# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest
import requests
import torch
from loguru import logger

import ttnn
from models.common.sampling import SamplingParams
from models.common.utility_functions import is_blackhole, is_wormhole_b0
from models.demos.utils.llm_demo_utils import create_benchmark_data, verify_accuracy, verify_perf
from models.demos.utils.model_targets import resolve_accuracy_targets, resolve_perf_targets
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.common import (
    PagedAttentionConfig,
    create_tt_model,
    preprocess_inputs_prefill,
    sample_host,
)
from models.tt_transformers.tt import inline_profile
from models.tt_transformers.tt.generator import Generator, SamplingParams, create_submeshes
from models.tt_transformers.tt.auto_shard.cost_model.mesh_selection import select_mesh_plan
from models.tt_transformers.tt.model_config import DecodersPrecision, determine_device_name, parse_decoder_json
from models.tt_transformers.tt.prefetcher import is_prefetcher_supported
from models.tt_transformers.tt.auto_shard.cost_model.sharding import workload_from_config

# Auto-shard variant: identical to simple_text_demo.py, but the model is built from the
# auto-shard stack. Importing model_auto_shard rebinds both model.TransformerBlock (auto-shard
# decoder block) and model.Transformer (replicated-residual-stream model) so create_tt_model()
# builds the auto-shard model + layers instead of the stock fractured ones.
import models.tt_transformers.tt.auto_shard.model_auto_shard  # noqa: F401  (import for side-effect wiring)

# Gemma-3 needs a different ModelArgs (unit-offset norms, embedding scale) and a different decoder
# block (pre/post feed-forward norms). This swaps both in, but only when HF_MODEL is a Gemma model;
# for every other model it is a no-op and this demo is unchanged.
import models.tt_transformers.tt.gemma_dispatch  # noqa: F401  (import for side-effect wiring)

# Issue: https://github.com/tenstorrent/tt-metal/issues/34763
models_not_supported_for_device_sampling = ["Mistral-7B"]


class TokenAccuracy:
    def __init__(self, model_name):
        self.gt_pos = -1
        self.store_predicted_tokens = []
        reference_data_file = os.path.join("models/tt_transformers/tests/reference_outputs/", model_name) + ".refpt"
        assert os.path.exists(reference_data_file)
        logger.info(f"Loading reference data from {reference_data_file}")
        reference_data = torch.load(reference_data_file)
        reference_tokens = reference_data["reference_tokens"]
        split_point = reference_tokens.shape[-1] // 2
        self.input_prompt = reference_tokens[0, :split_point]
        self.reference_tokens = reference_tokens[0, split_point:]
        self.top5_tokens = reference_data["top5_tokens"][split_point - 1 :, :]
        self.maxindex = len(self.reference_tokens) - 1

    def prepare_ref_tokens(self, tokenizer):
        text_data = tokenizer.decode(self.input_prompt.tolist())
        return text_data

    def collect_predicted_tokens(self, tokens):
        self.store_predicted_tokens.append(tokens)
        self.gt_pos += 1
        return self.reference_tokens[min(self.gt_pos, self.maxindex)].unsqueeze(-1).unsqueeze(-1)

    def compute_accuracy(self):
        count = 0
        count_t5 = 0
        matching_sz = min(len(self.reference_tokens), len(self.store_predicted_tokens))
        for i in range(matching_sz):
            if self.top5_tokens[i, 0].item() == self.store_predicted_tokens[i]:
                count += 1
            if self.store_predicted_tokens[i] in self.top5_tokens[i, :]:
                count_t5 += 1
        accuracy_top1 = count / matching_sz
        accuracy_top5 = count_t5 / matching_sz

        return accuracy_top1, accuracy_top5


def get_accuracy_thresholds(model_args, seq_len: int, batch_size: int = 1):
    """Resolve accuracy thresholds from centralized model targets."""
    centralized_targets = resolve_accuracy_targets(
        model_name=model_args.base_model_name,
        sku=model_args.device_name,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    if not centralized_targets or "top1" not in centralized_targets or "top5" not in centralized_targets:
        raise ValueError(
            "Could not find centralized accuracy targets for "
            f"{model_args.base_model_name} on {model_args.device_name} "
            f"(batch_size={batch_size}, seq_len={seq_len})"
        )

    # Preserve previous behavior for integer-rounded CI checks.
    return float(centralized_targets["top1"]) - 0.5, float(centralized_targets["top5"]) - 0.5


def load_and_cache_context(context_url, cache_dir, max_length=None):
    cache_file = cache_dir / hashlib.md5(context_url.encode()).hexdigest()

    if cache_file.exists():
        with open(cache_file, "r") as f:
            context_text = f.read()
        logger.info(f"Loaded context from cache: {context_url}")
    else:
        try:
            response = requests.get(context_url)
            if response.status_code == 200:
                context_text = response.text
                with open(cache_file, "w") as f:
                    f.write(context_text)
                logger.info(f"Downloaded and cached context: {context_url}")
            else:
                logger.warning(f"Failed to fetch context from URL: {context_url}. Status code: {response.status_code}")
                context_text = ""
        except Exception as e:
            logger.error(f"Error fetching context from URL: {context_url}. Error: {str(e)}")
            context_text = ""

    # Clip the context to the max length provided
    if max_length:
        context_text = context_text[:max_length]
        logger.info(f"Clipped the context text to {max_length} characters")

    return context_text


# load input prompts from json, return as a list
def load_inputs(user_input, batch, instruct):
    if isinstance(user_input, str):
        with open(user_input, "r") as f:
            user_input = json.load(f)

    if len(user_input) < batch:
        logger.warning(
            f"Number of users in the file is less than the provided batch={batch}. Repeating the prompts to match the batch size."
        )
        user_input = user_input * batch

    in_prompt = []
    all_prompts = []
    cache_dir = Path("models/tt_transformers/demo/context_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # The demo supports a custom prompt file, where the context is provided by a link to a book from the gutenberg project
    # It clips the excerpt to the max length provided to allow testing different long context lengths
    for i in range(len(user_input)):
        prompt = user_input[i]["prompt"]
        if "context" in user_input[i]:
            if "max_length" in user_input[i]:  # Clip the context to the max length provided
                context_text = load_and_cache_context(
                    user_input[i]["context"], cache_dir, max_length=user_input[i]["max_length"]
                )
            else:
                context_text = load_and_cache_context(user_input[i]["context"], cache_dir)
            if instruct:
                prompt = (
                    "```" + context_text + "```\n\n" + prompt
                )  # Add the markdown block to the context to comply with the prompt
            else:
                prompt = context_text
        all_prompts.append(prompt)  # return all the prompts taken from the input file to be used when repeat_batch > 1
        if i in range(batch):
            in_prompt.append(prompt)
    return in_prompt, all_prompts


def create_tt_page_table(global_batch_size, data_parallel, paged_attention_config: PagedAttentionConfig):
    page_table = None

    if paged_attention_config:
        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation).repeat(data_parallel)
        page_table = reverse_permutation.reshape(
            global_batch_size, paged_attention_config.max_num_blocks // (global_batch_size // data_parallel)
        )
    return page_table


def submesh_has_local_devices(submesh):
    """Return True if this submesh has at least one device local to the current host rank."""
    view = submesh.get_view()
    return any(
        view.is_local(ttnn.MeshCoordinate(row, col))
        for row in range(submesh.shape[0])
        for col in range(submesh.shape[1])
    )


def get_local_submesh_indices(submesh_devices):
    """Return indices of submeshes that own at least one local device on this host rank."""
    return [i for i, submesh in enumerate(submesh_devices) if submesh_has_local_devices(submesh)]


def select_local_data_parallel_items(items, batch_size, data_parallel, local_submesh_indices):
    """Select the per-DP-group slices corresponding to local submeshes."""
    assert (
        len(items) >= batch_size * data_parallel
    ), f"Expected at least {batch_size * data_parallel} items, got {len(items)}"
    return [item for dp_idx in local_submesh_indices for item in items[dp_idx * batch_size : (dp_idx + 1) * batch_size]]


def slice_sampling_params_for_local_submeshes(sampling_params, batch_size, data_parallel, local_submesh_indices):
    """Slice list-valued sampling params so each host rank keeps only its local DP groups."""
    result = dict(sampling_params)
    total_items = batch_size * data_parallel

    for key, value in sampling_params.items():
        if not isinstance(value, list):
            continue
        if len(value) == total_items:
            result[key] = select_local_data_parallel_items(value, batch_size, data_parallel, local_submesh_indices)
        elif len(value) == data_parallel:
            result[key] = [value[i] for i in local_submesh_indices]

    return result


def get_default_mesh_device_param():
    """
    Select a safe default mesh size for parameterized tests.

    In distributed runs, use the global system mesh size from the mesh graph descriptor
    instead of len(get_device_ids()), which can reflect host-local visibility.
    """
    if ttnn.using_distributed_env():
        try:
            return ttnn._ttnn.multi_device.SystemMeshDescriptor().shape().mesh_size()
        except Exception as e:
            logger.warning(f"Falling back to local device count for default mesh sizing: {e}")
    return len(ttnn.get_device_ids())


def prepare_generator_args(
    num_devices,
    data_parallel,
    mesh_device,
    instruct,
    global_batch_size,
    optimizations,
    max_seq_len,
    page_params,
    paged_attention,
    num_layers,
    use_prefetcher,
    use_hf_rope,
    max_generated_tokens=None,
):
    all_submesh_devices = create_submeshes(mesh_device, data_parallel)
    local_submesh_indices = get_local_submesh_indices(all_submesh_devices)
    submesh_devices = [all_submesh_devices[i] for i in local_submesh_indices]

    if not submesh_devices:
        raise RuntimeError("No local submeshes available on this host rank for the requested configuration")

    if len(submesh_devices) != len(all_submesh_devices):
        logger.info(
            f"Distributed mode detected: using local submeshes {local_submesh_indices} "
            f"({len(submesh_devices)}/{len(all_submesh_devices)}) on this host rank"
        )

    state_dict = None

    # Hybrid requires a model per submesh
    model_args = []
    model = []
    tt_kv_cache = []

    paged_attention_config = (
        PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks_per_dp"],
        )
        if paged_attention
        else None
    )

    max_batch_size_per_dp_group = global_batch_size // data_parallel

    for submesh in submesh_devices:
        model_args_i, model_i, tt_kv_cache_i, state_dict = create_tt_model(
            submesh,
            instruct=instruct,
            max_batch_size=max_batch_size_per_dp_group,
            optimizations=optimizations,
            max_seq_len=max_seq_len,
            paged_attention_config=paged_attention_config,
            dtype=ttnn.bfloat8_b,
            state_dict=state_dict,
            num_layers=num_layers,
            use_prefetcher=use_prefetcher,
            use_hf_rope=use_hf_rope,
            max_generated_tokens=max_generated_tokens,
        )
        model_args.append(model_args_i)
        model.append(model_i)
        tt_kv_cache.append(tt_kv_cache_i)

    local_data_parallel = len(submesh_devices)
    local_batch_size = max_batch_size_per_dp_group * local_data_parallel
    page_table = create_tt_page_table(
        global_batch_size=local_batch_size,
        data_parallel=local_data_parallel,
        paged_attention_config=paged_attention_config,
    )
    # Host code, safe to reuse tokenizer from the 1st model
    tokenizer = model_args[
        0
    ].tokenizer  # TODO Should we support Data Parallel different models? If so, we need to support multiple tokenizers
    processor = model_args[0].processor
    return model_args, model, page_table, tt_kv_cache, tokenizer, processor, local_data_parallel, local_submesh_indices


# List of supported Parameters for demo.py
#
# input_prompts (string): input json file with prompts to process. See models/tt_transformers/demo/*.json for list of input files
# instruct (bool): Whether to use instruct weights or general weights
# repeat_batches (int): Number of consecutive batches of users to run (default: 1)
# max_seq_len (int): Maximum context length supported by the model (Llama-3.1 and Llama-3.2 models have a maximum context length of 128k, i.e., 128 * 1024)
# batch_size (int): Number of users in a batch (Supports 1/2/4/8/16/32 batches)
# max_generated_tokens (int): Maximum number of tokens to generate for each user (Note that the users will stop generation before this limit if they reach a EoS token)
# paged_attention (bool): Whether to use paged attention or default attention (vLLM requires paged attention)
# page_params (dict): Page parameters for paged attention (block_size, max_num_blocks) For smaller context lengths use block_size=32 and max_num_blocks=1024, for larger context use block_size=64 and max_num_blocks=2048
# sampling_params (dict): Sampling parameters for decoding (temperature, top_p). If temperature is set to 0, argmax (greedy decode) is used.
# stop_at_eos (bool): Whether to stop decoding when the model generates an EoS token
# ci_only (bool): Whether to run the demo in CI only mode
# data_parallel (int): Number of data parallel groups to use
# token_accuracy (bool): Whether to compute token accuracy
# stress_test (bool): Whether to run the demo in stress test mode
# enable_trace (bool): Whether to enable tracing
# num_layers (int): Number of layers to use
# mode (str): Mode to run the demo in (full, prefill, decode), full will run both prefill and decode
# optimization (ModelOptimizations): Optimization level to use for the model (performance or accuracy)
# MESH_DEVICE (str): Fake device to use for testing (N150, N300, T3K, TG). Usage: `export MESH_DEVICE=N150`, will enable running a single-chip demo on a multi-chip system.
_trace_region_size = (
    100000000
    if (is_blackhole() and os.environ.get("HF_MODEL", "").endswith(("Qwen2.5-72B-Instruct", "Qwen2.5-32B-Instruct")))
    else 50000000
)


_MESH_SHAPES = {
    "N150": (1, 1),
    "N300": (1, 2),
    "N300x1x4": (1, 4),
    "N300x2x2": (2, 2),
    "N150x4": (2, 2),
    "T3K": (1, 8),
    "TG": (8, 4),
    "P150": (1, 1),
    "P300": (1, 2),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
    "BHGLX": (8, 4),
}

# MESH_DEVICE values that run the model on a submesh of a larger opened mesh: name -> (open, run).
#
# Fabric has to own every device in the system: opening a mesh smaller than the machine dies with
# "Fabric is being used but Device N is not active ... To launch on a subset of devices, first create
# a MeshDevice of the full system size, then create submeshes accordingly" (device_manager.cpp:436).
# So a 2-chip run on Boromir (2x N300 = a 2x2 of 4 chips) opens the whole 2x2 and carves a 1x2 out of
# it. The plain "N300" entry above stays a real 1x2 open, which is what a single-N300 machine wants.
#
# Both orientations are here because the two axes are different wires: axis 1 (1x2, two columns)
# crosses cards over QSFP-DD/WARP100, axis 0 (2x1, two rows) is the intra-card TRACE link.
_SUBMESH_SHAPES = {
    "N300x1x2": ((2, 2), (1, 2)),
    "N300x2x1": ((2, 2), (2, 1)),
}

# MESH_DEVICE=AUTO hands the shape choice to mesh_selection. Every other MESH_DEVICE value keeps
# meaning exactly what it meant before.
#
# The plan is picked HERE, at import: the fixture opens devices during the test, and the descriptor
# (TT_MESH_GRAPH_DESC_PATH, read at fabric init), the opened shape, and the fabric config are all
# fixed by that open. Choosing afterwards would be choosing after they were already frozen. A plan
# supplies the first two directly; the fabric follows from its open_shape via _fabric_config_for,
# which is what gets the 1x4 candidate a ring and the 2x2 candidate plain FABRIC_1D.
_AUTO = "AUTO"


def _auto_mesh():
    return os.environ.get("MESH_DEVICE") == _AUTO


def _resolve_mesh_shape_param():
    """The mesh this run opens, from MESH_DEVICE. A tuple (rows, cols), or an int device count."""
    if _auto_plan:
        return _auto_plan.open_shape
    name = os.environ.get("MESH_DEVICE")
    if name in _SUBMESH_SHAPES:
        return _SUBMESH_SHAPES[name][0]
    return _MESH_SHAPES.get(name, get_default_mesh_device_param())


def _resolve_submesh_shape():
    """The submesh this run carves out of the opened mesh, or None to use the whole thing."""
    if _auto_plan:
        return _auto_plan.submesh
    entry = _SUBMESH_SHAPES.get(os.environ.get("MESH_DEVICE"))
    return entry[1] if entry else None


def _maybe_submesh(mesh_device):
    """Carve out the submesh this run asked for, if it asked for one.

    Everything downstream (num_devices, cluster_shape, the auto-shard placement search) reads the
    mesh off this object, so substituting it here is all a smaller run needs. The mesh_device fixture
    closes submeshes before the parent, so the carved mesh doesn't leak. An auto-mesh plan carves
    through this same path -- its submesh is just decided at import instead of read off MESH_DEVICE.
    """
    shape = _resolve_submesh_shape()
    if shape is None or not isinstance(mesh_device, ttnn.MeshDevice):
        return mesh_device
    submesh = mesh_device.create_submeshes(ttnn.MeshShape(*shape))[0]
    logger.info(f"Running on a {shape} submesh of the opened {tuple(mesh_device.shape)} mesh")
    return submesh


def _fabric_config_for(shape):
    """Ring fabric on a line, plain 1D on a genuine 2D mesh -- mirrors toy_problem's fabric_for().

    A line (one dim == 1) closes into a ring, so bring the fabric up as a ring; ccl_topology() then
    returns Ring off its first branch (the one that actually checks the fabric). A real 2D mesh does
    not close into a ring on either axis, and ccl_topology() runs Linear collectives there, so a ring
    fabric would just be a lie.

    Never pass True here: a bool coerces to enum value 1 == FABRIC_1D_NEIGHBOR_EXCHANGE ("1D topology
    with no forwarding between non-adjacent devices"), so any collective needing a multi-hop route --
    e.g. a Ring's wrap-around on a mesh whose logical order isn't the physical cycle -- dies with
    "Could not find any forwarding direction from src ... to dst ...".
    """
    # TODO: this should actually check for a wraparound wire, or be something the user can set
    # An int param is a flat device count, which the fixture opens as a line.
    is_line = (1 in tuple(shape)) if isinstance(shape, tuple) else True
    return ttnn.FabricConfig.FABRIC_1D_RING if is_line else ttnn.FabricConfig.FABRIC_1D


# Hoisted out of the parametrize decorator so mesh selection can read the workload off the case that
# is about to run (see _demo_workload) -- the decorator's arguments are evaluated at import, which is
# also the only moment the mesh is still choosable. Re-applied to test_demo_text below, in its
# original position relative to the other decorators.
_DEMO_PARAMS = "input_prompts, instruct, repeat_batches, max_seq_len, batch_size, max_generated_tokens, paged_attention, page_params, sampling_params, stop_at_eos, ci_only, data_parallel, token_accuracy, stress_test, enable_trace, num_layers, mode"
_DEMO_CASES = [
        (  # Batch-1 run (Latency) - single user, small prompt
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            1024,  # max_seq_len
            1,  # batch_size
            200,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024},  # page_params
            {"temperature": 0, "top_p": 0.08, "top_k": 32},  # sampling_params (argmax)
            True,  # stop_at_eos
            False,  # ci_only
            1,
            False,  # token_accuracy
            False,  # stress_test
            True,  # enable_trace
            None,  # num_layers, if None -> defaults to all layers
            "full",  # performs both prefill and decode
        ),
        (  # Batch-32 run (Throughput) - 32 users, small prompt
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            1024,  # max_seq_len
            32,  # batch_size
            200,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024},  # page_params
            {
                "temperature": torch.linspace(0.0, 1.0, steps=32).tolist(),
                "top_p": torch.linspace(0.08, 1.0, steps=32).tolist(),
                "top_k": torch.arange(1, 33).tolist(),  # 1 to 32 inclusive
                "frequency_penalty": torch.linspace(0.0, 1.0, steps=32).tolist(),
                "presence_penalty": torch.linspace(0.0, 1.0, steps=32).tolist(),
                "repetition_penalty": torch.linspace(0.0, 1.0, steps=32).tolist(),
            },  # sampling_params (non-uniform)
            True,  # stop_at_eos
            False,  # ci_only
            1,  # data_parallel
            False,  # token_accuracy
            False,  # stress_test
            True,  # enable_trace
            None,  # num_layers, if None -> defaults to all layers
            "full",  # performs both prefill and decode
        ),
        (  # Batch-32 run (Throughput) - 32 users, small prompt with log-probs
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            1024,  # max_seq_len
            32,  # batch_size
            200,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024},  # page_params
            {
                "temperature": torch.linspace(0.0, 1.0, steps=32).tolist(),
                "top_p": torch.linspace(0.08, 1.0, steps=32).tolist(),
                "top_k": torch.arange(1, 33).tolist(),  # 1 to 32 inclusive
                "enable_log_probs": [True] * 32,
            },  # sampling_params (non-uniform)
            True,  # stop_at_eos
            False,  # ci_only
            1,  # data_parallel
            False,  # token_accuracy
            False,  # stress_test
            True,  # enable_trace
            None,  # num_layers, if None -> defaults to all layers
            "full",  # performs both prefill and decode
        ),
        (  # long-context-64k run - Single user, long prompt (may vary based on the model's tokenizer)
            "models/tt_transformers/demo/sample_prompts/input_data_long_64k.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            1,  # batch_size
            200,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks_per_dp": 2048},  # page_params
            {"temperature": 0, "top_p": 0.08, "top_k": 32},  # sampling_params (argmax)
            True,  # stop_at_eos
            False,  # ci_only
            1,  # data_parallel
            False,  # token_accuracy
            False,  # stress_test
            True,  # enable_trace
            None,  # num_layers, if None -> defaults to all layers
            "full",  # performs both prefill and decode
        ),
        (  # Long-context-32k run - Single user, long prompt (may vary based on the model's tokenizer)
            "models/tt_transformers/demo/sample_prompts/input_data_long_32k.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            64 * 1024,  # max_seq_len
            1,  # batch_size
            200,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks_per_dp": 1024},  # page_params
            {"temperature": 0, "top_p": 0.08, "top_k": 32},  # sampling_params (argmax)
            True,  # stop_at_eos
            False,  # ci_only
            1,  # data_parallel
            False,  # token_accuracy
            False,  # stress_test
            True,  # enable_trace
            None,  # num_layers, if None -> defaults to all layers
            "full",  # performs both prefill and decode
        ),
        (  # Long-context-16k run - Single user, long prompt (may vary based on the model's tokenizer)
            "models/tt_transformers/demo/sample_prompts/input_data_long_16k.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            32 * 1024,  # max_seq_len
            1,  # batch_size
            200,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024},  # page_params
            {"temperature": 0, "top_p": 0.08, "top_k": 32},  # sampling_params (argmax)
            True,  # stop_at_eos
            False,  # ci_only
            1,  # data_parallel
            False,  # token_accuracy
            False,  # stress_test
            True,  # enable_trace
            None,  # num_layers, if None -> defaults to all layers
            "full",  # performs both prefill and decode
        ),
        (  # reasoning-1 - single user, small prompt, long thinking time
            "models/tt_transformers/demo/input_data_questions_reasoning.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            16 * 1024,  # max_seq_len
            1,  # batch_size
            15000,  # max_generated_tokens
            True,  # paged_attention
            {
                "page_block_size": 32,
                "page_max_num_blocks_per_dp": 1024,
            },  # page_params  # TODO This will be serviced by vLLM
            {"temperature": 0, "top_p": 0.08, "top_k": 32},  # sampling_params (argmax)
            False,  # stop_at_eos
            False,  # ci_only
            1,  # data_parallel
            False,  # token_accuracy
            False,  # stress_test
            True,  # enable_trace
            None,  # num_layers, if None -> defaults to all layers
            "full",  # performs both prefill and decode
        ),
        (  # ci-1 [CI-only] - Measures the performance of a single user over 4096 iterations
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            8192,  # max_seq_len
            1,  # batch_size
            4096,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024},  # page_params
            {"temperature": 0, "top_p": 0.08, "top_k": 32},  # sampling_params (argmax)
            False,  # stop_at_eos
            True,  # ci_only
            1,  # data_parallel
            False,  # token_accuracy
            False,  # stress_test
            True,  # enable_trace
            None,  # num_layers, if None -> defaults to all layers
            "full",  # performs both prefill and decode
        ),
        (  # ci-32 [CI-only] - Measures the performance of 32 users over 4096 iterations
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            2048,  # max_seq_len
            32,  # batch_size
            1024,  # max_generated_tokens  # TODO Update this to 4096, and make sure it fits in DRAM with correct page_params
            True,  # paged_attention  # TODO Find the correct paged_attn params to avoid hangs in this config with long context generation
            {"page_block_size": 64, "page_max_num_blocks_per_dp": 1024},  # page_params
            {"temperature": 0, "top_p": 0.08, "top_k": 32},  # sampling_params (argmax)
            False,  # stop_at_eos
            True,  # ci_only
            1,  # data_parallel
            False,  # token_accuracy
            False,  # stress_test
            True,  # enable_trace
            None,  # num_layers, if None -> defaults to all layers
            "full",  # performs both prefill and decode
        ),
        (  # DP-4-b1 - single user, data-parallel=4, small prompt
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            1024,  # max_seq_len
            1,  # batch_size
            200,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024},  # page_params
            {"temperature": 0, "top_p": 0.08, "top_k": 32},  # sampling_params (argmax)
            True,  # stop_at_eos
            False,  # ci_only
            4,  # data_parallel
            False,  # token_accuracy
            False,  # stress_test
            True,  # enable_trace
            None,  # num_layers, if None -> defaults to all layers
            "full",  # performs both prefill and decode
        ),
        (  # DP-8-b1 - single user, data-parallel=8, small prompt
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            1024,  # max_seq_len
            1,  # batch_size
            200,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024},  # page_params
            {"temperature": 0, "top_p": 0.08, "top_k": 32},  # sampling_params (argmax)
            True,  # stop_at_eos
            False,  # ci_only
            8,  # data_parallel
            False,  # token_accuracy
            False,  # stress_test
            True,  # enable_trace
            None,  # num_layers, if None -> defaults to all layers
            "full",  # performs both prefill and decode
        ),
        (  # DP-4-b32 - 32 users, data-parallel=4, small prompt
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            1024,  # max_seq_len
            32,  # batch_size
            200,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024},  # page_params
            {"temperature": 0, "top_p": 0.08, "top_k": 32},  # sampling_params (argmax)
            True,  # stop_at_eos
            False,  # ci_only
            4,  # data_parallel
            False,  # token_accuracy
            False,  # stress_test
            True,  # enable_trace
            None,  # num_layers, if None -> defaults to all layers
            "full",  # performs both prefill and decode
        ),
        (  # ci-b1-DP-4 [CI-Only] - single user, data-parallel=4, small prompt
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            4096,  # max_seq_len
            1,  # batch_size
            2048,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024},  # page_params
            {"temperature": 0, "top_p": 0.08, "top_k": 32},  # sampling_params (argmax)
            False,  # stop_at_eos
            True,  # ci_only
            4,  # data_parallel
            False,  # token_accuracy
            False,  # stress_test
            True,  # enable_trace
            None,  # num_layers, if None -> defaults to all layers
            "full",  # performs both prefill and decode
        ),
        (  # ci-b1-DP-8 [CI-Only] - single user, data-parallel=8, small prompt
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            4096,  # max_seq_len
            1,  # batch_size
            2048,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024},  # page_params
            {"temperature": 0, "top_p": 0.08, "top_k": 32},  # sampling_params (argmax)
            False,  # stop_at_eos
            True,  # ci_only
            8,  # data_parallel
            False,  # token_accuracy
            False,  # stress_test
            True,  # enable_trace
            None,  # num_layers, if None -> defaults to all layers
            "full",  # performs both prefill and decode
        ),
        (  # ci-b1-DP-16 [CI-Only] - single user, data-parallel=16, small prompt
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            1024,  # max_seq_len
            1,  # batch_size
            200,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024},  # page_params
            {"temperature": 0, "top_p": 0.08, "top_k": 32},  # sampling_params (argmax)
            True,  # stop_at_eos
            True,  # ci_only
            16,  # data_parallel
            False,  # token_accuracy
            False,  # stress_test
            True,  # enable_trace
            None,  # num_layers, if None -> defaults to all layers
            "full",  # performs both prefill and decode
        ),
        (  # ci-b1-DP-32 [CI-Only] - single user, data-parallel=32, small prompt
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            1024,  # max_seq_len
            1,  # batch_size
            200,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024},  # page_params
            {"temperature": 0, "top_p": 0.08, "top_k": 32},  # sampling_params (argmax)
            True,  # stop_at_eos
            True,  # ci_only
            32,  # data_parallel
            False,  # token_accuracy
            False,  # stress_test
            True,  # enable_trace
            None,  # num_layers, if None -> defaults to all layers
            "full",  # performs both prefill and decode
        ),
        (  # ci-stress-1 [CI-only] stress test - Runs a short prefill (128) and loops the same iteration over 20000 times
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            1,  # batch_size
            20000,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks_per_dp": 2048},  # page_params
            {"temperature": 0, "top_p": 0.08, "top_k": 32},  # sampling_params (argmax)
            False,  # stop_at_eos
            True,  # ci_only
            1,  # data_parallel
            False,  # token_accuracy
            True,  # stress_test
            True,  # enable_trace
            None,  # num_layers, if None -> defaults to all layers
            "full",  # performs both prefill and decode
        ),
        (  # ci-token-matching run - Measures token matching accuracy of a single user over 500 iterations
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            False,  # instruct mode
            1,  # repeat_batches
            1024,  # max_seq_len
            1,  # batch_size
            500,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024},  # page_params
            {"temperature": 0, "top_p": 0.08, "top_k": 32},  # sampling_params (argmax)
            True,  # stop_at_eos
            True,  # ci_only
            1,  # data_parallel
            True,  # token_accuracy
            False,  # stress_test
            False,  # enable_trace -> Teacher forcing does not work if it is on
            None,  # num_layers, if None -> defaults to all layers
            "full",  # performs both prefill and decode
        ),
        (  # ci-eval-1 - 6 repeat batches with output comparison
            "models/tt_transformers/demo/sample_prompts/eval_repeat_prompts_batch1.json",  # input_prompts
            True,  # instruct mode
            6,  # repeat_batches
            1024,  # max_seq_len
            1,  # batch_size
            200,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024},  # page_params
            {"temperature": 0, "top_p": 0.08, "top_k": 32},  # sampling_params (argmax)
            True,  # stop_at_eos
            True,  # ci_only
            1,  # data_parallel
            False,  # token_accuracy
            False,  # stress_test
            True,  # enable_trace
            None,  # num_layers, if None -> defaults to all layers
            "full",  # performs both prefill and decode
        ),
        (  # ci-eval-32 - 32 users with 3 repeat batches and shifting prompts
            "models/tt_transformers/demo/sample_prompts/eval_repeat_prompts_batch32.json",  # input_prompts
            True,  # instruct mode
            3,  # repeat_batches
            1024,  # max_seq_len
            32,  # batch_size
            200,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024},  # page_params
            {"temperature": 0, "top_p": 0.08, "top_k": 32},  # sampling_params (argmax)
            True,  # stop_at_eos
            True,  # ci_only
            1,  # data_parallel
            False,  # token_accuracy
            False,  # stress_test
            True,  # enable_trace
            None,  # num_layers, if None -> defaults to all layers
            "full",  # performs both prefill and decode
        ),
        (  # [CI only] Long-context-16k run - Single user, long prompt (may vary based on the model's tokenizer)
            "models/tt_transformers/demo/sample_prompts/input_data_long_16k.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            32 * 1024,  # max_seq_len
            1,  # batch_size
            200,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024},  # page_params
            {"temperature": 0, "top_p": 0.08, "top_k": 32},  # sampling_params (argmax)
            True,  # stop_at_eos
            True,  # ci_only
            1,  # data_parallel
            False,  # token_accuracy
            False,  # stress_test
            True,  # enable_trace
            None,  # num_layers, if None -> defaults to all layers
            "full",  # performs both prefill and decode
        ),
        (  # device-perf - Measures device performance of a prefill or decode run (by default runs prefill but test_device_perf uses args to override defaults)
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            False,  # instruct mode
            1,  # repeat_batches
            1024,  # max_seq_len
            1,  # batch_size
            2,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024},  # page_params
            {"temperature": 0, "top_p": 0.08, "top_k": 32},  # sampling_params (argmax)
            True,  # stop_at_eos
            True,  # ci_only
            1,  # data_parallel
            False,  # token_accuracy
            False,  # stress_test
            True,  # enable_trace
            10,  # num_layers, if None -> defaults to all layers
            "prefill",  # mode
        ),
]
_DEMO_IDS = [
    "batch-1",  # latency
    "batch-32",  # throughput
    "batch-32-log-probs",  # throughput with log-probs
    "long-context-64k",  # 64k context, max_seq_len=128k
    "long-context-32k",  # 32k context, max_seq_len=32k
    "long-context-16k",  # 16k context, max_seq_len=32k
    "reasoning-1",  # reasoning
    "ci-1",  # CI batch 1
    "ci-32",  # CI batch 32
    "DP-4-b1",  # DP 4 latency
    "DP-8-b1",  # DP 8 latency
    "DP-4-b32",  # DP 4 throughput
    "ci-b1-DP-4",  # CI DP 4 batch 1
    "ci-b1-DP-8",  # CI DP 8 batch 1
    "ci-b1-DP-16",  # CI DP 16 batch 1
    "ci-b1-DP-32",  # CI DP 32 batch 1
    "ci-stress-1",  # CI Stress test batch-1
    "ci-token-matching",  # CI performs token accuracy matching with reference procomputed tokens
    "ci-eval-1",  # CI 6 repeat batches with output comparison
    "ci-eval-32",  # CI batch 32 with 3 repeat batches and output comparison
    "ci-long-context-16k",  # 16k context, max_seq_len=32k, used for testing --max_seq_len=16k override
    "device-perf",  # Device perf
]


def _case_value(case, name):
    """One field of a _DEMO_CASES tuple, by its name in _DEMO_PARAMS (the file warns that the
    parameter order is load-bearing, so read by name rather than by index)."""
    return case[[p.strip() for p in _DEMO_PARAMS.split(",")].index(name)]


def _cli_int(flag):
    """`--flag=N` or `--flag N` off the pytest command line, or None.

    The demo's own conftest defines --max_seq_len/--max_generated_tokens and the test applies them
    over the case's values; mesh selection runs before pytest parses argv, so read them the hard way.
    """
    for i, arg in enumerate(sys.argv):
        if arg == flag and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if arg.startswith(f"{flag}="):
            return int(arg.split("=", 1)[1])
    return None


def _demo_workload():
    """(prefill_len, decode_steps) for the case about to run, for the mesh cost model.

    The mesh has to be chosen at import, but which case runs is decided later by -k, so infer it: an
    id that appears in the -k expression is a case pytest will select. Only a unique match is
    usable -- one mesh is opened for the whole session, so if -k selects several cases with different
    workloads there is no single right answer and the params.py defaults (None) are the honest reply.

    Returns None to mean "use the defaults", which is also what a non-auto run gets.
    """
    kexpr = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "-k" and i + 1 < len(sys.argv)), "")
    # -k selects a case when one of its terms is a SUBSTRING of the id, not when the id appears in
    # -k: `-k batch-32` runs "batch-32" and "batch-32-log-probs" both, and is therefore ambiguous.
    # Terms naming another parametrize axis ("performance") simply match no id and drop out.
    terms = [t for t in re.split(r"[\s()]+", kexpr) if t and t not in ("and", "or", "not")]
    matched = [i for i, name in enumerate(_DEMO_IDS) if any(t in name for t in terms)]
    if len(matched) != 1:
        logger.warning(
            f"auto-mesh: -k {kexpr!r} matches {len(matched)} demo cases, cannot tell which workload to "
            f"size the mesh for; falling back to the params.py default"
        )
        return None
    case = _DEMO_CASES[matched[0]]
    max_seq_len = _cli_int("--max_seq_len") or _case_value(case, "max_seq_len")
    max_generated_tokens = _cli_int("--max_generated_tokens") or _case_value(case, "max_generated_tokens")
    # Reuse sharding's convention (largest prompt that fits, then that many decodes) rather than
    # restating it -- it only reads these two attributes off whatever it is handed.
    workload = workload_from_config(SimpleNamespace(max_seq_len=max_seq_len, max_generated_tokens=max_generated_tokens))
    logger.info(f"auto-mesh: sizing for demo case {_DEMO_IDS[matched[0]]!r} -> prefill_len={workload[0]}, decode_steps={workload[1]}")
    return workload


# Import-time, and it has to stay that way -- see the _AUTO note above. Also sets
# TT_MESH_GRAPH_DESC_PATH. Down here rather than beside _auto_mesh because _demo_workload needs
# _DEMO_CASES, and the decorators that consume _mesh_shape_param are all below this point.
_auto_plan = select_mesh_plan(_demo_workload()) if _auto_mesh() else None
_mesh_shape_param = _resolve_mesh_shape_param()


@pytest.mark.parametrize(_DEMO_PARAMS, _DEMO_CASES, ids=_DEMO_IDS)
# NOTE: Please do not add new pytest parameters between optimizations and the demo parameters above, certain tests ids depend on the order of the parameters.
@pytest.mark.parametrize(
    "optimizations",
    [
        lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name),
        lambda model_args: DecodersPrecision.accuracy(model_args.n_layers, model_args.model_name),
    ],
    ids=["performance", "accuracy"],
)
@pytest.mark.parametrize(
    "use_prefetcher",
    ([False]),
)
@pytest.mark.parametrize(
    "device_params",
    # RELAXED_INIT because STRICT auto-discovery can't map Boromir's 2xN300 (see toy_problem/utils.py).
    [
        {
            "fabric_config": _fabric_config_for(_mesh_shape_param),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "trace_region_size": _trace_region_size,
            "num_command_queues": 1,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [_mesh_shape_param],
    indirect=True,
)
def test_demo_text(
    input_prompts,
    instruct,
    repeat_batches,
    max_seq_len,
    batch_size,
    max_generated_tokens,
    paged_attention,
    page_params,
    sampling_params,
    optimizations,
    stop_at_eos,
    mesh_device,
    is_ci_env,
    is_ci_v2_env,
    ci_only,
    data_parallel,
    reset_seeds,
    request,
    token_accuracy,
    stress_test,
    enable_trace,
    model_location_generator,
    num_layers,
    mode,
    use_prefetcher,
):
    """
    Simple demo with limited dependence on reference code.
    """
    hf_dir = os.getenv("HF_MODEL", "")
    mesh_device = _maybe_submesh(mesh_device)
    num_devices = mesh_device.get_num_devices() if isinstance(mesh_device, ttnn.MeshDevice) else 1

    test_id = request.node.callspec.id
    if is_ci_env:
        if not ci_only:
            pytest.skip("CI only runs the CI-only tests")
        if "accuracy" in test_id and "ci-token-matching" not in test_id:
            pytest.skip("CI only runs the tests with performance optimizations except for ci-token-matching case")

    # TODO: Remove this once all batch sizes are supported on TG
    if os.environ.get("MESH_DEVICE") == "TG" and batch_size not in [1, 32]:
        pytest.skip("TG only supports batch 1 and 32")

    print_to_file = False  # Enable this flag to print the output of all users to a file

    # Override parameters from command line if they are provided
    input_prompts = request.config.getoption("--input_prompts") or input_prompts
    if request.config.getoption("--instruct") in [
        0,
        1,
    ]:  # If the flag is provided, use it. Take an int instead of bool due to parser limitations
        instruct = request.config.getoption("--instruct")
    repeat_batches = request.config.getoption("--repeat_batches") or repeat_batches
    max_seq_len = request.config.getoption("--max_seq_len") or max_seq_len
    batch_size = request.config.getoption("--batch_size") or batch_size
    max_generated_tokens = request.config.getoption("--max_generated_tokens") or max_generated_tokens
    data_parallel = request.config.getoption("--data_parallel") or data_parallel
    paged_attention = request.config.getoption("--paged_attention") or paged_attention
    page_params = request.config.getoption("--page_params") or page_params
    if isinstance(page_params, str):  # Required for proper load of a dictionary from the override command
        page_params = json.loads(page_params)
    sampling_params = request.config.getoption("--sampling_params") or sampling_params
    json_config_file = request.config.getoption("--decoder_config_file")
    token_accuracy = request.config.getoption("--token_accuracy") or token_accuracy
    stress_test = request.config.getoption("--stress_test") or stress_test
    arg_enable_trace = request.config.getoption("--enable_trace")
    if arg_enable_trace is not None:
        enable_trace = arg_enable_trace
    num_layers = request.config.getoption("--num_layers") or num_layers
    mode = request.config.getoption("--mode") or mode
    use_prefetcher = request.config.getoption("--use_prefetcher") or use_prefetcher
    if use_prefetcher and not is_blackhole():
        logger.warning("--use_prefetcher requested but DRAM prefetcher is only supported on Blackhole; disabling.")
        use_prefetcher = False
    use_prefetcher = (
        use_prefetcher and is_prefetcher_supported(hf_dir, num_devices) and "Llama" in hf_dir and "8B" in hf_dir
    )
    global_batch_size = batch_size * data_parallel  # input batch_size is interpreted as size per DP group
    use_hf_rope = request.config.getoption("--use_hf_rope")
    is_device_perf_test = "device-perf" in test_id

    if stress_test and token_accuracy:
        pytest.skip("Stress test cannot be run with token accuracy mode")

    if json_config_file:
        optimizations = parse_decoder_json(json_config_file)
    else:
        optimizations = request.config.getoption("--optimizations") or optimizations

    if request.config.getoption("--stop_at_eos") in [
        0,
        1,
    ]:  # If the flag is provided, use it. Take an int instead of bool due to parser limitations
        stop_at_eos = request.config.getoption("--stop_at_eos")

    if "phi-3-mini-128k-instruct" in hf_dir.lower():
        max_context_supported = 32 * 1024 * num_devices
        # This condition is present since Phi3 mini has a limit of context length 32k for N150
        # It makes sure neither the total_page_cache nor the max_seq_length exceeds this limit.
        if (max_context_supported < max_seq_len) or (
            max_context_supported < page_params["page_block_size"] * page_params["page_max_num_blocks_per_dp"]
        ):
            pytest.skip(
                f"Max sequence length: {max_seq_len} for batch: {batch_size} not supported for model: {hf_dir} on device: {mesh_device}"
            )

    # uneven split of devices per DP group not supported
    if data_parallel > num_devices or num_devices % data_parallel != 0:
        pytest.skip(f"Invalid number of DP groups: {data_parallel}, for {num_devices} devices")

    if is_ci_env:
        hf_model = os.getenv("HF_MODEL", "")
        is_33_70b = "3.3-70B" in hf_model
        is_32_1b = "3.2-1B" in hf_model
        is_31_8b = "3.1-8B" in hf_model

        tg_enabled = (data_parallel == 4 and is_33_70b) or (data_parallel in [4, 16, 32] and is_31_8b)

        if num_devices == 32 and not tg_enabled:
            pytest.skip("CI only runs Llama3 70b DP = 4, TP = 8 or Llama3 8b DP = 4/16/32, TP = 8/2/1 on TG")
        if num_devices == 8 and data_parallel > 1 and not (is_32_1b or is_31_8b) and is_wormhole_b0():
            pytest.skip("CI only runs hybrid Llama3 1b and 8b on T3K")

    if is_ci_v2_env:
        hf_model = os.getenv("HF_MODEL", "")
        model_location = model_location_generator(hf_model, download_if_ci_v2=True, ci_v2_timeout_in_s=900)
        # update env var HF_MODEL to the model location
        os.environ["HF_MODEL"] = str(model_location)

    if not stop_at_eos:
        logger.info(f"The decode generation will only stop at the max_generated_tokens limit == {max_generated_tokens}")

    if print_to_file:
        # Creat batch output file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_directory = "models/tt_transformers/demo/output"
        os.makedirs(output_directory, exist_ok=True)
        os.chmod(output_directory, 0o755)
        output_filename = f"{output_directory}/llama_text_demo_output_{timestamp}.txt"

    # Start profiler
    logger.info(f"Start profiler")
    profiler = BenchmarkProfiler()
    profiler.start("run")

    logger.info(f"Reading inputs...")
    profiler.start("loading_inputs")
    if len(input_prompts) == 1:  # Manual input
        input_prompts = input_prompts * global_batch_size
        all_prompts = input_prompts
    else:  # Inputs from file
        input_prompts, all_prompts = load_inputs(input_prompts, global_batch_size, instruct)
    profiler.end("loading_inputs")
    # To simulate a deployment environment, the demo supports repeating batched prompts.
    # This loop will rotate the prompts between the users for each batch, to simulate users sending different requests
    # If batch_size=1, the same prompt is repeated for each batch

    (
        model_args,
        model,
        page_table,
        tt_kv_cache,
        tokenizer,
        processor,
        local_data_parallel,
        local_submesh_indices,
    ) = prepare_generator_args(
        num_devices=num_devices,
        data_parallel=data_parallel,
        mesh_device=mesh_device,
        instruct=instruct,
        global_batch_size=global_batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        page_params=page_params,
        paged_attention=paged_attention,
        num_layers=num_layers,
        use_prefetcher=use_prefetcher,
        use_hf_rope=use_hf_rope,
        max_generated_tokens=max_generated_tokens,
    )

    global_batch_size = batch_size * local_data_parallel
    input_prompts = select_local_data_parallel_items(input_prompts, batch_size, data_parallel, local_submesh_indices)
    sampling_params = slice_sampling_params_for_local_submeshes(
        sampling_params, batch_size, data_parallel, local_submesh_indices
    )

    # Skip ci-eval tests on P100 devices
    if ("ci-eval-1" in test_id or "ci-eval-32" in test_id) and model_args[0].device_name == "P100":
        pytest.skip("ci-eval-1 and ci-eval-32 tests are not supported on P100 devices")

    if token_accuracy:
        token_acc = TokenAccuracy(model_name=model_args[0].model_name)

    for m_args in model_args:
        if m_args.max_context_len < max_seq_len:
            pytest.skip(
                f"Max seq len {max_seq_len} not supported by model {m_args.model_name}. The model's max context len is {m_args.max_context_len}"
            )

    generator = Generator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)

    if token_accuracy:
        input_prompts[0] = token_acc.prepare_ref_tokens(tokenizer)

    repeat_batch_prompts = []
    for i in range(repeat_batches):
        # For token accuracy, use input_prompts without rotation
        if token_accuracy:
            repeat_batch_prompts.append(input_prompts)
        else:
            global_prompts_for_batch = [all_prompts[(j + i) % len(all_prompts)] for j in range(len(all_prompts))][
                : batch_size * data_parallel
            ]
            repeat_batch_prompts.append(
                select_local_data_parallel_items(
                    global_prompts_for_batch, batch_size, data_parallel, local_submesh_indices
                )
            )

    num_tokens_generated_decode = []

    logger.info("Starting inference...")
    for batch_idx, input_prompts in enumerate(repeat_batch_prompts):
        logger.info(f"Processing batch {batch_idx}")
        profiler.start(f"preprocess_prefill_inputs", iteration=batch_idx)
        # Preprocess initial prompt inputs
        (
            input_tokens_prefill_pt,
            encoded_prompts,
            decoding_pos,
            prefill_lens,
        ) = preprocess_inputs_prefill(
            input_prompts, tokenizer, model_args, instruct, max_generated_tokens, max_prefill_len=max_seq_len
        )

        max_encoded_prompt_len = max(len(p) for p in encoded_prompts)
        assert (
            max_generated_tokens + max_encoded_prompt_len <= max_seq_len
        ), f"Prompt prefill tokens ({max_encoded_prompt_len}) + maximum number of decoded iterations ({max_generated_tokens}) needs to be <= than max_seq_len ({max_seq_len})"

        if paged_attention:
            paged_cache_max_seq_len = (
                page_params["page_block_size"] * page_params["page_max_num_blocks_per_dp"] / batch_size
            )
            assert (
                max_generated_tokens + max_encoded_prompt_len <= paged_cache_max_seq_len
            ), f"max_generated_tokens ({max_generated_tokens}) needs to be <= than paged_cache_max_seq_len ({paged_cache_max_seq_len})"
        profiler.end(f"preprocess_prefill_inputs", iteration=batch_idx)

        # when doing repeating batches, set kv-caches to zero, to avoid context leaking

        if batch_idx != 0:
            for i in range(len(model)):
                for layer in model[i].layers:
                    k_cache, v_cache = layer.attention.layer_past
                    k_cache = ttnn.mul(k_cache, 0, output_tensor=k_cache)
                    v_cache = ttnn.mul(v_cache, 0, output_tensor=v_cache)
            generator.prev_page_table = None

        input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(global_batch_size, -1)
        # Use device sampling for all cases when supported (prefill + decode)
        device_sampling_params = (
            SamplingParams(
                temperature=sampling_params["temperature"],
                top_k=sampling_params["top_k"],
                top_p=sampling_params["top_p"],
                seed=sampling_params["seed"] if "seed" in sampling_params else None,
                frequency_penalty=sampling_params["frequency_penalty"]
                if "frequency_penalty" in sampling_params
                else 0.0,
                presence_penalty=sampling_params["presence_penalty"] if "presence_penalty" in sampling_params else 0.0,
                repetition_penalty=sampling_params["repetition_penalty"]
                if "repetition_penalty" in sampling_params
                else 1.0,
                enable_log_probs=sampling_params["enable_log_probs"]
                if "enable_log_probs" in sampling_params
                else False,
            )
            if model[0]._supports_on_device_sampling
            else None
        )

        # Override the sampling params for some models
        # Issue: https://github.com/tenstorrent/tt-metal/issues/34763
        if model_args[0].base_model_name in models_not_supported_for_device_sampling:
            device_sampling_params = None

        if device_sampling_params is None and isinstance(sampling_params["temperature"], List):
            # host sampling only supports single sample param for all users in a batch
            sampling_params["temperature"] = sampling_params["temperature"][0]
            sampling_params["top_p"] = sampling_params["top_p"][0]
            sampling_params["enable_log_probs"] = sampling_params["enable_log_probs"][0]

        prefill_sampling_params = device_sampling_params if device_sampling_params is not None else None

        if mode == "prefill" or mode == "full":
            logger.info("Starting prefill warmup...")
            profiler.start(f"compile_prefill", iteration=batch_idx)
            logits = generator.prefill_forward_text(
                input_tokens_prefill_pt,  # Prefill warmup for all users, in case some users have different seqlens than others
                page_table=page_table,
                kv_cache=tt_kv_cache,
                prompt_lens=decoding_pos,
                sampling_params=prefill_sampling_params,
                warmup_prefill=not is_device_perf_test,
                enable_trace=enable_trace,
            )
            profiler.end(f"compile_prefill", iteration=batch_idx)
            logger.info("Finished prefill warmup")

            logger.info(f"Starting prefill...")
            profiler.start(f"inference_prefill", iteration=batch_idx)
            prefill_out = generator.prefill_forward_text(
                input_tokens_prefill_pt,
                page_table=page_table,
                kv_cache=tt_kv_cache,
                prompt_lens=decoding_pos,
                sampling_params=prefill_sampling_params,
                warmup_prefill=not is_device_perf_test,
                enable_trace=enable_trace,
            )
            if prefill_sampling_params is not None and isinstance(prefill_out, tuple):
                prefilled_token, prefill_log_probs = prefill_out
            else:
                logits = prefill_out
                prefilled_token = torch.argmax(logits, dim=-1)
            profiler.end(f"inference_prefill", iteration=batch_idx)
            logger.info(f"Prefill finished")
        else:
            # CI expects profiler to have these measurement keys so
            # they must be inserted regardless of whether we run prefill or not
            profiler.start(f"compile_prefill", iteration=batch_idx)
            profiler.end(f"compile_prefill", iteration=batch_idx)
            profiler.start(f"inference_prefill", iteration=batch_idx)
            profiler.end(f"inference_prefill", iteration=batch_idx)
            logger.info(f"Skipping prefill forward pass when decode mode is enabled")

            prefilled_token = torch.zeros(global_batch_size, 1, 1)

        # Keep track of generated outputs to print out every iteration
        all_outputs = [encoded_prompts[b][: prefill_lens[b]] for b in range(global_batch_size)]
        for user in range(global_batch_size):
            user_tok = int(prefilled_token[user].item())
            all_outputs[user].append(user_tok)

        user_done = [False] * global_batch_size  # Keeps track when a user reaches EoD token

        # Initial positions
        current_pos = torch.tensor([decoding_pos[b] if mode == "full" else 0 for b in range(global_batch_size)])

        # Start decoding
        iteration = 0
        users_decoding = True

        out_tok = prefilled_token

        logger.info(f"Starting decode loop...")

        # Log total inference (accounting for compile_decode as well)
        profiler.start(f"inference_decode", iteration=batch_idx)

        if mode == "prefill":
            # CI expects profiler to have these measurement keys so
            # they must be inserted regardless of whether we run decode or not
            profiler.start(f"compile_decode", iteration=batch_idx)
            profiler.end(f"compile_decode", iteration=batch_idx)
            profiler.start(f"inference_decode_time_{1}", iteration=batch_idx)
            profiler.end(f"inference_decode_time_{1}", iteration=batch_idx)
            logger.info(f"Skipping decode forward pass when prefill mode is enabled")

        while users_decoding and mode != "prefill":
            if iteration == 0:  # First iteration also accounts for compile time
                profiler.start(f"compile_decode", iteration=batch_idx)
            else:
                profiler.start(f"inference_decode_time_{iteration}", iteration=batch_idx)
            # below the collect method also applies teacher forcing which is necessary for exact token matching
            if token_accuracy:
                out_tok[0] = token_acc.collect_predicted_tokens(out_tok[0].item())

            # Run decode forward
            logits, log_probs = generator.decode_forward(
                out_tok,
                current_pos,
                enable_trace=enable_trace,
                page_table=page_table,
                kv_cache=tt_kv_cache,
                reset_batch=(iteration == 0),
                sampling_params=device_sampling_params,
                prompt_tokens=input_tokens_prefill_pt,
                output_tokens=out_tok,
            )

            # Get the next token
            if device_sampling_params is not None:
                out_tok = logits.unsqueeze(1)

            else:
                # TODO Fix use case with temperature > 0
                _, out_tok = sample_host(
                    logits,
                    temperature=sampling_params["temperature"],
                    top_p=sampling_params["top_p"],
                    on_host=True,
                )

            if iteration == 0:  # First iteration will account the compile time
                profiler.end(f"compile_decode", iteration=batch_idx)
                decode_iteration_time = profiler.get_duration("compile_decode", iteration=batch_idx)
                # Everything so far is prefill plus this JIT-compile step; drop it so the
                # AUTO_SHARD_PROFILE table reports steady-state decode only.
                inline_profile.reset()
            else:
                profiler.end(f"inference_decode_time_{iteration}", iteration=batch_idx)
                decode_iteration_time = profiler.get_duration(f"inference_decode_time_{iteration}", iteration=batch_idx)

            # Print perf after every iteration (skip in CI to avoid performance overhead)
            tokens_per_second_per_user = 1 / decode_iteration_time
            logger.debug(
                f"Iteration {iteration}: {1000 * decode_iteration_time:.0f}ms @ {tokens_per_second_per_user:.1f} tok/s/user ({global_batch_size * tokens_per_second_per_user:.1f} tok/s throughput)"
            )

            if not stress_test:  # During stress test runs we will iterate over the same position for X iterations
                current_pos += 1
            # Save output token to print out later
            for user in range(global_batch_size):
                user_tok = out_tok[user].item()
                if (
                    user_tok not in tokenizer.stop_tokens and user_done[user] == False
                ):  # Read until an eos token (e.g. <|eot_id|>); create_tokenizer adds stop_tokens to HF tokenizers
                    all_outputs[user].append(user_tok)
                else:
                    if (
                        stop_at_eos
                    ):  # For performance gathering in CI, we want to sometimes force decoding for a fixed number of iterations
                        user_done[user] = True
                        logger.trace(f"[User {user}] Finished decoding at iteration {iteration}")
                        if all(user_done):
                            users_decoding = False
                    else:
                        all_outputs[user].append(user_tok)

            # Print out generated outputs for each user at the end of every iteration
            for user in range(global_batch_size):
                text = "".join(tokenizer.decode(all_outputs[user]))
                if len(text) > 100:
                    text = "..." + text[-97:]
                text = text.replace("\n", " ")
                logger.debug("[User {}] {}".format(user, text))

            iteration += 1

            # Upper limit of generated tokens for each user
            if iteration >= max_generated_tokens:
                users_decoding = False

        # Final print
        if not users_decoding:
            profiler.start(f"log_saving_file", iteration=batch_idx)
            logger.info("Finished decoding, printing the final outputs...\n")
            for i, (output, prompt) in enumerate(zip(all_outputs, input_prompts)):
                text = tokenizer.decode(output)
                prompt_including_assistant_tags = tokenizer.decode(
                    model_args[0].encode_prompt(prompt, instruct=instruct)
                )
                text_after_prompt = text.replace(prompt_including_assistant_tags, "", 1)
                if print_to_file:
                    with open(output_filename, "a") as f:
                        f.write(f"\nbatch: {batch_idx} user: {i}\nprompt: {prompt} \noutput:\n{text_after_prompt}\n")
                else:
                    # Strip leading newlines from output when sent to terminal
                    short_prompt = (
                        (prompt[:100] + "\n<long prompt not printed in full>\n" + prompt[-100:])
                        if len(prompt) > 200
                        else prompt
                    )
                    logger.info(
                        f"\n==REPEAT BATCH {batch_idx}\n==USER {i} - PROMPT\n{short_prompt} \n==USER {i} - OUTPUT\n{text_after_prompt.strip()}\n"
                    )
            profiler.end(f"log_saving_file", iteration=batch_idx)

        num_tokens_generated_decode.append(iteration)  # Save the number of tokens generated for each repeat batch

        # Store outputs for repeat batch tests
        if "ci-eval-1" in test_id:
            if not hasattr(test_demo_text, "batch_outputs"):
                test_demo_text.batch_outputs = []
            if batch_idx == 0:
                test_demo_text.batch_outputs = []

            if all_outputs and len(all_outputs) > 0:
                final_output_text = tokenizer.decode(all_outputs[0])
                test_demo_text.batch_outputs.append(final_output_text)
                logger.info(f"Stored output for batch {batch_idx}: {final_output_text[:100]}...")

        if "ci-eval-32" in test_id:
            if not hasattr(test_demo_text, "batch32_outputs"):
                test_demo_text.batch32_outputs = []
            if batch_idx == 0:
                test_demo_text.batch32_outputs = []

            batch_outputs = []
            if all_outputs and len(all_outputs) > 0:
                for user_idx in range(len(all_outputs)):
                    final_output_text = tokenizer.decode(all_outputs[user_idx])
                    batch_outputs.append(final_output_text)
                test_demo_text.batch32_outputs.append(batch_outputs)
                logger.info(f"Stored outputs for batch {batch_idx}: {len(batch_outputs)} users")

        if token_accuracy:
            acc = token_acc.compute_accuracy()
            logger.info(f"=== Top1 and Top5 Token Accuracy ===")
            logger.info(f" Top1 Accuracy: {acc[0] * 100:.2f}%, Top5 Accuracy: {acc[1] * 100:.2f}%")

        profiler.end(f"inference_decode", iteration=batch_idx)

    # Finish profiling at the end of inference for all repeated batches
    profiler.end("run")

    # Quick sanity check that the model doesn't produce special tokens=garbage output
    is_special_tokens_produced = [False] * len(all_outputs)
    for i, output in enumerate(all_outputs):
        output = output[len(encoded_prompts[i]) :]
        is_eos = [token in tokenizer.stop_tokens for token in output]
        if any(is_eos):
            output = output[: is_eos.index(True)]
        is_special_tokens_produced[i] = any(token in tokenizer.all_special_ids for token in output)
    if any(is_special_tokens_produced):
        logger.warning(f"{sum(is_special_tokens_produced)}/{len(all_outputs)} users produced special tokens")
        if is_ci_env:
            assert False, "model produced special tokens"

    # Prepare profile benchmark metrics for the first repeat batch only
    compile_prefill_time = profiler.get_duration("compile_prefill") if mode != "decode" else 0
    compile_decode_time = profiler.get_duration("compile_decode") if mode != "prefill" else 0

    total_inference_prefill_time = profiler.get_duration("inference_prefill") if mode != "decode" else 0
    total_inference_decode_time = 0
    for i in range(1, num_tokens_generated_decode[0]):  # Iteration 0 is the compile time
        total_inference_decode_time += profiler.get_duration(f"inference_decode_time_{i}")

    # Average prefill time for each user
    avg_time_to_first_token = total_inference_prefill_time / global_batch_size

    # Average decode time per batch iteration
    avg_decode_iteration_time = (
        total_inference_decode_time / (num_tokens_generated_decode[0] - 1) if iteration > 1 else 0
    )

    prefill_tok_s = prefill_lens[0] / total_inference_prefill_time * global_batch_size if mode != "decode" else 0
    decode_tok_s_user = (
        (num_tokens_generated_decode[0] - 1) / total_inference_decode_time if mode != "prefill" and iteration > 1 else 0
    )  # Remove the compile time
    decode_tok_s = (
        ((num_tokens_generated_decode[0] - 1) / total_inference_decode_time * global_batch_size)
        if mode != "prefill" and iteration > 1
        else 0
    )  # Remove the compile time

    measurements = {
        # Required measurements
        "compile_prefill": compile_prefill_time,
        "compile_decode": compile_decode_time,
        "inference_prefill": total_inference_prefill_time,
        "inference_decode": total_inference_decode_time,
        "prefill_time_to_token": avg_time_to_first_token,
        "prefill_t/s": prefill_tok_s,  # tokens/s
        "decode_t/s/u": decode_tok_s_user,  # tokens/s/u
        "decode_t/s": decode_tok_s,  # tokens/s
        # Optional measurements
        "Total compile time": compile_prefill_time + compile_decode_time,
        "Full demo runtime": profiler.get_duration("run"),
    }

    # Decode performance for some specific tokens
    tok_1_perf = (
        profiler.get_duration(f"inference_decode_time_{1}") if 1 < num_tokens_generated_decode[0] else 0
    )  # Iteration 0 is compile time
    tok_128_perf = profiler.get_duration(f"inference_decode_time_{127}") if 127 < num_tokens_generated_decode[0] else 0
    tok_1024_perf = (
        profiler.get_duration(f"inference_decode_time_{1023}") if 1023 < num_tokens_generated_decode[0] else 0
    )
    tok_4096_perf = (
        profiler.get_duration(f"inference_decode_time_{4095}") if 4095 < num_tokens_generated_decode[0] else 0
    )

    if not stop_at_eos:
        logger.info(f"Please note that 'stop_at_eos' is disabled. Output repetition is expected.")

    logger.info("")
    logger.info(f"=== Performance metrics ===")
    if tok_1_perf > 0:
        logger.info(
            f"1st token decode time: {tok_1_perf * 1000:.2f}ms [{round(1 / tok_1_perf, 2)} t/s/u, {round((1 / tok_1_perf) * global_batch_size, 2)} t/s]"
        )
    if tok_128_perf > 0:
        logger.info(
            f"128th token decode time: {tok_128_perf * 1000:.2f}ms [{round(1 / tok_128_perf, 2)} t/s/u, {round((1 / tok_128_perf) * global_batch_size, 2)} t/s]"
        )
    if tok_1024_perf > 0:
        logger.info(
            f"1024th token decode time: {tok_1024_perf * 1000:.2f}ms [{round(1 / tok_1024_perf, 2)} t/s/u, {round((1 / tok_1024_perf) * global_batch_size, 2)} t/s]"
        )
    if tok_4096_perf > 0:
        logger.info(
            f"4096th token decode time: {tok_4096_perf * 1000:.2f}ms [{round(1 / tok_4096_perf, 2)} t/s/u, {round((1 / tok_4096_perf) * global_batch_size, 2)} t/s]"
        )

    # Print some of the perf metrics
    logger.info("==")
    logger.info(f"Prefill compile time: {round(compile_prefill_time, 2)}s")
    logger.info(f"Decode compile time: {round(compile_decode_time, 2)}s")
    logger.info("")
    logger.info(f"Average Time to First Token (TTFT): {round(avg_time_to_first_token * 1000, 2)}ms")
    logger.info(
        f"Average speed: {round(avg_decode_iteration_time * 1000, 2)}ms @ {round(decode_tok_s_user, 2)} tok/s/user ({round(decode_tok_s, 2)} tok/s throughput)"
    )

    # Benchmark targets
    tt_device_name = determine_device_name(mesh_device)  # submesh device should not decide performance target
    model_name = model_args[0].base_model_name
    resolved_perf_targets = resolve_perf_targets(
        model_name=model_name,
        sku=tt_device_name,
        batch_size=global_batch_size,
        seq_len=max_seq_len,
    )
    if not resolved_perf_targets:
        logger.info(
            f"Model {model_name} does not have centralized performance targets set for "
            f"device {tt_device_name}, batch_size={global_batch_size}, seq_len={max_seq_len}"
        )
        targets = {}
    else:
        targets = {
            "prefill_t/s": resolved_perf_targets.get("prefill_t/s"),
            "decode_t/s": resolved_perf_targets.get("decode_t/s"),
            "decode_t/s/u": resolved_perf_targets.get("decode_t/s/u"),
        }

    # Save benchmark data for CI dashboard
    if is_ci_env:
        # Instead of running warmup iterations, the demo profiles the initial compile iteration
        bench_n_warmup_iter = {"inference_prefill": 0, "inference_decode": 1}
        benchmark_data = create_benchmark_data(profiler, measurements, bench_n_warmup_iter, targets)

        # Save the decode performance of every iteration for plotting in superset
        for i in range(1, num_tokens_generated_decode[0]):
            benchmark_data.add_measurement(
                profiler,
                0,
                "inference_decode",
                f"time_to_token_{i}",
                profiler.get_duration(f"inference_decode_time_{i}") * 1000,
                step_warm_up_num_iterations=None,
                target=None,
            )

        # Also save the avg decode performance for the 128 iterations (excluding the compile time)
        num_iterations_for_avg = min(128, num_tokens_generated_decode[0])
        inference_decode_time_first_128 = sum(
            profiler.get_duration(f"inference_decode_time_{i}") for i in range(1, num_iterations_for_avg)
        )
        benchmark_data.add_measurement(
            profiler,
            0,
            "inference_decode",
            "avg_decode_time_first_128",
            inference_decode_time_first_128 * 1000 / max(1, num_iterations_for_avg - 1),
            step_warm_up_num_iterations=None,
            target=None,
        )
        if token_accuracy:
            benchmark_data.add_measurement(
                profiler,
                0,
                "inference_decode",
                "top1_token_accuracy",
                acc[0] * 100,
                step_warm_up_num_iterations=None,
                target=None,
            )
            benchmark_data.add_measurement(
                profiler,
                0,
                "inference_decode",
                "top5_token_accuracy",
                acc[1] * 100,
                step_warm_up_num_iterations=None,
                target=None,
            )
        benchmark_data.save_partial_run_json(
            profiler,
            run_type="demo",
            ml_model_name=model_name,
            ml_model_type="llm",
            device_name=tt_device_name,
            num_layers=model_args[0].n_layers,
            batch_size=global_batch_size,
            config_params={"data_parallel": data_parallel, "tensor_parallel": num_devices // data_parallel},
            input_sequence_length=max(prefill_lens),
            output_sequence_length=num_tokens_generated_decode[0],
        )

        # check measurements against CI performance targets -- for batch size 32
        if "performance" in test_id and "ci-32" in test_id:
            logger.info(
                f"Checking measurements against CI performance targets for batch size 32 of {model_name} on {tt_device_name}"
            )
            verify_perf(
                measurements,
                expected_measurements={
                    "prefill_time_to_token": True,
                    "decode_t/s": True,
                    "decode_t/s/u": True,
                },
                model_name=model_name,
                sku=tt_device_name,
                batch_size=global_batch_size,
                seq_len=max_seq_len,
            )
    if token_accuracy:
        total_top1_acc = math.ceil(acc[0] * 100)
        total_top5_acc = math.ceil(acc[1] * 100)

        if not json_config_file:
            # Get accuracy thresholds from centralized model targets unless the config is loaded from json.
            min_top1_acc, min_top5_acc = get_accuracy_thresholds(model_args[0], seq_len=max(prefill_lens))
            verify_accuracy(
                measurements={
                    "top1_token_accuracy": total_top1_acc,
                    "top5_token_accuracy": total_top5_acc,
                },
                expected_accuracy_metrics={"top1": min_top1_acc, "top5": min_top5_acc},
            )
            logger.info("Checks of top-1 and top-5 accuracy against centralized model targets passed")

    if (
        "ci-eval-1" in test_id
        and hasattr(test_demo_text, "batch_outputs")
        and len(test_demo_text.batch_outputs) == repeat_batches
    ):
        logger.info("=== Repeat Batch Output Comparison ===")

        # Compare paired batches (A<->A, B<->B, C<->C)
        comparisons = [(i, i + 1) for i in range(0, repeat_batches, 2)]
        comparison_names = [f"Batch{i // 2 + 1}<->Batch{i // 2 + 1}" for i in range(0, repeat_batches, 2)]

        all_matches = True
        for i, (batch1_idx, batch2_idx) in enumerate(comparisons):
            output1 = test_demo_text.batch_outputs[batch1_idx]
            output2 = test_demo_text.batch_outputs[batch2_idx]

            if output1 == output2:
                logger.info(
                    f"{comparison_names[i]} comparison PASSED: Batches {batch1_idx + 1} and {batch2_idx + 1} outputs match"
                )
            else:
                logger.warning(
                    f"{comparison_names[i]} comparison FAILED: Batches {batch1_idx + 1} and {batch2_idx + 1} outputs differ"
                )
                logger.info(f"  Batch {batch1_idx + 1} output: {output1[:100]}...")
                logger.info(f"  Batch {batch2_idx + 1} output: {output2[:100]}...")
                all_matches = False

        assert all_matches, "Repeat batch outputs should be identical"

    if (
        "ci-eval-32" in test_id
        and hasattr(test_demo_text, "batch32_outputs")
        and len(test_demo_text.batch32_outputs) > 1
    ):
        logger.info("=== Batch32 Shifting Test Output Comparison ===")

        num_batches = len(test_demo_text.batch32_outputs)
        num_users = len(test_demo_text.batch32_outputs[0])

        logger.info(f"Comparing {num_batches} batches with {num_users} users each")

        consistency_checks = []

        for batch_idx in range(num_batches - 1):
            current_batch = test_demo_text.batch32_outputs[batch_idx]
            next_batch = test_demo_text.batch32_outputs[batch_idx + 1]

            logger.info(f"Comparing batch {batch_idx} vs batch {batch_idx + 1}")

            for user_offset in range(num_users):
                current_user_idx = (user_offset + 1) % num_users
                next_user_idx = user_offset

                current_output = current_batch[current_user_idx]
                next_output = next_batch[next_user_idx]

                expected_prompt_idx = (user_offset + batch_idx + 1) % num_users

                if current_output == next_output:
                    consistency_checks.append(True)
                    logger.info(
                        f"User {current_user_idx} (batch {batch_idx}) matches User {next_user_idx} (batch {batch_idx + 1}) - both used prompt[{expected_prompt_idx}]"
                    )
                else:
                    consistency_checks.append(False)
                    logger.warning(
                        f"User {current_user_idx} (batch {batch_idx}) differs from User {next_user_idx} (batch {batch_idx + 1}) - both should use prompt[{expected_prompt_idx}]"
                    )
                    logger.info(f"  Batch {batch_idx} User {current_user_idx}: {current_output[:50]}...")
                    logger.info(f"  Batch {batch_idx + 1} User {next_user_idx}: {next_output[:50]}...")

        all_consistent = all(consistency_checks)
        failed_checks = sum(1 for check in consistency_checks if not check)
        total_checks = len(consistency_checks)

        assert (
            all_consistent
        ), f"Batch32 repeat batch outputs should be identical - {failed_checks} out of {total_checks} consistency checks failed"
