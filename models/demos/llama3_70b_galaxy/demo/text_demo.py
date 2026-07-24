# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from loguru import logger
from datetime import datetime
import hashlib
import requests
import json
import math
import torch
import pytest
import os
import ttnn

from models.demos.llama3_70b_galaxy.tt.generator import Generator, SamplingParams
from models.demos.llama3_70b_galaxy.tt.model_config import LlamaOptimizations
from models.tt_transformers.tt.common import (
    preprocess_inputs_prefill,
    PagedAttentionConfig,
)
from models.perf.benchmarking_utils import BenchmarkProfiler, BenchmarkData
from models.common.utility_functions import (
    comp_pcc,
)
from models.demos.utils.device_sku import get_current_device_sku_name
from models.demos.utils.llm_demo_utils import verify_perf
from models.demos.utils.model_targets import resolve_perf_targets
from models.demos.utils.trace_region_sizes import TRACE_MODEL_KEY_PARAM


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
def load_inputs(user_input, len_per_batch, instruct):
    if isinstance(user_input, str):
        with open(user_input, "r") as f:
            user_input = json.load(f)
    batch = len(len_per_batch)
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
            # TODO This might override the expected input size give in the prompt file
            if "max_length" in user_input[i]:  # Clip the context to the max length provided
                context_text = load_and_cache_context(
                    user_input[i]["context"],
                    cache_dir,
                    max_length=(user_input[i]["max_length"]) if batch == 1 else len_per_batch[i],
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


class TokenAccuracy:
    def __init__(self, model_name):
        self.gt_pos = -1
        self.store_predicted_tokens = []
        reference_data_file = os.path.join("models/tt_transformers/tests/reference_outputs/", model_name) + ".refpt"
        assert os.path.exists(reference_data_file), f"Reference data file {reference_data_file} does not exist"
        logger.info(f"Loading reference data from {reference_data_file}")
        reference_data = torch.load(reference_data_file)
        reference_tokens = reference_data["reference_tokens"]
        split_point = reference_tokens.shape[-1] // 2
        self.input_prompt = reference_tokens[0, :split_point]
        self.reference_tokens = reference_tokens[0, split_point:]
        self.top5_tokens = reference_data["top5_tokens"][split_point - 1 :, :]
        self.maxindex = len(self.reference_tokens) - 1

    def prepare_ref_tokens(self, tokenizer):
        return tokenizer.decode(self.input_prompt.tolist())

    def collect_predicted_tokens(self, token):
        token = token.item() if hasattr(token, "item") else token
        self.store_predicted_tokens.append(int(token))
        self.gt_pos += 1
        return self.reference_tokens[min(self.gt_pos, self.maxindex)].unsqueeze(-1).unsqueeze(-1)

    def compute_accuracy(self):
        count = 0
        count_t5 = 0
        matching_sz = min(len(self.reference_tokens), len(self.store_predicted_tokens))
        assert matching_sz > 0, "No tokens collected for token accuracy"
        for i in range(matching_sz):
            if self.top5_tokens[i, 0].item() == self.store_predicted_tokens[i]:
                count += 1
            if self.store_predicted_tokens[i] in self.top5_tokens[i, :]:
                count_t5 += 1

        return count / matching_sz, count_t5 / matching_sz


def get_accuracy_thresholds(model_args):
    """Parse token accuracy thresholds from the common PERF.md Performance table."""
    perf_file = "models/tt_transformers/PERF.md"
    with open(perf_file, "r") as f:
        content = f.read()

    sections = content.split("## ")
    target_section = next(s for s in sections if s.lower().startswith("performance\n"))

    base_model_name = model_args.base_model_name
    device_name = model_args.device_name
    correct_line = (
        lambda line: "|" in line
        and base_model_name.lower() in line.split("|")[1].strip().lower()
        and device_name.lower() in line.split("|")[2].strip().lower()
        and not "(DP=".lower() in line.lower()
    )
    rows = [line.split("|")[1:] for line in target_section.split("\n") if correct_line(line)]
    if not rows:
        raise ValueError(f"Could not find accuracy data for {base_model_name} on {device_name} in {perf_file}")

    assert len(rows) == 1, f"Found multiple rows for {base_model_name} on {device_name} in {perf_file}"
    row = rows[0]
    top1_acc = float(row[2].strip())
    top5_acc = float(row[3].strip())

    return top1_acc - 0.5, top5_acc - 0.5


def load_demo_targets(filename, galaxy_type):
    """
    Load expected demo targets from a JSON file.
    """
    if not os.path.exists(filename):
        logger.warning(f"Expected outputs file {filename} does not exist. Skipping loading targets.")
        return []

    with open(filename, "r") as f:
        try:
            demo_targets = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {filename}: {e}. Returning empty list.")
            return []

    demo_targets = demo_targets["targets"][galaxy_type]

    return demo_targets


def create_tt_model(
    mesh_device,
    instruct,
    max_batch_size,
    optimizations,
    max_seq_len,
    num_layers,
    dummy_weights,
    page_params,
    dtype=ttnn.bfloat8_b,
    use_paged_kv_cache=False,
    prefill_profile=False,
):
    from models.demos.llama3_70b_galaxy.tt.llama_model import TtTransformer
    from models.demos.llama3_70b_galaxy.tt.model_config import TtModelArgs

    tt_model_args = TtModelArgs(
        mesh_device,
        instruct=instruct,
        max_batch_size=32,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        dummy_weights=dummy_weights,
    )
    # When running running prefill-only profile, run just 1 layer
    tt_model_args.n_layers = num_layers if not prefill_profile else 1

    state_dict = tt_model_args.load_state_dict()
    page_table = None
    paged_attention_config = None
    tt_kv_cache = None

    if use_paged_kv_cache:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            max_batch_size, paged_attention_config.max_num_blocks // max_batch_size
        )

    model = TtTransformer(
        args=tt_model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=tt_model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
        mode="prefill",
        enable_prefetcher_performance_mode=True,
    )

    if use_paged_kv_cache:
        tt_kv_cache = [l.attention.layer_past for l in model.layers]

    return tt_model_args, model, page_table, [tt_kv_cache]


# List of supported Parameters for demo.py
#
# input_prompts (string): input json file with prompts to process. See models/demos/llama3_70b_galaxy/demo/sample_prompts/*.json for list of input files
# instruct (bool): Whether to use instruct weights or general weights
# repeat_batches (int): Number of consecutive batches of users to run (default: 1)
# max_seq_len (int): Maximum context length supported by the model (Llama-3.1 and Llama-3.2 models have a maximum context length of 128k, i.e., 128 * 1024)
# batch_size (int): Number of users in a batch (Supports 1/2/4/8/16/32 batches)
# max_generated_tokens (int): Maximum number of tokens to generate for each user (Note that the users will stop generation before this limit if they reach a EoS token)
# paged_attention (bool): Whether to use paged attention or default attention (vLLM requires paged attention)
# page_params (dict): Page parameters for paged attention (block_size, max_num_blocks) For smaller context lengths use block_size=32 and max_num_blocks=1024, for larger context use block_size=64 and max_num_blocks=2048
# sampling_params (dict): Sampling parameters for decoding (temperature, top_p). If temperature is set to 0, argmax (greedy decode) is used.
# stop_at_eos (bool): Whether to stop decoding when the model generates an EoS token
# token_accuracy (bool): Whether to run teacher-forced top-1/top-5 token matching against reference outputs
# is_cur_pos_sharded (bool): Whether to replicate the cur pos tensor on sub core grid as an optimization
# is_page_table_sharded (bool):  Whether to replicate the page table tensor on sub core grid as an optimization (Currently page table sharding is only supported for BS=32)


# optimization (LlamaOptimizations): Optimization level to use for the model (performance or accuracy)
@pytest.mark.parametrize(
    "input_prompts, instruct, repeat_batches, max_seq_len, batch_size, max_generated_tokens, paged_attention, page_params, sampling_params, stop_at_eos, apc_test, pcc_check, token_accuracy, prefill_profile, num_layers, print_outputs, is_cur_pos_sharded, is_page_table_sharded, use_prefix_caching, prefix_cached_ratio",
    [
        (  # Batch-32 run (Throughput) - 32 users, small prompt
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            32,  # batch_size
            128,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 2048},  # page_params
            {"temperature": 0.0, "top_p": 0.08},  # sampling_params (argmax)
            False,  # stop_at_eos
            False,  # apc_test
            False,  # pcc_check
            False,  # token_accuracy
            False,  # prefill-only profile
            80,  # num layers
            False,  # print_outputs
            True,  # is_cur_pos_sharded
            True,  # is_page_table_sharded
            False,  # use_prefix_caching
            0.0,  # prefix_cached_ratio
        ),
        (  # Batch-32 with non-uniform sampling
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            32,  # batch_size
            128,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 2048},  # page_params
            {
                "temperature": torch.linspace(0.0, 1.0, steps=32).tolist(),
                "top_p": torch.linspace(0.08, 1.0, steps=32).tolist(),
                "top_k": torch.arange(1, 33).tolist(),  # 1 to 32 inclusive
                "presence_penalty": torch.linspace(-2.0, 2.0, steps=32).tolist(),
                "frequency_penalty": torch.linspace(-2.0, 2.0, steps=32).tolist(),
                "repetition_penalty": torch.linspace(0.8, 1.5, steps=32).tolist(),
                "seed": torch.randint(0, 33, size=(32,)).tolist(),
            },  # sampling_params (non-uniform)
            False,  # stop_at_eos
            False,  # apc_test
            False,  # pcc_check
            False,  # token_accuracy
            False,  # prefill-only profile
            80,  # num layers
            False,  # print_outputs
            True,  # is_cur_pos_sharded
            True,  # is_page_table_sharded
            False,  # use_prefix_caching
            0.0,  # prefix_cached_ratio
        ),
        (  # Batch-32 with non-uniform sampling and log-probs calculation
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            32,  # batch_size
            128,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 2048},  # page_params
            {
                "temperature": torch.linspace(0.0, 1.0, steps=32).tolist(),
                "top_p": torch.linspace(0.08, 1.0, steps=32).tolist(),
                "top_k": torch.arange(1, 33).tolist(),  # 1 to 32 inclusive
                "log_probs": [True] * 32,
            },  # sampling_params (non-uniform)
            False,  # stop_at_eos
            False,  # apc_test
            False,  # pcc_check
            False,  # token_accuracy
            False,  # prefill-only profile
            80,  # num layers
            False,  # print_outputs
            True,  # is_cur_pos_sharded
            True,  # is_page_table_sharded
            False,  # use_prefix_caching
            0.0,  # prefix_cached_ratio
        ),
        (  # Batch-1 run (Throughput) - 1 user, small prompt
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            1,  # batch_size
            128,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 2048},  # page_params
            {"temperature": 0.0, "top_p": 0.05},  # sampling_params (argmax)
            False,  # stop_at_eos
            False,  # apc_test
            False,  # pcc_check
            False,  # token_accuracy
            False,  # prefill-only profile
            80,  # num layers
            False,  # print_outputs
            False,  # is_cur_pos_sharded
            False,  # is_page_table_sharded
            False,  # use_prefix_caching
            0.0,  # prefix_cached_ratio
        ),
        (  # evals-1 run (Throughput) - 1 user, smaller prompts, batch repeat 32
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/eval_repeat_prompts.json",  # input_prompts
            True,  # instruct mode
            16,  # repeat_batches
            128 * 1024,  # max_seq_len
            1,  # batch_size
            1024,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 2048},  # page_params
            {"temperature": 0.0, "top_p": 0.05},  # sampling_params (argmax)
            False,  # stop_at_eos
            False,  # apc_test
            False,  # pcc_check
            False,  # token_accuracy
            False,  # prefill-only profile
            80,  # num layers
            False,  # print_outputs
            False,  # is_cur_pos_sharded
            False,  # is_page_table_sharded
            False,  # use_prefix_caching
            0.0,  # prefix_cached_ratio
        ),
        (  # evals-32 run (Throughput) - 32 users, smaller prompts, batch repeat 32
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/eval_repeat_prompts_debug.json",  # input_prompts
            True,  # instruct mode
            16,  # repeat_batches
            128 * 1024,  # max_seq_len
            32,  # batch_size
            1024,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 2048},  # page_params
            {"temperature": 0.0, "top_p": 0.05},  # sampling_params (argmax)
            False,  # stop_at_eos
            False,  # apc_test
            False,  # pcc_check
            False,  # token_accuracy
            False,  # prefill-only profile
            80,  # num layers
            False,  # print_outputs
            False,  # is_cur_pos_sharded
            False,  # is_page_table_sharded
            False,  # use_prefix_caching
            0.0,  # prefix_cached_ratio
        ),
        (  # evals-long-prompts run (Throughput) - 1 user, smaller prompts, batch repeat 12
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/eval_repeat_prompts_very_long.json",  # input_prompts
            True,  # instruct mode
            6,  # repeat_batches
            128 * 1024,  # max_seq_len
            1,  # batch_size
            1024,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 2048},  # page_params
            {"temperature": 0.0, "top_p": 0.05},  # sampling_params (argmax)
            False,  # stop_at_eos
            False,  # apc_test
            False,  # pcc_check
            False,  # token_accuracy
            False,  # prefill-only profile
            80,  # num layers
            False,  # print_outputs
            False,  # is_cur_pos_sharded
            False,  # is_page_table_sharded
            False,  # use_prefix_caching
            0.0,  # prefix_cached_ratio
        ),
        (  # Repeat2 (Batch-1) run (Throughput) - 1 user, small prompt
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            2,  # repeat_batches
            128 * 1024,  # max_seq_len
            1,  # batch_size
            128,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 2048},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            False,  # stop_at_eos
            False,  # apc_test
            False,  # pcc_check
            False,  # token_accuracy
            False,  # prefill-only profile
            80,  # num layers
            False,  # print_outputs
            False,  # is_cur_pos_sharded    #NOTE: currently cur pos/ page table sharding is not supported on repeat batch runs
            False,  # is_page_table_sharded
            False,  # use_prefix_caching
            0.0,  # prefix_cached_ratio
        ),
        (  # long-4k-b1 - Single user, 4K long prompt
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_4k.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            1,  # batch_size
            128,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 2048},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            False,  # stop_at_eos
            False,  # apc_test
            False,  # pcc_check
            False,  # token_accuracy
            False,  # prefill-only profile
            80,  # num layers
            False,  # print_outputs
            False,  # is_cur_pos_sharded
            False,  # is_page_table_sharded
            False,  # use_prefix_caching
            0.0,  # prefix_cached_ratio
        ),
        (  # long-8k-b1 - Single user, 8K long prompt
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_8k.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            1,  # batch_size
            128,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 2048},  # page_params
            {"temperature": 1.0, "top_p": 0.04},  # sampling_params (argmax)
            False,  # stop_at_eos
            False,  # apc_test
            False,  # pcc_check
            False,  # token_accuracy
            False,  # prefill-only profile
            80,  # num layers
            False,  # print_outputs
            False,  # is_cur_pos_sharded
            False,  # is_page_table_sharded
            False,  # use_prefix_caching
            0.0,  # prefix_cached_ratio
        ),
        (  # long-16k-b1 - 1 user, 16K long prompt
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_16k.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            1,  # batch_size
            128,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 2048},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            False,  # stop_at_eos
            False,  # apc_test
            False,  # pcc_check
            False,  # token_accuracy
            False,  # prefill-only profile
            80,  # num layers
            False,  # print_outputs
            False,  # is_cur_pos_sharded
            False,  # is_page_table_sharded
            False,  # use_prefix_caching
            0.0,  # prefix_cached_ratio
        ),
        (  # long-32k-b1 - Single user, 32K long prompt
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_32k.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            1,  # batch_size
            128,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 2048},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            False,  # stop_at_eos
            False,  # apc_test
            False,  # pcc_check
            False,  # token_accuracy
            False,  # prefill-only profile
            80,  # num layers
            False,  # print_outputs
            False,  # is_cur_pos_sharded
            False,  # is_page_table_sharded
            False,  # use_prefix_caching
            0.0,  # prefix_cached_ratio
        ),
        (  # long-64k-b1 - Single user, 64K long prompt
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_64k.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            1,  # batch_size
            128,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 2048},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            False,  # stop_at_eos
            False,  # apc_test
            False,  # pcc_check
            False,  # token_accuracy
            False,  # prefill-only profile
            80,  # num layers
            False,  # print_outputs
            False,  # is_cur_pos_sharded
            False,  # is_page_table_sharded
            False,  # use_prefix_caching
            0.0,  # prefix_cached_ratio
        ),
        (  # long-128k-b1 - Single user, 128K long prompt
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_128k.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            1,  # batch_size
            128,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 2048},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            False,  # stop_at_eos
            False,  # apc_test
            False,  # pcc_check
            False,  # token_accuracy
            False,  # prefill-only profile
            80,  # num layers
            False,  # print_outputs
            False,  # is_cur_pos_sharded
            False,  # is_page_table_sharded
            False,  # use_prefix_caching
            0.0,  # prefix_cached_ratio
        ),
        (  # prefill-profile-standard [default 4K seqlen] - Runs 1L prefill-only
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_4k.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            1,  # batch_size
            10,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 2048},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            False,  # stop_at_eos
            False,  # apc_test
            False,  # pcc_check
            False,  # token_accuracy
            True,  # prefill-only profile
            1,  # num layers
            False,  # print_outputs
            False,  # is_cur_pos_sharded
            False,  # is_page_table_sharded
            False,  # use_prefix_caching
            0.0,  # prefix_cached_ratio
        ),
        (  # prefill-profile-prefix-caching - Runs 1L, Phase 2 only (prefix-cached prefill, signposts around Phase 2)
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_4k.json",  # input_prompts (need >=128 tokens for 50% cache to align to page_block_size 64)
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            1,  # batch_size
            10,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 2048},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            False,  # stop_at_eos
            False,  # apc_test
            False,  # pcc_check
            False,  # token_accuracy
            True,  # prefill-only profile
            1,  # num layers
            False,  # print_outputs
            False,  # is_cur_pos_sharded
            False,  # is_page_table_sharded
            True,  # use_prefix_caching
            0.5,  # prefix_cached_ratio
        ),
        (  # apc-test Run for PCC check, perf and functionality check: Batch-32 run (Throughput) - 32 users, prompt is "This is a test"
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_reference.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            32,  # batch_size
            130,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 2048},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            False,  # stop_at_eos
            True,  # apc_test
            True,  # pcc_check
            False,  # token_accuracy
            False,  # prefill-only profile
            80,  # num layers
            False,  # print_outputs
            False,  # is_cur_pos_sharded
            False,  # is_page_table_sharded
            False,  # use_prefix_caching
            0.0,  # prefix_cached_ratio
        ),
        (  # ci-token-matching - CI Run for teacher-forced top-1/top-5 token accuracy
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            1024,  # max_seq_len
            1,  # batch_size
            500,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 2048},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            True,  # stop_at_eos
            False,  # apc_test
            False,  # pcc_check
            True,  # token_accuracy
            False,  # prefill-only profile
            80,  # num layers
            False,  # print_outputs
            False,  # is_cur_pos_sharded
            False,  # is_page_table_sharded
            False,  # use_prefix_caching
            0.0,  # prefix_cached_ratio
        ),
        (  # pcc-80L - CI Run for PCC check for 80 Layers + Teacher Forced: Batch-32 run (Throughput) - 32 users, prompt is "This is a test"
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_reference.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            1024,  # max_seq_len
            32,  # batch_size
            20,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 2048},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            False,  # stop_at_eos
            False,  # apc_test
            True,  # pcc_check
            False,  # token_accuracy
            False,  # prefill-only profile
            80,  # num layers
            False,  # print_outputs
            False,  # is_cur_pos_sharded
            False,  # is_page_table_sharded
            False,  # use_prefix_caching
            0.0,  # prefix_cached_ratio
        ),
        (  # batch-1-prefix-caching-perf - 1 user, long enough prompt, prefix caching (performance)
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_4k.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            1,  # batch_size
            128,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 2048},  # page_params
            {"temperature": 0.0, "top_p": 0.05},  # sampling_params (argmax)
            False,  # stop_at_eos
            False,  # apc_test
            False,  # pcc_check
            False,  # token_accuracy
            False,  # prefill-only profile
            80,  # num layers
            False,  # print_outputs
            False,  # is_cur_pos_sharded
            False,  # is_page_table_sharded
            True,  # use_prefix_caching
            0.75,  # prefix_cached_ratio
        ),
        (  # batch-1-prefix-caching-pcc - 1 user, prefix caching with PCC (correctness)
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_reference.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            1,  # batch_size
            20,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 2048},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            False,  # stop_at_eos
            False,  # apc_test
            True,  # pcc_check
            False,  # token_accuracy
            False,  # prefill-only profile
            80,  # num layers
            False,  # print_outputs
            False,  # is_cur_pos_sharded
            False,  # is_page_table_sharded
            True,  # use_prefix_caching
            0.5,  # prefix_cached_ratio
        ),
        (  # seqlen-sweep - Sweeps all powers-of-two seqlens up to model's max (128k), one prefill+decode per batch
            [
                "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_1k.json",
                "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_2k.json",
                "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_4k.json",
                "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_8k.json",
                "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_16k.json",
                "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_32k.json",
                "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_64k.json",
                "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_128k.json",
            ],  # input_prompts (list of files, one per sweep step)
            True,  # instruct mode
            8,  # repeat_batches (one per seqlen step; adjusted dynamically by sweep logic)
            128 * 1024,  # max_seq_len (model maximum; filters out files that exceed this)
            1,  # batch_size
            32,  # max_generated_tokens (minimal decode to verify prefill succeeds at each length)
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 2048},  # page_params
            {"temperature": 0, "top_p": 0.08},  # sampling_params (argmax)
            True,  # stop_at_eos
            False,  # apc_test
            False,  # pcc_check
            False,  # token_accuracy
            False,  # prefill-only profile
            80,  # num layers
            False,  # print_outputs
            False,  # is_cur_pos_sharded
            False,  # is_page_table_sharded
            False,  # use_prefix_caching
            0.0,  # prefix_cached_ratio
        ),
    ],
    ids=[
        "batch-32",  # throughput
        "batch-32-non-uniform-sampling",  # throughput w/ non-uniform sampling
        "batch-32-log-probs",  # throughput w/ non-uniform sampling and log-probs calculation
        "batch-1",  # latency
        "evals-1",  # Single user, 32 repeated batches, smaller prompts (<4K)
        "evals-32",  # 32 users, 32 repeated batches, smaller prompts (<4K)
        "evals-long-prompts",  # Single user, 12 repeated batches, very long prompts (4K ~ 64K)
        "repeat2",  # latency with 2 repeat batches
        "long-4k-b1",  # 4k context for 1 user
        "long-8k-b1",  # 4k context for 1 user
        "long-16k-b1",  # 16K context for 1 user
        "long-32k-b1",  # 32k context for 1 user
        "long-64k-b1",  # 64k context for 1 user
        "long-128k-b1",  # 128k context for 1 user
        "prefill-profile-standard",  # prefill-only profile run
        "prefill-profile-prefix-caching",  # prefill-only, Phase 2 (prefix-cached) only, 50% cache
        "apc-test",  # apc check for 80L + teacher forced for prefill + pcc check on prefill and 1st decode token
        "ci-token-matching",  # CI performs token accuracy matching with precomputed reference tokens
        "pcc-80L",  # pcc check for 80L + teacher forced
        "batch-1-prefix-caching-perf",  # 1 user, prefix caching (performance)
        "batch-1-prefix-caching-pcc",  # 1 user, prefix caching with PCC (correctness)
        "seqlen-sweep",  # Sweep prefill across all powers-of-two seqlens up to model's max (128k)
    ],
)
@pytest.mark.parametrize(
    "optimizations",
    [
        LlamaOptimizations.performance,
    ],
)
@pytest.mark.parametrize(
    "pcc_decode_len",
    [10],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            TRACE_MODEL_KEY_PARAM: "llama3.3-70b-galaxy",
            "num_command_queues": 1,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "worker_l1_size": 1345000,
            "fabric_config": True,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
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
    device_params,
    is_ci_env,
    apc_test,
    prefill_profile,
    num_layers,
    pcc_check,
    token_accuracy,
    pcc_decode_len,
    reset_seeds,
    request,
    galaxy_type,
    print_outputs,
    is_cur_pos_sharded,
    is_page_table_sharded,
    use_prefix_caching,
    prefix_cached_ratio,
):
    """
    Simple demo with limited dependence on reference code.
    """
    if use_prefix_caching and batch_size != 1:
        pytest.skip("Prefix caching only supported for batch_size=1")

    # Reset prefetcher global so each test gets a clean state (avoids reusing
    # address tensor from a previous test, which causes device mismatch TT_FATAL).
    import models.demos.llama3_70b_galaxy.tt.prefetcher_common as prefetcher_common

    prefetcher_common.global_tt_tensor_address = None

    # TODO: Remove this once all batch sizes are supported on TG
    if os.environ.get("MESH_DEVICE") == "TG" and batch_size not in [1, 32]:
        pytest.skip("Llama TG only supports batch-32")
    if apc_test and not pcc_check:
        raise ValueError("APC test requires PCC check to be enabled")
    if apc_test:
        demo_targets = load_demo_targets("models/demos/llama3_70b_galaxy/demo/text_demo_targets.json", galaxy_type)

    if prefill_profile:  # Special mode where we only run prefill with tracy
        from tracy import signpost

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
    paged_attention = request.config.getoption("--paged_attention") or paged_attention
    page_params = request.config.getoption("--page_params") or page_params
    sampling_params = request.config.getoption("--sampling_params") or sampling_params
    if request.config.getoption("--stop_at_eos") in [
        0,
        1,
    ]:  # If the flag is provided, use it. Take an int instead of bool due to parser limitations
        stop_at_eos = request.config.getoption("--stop_at_eos")
    print_outputs = request.config.getoption("--print_outputs") or print_outputs
    token_accuracy = request.config.getoption("--token_accuracy") or token_accuracy

    if token_accuracy and pcc_check:
        raise ValueError("Token accuracy and PCC check are separate demo modes")
    if token_accuracy and batch_size != 1:
        raise ValueError("Token accuracy mode only supports batch_size=1")

    test_id = request.node.callspec.id
    is_seqlen_sweep = "seqlen-sweep" in test_id

    enable_trace = True  # Use tracing for better perf
    if token_accuracy:
        enable_trace = False  # Teacher forcing uses fresh host tokens each iteration.
    prefill_enable_trace = not is_seqlen_sweep  # disable only for sweep to avoid 80MB trace budget
    print_to_file = False  # Enable this flag to print the output of all users to a file
    instruct = num_layers == 80 and instruct  # if using instruct weights it must be full model
    input_lengths = (
        [
            534,
            1008,
            1111 * 4,
            3333 * 4,
        ]
        * 8
        if batch_size == 32
        else [15384 * 8]
    )

    # Creat batch output file
    benchmark_data = BenchmarkData()
    profiler_step_name = "tg-llama-demo-prefill-e2e"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_directory = "models/demos/llama3_70b_galaxy/demo/output"
    os.makedirs(output_directory, exist_ok=True)
    os.chmod(output_directory, 0o755)
    output_filename = f"{output_directory}/demo_user_output_{timestamp}.txt"

    if not stop_at_eos:
        logger.info(f"The decode generation will only stop at the max_generated_tokens limit == {max_generated_tokens}")

    if print_to_file:
        # Creat batch output file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_directory = "models/demos/llama3_70b_galaxy/demo/output"
        os.makedirs(output_directory, exist_ok=True)
        os.chmod(output_directory, 0o755)
        output_filename = f"{output_directory}/llama_text_demo_output_{timestamp}.txt"

    # Start profiler
    logger.info(f"Start profiler")
    profiler = BenchmarkProfiler()
    profiler.start("run")

    logger.info(f"Reading inputs...")
    profiler.start("loading_inputs")
    sweep_prompt_files = None
    if is_seqlen_sweep:
        # In seqlen-sweep mode, input_prompts is a list of file paths (one per sweep step).
        # Load the first file now for initial setup; per-step loading happens later.
        sweep_prompt_files = input_prompts
        input_prompts, all_prompts = load_inputs(
            sweep_prompt_files[0],
            input_lengths,
            instruct,
        )
    else:
        input_prompts, all_prompts = load_inputs(
            input_prompts,
            input_lengths,
            instruct,
        )
    profiler.end("loading_inputs")

    # Load expected outputs for comparison
    expected_outputs_data = []
    # Always use this specific path for the expected outputs.
    expected_outputs_file_path_to_load = "models/demos/llama3_70b_galaxy/demo/outputs_batch_1.json"

    if os.path.exists(expected_outputs_file_path_to_load):
        logger.info(f"Attempting to load expected outputs from: {expected_outputs_file_path_to_load}")
        try:
            with open(expected_outputs_file_path_to_load, "r") as f:
                first_char = f.read(1)
                if not first_char:
                    logger.warning(
                        f"Expected outputs file {expected_outputs_file_path_to_load} is empty. Disabling comparison."
                    )
                else:
                    f.seek(0)
                    loaded_json = json.load(f)[galaxy_type]
                    if isinstance(loaded_json, list) and all(isinstance(item, str) for item in loaded_json):
                        expected_outputs_data = loaded_json
                        logger.info(
                            f"Successfully loaded {len(expected_outputs_data)} expected string outputs from {expected_outputs_file_path_to_load}"
                        )
                    else:
                        logger.warning(
                            f"Expected {expected_outputs_file_path_to_load} to contain a JSON list of strings. Got {type(loaded_json)}. Disabling comparison."
                        )
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {expected_outputs_file_path_to_load}: {e}. Disabling comparison.")
        except Exception as e:
            logger.error(
                f"Unexpected error loading or parsing {expected_outputs_file_path_to_load}: {str(e)}. Disabling comparison."
            )
    else:
        logger.warning(
            f"Expected outputs file not found: {expected_outputs_file_path_to_load}. Output comparison will be skipped."
        )

    # To simulate a deployment environment, the demo supports repeating batched prompts.
    # This loop will rotate the prompts between the users for each batch, to simulate users sending different requests
    repeat_batch_prompts = []
    if is_seqlen_sweep:
        # Seqlen sweep: load each prompt file individually, filtering out lengths that exceed the model's max.
        # Extract target seqlen from filename (e.g. "input_data_long_16k.json" -> stem "input_data_long_16k" -> "16k" -> 16384).
        filtered_files = []
        for f in sweep_prompt_files:
            label = Path(f).stem.split("_")[-1]  # e.g. "16k"
            if label.endswith("k") and label[:-1].isdigit():
                min_seqlen = int(label[:-1]) * 1024
                if min_seqlen <= max_seq_len:
                    filtered_files.append(f)
        if not filtered_files:
            pytest.skip(f"No sweep prompt files fit within model's max context length ({max_seq_len})")
        repeat_batches = len(filtered_files)
        logger.info(
            f"Seqlen sweep: running {repeat_batches} steps with files: {[Path(f).name for f in filtered_files]}"
        )
        for f in filtered_files:
            batch_prompts, _ = load_inputs(f, input_lengths, instruct)
            repeat_batch_prompts.append(batch_prompts[:batch_size])
    else:
        for i in range(repeat_batches):
            repeat_batch_prompts.append(
                [all_prompts[(j + i) % len(all_prompts)] for j in range(len(all_prompts))][:batch_size]
            )

    model_args, model, page_table, tt_kv_cache = create_tt_model(
        mesh_device,
        instruct=instruct,
        max_batch_size=batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        dummy_weights=not instruct,
        page_params=page_params,
        dtype=ttnn.bfloat8_b,
        use_paged_kv_cache=paged_attention,
        prefill_profile=prefill_profile,
    )

    model_args.tokenizer = model_args.create_tokenizer()
    tokenizer = model_args.tokenizer
    generator = Generator(model, model_args, mesh_device, tokenizer=tokenizer)

    token_acc = None
    acc = None
    if token_accuracy:
        token_acc = TokenAccuracy(model_args.model_name)
        repeat_batch_prompts = [[token_acc.prepare_ref_tokens(tokenizer)]]

    num_tokens_generated_decode = []

    logger.info("Starting inference...")
    for batch_idx, input_prompts in enumerate(repeat_batch_prompts):
        logger.info(f"Processing batch {batch_idx}")
        profiler.start(f"preprocess_prefill_inputs", iteration=batch_idx)
        # Preprocess initial prompt inputs
        try:
            (
                input_tokens_prefill_pt,
                encoded_prompts,
                decoding_pos,
                prefill_lens,
            ) = preprocess_inputs_prefill(
                input_prompts,
                tokenizer,
                [model_args],
                False if token_accuracy else instruct,
                max_generated_tokens,
            )

        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")

        max_encoded_prompt_len = max(len(p) for p in encoded_prompts)

        # Load reference outputs for PCC check
        if pcc_check:
            vocab_size = 128256
            if is_ci_env:
                ref_output_path = f"/mnt/MLPerf/tt_dnn-models/llama/Llama3.3-70B-Instruct/llama3.3_70b_text_demo_ref_outputs/llama3.3_70b_ref_outputs_{num_layers}L_decode.refpt"
            else:
                ref_output_path = f"/proj_sw/user_dev/llama3.3_70b_text_demo_ref_outputs/llama3.3_70b_ref_outputs_{num_layers}L_decode.refpt"
            assert os.path.exists(ref_output_path), f"Reference output file with path {ref_output_path} does not exist!"
            torch_reference = torch.load(ref_output_path)
            ref_logits = torch_reference["all_ref_logits"]
            # Prefix-caching PCC uses batch_size=1; ref file may be batch-32 (320, 1, vocab). Use first user's steps.
            if use_prefix_caching and batch_size == 1 and ref_logits.shape == (320, 1, vocab_size):
                ref_logits = ref_logits.reshape(pcc_decode_len, 32, vocab_size)[:, 0, :].unsqueeze(1)  # (10, 1, 128256)
                torch_reference["all_ref_logits"] = ref_logits
                if len(torch_reference["reference_tokens"]) > max_encoded_prompt_len + pcc_decode_len:
                    torch_reference["reference_tokens"] = torch_reference["reference_tokens"][
                        : max_encoded_prompt_len + pcc_decode_len
                    ]
            assert torch_reference["all_ref_logits"].shape == (
                batch_size * pcc_decode_len,
                1,
                vocab_size,
            ), f"In PCC check mode, expected reference logits to have shape {(batch_size * pcc_decode_len, vocab_size)}, received {torch_reference['all_ref_logits'].shape}"
            assert (
                encoded_prompts[0] == torch_reference["reference_tokens"][:max_encoded_prompt_len].tolist()
            ), f"Provided prompt tokens do not match reference model prompt tokenss, Your prompt is encoded as: {encoded_prompts[0]}, but your reference tokens are {torch_reference['reference_tokens'][:max_encoded_prompt_len]}"
            assert max_encoded_prompt_len + pcc_decode_len == len(
                torch_reference["reference_tokens"]
            ), f"Length of prompt prefill tokens {max_encoded_prompt_len + pcc_decode_len} must match number of prompt tokens in reference tokens {len(torch_reference['reference_tokens'])}"
            torch_output = torch_reference["all_ref_logits"].reshape(pcc_decode_len, batch_size, vocab_size)[:, 0, :]
            ref_tokens = torch_reference["reference_tokens"]

        assert (
            max_generated_tokens + max_encoded_prompt_len <= max_seq_len
        ), f"Prompt prefill tokens ({max_encoded_prompt_len}) + maximum number of decoded iterations ({max_generated_tokens}) needs to be <= than max_seq_len ({max_seq_len})"
        batch_size_per_device_group = (
            32 if batch_size == 32 else 1
        )  # This is a workoaround until page table needs to know that attention is DP

        if paged_attention:
            paged_cache_max_seq_len = (
                page_params["page_block_size"] * page_params["page_max_num_blocks"] / batch_size_per_device_group
            )
            assert (
                max_generated_tokens + max_encoded_prompt_len <= paged_cache_max_seq_len
            ), f"max_generated_tokens ({max_generated_tokens}) needs to be <= than paged_cache_max_seq_len ({paged_cache_max_seq_len})"

        profiler.end(f"preprocess_prefill_inputs", iteration=batch_idx)

        # when doing repeating batches, set kv-caches to zero, to avoid context leaking
        if batch_idx != 0:
            model.switch_mode("prefill")
            for layer in model.layers:
                k_cache, v_cache = layer.attention.layer_past
                k_cache = ttnn.mul(k_cache, 0, output_tensor=k_cache)
                v_cache = ttnn.mul(v_cache, 0, output_tensor=v_cache)

        input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(batch_size, -1)
        temperature = sampling_params["temperature"]
        top_k = sampling_params.get("top_k", 32)
        top_p = sampling_params["top_p"]
        presence_penalty = sampling_params.get("presence_penalty", 0.0)
        frequency_penalty = sampling_params.get("frequency_penalty", 0.0)
        repetition_penalty = sampling_params.get("repetition_penalty", 1.0)
        seed = sampling_params.get("seed", 0)
        log_probs = sampling_params.get("log_probs", False)
        device_sampling_params = SamplingParams(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
            seed=seed,
            enable_log_probs=log_probs,
        )
        if batch_idx == 0:
            logger.info("Starting prefill warmup...")
            profiler.start(f"compile_prefill", iteration=batch_idx)
            try:
                # We run prefill warm up for all supported sequence lengths once on 1 user
                # Generator warmup runs with batch=32 internally; buffer must be at least 32.
                tt_out_logits_all_users = torch.zeros(max(32, batch_size), 1, 131072) if pcc_check else None
                toks = generator.prefill_forward_text(
                    input_tokens_prefill_pt,
                    page_table=page_table,
                    kv_cache=tt_kv_cache,
                    prompt_lens=decoding_pos,
                    enable_trace=prefill_enable_trace,
                    tt_out_logits_all_users=tt_out_logits_all_users,
                    sampling_params=device_sampling_params,
                )
            except Exception as e:
                logger.error(f"Error during prefill warmup: {str(e)}")
                raise e
            profiler.end(f"compile_prefill", iteration=batch_idx)
            logger.info("Finished prefill warmup")
        logger.info(f"Starting prefill...")

        try:
            # Generator warmup (on first prefill) uses batch=32; buffer must be at least 32.
            tt_out_logits_all_users = torch.zeros(max(32, batch_size), 1, 131072) if pcc_check else None
            if use_prefix_caching:
                # Two-phase prefill: phase 1 fills KV cache; phase 2 prefills with cached prefix (timed).
                # Phase 1: full prefill to fill KV cache (do not use output for decode).
                generator.prefill_forward_text(
                    input_tokens_prefill_pt,
                    page_table=page_table,
                    kv_cache=tt_kv_cache,
                    prompt_lens=decoding_pos,
                    enable_trace=prefill_enable_trace,
                    tt_out_logits_all_users=None,  # no PCC on phase 1
                    sampling_params=device_sampling_params,
                    start_pos=None,
                )
                # Phase 2: prefill with start_pos = num_cached_tokens (only new tokens); time this as inference_prefill.
                num_cached_tokens = int(decoding_pos[0] * prefix_cached_ratio)
                num_cached_tokens = min(num_cached_tokens, decoding_pos[0] - 1)  # at least 1 new token
                # Number of cached tokens must be a multiple of KV cache page size
                page_block_size = page_params["page_block_size"]
                num_cached_tokens = (num_cached_tokens // page_block_size) * page_block_size
                if prefill_profile:
                    signpost("start")
                profiler.start(f"inference_prefill", iteration=batch_idx)
                toks = generator.prefill_forward_text(
                    input_tokens_prefill_pt,
                    page_table=page_table,
                    kv_cache=tt_kv_cache,
                    prompt_lens=decoding_pos,
                    enable_trace=prefill_enable_trace,
                    tt_out_logits_all_users=tt_out_logits_all_users,
                    sampling_params=device_sampling_params,
                    start_pos=[num_cached_tokens],
                )
            else:
                if prefill_profile:
                    signpost("start")
                profiler.start(f"inference_prefill", iteration=batch_idx)
                toks = generator.prefill_forward_text(
                    input_tokens_prefill_pt,
                    page_table=page_table,
                    kv_cache=tt_kv_cache,
                    prompt_lens=decoding_pos,
                    enable_trace=prefill_enable_trace,
                    tt_out_logits_all_users=tt_out_logits_all_users,
                    sampling_params=device_sampling_params,
                )
            if prefill_profile:
                signpost("stop")
        except Exception as e:
            logger.error(f"Error during prefill: {str(e)}")
            raise e

        # Check the output tokens after prefill
        if pcc_check:
            torch_output_logits = torch_output[0]
            logits = tt_out_logits_all_users[0, 0, :vocab_size]
            expected_prefill_pcc = 0.90 if not apc_test else demo_targets["prefill_pcc"]
            does_pass, pcc_message = comp_pcc(logits, torch_output_logits, expected_prefill_pcc)
            logger.info(f"PCC: {pcc_message}")
            logger.info(
                f"Teacher forced token at prefill {'PASSED' if does_pass else 'FAILED'} PCC check with torch reference model"
            )
            if not apc_test:
                assert does_pass, f"Prefill PCC check failed: {pcc_message}, while expected >= {expected_prefill_pcc}."
        if apc_test:
            assert_message = (
                f"Prefill PCC check failed: {pcc_message}, while expected {demo_targets['prefill_pcc']}.\n"
                f"If it is expected to be different in Llama model, please update the text_demo_targets.json file.\n"
                f"See the comment on the text_demo.py by the assert for instructions."
            )
            assert pcc_message == demo_targets["prefill_pcc"], assert_message
            # A 'Prefill PCC mismatch' indicates that a change in the underlying prefill operation is affecting the results.
            # In some cases, small variations in PCC or improved model performance are expected. When this happens, update the target values in models/demos/llama3_70b_galaxy/demo/text_demo_targets.json.
            # Once updated, include the modified target file in your PR. The model code owners will then review and approve the changes.
            # If no changes to the model are expected from the PR, but targets differ, further investigation is needed to understand the root cause.

        # Save prefill token (unpack tuple when device sampling returns logprobs)
        if isinstance(toks, tuple):
            toks = toks[0]

        prefilled_token = toks.view(-1, 1)
        profiler.end(f"inference_prefill", iteration=batch_idx)
        logger.info(f"Prefill finished")

        if prefill_profile:  # If we are profiling prefill, we stop here
            model.tt_ccl.close()
            return True

        # Keep track of generated outputs to print out every iteration
        all_outputs = [encoded_prompts[b][: prefill_lens[b]] for b in range(batch_size)]
        for user in range(batch_size):
            user_tok = int(prefilled_token[user].item())
            all_outputs[user].append(user_tok)

        # Keeps track when a user reaches EoD token
        user_done = [False] * batch_size

        # Initial positions
        current_pos = torch.tensor([decoding_pos[b] for b in range(batch_size)])
        if batch_size == 1:
            # pad current_pos to 32 with -1s
            current_pos = torch.nn.functional.pad(current_pos, (0, 32 - current_pos.shape[0]), value=-1)
            # pad page_table to 32 with 0s
            page_table = torch.nn.functional.pad(page_table, (0, 0, 0, 32 - page_table.shape[0]), value=0)

        # Start decoding
        iteration = 0
        users_decoding = True

        # Replace the prefill token with reference token if PCC check enabled
        out_tok = prefilled_token if not pcc_check else ref_tokens[max_encoded_prompt_len]
        if token_accuracy:
            out_tok = token_acc.collect_predicted_tokens(out_tok[0].item())

        if out_tok.shape == torch.Size([]) or (len(out_tok.shape) > 0 and out_tok.shape[0] != 32):
            out_tok = out_tok.repeat(32, 1)

        try:
            model.switch_mode("decode")
        except Exception as e:
            logger.error(f"Error switching to decode mode: {str(e)}")
            model.tt_ccl.close()
        logger.info(f"Starting decode loop from positions: {decoding_pos}")

        # Log total inference (accounting for compile_decode as well)
        profiler.start(f"inference_decode", iteration=batch_idx)

        top_5_accs = []
        top_1_accs = []
        read_events = []
        tt_out_toks = []

        while users_decoding:
            if token_accuracy and iteration > 0:
                out_tok = token_acc.collect_predicted_tokens(out_tok[0].item())
                if out_tok.shape == torch.Size([]) or (len(out_tok.shape) > 0 and out_tok.shape[0] != 32):
                    out_tok = out_tok.repeat(32, 1)

            if iteration == 0:  # First iteration also accounts for compile time
                profiler.start(f"compile_decode", iteration=batch_idx)
            else:
                profiler.start(f"inference_decode_time_{iteration}", iteration=batch_idx)

            # Determine whether to enable trace
            if apc_test:
                is_enable_trace = iteration != 0  # First iteration is compile time and checks PCC
            else:
                is_enable_trace = enable_trace if not pcc_check else False

            # Run decode forward
            try:
                # Save logits only for PCC check when tracing is disabled
                tt_out_logits_saved = torch.zeros(vocab_size) if (pcc_check and not is_enable_trace) else None
                decode_async_read = not token_accuracy
                decode_output = generator.decode_forward(
                    out_tok,
                    current_pos,
                    enable_trace=is_enable_trace,
                    page_table=page_table,
                    kv_cache=tt_kv_cache,
                    read_from_device=True,
                    async_read=decode_async_read,
                    sampling_params=device_sampling_params,
                    reset_inputs=iteration == 0,
                    tt_out_logits_saved=tt_out_logits_saved,
                    is_cur_pos_sharded=is_cur_pos_sharded,
                    is_page_table_sharded=is_page_table_sharded,
                    prompt_tokens=input_tokens_prefill_pt,
                    output_tokens=prefilled_token,
                )
                if decode_async_read:
                    tt_out_tok, read_event = decode_output
                    read_events.append(read_event)
                    tt_out_toks.append(tt_out_tok)
                else:
                    tt_out_tok, tt_log_probs = decode_output
                if apc_test and iteration == 0:
                    tt_out_logits_saved_iter_0 = tt_out_logits_saved
            except Exception as e:
                pytest.fail(f"Decode forward failed at iteration {iteration}: {str(e)}")

            if iteration == 0:
                profiler.end(f"compile_decode", iteration=batch_idx)
                decode_iteration_time = profiler.get_duration("compile_decode", iteration=batch_idx)
                logger.info(f"Iteration {iteration} (compile): {1000*decode_iteration_time:.4f}ms")

            teacher_forcing = (
                not apc_test
                and pcc_check
                and max_encoded_prompt_len + iteration + 1 < len(ref_tokens)
                and num_layers == 80
            )

            if iteration > 0 or token_accuracy:
                if not token_accuracy:
                    ttnn.event_synchronize(read_events.pop(0)[0])
                    tt_out_tok, tt_log_probs = generator.process_decode_output_host(tt_out_toks.pop(0))

                if teacher_forcing:
                    out_tok = ref_tokens[max_encoded_prompt_len + iteration + 1]
                elif pcc_check and tt_out_tok.shape[-1] >= vocab_size:
                    out_tok = tt_out_tok.float().argmax(dim=-1)
                else:
                    out_tok = tt_out_tok.reshape(-1).to(torch.long)

                if out_tok.shape == torch.Size([]) or (len(out_tok.shape) > 0 and out_tok.shape[0] != 32):
                    out_tok = out_tok.repeat(32, 1)

                if teacher_forcing or (apc_test and iteration == 1):
                    if apc_test:
                        tt_out_logits_saved = tt_out_logits_saved_iter_0
                        torch_output_logits = torch_output[1]
                    else:
                        torch_output_logits = torch_output[iteration + 1]
                    expected_decode_pcc = 0.91 if not apc_test else demo_targets["decode_pcc"]
                    does_pass, pcc_message = comp_pcc(tt_out_logits_saved, torch_output_logits, expected_decode_pcc)
                    logger.info(f"PCC: {pcc_message}")
                    logger.info(
                        f"Teacher forced token at decode iteration {iteration} "
                        f"{'PASSED' if does_pass else 'FAILED'} PCC check with torch reference model"
                    )
                    if not apc_test:
                        assert does_pass, (
                            f"Decode PCC check failed at iteration {iteration}: {pcc_message}, "
                            f"while expected >= {expected_decode_pcc}."
                        )

                if apc_test:
                    assert_message = (
                        f"Decode PCC check failed: {pcc_message}, while expected {demo_targets['decode_pcc']}.\n"
                        f"If any ops in Llama model might be impacted, please update decode_pcc in the text_demo_targets.json file.\n"
                        f"See the comment on the text_demo.py by the assert for instructions."
                    )
                    assert pcc_message == demo_targets["decode_pcc"], assert_message

                if teacher_forcing:
                    _, tt_top5_tokens = torch.topk(tt_out_logits_saved, k=5, dim=-1)
                    _, ref_top5_tokens = torch.topk(torch_output_logits, k=5, dim=-1)
                    top_1_acc = tt_top5_tokens[0] == ref_top5_tokens[0]
                    top_5_acc = torch.any(tt_top5_tokens == ref_top5_tokens)
                    top_1_accs.append(top_1_acc)
                    top_5_accs.append(top_5_acc)
                    logger.info(f"Top-1 Accuracy: {top_1_acc}")
                    logger.info(
                        f"Top-5 Correctness:{torch.any(tt_top5_tokens == ref_top5_tokens).item(),} Accuracy: {top_5_acc}"
                    )

                # Save output token to print out later
                if not pcc_check:
                    for user in range(batch_size):
                        user_tok = out_tok.tolist()[user]
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

                # Print out generated outputs for each user at the end of every iteration
                if print_outputs and not is_ci_env and not pcc_check:
                    for user in range(batch_size):
                        text = "".join(tokenizer.decode(all_outputs[user]))
                        if len(text) > 100:
                            text = "..." + text[-97:]
                        text = text.replace("\n", " ")
                        logger.info("[User {}] {}".format(user, text))

                # The e2e decode inference accounts for device execution + host post-processing time
                if iteration > 0:
                    profiler.end(f"inference_decode_time_{iteration}", iteration=batch_idx)
                    decode_iteration_time = profiler.get_duration(
                        f"inference_decode_time_{iteration}", iteration=batch_idx
                    )
                    # Always print perf after every iteration
                    tokens_per_second_per_user = 1 / decode_iteration_time

                    logger.info(
                        f"Decode Iteration {iteration}: Time: {1000*decode_iteration_time:.4f}ms, tok/s/user: {tokens_per_second_per_user:.2f}, Throughput: {batch_size*tokens_per_second_per_user:.2f} tok/s"
                    )
                    if apc_test and (demo_targets["token_pos"] - len(input_tokens_prefill_pt)) == iteration:
                        # Check if the throughput is within the expected range
                        lower_bound = demo_targets["throughput"] - demo_targets["absolute_margin"]
                        upper_bound = demo_targets["throughput"] + demo_targets["absolute_margin"]
                        # TODO: Enable once experimentaly established avg and absolute margin
                        assert_message = (
                            f"Current throughput: {tokens_per_second_per_user:.1f} tok/s/user for APC test is not within the expected range: ({lower_bound}, {upper_bound}).\n"
                            f"Update text_demo_targets.json file with the expected throughput.\n"
                            f"See the comment on the text_demo.py by the assert for instructions."
                        )
                        assert lower_bound <= tokens_per_second_per_user <= upper_bound, assert_message
                        # A mismatch of throughput suggests a regression in performance, likely due to changes in one or more ops used by the model, resulting in reduced end-to-end throughput.
                        # In some cases, small variations in PCC or improved model performance are expected. When this happens, update the target values in models/demos/llama3_70b_galaxy/demo/text_demo_targets.json.
                        # Once updated, include the modified target file in your PR. The model code owners will then review and approve the changes.
                        # If no changes to the model are expected from the PR, but targets differ, further investigation is needed to understand the root cause.

            current_pos += 1
            iteration += 1

            # Upper limit of generated tokens for each user; if users_decoding is already False (say by hitting eos), then we don't need to check the max_generated_tokens.
            if users_decoding:
                users_decoding = iteration < max_generated_tokens

            # Final print
            if not users_decoding and not pcc_check:
                profiler.start(f"log_saving_file", iteration=batch_idx)
                logger.info("Finished decoding, printing the final outputs...\n")
                for i, (output, prompt) in enumerate(zip(all_outputs, input_prompts)):
                    text = tokenizer.decode(output)
                    prompt_including_assistant_tags = tokenizer.decode(
                        model_args.encode_prompt(prompt, instruct=instruct)
                    )
                    text_after_prompt = text.replace(prompt_including_assistant_tags, "", 1)
                    if print_to_file:
                        with open(output_filename, "a") as f:
                            f.write(
                                f"\nbatch: {batch_idx} user: {i}\nprompt: {prompt} \noutput:\n{text_after_prompt}\n"
                            )
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
                # TODO This check is only for the config `repeat2`.

            # Since right now that config is the only using a repeat_batches=2 this if statement works
            if not users_decoding and batch_size == 1 and repeat_batches == 2:
                # Compare to text in outputs_batch_1.json for the first user of the first batch
                if batch_idx == 0 and expected_outputs_data:  # Only compare if data was loaded
                    if i == 0:  # Only for the first user of the batch (i.e., user 0)
                        if len(expected_outputs_data) > 0:
                            expected_text = expected_outputs_data[0]  # Compare with the first entry in the JSON list
                            actual_text_clean = text_after_prompt.strip()
                            expected_text_clean = expected_text.strip()

                            if actual_text_clean != expected_text_clean:
                                logger.warning(
                                    f"Output for user {i} in batch {batch_idx} DOES NOT MATCH expected output from {expected_outputs_file_path_to_load}."
                                )
                                logger.info(f"Expected: {repr(expected_text_clean)}")
                                logger.info(f"Actual  : {repr(actual_text_clean)}")
                                mismatches_found = 0
                                # Iterate based on the longer of the two strings to catch all differences
                                for char_idx in range(min(len(actual_text_clean), len(expected_text_clean))):
                                    actual_char = (
                                        actual_text_clean[char_idx]
                                        if char_idx < len(actual_text_clean)
                                        else "<END_OF_ACTUAL>"
                                    )
                                    expected_char = (
                                        expected_text_clean[char_idx]
                                        if char_idx < len(expected_text_clean)
                                        else "<END_OF_EXPECTED>"
                                    )
                                    if actual_char != expected_char:
                                        logger.info(
                                            f"Mismatch at position {char_idx}: Actual: '{repr(actual_char)}', Expected: '{repr(expected_char)}'"
                                        )
                                        mismatches_found += 1
                                    if mismatches_found >= 20:  # Limit number of logged mismatches
                                        logger.info(
                                            "More mismatches exist but will not be logged for this comparison (limit reached)."
                                        )
                                        assert (
                                            False
                                        ), "More mismatches exist but will not be logged for this comparison (limit reached)."
                                        break
                        else:
                            logger.info(
                                f"Output for user {i} in batch {batch_idx} matches expected output from {expected_outputs_file_path_to_load}."
                            )
                    else:  # expected_outputs_data is not empty list, but i==0 and len(expected_outputs_data) == 0 (should be caught by outer if)
                        logger.warning(
                            f"Expected outputs data was loaded from {expected_outputs_file_path_to_load} but is an empty list. Cannot compare for user {i}, batch {batch_idx}."
                        )
                elif batch_idx == 0 and not expected_outputs_data and i == 0:  # Only log once per batch if no data
                    logger.warning("Expected outputs data is empty or not loaded, cannot compare.")

        num_tokens_generated_decode.append(iteration)  # Save the number of tokens generated for each repeat batch
        if token_accuracy:
            acc = token_acc.compute_accuracy()
            logger.info("=== Top1 and Top5 Token Accuracy ===")
            logger.info(f" Top1 Accuracy: {acc[0] * 100:.2f}%, Top5 Accuracy: {acc[1] * 100:.2f}%")

    profiler.end(f"inference_decode", iteration=batch_idx)

    # Finish profiling at the end of inference for all repeated batches
    profiler.end("run")

    # Prepare profile benchmark metrics
    # When repeat_batches > 1: use prefill time from batch 1 (after warmup). Otherwise use batch 0.
    compile_prefill_time = profiler.get_duration("compile_prefill")
    compile_decode_time = profiler.get_duration("compile_decode")

    total_inference_prefill_time = profiler.get_duration("inference_prefill")
    total_inference_decode_time = 0
    for i in range(1, iteration):  # Iteration 0 is the compile time
        total_inference_decode_time += profiler.get_duration(f"inference_decode_time_{i}")

    # Average prefill time for each user
    avg_time_to_first_token = total_inference_prefill_time
    # Average decode time per batch iteration
    avg_decode_iteration_time = total_inference_decode_time / (iteration - 1)

    prefill_tok_s = prefill_lens[0] / total_inference_prefill_time * batch_size
    decode_tok_s_user = (num_tokens_generated_decode[0] - 1) / total_inference_decode_time  # Remove the compile time
    decode_tok_s = (
        (num_tokens_generated_decode[0] - 1) / total_inference_decode_time * batch_size
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

    if not apc_test and num_layers == 80 and pcc_check and len(top_1_accs) > 0 and len(top_5_accs) > 0:
        measurements["Top 1 Accuracy"] = sum(top_1_accs) / len(top_1_accs)
        measurements["Top 5 Accuracy"] = sum(top_5_accs) / len(top_5_accs)

    if token_accuracy:
        assert acc is not None, "Token accuracy mode completed without accuracy results"
        total_top1_acc = math.ceil(acc[0] * 100)
        total_top5_acc = math.ceil(acc[1] * 100)
        measurements["Top 1 Accuracy"] = total_top1_acc
        measurements["Top 5 Accuracy"] = total_top5_acc

        min_top1_acc, min_top5_acc = get_accuracy_thresholds(model_args)
        assert (
            total_top1_acc >= min_top1_acc
        ), f"Top-1 accuracy {total_top1_acc:.1f}% is too low (expected >={min_top1_acc}%)"
        assert (
            total_top5_acc >= min_top5_acc
        ), f"Top-5 accuracy {total_top5_acc:.1f}% is too low (expected >={min_top5_acc}%)"
        logger.info("Checks of top-1 and top-5 accuracy against PERF.md passed")

    # Decode performance for some specific tokens
    tok_1_perf = profiler.get_duration(f"inference_decode_time_{1}")  # Iteration 0 is compile time
    tok_128_perf = profiler.get_duration(f"inference_decode_time_{127}") if 127 < iteration else 0
    tok_1024_perf = profiler.get_duration(f"inference_decode_time_{1023}") if 1023 < iteration else 0
    tok_4096_perf = profiler.get_duration(f"inference_decode_time_{4095}") if 4095 < iteration else 0

    if not stop_at_eos:
        logger.info(f"Please note that 'stop_at_eos' is disabled. Output repetition is expected.")

    logger.info("")
    logger.info(f"=== Performance metrics ===")
    logger.info(
        f"1st token decode time: {tok_1_perf*1000:.2f}ms [{round(1/tok_1_perf, 2)} t/s/u, {round((1/tok_1_perf)*batch_size, 2)} t/s]"
    )
    if tok_128_perf > 0:
        logger.info(
            f"128th token decode time: {tok_128_perf*1000:.2f}ms [{round(1/tok_128_perf, 2)} t/s/u, {round((1/tok_128_perf)*batch_size, 2)} t/s]"
        )
    if tok_1024_perf > 0:
        logger.info(
            f"1024th token decode time: {tok_1024_perf*1000:.2f}ms [{round(1/tok_1024_perf, 2)} t/s/u, {round((1/tok_1024_perf)*batch_size, 2)} t/s]"
        )
    if tok_4096_perf > 0:
        logger.info(
            f"4096th token decode time: {tok_4096_perf*1000:.2f}ms [{round(1/tok_4096_perf, 2)} t/s/u, {round((1/tok_4096_perf)*batch_size, 2)} t/s]"
        )

    # Print some of the perf metrics
    logger.info("==")
    logger.info(f"Prefill compile time: {round(compile_prefill_time, 2)}s")
    logger.info(f"Decode compile time: {round(compile_decode_time, 2)}s")
    logger.info("")
    logger.info(f"Average Time to First Token (TTFT): {round(avg_time_to_first_token*1000, 2)}ms")
    logger.info(
        f"Average speed: {round(avg_decode_iteration_time * 1000, 2)}ms @ {round(decode_tok_s_user, 2)} tok/s/user ({round(decode_tok_s, 2)} tok/s throughput)"
    )

    test_id = request.node.callspec.id
    if "repeat2" in test_id:  #  test_id will be changed to eval-1 and eval-32 in the future
        sku = get_current_device_sku_name()
        resolved_targets = resolve_perf_targets(
            model_name=model_args.base_model_name,
            sku=sku,
            batch_size=batch_size,
            seq_len=len(input_prompts[0]),
        )
        if resolved_targets:
            verify_perf(
                measurements,
                expected_measurements={
                    "prefill_time_to_token": True,
                    "decode_t/s/u": True,
                },
                model_name=model_args.base_model_name,
                sku=sku,
                batch_size=batch_size,
                seq_len=len(input_prompts[0]),
            )
        else:
            logger.warning(
                f"No centralized performance targets found for model={model_args.base_model_name}, "
                f"sku={sku}, batch_size={batch_size}, seq_len={len(input_prompts[0])}"
            )
    else:
        logger.info(f"Test '{test_id}' currently doesn't have performance targets set! Skipping performance checks...")

    # Save benchmark data for CI dashboard
    if is_ci_env and repeat_batches > 1:
        benchmark_data.add_measurement(
            profiler,
            1,  # grab the second repeat batch of prefill
            "inference_prefill",
            "time_to_token",
            avg_time_to_first_token,
        )
        benchmark_data.add_measurement(
            profiler,
            repeat_batches - 1,  # use the last batch iteration (the one with an end timestamp)
            "inference_decode",
            "tokens/s",
            decode_tok_s,
        )
        benchmark_data.add_measurement(
            profiler,
            repeat_batches - 1,
            "inference_decode",
            "tokens/s/user",
            decode_tok_s_user,
        )

        benchmark_data.save_partial_run_json(
            profiler,
            run_type="demo_perf",
            ml_model_name=model_args.base_model_name,
            ml_model_type="llm",
            batch_size=batch_size,
            input_sequence_length=len(input_prompts[0]),
            output_sequence_length=num_tokens_generated_decode[0],
        )
