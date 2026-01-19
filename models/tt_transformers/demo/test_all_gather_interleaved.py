# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import pytest
import requests
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_wormhole_b0
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.common import PagedAttentionConfig, create_tt_model
from models.tt_transformers.tt.generator import Generator, create_submeshes
from models.tt_transformers.tt.model_config import DecodersPrecision


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
    # It clips the excerpt to the max length provided to allow testing different long context lengthts
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


def fingerprint_device_tensor(tt_tensor: ttnn.Tensor, label: str, mesh_device):
    addr = tt_tensor.buffer_address()

    # Compose full tensor to host for debugging; slice if this is too big
    host = ttnn.to_torch(
        tt_tensor,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3),
    )

    checksum = float(host.sum())
    vmin = float(host.min())
    vmax = float(host.max())

    logger.info(f"{label}: addr={addr}, shape={list(host.shape)}, " f"checksum={checksum}, min={vmin}, max={vmax}")
    return addr, checksum, vmin, vmax, host


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
):
    submesh_devices = create_submeshes(mesh_device, data_parallel)
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

    for submesh in submesh_devices:
        model_args_i, model_i, tt_kv_cache_i, state_dict = create_tt_model(
            submesh,
            instruct=instruct,
            max_batch_size=global_batch_size // data_parallel,
            optimizations=optimizations,
            max_seq_len=max_seq_len,
            paged_attention_config=paged_attention_config,
            dtype=ttnn.bfloat8_b,
            state_dict=state_dict,
            num_layers=num_layers,
        )
        model_args.append(model_args_i)
        model.append(model_i)
        tt_kv_cache.append(tt_kv_cache_i)

    page_table = create_tt_page_table(
        global_batch_size=global_batch_size,
        data_parallel=data_parallel,
        paged_attention_config=paged_attention_config,
    )
    # Host code, safe to reuse tokenizer from the 1st model
    tokenizer = model_args[
        0
    ].tokenizer  # TODO Should we support Data Parallel different models? If so, we need to support multiple tokenizers
    processor = model_args[0].processor
    return model_args, model, page_table, tt_kv_cache, tokenizer, processor


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "trace_region_size": 50000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "T3K": (1, 8),
            "TG": (8, 4),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
            "BHGLX": (8, 4),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
def test_demo_text(
    mesh_device,
    is_ci_env,
    is_ci_v2_env,
    reset_seeds,
    request,
    model_location_generator,
):
    """
    Simple demo with limited dependence on reference code.
    Hardcoded for batch-1 test case.
    """
    # Hardcoded batch-1 test parameters
    input_prompts = "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json"
    instruct = True
    repeat_batches = 2
    max_seq_len = 1024
    batch_size = 1
    max_generated_tokens = 50
    paged_attention = True
    page_params = {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024}
    stop_at_eos = True
    data_parallel = 1
    token_accuracy = False
    num_layers = None

    # Hardcode optimizations to performance mode
    def optimizations_fn(model_args):
        return DecodersPrecision.performance(model_args.n_layers, model_args.model_name)

    optimizations = optimizations_fn

    print_to_file = False  # Enable this flag to print the output of all users to a file

    num_devices = mesh_device.get_num_devices() if isinstance(mesh_device, ttnn.MeshDevice) else 1
    global_batch_size = batch_size * data_parallel  # input batch_size is interpreted as size per DP group

    hf_dir = os.getenv("HF_MODEL", "")
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

    model_args, model, page_table, tt_kv_cache, tokenizer, processor = prepare_generator_args(
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
    )

    repeat_batch_prompts = []
    for i in range(repeat_batches):
        # For token accuracy, use input_prompts without rotation
        if token_accuracy:
            repeat_batch_prompts.append(input_prompts)
        else:
            repeat_batch_prompts.append(
                [all_prompts[(j + i) % len(all_prompts)] for j in range(len(all_prompts))][:global_batch_size]
            )

    input_tensors = []
    for i in range(8):
        input_tensors.append(torch.load(f"topk/tensor_{i}.pt"))

    memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )

    host_shards = [
        ttnn.from_torch(
            tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
        )
        for tensor in input_tensors
    ]

    tensor = ttnn.from_host_shards(
        host_shards,
        mesh_shape=mesh_device.shape,
    ).to(mesh_device)

    ret = []

    x = ttnn.from_torch(
        torch.ones(1, 1, 1024, 256),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.BufferType.DRAM,
        ),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        device=mesh_device,
    )

    generator = Generator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)

    # compile run
    a = ttnn.slice(x, (0, 0, 0, 0), (1, 1, 0 + 32, x.shape[-1]))
    b = generator.model[0].norm(a, mode="prefill")
    c = ttnn.interleaved_to_sharded(b, generator.model[0].model_config["LM_HEAD_INPUT_MEMCFG"])
    d = generator.model[0].lm_head(c)
    e = ttnn.to_layout(d, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logger.info("Done compiling run")

    ttnn.synchronize_device(model_args[0].mesh_device)
    trace_id = ttnn.begin_trace_capture(model_args[0].mesh_device, cq_id=0)
    a = ttnn.slice(x, (0, 0, 0, 0), (1, 1, 0 + 32, x.shape[-1]))
    b = generator.model[0].norm(a, mode="prefill")
    c = ttnn.interleaved_to_sharded(b, generator.model[0].model_config["LM_HEAD_INPUT_MEMCFG"])
    d = generator.model[0].lm_head(c)
    e = ttnn.to_layout(d, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.end_trace_capture(model_args[0].mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(model_args[0].mesh_device)
    logger.info("Done capturing trace")

    for _ in range(2):
        addr0, c0, mn0, mx0, _ = fingerprint_device_tensor(tensor, "AG_input_before_trace0", mesh_device)

        ttnn.execute_trace(model_args[0].mesh_device, trace_id, cq_id=0, blocking=False)
        logger.info("Done executing trace " + str(_))

        ttnn.synchronize_device(model_args[0].mesh_device)

        ret.append(
            ttnn.all_gather(
                tensor,
                dim=3,
                num_links=1,
                memory_config=tensor.memory_config(),
                cluster_axis=None,
                topology=ttnn.Topology.Linear,
            ).cpu(blocking=True)
        )
        logger.info("Done all gather " + str(_))

        ttnn.synchronize_device(model_args[0].mesh_device)

    ret[0] = ttnn.to_device(ret[0], mesh_device)
    ret[1] = ttnn.to_device(ret[1], mesh_device)

    ttnn.synchronize_device(mesh_device)

    gathered_tensor_a_torch = ttnn.to_torch(
        ret[0],
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3),
    )

    gathered_tensor_b_torch = ttnn.to_torch(
        ret[1],
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3),
    )

    assert torch.allclose(gathered_tensor_a_torch, gathered_tensor_b_torch), "All-gather results do not match"
