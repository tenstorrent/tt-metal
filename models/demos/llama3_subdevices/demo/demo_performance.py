# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from time import perf_counter
from datetime import datetime
from loguru import logger
import os
import ttnn
import pytest

from models.demos.llama3_subdevices.tt.llama_common import (
    PagedAttentionConfig,
)
from models.demos.llama3_subdevices.tt.llama_model import TtTransformer
from models.demos.llama3_subdevices.tt.llama_embedding import TtLlamaEmbedding
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.tokenizer import Tokenizer
from models.demos.llama3_subdevices.tt.model_config import TtModelArgs

from models.perf.benchmarking_utils import BenchmarkProfiler
from models.demos.llama3_subdevices.tt.model_config import LlamaOptimizations

from .demo_decode import load_inputs
from tracy import signpost


def run_llama3_decode_performance(
    user_input,
    mesh_device,
    max_seq_len,
    batch_size,
    num_batches,
    paged_attention,
    paged_attention_config,
    benchmark_token_range,
    warmup_iters,
    inner_iters,
    optimizations,
    instruct_mode,
    is_ci_env,
    print_to_file,
    weights,
    layers,
):
    bench_start, bench_end = benchmark_token_range

    # Creat batch output file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_directory = "models/demos/llama3/demo/output"
    os.makedirs(output_directory, exist_ok=True)
    os.chmod(output_directory, 0o755)
    output_filename = f"{output_directory}/demo_user_output_{timestamp}.txt"

    dtype = ttnn.bfloat8_b
    assert batch_size <= 32, "Max batch size currently supported is 32"
    assert max_seq_len <= 128 * 1024, "Max sequence length must be less than 128k tokens"

    dummy_weights = weights == "random"

    # We disregard any warmup iteration for profiling, in favour of just measuring compile time on the first iteration
    N_warmup_iter = {"inference_prefill": 0, "inference_decode": 0}

    # Start profiler
    logger.info(f"Start profiler")
    profiler = BenchmarkProfiler()
    profiler.start("run")

    logger.info(f"Reading inputs...")
    profiler.start("loading_inputs")
    if len(user_input) == 1:
        input_prompts = user_input * batch_size
    else:
        input_prompts = load_inputs(user_input, batch_size, instruct_mode)
    profiler.end("loading_inputs")

    # Generate the batched prompts (rotate the inputs between the users, for each batch)
    # If batch_size == 1, the same prompt is repeated for each batch
    batch_prompts = []
    for i in range(num_batches):
        batch_prompts.append([input_prompts[(j + i) % len(input_prompts)] for j in range(len(input_prompts))])

    # Load model args, weights, and tokenizer
    model_args = TtModelArgs(
        mesh_device,
        instruct=instruct_mode,
        max_batch_size=batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        dummy_weights=dummy_weights,
    )
    model_args.n_layers = layers

    tokenizer = Tokenizer(model_args.tokenizer_path)

    # Check max sequence length compatibility with model and architecture. Refer to README for more information
    llama_model_name = model_args.model_name  # ["3.2-1B", "3.2-3B", "3.1-8B", "3.2-11B", "3.1-70B"]
    tt_device_name = model_args.device_name  # ["N150", "N300", "T3K", "TG"]

    if llama_model_name == "3.1-70B":
        assert tt_device_name in ["TG"], "Llama-3.1-70B is only supported on TG"
        assert max_seq_len <= 128 * 1024, "TG supports the official max context length of 128k tokens for Llama-3.1-70B"

    logger.info("Loading weights...")
    profiler.start("weight_loading")
    state_dict = model_args.load_state_dict()
    profiler.end("weight_loading")

    page_table_tt = None

    if paged_attention:
        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
        )

    # Load TTNN Llama-3.1 model
    logger.info("Loading weights to device...")
    profiler.start("loading_weights_to_device")
    tt_model = TtTransformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )
    tt_embd = TtLlamaEmbedding(
        mesh_device=mesh_device,
        args=model_args,
        weight_cache_path=model_args.weight_cache_path(dtype),
        state_dict=state_dict,
        dtype=ttnn.bfloat16,  # Row major layout requires bfloat16
    )

    profiler.end("loading_weights_to_device")
    logger.info("Finished loading weights to device.")

    # Keep track of generated outputs to print out every iteration
    if dummy_weights:
        encoded_prompts = [
            [128000, 2028, 374, 264, 1296]
        ] * model_args.max_batch_size  # "This is a test" encoded prompt
    else:
        encoded_prompts = [tokenizer.encode(prompt, bos=True, eos=False) for prompt in input_prompts]

    # Prefill by decode: start at first token; pad to 32 (tile size)
    max_prompt_length = max([len(prompt) for prompt in encoded_prompts])
    padded_token_prompts = [prompt + [128009] * (max_prompt_length - len(prompt)) for prompt in encoded_prompts]
    encoded_prompts_tensor_whole_sequence = torch.tensor([padded_token_prompts[b] for b in range(batch_size)])

    user_done = [False] * batch_size  # Keeps track when a user reaches EoD token

    logger.info("Starting decode...")

    # Shard the page table for TG decode
    if paged_attention and model_args.is_galaxy and batch_size > 1:
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, -2) if batch_size > 1 else (None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

        logger.info("Page table tensor done")

    # Initial positions
    decoding_pos = [bench_start] * batch_size
    current_pos = torch.tensor(decoding_pos)

    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )

    logger.info("Current pos tensor done")

    # Get cos/sin matrices for the current position of each user
    rot_mats, rot_mat_idxs = tt_model.rope_setup.get_rm_rot_mats(current_pos, return_rot_idxs=True)

    logger.info("Rot mats done")

    # Prepare the encoded prompts for the decode input
    tt_out_tok = ttnn.from_torch(
        encoded_prompts_tensor_whole_sequence[:, :1].reshape(1, 1, 1, batch_size),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sub_core_grids = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )

    # Compile
    logger.info(f"Compiling model trace...")
    if layers == 1:
        num_compile_iters = 10
    else:
        num_compile_iters = 1
    for i in range(num_compile_iters):
        tt_decode_input = tt_embd(tt_out_tok)
        logger.info(f"tt_decode_input done")

        tt_out = tt_model(
            tt_decode_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )
        logger.info(f"tt_out done")

        tt_out_gathered = tt_model.tt_ccl.line_all_gather(
            tt_out[0], dim=3, num_links=2, cluster_axis=0, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tt_out_rm = ttnn.untilize(tt_out_gathered, use_multicore=True, sub_core_grids=sub_core_grids)
        ttnn.deallocate(tt_out_gathered)
        tt_out_tok = ttnn.argmax(
            tt_out_rm, dim=3, keepdim=True, use_multicore=True, output_tensor=tt_out_tok, sub_core_grids=sub_core_grids
        )
        logger.info(f"sampling done")

    ttnn.plus_one(
        current_pos_tensor,
        sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
    )

    # Capture Trace
    logger.info(f"Capturing model trace...")
    profiler.start(f"capture_trace")

    tt_model.tt_ccl.reset_gather_and_buffer_idx()

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)

    # Get cos/sin matrices for the current position of each user
    rot_mats = tt_model.rope_setup.get_rm_rot_mats(rot_mat_idxs)
    tt_decode_input = tt_embd(tt_out_tok)
    tt_out = tt_model(
        tt_decode_input,
        current_pos_tensor,
        rot_mats=rot_mats,
        mode="decode",
        page_table=page_table_tt,
    )
    tt_out_gathered = tt_model.tt_ccl.line_all_gather(
        tt_out[0], dim=3, num_links=2, cluster_axis=0, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out_rm = ttnn.untilize(tt_out_gathered, use_multicore=True, sub_core_grids=sub_core_grids)
    ttnn.deallocate(tt_out_gathered)
    tt_out_tok = ttnn.argmax(
        tt_out_rm, dim=3, keepdim=True, use_multicore=True, output_tensor=tt_out_tok, sub_core_grids=sub_core_grids
    )

    ttnn.plus_one(
        current_pos_tensor,
        sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
    )

    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Reset the decoding position for the proper run of the model
    current_pos_reset = ttnn.from_torch(
        current_pos,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )

    tt_out_tok_reset = ttnn.from_torch(
        encoded_prompts_tensor_whole_sequence[:, :1].reshape(1, 1, 1, batch_size),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
    )

    # Reset the current position and output token tensors for the real decode run
    ttnn.copy_host_to_device_tensor(current_pos_reset, current_pos_tensor)
    ttnn.copy_host_to_device_tensor(tt_out_tok_reset, tt_out_tok)
    rot_mat_idxs_reset = tt_model.rope_setup.get_rm_rot_idxs(current_pos, on_host=True)
    ttnn.copy_host_to_device_tensor(rot_mat_idxs_reset, rot_mat_idxs)

    profiler.end(f"capture_trace")

    ttnn.synchronize_device(mesh_device)

    # When getting dispatch device perf, pushing weights fills up profiler buffers.
    ttnn.DumpDeviceProfiler(device)

    # Sync after dump or execute trace will launch on devices with huge skew
    ttnn.synchronize_device(mesh_device)

    # Start decoding
    bench_decode_time = 0
    all_outputs = []

    logger.info(f"Starting decode loop...")

    for iteration in range(bench_start - warmup_iters, bench_end):
        if iteration == bench_start:
            signpost("tracy_perf_run")
        iteration_time_start = perf_counter()

        # Execute trace
        for _ in range(inner_iters):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        # Block on output
        tt_out_tok.cpu(blocking=True, cq_id=0)

        # Print out generated outputs for each user at the end of every iteration
        iteration_time = perf_counter() - iteration_time_start

        if iteration >= bench_start:
            bench_decode_time += iteration_time

        tokens_per_second_per_user = inner_iters / iteration_time

        # Always print perf at every iteration
        logger.info(
            f"Iteration {iteration}: {1000*iteration_time:.3f}ms @ {tokens_per_second_per_user:.3f} tok/s/user ({batch_size*tokens_per_second_per_user:.3f} tok/s throughput)"
        )

    # Release trace
    ttnn.release_trace(mesh_device, trace_id)

    # Finish profiling at the end of all batches inference
    profiler.end("run")

    bench_tokens_generated = bench_end - bench_start
    bench_decode_latency = bench_decode_time / (bench_tokens_generated * inner_iters)
    bench_toksu = 1 / bench_decode_latency

    logger.info(
        f"Statistics over token range [{bench_start}, {bench_end}) with {inner_iters} inner iterations between blocking"
    )
    logger.info(
        f"tokens_generated: {bench_tokens_generated}, decode_latency: {bench_decode_latency * 1000:.3f}ms @ {bench_toksu:.3f} tok/s/u"
    )

    ttnn.synchronize_device(mesh_device)


@pytest.mark.parametrize(
    "input_prompts, instruct, repeat_batches, max_seq_len, batch_size, benchmark_token_range, warmup_iters, inner_iters, paged_attention, page_params",
    [
        (
            "models/demos/llama3_subdevices/demo/sample_prompts/input_data_prefill_128.json",
            True,
            1,
            1024,
            32,
            (118, 128),
            10,
            10,
            True,
            None,
        ),
        (
            "models/demos/llama3_subdevices/demo/sample_prompts/input_data_prefill_128.json",
            True,
            1,
            1024,
            32,
            (127, 128),
            0,
            1,
            True,
            None,
        ),
    ],
    ids=[
        "batch-32-e2e",  # throughput
        "batch-32-device-perf",  # throughput
    ],
)
@pytest.mark.parametrize(
    "weights, layers",
    [
        ("instruct", 1),
        ("instruct", 5),
        ("instruct", 10),
        ("instruct", 15),
        ("instruct", 40),
        ("instruct", 80),
    ],
    ids=["1L", "5L", "10L", "15L", "40L", "80L"],
)
@pytest.mark.parametrize(
    "optimizations",
    [
        LlamaOptimizations.performance,
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "trace_region_size": 23887872,
            "fabric_config": True,
        }
    ],
    indirect=True,
)
def test_llama_decode_performance(
    input_prompts,
    instruct,
    repeat_batches,
    max_seq_len,
    batch_size,
    benchmark_token_range,
    warmup_iters,
    inner_iters,
    paged_attention,
    page_params,
    optimizations,
    weights,
    layers,
    mesh_device,
    is_ci_env,
    reset_seeds,
):
    if is_ci_env and ("long" in input_prompts or optimizations == LlamaOptimizations.accuracy):
        pytest.skip("Do not run the 'long-context' or accuracy tests on CI to reduce load")

    # TODO: Remove this once all batch sizes are supported on TG
    if os.environ.get("FAKE_DEVICE") == "TG" and batch_size not in [1, 32]:
        pytest.skip("TG only supports batch 1 and 32")

    if paged_attention:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
    else:
        paged_attention_config = None

    return run_llama3_decode_performance(
        user_input=input_prompts,
        mesh_device=mesh_device,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        num_batches=repeat_batches,
        paged_attention=paged_attention,
        paged_attention_config=paged_attention_config,
        benchmark_token_range=benchmark_token_range,
        warmup_iters=warmup_iters,
        inner_iters=inner_iters,
        optimizations=optimizations,
        instruct_mode=instruct,
        is_ci_env=is_ci_env,
        print_to_file=False,
        weights=weights,
        layers=layers,
    )
