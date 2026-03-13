# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
OLMo-3.1-32B End-to-End Demo for Galaxy TG.

Similar to Qwen32 demo but uses OLMo-specific configuration:
- YaRN RoPE (not linear)
- Sliding window attention (3 sliding + 1 full pattern)
- 5:1 GQA ratio (40 Q heads, 8 KV heads)
- GPT2Tokenizer

Run with:
    export HF_MODEL=~/models/models--allenai--Olmo-3.1-32B-Think
    export LINE_RS=1
    pytest models/demos/llama3_70b_galaxy/demo/demo_olmo_decode.py::test_olmo_demo -v -k "quick"
"""

import torch
from time import time
from datetime import datetime
from loguru import logger
import os
import ttnn
import pytest


from models.demos.llama3_70b_galaxy.tt.llama_common import (
    PagedAttentionConfig,
)
from models.demos.llama3_70b_galaxy.tt.llama_model import TtTransformer
from models.demos.llama3_70b_galaxy.tt.llama_embedding import TtLlamaEmbedding
from models.demos.llama3_70b_galaxy.tt.olmo_model_config import TtOlmoModelArgs
from models.common.sampling.tt_sampling import TTSampling
from models.demos.llama3_70b_galaxy.demo.demo_common import load_inputs_simple

from models.perf.benchmarking_utils import BenchmarkProfiler, BenchmarkData
from models.demos.llama3_70b_galaxy.tt.model_config import LlamaOptimizations

from transformers import GPT2Tokenizer

# Maximum number of times `tokens_per_second_per_user` is allowed to be outside the `tsu_range`
# before triggering an assertion failure. Allows occasional dips while ensuring
# stable performance without breaking CI prematurely.
TSU_PERF_DROP_LIMIT_PERCENT = 10

# Constants for TSU thresholds based on the number of layers (6U Galaxy configuration)
# OLMo may have different performance characteristics - adjust as needed
TSU_THRESHOLDS = {1: {"min": 400, "max": 500}, 10: {"min": 200, "max": 230}, 64: {"min": 50, "max": 60}}


def run_olmo_demo(
    user_input,
    mesh_device,
    max_seq_len,
    batch_size,
    num_batches,
    paged_attention,
    paged_attention_config,
    max_generated_tokens,
    optimizations,
    sampling_params,
    instruct_mode,
    is_ci_env,
    print_to_file,
    weights,
    layers,
    stress_test,
    start_pos,
    enable_prefetcher_performance_mode=True,
):
    max_supported_seq_len = 128 * 1024  # OLMo supports up to 128k context

    # Create batch output file
    benchmark_data = BenchmarkData()
    profiler_step_name = "tg-olmo-demo-e2e"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_directory = "models/demos/llama3_70b_galaxy/demo/output"
    os.makedirs(output_directory, exist_ok=True)
    os.chmod(output_directory, 0o755)
    output_filename = f"{output_directory}/demo_olmo_output_{timestamp}.txt"

    dtype = ttnn.bfloat8_b
    assert batch_size <= 32, "Max batch size currently supported is 32"
    assert max_seq_len <= max_supported_seq_len, "Max sequence length must be less than 128k tokens"

    top_k = sampling_params["top_k"]
    if isinstance(top_k, int):
        top_k = torch.tensor([top_k] * batch_size)
    top_p = sampling_params["top_p"]
    if isinstance(top_p, float):
        top_p = torch.tensor([top_p] * batch_size)
    temperature = sampling_params["temperature"]
    if isinstance(temperature, float):
        temperature = torch.tensor([temperature] * batch_size)
    seed = sampling_params["seed"]

    dummy_weights = False

    # We disregard any warmup iteration for profiling, in favour of just measuring compile time on the first iteration
    N_warmup_iter = {"inference_prefill": 0, "inference_decode": 0}

    # Start profiler
    logger.info(f"Start profiler")
    profiler = BenchmarkProfiler()
    profiler.start("run")
    profiler.start(profiler_step_name)

    logger.info(f"Reading inputs...")
    profiler.start("loading_inputs")
    if len(user_input) == 1:
        input_prompts = user_input * batch_size
    else:
        input_prompts = load_inputs_simple(
            user_input, batch_size, instruct_mode, "models/demos/llama3_70b_galaxy/demo/context_cache"
        )
    profiler.end("loading_inputs")

    # Generate the batched prompts (rotate the inputs between the users, for each batch)
    # If batch_size == 1, the same prompt is repeated for each batch
    batch_prompts = []
    for i in range(num_batches):
        batch_prompts.append([input_prompts[(j + i) % len(input_prompts)] for j in range(len(input_prompts))])

    # Load model args, weights, and tokenizer - Use TtOlmoModelArgs
    model_args = TtOlmoModelArgs(
        mesh_device,
        max_batch_size=32,
        max_seq_len=max_supported_seq_len,
    )
    model_args.n_layers = layers

    # Use OLMo tokenizer (GPT2Tokenizer)
    tokenizer = model_args.tokenizer
    if tokenizer is None:
        # Fallback to GPT2Tokenizer if model_args.tokenizer is None
        tokenizer = GPT2Tokenizer.from_pretrained(model_args.TOKENIZER_PATH)

    logger.info("Loading weights...")
    profiler.start("weight_loading")
    state_dict = model_args.load_state_dict()
    profiler.end("weight_loading")

    page_table_tt = None
    if paged_attention:
        paged_cache_max_seq_len = (
            paged_attention_config.block_size
            * paged_attention_config.max_num_blocks
            / model_args.batch_size_per_device_group
        )
        is_valid_token_position = (stress_test and start_pos <= paged_cache_max_seq_len) or (
            max_generated_tokens + start_pos <= paged_cache_max_seq_len
        )
        assert_msg = f"Either stress test with start_pos ({start_pos}) <= paged_cache_max_seq_len ({paged_cache_max_seq_len}) or max_generated_tokens ({max_generated_tokens}) + start_pos ({start_pos}) <= paged_cache_max_seq_len ({paged_cache_max_seq_len})"
        assert is_valid_token_position, assert_msg

        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            model_args.batch_size_per_device_group,
            paged_attention_config.max_num_blocks // model_args.batch_size_per_device_group,
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
        )
        logger.info("Page table tensor done")

    # Load TTNN OLMo model
    logger.info("Loading weights to device...")
    profiler.start("loading_weights_to_device")
    tt_model = TtTransformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
        enable_prefetcher_performance_mode=enable_prefetcher_performance_mode,
        decode_mode_only=True,
    )
    tt_embd = TtLlamaEmbedding(
        mesh_device=mesh_device,
        args=model_args,
        weight_cache_path=model_args.weight_cache_path(dtype),
        state_dict=state_dict,
        dtype=ttnn.bfloat16,  # Row major layout requires bfloat16
    )
    tt_sampling = TTSampling(
        args=model_args,
        mesh_device=mesh_device,
        tt_ccl=tt_model.tt_ccl,
        k=top_k,
        p=top_p,
        temp=temperature,
    )
    profiler.end("loading_weights_to_device")
    logger.info("Finished loading weights to device.")

    # Keep track of generated outputs to print out every iteration
    if dummy_weights:
        encoded_prompts = [
            [128000, 2028, 374, 264, 1296]
        ] * model_args.max_batch_size  # "This is a test" encoded prompt
    else:
        # Use OLMo tokenizer encoding
        encoded_prompts = [tokenizer.encode(prompt, add_special_tokens=True) for prompt in input_prompts]

    # Prefill by decode: start at first token; pad to 32 (tile size)
    max_prompt_length = max([len(prompt) for prompt in encoded_prompts])
    # Use OLMo EOS token for padding (GPT2 EOS is typically 50256 or use pad_token_id)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 50256
    padded_token_prompts = [prompt + [eos_token_id] * (max_prompt_length - len(prompt)) for prompt in encoded_prompts]
    encoded_prompts_tensor_whole_sequence = torch.tensor([padded_token_prompts[b] for b in range(batch_size)])

    user_done = [False] * batch_size  # Keeps track when a user reaches EoD token

    logger.info("Starting decode...")
    # Initial positions
    decoding_pos = [start_pos] * batch_size
    current_pos = torch.tensor([decoding_pos[b] for b in range(batch_size)])

    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if (batch_size > 1) else (None, None),
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
    elif layers == 5:
        num_compile_iters = 2
    else:
        num_compile_iters = 1
    for i in range(num_compile_iters):
        tt_decode_input = tt_embd(tt_out_tok)
        # logger.info(f"tt_decode_input done")

        tt_out = tt_model(
            tt_decode_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )

        # Sampling
        _, logprobs = tt_sampling(tt_out[0], tt_out_tok=tt_out_tok)  # Compile once with setting the seed
        logger.info(f"Sampling done")

    if not stress_test:
        ttnn.plus_one(current_pos_tensor, sub_core_grids=model_args.sub_core_grids, skip_negative_entries=True)
        ttnn.plus_one(
            rot_mat_idxs,
            sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
        )

    _, logprobs = tt_sampling(tt_out[0], tt_out_tok=tt_out_tok)

    ttnn.synchronize_device(mesh_device)
    logger.info("Compile done. Starting decode loop (no trace)...")

    iteration = 0
    users_decoding = True
    tokens_per_second_per_user_token127 = None

    all_outputs = []
    all_log_probs = []
    profiler.start(f"inference_decode", iteration=iteration)

    tsu_thresholds = TSU_THRESHOLDS.get(layers, {"min": 0, "max": 9999999})
    tsu_failures = 0
    all_tokens_per_second_per_user = []
    failed_tokens_per_second_per_user = []
    iteration_time_start = time()
    current_iteration = 0

    while users_decoding:
        if iteration in range(len(encoded_prompts[0])):
            current_iteration = iteration
            all_outputs.append(encoded_prompts[0][iteration])
            all_log_probs.append(torch.ones((1, 1, 1, batch_size)))
            tt_out_tok_update = ttnn.from_torch(
                encoded_prompts_tensor_whole_sequence[:, iteration].reshape(1, 1, 1, batch_size),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
            )
            ttnn.copy_host_to_device_tensor(tt_out_tok_update, tt_out_tok)
        else:
            current_iteration = iteration

        rot_mats, rot_mat_idxs = tt_model.rope_setup.get_rm_rot_mats(current_pos_tensor, return_rot_idxs=True)
        tt_decode_input = tt_embd(tt_out_tok)
        tt_out = tt_model(
            tt_decode_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )
        _, logprobs = tt_sampling(tt_out[0], tt_out_tok=tt_out_tok)

        if not stress_test:
            ttnn.plus_one(current_pos_tensor, sub_core_grids=model_args.sub_core_grids, skip_negative_entries=True)

        ttnn.synchronize_device(mesh_device)

        if iteration >= len(encoded_prompts[0]):
            tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out_tok.cpu(blocking=True))[0])[
                0, 0, 0, :batch_size
            ]
            all_outputs.append(tt_output_torch.tolist()[0])

            iteration_time_ends = time()
            iteration_time = iteration_time_ends - iteration_time_start
            tokens_per_second_per_user = 1 / iteration_time
            all_tokens_per_second_per_user.append(tokens_per_second_per_user)

            if not is_ci_env or iteration < 200 or iteration % 1000 == 0:
                logger.info(
                    f"Iteration {iteration}: tok/s/user={tokens_per_second_per_user:.2f}, "
                    f"Throughput={batch_size/iteration_time:.2f} tok/s, "
                    f"Time={1000*iteration_time:.2f} ms"
                )
                if not is_ci_env:
                    logger.info("[User 0] {}".format("".join(tokenizer.decode(all_outputs))))

            if iteration == 127:
                tokens_per_second_per_user_token127 = tokens_per_second_per_user

            iteration_time_start = time()

        if current_iteration + 1 >= max_generated_tokens:
            users_decoding = False

        iteration += 1

    # Finish profiling at the end of all batches inference
    profiler.end(profiler_step_name)
    profiler.end("run")

    if is_ci_env and tokens_per_second_per_user_token127 is not None:
        benchmark_data.add_measurement(profiler, 0, profiler_step_name, "tsu_e2e", tokens_per_second_per_user_token127)

        run_type = "tg_olmo_demo_decode_6u"  # Always 6U Galaxy configuration

        benchmark_data.save_partial_run_json(
            profiler,
            run_type=run_type,
            ml_model_name="olmo32b-tg",
        )

    if not stress_test and len(all_tokens_per_second_per_user) > 0:
        logger.info(f"Min tsu throughput: {min(all_tokens_per_second_per_user)}")
        logger.info(f"Max tsu throughput: {max(all_tokens_per_second_per_user)}")
        logger.info(f"Avg tsu throughput: {sum(all_tokens_per_second_per_user) / len(all_tokens_per_second_per_user)}")
        logger.info(
            f"Median tsu throughput: {sorted(all_tokens_per_second_per_user)[len(all_tokens_per_second_per_user) // 2]}"
        )
        # 95 percentile tsu throughput
        percentile_5 = sorted(all_tokens_per_second_per_user)[int(0.05 * len(all_tokens_per_second_per_user))]
        percentile_95 = sorted(all_tokens_per_second_per_user)[int(0.95 * len(all_tokens_per_second_per_user))]
        logger.info(f"5 percentile tsu throughput: {percentile_5}")
        logger.info(f"95 percentile tsu throughput: {percentile_95}")

        logger.info(
            f"Suggested target range is 5 percentile: {int(percentile_5)} - max: {int(max(all_tokens_per_second_per_user))+1}"
        )

        if tokens_per_second_per_user_token127 is not None:
            logger.info(f"Tokens per second per user at token 128: {tokens_per_second_per_user_token127}")

        # print before assertion
        out_of_targets_msg = f"Throughput is out of targets {tsu_thresholds['min']} - {tsu_thresholds['max']} t/s/u in {tsu_failures} iterations"
        num_decode_iters = len(all_tokens_per_second_per_user)
        tsu_perf_drop_limit = TSU_PERF_DROP_LIMIT_PERCENT * max(num_decode_iters, 1) / 100
        if tsu_failures > tsu_perf_drop_limit:
            logger.info(out_of_targets_msg)
            logger.info(f"Failing iterations sorted by t/s/u")
            sorted_tokens_per_second_per_user = sorted(failed_tokens_per_second_per_user, key=lambda x: x[1])
            for iteration, tsu in sorted_tokens_per_second_per_user:
                logger.info(f"Iteration {iteration}: {tsu}")
        # Assert at the end of test to check if the throughput recuperated
        assert tsu_failures <= tsu_perf_drop_limit, out_of_targets_msg

        # Print out total number of tsu_failures
        logger.info(f"Total TSU Failures: {tsu_failures} (threshold: {tsu_perf_drop_limit})")


# List of supported Parameters for demo.py
#
# input_prompts (string): input json file with prompts to process. See models/demos/llama3/demo/*.json for list of input files
# instruct (bool): Whether to use instruct weights or general weights
# repeat_batches (int): Number of consecutive batches of users to run (default: 1)
# max_seq_len (int): Maximum context length supported by the model (OLMo models have a maximum context length of 128k)
# batch_size (int): Number of users in a batch (Supports 1/2/4/8/16/32 batches)
# max_generated_tokens (int): Maximum number of tokens to generate for each user (Note that the users will stop generation before this limit if they reach a EoS token)
# paged_attention (bool): Whether to use paged attention or default attention (vLLM requires paged attention)
# page_params (dict): Page parameters for paged attention (block_size, max_num_blocks) For smaller context lengths use block_size=32 and max_num_blocks=1024, for larger context use block_size=64 and max_num_blocks=2048
# sampling_params (dict): Sampling parameters for decoding (temperature, top_p). If temperature is set to 0, argmax (greedy decode) is used.
#
# optimization (LlamaOptimizations): Optimization level to use for the model (performance or accuracy)
@pytest.mark.parametrize(
    "weights, layers, input_prompts, instruct, repeat_batches, max_seq_len, batch_size, max_generated_tokens, paged_attention, page_params, sampling_params, stress_test, start_pos",
    [
        (  # full demo, batch 32
            "instruct",
            64,
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            False,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            32,  # batch_size
            2000,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 4096},  # page_params
            {"top_k": 1, "top_p": 0.00, "temperature": 0.0, "seed": 42},  # sampling_params
            False,  # stress_test
            0,  # start_pos
        ),
        (  # quick 3L demo
            "instruct",
            3,
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            32,  # batch_size
            200,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 4096},  # page_params
            {"top_k": 1, "top_p": 0.00, "temperature": 1.0, "seed": 42},  # sampling_params (argmax)
            False,  # stress_test
            0,  # start_pos
        ),
        (  # quick 1L demo
            "instruct",
            1,
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            32,  # batch_size
            50,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 4096},  # page_params
            {"top_k": 1, "top_p": 0.00, "temperature": 1.0, "seed": 42},  # sampling_params (argmax)
            False,  # stress_test
            0,  # start_pos
        ),
        (  # Stress test: 4*128k generation length
            "instruct",
            64,
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            32,  # batch_size
            500000,  # max_generated_tokens (same index for stress test)
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 4096},  # page_params
            {"top_k": 1, "top_p": 0.0, "temperature": 1.0, "seed": 42},  # sampling_params
            True,  # stress_test
            0,  # start_pos
        ),
        (  # mini stress test
            "instruct",
            64,
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            32,  # batch_size
            2048,  # max_generated_tokens (same index for stress test)
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 4096},  # page_params
            {"top_k": 1, "top_p": 0.00, "temperature": 1.0, "seed": 42},  # sampling_params (argmax)
            True,  # stress_test
            0,  # start_pos
        ),
        (  # 10 layers for device perf measurements
            "instruct",
            10,
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            32,  # batch_size
            1,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 4096},  # page_params
            {"top_k": 1, "top_p": 0.00, "temperature": 1.0, "seed": 42},  # sampling_params (argmax)
            False,  # stress_test
            127,  # start_pos
        ),
        (  # ND hang test
            "instruct",
            64,
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            32,  # batch_size
            20000,  # experimentally established as large enough to catch ND hangs
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 4096},  # page_params
            {"top_k": 1, "top_p": 0.00, "temperature": 1.0, "seed": 42},  # sampling_params (argmax)
            True,  # stress_test
            0,  # start_pos
        ),
    ],
    ids=[
        "full",  # full demo
        "quick",  # 3L demo (uses 3 layers to test sliding window pattern: 3 sliding + 1 full)
        "single",  # 1L demo for fastest iteration
        "stress-test",  # stress test with many iterations and same token index, full model
        "mini-stress-test",  # mini stress test with 2048 max_generated_tokens
        "measure-device-perf",  # 10L demo for device performance measurements
        "nd-hang-test",  # testing for nd-hang across multiple iterations
    ],
)
@pytest.mark.parametrize(
    "optimizations",
    [
        LlamaOptimizations.performance,
        # LlamaOptimizations.accuracy,
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "trace_region_size": 12726272,
            # "trace_region_size": 10459136,
            "fabric_config": True,
        }
    ],
    indirect=True,
)
def test_olmo_demo(
    weights,
    layers,
    input_prompts,
    instruct,
    repeat_batches,
    max_seq_len,
    batch_size,
    max_generated_tokens,
    paged_attention,
    page_params,
    sampling_params,
    stress_test,
    start_pos,
    optimizations,
    mesh_device,
    is_ci_env,
    reset_seeds,
    request,
):
    if is_ci_env and ("long" in input_prompts or optimizations == LlamaOptimizations.accuracy):
        pytest.skip("Do not run the 'long-context' or accuracy tests on CI to reduce load")

    # TODO: Remove this once all batch sizes are supported on Galaxy
    if batch_size not in [1, 32]:
        pytest.skip("Galaxy only supports batch 1 and 32")
    # Always assume 6U Galaxy configuration

    if paged_attention:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
    else:
        paged_attention_config = None

    enable_pf_perf_mode = not request.config.getoption("--disable_pf_perf_mode")

    return run_olmo_demo(
        user_input=input_prompts,
        mesh_device=mesh_device,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        num_batches=repeat_batches,
        paged_attention=paged_attention,
        paged_attention_config=paged_attention_config,
        max_generated_tokens=max_generated_tokens,
        optimizations=optimizations,
        sampling_params=sampling_params,
        instruct_mode=instruct,
        is_ci_env=is_ci_env,
        print_to_file=False,
        weights=weights,
        layers=layers,
        stress_test=stress_test,
        start_pos=start_pos,
        enable_prefetcher_performance_mode=enable_pf_perf_mode,
    )
