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
from models.demos.llama3_70b_galaxy.tt.olmo_model_config import TtOlmoModelArgs
from models.demos.llama3_70b_galaxy.tt.generator import Generator
from models.common.sampling.tt_sampling import TTSampling
from models.demos.llama3_70b_galaxy.demo.demo_common import load_inputs_simple
from models.tt_transformers.tt.common import preprocess_inputs_prefill

from models.perf.benchmarking_utils import BenchmarkProfiler, BenchmarkData
from models.demos.llama3_70b_galaxy.tt.model_config import LlamaOptimizations

from transformers import GPT2Tokenizer


def _safe_decode(tokenizer, token_ids, skip_special_tokens=True):
    """Decode token IDs safely.

    OLMo's tokenizer has vocab size 50280 (GPT-2 base 50257 + 23 added special tokens).
    GPT2Tokenizer._convert_id_to_token uses self.decoder (the base 50257 vocab) and returns
    None for IDs 50257-50279.  If those IDs are not in all_special_ids, skip_special_tokens
    alone won't filter them and convert_tokens_to_string -> ''.join(tokens) raises TypeError.
    This wrapper explicitly drops any None tokens before joining.
    """
    tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
    return tokenizer.convert_tokens_to_string([t for t in tokens if t is not None])


def get_padded_prefill_len(seq_len: int) -> int:
    if seq_len <= 128:
        return 128
    if seq_len <= 1024:
        return 1024
    else:
        return 2 ** (seq_len - 1).bit_length()


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

    sampling_batch = max(batch_size, 32)
    top_k = sampling_params["top_k"]
    if isinstance(top_k, int):
        top_k = torch.tensor([top_k] * sampling_batch)
    top_p = sampling_params["top_p"]
    if isinstance(top_p, float):
        top_p = torch.tensor([top_p] * sampling_batch)
    temperature = sampling_params["temperature"]
    if isinstance(temperature, float):
        temperature = torch.tensor([temperature] * sampling_batch)
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

    page_table = None
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
        logger.info("Page table created")

    # Load TTNN OLMo model (decode_mode_only=False for prefill+decode)
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
        decode_mode_only=False,
    )
    profiler.end("loading_weights_to_device")
    logger.info("Finished loading weights to device. Model is in prefill mode.")

    # Create Generator — mirrors text_demo.py; handles semaphore reset, all-ISL warmup,
    # and trace capture/execute via prefill_forward_text.
    generator = Generator(tt_model, model_args, mesh_device)

    # Encode prompts — mirrors text_demo.py: uses preprocess_inputs_prefill which calls
    # model_args.encode_prompt() (applies ChatML via tokenizer.apply_chat_template for OLMo).
    if dummy_weights:
        encoded_prompts = [
            [128000, 2028, 374, 264, 1296]
        ] * model_args.max_batch_size  # "This is a test" encoded prompt
        input_tokens_prefill_pt = [
            torch.tensor(encoded_prompts[b], dtype=torch.int32).unsqueeze(0) for b in range(batch_size)
        ]
        decoding_pos = [len(encoded_prompts[b]) for b in range(batch_size)]
    else:
        input_tokens_prefill_pt, encoded_prompts, decoding_pos, _ = preprocess_inputs_prefill(
            input_prompts,
            model_args.tokenizer,
            [model_args],
            instruct_mode,
            max_generated_tokens,
            max_prefill_len=model_args.max_context_len,
        )
        # Clamp each prompt to the target ISL to prevent chat-template overhead from pushing
        # past a power-of-2 boundary that would break support_seqlens CCL buffer lookups.
        target_prefill_len = 1 << (max_generated_tokens.bit_length() - 1)
        encoded_prompts = [ep[:target_prefill_len] for ep in encoded_prompts]
        input_tokens_prefill_pt = [
            torch.tensor(encoded_prompts[b], dtype=torch.int32).unsqueeze(0) for b in range(batch_size)
        ]
        decoding_pos = [len(encoded_prompts[b]) for b in range(batch_size)]

    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 50256
    user_done = [False] * batch_size

    # ===================== PREFILL PHASE =====================
    logger.info("=" * 60)
    logger.info("PREFILL PHASE")
    logger.info("=" * 60)
    profiler.start("inference_prefill")

    max_prompt_length = max([len(prompt) for prompt in encoded_prompts])
    padded_prefill_len = get_padded_prefill_len(max_prompt_length)
    prompt_lengths = [len(encoded_prompts[b]) for b in range(batch_size)]
    logger.info(f"Max prompt length: {max_prompt_length}, padded: {padded_prefill_len}")

    # YaRN RoPE is now computed internally by prepare_prefill_inputs_host on first call
    # (use_yarn=True for OLMo in get_prefill_rot_mat), so no manual setup is needed here.

    # KV cache — wrap in list so generator.prefill_forward_text can unwrap with kv_cache[0]
    kv_list = [layer.attention.layer_past for layer in tt_model.layers]

    # Stack tokens to [batch, max_prompt_len] as expected by Generator
    tokens = torch.stack(input_tokens_prefill_pt).view(batch_size, -1)

    # --- Prefill via Generator (mirrors text_demo.py) ---
    # On first call, prefill_forward_text automatically:
    #   1. Calls warmup_prefill_traces which resets all CCL semaphores
    #   2. Warms up traces for all support_seqlens [128, 1024, 2048, 4096, 8192]
    #      — each ISL runs eager compile (self-resets semaphores) then trace capture
    #   3. Executes the actual prefill for the requested ISL
    # This is the same flow as text_demo.py, eliminating device-state hangs between runs.
    first_decode_tokens = []
    all_outputs_per_user = [[] for _ in range(batch_size)]

    logger.info("Starting prefill (compile + warmup on first call)...")
    profiler.start("compile_prefill")
    output_logits = generator.prefill_forward_text(
        tokens,
        page_table=page_table,
        kv_cache=[kv_list],
        prompt_lens=decoding_pos,
        enable_trace=True,
        sampling_params=None,  # return logits; greedy argmax done on host below
        empty_slots=list(range(batch_size)),
    )
    profiler.end("compile_prefill")

    # Greedy argmax on host-side logits → first decode token per user
    # output_logits: [batch, 1, vocab_size]
    first_decode_tokens = output_logits[:batch_size, 0, :].argmax(dim=-1).tolist()
    for u in range(batch_size):
        all_outputs_per_user[u] = list(encoded_prompts[u]) + [first_decode_tokens[u]]
        decoded_tok = _safe_decode(tokenizer, [first_decode_tokens[u]])
        logger.info(
            f"Prefill user {u}: prompt_len={decoding_pos[u]}, first_token={first_decode_tokens[u]} ({decoded_tok})"
        )

    profiler.end("inference_prefill")
    logger.info(f"Prefill complete for {batch_size} users.")

    # ===================== SWITCH TO DECODE MODE =====================
    logger.info("Switching to decode mode...")
    tt_model.switch_mode("decode")
    logger.info("Switched to decode mode.")

    # Create sampling with decode CCL
    tt_sampling = TTSampling(
        args=model_args,
        mesh_device=mesh_device,
        tt_ccl=tt_model.tt_ccl,
        k=top_k,
        p=top_p,
        temp=temperature,
    )

    # ===================== DECODE PHASE =====================
    logger.info("=" * 60)
    logger.info("DECODE PHASE")
    logger.info("=" * 60)

    # Decode starts from the end of each user's prompt (always 32 entries, -1 for inactive users)
    current_pos = torch.full((32,), -1, dtype=torch.long)
    for b in range(batch_size):
        current_pos[b] = prompt_lengths[b]

    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0),
            mesh_shape=model_args.cluster_shape,
        ),
    )

    rot_mats, rot_mat_idxs = tt_model.rope_setup.get_rm_rot_mats(current_pos, return_rot_idxs=True)

    # First decode tokens from prefill (always 32 entries for TG mesh)
    first_tokens_padded = first_decode_tokens[:batch_size] + [0] * (32 - batch_size)
    first_tokens_tensor = torch.tensor(first_tokens_padded).reshape(1, 1, 1, 32)
    tt_out_tok = ttnn.from_torch(
        first_tokens_tensor,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Decode page table (sharded for decode)
    if paged_attention:
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
        )
    else:
        page_table_tt = None

    # Compile decode
    logger.info("Compiling decode trace...")
    if layers == 1:
        num_compile_iters = 10
    elif layers == 5:
        num_compile_iters = 2
    else:
        num_compile_iters = 1
    for i in range(num_compile_iters):
        tt_decode_input = tt_model.embd(tt_out_tok)
        tt_out = tt_model(
            tt_decode_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )
        ttnn.manual_seed(seed, sub_core_grids=model_args.sub_core_grids, device=mesh_device)
        _ = tt_sampling(tt_out[0], tt_out_tok=tt_out_tok)
        logger.info(f"Decode compile iteration {i} done")

    if not stress_test:
        ttnn.plus_one(current_pos_tensor, sub_core_grids=model_args.sub_core_grids, skip_negative_entries=True)
        ttnn.plus_one(
            rot_mat_idxs,
            sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
        )

    _ = tt_sampling(tt_out[0], tt_out_tok=tt_out_tok)

    # Capture decode trace
    logger.info("Capturing decode trace...")
    profiler.start("capture_decode_trace")

    tt_model.tt_ccl.reset_gather_and_buffer_idx()

    decode_trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)

    rot_mats = tt_model.rope_setup.get_rm_rot_mats(rot_mat_idxs)
    tt_decode_input = tt_model.embd(tt_out_tok)
    tt_out = tt_model(
        tt_decode_input,
        current_pos_tensor,
        rot_mats=rot_mats,
        mode="decode",
        page_table=page_table_tt,
    )
    _ = tt_sampling(tt_out[0], tt_out_tok=tt_out_tok)

    if not stress_test:
        ttnn.plus_one(current_pos_tensor, sub_core_grids=model_args.sub_core_grids, skip_negative_entries=True)
        ttnn.plus_one(
            rot_mat_idxs,
            sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
        )

    ttnn.end_trace_capture(mesh_device, decode_trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Reset decode state for actual generation
    current_pos_reset = ttnn.from_torch(
        current_pos,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0),
            mesh_shape=model_args.cluster_shape,
        ),
    )
    ttnn.copy_host_to_device_tensor(current_pos_reset, current_pos_tensor)

    tt_out_tok_reset = ttnn.from_torch(
        first_tokens_tensor,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
    )
    ttnn.copy_host_to_device_tensor(tt_out_tok_reset, tt_out_tok)

    rot_mat_idxs_reset = tt_model.rope_setup.get_rm_rot_idxs(current_pos, on_host=True)
    ttnn.copy_host_to_device_tensor(rot_mat_idxs_reset, rot_mat_idxs)

    profiler.end("capture_decode_trace")
    ttnn.synchronize_device(mesh_device)
    logger.info("Decode trace captured. Starting decode loop...")

    # ===================== DECODE LOOP =====================
    iteration = 0
    users_decoding = True
    tokens_per_second_per_user_token127 = None

    all_outputs = list(encoded_prompts[0]) + [first_decode_tokens[0]]
    all_log_probs = []
    profiler.start(f"inference_decode", iteration=iteration)

    tsu_thresholds = TSU_THRESHOLDS.get(layers, {"min": 0, "max": 9999999})
    tsu_failures = 0
    all_tokens_per_second_per_user = []
    failed_tokens_per_second_per_user = []
    iteration_time_start = time()

    num_decode_tokens = max_generated_tokens - max_prompt_length
    if num_decode_tokens <= 0:
        num_decode_tokens = max_generated_tokens

    while users_decoding:
        ttnn.execute_trace(mesh_device, decode_trace_id, cq_id=0, blocking=True)

        tt_out_tok_cpu = tt_out_tok.cpu(blocking=True, cq_id=0)
        tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out_tok_cpu)[0])[0, 0, 0, :batch_size]
        tt_out_tok_cpu.deallocate(True)
        all_toks = tt_output_torch.tolist()
        all_outputs.append(all_toks[0])
        for u in range(batch_size):
            all_outputs_per_user[u].append(all_toks[u])

        iteration_time_ends = time()
        iteration_time = iteration_time_ends - iteration_time_start
        tokens_per_second_per_user = 1 / iteration_time
        all_tokens_per_second_per_user.append(tokens_per_second_per_user)

        if not is_ci_env or iteration < 200 or iteration % 1000 == 0:
            logger.info(
                f"Decode iter {iteration}: tok/s/user={tokens_per_second_per_user:.2f}, "
                f"Throughput={batch_size/iteration_time:.2f} tok/s, "
                f"Time={1000*iteration_time:.2f} ms"
            )
            if not is_ci_env:
                logger.info("[User 0] {}".format(_safe_decode(tokenizer, all_outputs)))

        if iteration == 127:
            tokens_per_second_per_user_token127 = tokens_per_second_per_user

        iteration_time_start = time()
        iteration += 1

        if iteration >= num_decode_tokens:
            users_decoding = False

    # Wait for all in-flight device ops (including last decode trace's CCL semaphore
    # resets) to complete, then explicitly zero all global semaphores while the device
    # is idle. This leaves L1 semaphores at 0 so the next pytest process can start
    # without a hardware reset.
    ttnn.synchronize_device(mesh_device)
    tt_model.tt_ccl.reset_global_semaphores()
    if hasattr(tt_model, "tt_ccl_prefill"):
        tt_model.tt_ccl_prefill.reset_global_semaphores()
    ttnn.synchronize_device(mesh_device)
    ttnn.release_trace(mesh_device, decode_trace_id)

    # Print per-user output for coherency check
    logger.info("\n" + "=" * 60)
    logger.info("PER-USER OUTPUT COHERENCY CHECK")
    logger.info("=" * 60)
    for u in range(batch_size):
        # skip_special_tokens avoids None entries from EOS/special tokens at long ISL
        prompt_text = _safe_decode(tokenizer, encoded_prompts[u])
        generated_ids = all_outputs_per_user[u][len(encoded_prompts[u]) :]
        generated_text = _safe_decode(tokenizer, generated_ids)
        logger.info(f"\n--- User {u} ---")
        logger.info(f"  PROMPT   : {prompt_text[:200]}")
        logger.info(f"  GENERATED: {generated_text}")

    # Finish profiling at the end of all batches inference
    profiler.end(profiler_step_name)
    profiler.end("run")

    # Report TTFT and decode performance summary
    compile_prefill_time = profiler.get_duration("compile_prefill")
    compile_decode_time = profiler.get_duration("capture_decode_trace")
    avg_ttft = profiler.get_duration("inference_prefill")
    num_decode_iters = len(all_tokens_per_second_per_user)
    if num_decode_iters > 0:
        avg_decode_iter_time = sum(1.0 / t for t in all_tokens_per_second_per_user) / num_decode_iters
        avg_decode_tok_s_user = num_decode_iters / sum(1.0 / t for t in all_tokens_per_second_per_user)
        avg_decode_tok_s = avg_decode_tok_s_user * batch_size
    else:
        avg_decode_iter_time = 0
        avg_decode_tok_s_user = 0
        avg_decode_tok_s = 0

    logger.info("=" * 60)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Prefill compile time: {round(compile_prefill_time, 2)}s")
    logger.info(f"Decode compile time: {round(compile_decode_time, 2)}s")
    logger.info(f"Time to First Token (TTFT): {round(avg_ttft * 1000, 2)} ms")
    logger.info(
        f"Decode speed: {round(avg_decode_iter_time * 1000, 2)} ms/tok @ {round(avg_decode_tok_s_user, 2)} tok/s/user ({round(avg_decode_tok_s, 2)} tok/s throughput)"
    )
    logger.info(f"Batch size: {batch_size}, Layers: {layers}, Prefill len: {padded_prefill_len}")
    logger.info("=" * 60)

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
        (  # full demo, batch 1
            "instruct",
            64,
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            False,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            1,  # batch_size
            200,  # max_generated_tokens
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
        (  # quick 1L demo, batch 1
            "instruct",
            1,
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            1,  # batch_size
            50,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 4096},  # page_params
            {"top_k": 1, "top_p": 0.00, "temperature": 1.0, "seed": 42},  # sampling_params (argmax)
            False,  # stress_test
            0,  # start_pos
        ),
        (  # 1L profiler run: 2 decode tokens for tracy (1L prefill + 1L decode)
            "instruct",
            1,
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            1,  # batch_size
            2,  # max_generated_tokens
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
        (  # full model batch 32 perf
            "instruct",
            64,
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            32,  # batch_size
            1128,  # max_generated_tokens (128 prefill + 1000 decode)
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 4096},  # page_params
            {"top_k": 1, "top_p": 0.00, "temperature": 1.0, "seed": 42},  # sampling_params (argmax)
            False,  # stress_test
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
        # ── ISL sweep: batch=1, 10 decode tokens, 64 layers ──────────────────────────────
        # paged_cache_max_seq_len = 8 × max_num_blocks  (batch_size_per_device_group=8 on TG)
        (  # isl-128-b1: ~128-token prefill + 200 decode (greedy, for coherence check)
            "instruct",
            64,
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_prefill_128.json",
            False,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            1,  # batch_size
            400,  # ~200 prefill (with chat template) + 200 decode
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 1024},  # capacity: 8192 tokens
            {"top_k": 1, "top_p": 0.00, "temperature": 0.0, "seed": 42},
            False,  # stress_test
            0,  # start_pos
        ),
        (  # isl-1k-b1: ~1k-token prefill + 1000 decode
            "instruct",
            64,
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_1k.json",
            False,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            1,  # batch_size
            2024,  # ~1024 prefill + 1000 decode
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 1024},  # capacity: 8192 tokens
            {"top_k": 1, "top_p": 0.00, "temperature": 0.0, "seed": 42},
            False,  # stress_test
            0,  # start_pos
        ),
        (  # isl-2k-b1: ~2k-token prefill + 10 decode
            "instruct",
            64,
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_2k.json",
            False,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            1,  # batch_size
            2058,  # ~2048 prefill + 10 decode
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 1024},  # capacity: 8192 tokens
            {"top_k": 1, "top_p": 0.00, "temperature": 0.0, "seed": 42},
            False,  # stress_test
            0,  # start_pos
        ),
        (  # isl-2k-long-b1: ~2k-token prefill + ~200 decode tokens (greedy coherence check)
            "instruct",
            64,
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_2k.json",
            False,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            1,  # batch_size
            2248,  # ~2048 prefill + 200 decode
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 1024},  # capacity: 8192 tokens
            {"top_k": 1, "top_p": 0.00, "temperature": 0.0, "seed": 42},
            False,  # stress_test
            0,  # start_pos
        ),
        (  # isl-4k-b1: ~4k-token prefill + 10 decode
            "instruct",
            64,
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_4k.json",
            False,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            1,  # batch_size
            4106,  # ~4096 prefill + 10 decode
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 1024},  # capacity: 8192 tokens
            {"top_k": 1, "top_p": 0.00, "temperature": 0.0, "seed": 42},
            False,  # stress_test
            0,  # start_pos
        ),
        (  # isl-8k-b1: ~8k-token prefill + 1000 decode
            "instruct",
            64,
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_8k.json",
            False,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            1,  # batch_size
            9192,  # 8192 prefill + 1000 decode
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 2048},  # capacity: 16384 tokens
            {"top_k": 1, "top_p": 0.00, "temperature": 0.0, "seed": 42},
            False,  # stress_test
            0,  # start_pos
        ),
        (  # isl-16k-b1: ~16k-token prefill + 10 decode
            "instruct",
            64,
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_16k.json",
            False,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            1,  # batch_size
            16393,  # 16383 prefill tokens + 10 decode (pads to 16384)
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 4096},  # capacity: 32768 tokens
            {"top_k": 1, "top_p": 0.00, "temperature": 0.0, "seed": 42},
            False,  # stress_test
            0,  # start_pos
        ),
        (  # isl-32k-b1: ~32k-token prefill + 10 decode (eager mode, no trace)
            # capacity: 4128 blocks × 64 tok/block / 8 (batch_size_per_device_group) = 33,024 tokens/user ≥ 32,777
            "instruct",
            64,
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_32k.json",
            False,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            1,  # batch_size
            32777,  # 32767 prefill tokens + 10 decode (pads to 32768)
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 4128},  # capacity: 33,024 tokens/user
            {"top_k": 1, "top_p": 0.00, "temperature": 0.0, "seed": 42},
            False,  # stress_test
            0,  # start_pos
        ),
        (  # isl-64k-b1: ~64k-token prefill + 10 decode (eager mode, no trace, model max context)
            # capacity: 8208 blocks × 64 tok/block / 8 (batch_size_per_device_group) = 65,664 tokens/user ≥ 65,545
            "instruct",
            64,
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_64k.json",
            False,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            1,  # batch_size
            65545,  # 65535 prefill tokens + 10 decode (pads to 65536)
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 8208},  # capacity: 65,664 tokens/user
            {"top_k": 1, "top_p": 0.00, "temperature": 0.0, "seed": 42},
            False,  # stress_test
            0,  # start_pos
        ),
        # ── ISL sweep: batch=32, 64 layers, 20 decode tokens (coherence check) ──────────────
        (  # isl-128-b32: ~128-token prefill + 20 decode, batch=32, 64 layers
            "instruct",
            64,
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_prefill_128.json",
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            32,  # batch_size
            148,  # ~128 prefill + 20 decode
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 4096},
            {"top_k": 1, "top_p": 0.00, "temperature": 1.0, "seed": 42},
            False,  # stress_test
            0,  # start_pos
        ),
        (  # isl-1k-b32: ~1k-token prefill + 20 decode, batch=32, 64 layers
            "instruct",
            64,
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_1k_b32.json",
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            32,  # batch_size
            1044,  # ~1024 prefill + 20 decode
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 4096},
            {"top_k": 1, "top_p": 0.00, "temperature": 1.0, "seed": 42},
            False,  # stress_test
            0,  # start_pos
        ),
        (  # isl-2k-b32: ~2k-token prefill + 20 decode, batch=32, 64 layers
            "instruct",
            64,
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_2k_b32.json",
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            32,  # batch_size
            2068,  # ~2048 prefill + 20 decode
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 4096},
            {"top_k": 1, "top_p": 0.00, "temperature": 1.0, "seed": 42},
            False,  # stress_test
            0,  # start_pos
        ),
        (  # isl-4k-b32: ~4k-token prefill + 20 decode, batch=32, 64 layers
            "instruct",
            64,
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_4k_b32.json",
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            32,  # batch_size
            4116,  # ~4096 prefill + 20 decode
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 4096},
            {"top_k": 1, "top_p": 0.00, "temperature": 1.0, "seed": 42},
            False,  # stress_test
            0,  # start_pos
        ),
    ],
    ids=[
        "full",  # full demo
        "quick",  # 3L demo (uses 3 layers to test sliding window pattern: 3 sliding + 1 full)
        "single",  # 1L demo for fastest iteration (batch=32)
        "single-batch1",  # 1L demo for batch=1 testing
        "profiler",  # 1L, batch=1, 2 decode tokens for tracy profiling (1L prefill + 1L decode)
        "full-batch32",  # 64L, batch=32, 200 tokens for E2E perf
        "stress-test",  # stress test with many iterations and same token index, full model
        "mini-stress-test",  # mini stress test with 2048 max_generated_tokens
        "measure-device-perf",  # 10L demo for device performance measurements
        "nd-hang-test",  # testing for nd-hang across multiple iterations
        "isl-128-b1",  # ISL sweep: 128-token prefill, batch=1
        "isl-1k-b1",  # ISL sweep: 1k-token prefill, batch=1
        "isl-2k-b1",  # ISL sweep: 2k-token prefill, batch=1
        "isl-2k-long-b1",  # 2k prefill + ~2k decode for coherence verification
        "isl-4k-b1",  # ISL sweep: 4k-token prefill, batch=1
        "isl-8k-b1",  # ISL sweep: 8k-token prefill, batch=1 (traced, pre-allocated CCL buffers)
        "isl-16k-b1",  # ISL sweep: 16k-token prefill, batch=1 (traced, pre-allocated CCL buffers)
        "isl-32k-b1",  # ISL sweep: 32k-token prefill, batch=1 (eager mode, CCL barrier sync)
        "isl-64k-b1",  # ISL sweep: 64k-token prefill, batch=1 (eager mode, model max context)
        "isl-128-b32",  # ISL sweep: 128-token prefill, batch=32, 1 layer
        "isl-1k-b32",  # ISL sweep: 1k-token prefill, batch=32, 1 layer
        "isl-2k-b32",  # ISL sweep: 2k-token prefill, batch=32, 1 layer
        "isl-4k-b32",  # ISL sweep: 4k-token prefill, batch=32, 1 layer
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
            "trace_region_size": 184915840,
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
