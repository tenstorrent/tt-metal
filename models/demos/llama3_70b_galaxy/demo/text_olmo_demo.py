# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
OLMo-3.1-32B Text Demo for Galaxy TG.

Follows the Llama text_demo.py pattern using Generator for deterministic
warm-up, trace capture, and decode. OLMo-specific adaptations:
- TtOlmoModelArgs (YaRN RoPE, sliding window, 5:1 GQA)
- GPT2Tokenizer with ChatML chat template
- _safe_decode for OLMo's expanded vocab (50280 tokens)

Run with:
    export HF_MODEL=~/models/models--allenai--Olmo-3.1-32B-Think
    pytest models/demos/llama3_70b_galaxy/demo/text_olmo_demo.py::test_olmo_text_demo -v -k "quick"
"""

import torch
import os
import pytest
from datetime import datetime
from loguru import logger

import ttnn

from models.demos.llama3_70b_galaxy.tt.generator import Generator, SamplingParams
from models.demos.llama3_70b_galaxy.tt.olmo_model_config import TtOlmoModelArgs
from models.demos.llama3_70b_galaxy.tt.llama_model import TtTransformer
from models.demos.llama3_70b_galaxy.tt.model_config import LlamaOptimizations
from models.demos.llama3_70b_galaxy.demo.demo_common import load_inputs_simple
from models.tt_transformers.tt.common import (
    preprocess_inputs_prefill,
    PagedAttentionConfig,
)
from models.perf.benchmarking_utils import BenchmarkProfiler, BenchmarkData


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


def create_olmo_tt_model(
    mesh_device,
    max_batch_size,
    max_seq_len,
    num_layers,
    page_params,
    dtype=ttnn.bfloat8_b,
    use_paged_kv_cache=False,
    enable_prefetcher_performance_mode=True,
):
    model_args = TtOlmoModelArgs(
        mesh_device,
        max_batch_size=32,
        max_seq_len=max_seq_len,
    )
    model_args.n_layers = num_layers

    state_dict = model_args.load_state_dict()
    page_table = None
    paged_attention_config = None
    tt_kv_cache = None

    if use_paged_kv_cache:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
        # Implied shuffling of blocks (same as Llama text_demo.py).
        # For batch=1: page_table (1, all_blocks) → user 0 gets full capacity for long ISL.
        # For batch=32: page_table (32, blocks/32) → each user gets unique blocks.
        # The Generator's prepare_decode_inputs_host shards rows across columns.
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            max_batch_size, paged_attention_config.max_num_blocks // max_batch_size
        )

    model = TtTransformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
        enable_prefetcher_performance_mode=enable_prefetcher_performance_mode,
        decode_mode_only=False,
    )

    if use_paged_kv_cache:
        tt_kv_cache = [layer.attention.layer_past for layer in model.layers]

    return model_args, model, page_table, [tt_kv_cache]


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
    enable_prefetcher_performance_mode=True,
    is_cur_pos_sharded=True,
    stop_at_eos=False,
):
    enable_trace = True
    prefill_enable_trace = True
    print_outputs = not is_ci_env

    # Benchmark setup
    benchmark_data = BenchmarkData()
    profiler_step_name = "tg-olmo-demo-e2e"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_directory = "models/demos/llama3_70b_galaxy/demo/output"
    os.makedirs(output_directory, exist_ok=True)
    os.chmod(output_directory, 0o755)
    output_filename = f"{output_directory}/olmo_text_demo_output_{timestamp}.txt"

    assert batch_size <= 32, "Max batch size currently supported is 32"

    # Start profiler
    logger.info("Start profiler")
    profiler = BenchmarkProfiler()
    profiler.start("run")

    # ── Load inputs ──────────────────────────────────────────────────────
    logger.info("Reading inputs...")
    profiler.start("loading_inputs")
    if len(user_input) == 1:
        input_prompts = user_input * batch_size
    else:
        input_prompts = load_inputs_simple(
            user_input, batch_size, instruct_mode, "models/demos/llama3_70b_galaxy/demo/context_cache"
        )
    profiler.end("loading_inputs")

    # Build repeat-batch prompts (rotate prompts between users for each batch)
    repeat_batch_prompts = []
    for i in range(num_batches):
        repeat_batch_prompts.append([input_prompts[(j + i) % len(input_prompts)] for j in range(len(input_prompts))])

    # ── Create model ─────────────────────────────────────────────────────
    page_params = None
    if paged_attention:
        page_params = {
            "page_block_size": paged_attention_config.block_size,
            "page_max_num_blocks": paged_attention_config.max_num_blocks,
        }

    model_args, model, page_table, tt_kv_cache = create_olmo_tt_model(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_layers=layers,
        page_params=page_params,
        dtype=ttnn.bfloat8_b,
        use_paged_kv_cache=paged_attention,
        enable_prefetcher_performance_mode=enable_prefetcher_performance_mode,
    )

    tokenizer = model_args.tokenizer
    generator = Generator(model, model_args, mesh_device, tokenizer=tokenizer)

    num_tokens_generated_decode = []

    logger.info("Starting inference...")
    for batch_idx, batch_input_prompts in enumerate(repeat_batch_prompts):
        logger.info(f"Processing batch {batch_idx}")

        # ── Preprocess inputs ────────────────────────────────────────────
        profiler.start("preprocess_prefill_inputs", iteration=batch_idx)
        # Cap max_generated_tokens for prefill clipping (stress tests may exceed max_seq_len)
        prefill_gen_budget = min(max_generated_tokens, max_seq_len - 1024)
        (
            input_tokens_prefill_pt,
            encoded_prompts,
            decoding_pos,
            prefill_lens,
        ) = preprocess_inputs_prefill(
            batch_input_prompts,
            tokenizer,
            [model_args],
            instruct_mode,
            prefill_gen_budget,
            max_prefill_len=model_args.max_context_len,
        )

        max_encoded_prompt_len = max(len(p) for p in encoded_prompts)
        input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(batch_size, -1)

        batch_size_per_device_group = 32 if batch_size == 32 else 1
        if paged_attention:
            paged_cache_max_seq_len = (
                paged_attention_config.block_size * paged_attention_config.max_num_blocks / batch_size_per_device_group
            )
            # For stress tests with very large max_generated_tokens, skip the capacity check
            # (the page table wraps naturally via modular indexing)
            if max_generated_tokens + max_encoded_prompt_len > paged_cache_max_seq_len:
                logger.warning(
                    f"max_generated_tokens ({max_generated_tokens}) + prompt len ({max_encoded_prompt_len}) "
                    f"exceeds paged_cache_max_seq_len ({paged_cache_max_seq_len}). "
                    f"Continuing anyway (stress test mode)."
                )

        profiler.end("preprocess_prefill_inputs", iteration=batch_idx)

        # ── Construct sampling params ────────────────────────────────────
        temperature = sampling_params["temperature"]
        top_k = sampling_params.get("top_k", 32)
        top_p = sampling_params["top_p"]
        seed = sampling_params.get("seed", 42)
        device_sampling_params = SamplingParams(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )

        # ── KV cache reset for repeat batches ────────────────────────────
        if batch_idx != 0:
            model.switch_mode("prefill")
            for layer in model.layers:
                k_cache, v_cache = layer.attention.layer_past
                k_cache = ttnn.mul(k_cache, 0, output_tensor=k_cache)
                v_cache = ttnn.mul(v_cache, 0, output_tensor=v_cache)

        # Check if ISL fits within traced seqlens
        max_trace_seqlen = max(model.tt_ccl.support_seqlens)
        use_prefill_trace = prefill_enable_trace and max_encoded_prompt_len <= max_trace_seqlen

        # ── Prefill warmup (first batch only) ────────────────────────────
        # No sampling_params: OLMo's on-device sampling during prefill passes V
        # with wrong head count to paged_fused_update_cache, causing hang.
        if batch_idx == 0:
            logger.info(f"Starting prefill warmup (trace={use_prefill_trace}, ISL={max_encoded_prompt_len})...")
            profiler.start("compile_prefill", iteration=batch_idx)
            if not use_prefill_trace:
                # Long ISL (>4K): warmup with short tokens, then mode-cycle
                # to release prefill CCL trace buffers (decode→prefill with no alloc).
                warmup_tokens = torch.zeros(1, 128, dtype=torch.long)
                warmup_lens = torch.tensor([128])
                generator.prefill_forward_text(
                    warmup_tokens,
                    page_table=page_table,
                    kv_cache=tt_kv_cache,
                    prompt_lens=warmup_lens,
                    enable_trace=True,
                )
                model.switch_mode("decode")
                model.allocate_prefill_buffers = False
                model.switch_mode("prefill")
                model.allocate_prefill_buffers = True
            else:
                toks = generator.prefill_forward_text(
                    input_tokens_prefill_pt,
                    page_table=page_table,
                    kv_cache=tt_kv_cache,
                    prompt_lens=decoding_pos,
                    enable_trace=True,
                )
            profiler.end("compile_prefill", iteration=batch_idx)
            logger.info("Finished prefill warmup")

        # ── Actual prefill ───────────────────────────────────────────────
        logger.info("Starting prefill...")
        profiler.start("inference_prefill", iteration=batch_idx)
        toks = generator.prefill_forward_text(
            input_tokens_prefill_pt,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=decoding_pos,
            enable_trace=use_prefill_trace,
        )

        # toks is logits when sampling_params=None — take argmax
        prefilled_token = toks[:, :, :].argmax(dim=-1).view(-1, 1)
        profiler.end("inference_prefill", iteration=batch_idx)
        logger.info("Prefill finished")

        # ── Track outputs ────────────────────────────────────────────────
        all_outputs = [encoded_prompts[b][: prefill_lens[b]] for b in range(batch_size)]
        for user in range(batch_size):
            user_tok = int(prefilled_token[user].item())
            all_outputs[user].append(user_tok)

        user_done = [False] * batch_size

        # ── Decode setup ─────────────────────────────────────────────────
        current_pos = torch.tensor([decoding_pos[b] for b in range(batch_size)])
        if batch_size == 1:
            current_pos = torch.nn.functional.pad(current_pos, (0, 32 - current_pos.shape[0]), value=-1)
            page_table = torch.nn.functional.pad(page_table, (0, 0, 0, 32 - page_table.shape[0]), value=0)

        out_tok = prefilled_token
        if out_tok.shape == torch.Size([]) or (len(out_tok.shape) > 0 and out_tok.shape[0] != 32):
            out_tok = out_tok.repeat(32, 1)

        model.switch_mode("decode")
        logger.info(f"Starting decode loop from positions: {decoding_pos}")

        # ── Decode loop (Generator, async pipelined reads) ─────────────
        iteration = 0
        users_decoding = True
        read_events = []
        tt_out_toks = []

        profiler.start("inference_decode", iteration=batch_idx)

        while users_decoding:
            if iteration == 0:
                profiler.start("compile_decode", iteration=batch_idx)
            else:
                profiler.start(f"inference_decode_time_{iteration}", iteration=batch_idx)

            tt_out_tok, read_event = generator.decode_forward(
                out_tok,
                current_pos,
                enable_trace=enable_trace,
                page_table=page_table,
                kv_cache=tt_kv_cache,
                read_from_device=True,
                async_read=True,
                sampling_params=device_sampling_params,
                reset_inputs=iteration == 0,
                is_cur_pos_sharded=is_cur_pos_sharded and batch_size > 1,
                prompt_tokens=input_tokens_prefill_pt,
                output_tokens=prefilled_token,
            )
            read_events.append(read_event)
            tt_out_toks.append(tt_out_tok)

            if iteration == 0:
                profiler.end("compile_decode", iteration=batch_idx)
                decode_compile_time = profiler.get_duration("compile_decode", iteration=batch_idx)
                logger.info(f"Iteration {iteration} (compile): {1000*decode_compile_time:.4f}ms")

            if iteration > 0:
                ttnn.event_synchronize(read_events.pop(0)[0])
                tt_out_result = generator.process_decode_output_host(tt_out_toks.pop(0))
                out_tok = tt_out_result[0]

                if out_tok.shape == torch.Size([]) or (len(out_tok.shape) > 0 and out_tok.shape[0] != 32):
                    out_tok = out_tok.repeat(32, 1)

                for user in range(batch_size):
                    user_tok = out_tok.tolist()[user]
                    if user_tok not in tokenizer.stop_tokens and not user_done[user]:
                        all_outputs[user].append(user_tok)
                    else:
                        if stop_at_eos:
                            user_done[user] = True
                            if all(user_done):
                                users_decoding = False

                profiler.end(f"inference_decode_time_{iteration}", iteration=batch_idx)
                decode_iteration_time = profiler.get_duration(f"inference_decode_time_{iteration}", iteration=batch_idx)
                tokens_per_second_per_user = 1 / decode_iteration_time
                logger.info(
                    f"Decode iter {iteration}: tok/s/user={tokens_per_second_per_user:.2f}, "
                    f"Throughput={batch_size*tokens_per_second_per_user:.2f} tok/s, "
                    f"Time={1000*decode_iteration_time:.2f} ms"
                )

            if print_outputs and not is_ci_env and iteration > 0 and iteration % 10 == 0:
                for user in range(min(batch_size, 4)):
                    generated = all_outputs[user][len(encoded_prompts[user]) :]
                    text = _safe_decode(tokenizer, generated)
                    if len(text) > 120:
                        text = "..." + text[-117:]
                    logger.info(f"  [User {user}] {text}")

            current_pos += 1
            iteration += 1

            if users_decoding:
                users_decoding = iteration < max_generated_tokens

        # Final output
        logger.info("\nFinished decoding, printing final outputs...\n")
        for i in range(batch_size):
            prompt_text = _safe_decode(tokenizer, encoded_prompts[i])
            generated_ids = all_outputs[i][len(encoded_prompts[i]) :]
            generated_text = _safe_decode(tokenizer, generated_ids)
            short_prompt = (
                (prompt_text[:100] + "\n<long prompt...>\n" + prompt_text[-100:])
                if len(prompt_text) > 200
                else prompt_text
            )
            logger.info(
                f"\n==BATCH {batch_idx}\n==USER {i} - PROMPT\n{short_prompt}\n==USER {i} - OUTPUT\n{generated_text.strip()}\n"
            )

        num_tokens_generated_decode.append(iteration)

    profiler.end("inference_decode", iteration=batch_idx)
    profiler.end("run")

    # ── Performance summary ──────────────────────────────────────────────
    compile_prefill_time = profiler.get_duration("compile_prefill")
    compile_decode_time = profiler.get_duration("compile_decode")
    total_inference_prefill_time = profiler.get_duration("inference_prefill")

    total_inference_decode_time = 0
    for i in range(1, iteration):
        total_inference_decode_time += profiler.get_duration(f"inference_decode_time_{i}")

    avg_time_to_first_token = total_inference_prefill_time
    if iteration > 1:
        avg_decode_iteration_time = total_inference_decode_time / (iteration - 1)
        prefill_tok_s = prefill_lens[0] / total_inference_prefill_time * batch_size
        decode_tok_s_user = (num_tokens_generated_decode[0] - 1) / total_inference_decode_time
        decode_tok_s = decode_tok_s_user * batch_size
    else:
        avg_decode_iteration_time = 0
        prefill_tok_s = 0
        decode_tok_s_user = 0
        decode_tok_s = 0

    logger.info("")
    logger.info("=== Performance metrics ===")
    logger.info(f"Prefill compile time: {round(compile_prefill_time, 2)}s")
    logger.info(f"Decode compile time: {round(compile_decode_time, 2)}s")
    logger.info(f"Average Time to First Token (TTFT): {round(avg_time_to_first_token*1000, 2)}ms")
    logger.info(
        f"Average speed: {round(avg_decode_iteration_time * 1000, 2)}ms @ {round(decode_tok_s_user, 2)} tok/s/user ({round(decode_tok_s, 2)} tok/s throughput)"
    )
    logger.info(f"Batch size: {batch_size}, Layers: {layers}, Prefill len: {prefill_lens[0]}")

    # Decode perf at specific token positions
    tok_1_perf = profiler.get_duration("inference_decode_time_1") if 1 < iteration else 0
    tok_128_perf = profiler.get_duration("inference_decode_time_127") if 127 < iteration else 0
    if tok_1_perf > 0:
        logger.info(
            f"1st token decode time: {tok_1_perf*1000:.2f}ms [{round(1/tok_1_perf, 2)} t/s/u, {round((1/tok_1_perf)*batch_size, 2)} t/s]"
        )
    if tok_128_perf > 0:
        logger.info(
            f"128th token decode time: {tok_128_perf*1000:.2f}ms [{round(1/tok_128_perf, 2)} t/s/u, {round((1/tok_128_perf)*batch_size, 2)} t/s]"
        )

    # Save benchmark data for CI
    if is_ci_env and tok_128_perf > 0:
        benchmark_data.add_measurement(profiler, 0, profiler_step_name, "tsu_e2e", round(1 / tok_128_perf, 2))
        benchmark_data.save_partial_run_json(
            profiler,
            run_type="tg_olmo_text_demo",
            ml_model_name="olmo32b-tg",
        )


# =============================================================================
# Test parametrization
# =============================================================================
@pytest.mark.parametrize(
    "input_prompts, instruct, repeat_batches, max_seq_len, batch_size, max_generated_tokens, paged_attention, page_params, sampling_params, stop_at_eos, num_layers, is_cur_pos_sharded",
    [
        (  # batch-32: Throughput run, 32 users, short prompt
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_prefill_128.json",
            True,  # instruct mode
            1,  # repeat_batches
            128 * 1024,  # max_seq_len
            32,  # batch_size
            128,  # max_generated_tokens (decode tokens)
            True,  # paged_attention
            {"page_block_size": 64, "page_max_num_blocks": 2048},
            {"temperature": 0.0, "top_p": 0.08},  # argmax
            False,  # stop_at_eos
            64,  # num_layers
            True,  # is_cur_pos_sharded
        ),
        (  # batch-1: Latency run, 1 user, short prompt
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_prefill_128.json",
            True,
            1,
            128 * 1024,
            1,
            128,
            True,
            {"page_block_size": 64, "page_max_num_blocks": 4096},
            {"temperature": 0.0, "top_p": 0.08},
            False,
            64,
            False,  # is_cur_pos_sharded (batch=1)
        ),
        (  # quick: 3-layer smoke test, batch 32
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_prefill_128.json",
            True,
            1,
            128 * 1024,
            32,
            128,
            True,
            {"page_block_size": 64, "page_max_num_blocks": 4096},
            {"temperature": 0.0, "top_p": 0.08},
            False,
            3,
            True,
        ),
        (  # single: 1-layer, batch 32
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_prefill_128.json",
            True,
            1,
            128 * 1024,
            32,
            50,
            True,
            {"page_block_size": 64, "page_max_num_blocks": 4096},
            {"temperature": 0.0, "top_p": 0.08},
            False,
            1,
            True,
        ),
        (  # single-batch1: 1-layer, batch 1
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_prefill_128.json",
            True,
            1,
            128 * 1024,
            1,
            50,
            True,
            {"page_block_size": 64, "page_max_num_blocks": 4096},
            {"temperature": 0.0, "top_p": 0.08},
            False,
            1,
            False,
        ),
        (  # long-4k-b1: 4K context, 1 user
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_4k.json",
            True,
            1,
            128 * 1024,
            1,
            128,
            True,
            {"page_block_size": 64, "page_max_num_blocks": 2048},
            {"temperature": 0.6, "top_p": 0.95, "top_k": 50, "seed": 42},
            False,
            64,
            False,
        ),
        (  # long-8k-b1: 8K context, 1 user
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_8k.json",
            True,
            1,
            128 * 1024,
            1,
            128,
            True,
            {"page_block_size": 64, "page_max_num_blocks": 2048},
            {"temperature": 0.6, "top_p": 0.95, "top_k": 50, "seed": 42},
            False,
            64,
            False,
        ),
        (  # long-16k-b1: 16K context, 1 user
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_16k.json",
            True,
            1,
            128 * 1024,
            1,
            128,
            True,
            {"page_block_size": 64, "page_max_num_blocks": 2048},
            {"temperature": 0.6, "top_p": 0.95, "top_k": 50, "seed": 42},
            False,
            64,
            False,
        ),
        (  # long-32k-b1: 32K context, 1 user
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_32k.json",
            True,
            1,
            128 * 1024,
            1,
            128,
            True,
            {"page_block_size": 64, "page_max_num_blocks": 2048},
            {"temperature": 0.6, "top_p": 0.95, "top_k": 50, "seed": 42},
            False,
            64,
            False,
        ),
        (  # long-64k-b1: 64K context (max ISL), 1 user
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_64k.json",
            True,
            1,
            128 * 1024,
            1,
            128,
            True,
            {"page_block_size": 64, "page_max_num_blocks": 2048},
            {"temperature": 0.6, "top_p": 0.95, "top_k": 50, "seed": 42},
            False,
            64,
            False,
        ),
        (  # stress-test: Long stability run
            "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_prefill_128.json",
            True,
            1,
            128 * 1024,
            32,
            500000,
            True,
            {"page_block_size": 64, "page_max_num_blocks": 4096},
            {"temperature": 0.0, "top_p": 0.08},
            False,
            64,
            True,
        ),
    ],
    ids=[
        "batch-32",
        "batch-1",
        "quick",
        "single",
        "single-batch1",
        "long-4k-b1",
        "long-8k-b1",
        "long-16k-b1",
        "long-32k-b1",
        "long-64k-b1",
        "stress-test",
    ],
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
            "num_command_queues": 1,
            "worker_l1_size": 1345000,
            "fabric_config": True,
        }
    ],
    indirect=True,
)
def test_olmo_text_demo(
    input_prompts,
    instruct,
    repeat_batches,
    max_seq_len,
    batch_size,
    max_generated_tokens,
    paged_attention,
    page_params,
    sampling_params,
    stop_at_eos,
    num_layers,
    is_cur_pos_sharded,
    optimizations,
    mesh_device,
    is_ci_env,
    reset_seeds,
    request,
):
    if is_ci_env and ("long" in input_prompts or optimizations == LlamaOptimizations.accuracy):
        pytest.skip("Do not run 'long-context' or accuracy tests on CI")

    if batch_size not in [1, 32]:
        pytest.skip("Galaxy only supports batch 1 and 32")

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
        weights="instruct",
        layers=num_layers,
        enable_prefetcher_performance_mode=enable_pf_perf_mode,
        is_cur_pos_sharded=is_cur_pos_sharded,
        stop_at_eos=stop_at_eos,
    )
