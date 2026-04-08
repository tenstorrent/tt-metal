# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 text generation demo.

Simple prefill + decode loop following gpt-oss text_demo.py pattern.

Usage:
    pytest models/demos/gemma4/demo/text_demo.py -v --timeout=600

    # With fewer layers for testing:
    pytest models/demos/gemma4/demo/text_demo.py -v --timeout=600 -k "test_demo"
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tt.common import create_tt_model
from models.demos.utils.llm_demo_utils import create_benchmark_data
from models.perf.benchmarking_utils import BenchmarkProfiler


def run_generation(
    mesh_device,
    model_path,
    prompts,
    max_new_tokens=32,
    num_layers=None,
):
    """
    Run text generation with Gemma4.

    Args:
        mesh_device: TT device
        model_path: Path to model weights
        prompts: List of prompt strings
        max_new_tokens: Number of tokens to generate per prompt
        num_layers: Override layer count (for quick testing)

    Returns:
        List of generated text strings
    """
    from transformers import AutoTokenizer

    is_ci_env = os.environ.get("CI") == "true"
    batch_size = 1  # Gemma4 demo is single-user

    profiler = BenchmarkProfiler()
    profiler.start("run")

    # Load tokenizer
    profiler.start("loading_inputs")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logger.info(f"Tokenizer loaded from {model_path}")
    profiler.end("loading_inputs")

    # Create model
    max_seq_len = 128  # Keep very small for initial testing
    logger.info(f"Creating model with {num_layers or 'all'} layers, max_seq_len={max_seq_len}...")
    t0 = time.time()
    model_args, model, tt_kv_cache, state_dict = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        model_path=model_path,
        create_kv_cache=True,
    )
    logger.info(f"Model created in {time.time() - t0:.1f}s")

    is_mesh = hasattr(mesh_device, "shape")
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    generated_texts = []

    for prompt_idx, prompt in enumerate(prompts):
        logger.info(f"\n{'='*60}")
        logger.info(f"Prompt {prompt_idx}: {prompt}")

        # Tokenize
        input_ids = tokenizer.encode(prompt, return_tensors="pt").squeeze(0)  # [seq_len]
        prompt_len = input_ids.shape[0]
        # Pad to tile alignment for prefill
        padded_len = ((prompt_len + 31) // 32) * 32
        if padded_len > prompt_len:
            input_ids_padded = torch.nn.functional.pad(input_ids, (0, padded_len - prompt_len), value=0)
        else:
            input_ids_padded = input_ids
        logger.info(f"Prompt tokens: {prompt_len} (padded to {padded_len})")

        # Prefill
        logger.info("Prefilling...")
        profiler.start(f"compile_prefill", iteration=prompt_idx)

        import traceback as tb

        tokens_tt = ttnn.from_torch(
            input_ids_padded.unsqueeze(0).to(torch.int32),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=replicate,
        )
        embeds = model.embed_tokens(tokens_tt)
        embeds = ttnn.reshape(embeds, (1, 1, padded_len, model_args.hidden_size))
        embeds = ttnn.to_layout(embeds, ttnn.TILE_LAYOUT)

        # CPU tensors for per-layer input computation (E2B/E4B)
        import torch.nn.functional as F

        embeds_torch = (
            F.embedding(
                input_ids_padded.unsqueeze(0).long(),
                state_dict.get(
                    "model.language_model.embed_tokens.weight",
                    state_dict.get("model.embed_tokens.weight", torch.zeros(1)),
                ),
            )
            * model.embed_scale
        ).float()

        # Get last token tile for first decode token
        get_last_token = ((prompt_len - 1) // 32) * 32
        try:
            logits = model.ttnn_prefill_forward(
                embeds,
                page_table=None,
                kv_cache=tt_kv_cache,
                get_last_token=get_last_token,
                input_ids_torch=input_ids_padded.unsqueeze(0),
                embeds_torch=embeds_torch,
            )
        except Exception as e:
            logger.error(f"Prefill failed: {e}")
            tb.print_exc()
            raise

        # Sample first token (argmax from last position)
        if is_mesh:
            logits_cpu = ttnn.to_torch(ttnn.get_device_tensors(logits)[0])
        else:
            logits_cpu = ttnn.to_torch(logits)
        logits.deallocate(True)

        # Get logits at the actual last prompt position within the tile
        pos_in_tile = (prompt_len - 1) - get_last_token
        next_token = logits_cpu[0, 0, pos_in_tile, :].argmax().item()

        profiler.end(f"compile_prefill", iteration=prompt_idx)

        # Also record as inference_prefill (compile_prefill includes first-run compile cost)
        profiler.start(f"inference_prefill", iteration=prompt_idx)
        profiler.end(f"inference_prefill", iteration=prompt_idx)

        logger.info(
            f"Prefill done in {profiler.get_duration('compile_prefill', iteration=prompt_idx):.2f}s, "
            f"first token: {next_token} = '{tokenizer.decode([next_token])}'"
        )

        # Decode loop
        generated_tokens = [next_token]
        current_pos = prompt_len
        iteration = 0

        logger.info("Decoding...")
        profiler.start(f"inference_decode", iteration=prompt_idx)

        for step in range(max_new_tokens - 1):
            if iteration == 0:
                profiler.start(f"compile_decode", iteration=prompt_idx)
            else:
                profiler.start(f"inference_decode_time_{iteration}", iteration=prompt_idx)

            # Prepare decode input
            token_tensor = torch.tensor([[next_token]], dtype=torch.int32)
            token_tt = ttnn.from_torch(
                token_tensor,
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
                mesh_mapper=replicate,
            )

            position_idx = torch.tensor([current_pos], dtype=torch.int32)
            position_tt = ttnn.from_torch(
                position_idx,
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.int32,
                mesh_mapper=replicate,
            )

            decode_logits, _ = model.ttnn_decode_forward(
                token_tt,
                current_pos=position_tt,
                kv_cache=tt_kv_cache,
                input_ids_torch=token_tensor,
            )

            if is_mesh:
                logits_cpu = ttnn.to_torch(ttnn.get_device_tensors(decode_logits)[0])
            else:
                logits_cpu = ttnn.to_torch(decode_logits)
            decode_logits.deallocate(True)

            next_token = logits_cpu.squeeze().argmax().item()
            generated_tokens.append(next_token)
            current_pos += 1

            if iteration == 0:
                profiler.end(f"compile_decode", iteration=prompt_idx)
                decode_iteration_time = profiler.get_duration("compile_decode", iteration=prompt_idx)
            else:
                profiler.end(f"inference_decode_time_{iteration}", iteration=prompt_idx)
                decode_iteration_time = profiler.get_duration(
                    f"inference_decode_time_{iteration}", iteration=prompt_idx
                )

            tokens_per_second_per_user = 1 / decode_iteration_time
            logger.debug(
                f"Iteration {iteration}: {1000*decode_iteration_time:.0f}ms @ "
                f"{tokens_per_second_per_user:.1f} tok/s/user ({batch_size*tokens_per_second_per_user:.1f} tok/s throughput)"
            )

            iteration += 1

            # Check for EOS
            if next_token == tokenizer.eos_token_id:
                break

        profiler.end(f"inference_decode", iteration=prompt_idx)

        # Final output
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        full_text = prompt + generated_text
        generated_texts.append(full_text)

        short_prompt = (
            (prompt[:100] + "\n<long prompt not printed in full>\n" + prompt[-100:]) if len(prompt) > 200 else prompt
        )
        logger.info(f"\n==PROMPT {prompt_idx}\n{short_prompt}\n==OUTPUT {prompt_idx}\n{generated_text.strip()}\n")

    num_tokens_generated_decode = iteration  # from last prompt

    profiler.end("run")

    # ── Performance metrics ──────────────────────────────────────────────
    compile_prefill_time = profiler.get_duration("compile_prefill")
    compile_decode_time = profiler.get_duration("compile_decode")

    # inference_prefill is a zero-duration marker (prefill compile+run are not separated yet)
    total_inference_prefill_time = compile_prefill_time

    total_inference_decode_time = 0
    for i in range(1, num_tokens_generated_decode):  # Iteration 0 is the compile time
        total_inference_decode_time += profiler.get_duration(f"inference_decode_time_{i}")

    avg_time_to_first_token = total_inference_prefill_time / batch_size
    avg_decode_iteration_time = (
        total_inference_decode_time / (num_tokens_generated_decode - 1) if num_tokens_generated_decode > 1 else 0
    )

    prefill_tok_s = prompt_len / total_inference_prefill_time * batch_size if total_inference_prefill_time > 0 else 0
    decode_tok_s_user = (
        (num_tokens_generated_decode - 1) / total_inference_decode_time
        if num_tokens_generated_decode > 1 and total_inference_decode_time > 0
        else 0
    )
    decode_tok_s = decode_tok_s_user * batch_size

    measurements = {
        # Required measurements
        "compile_prefill": compile_prefill_time,
        "compile_decode": compile_decode_time,
        "inference_prefill": total_inference_prefill_time,
        "inference_decode": total_inference_decode_time,
        "prefill_time_to_token": avg_time_to_first_token,
        "prefill_t/s": prefill_tok_s,
        "decode_t/s/u": decode_tok_s_user,
        "decode_t/s": decode_tok_s,
        # Optional measurements
        "Total compile time": compile_prefill_time + compile_decode_time,
        "Full demo runtime": profiler.get_duration("run"),
    }

    # Decode performance at specific token milestones
    tok_1_perf = profiler.get_duration("inference_decode_time_1") if 1 < num_tokens_generated_decode else 0
    tok_128_perf = profiler.get_duration("inference_decode_time_127") if 127 < num_tokens_generated_decode else 0

    logger.info("")
    logger.info("=== Performance metrics ===")
    if tok_1_perf > 0:
        logger.info(
            f"1st token decode time: {tok_1_perf * 1000:.2f}ms "
            f"[{round(1 / tok_1_perf, 2)} t/s/u, {round((1 / tok_1_perf) * batch_size, 2)} t/s]"
        )
    if tok_128_perf > 0:
        logger.info(
            f"128th token decode time: {tok_128_perf * 1000:.2f}ms "
            f"[{round(1 / tok_128_perf, 2)} t/s/u, {round((1 / tok_128_perf) * batch_size, 2)} t/s]"
        )
    logger.info("==")
    logger.info(f"Prefill compile time: {round(compile_prefill_time, 2)}s")
    logger.info(f"Decode compile time: {round(compile_decode_time, 2)}s")
    logger.info("")
    logger.info(f"Average Time to First Token (TTFT): {round(avg_time_to_first_token * 1000, 2)}ms")
    logger.info(
        f"Average speed: {round(avg_decode_iteration_time * 1000, 2)}ms @ "
        f"{round(decode_tok_s_user, 2)} tok/s/user ({round(decode_tok_s, 2)} tok/s throughput)"
    )
    logger.info(f"Generated {num_tokens_generated_decode} tokens")
    logger.info(f"Full demo runtime: {round(profiler.get_duration('run'), 2)}s")

    # Save benchmark data for CI dashboard
    if is_ci_env:
        targets = {}  # No perf targets for Gemma4 yet
        bench_n_warmup_iter = {"inference_prefill": 0, "inference_decode": 1}
        benchmark_data = create_benchmark_data(profiler, measurements, bench_n_warmup_iter, targets)

        # Save the decode performance of every iteration for plotting
        for i in range(1, num_tokens_generated_decode):
            benchmark_data.add_measurement(
                profiler,
                0,
                "inference_decode",
                f"time_to_token_{i}",
                profiler.get_duration(f"inference_decode_time_{i}") * 1000,
                step_warm_up_num_iterations=None,
                target=None,
            )

        # Average decode performance for first 128 iterations (excluding compile)
        num_iterations_for_avg = min(128, num_tokens_generated_decode)
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

        model_name = "Gemma4"
        benchmark_data.save_partial_run_json(
            profiler,
            run_type="demo",
            ml_model_name=model_name,
            ml_model_type="llm",
            num_layers=num_layers or model_args.n_layers,
            batch_size=batch_size,
            config_params={},
            input_sequence_length=prompt_len,
            output_sequence_length=num_tokens_generated_decode,
        )

    return generated_texts


# ── Pytest entry points ──────────────────────────────────────────────────


@pytest.fixture
def model_path():
    return os.getenv("HF_MODEL") or os.getenv("GEMMA4_MODEL_PATH", "/proj_sw/user_dev/gemma4/gemma-4-26B-A4B-it")


def test_demo_single_layer(device, model_path):
    """Quick demo with 1 layer — verifies the pipeline works on single device."""
    prompts = ["The capital of France is"]
    results = run_generation(
        mesh_device=device,
        model_path=model_path,
        prompts=prompts,
        max_new_tokens=8,
        num_layers=1,
    )
    assert len(results) == 1
    assert len(results[0]) > len(prompts[0])


def test_demo_full_model(device, model_path):
    """Full model demo — requires sufficient DRAM for all layers."""
    prompts = ["Explain quantum computing in simple terms:"]
    results = run_generation(
        mesh_device=device,
        model_path=model_path,
        prompts=prompts,
        max_new_tokens=64,
    )
    assert len(results) == 1
    logger.info(f"Full model output: {results[0]}")
