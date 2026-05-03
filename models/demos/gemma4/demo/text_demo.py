# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 text generation demo.

Simple prefill + decode loop following gpt-oss text_demo.py pattern.

Usage:
    pytest models/demos/gemma4/demo/text_demo.py -v --timeout=600

    # With fewer layers for testing:
    pytest models/demos/gemma4/demo/text_demo.py -v --timeout=600 -k "test_demo"
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.common.sampling import SamplingParams, format_sampling_params
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4.tt.common import create_tt_model
from models.demos.gemma4.tt.model_config import DEFAULT_GEMMA4_MODEL
from models.demos.utils.llm_demo_utils import create_benchmark_data
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.common import PagedAttentionConfig
from models.tt_transformers.tt.model_config import determine_device_name


def _load_tokenizer(model_path):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except AttributeError as exc:
        if "'list' object has no attribute 'keys'" not in str(exc):
            raise
        logger.warning(
            "Installed Transformers cannot parse Gemma4 tokenizer extra_special_tokens list; retrying with empty mapping"
        )
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, extra_special_tokens={})


def _set_fabric_1d():
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    except TypeError:
        ttnn.set_fabric_config(
            ttnn.FabricConfig.FABRIC_1D,
            None,
            None,
            ttnn.FabricTensixConfig.DISABLED,
            ttnn.FabricUDMMode.DISABLED,
            ttnn.FabricManagerMode.DEFAULT,
        )


def _encode_prompt(tokenizer, prompt, instruct=True):
    if instruct and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        chat_result = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return chat_result["input_ids"].squeeze(0)
    if instruct:
        bos_token = tokenizer.bos_token or ""
        prompt = f"{bos_token}<|turn>user\n{prompt.strip()}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
        return tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").squeeze(0)
    return tokenizer.encode(prompt, return_tensors="pt").squeeze(0)


def _sample_next_token_host(logits, *, temperature=0.0, top_p=1.0, top_k=1, generator=None):
    logits = logits.float().flatten()
    if temperature == 0 or top_k == 1:
        return int(torch.argmax(logits).item())

    logits = logits / temperature
    if top_k and top_k > 0:
        top_k = min(int(top_k), logits.numel())
        values, indices = torch.topk(logits, top_k)
        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(0, indices, values)
        logits = filtered

    probs = torch.softmax(logits, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative_probs > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_probs[remove] = 0
        probs = torch.zeros_like(probs).scatter(0, sorted_indices, sorted_probs)
        probs = probs / probs.sum()

    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def _make_sampling_params(temperature, top_p, top_k, seed, greedy):
    if greedy:
        temperature = 0.0
        top_p = 0.0
        top_k = 1
    return SamplingParams(temperature=temperature, top_p=top_p, top_k=top_k, seed=seed)


def _configure_on_device_sampling(model, sampling_params):
    if model.sampling is None:
        return
    formatted = format_sampling_params(sampling_params, model.sampling.tt_sampling.max_batch_size)
    model.sampling.reset_sampling_params(formatted)
    model.sampling.seed_manager.reset_seed(formatted.seed, [0])
    model.sampling.seed_manager.get_new_values([0])


def run_generation(
    mesh_device,
    model_path,
    prompts,
    tokenizer_path=None,
    max_new_tokens=32,
    num_layers=None,
    max_seq_len=4096,
    page_params=None,
    enable_decode_trace=True,
    instruct=True,
    temperature=0.0,
    top_p=1.0,
    top_k=1,
    seed=None,
    greedy=True,
    allow_cpu_sampling_fallback=False,
):
    """
    Run text generation with Gemma4.

    Args:
        mesh_device: TT device
        model_path: Path to model weights
        prompts: List of prompt strings
        max_new_tokens: Number of tokens to generate per prompt
        num_layers: Override layer count (for quick testing)
        max_seq_len: Maximum sequence length (determines KV cache size)
        page_params: Paged attention params dict with "page_block_size" and "page_max_num_blocks"

    Returns:
        List of generated text strings
    """
    is_ci_env = os.environ.get("CI") == "true"
    batch_size = 1  # Gemma4 demo is single-user

    profiler = BenchmarkProfiler()
    profiler.start("run")

    # Load tokenizer
    profiler.start("loading_inputs")
    tokenizer_source = tokenizer_path or model_path
    tokenizer = _load_tokenizer(tokenizer_source)
    logger.info(f"Tokenizer loaded from {tokenizer_source}")
    profiler.end("loading_inputs")

    # Paged attention config
    if page_params is None:
        page_params = {"page_block_size": 64, "page_max_num_blocks": max_seq_len // 64}
    paged_attention_config = PagedAttentionConfig(
        block_size=page_params["page_block_size"],
        max_num_blocks=page_params["page_max_num_blocks"],
    )

    # Create page table (identity mapping for single-user)
    page_table = torch.arange(paged_attention_config.max_num_blocks, dtype=torch.int32).reshape(
        batch_size, paged_attention_config.max_num_blocks
    )

    # Create model
    logger.info(f"Creating model with {num_layers or 'all'} layers, max_seq_len={max_seq_len}...")
    t0 = time.time()
    model_args, model, tt_kv_cache, state_dict = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        model_path=model_path,
        create_kv_cache=True,
        paged_attention_config=paged_attention_config,
    )
    logger.info(f"Model created in {time.time() - t0:.1f}s")

    is_mesh = hasattr(mesh_device, "shape")
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    sampling_params = _make_sampling_params(temperature, top_p, top_k, seed, greedy)
    on_device_sampling_available = model.sampling is not None
    if on_device_sampling_available:
        _configure_on_device_sampling(model, sampling_params)
        sample_mode = "device"
    elif allow_cpu_sampling_fallback:
        sample_mode = "host-debug"
        logger.warning("On-device sampling is unavailable; using explicit CPU debug fallback")
    else:
        raise RuntimeError(
            "On-device sampling is unavailable for this mesh/config. "
            "Pass allow_cpu_sampling_fallback=True or --allow-cpu-sampling-fallback for debug-only host sampling."
        )
    host_rng = torch.Generator()
    if seed is not None:
        host_rng.manual_seed(int(seed))

    # Page table on device
    page_table_tt = ttnn.from_torch(
        page_table,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=replicate,
    )

    generated_texts = []

    for prompt_idx, prompt in enumerate(prompts):
        logger.info(f"\n{'='*60}")
        logger.info(f"Prompt {prompt_idx}: {prompt}")

        input_ids = _encode_prompt(tokenizer, prompt, instruct=instruct)

        prompt_len = input_ids.shape[0]
        # Pad to standard prefill lengths (matches tt_transformers/gpt_oss pattern)
        if prompt_len <= 128:
            padded_len = 128
        elif prompt_len <= 1024:
            padded_len = 1024
        else:
            padded_len = 2 ** (prompt_len - 1).bit_length()
        input_ids_padded = torch.nn.functional.pad(input_ids, (0, padded_len - prompt_len), value=0)
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
                page_table=page_table_tt,
                kv_cache=tt_kv_cache,
                get_last_token=get_last_token,
                input_ids_torch=input_ids_padded.unsqueeze(0),
                embeds_torch=embeds_torch,
            )
        except Exception as e:
            logger.error(f"Prefill failed: {e}")
            tb.print_exc()
            raise

        # Sample first token from prefill logits.  Prefill returns gathered logits
        # today, so the first token still uses the host sampler.  Decode uses
        # device-side sampling whenever the mesh supports it.
        if is_mesh:
            logits_cpu = ttnn.to_torch(ttnn.get_device_tensors(logits)[0])
        else:
            logits_cpu = ttnn.to_torch(logits)
        logits.deallocate(True)

        # Get logits at the actual last prompt position within the tile
        pos_in_tile = (prompt_len - 1) - get_last_token
        next_token = _sample_next_token_host(
            logits_cpu[0, 0, pos_in_tile, :],
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
            generator=host_rng,
        )

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
        trace_id = None
        trace_output = None
        trace_device_inputs = None

        # ── Decode helpers ─────────────────────────────────────────────────
        # Embedding + PLI computed on host (fast for single-token decode),
        # transferred as ROW_MAJOR to device.  Trace captures decoder layers onward.
        # Sampling: SamplingGenerator for TP >= 2, host torch.argmax for TP = 1.
        on_device_sampling = model.sampling is not None
        if on_device_sampling:
            _configure_on_device_sampling(model, sampling_params)

        def _advance_device_seed():
            if on_device_sampling:
                model.sampling.seed_manager.get_new_values([0])

        def _make_decode_inputs(tok, pos):
            """Create host tensors for one decode iteration."""
            embeds_torch, pli_torch = model.compute_host_embeddings(tok)
            # ROW_MAJOR on host — TILE conversion happens on device inside the trace.
            embeds_h = ttnn.from_torch(
                embeds_torch,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=replicate,
            )
            pos_padded = torch.nn.functional.pad(
                torch.tensor([pos], dtype=torch.int32).reshape(1, 1), (0, 31), "constant", 0
            )
            inputs = {
                "embeds": embeds_h,
                "position": ttnn.from_torch(
                    pos_padded,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    dtype=ttnn.uint32,
                    mesh_mapper=replicate,
                ),
                "position_int32": ttnn.from_torch(
                    torch.tensor([pos], dtype=torch.int32),
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    dtype=ttnn.int32,
                    mesh_mapper=replicate,
                ),
            }
            if pli_torch is not None:
                inputs["pli"] = ttnn.from_torch(
                    pli_torch,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    dtype=ttnn.bfloat16,
                    mesh_mapper=replicate,
                )
            return inputs

        def _fwd(device_inputs):
            return model.ttnn_decode_forward(
                x=device_inputs["embeds"],
                current_pos=device_inputs["position"],
                rot_mat_idxs=device_inputs["position_int32"],  # pos_int32 passed as rot_mat_idxs
                page_table=page_table_tt,
                kv_cache=tt_kv_cache,
                sampling_on_device=on_device_sampling,
                pli_combined=device_inputs.get("pli"),
            )

        def _inputs_to_device(inputs):
            return {k: ttnn.to_device(v, device=mesh_device) for k, v in inputs.items() if v is not None}

        def _copy_inputs_to_trace(host_inputs):
            for k, v in host_inputs.items():
                if v is not None and k in trace_device_inputs:
                    ttnn.copy_host_to_device_tensor(v, trace_device_inputs[k])

        def _extract_token(decode_output):
            """Extract next token from model output (token IDs or logits)."""
            output_cpu = (
                ttnn.to_torch(ttnn.get_device_tensors(decode_output)[0]) if is_mesh else ttnn.to_torch(decode_output)
            )
            if on_device_sampling:
                return output_cpu.reshape(-1)[0].item()
            else:
                return _sample_next_token_host(
                    output_cpu.squeeze(),
                    temperature=sampling_params.temperature,
                    top_p=sampling_params.top_p,
                    top_k=sampling_params.top_k,
                    generator=host_rng,
                )

        logger.info(
            f"Decoding (trace={'ON' if enable_decode_trace else 'OFF'}, " f"embedding=host, sampling={sample_mode})..."
        )
        profiler.start(f"inference_decode", iteration=prompt_idx)

        # Disable Python GC during decode to avoid pause spikes; collect once before.
        # The decode loop, trace capture, and trace execution all sit inside a try
        # so that GC is always restored and any captured trace is always released
        # — otherwise an exception leaves GC disabled for the rest of the pytest
        # worker (contaminating unrelated tests) and leaks the trace handle.
        gc.collect()
        gc_was_enabled = gc.isenabled()
        gc.disable()

        try:
            # ── Main decode loop (mode-agnostic) ──────────────────────────────
            for step in range(max_new_tokens - 1):
                if iteration == 0:
                    profiler.start(f"compile_decode", iteration=prompt_idx)
                else:
                    profiler.start(f"inference_decode_time_{iteration}", iteration=prompt_idx)

                t_make_start = time.perf_counter()
                inputs_h = _make_decode_inputs(next_token, current_pos)
                t_make_end = time.perf_counter()

                if enable_decode_trace and trace_id is not None:
                    # ── Traced execution: copy inputs and replay ──
                    _copy_inputs_to_trace(inputs_h)
                    _advance_device_seed()
                    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
                    decode_logits = trace_output
                    t_enq_end = time.perf_counter()

                elif enable_decode_trace and iteration == 0:
                    # ── Iteration 0: compile run + trace capture ──
                    # 1. Compile run (un-traced)
                    inputs_d = _inputs_to_device(inputs_h)
                    _advance_device_seed()
                    decode_logits, _ = _fwd(inputs_d)
                    next_token = _extract_token(decode_logits)
                    generated_tokens.append(next_token)
                    current_pos += 1
                    profiler.end(f"compile_decode", iteration=prompt_idx)
                    decode_iteration_time = profiler.get_duration("compile_decode", iteration=prompt_idx)
                    logger.debug(
                        f"Iteration {iteration} (compile): {1000*decode_iteration_time:.0f}ms @ "
                        f"{1/decode_iteration_time:.1f} tok/s/user"
                    )
                    iteration += 1

                    # 2. Capture trace with fresh device buffers
                    logger.info("Capturing decode trace...")
                    inputs_h2 = _make_decode_inputs(next_token, current_pos)
                    trace_device_inputs = _inputs_to_device(inputs_h2)

                    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                    _advance_device_seed()
                    trace_output, _ = _fwd(trace_device_inputs)
                    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
                    logger.info("Decode trace captured")

                    # 3. Execute trace for current iteration
                    profiler.start(f"inference_decode_time_{iteration}", iteration=prompt_idx)
                    _copy_inputs_to_trace(inputs_h2)
                    _advance_device_seed()
                    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
                    decode_logits = trace_output
                    t_enq_end = time.perf_counter()

                else:
                    # ── No tracing: straightforward forward ──
                    inputs_d = _inputs_to_device(inputs_h)
                    _advance_device_seed()
                    decode_logits, _ = _fwd(inputs_d)
                    t_enq_end = time.perf_counter()

                next_token = _extract_token(decode_logits)
                t_sync_end = time.perf_counter()
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
                host_inputs_ms = 1000 * (t_make_end - t_make_start)
                copy_enq_ms = 1000 * (t_enq_end - t_make_end)
                exec_sync_ms = 1000 * (t_sync_end - t_enq_end)
                logger.debug(
                    f"Iteration {iteration}: {1000*decode_iteration_time:.0f}ms @ "
                    f"{tokens_per_second_per_user:.1f} tok/s/user ({batch_size*tokens_per_second_per_user:.1f} tok/s throughput) "
                    f"| host_inputs={host_inputs_ms:.1f}ms copy+enq={copy_enq_ms:.1f}ms exec+sync={exec_sync_ms:.1f}ms"
                )

                iteration += 1

                # Check for EOS
                if next_token == tokenizer.eos_token_id:
                    break

        finally:
            if gc_was_enabled:
                gc.enable()
            if trace_id is not None:
                ttnn.release_trace(mesh_device, trace_id)

        profiler.end(f"inference_decode", iteration=prompt_idx)

        # Final output
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        full_text = prompt + generated_text
        generated_texts.append(full_text)

        short_prompt = (
            (prompt[:100] + "\n<long prompt not printed in full>\n" + prompt[-100:]) if len(prompt) > 200 else prompt
        )
        logger.info(f"\n==PROMPT {prompt_idx}\n{short_prompt}\n==OUTPUT {prompt_idx}\n{generated_text.strip()}\n")

    num_decode_iterations = iteration  # from last prompt, excluding the prefill-sampled token
    num_tokens_generated = len(generated_tokens)

    profiler.end("run")

    # ── Performance metrics ──────────────────────────────────────────────
    compile_prefill_time = profiler.get_duration("compile_prefill")
    compile_decode_time = profiler.get_duration("compile_decode") if num_decode_iterations > 0 else 0

    # inference_prefill is a zero-duration marker (prefill compile+run are not separated yet)
    total_inference_prefill_time = compile_prefill_time

    total_inference_decode_time = 0
    for i in range(1, num_decode_iterations):  # Iteration 0 is the compile time
        total_inference_decode_time += profiler.get_duration(f"inference_decode_time_{i}")

    avg_time_to_first_token = total_inference_prefill_time / batch_size
    avg_decode_iteration_time = (
        total_inference_decode_time / (num_decode_iterations - 1) if num_decode_iterations > 1 else 0
    )

    prefill_tok_s = prompt_len / total_inference_prefill_time * batch_size if total_inference_prefill_time > 0 else 0
    decode_tok_s_user = (
        (num_decode_iterations - 1) / total_inference_decode_time
        if num_decode_iterations > 1 and total_inference_decode_time > 0
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
    tok_1_perf = profiler.get_duration("inference_decode_time_1") if 1 < num_decode_iterations else 0
    tok_128_perf = profiler.get_duration("inference_decode_time_127") if 127 < num_decode_iterations else 0

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
    logger.info(f"Generated {num_tokens_generated} tokens")
    logger.info(f"Full demo runtime: {round(profiler.get_duration('run'), 2)}s")

    # Save benchmark data for CI dashboard
    if is_ci_env:
        targets = {}  # No perf targets for Gemma4 yet
        bench_n_warmup_iter = {"inference_prefill": 0, "inference_decode": 1}
        benchmark_data = create_benchmark_data(profiler, measurements, bench_n_warmup_iter, targets)

        # Save the decode performance of every iteration for plotting
        for i in range(1, num_decode_iterations):
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
        num_iterations_for_avg = min(128, num_decode_iterations)
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

        benchmark_data.save_partial_run_json(
            profiler,
            run_type="demo",
            ml_model_name="gemma4",
            ml_model_type="llm",
            device_name=determine_device_name(mesh_device),
            num_layers=num_layers or model_args.num_hidden_layers,
            batch_size=batch_size,
            config_params={},
            input_sequence_length=prompt_len,
            output_sequence_length=num_tokens_generated,
        )

    return generated_texts


# ── Pytest entry points ──────────────────────────────────────────────────


@pytest.fixture
def model_path():
    return os.getenv("HF_MODEL") or os.getenv("GEMMA4_MODEL_PATH", DEFAULT_GEMMA4_MODEL)


def test_demo_single_layer(device, model_path):
    """Quick demo with 1 layer — verifies the pipeline works on single device."""
    prompts = ["The capital of France is"]
    results = run_generation(
        mesh_device=device,
        model_path=model_path,
        prompts=prompts,
        max_new_tokens=8,
        num_layers=1,
        allow_cpu_sampling_fallback=True,
    )
    assert len(results) == 1
    assert len(results[0]) > len(prompts[0])


def _parse_args():
    parser = argparse.ArgumentParser(description="Gemma4 TTNN instruct text generation demo")
    parser.add_argument(
        "--model-path", default=os.getenv("HF_MODEL") or os.getenv("GEMMA4_MODEL_PATH", DEFAULT_GEMMA4_MODEL)
    )
    parser.add_argument("--tokenizer-path", default=os.getenv("GEMMA4_TOKENIZER_PATH"))
    parser.add_argument("--prompt", action="append", default=None, help="Prompt text; may be passed more than once")
    parser.add_argument("--prompt-file", type=Path, help="Text file with one prompt per line")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample", action="store_true", help="Use top-k/top-p/temperature sampling instead of greedy")
    parser.add_argument("--base-completion", action="store_true", help="Bypass the tokenizer chat template")
    parser.add_argument("--mesh-rows", type=int, default=1)
    parser.add_argument("--mesh-cols", type=int, default=8)
    parser.add_argument("--trace-region-size", type=int, default=50_000_000)
    parser.add_argument("--page-block-size", type=int, default=64)
    parser.add_argument("--disable-decode-trace", action="store_true")
    parser.add_argument(
        "--allow-cpu-sampling-fallback",
        action="store_true",
        help="Debug-only fallback for meshes without device-side sampling",
    )
    return parser.parse_args()


def _collect_prompts(args):
    prompts = list(args.prompt or [])
    if args.prompt_file:
        prompts.extend(line.strip() for line in args.prompt_file.read_text().splitlines() if line.strip())
    if not prompts:
        prompts = ["Explain in two sentences why paged attention helps LLM serving."]
    return prompts


def main():
    args = _parse_args()
    _set_fabric_1d()
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(args.mesh_rows, args.mesh_cols),
        trace_region_size=args.trace_region_size,
    )
    try:
        outputs = run_generation(
            mesh_device=mesh_device,
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            prompts=_collect_prompts(args),
            max_new_tokens=args.max_new_tokens,
            num_layers=args.num_layers,
            max_seq_len=args.max_seq_len,
            page_params={
                "page_block_size": args.page_block_size,
                "page_max_num_blocks": args.max_seq_len // args.page_block_size,
            },
            enable_decode_trace=not args.disable_decode_trace,
            instruct=not args.base_completion,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
            greedy=not args.sample,
            allow_cpu_sampling_fallback=args.allow_cpu_sampling_fallback,
        )
        print("\n=== Gemma4 generated text ===")
        for idx, output in enumerate(outputs):
            print(f"\n--- Output {idx} ---")
            print(output)
    finally:
        ttnn.close_mesh_device(mesh_device)


@parametrize_mesh_with_fabric()
def test_demo(mesh_device, model_path):
    """Full model demo — runs on any multi-device mesh.

    Filter by mesh shape:
        pytest -k "1x2"   # N300 / TP=2
        pytest -k "1x8"   # T3K  / TP=8
    """
    prompts = ["Explain quantum computing in simple terms."]
    results = run_generation(
        mesh_device=mesh_device,
        model_path=model_path,
        prompts=prompts,
        max_new_tokens=128,
        max_seq_len=4 * 1024,
        enable_decode_trace=True,
        allow_cpu_sampling_fallback=mesh_device.get_num_devices() == 1,
    )
    assert len(results) == 1
    logger.info(f"Full model output: {results[0]}")


if __name__ == "__main__":
    main()
