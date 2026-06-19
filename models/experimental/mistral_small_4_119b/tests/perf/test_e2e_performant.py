# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end wall-clock performance test for Mistral-Small-4-119B multimodal (vision + language) on TTNN.

Mirrors the vision + prefill + decode flow in ``demo_multimodal.py`` (unified orchestrator)
but isolates the steady-state timing windows (compile/load passes excluded from inference
numbers):

  1. Build TT model (untimed orchestrator construction; lazy load happens on first
     ``encode_image`` call).
  2. Vision:  1 cold ``encode_image`` (includes the one-time unified vision + text
     device load) + N warm replays (pure vision forward).
  3. Prefill: 1 compile pass + N program-cache-hot replays (timed). Note that
     ``prefill_multimodal`` is **not** trace-captured — ``TtMistral4TextModel``'s
     prefill path creates a fresh device input every call and reads the next token
     id back to host, neither of which is trace-friendly.
  4. Decode:  1 compile pass (eager) -> ``capture_decode_trace`` -> M trace replays
     (timed) over **2 command queues** — CQ1 uploads each step's token/position/RoPE
     while CQ0 replays the captured trace (``decode_next_token_2cq``).

Reports TTFT (vision_replay + prefill_replay), prefill tok/s,
**steady-state decode tok/s/user** (trace replay only), and
**end-to-end decode tok/s/user** (vision + prefill + decode compile/capture + all decode
steps), plus CSV/JSON via ``prep_perf_report`` + ``BenchmarkData``.

``num_command_queues`` is set to 2: the decode replay loop uses CQ1 for the
per-step host→device input uploads and CQ0 for the compute trace (event-fenced).

Run::

    pytest models/experimental/mistral_small_4_119b/tests/perf/test_e2e_performant.py -k L1V1
    pytest models/experimental/mistral_small_4_119b/tests/perf/test_e2e_performant.py -k L36V24
"""

from __future__ import annotations

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.mistral_small_4_119b.constants import (
    HF_MODEL_ID,
    MMP_SPATIAL_MERGE_SIZE,
    VISION_PATCH_SIZE,
    text_decoder_layer_state_dict_prefix,
    vision_layer_state_dict_prefix,
)
from models.experimental.mistral_small_4_119b.tt.mistral3_for_conditional_generation_unified import (
    TtMistral3ForConditionalGenerationUnified,
)
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.perf_utils import prep_perf_report
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered

# Target hardware: P150x8 (1×8 BlackHole P150 mesh).
MESH_SHAPE = (1, 8)


def _mesh_device_param():
    return pytest.param(MESH_SHAPE, id=f"mesh{MESH_SHAPE[0]}x{MESH_SHAPE[1]}")


def _state_dict_prefixes(n_text: int, n_vision: int) -> tuple:
    p = ["language_model.model.embed_tokens."]
    for i in range(n_text):
        p.append(text_decoder_layer_state_dict_prefix(i))
    p.append("language_model.model.norm.")
    p.append("language_model.lm_head.")
    p.append("vision_tower.patch_conv.")
    p.append("vision_tower.ln_pre.")
    for i in range(n_vision):
        p.append(vision_layer_state_dict_prefix(i))
    p.append("multi_modal_projector.")
    return tuple(p)


def _precompute_rope_table(rotary_cls, text_config, max_seq_len: int):
    rotary = rotary_cls(text_config).eval().to(torch.bfloat16)
    dummy = torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16)
    pos_ids = torch.arange(max_seq_len, dtype=torch.long).unsqueeze(0)
    return rotary(dummy, pos_ids)


def _build_synthetic_inputs(
    prompt_len: int,
    img_patches: int,
    image_token_id: int,
    vocab: int,
):
    """Deterministic synthetic pixel_values + input_ids — we only care about throughput here."""
    assert img_patches % MMP_SPATIAL_MERGE_SIZE == 0, "img_patches must be even for 2×2 spatial merge"
    num_image_tokens = (img_patches // MMP_SPATIAL_MERGE_SIZE) ** 2
    text_token_count = prompt_len - num_image_tokens
    assert text_token_count > 0, f"prompt_len={prompt_len} must exceed num_image_tokens={num_image_tokens}"

    side_px = img_patches * VISION_PATCH_SIZE
    gen = torch.Generator().manual_seed(0)
    pixel_values = (torch.rand(1, 3, side_px, side_px, generator=gen, dtype=torch.float32) * 2 - 1).to(torch.bfloat16)
    # Sample text-token ids from [image_token_id+1, vocab) so synthetic text never collides
    # with image_token_id — the orchestrator asserts num image-token slots == projector output.
    text_ids = torch.randint(image_token_id + 1, vocab, (text_token_count,), generator=gen, dtype=torch.long)
    img_ids = torch.full((num_image_tokens,), image_token_id, dtype=torch.long)
    input_ids = torch.cat([img_ids, text_ids]).unsqueeze(0)
    return pixel_values, input_ids, num_image_tokens, side_px


def _build_model(mesh_device, num_text_layers: int, num_vision_layers: int, max_seq_len: int):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    text_cfg = cfg.text_config
    for attr in ("attn_implementation", "_attn_implementation"):
        if hasattr(text_cfg, attr):
            setattr(text_cfg, attr, "eager")
    image_token_id = int(getattr(cfg, "image_token_index", 10))

    state_dict = load_hf_state_dict_filtered(HF_MODEL_ID, _state_dict_prefixes(num_text_layers, num_vision_layers))

    model = TtMistral3ForConditionalGenerationUnified(
        mesh_device=mesh_device,
        state_dict=state_dict,
        text_config=text_cfg,
        image_token_id=image_token_id,
        num_text_layers=num_text_layers,
        num_vision_layers=num_vision_layers,
        max_seq_len=max_seq_len,
        vision_dtype=ttnn.bfloat8_b,
    )
    del state_dict
    return model, cfg, text_cfg, image_token_id


def _run_mistral_perf(
    mesh_device,
    num_text_layers: int,
    num_vision_layers: int,
    prompt_len: int,
    decode_iters: int,
    prefill_iters: int,
    img_patches: int = 10,
):
    from transformers.models.mistral4.modeling_mistral4 import Mistral4RotaryEmbedding

    max_seq_len = prompt_len + decode_iters + 64

    t_build = time.time()
    model, cfg, text_cfg, image_token_id = _build_model(mesh_device, num_text_layers, num_vision_layers, max_seq_len)
    build_time = time.time() - t_build
    logger.info(
        f"Orchestrator built in {build_time:.1f}s "
        f"(text_layers={num_text_layers}, vision_layers={num_vision_layers}, max_seq_len={max_seq_len})"
    )

    vocab = int(getattr(text_cfg, "vocab_size", 131072))
    pixel_values, input_ids, num_image_tokens, side_px = _build_synthetic_inputs(
        prompt_len, img_patches, image_token_id, vocab
    )
    logger.info(
        f"Synthetic inputs: pixel_values{tuple(pixel_values.shape)}, "
        f"input_ids{tuple(input_ids.shape)} ({num_image_tokens} image-token slots)"
    )

    # ── Vision: cold (one-time unified vision+text device load + vision forward) ──
    t = time.time()
    img_embeds_host = model.encode_image(pixel_values)
    vision_compile_time = time.time() - t

    # Warm vision replays — load is done, just timing vision forward.
    t = time.time()
    for _ in range(prefill_iters):
        img_embeds_host = model.encode_image(pixel_values)
    vision_replay_time = (time.time() - t) / prefill_iters
    logger.info(
        f"Vision: compile+load={vision_compile_time*1000:.0f}ms, "
        f"replay={vision_replay_time*1000:.1f}ms (load ≈ {(vision_compile_time - vision_replay_time)*1000:.0f}ms)"
    )

    # RoPE tables (text_model is already loaded; load_text is a no-op).
    cos_full, sin_full = _precompute_rope_table(Mistral4RotaryEmbedding, text_cfg, prompt_len + decode_iters + 1)
    model.load_text()
    model.cache_rope_tables(cos_full, sin_full)

    # ── Prefill: cold compile, then program-cache-hot replays ────────────────────
    t = time.time()
    model.prefill_multimodal(img_embeds_host, input_ids)
    prefill_compile_time = time.time() - t

    # TTFT: time for one warm prefill replay (steady-state user-visible text-prefill cost).
    t = time.time()
    next_id = model.prefill_multimodal(img_embeds_host, input_ids)
    prefill_ttft_only_s = time.time() - t

    # Additional prefill replays for throughput averaging (includes the TTFT pass).
    prefill_replay_times = [prefill_ttft_only_s]
    for _ in range(prefill_iters - 1):
        t = time.time()
        next_id = model.prefill_multimodal(img_embeds_host, input_ids)
        prefill_replay_times.append(time.time() - t)
    prefill_replay_time = sum(prefill_replay_times) / len(prefill_replay_times)

    # User-visible TTFT = vision forward + multimodal prefill.
    ttft_s = vision_replay_time + prefill_replay_time
    logger.info(
        f"Prefill (seq={prompt_len}): compile={prefill_compile_time*1000:.0f}ms, "
        f"replay={prefill_replay_time*1000:.1f}ms "
        f"({prompt_len/prefill_replay_time:.1f} tok/s); "
        f"multimodal TTFT={ttft_s*1000:.1f}ms"
    )

    # ── Decode: eager compile -> capture trace -> trace replays ──────────────────
    cur = torch.tensor([[next_id]], dtype=torch.long)
    t = time.time()
    tok = model.decode_next_token(cur, prompt_len)
    decode_compile_time = time.time() - t
    cur = torch.tensor([[tok]], dtype=torch.long)

    t = time.time()
    model.capture_decode_trace()
    ttnn.synchronize_device(mesh_device)
    decode_capture_time = time.time() - t

    # 2-CQ trace replay: per step, CQ1 writes the token/position/RoPE inputs while
    # CQ0 replays the captured decode trace (event-fenced). begin_decode_2cq arms the
    # initial CQ0 event before the loop.
    model.begin_decode_2cq()
    t = time.time()
    for step in range(decode_iters):
        current_pos = prompt_len + 1 + step
        tok = model.decode_next_token_2cq(cur, current_pos)
        cur = torch.tensor([[tok]], dtype=torch.long)
    ttnn.synchronize_device(mesh_device)
    decode_total_time = time.time() - t
    decode_replay_time = decode_total_time / decode_iters
    steady_state_decode_tok_per_s = decode_iters / decode_total_time

    # End-to-end: vision + prefill compile/capture overhead + all decode steps.
    end_to_end_generation_time_s = ttft_s + decode_compile_time + decode_capture_time + decode_total_time
    end_to_end_throughput_tok_per_s = decode_iters / end_to_end_generation_time_s

    logger.info(
        f"Decode: compile={decode_compile_time*1000:.0f}ms, "
        f"capture={decode_capture_time*1000:.0f}ms, "
        f"replay={decode_replay_time*1000:.2f}ms "
        f"(steady-state {steady_state_decode_tok_per_s:.2f} tok/s/user, "
        f"end-to-end {end_to_end_throughput_tok_per_s:.2f} tok/s/user)"
    )

    return {
        "build_time_s": build_time,
        "vision_compile_time_s": vision_compile_time,
        "vision_replay_time_s": vision_replay_time,
        "prefill_compile_time_s": prefill_compile_time,
        "prefill_replay_time_s": prefill_replay_time,
        "prefill_throughput_tok_per_s": prompt_len / prefill_replay_time,
        "ttft_s": ttft_s,
        "ttft_ms": ttft_s * 1000,
        "decode_compile_time_s": decode_compile_time,
        "decode_capture_time_s": decode_capture_time,
        "decode_replay_time_s": decode_replay_time,
        "decode_total_time_s": decode_total_time,
        "steady_state_decode_throughput_tok_per_s": steady_state_decode_tok_per_s,
        "decode_throughput_tok_per_s_per_user": steady_state_decode_tok_per_s,
        "end_to_end_generation_time_s": end_to_end_generation_time_s,
        "end_to_end_throughput_tok_per_s": end_to_end_throughput_tok_per_s,
        "padded_prompt_len": prompt_len,
        "num_image_tokens": num_image_tokens,
        "image_side_px": side_px,
        "decode_iters": decode_iters,
        "num_text_layers": num_text_layers,
        "num_vision_layers": num_vision_layers,
    }


def _e2e_perf_device_params():
    return {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "trace_region_size": 100_000_000,
        # 2 queues: CQ1 for decode input uploads, CQ0 for the compute trace (see decode loop).
        "num_command_queues": 2,
    }


@pytest.mark.timeout(0)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    (
        "num_text_layers, num_vision_layers, prompt_len, decode_iters, prefill_iters, "
        "expected_compile_time, expected_inference_time"
    ),
    [
        (1, 1, 128, 32, 3, 600.0, 0.05),
        (36, 24, 128, 32, 3, 3600.0, 0.15),
        (36, 24, 4096, 32, 3, 3600.0, 0.15),
        (36, 24, 16384, 32, 3, 3600.0, 0.15),
        pytest.param(36, 24, 65536, 32, 3, 3600.0, 0.15, marks=pytest.mark.slow),
        pytest.param(36, 24, 131072, 32, 3, 3600.0, 0.15, marks=pytest.mark.slow),
        pytest.param(36, 24, 262144, 32, 3, 3600.0, 0.15, marks=pytest.mark.slow),
    ],
    ids=[
        "L1V1",
        "L36V24",
        "L36V24_4096",
        "L36V24_16384",
        "L36V24_65536",
        "L36V24_131072",
        "L36V24_262144",
    ],
)
@pytest.mark.parametrize("mesh_device", [_mesh_device_param()], indirect=True)
@pytest.mark.parametrize("device_params", [_e2e_perf_device_params()], indirect=True)
def test_mistral_small_4_119b_e2e_performant(
    mesh_device,
    num_text_layers,
    num_vision_layers,
    prompt_len,
    decode_iters,
    prefill_iters,
    expected_compile_time,
    expected_inference_time,
):
    results = _run_mistral_perf(
        mesh_device,
        num_text_layers=num_text_layers,
        num_vision_layers=num_vision_layers,
        prompt_len=prompt_len,
        decode_iters=decode_iters,
        prefill_iters=prefill_iters,
    )

    batch_size = 1
    model_name = f"mistral_small_4_119b_L{num_text_layers}V{num_vision_layers}"
    comments = f"prefill{results['padded_prompt_len']}_decode{decode_iters}"

    # Treat decode trace replay as the steady-state inference time, and everything else
    # (build + vision + both compile/capture passes + warm prefill/vision replays) as compile/warmup.
    inference_time = results["decode_replay_time_s"]
    inference_and_compile_time = (
        results["build_time_s"]
        + results["vision_compile_time_s"]
        + results["vision_replay_time_s"] * prefill_iters
        + results["prefill_compile_time_s"]
        + results["prefill_replay_time_s"] * prefill_iters
        + results["decode_compile_time_s"]
        + results["decode_capture_time_s"]
        + inference_time * decode_iters
    )

    prep_perf_report(
        model_name=model_name,
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
    )

    profiler = BenchmarkProfiler()
    profiler.start("run")
    profiler.end("run")
    step = "mistral_small_4_119b_e2e"
    profiler.start(step)
    profiler.end(step)
    benchmark_data = BenchmarkData()
    for k, v in results.items():
        benchmark_data.add_measurement(profiler, 0, step, k, float(v))
    benchmark_data.save_partial_run_json(
        profiler,
        run_type="end_to_end_perf",
        ml_model_name=model_name,
        batch_size=batch_size,
    )

    logger.info(
        f"{model_name}: TTFT={results['ttft_ms']:.1f}ms, "
        f"steady-state decode={results['steady_state_decode_throughput_tok_per_s']:.2f} tok/s/user, "
        f"end-to-end decode={results['end_to_end_throughput_tok_per_s']:.2f} tok/s/user "
        f"(prefill {results['prefill_throughput_tok_per_s']:.1f} tok/s on {results['padded_prompt_len']}-tok prompt, "
        f"{results['num_image_tokens']} of which are image tokens)"
    )
