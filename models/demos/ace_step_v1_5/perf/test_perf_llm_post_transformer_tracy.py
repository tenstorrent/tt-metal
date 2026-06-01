# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tracy harness for ACE-Step 5 Hz LM **post-transformer** work.

Profiles device ops that run *after* the transformer stack on each token:

- **prefill_last_token** (default): final RMSNorm + sharded ``LMHead`` on the prefill
  last-token hidden tile (CoT / metadata phase pattern).
- **decode_step**: decode-path final norm + ``LMHead`` (+ untilize) on a decode hidden tile.
- **cfg_combine**: TTNN CFG linear combine on the narrow valid-audio logits slice ``[B,K]``.
- **penalty_sample**: fused repetition penalty + top-k/top-p/temperature + argmax sample.
- **decode_post_chain**: decode norm/LMHead → host read → CFG → penalty/sample (one synthetic step).

Run from repo root (example):

    TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r -v -m pytest \\
        models/demos/ace_step_v1_5/perf/test_perf_llm_post_transformer_tracy.py \\
        -v -s

Do **not** set ``ACE_STEP_USE_TRACE=1`` (Tracy profiling and TTNN trace capture conflict).

Environment knobs (shared with ``test_perf_llm_tracy.py`` where noted):

- ``ACE_STEP_LLM_HEAD_PERF_MODE``: see modes above (default ``prefill_last_token``).
- ``ACE_STEP_LLM_PERF_ITERS`` / ``ACE_STEP_PERF_WARMUP``: timed pass / warmup counts.
- ``ACE_STEP_LLM_PERF_PREFILL_SEQ`` (default ``128``): prompt length for hidden-tile capture.
- ``ACE_STEP_LLM_HEAD_PERF_CFG_K`` (default ``150``): valid-audio slice width for CFG mode.
- ``ACE_STEP_LM_NARROW_AUDIO_VOCAB``: narrow ``LMHead`` band (default ``1`` via Tracy setdefault).
- ``ACE_STEP_LM_LM_HEAD_SHARDED_NORM``: sharded prefill final RMSNorm before ``LMHead`` (default ``1``).
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.utility_functions import Profiler
from models.demos.ace_step_v1_5.perf.test_perf_llm_tracy import (
    _ensure_lm_checkpoint,
    _random_token_ids,
    _tracy_signpost,
)
from models.demos.ace_step_v1_5.run_prompt_to_wav import _DEFAULT_CKPT_DIR
from models.demos.ace_step_v1_5.ttnn_impl.five_hz_causal_lm_experimental import AceStepFiveHzExperimentalTtnnCausalLM
from models.demos.ace_step_v1_5.ttnn_impl.math_perf_env import (
    ace_step_enable_tracy_profiler_env,
    ace_step_flush_device_profiler,
)


def _is_ci() -> bool:
    return os.environ.get("CI", "").lower() in ("true", "1", "yes")


def _ace_step_enable_llm_head_tracy_env() -> None:
    os.environ.setdefault("ACE_STEP_LM_NARROW_AUDIO_VOCAB", "1")


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_perf_ace_step_llm_post_transformer_tracy_profile(device):
    """Profile LM final norm, LMHead, CFG, and sampling postprocess with Tracy."""
    ace_step_enable_tracy_profiler_env()
    if os.environ.get("ACE_STEP_USE_TRACE", "").lower() in ("1", "true", "yes"):
        pytest.fail("ACE_STEP_USE_TRACE=1 is incompatible with Tracy device profiling.")

    ckpt_root = Path(os.environ.get("ACE_STEP_CKPT_DIR", str(_DEFAULT_CKPT_DIR))).expanduser().resolve()
    variant = os.environ.get("ACE_STEP_LLM_PERF_VARIANT", "acestep-5Hz-lm-1.7B")
    lm_dir = _ensure_lm_checkpoint(ckpt_root, variant)

    _run_post_transformer_tracy_harness(device, lm_dir=lm_dir, variant_label=variant)


def _run_post_transformer_tracy_harness(device: ttnn.Device, *, lm_dir: Path, variant_label: str) -> None:
    _ace_step_enable_llm_head_tracy_env()

    perf_mode = os.environ.get("ACE_STEP_LLM_HEAD_PERF_MODE", "prefill_last_token").strip().lower()
    valid_modes = ("prefill_last_token", "decode_step", "cfg_combine", "penalty_sample", "decode_post_chain")
    if perf_mode not in valid_modes:
        pytest.fail(f"Unknown ACE_STEP_LLM_HEAD_PERF_MODE={perf_mode!r}; use one of {valid_modes}.")

    iters = max(1, int(os.environ.get("ACE_STEP_LLM_PERF_ITERS", "20")))
    warmup = max(0, int(os.environ.get("ACE_STEP_PERF_WARMUP", "2")))
    seed = int(os.environ.get("ACE_STEP_PERF_SEED", "0"))
    prefill_seq = max(8, int(os.environ.get("ACE_STEP_LLM_PERF_PREFILL_SEQ", "128")))
    cfg_k = max(8, int(os.environ.get("ACE_STEP_LLM_HEAD_PERF_CFG_K", "150")))
    cfg_scale = float(os.environ.get("ACE_STEP_LLM_HEAD_PERF_CFG_SCALE", "3.0"))
    trace_each_iter = os.environ.get("ACE_STEP_TRACY_EACH_LLM_ITER", "").lower() in ("1", "true", "yes")
    is_ci = _is_ci()

    cfg_obj = AutoConfig.from_pretrained(str(lm_dir), trust_remote_code=True)
    vocab = int(getattr(cfg_obj, "vocab_size", 0) or 0)
    if vocab <= 8:
        pytest.skip("invalid vocab_size from LM config")

    max_seq_len = max(prefill_seq + 64, 1024)
    prefill_ids = _random_token_ids(vocab=vocab, seq_len=prefill_seq, seed=seed)
    decode_token = torch.randint(4, min(8192, vocab - 1), (1, 1), dtype=torch.long)

    profiler = Profiler()
    profiler.clear()

    profiler.disable()
    profiler.start("ace_step_llm_head_init", force_enable=True)
    _tracy_signpost("LLM_HEAD_INIT")

    try:
        causal_lm = AceStepFiveHzExperimentalTtnnCausalLM(
            str(lm_dir),
            device,
            max_seq_len=int(max_seq_len),
        )
    except RuntimeError as exc:
        pytest.skip(f"TTNN causal LM init skipped: {exc}")

    qwen = causal_lm.qwen
    narrow_indices = torch.arange(4, 4 + cfg_k, dtype=torch.long)

    profiler.end("ace_step_llm_head_init", force_enable=True)
    ttnn.synchronize_device(device)
    ace_step_flush_device_profiler(device)

    # Compile transformer + post-transformer paths and capture representative tensors.
    profiler.start("ace_step_llm_head_compile_pass", force_enable=True)
    _tracy_signpost("LLM_HEAD_COMPILE_PASS")

    with torch.inference_mode():
        causal_lm.reset_decode_state()
        causal_lm.forward(input_ids=prefill_ids.clone(), past_key_values=None, use_cache=True)

        prefill_hidden = qwen.capture_prefill_hidden_tile(prefill_ids)
        _ = qwen.forward_post_transformer_prefill(prefill_hidden)

        decode_pos = int(prefill_seq)
        decode_hidden = qwen.capture_decode_hidden_tile(decode_token, decode_pos)
        _ = qwen.forward_post_transformer_decode(decode_hidden)

        if perf_mode in ("cfg_combine", "decode_post_chain"):
            from models.demos.ace_step_v1_5.ttnn_impl.lm_logits_ttnn import cfg_linear_combination_bf16

            cond = torch.randn(1, cfg_k, dtype=torch.float32)
            uncond = torch.randn(1, cfg_k, dtype=torch.float32)
            _ = cfg_linear_combination_bf16(cond, uncond, cfg_scale, device=device, use_trace=False)

        if perf_mode in ("penalty_sample", "decode_post_chain"):
            from models.demos.ace_step_v1_5.ttnn_impl.lm_postprocess_tt_transformers import apply_penalty_filter_sample

            scores = torch.randn(1, cfg_k, dtype=torch.float32)
            hist = torch.randint(4, 4 + cfg_k, (1, 32), dtype=torch.long)
            _ = apply_penalty_filter_sample(
                scores,
                hist,
                repetition_penalty=1.05,
                top_k=50,
                top_p=0.95,
                temperature=0.9,
                seed=seed,
                device=device,
            )

    cached_prefill_hidden = prefill_hidden
    cached_decode_hidden = decode_hidden

    profiler.end("ace_step_llm_head_compile_pass", force_enable=True)
    ttnn.synchronize_device(device)
    ace_step_flush_device_profiler(device)

    def _run_once() -> None:
        if perf_mode == "prefill_last_token":
            causal_lm.set_narrow_audio_vocab_indices(None)
            _ = qwen.forward_post_transformer_prefill(cached_prefill_hidden)
            return

        if perf_mode == "decode_step":
            causal_lm.set_narrow_audio_vocab_indices(narrow_indices)
            _ = qwen.forward_post_transformer_decode(cached_decode_hidden)
            return

        if perf_mode == "cfg_combine":
            from models.demos.ace_step_v1_5.ttnn_impl.lm_logits_ttnn import cfg_linear_combination_bf16

            cond = torch.randn(1, cfg_k, dtype=torch.float32)
            uncond = torch.randn(1, cfg_k, dtype=torch.float32)
            _ = cfg_linear_combination_bf16(cond, uncond, cfg_scale, device=device, use_trace=True)
            return

        if perf_mode == "penalty_sample":
            from models.demos.ace_step_v1_5.ttnn_impl.lm_postprocess_tt_transformers import apply_penalty_filter_sample

            scores = torch.randn(1, cfg_k, dtype=torch.float32)
            hist = torch.randint(4, 4 + cfg_k, (1, 32), dtype=torch.long)
            _ = apply_penalty_filter_sample(
                scores,
                hist,
                repetition_penalty=1.05,
                top_k=50,
                top_p=0.95,
                temperature=0.9,
                seed=seed,
                device=device,
            )
            return

        # decode_post_chain: norm/LMHead → host read → CFG → penalty/sample (post-transformer only).
        causal_lm.set_narrow_audio_vocab_indices(narrow_indices)
        logits_tt = qwen.forward_post_transformer_decode(cached_decode_hidden)
        logits = ttnn.to_torch(logits_tt).float()
        if logits.dim() == 4:
            logits = logits.reshape(1, -1)
        band_w = min(cfg_k, int(logits.shape[-1]))
        cond = logits[..., :band_w].contiguous()
        uncond = cond * 0.95

        from models.demos.ace_step_v1_5.ttnn_impl.lm_logits_ttnn import cfg_linear_combination_bf16
        from models.demos.ace_step_v1_5.ttnn_impl.lm_postprocess_tt_transformers import apply_penalty_filter_sample

        cfg_logits = cfg_linear_combination_bf16(cond, uncond, cfg_scale, device=device, use_trace=True)
        hist = torch.randint(4, 4 + cfg_k, (1, 32), dtype=torch.long)
        _ = apply_penalty_filter_sample(
            cfg_logits,
            hist,
            repetition_penalty=1.05,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            seed=seed,
            device=device,
        )

    profiler.start("ace_step_llm_head_warmup", force_enable=True)
    _tracy_signpost("LLM_HEAD_WARMUP")
    for _ in range(warmup):
        _run_once()
    profiler.end("ace_step_llm_head_warmup", force_enable=True)
    ttnn.synchronize_device(device)
    ace_step_flush_device_profiler(device)

    profiler.enable()
    profiler.start("ace_step_llm_head_perf_pass")
    _tracy_signpost("LLM_HEAD_PERF_PASS")

    for iter_idx in range(iters):
        if trace_each_iter and not is_ci:
            _tracy_signpost(f"LLM_HEAD_ITER_{iter_idx}")
        _run_once()
        ace_step_flush_device_profiler(device)

    ttnn.synchronize_device(device)
    profiler.end("ace_step_llm_head_perf_pass")
    ace_step_flush_device_profiler(device)

    if hasattr(causal_lm, "release_trace"):
        causal_lm.release_trace()
    del causal_lm
    gc.collect()

    profiler.print()
    init_wall = profiler.get("ace_step_llm_head_init")
    compile_wall = profiler.get("ace_step_llm_head_compile_pass")
    warmup_wall = profiler.get("ace_step_llm_head_warmup")
    perf_wall = profiler.get("ace_step_llm_head_perf_pass")
    per_iter_ms = (perf_wall * 1000.0 / max(1, iters)) if iters else 0.0

    logger.info(
        "AceStep LLM post-transformer Tracy (mode={}, variant={}, prefill_seq={}, cfg_k={}, iters={}): "
        "init={:.3f}s compile={:.3f}s warmup({}x)={:.3f}s perf_pass={:.3f}s (~{:.1f}ms/iter)",
        perf_mode,
        variant_label,
        prefill_seq,
        cfg_k,
        iters,
        init_wall,
        compile_wall,
        warmup,
        warmup_wall,
        perf_wall,
        per_iter_ms,
    )

    budget = os.environ.get("ACE_STEP_PERF_MAX_SECONDS")
    if budget:
        assert perf_wall <= float(
            budget
        ), f"ace_step_llm_head_perf_pass {perf_wall}s exceeds ACE_STEP_PERF_MAX_SECONDS={budget}s"
