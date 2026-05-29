# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tracy-oriented performance harness for ACE-Step **5 Hz LM** (TTNN causal stack).

Profiles the per-prompt LLM planning path (CoT metadata + audio-code generation) used by
``run_prompt_to_wav.py`` before the DiT denoise loop:

    ``LocalFiveHzLMHandler`` → ``AceStepFiveHzExperimentalTtnnCausalLM``
    (``ttnn_impl/five_hz_causal_lm_experimental.py`` → ``qwen_tt_transformers_lm``)

Run from the repository root (example):

    TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r -v -m pytest \\
        models/demos/ace_step_v1_5/perf/test_perf_llm_tracy.py::test_perf_ace_step_llm_tracy_profile \\
        -v -s

CSV / Tracy artifacts land under ``generated/profiler/reports/<timestamp>/`` — see
``docs/source/tt-metalium/tools/tracy_profiler.rst``.

**Important:** do **not** set ``ACE_STEP_USE_TRACE=1`` for this test. Device Tracy profiling
and TTNN LM trace capture are incompatible (same constraint as DiT / VAE / conditioning Tracy
harnesses).

Optional environment variables:

- ``ACE_STEP_CKPT_DIR``: checkpoint root (default: ``~/.cache/huggingface/hub/ACE-Step-1.5-checkpoints``).
- ``ACE_STEP_LLM_PERF_VARIANT``: LM folder name (default ``acestep-5Hz-lm-1.7B``).
- ``ACE_STEP_PERF_NO_DOWNLOAD`` / ``ACE_STEP_PERF_DOWNLOAD``: Hugging Face fetch control.
- ``ACE_STEP_LLM_PERF_MODE`` (default ``prefill_decode``):
  - ``prefill``: TTNN causal LM prefill only (one-shot prompt encoding).
  - ``prefill_decode``: prefill + ``ACE_STEP_LLM_PERF_DECODE_STEPS`` decode tokens (TTNN matmul/attn).
  - ``handler_cot``: handler Phase 1 CoT only (``infer_type=dit``).
  - ``handler_codes``: handler Phase 2 audio codes only (synthetic CoT metadata).
  - ``handler_full``: full ``generate_with_stop_condition`` (CoT + codes, production path).
- ``ACE_STEP_LLM_PERF_ITERS`` (default ``3``): timed perf-pass iterations.
- ``ACE_STEP_PERF_WARMUP`` (default ``1``): warmup iterations before the timed pass.
- ``ACE_STEP_LLM_PERF_PREFILL_SEQ`` (default ``128``): tokenized prompt length for ``prefill*`` modes.
- ``ACE_STEP_LLM_PERF_DECODE_STEPS`` (default ``50``): decode tokens per ``prefill_decode`` iteration.
- ``ACE_STEP_LLM_PERF_DURATION_SEC`` (default ``10``): target duration for handler modes (5 Hz codes).
- ``ACE_STEP_LLM_PERF_PROMPT`` / ``ACE_STEP_LLM_PERF_LYRICS``: handler caption / lyrics inputs.
- ``ACE_STEP_TRACY_EACH_LLM_ITER``: set to ``1`` for one Tracy signpost per perf iteration.
- ``ACE_STEP_TRACY_EACH_DECODE_STEP``: set to ``1`` for one signpost per decode token (``prefill_decode``).
- ``ACE_STEP_PROFILER_FLUSH_EVERY``: flush device profiler every N perf iterations (default ``1``).
- ``ACE_STEP_PERF_MAX_SECONDS``: optional wall-time budget on the timed perf pass.

If Tracy merge reports ``Device data missing``, run without ``-p`` or post-process with
``python tools/tracy/process_ops_logs.py --date``.
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
from models.demos.ace_step_v1_5.run_prompt_to_wav import _DEFAULT_CKPT_DIR, _ensure_variant
from models.demos.ace_step_v1_5.ttnn_impl.five_hz_causal_lm_experimental import AceStepFiveHzExperimentalTtnnCausalLM
from models.demos.ace_step_v1_5.ttnn_impl.math_perf_env import (
    ace_step_enable_tracy_profiler_env,
    ace_step_flush_device_profiler,
)


def _is_ci() -> bool:
    return os.environ.get("CI", "").lower() in ("true", "1", "yes")


def _perf_download_disabled() -> bool:
    if os.environ.get("ACE_STEP_PERF_NO_DOWNLOAD", "").lower() in ("1", "true", "yes"):
        return True
    dl = os.environ.get("ACE_STEP_PERF_DOWNLOAD", "").lower()
    return dl in ("0", "false", "no")


def _resolve_lm_dir(ckpt_dir: Path, variant: str) -> Path:
    return (ckpt_dir / variant).resolve()


def _lm_checkpoint_ready(lm_dir: Path) -> bool:
    if not (lm_dir / "config.json").is_file():
        return False
    if (lm_dir / "model.safetensors").is_file():
        return True
    return bool(list(lm_dir.glob("model-*.safetensors")))


def _ensure_lm_checkpoint(ckpt_dir: Path, variant: str) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    lm_dir = _resolve_lm_dir(ckpt_dir, variant)
    if _lm_checkpoint_ready(lm_dir):
        return lm_dir
    if _perf_download_disabled():
        pytest.skip(f"Missing 5 Hz LM under {lm_dir}. Unset ACE_STEP_PERF_NO_DOWNLOAD to fetch via Hugging Face.")
    pytest.importorskip("huggingface_hub")
    logger.info("ACE-Step LLM perf: fetching {} …", variant)
    try:
        _ensure_variant(variant, ckpt_dir)
    except Exception as exc:
        pytest.skip(f"Hugging Face download failed for {variant}: {exc}")
    if not _lm_checkpoint_ready(lm_dir):
        pytest.fail(f"Download finished but LM weights still missing under {lm_dir}")
    return lm_dir


def _tracy_signpost(label: str) -> None:
    if _is_ci():
        return
    try:
        from tracy import signpost  # type: ignore[import-untyped]
    except ImportError:
        return
    try:
        signpost(label)
    except Exception:
        pass


def _random_token_ids(*, vocab: int, seq_len: int, seed: int) -> torch.Tensor:
    torch.manual_seed(int(seed))
    hi = min(8192, max(8, int(vocab) - 1))
    lo = max(4, min(8, hi - 1))
    return torch.randint(lo, hi, (1, int(seq_len)), dtype=torch.long)


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_perf_ace_step_llm_tracy_profile(device):
    """Profile TTNN 5 Hz LM (prefill/decode or full handler) with Tracy signposts."""
    ace_step_enable_tracy_profiler_env()
    if os.environ.get("ACE_STEP_USE_TRACE", "").lower() in ("1", "true", "yes"):
        pytest.fail("ACE_STEP_USE_TRACE=1 is incompatible with Tracy device profiling for LLM perf.")

    ckpt_root = Path(os.environ.get("ACE_STEP_CKPT_DIR", str(_DEFAULT_CKPT_DIR))).expanduser().resolve()
    variant = os.environ.get("ACE_STEP_LLM_PERF_VARIANT", "acestep-5Hz-lm-1.7B")
    lm_dir = _ensure_lm_checkpoint(ckpt_root, variant)

    _run_llm_tracy_harness(device, lm_dir=lm_dir, variant_label=variant)


def _run_llm_tracy_harness(device: ttnn.Device, *, lm_dir: Path, variant_label: str) -> None:
    perf_mode = os.environ.get("ACE_STEP_LLM_PERF_MODE", "prefill_decode").strip().lower()
    valid_modes = ("prefill", "prefill_decode", "handler_cot", "handler_codes", "handler_full")
    if perf_mode not in valid_modes:
        pytest.fail(f"Unknown ACE_STEP_LLM_PERF_MODE={perf_mode!r}; use one of {valid_modes}.")

    iters = max(1, int(os.environ.get("ACE_STEP_LLM_PERF_ITERS", "3")))
    warmup = max(0, int(os.environ.get("ACE_STEP_PERF_WARMUP", "1")))
    seed = int(os.environ.get("ACE_STEP_PERF_SEED", "0"))
    trace_each_iter = os.environ.get("ACE_STEP_TRACY_EACH_LLM_ITER", "").lower() in ("1", "true", "yes")
    trace_each_decode = os.environ.get("ACE_STEP_TRACY_EACH_DECODE_STEP", "").lower() in ("1", "true", "yes")
    try:
        flush_every = int(os.environ.get("ACE_STEP_PROFILER_FLUSH_EVERY", "1"))
    except ValueError:
        flush_every = 1

    cfg = AutoConfig.from_pretrained(str(lm_dir), trust_remote_code=True)
    vocab = int(getattr(cfg, "vocab_size", 0) or 0)
    if vocab <= 8:
        pytest.skip("invalid vocab_size from LM config")

    prefill_seq = max(8, int(os.environ.get("ACE_STEP_LLM_PERF_PREFILL_SEQ", "128")))
    decode_steps = max(1, int(os.environ.get("ACE_STEP_LLM_PERF_DECODE_STEPS", "50")))
    duration_sec = float(os.environ.get("ACE_STEP_LLM_PERF_DURATION_SEC", "10"))
    prompt = os.environ.get("ACE_STEP_LLM_PERF_PROMPT", "lofi hip hop, warm vinyl, instrumental")
    lyrics = os.environ.get("ACE_STEP_LLM_PERF_LYRICS", "[Instrumental]")

    profiler = Profiler()
    profiler.clear()
    is_ci = _is_ci()

    if perf_mode.startswith("handler"):
        _run_handler_tracy_harness(
            device,
            lm_dir=lm_dir,
            variant_label=variant_label,
            perf_mode=perf_mode,
            iters=iters,
            warmup=warmup,
            seed=seed,
            duration_sec=duration_sec,
            prompt=prompt,
            lyrics=lyrics,
            trace_each_iter=trace_each_iter,
            flush_every=flush_every,
            profiler=profiler,
            is_ci=is_ci,
        )
        return

    _run_causal_lm_tracy_harness(
        device,
        lm_dir=lm_dir,
        variant_label=variant_label,
        perf_mode=perf_mode,
        iters=iters,
        warmup=warmup,
        seed=seed,
        vocab=vocab,
        prefill_seq=prefill_seq,
        decode_steps=decode_steps,
        trace_each_iter=trace_each_iter,
        trace_each_decode=trace_each_decode,
        flush_every=flush_every,
        profiler=profiler,
        is_ci=is_ci,
    )


def _run_causal_lm_tracy_harness(
    device: ttnn.Device,
    *,
    lm_dir: Path,
    variant_label: str,
    perf_mode: str,
    iters: int,
    warmup: int,
    seed: int,
    vocab: int,
    prefill_seq: int,
    decode_steps: int,
    trace_each_iter: bool,
    trace_each_decode: bool,
    flush_every: int,
    profiler: Profiler,
    is_ci: bool,
) -> None:
    max_seq_len = max(prefill_seq + decode_steps + 32, 1024)

    profiler.disable()
    profiler.start("ace_step_llm_init", force_enable=True)
    _tracy_signpost("LLM_INIT")

    try:
        causal_lm = AceStepFiveHzExperimentalTtnnCausalLM(
            str(lm_dir),
            device,
            max_seq_len=int(max_seq_len),
        )
    except RuntimeError as exc:
        pytest.skip(f"TTNN causal LM init skipped: {exc}")

    profiler.end("ace_step_llm_init", force_enable=True)
    ttnn.synchronize_device(device)
    ace_step_flush_device_profiler(device)

    prefill_ids = _random_token_ids(vocab=vocab, seq_len=prefill_seq, seed=seed)

    def _run_prefill_once() -> None:
        causal_lm.reset_decode_state()
        with torch.inference_mode():
            causal_lm.forward(input_ids=prefill_ids.clone(), past_key_values=None, use_cache=True)

    def _run_prefill_decode_once() -> None:
        causal_lm.reset_decode_state()
        with torch.inference_mode():
            out = causal_lm.forward(input_ids=prefill_ids.clone(), past_key_values=None, use_cache=True)
            past = out.past_key_values
            cur = prefill_ids[:, -1:]
            for step_idx in range(decode_steps):
                if trace_each_decode and not is_ci:
                    _tracy_signpost(f"LLM_DECODE_STEP_{step_idx}")
                nxt = torch.randint(4, min(8192, vocab - 1), (1, 1), dtype=torch.long)
                out = causal_lm.forward(input_ids=nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
                cur = nxt

    run_once = _run_prefill_once if perf_mode == "prefill" else _run_prefill_decode_once

    profiler.start("ace_step_llm_compile_pass", force_enable=True)
    _tracy_signpost("LLM_COMPILE_PASS")
    if perf_mode == "prefill":
        _tracy_signpost("LLM_PREFILL_COMPILE")
    else:
        _tracy_signpost("LLM_PREFILL_DECODE_COMPILE")
    run_once()
    profiler.end("ace_step_llm_compile_pass", force_enable=True)
    ttnn.synchronize_device(device)
    ace_step_flush_device_profiler(device)

    profiler.start("ace_step_llm_warmup", force_enable=True)
    _tracy_signpost("LLM_WARMUP")
    for _ in range(warmup):
        run_once()
    profiler.end("ace_step_llm_warmup", force_enable=True)
    ttnn.synchronize_device(device)
    ace_step_flush_device_profiler(device)

    profiler.enable()
    profiler.start("ace_step_llm_perf_pass")
    _tracy_signpost("LLM_PERF_PASS")

    for iter_idx in range(iters):
        if trace_each_iter and not is_ci:
            _tracy_signpost(f"LLM_ITER_{iter_idx}")
        run_once()
        if flush_every > 0 and (iter_idx + 1) % flush_every == 0:
            ace_step_flush_device_profiler(device)

    ttnn.synchronize_device(device)
    profiler.end("ace_step_llm_perf_pass")
    ace_step_flush_device_profiler(device)

    if hasattr(causal_lm, "release_trace"):
        causal_lm.release_trace()
    del causal_lm
    gc.collect()

    _log_llm_profiler_summary(
        profiler=profiler,
        perf_mode=perf_mode,
        variant_label=variant_label,
        iters=iters,
        warmup=warmup,
        extra=f"prefill_seq={prefill_seq} decode_steps={decode_steps if perf_mode != 'prefill' else 0}",
    )


def _run_handler_tracy_harness(
    device: ttnn.Device,
    *,
    lm_dir: Path,
    variant_label: str,
    perf_mode: str,
    iters: int,
    warmup: int,
    seed: int,
    duration_sec: float,
    prompt: str,
    lyrics: str,
    trace_each_iter: bool,
    flush_every: int,
    profiler: Profiler,
    is_ci: bool,
) -> None:
    from models.demos.ace_step_v1_5.ttnn_impl.five_hz_lm import LocalFiveHzLMHandler

    ckpt_dir = lm_dir.parent

    profiler.disable()
    profiler.start("ace_step_llm_init", force_enable=True)
    _tracy_signpost("LLM_HANDLER_INIT")

    handler = LocalFiveHzLMHandler()
    status, ok = handler.initialize(
        checkpoint_dir=str(ckpt_dir),
        lm_model_path=variant_label,
        backend="pt",
        device="cpu",
        ttnn_causal_device=device,
        use_ttnn_causal_lm=True,
        ttnn_lm_prefill_trace=False,
        ttnn_lm_decode_trace=False,
    )
    if not ok:
        pytest.fail(f"LocalFiveHzLMHandler.initialize failed: {status}")
    handler.set_ttnn_logits_device(device)

    profiler.end("ace_step_llm_init", force_enable=True)
    ttnn.synchronize_device(device)
    ace_step_flush_device_profiler(device)

    cot_metadata = {
        "bpm": 90,
        "keyscale": "C major",
        "timesignature": "4/4",
        "duration": int(duration_sec),
        "vocal_language": "en",
        "caption": prompt,
    }

    def _run_handler_once() -> None:
        if perf_mode == "handler_cot":
            _tracy_signpost("LLM_HANDLER_COT")
            formatted = handler.build_formatted_prompt(prompt, lyrics, generation_phase="cot")
            handler.generate_from_formatted_prompt(
                formatted,
                cfg={
                    "temperature": 0.85,
                    "cfg_scale": 1.0,
                    "generation_phase": "cot",
                    "caption": prompt,
                    "lyrics": lyrics,
                },
                use_constrained_decoding=True,
                stop_at_reasoning=True,
            )
            return

        if perf_mode == "handler_codes":
            _tracy_signpost("LLM_HANDLER_CODES")
            cot_text = handler._format_metadata_as_cot(cot_metadata)
            formatted = handler.build_formatted_prompt_with_cot(prompt, lyrics, cot_text)
            handler.generate_from_formatted_prompt(
                formatted,
                cfg={
                    "temperature": 0.85,
                    "cfg_scale": 1.0,
                    "target_duration": float(duration_sec),
                    "generation_phase": "codes",
                    "caption": prompt,
                    "lyrics": lyrics,
                    "cot_text": cot_text,
                },
                use_constrained_decoding=True,
                stop_at_reasoning=False,
            )
            return

        _tracy_signpost("LLM_HANDLER_FULL")
        result = handler.generate_with_stop_condition(
            caption=prompt,
            lyrics=lyrics,
            infer_type="llm_dit",
            temperature=0.85,
            cfg_scale=1.0,
            target_duration=float(duration_sec),
            user_metadata={"duration": int(duration_sec)},
            use_cot_metas=True,
            use_cot_caption=True,
            use_cot_language=True,
            use_constrained_decoding=True,
            constrained_decoding_debug=False,
            batch_size=1,
            seeds=[int(seed)],
        )
        if not result.get("success", False):
            raise RuntimeError(result.get("error", "LM generate_with_stop_condition failed"))

    profiler.start("ace_step_llm_compile_pass", force_enable=True)
    _tracy_signpost("LLM_HANDLER_COMPILE_PASS")
    _run_handler_once()
    profiler.end("ace_step_llm_compile_pass", force_enable=True)
    ttnn.synchronize_device(device)
    ace_step_flush_device_profiler(device)

    profiler.start("ace_step_llm_warmup", force_enable=True)
    _tracy_signpost("LLM_HANDLER_WARMUP")
    for _ in range(warmup):
        _run_handler_once()
    profiler.end("ace_step_llm_warmup", force_enable=True)
    ttnn.synchronize_device(device)
    ace_step_flush_device_profiler(device)

    profiler.enable()
    profiler.start("ace_step_llm_perf_pass")
    _tracy_signpost("LLM_HANDLER_PERF_PASS")

    for iter_idx in range(iters):
        if trace_each_iter and not is_ci:
            _tracy_signpost(f"LLM_HANDLER_ITER_{iter_idx}")
        _run_handler_once()
        if flush_every > 0 and (iter_idx + 1) % flush_every == 0:
            ace_step_flush_device_profiler(device)

    ttnn.synchronize_device(device)
    profiler.end("ace_step_llm_perf_pass")
    ace_step_flush_device_profiler(device)

    handler.unload()
    del handler
    gc.collect()

    _log_llm_profiler_summary(
        profiler=profiler,
        perf_mode=perf_mode,
        variant_label=variant_label,
        iters=iters,
        warmup=warmup,
        extra=f"duration_sec={duration_sec:g}",
    )


def _log_llm_profiler_summary(
    *,
    profiler: Profiler,
    perf_mode: str,
    variant_label: str,
    iters: int,
    warmup: int,
    extra: str,
) -> None:
    profiler.print()
    init_wall = profiler.get("ace_step_llm_init")
    compile_wall = profiler.get("ace_step_llm_compile_pass")
    warmup_wall = profiler.get("ace_step_llm_warmup")
    perf_wall = profiler.get("ace_step_llm_perf_pass")
    per_iter_ms = (perf_wall * 1000.0 / max(1, iters)) if iters else 0.0

    logger.info(
        "AceStep LLM Tracy harness (mode={}, variant={}, {}, iters={}): "
        "init={:.3f}s compile={:.3f}s warmup({}x)={:.3f}s perf_pass={:.3f}s (~{:.1f}ms/iter)",
        perf_mode,
        variant_label,
        extra,
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
        ), f"ace_step_llm_perf_pass {perf_wall}s exceeds ACE_STEP_PERF_MAX_SECONDS={budget}s"
