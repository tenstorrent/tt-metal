# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tracy-oriented performance harness for ACE-Step **one-time conditioning** (Priority 2).

Profiles the per-prompt path amortized on long sessions (runs once before the DiT denoise loop):

    ``AceStepQwen3Encoder`` (``ttnn_impl/qwen3_embedding_ace_step.py``) — 28-layer caption encoder
    → ``TtAceStepInstrumentalConditionEncoder`` (``ttnn_impl/condition_encoder.py``) — text projector
    + cached lyric/timbre constants + concat

This matches ``AceStepE2EModel.encode_text`` + ``TtAceStepInstrumentalConditionEncoder.forward`` in
``ttnn_impl/e2e_model_tt.py`` (not repeated per Euler step).

Run from the repository root (example):

    TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r -v -m pytest \\
        models/demos/ace_step_v1_5/perf/test_perf_conditioning_tracy.py::test_perf_ace_step_conditioning_tracy_profile \\
        -v -s

CSV / Tracy artifacts land under ``generated/profiler/reports/<timestamp>/`` — see
``docs/source/tt-metalium/tools/tracy_profiler.rst``.

Production defaults (``math_perf_env``): LoFi + ``bfloat8_b`` on Qwen3 (``ace_step_qwen3_optimizations``),
condition encoder, and text projector; L1 TILE activations where supported — same path as PCC (no env toggle).

**Important:** do **not** set ``ACE_STEP_USE_TRACE=1`` for this test (same constraint as the DiT Tracy
harness: device profiler flush + trace capture are incompatible).

Optional environment variables:

- ``ACE_STEP_CKPT_DIR`` / ``ACE_STEP_PERF_VARIANT``: DiT checkpoint (condition encoder weights live in
  the DiT ``model.safetensors``) + bundle folder (default variant: ``acestep-v15-turbo``).
- ``ACE_STEP_PERF_NO_DOWNLOAD`` / ``ACE_STEP_PERF_DOWNLOAD``: Hugging Face fetch control (same as other
  ACE perf tests). Qwen3-Embedding-0.6B is fetched alongside the DiT bundle when missing.
- ``ACE_STEP_COND_PERF_MODE`` (default ``full``):
  - ``full``: Qwen3 ``forward`` + condition ``forward`` (production one-shot path).
  - ``qwen``: Qwen3 text encoder only.
  - ``condition``: condition encoder only (synthetic ``[1,1,S,1024]`` Qwen hidden states).
- ``ACE_STEP_COND_PERF_ITERS`` (default ``8``): timed perf-pass iterations (simulates multiple prompts
  in a session; each iteration is an independent encode+condition pair).
- ``ACE_STEP_PERF_WARMUP`` (default ``1``): extra warmup iterations before the timed pass.
- ``ACE_STEP_COND_PERF_PROMPT`` (default ``"lofi hip hop, warm vinyl"``): tokenizer input for Qwen3.
- ``ACE_STEP_COND_PERF_TEXT_SEQ`` (default ``256``): tokenizer ``max_length`` (Qwen ``max_seq_len``).
- ``ACE_STEP_COND_PERF_VALID_TOKENS`` (default ``50``): valid token count for ``condition``-only mode.
- ``ACE_STEP_TRACY_EACH_COND_ITER``: set to ``1`` for one Tracy signpost per perf iteration.
- ``ACE_STEP_PROFILER_FLUSH_EVERY``: flush device profiler every N perf iterations (default ``1``).
- ``ACE_STEP_PERF_MAX_SECONDS``: optional wall-time budget on the timed perf pass.

If Tracy's merge step fails with ``Device data missing``, run without ``-p`` for host-only timelines, or
post-process with ``python tools/tracy/process_ops_logs.py --date``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import Profiler
from models.demos.ace_step_v1_5.run_prompt_to_wav import _DEFAULT_CKPT_DIR, _ensure_variant
from models.demos.ace_step_v1_5.ttnn_impl.condition_encoder import TtAceStepInstrumentalConditionEncoder
from models.demos.ace_step_v1_5.ttnn_impl.qwen3_embedding_ace_step import AceStepQwen3Encoder
from models.demos.ace_step_v1_5.ttnn_impl.qwen_prefill_l1 import ace_step_qwen_prefill_l1_op_context


def _is_ci() -> bool:
    return os.environ.get("CI", "").lower() in ("true", "1", "yes")


def _perf_download_disabled() -> bool:
    if os.environ.get("ACE_STEP_PERF_NO_DOWNLOAD", "").lower() in ("1", "true", "yes"):
        return True
    dl = os.environ.get("ACE_STEP_PERF_DOWNLOAD", "").lower()
    return dl in ("0", "false", "no")


def _resolve_dit_checkpoint(ckpt_dir: Path, variant: str) -> Path:
    model_dir = ckpt_dir / variant
    dit_st = model_dir / "model.safetensors"
    if not dit_st.is_file():
        shards = sorted(model_dir.glob("model-*.safetensors"))
        if shards:
            dit_st = shards[0]
    return dit_st


def _resolve_qwen_dir(ckpt_dir: Path) -> Path:
    return ckpt_dir / "Qwen3-Embedding-0.6B"


def _checkpoints_ready(ckpt_dir: Path, variant: str) -> bool:
    dit_ok = _resolve_dit_checkpoint(ckpt_dir, variant).is_file()
    qwen_dir = _resolve_qwen_dir(ckpt_dir)
    qwen_st = qwen_dir / "model.safetensors"
    if not qwen_st.is_file() and not any(qwen_dir.glob("model-*.safetensors")):
        return False
    return dit_ok and (qwen_dir / "config.json").is_file()


def _ensure_checkpoints(ckpt_dir: Path, variant: str) -> tuple[Path, Path]:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pytest.importorskip("huggingface_hub")
    for bundle in (variant, "Qwen3-Embedding-0.6B"):
        logger.info("ACE-Step conditioning perf: fetching {} …", bundle)
        try:
            _ensure_variant(bundle, ckpt_dir)
        except Exception as exc:
            pytest.skip(f"Hugging Face download failed for {bundle}: {exc}")
    dit_st = _resolve_dit_checkpoint(ckpt_dir, variant)
    qwen_dir = _resolve_qwen_dir(ckpt_dir)
    if not dit_st.is_file():
        pytest.fail(f"Download finished but DiT weights still missing under {ckpt_dir / variant}")
    if not (qwen_dir / "config.json").is_file():
        pytest.fail(f"Download finished but Qwen3 config still missing under {qwen_dir}")
    return dit_st, qwen_dir


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


def _ace_step_flush_device_profiler(device) -> None:
    if os.environ.get("TTNN_OP_PROFILER") != "1" and os.environ.get("TT_METAL_DEVICE_PROFILER") != "1":
        return
    if os.environ.get("ACE_STEP_USE_TRACE", "").lower() in ("1", "true", "yes"):
        return
    try:
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)
    except Exception:
        pass


def _tokenize_prompt(*, text_model_dir: Path, prompt: str, max_length: int) -> tuple[np.ndarray, np.ndarray]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(text_model_dir))
    tokens = tok(prompt, padding="max_length", truncation=True, max_length=int(max_length), return_tensors="pt")
    input_ids_np = np.asarray(tokens["input_ids"], dtype=np.uint32).reshape(1, -1)
    attn_mask_np = np.asarray(tokens["attention_mask"], dtype=np.float32).reshape(1, -1)
    return input_ids_np, attn_mask_np


def _synthetic_text_hidden(
    *,
    device: ttnn.Device,
    seq_len: int,
    hidden: int = 1024,
    seed: int = 0,
) -> ttnn.Tensor:
    torch.manual_seed(int(seed))
    host = torch.randn(1, 1, int(seq_len), int(hidden), dtype=torch.bfloat16)
    return ttnn.from_torch(
        host.contiguous(),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_perf_ace_step_conditioning_tracy_profile(device):
    """Profile Qwen3 caption encoder + instrumental condition encoder with Tracy signposts."""
    ckpt_root = Path(os.environ.get("ACE_STEP_CKPT_DIR", str(_DEFAULT_CKPT_DIR))).expanduser().resolve()
    variant = os.environ.get("ACE_STEP_PERF_VARIANT", "acestep-v15-turbo")

    if not _checkpoints_ready(ckpt_root, variant):
        if _perf_download_disabled():
            pytest.skip(
                f"Missing ACE-Step checkpoints under {ckpt_root} (need {variant} + Qwen3-Embedding-0.6B). "
                "Unset ACE_STEP_PERF_NO_DOWNLOAD to fetch."
            )
        dit_st, qwen_dir = _ensure_checkpoints(ckpt_root, variant)
    else:
        dit_st = _resolve_dit_checkpoint(ckpt_root, variant)
        qwen_dir = _resolve_qwen_dir(ckpt_root)

    _run_conditioning_tracy_harness(
        device,
        dit_checkpoint=dit_st,
        qwen_model_dir=qwen_dir,
        variant_label=variant,
    )


def _run_conditioning_tracy_harness(
    device: ttnn.Device,
    *,
    dit_checkpoint: Path,
    qwen_model_dir: Path,
    variant_label: str,
) -> None:
    perf_mode = os.environ.get("ACE_STEP_COND_PERF_MODE", "full").strip().lower()
    if perf_mode not in ("full", "qwen", "qwen_ttt", "condition"):
        pytest.fail(f"Unknown ACE_STEP_COND_PERF_MODE={perf_mode!r}; use 'full', 'qwen', 'qwen_ttt', or 'condition'.")

    iters = max(1, int(os.environ.get("ACE_STEP_COND_PERF_ITERS", "8")))
    warmup = max(0, int(os.environ.get("ACE_STEP_PERF_WARMUP", "1")))
    text_seq = int(os.environ.get("ACE_STEP_COND_PERF_TEXT_SEQ", "256"))
    prompt = os.environ.get("ACE_STEP_COND_PERF_PROMPT", "lofi hip hop, warm vinyl")
    seed = int(os.environ.get("ACE_STEP_PERF_SEED", "0"))
    trace_each_iter = os.environ.get("ACE_STEP_TRACY_EACH_COND_ITER", "").lower() in ("1", "true", "yes")
    try:
        flush_every = int(os.environ.get("ACE_STEP_PROFILER_FLUSH_EVERY", "1"))
    except ValueError:
        flush_every = 1

    run_qwen = perf_mode in ("full", "qwen", "qwen_ttt")
    run_cond = perf_mode in ("full", "condition")
    qwen_ttt = perf_mode == "qwen_ttt"  # profile the tt_transformers prefill (eager) vs eager custom

    profiler = Profiler()
    profiler.clear()
    is_ci = _is_ci()

    input_ids_np: np.ndarray | None = None
    attn_mask_np: np.ndarray | None = None
    if run_qwen:
        input_ids_np, attn_mask_np = _tokenize_prompt(
            text_model_dir=qwen_model_dir,
            prompt=prompt,
            max_length=text_seq,
        )
    elif run_cond:
        # Condition-only perf: fixed valid-token count (slice/concat path still exercised).
        valid = min(max(1, int(os.environ.get("ACE_STEP_COND_PERF_VALID_TOKENS", "50"))), text_seq)
        attn_mask_np = np.zeros((1, text_seq), dtype=np.float32)
        attn_mask_np[0, :valid] = 1.0

    qwen_st = qwen_model_dir / "model.safetensors"
    if not qwen_st.is_file():
        shards = sorted(qwen_model_dir.glob("model-*.safetensors"))
        if shards:
            qwen_st = shards[0]

    # --- INIT -------------------------------------------------------------------------
    profiler.disable()
    profiler.start("ace_step_cond_init", force_enable=True)
    _tracy_signpost("CONDITIONING_INIT")

    qwen_enc: AceStepQwen3Encoder | None = None
    cond_enc: TtAceStepInstrumentalConditionEncoder | None = None

    if run_qwen:
        qwen_enc = AceStepQwen3Encoder(
            device=device,
            hf_model_dir=str(qwen_model_dir),
            qwen_safetensors_path=str(qwen_st),
            max_seq_len=text_seq,
        )
    if run_cond:
        cond_enc = TtAceStepInstrumentalConditionEncoder(
            device=device,
            checkpoint_safetensors_path=str(dit_checkpoint),
            dtype=ttnn.bfloat16,
        )

    profiler.end("ace_step_cond_init", force_enable=True)
    ttnn.synchronize_device(device)
    _ace_step_flush_device_profiler(device)

    assert qwen_enc is not None or cond_enc is not None

    def _run_qwen_once() -> ttnn.Tensor:
        assert qwen_enc is not None and input_ids_np is not None
        if qwen_ttt:
            return qwen_enc.prefill_eager(input_ids_np)
        return qwen_enc.forward(input_ids_np, attn_mask_np)

    def _run_condition_once(text_hs_tt: ttnn.Tensor) -> ttnn.Tensor:
        assert cond_enc is not None and attn_mask_np is not None
        with ace_step_qwen_prefill_l1_op_context():
            enc_hs_tt, _enc_mask_np, _null_emb = cond_enc.forward(text_hs_tt, attn_mask_np)
        return enc_hs_tt

    def _run_full_once() -> None:
        text_hs = _run_qwen_once() if run_qwen else _synthetic_text_hidden(device=device, seq_len=text_seq, seed=seed)
        if run_cond:
            enc_hs = _run_condition_once(text_hs)
            try:
                ttnn.deallocate(enc_hs)
            except Exception:
                pass
        try:
            ttnn.deallocate(text_hs)
        except Exception:
            pass

    def _run_qwen_only_once() -> None:
        text_hs = _run_qwen_once()
        try:
            ttnn.deallocate(text_hs)
        except Exception:
            pass

    def _run_cond_only_once() -> None:
        text_hs = _synthetic_text_hidden(device=device, seq_len=text_seq, seed=seed)
        enc_hs = _run_condition_once(text_hs)
        try:
            ttnn.deallocate(enc_hs)
            ttnn.deallocate(text_hs)
        except Exception:
            pass

    if perf_mode == "full":
        run_once = _run_full_once
    elif perf_mode in ("qwen", "qwen_ttt"):
        run_once = _run_qwen_only_once
    else:
        run_once = _run_cond_only_once

    # --- COMPILE PASS -----------------------------------------------------------------
    profiler.start("ace_step_cond_compile_pass", force_enable=True)
    _tracy_signpost("CONDITIONING_COMPILE_PASS")
    if run_qwen:
        _tracy_signpost("QWEN_COMPILE_PASS")
    if run_cond:
        _tracy_signpost("COND_COMPILE_PASS")
    run_once()
    profiler.end("ace_step_cond_compile_pass", force_enable=True)
    ttnn.synchronize_device(device)
    _ace_step_flush_device_profiler(device)

    # --- WARMUP -----------------------------------------------------------------------
    profiler.start("ace_step_cond_warmup", force_enable=True)
    _tracy_signpost("CONDITIONING_WARMUP")
    for _ in range(warmup):
        run_once()
    profiler.end("ace_step_cond_warmup", force_enable=True)
    ttnn.synchronize_device(device)
    _ace_step_flush_device_profiler(device)

    # --- PERF PASS --------------------------------------------------------------------
    profiler.enable()
    profiler.start("ace_step_cond_perf_pass")
    _tracy_signpost("CONDITIONING_PERF_PASS")

    for iter_idx in range(iters):
        if trace_each_iter and not is_ci:
            _tracy_signpost(f"CONDITIONING_ITER_{iter_idx}")
        run_once()
        if flush_every > 0 and (iter_idx + 1) % flush_every == 0:
            _ace_step_flush_device_profiler(device)

    ttnn.synchronize_device(device)
    profiler.end("ace_step_cond_perf_pass")
    _ace_step_flush_device_profiler(device)

    profiler.print()
    init_wall = profiler.get("ace_step_cond_init")
    compile_wall = profiler.get("ace_step_cond_compile_pass")
    warmup_wall = profiler.get("ace_step_cond_warmup")
    perf_wall = profiler.get("ace_step_cond_perf_pass")
    per_iter_ms = (perf_wall * 1000.0 / max(1, iters)) if iters else 0.0

    logger.info(
        "AceStep conditioning Tracy harness (mode={}, variant={}, text_seq={}, iters={}): "
        "init={:.3f}s compile={:.3f}s warmup({}x)={:.3f}s perf_pass={:.3f}s (~{:.1f}ms/iter)",
        perf_mode,
        variant_label,
        text_seq,
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
        ), f"ace_step_cond_perf_pass {perf_wall}s exceeds ACE_STEP_PERF_MAX_SECONDS={budget}s"
