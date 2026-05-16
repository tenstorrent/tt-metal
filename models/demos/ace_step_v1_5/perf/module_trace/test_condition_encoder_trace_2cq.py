# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Trace + 2CQ wrapping of ``TtAceStepInstrumentalConditionEncoder.forward_device``.

Exercises the fast-preprocess text-only path used by ``ttnn_impl/e2e_model_tt.py``:

    text_hidden (Qwen3 output) -> text_projector -> slice -> concat([lyric_const, timbre_const,
                                                                     text_valid, text_pad])

Per the refactor in ``ttnn_impl/condition_encoder.py``, the 8 lyric + 4 timbre transformer layers
are **precomputed once** at ``__init__`` (their inputs are zero-filled dummies, so the output is
deterministic). The traced body therefore reduces to:

    text_projector linear  +  2 ttnn.slice  +  4 ttnn.to_layout  +  1 ttnn.concat

That is what gets timed below.

Skips when the ACE-Step base DiT checkpoint isn't present (needed for ``encoder.lyric_encoder.*``,
``encoder.timbre_encoder.*``, ``encoder.text_projector.weight``, ``null_condition_emb`` keys):

    ~/.cache/huggingface/hub/ACE-Step-1.5-checkpoints/acestep-v15-base/model.safetensors
    (or any acestep-v15-* variant)

Run:

    pytest models/demos/ace_step_v1_5/perf/module_trace/test_condition_encoder_trace_2cq.py -v -s
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from models.demos.ace_step_v1_5.tests._dit_decoder_pcc_common import assert_pcc_print
from models.demos.ace_step_v1_5.ttnn_impl.condition_encoder import TtAceStepInstrumentalConditionEncoder

_CKPT_ROOT_ENV = "ACE_STEP_CHECKPOINT_DIR"
_DEFAULT_CKPT_ROOT = Path("~/.cache/huggingface/hub/ACE-Step-1.5-checkpoints").expanduser()

_DEFAULT_ITERS = int(os.environ.get("ACE_STEP_TRACE_TEST_ITERS", "16"))
_TRACE_VS_COMPILE_PCC = 0.999

# ``trace_device`` comes from the demo conftest (shared session device with 2 CQs +
# trace region). Opening a per-test second device on the same physical hardware
# corrupts TT's dispatch state and breaks the session ``close_device`` sync.

# Shapes match the e2e demo: Qwen3 produces [B, 1, S, 1024] hidden states; condition encoder
# accepts the full sequence and slices into valid + pad chunks.
_BATCH = 1
_TEXT_SEQ_LEN = 256
_QWEN_HIDDEN = 1024
_VALID_TOKENS = 50  # arbitrary realistic value; doesn't affect trace stability


def _ckpt_root() -> Path:
    return Path(os.environ.get(_CKPT_ROOT_ENV, str(_DEFAULT_CKPT_ROOT))).expanduser()


def _find_dit_safetensors() -> Path | None:
    """Return the first ``acestep-v15-*/model.safetensors`` under the checkpoint root."""
    root = _ckpt_root()
    for variant in ("acestep-v15-turbo", "acestep-v15-base"):
        p = root / variant / "model.safetensors"
        if p.is_file():
            return p
    for variant_dir in sorted(root.glob("acestep-v15-*")):
        p = variant_dir / "model.safetensors"
        if p.is_file():
            return p
    return None


_SKIP_REASON = (
    f"ACE-Step DiT safetensors not found under {_DEFAULT_CKPT_ROOT}. "
    f"Set {_CKPT_ROOT_ENV} or run the demo once to populate the cache."
)


@pytest.mark.skipif(_find_dit_safetensors() is None, reason=_SKIP_REASON)
def test_condition_encoder_trace_2cq(trace_device):
    """Trace + 2CQ replay of the condition encoder, with PCC parity vs the no-trace compile pass."""
    import ttnn

    device = trace_device
    dit_safetensors = _find_dit_safetensors()
    assert dit_safetensors is not None

    # Build the encoder once. Construction also runs the 8+4 transformer layers ONCE (the dummy
    # zero-input call paths) and caches their outputs as `_lyric_const_tt` / `_timbre_const_tt`.
    enc = TtAceStepInstrumentalConditionEncoder(
        device=device,
        checkpoint_safetensors_path=str(dit_safetensors),
        dtype=ttnn.bfloat16,
    )

    # --- HOST INPUTS --------------------------------------------------------------------
    # Stand-in for Qwen3 hidden states: random [B, 1, S, D_text] device tensor. The condition
    # encoder accepts this directly via text_projector.forward_from_hidden().
    torch.manual_seed(7)
    text_hidden_torch = torch.randn(_BATCH, 1, _TEXT_SEQ_LEN, _QWEN_HIDDEN, dtype=torch.bfloat16)
    text_hidden_host = ttnn.from_torch(
        text_hidden_torch.contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    text_hidden_dev = ttnn.from_torch(
        text_hidden_torch.contiguous(),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    valid = int(_VALID_TOKENS)

    # --- COMPILE PASS (CQ 0, no trace) --------------------------------------------------
    enc_compile_dev, _null_dev = enc.forward_device(text_hidden_dev, valid)
    ttnn.synchronize_device(device)
    y_compile = ttnn.to_torch(enc_compile_dev).to(torch.float32)
    assert torch.isfinite(y_compile).all(), "Compile-pass condition-encoder output has non-finite values."
    # enc shape after concat: [B, 2 + S, D_dec] — 2 for the cached lyric+timbre tokens.
    expected_seq = 2 + _TEXT_SEQ_LEN
    assert y_compile.shape[0] == _BATCH and y_compile.shape[1] == expected_seq, (
        f"Unexpected compile output shape: got {tuple(y_compile.shape)}, expected first two dims "
        f"({_BATCH}, {expected_seq})."
    )
    try:
        ttnn.deallocate(enc_compile_dev)
    except Exception:
        pass

    # --- WARMUP PASS (CQ 0) -------------------------------------------------------------
    enc_warmup_dev, _ = enc.forward_device(text_hidden_dev, valid)
    ttnn.synchronize_device(device)
    try:
        ttnn.deallocate(enc_warmup_dev)
    except Exception:
        pass

    # --- CAPTURE TRACE ------------------------------------------------------------------
    # If this raises "Writes/Reads are not supported during trace capture", check that the
    # condition encoder's __init__ ran the lyric/timbre caching pass (search for
    # `_lyric_const_tt` / `_timbre_const_tt` in `ttnn_impl/condition_encoder.py`).
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    enc_trace_dev, _null_dev_trace = enc.forward_device(text_hidden_dev, valid)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    trace_output_addr = enc_trace_dev.buffer_address()

    op_event = ttnn.record_event(device, 0)

    # --- TIMED TRACE EXECUTION (CQ 0 compute, CQ 1 writes) ------------------------------
    iters = max(1, int(_DEFAULT_ITERS))
    latencies_ms: list[float] = []
    for _ in range(iters):
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(text_hidden_host, text_hidden_dev, cq_id=1)
        write_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, write_event)

        t0 = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        op_event = ttnn.record_event(device, 0)
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1e3)

    # --- VALIDATE BUFFER ADDRESS STABILITY ---------------------------------------------
    assert trace_output_addr == enc_trace_dev.buffer_address(), (
        f"Trace output buffer moved across executes: {trace_output_addr} -> {enc_trace_dev.buffer_address()}. "
        "Audit the condition encoder for per-call host transfers."
    )

    # --- VALIDATE TRACE OUTPUT MATCHES COMPILE OUTPUT ----------------------------------
    y_trace = ttnn.to_torch(enc_trace_dev).to(torch.float32)
    assert_pcc_print(
        "condition_encoder_trace_2cq.trace_vs_compile",
        y_compile,
        y_trace,
        pcc=_TRACE_VS_COMPILE_PCC,
    )

    # --- REPORT -------------------------------------------------------------------------
    avg_ms = float(np.mean(latencies_ms))
    best_ms = float(np.min(latencies_ms))
    p90_ms = float(np.percentile(latencies_ms, 90)) if len(latencies_ms) > 1 else avg_ms
    print(
        f"[ace_step_v1_5][trace_2cq] condition encoder: iters={iters} "
        f"avg={avg_ms:.3f}ms best={best_ms:.3f}ms p90={p90_ms:.3f}ms "
        f"(text_seq={_TEXT_SEQ_LEN}, valid={valid}, qwen_hidden={_QWEN_HIDDEN})",
        flush=True,
    )

    # --- CLEANUP ------------------------------------------------------------------------
    try:
        ttnn.release_trace(device, tid)
    except Exception:
        pass
    for t in (enc_trace_dev, text_hidden_dev):
        try:
            ttnn.deallocate(t)
        except Exception:
            pass
