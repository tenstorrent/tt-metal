# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Trace + 2CQ wrapping of ``TtQwen3EmbeddingEncoder.forward_device`` (28-layer text encoder).

Exercises the same code path as ``test_qwen3_embedding_encoder_pcc.py`` but wraps the device-only
``forward_device`` in a TTNN trace and re-executes it N times with host-side input rewrites on CQ 1.
The two host transfers previously embedded in ``forward()`` (token-id upload and attention-bias
upload) now live OUTSIDE the trace so the captured graph is pure compute.

Inputs to the trace:
- ``ids_dev``         : ``[B, S]`` uint32 ROW_MAJOR (persistent device tensor)
- ``attn_bias_dev``   : ``[B, 1, S, S]`` bfloat16 TILE (persistent device tensor; bias precomputed
                       on host from the attention mask via ``causal_padding_attn_bias_np``)

The Qwen3 encoder runs **once per prompt** in the e2e pipeline (not per Euler step), so the
direct per-iter speedup from trace is smaller than for the DiT body. The point is consistent
latency + the ability to overlap host token-prep with device compute on CQ 1.

Skips when the local Qwen3-Embedding-0.6B checkpoint isn't present (same gating as the PCC test):

    ~/.cache/huggingface/hub/ACE-Step-1.5-checkpoints/Qwen3-Embedding-0.6B/{config.json,model.safetensors}

Run:

    pytest models/demos/ace_step_v1_5/perf/module_trace/test_qwen3_encoder_trace_2cq.py -v -s
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from models.demos.ace_step_v1_5.tests._dit_decoder_pcc_common import assert_pcc_print
from models.demos.ace_step_v1_5.ttnn_impl.qwen3_embedding_encoder import (
    TtQwen3EmbeddingEncoder,
    causal_padding_attn_bias_np,
)

_DEFAULT_QWEN_DIR = Path.home() / ".cache" / "huggingface" / "hub" / "ACE-Step-1.5-checkpoints" / "Qwen3-Embedding-0.6B"

_DEFAULT_ITERS = int(os.environ.get("ACE_STEP_TRACE_TEST_ITERS", "16"))
# Trace replay must match the compile-pass output deterministically (same inputs, same kernels).
_TRACE_VS_COMPILE_PCC = 0.999

# ``trace_device`` comes from the demo conftest (shared session device with 2 CQs +
# trace region). Opening a per-test second device on the same physical hardware
# corrupts TT's dispatch state and breaks the session ``close_device`` sync.


def _ckpt_dir() -> Path | None:
    d = _DEFAULT_QWEN_DIR.resolve()
    if (d / "model.safetensors").is_file():
        return d
    if any(d.glob("model-*.safetensors")):
        return d
    return None


_SKIP_REASON = (
    f"Qwen3-Embedding-0.6B not found at {_DEFAULT_QWEN_DIR}. "
    "Populate that directory (e.g. run the ACE-Step demo once so weights download there)."
)


@pytest.mark.skipif(_ckpt_dir() is None, reason=_SKIP_REASON)
def test_qwen3_encoder_trace_2cq(trace_device):
    """Trace + 2CQ replay of the Qwen3 text encoder, with PCC parity vs the no-trace compile pass."""
    import ttnn

    ckpt = _ckpt_dir()
    assert ckpt is not None
    text_dir = ckpt

    device = trace_device

    # Build the encoder once. RoPE caches + 28 layer instances are uploaded inside __init__.
    enc = TtQwen3EmbeddingEncoder(
        device=device,
        hf_model_dir=str(text_dir),
        qwen_safetensors_path=str(text_dir / "model.safetensors"),
    )
    cfg = enc.cfg
    S = int(cfg.max_seq_len)
    B = 1

    # --- HOST INPUTS --------------------------------------------------------------------
    # Use the AutoTokenizer so the token ids are realistic (same recipe as the demo / PCC test).
    # If transformers isn't installed in the env, fall back to a deterministic numpy id sequence
    # so the trace path still exercises the same shapes.
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(text_dir))
        prompt = "lofi hip hop, warm vinyl"
        tokens = tok(prompt, padding="max_length", truncation=True, max_length=S, return_tensors="np")
        ids_np = np.asarray(tokens["input_ids"], dtype=np.uint32).reshape(B, S)
        attn_np = np.asarray(tokens["attention_mask"], dtype=np.float32).reshape(B, S)
    except Exception:
        ids_np = np.arange(B * S, dtype=np.uint32).reshape(B, S) % 32000
        attn_np = np.ones((B, S), dtype=np.float32)

    bias_np = causal_padding_attn_bias_np(attn_np, S).astype(np.float32)

    # --- PERSISTENT DEVICE INPUTS + MATCHING HOST TENSORS -------------------------------
    # Two persistent device tensors get rewritten every iteration via copy_host_to_device_tensor
    # on CQ 1; the trace body reads them as its inputs.
    mapper = ttnn.ReplicateTensorToMesh(device) if hasattr(ttnn, "ReplicateTensorToMesh") else None
    ids_dev = ttnn.as_tensor(
        ids_np,
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=enc.mem,
        mesh_mapper=mapper,
    )
    attn_bias_dev = ttnn.as_tensor(
        bias_np,
        device=device,
        dtype=enc.dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=enc.mem,
        mesh_mapper=mapper,
    )
    # Matching host tensors for per-iteration copy_host_to_device_tensor (CQ 1).
    ids_host = ttnn.as_tensor(ids_np, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    attn_bias_host = ttnn.as_tensor(bias_np, dtype=enc.dtype, layout=ttnn.TILE_LAYOUT)

    # --- COMPILE PASS (CQ 0, no trace) --------------------------------------------------
    y_compile_dev = enc.forward_device(ids_dev, attn_bias_dev)
    ttnn.synchronize_device(device)
    y_compile = ttnn.to_torch(y_compile_dev).to(torch.float32)
    assert torch.isfinite(y_compile).all(), "Compile-pass encoder output has non-finite values."
    # Expected shape [B, 1, S, H] (TTNN keeps the leading singleton from the reshape).
    expected_shape = (B, 1, S, int(cfg.hidden_size))
    assert (
        tuple(y_compile.shape) == expected_shape
    ), f"Unexpected compile output shape: got {tuple(y_compile.shape)}, expected {expected_shape}"
    try:
        ttnn.deallocate(y_compile_dev)
    except Exception:
        pass

    # --- WARMUP PASS (CQ 0) -------------------------------------------------------------
    y_warmup_dev = enc.forward_device(ids_dev, attn_bias_dev)
    ttnn.synchronize_device(device)
    try:
        ttnn.deallocate(y_warmup_dev)
    except Exception:
        pass

    # --- CAPTURE TRACE ------------------------------------------------------------------
    # If this raises "Writes/Reads are not supported during trace capture", grep:
    #   rg "ttnn\.(ones_like|zeros_like|full|as_tensor|from_torch)" models/demos/ace_step_v1_5/ttnn_impl
    # and hoist the offending call to __init__ or replace with a scalar-op equivalent.
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    y_trace_dev = enc.forward_device(ids_dev, attn_bias_dev)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    trace_output_addr = y_trace_dev.buffer_address()

    op_event = ttnn.record_event(device, 0)

    # --- TIMED TRACE EXECUTION (CQ 0 compute, CQ 1 writes) ------------------------------
    iters = max(1, int(_DEFAULT_ITERS))
    latencies_ms: list[float] = []
    for _ in range(iters):
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(ids_host, ids_dev, cq_id=1)
        ttnn.copy_host_to_device_tensor(attn_bias_host, attn_bias_dev, cq_id=1)
        write_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, write_event)

        t0 = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        op_event = ttnn.record_event(device, 0)
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1e3)

    # --- VALIDATE BUFFER ADDRESS STABILITY ---------------------------------------------
    assert trace_output_addr == y_trace_dev.buffer_address(), (
        f"Trace output buffer moved across executes: {trace_output_addr} -> {y_trace_dev.buffer_address()}. "
        "The captured graph included a non-deterministic allocation. Audit the encoder body for "
        "per-call host transfers."
    )

    # --- VALIDATE TRACE OUTPUT MATCHES COMPILE OUTPUT ----------------------------------
    y_trace = ttnn.to_torch(y_trace_dev).to(torch.float32)
    assert_pcc_print(
        "qwen3_encoder_trace_2cq.trace_vs_compile",
        y_compile,
        y_trace,
        pcc=_TRACE_VS_COMPILE_PCC,
    )

    # --- REPORT -------------------------------------------------------------------------
    avg_ms = float(np.mean(latencies_ms))
    best_ms = float(np.min(latencies_ms))
    p90_ms = float(np.percentile(latencies_ms, 90)) if len(latencies_ms) > 1 else avg_ms
    print(
        f"[ace_step_v1_5][trace_2cq] qwen3 encoder: iters={iters} "
        f"avg={avg_ms:.3f}ms best={best_ms:.3f}ms p90={p90_ms:.3f}ms "
        f"(layers={cfg.num_hidden_layers}, seq={S}, hidden={cfg.hidden_size})",
        flush=True,
    )

    # --- CLEANUP ------------------------------------------------------------------------
    try:
        ttnn.release_trace(device, tid)
    except Exception:
        pass
    for t in (y_trace_dev, ids_dev, attn_bias_dev):
        try:
            ttnn.deallocate(t)
        except Exception:
            pass
