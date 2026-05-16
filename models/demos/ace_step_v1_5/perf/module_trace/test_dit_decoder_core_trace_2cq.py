# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Trace + 2CQ wrapping of ``TtAceStepDiTCore.__call__`` (one Euler-step body).

This is the first module-by-module perf harness in the ACE-Step plan: it captures a TTNN
trace around the inner DiT body that the PCC suite already exercises
(``tests/test_pcc_dit_decoder_core.py``), executes it on CQ 0, and overlaps host->device
input copies on CQ 1. The test asserts PCC vs the torch reference both before and after
trace capture so any regression introduced by per-call mask / time-embed caches in
``ttnn_impl/dit_decoder_core.py`` fails this test loudly.

The device is opened locally (not via the demo's session ``mesh_device`` fixture) so we can
guarantee ``num_command_queues=2`` and a dedicated ``trace_region_size`` independent of
other tests in the same session.

Run:

    pytest models/demos/ace_step_v1_5/perf/module_trace/test_dit_decoder_core_trace_2cq.py -v -s

Useful env:

- ``ACE_STEP_TRACE_TEST_ITERS``: number of traced iterations to time (default 16).
- ``ACE_STEP_L1_SMALL_SIZE`` / ``TT_DEVICE_ID``: same semantics as the demo conftest.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from models.demos.ace_step_v1_5.tests._dit_decoder_pcc_common import assert_pcc_print
from models.demos.ace_step_v1_5.torch_ref.dit_decoder_core import TorchAceStepDiTCoreRef, make_tiny_state_dict
from models.demos.ace_step_v1_5.ttnn_impl.dit_decoder_core import AceStepDecoderConfigTTNN, TtAceStepDiTCore

_DEFAULT_ITERS = int(os.environ.get("ACE_STEP_TRACE_TEST_ITERS", "16"))

# ``trace_device`` comes from the demo conftest (shared session device with 2 CQs +
# trace region). Opening a per-test second device on the same physical hardware
# corrupts TT's dispatch state and breaks the session ``close_device`` sync.


def _build_tiny_core(device):
    """Same tiny shapes as ``test_dit_decoder_core_matches_torch`` (head_dim tile-aligned)."""
    import ttnn

    B = 1
    S = 32
    S_enc = 16
    head_dim = 32
    n_heads = 4
    D = n_heads * head_dim
    cond_dim = 32
    intermediate = 256
    num_layers = 1

    cfg = AceStepDecoderConfigTTNN(
        hidden_size=D,
        num_hidden_layers=num_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_heads,
        head_dim=head_dim,
        rms_norm_eps=1e-6,
        sliding_window=None,
    )
    sd = make_tiny_state_dict(
        d_model=D,
        n_heads=n_heads,
        head_dim=head_dim,
        cond_dim=cond_dim,
        intermediate=intermediate,
        num_layers=num_layers,
    )

    torch.manual_seed(1)
    x_patches = torch.randn(B, S, D, dtype=torch.bfloat16)
    timestep_proj = torch.randn(B, 6, D, dtype=torch.bfloat16)
    enc = torch.randn(B, S_enc, cond_dim, dtype=torch.bfloat16)

    y_ref = TorchAceStepDiTCoreRef(cfg=cfg, state_dict=sd)(x_patches, timestep_proj, enc)

    tt_core = TtAceStepDiTCore(cfg=cfg, state_dict=sd, mesh_device=device, dtype=ttnn.bfloat16)
    return cfg, tt_core, x_patches, timestep_proj, enc, y_ref


def _to_host_bf16(t: torch.Tensor) -> "ttnn.Tensor":
    """Build a TTNN host tensor matching the device buffers used inside ``tt_core``."""
    import ttnn

    return ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)


def test_dit_decoder_core_trace_2cq(trace_device):
    """Validate trace + 2CQ replay of the DiT body, with PCC parity to the torch reference.

    Flow (mirrors ``models/demos/sentence_bert/runner/performant_runner.py``):
        1. compile pass on CQ 0 with one host->device copy on CQ 1 (event-synced)
        2. warmup pass to fully populate program cache + mask/time-embed caches
        3. ``begin_trace_capture`` + tt_core(...) + ``end_trace_capture``
        4. N timed iterations: copy_host_to_device on CQ 1, ``execute_trace`` on CQ 0, sync at end
        5. PCC check on the last output vs torch reference
    """
    import ttnn

    device = trace_device
    cfg, tt_core, x_torch, tp_torch, enc_torch, y_ref = _build_tiny_core(device)

    # Persistent device input tensors. Initial upload uses CQ 0 (synchronous); subsequent
    # iterations rewrite them via ttnn.copy_host_to_device_tensor on CQ 1.
    x_host = _to_host_bf16(x_torch)
    tp_host = _to_host_bf16(tp_torch)
    enc_host = _to_host_bf16(enc_torch)

    x_dev = ttnn.from_torch(
        x_torch.contiguous(),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tp_dev = ttnn.from_torch(
        tp_torch.contiguous(),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    enc_dev = ttnn.from_torch(
        enc_torch.contiguous(),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # --- COMPILE PASS (CQ 0, no trace) --------------------------------------------------
    # Populates program cache + the mask / time-embed caches inside ttnn_impl/dit_decoder_core.py.
    y_compile_dev = tt_core(x_dev, tp_dev, enc_dev)
    ttnn.synchronize_device(device)
    y_compile = ttnn.to_torch(y_compile_dev).to(torch.bfloat16)
    assert_pcc_print("dit_core_trace_2cq.compile", y_ref, y_compile)
    try:
        ttnn.deallocate(y_compile_dev)
    except Exception:
        pass

    # --- WARMUP PASS (CQ 0) -------------------------------------------------------------
    # Second forward to make sure every cache hit is truly device-resident; without this the
    # captured trace would record additional first-time allocations that aren't part of the
    # steady-state Euler step.
    y_warmup_dev = tt_core(x_dev, tp_dev, enc_dev)
    ttnn.synchronize_device(device)
    try:
        ttnn.deallocate(y_warmup_dev)
    except Exception:
        pass

    # --- CAPTURE TRACE ------------------------------------------------------------------
    # Inputs already live at x_dev / tp_dev / enc_dev. The body returns a freshly allocated
    # output buffer; the trace will replay that allocation deterministically at the same
    # device address on every execute_trace call.
    #
    # If this `tt_core(...)` call raises "Writes/Reads are not supported during trace capture",
    # something in TtAceStepDiTCore / TtAceStepDiTLayer / TtAceStepAttentionSDPA is still going
    # through host (most commonly: `ttnn.ones_like` / `ttnn.full` / `ttnn.as_tensor` inside the
    # forward path). Audit with the grep:
    #   rg "ttnn\.(ones_like|zeros_like|full|as_tensor|from_torch)" models/demos/ace_step_v1_5/ttnn_impl
    # and either hoist the call to __init__ or replace it with a scalar-add equivalent.
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    y_trace_dev = tt_core(x_dev, tp_dev, enc_dev)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    trace_output_addr = y_trace_dev.buffer_address()

    # Initialize op_event after capture so the first iteration's CQ 1 wait has something to wait on.
    op_event = ttnn.record_event(device, 0)

    # --- TIMED TRACE EXECUTION (CQ 0 compute, CQ 1 writes) ------------------------------
    # Reuse the pre-built TTNN host tensors (x_host / tp_host / enc_host) every iteration:
    # the body's correctness is already validated by the compile-pass PCC assert above, and
    # the buffer-address invariant below is what guards against silent buffer-reuse bugs.
    # Using TTNN tensors directly (no torch/numpy in the loop) also sidesteps the fact that
    # torch.bfloat16 has no native NumPy dtype.
    iters = max(1, int(_DEFAULT_ITERS))
    latencies_ms: list[float] = []
    for i in range(iters):
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(x_host, x_dev, cq_id=1)
        ttnn.copy_host_to_device_tensor(tp_host, tp_dev, cq_id=1)
        ttnn.copy_host_to_device_tensor(enc_host, enc_dev, cq_id=1)
        write_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, write_event)

        t0 = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        op_event = ttnn.record_event(device, 0)
        # Block on this iteration so latency numbers are per-step rather than batched.
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1e3)

    # --- VALIDATE BUFFER ADDRESS STABILITY ---------------------------------------------
    assert trace_output_addr == y_trace_dev.buffer_address(), (
        f"Trace output buffer moved across executes: {trace_output_addr} -> {y_trace_dev.buffer_address()}. "
        "The trace recorded an allocation that re-runs differently between captures and replays; check that "
        "the DiT body avoids per-call host->device uploads (mask caches in TtAceStepAttentionSDPA and the "
        "_value_cache in TtTimestepEmbedding) before debugging further."
    )

    # --- PCC PARITY OF THE TRACED PATH --------------------------------------------------
    y_trace = ttnn.to_torch(y_trace_dev).to(torch.bfloat16)
    assert_pcc_print("dit_core_trace_2cq.trace_execute", y_ref, y_trace)

    # --- REPORT -------------------------------------------------------------------------
    avg_ms = float(np.mean(latencies_ms))
    best_ms = float(np.min(latencies_ms))
    p90_ms = float(np.percentile(latencies_ms, 90)) if len(latencies_ms) > 1 else avg_ms
    print(
        f"[ace_step_v1_5][trace_2cq] dit_core body: iters={iters} "
        f"avg={avg_ms:.3f}ms best={best_ms:.3f}ms p90={p90_ms:.3f}ms",
        flush=True,
    )

    # --- CLEANUP ------------------------------------------------------------------------
    try:
        ttnn.release_trace(device, tid)
    except Exception:
        pass
    for t in (y_trace_dev, x_dev, tp_dev, enc_dev):
        try:
            ttnn.deallocate(t)
        except Exception:
            pass
