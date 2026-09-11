# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FEASIBILITY PROBE: can :class:`TtDFlashDrafter` be trace-captured as it stands?

The prize is the largest in the drafter by far. It runs at **120 ms of wall against 10.3 ms of
device time** (``test_dflash_drafter_wall_time.py``) — ~91 % of a step is host dispatch of 174 ops,
and the block is a fixed shape, which is exactly the profile a trace erases. Nothing else left in
this model is worth a fraction of that.

This file does not implement tracing. It **captures the specific reasons the current design blocks
it**, as a runnable artifact, so the port that follows starts from evidence instead of a guess.

MEASURED (T3K, warmed 4 steps to context 48, then attempted capture):

    KV history per layer: (1, 8, 48, 128) -> (1, 8, 64, 128)   (context 48 -> 64)
    SHAPE-STABLE ACROSS STEPS: False
    CAPTURE: FAILED -> TT_FATAL: Cannot load new binaries during trace capture.
                       This program is not yet in program cache. Warm up before capturing a trace.

Read those two lines together, because the second is a CONSEQUENCE of the first and not an
independent problem. The error invites you to warm up more; warming up cannot help. Every step's
K/V is 16 rows longer than the last, so every step compiles programs it has never seen, and no
number of warm-up steps ever populates the cache for the step you are about to capture. The capture
failed on the shape-driven compile before it even reached the ``_sliding_mask`` host write, so
blocker 2 below is still untested -- it is next in line, not absent.

Note also that a capture SUCCEEDING would not have been good news. ``shape_stable=False`` means a
trace would freeze one step's shapes and replay them against a history that has moved on: wrong
results, silently. The shape check therefore runs first, deliberately.

BLOCKERS, in the order they bite:

1. **The KV history grows every step.** ``_layer_attention`` does
   ``concat([hist_k, k])`` then slices back, so every step's tensors are 16 rows longer than the
   last and every buffer address moves. A trace bakes both. This is a deliberate design choice
   (see the module docstring: the target's paged path can only write at bucket-aligned offsets, and
   speculation advances 1-16 tokens), so removing it is a real port, not a tweak — a fixed-capacity
   KV buffer written in place, which the op-mapping doc has wanted anyway to kill the concat's
   untilize/retilize pairs.
2. **``_sliding_mask`` uploads from host inside the step.** ``ttnn.from_torch`` during capture is
   illegal — "Writes are not supported during trace capture", the same wall
   ``tt/gdn/decode.py`` and ``tt/wh_compat.py`` document hitting. It needs a resident
   mask buffer written before replay.
3. **``_rope_slice`` slices at a position-dependent offset.** The span is constant but the offset
   advances with ``start``, so the runtime args differ per step and a trace would freeze one step's
   positions.
4. Host-side bookkeeping (``_ctx_len``, the ``_mm_pc`` cache, the ``assert``) runs per step; it is
   harmless at capture but means a replay does not advance the drafter's own state.

Run::

    MESH_DEVICE=T3K DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash \\
      pytest -svq models/demos/blackhole/qwen36/tests/perf/test_dflash_drafter_trace_probe.py
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.perf.test_profile_dflash_drafter import _upload_noise, _upload_taps
from models.demos.blackhole.qwen36.tt.dflash.config import (
    DFlashDrafterConfig,
    load_drafter_state_dict,
    resolve_drafter_path,
)
from models.demos.blackhole.qwen36.tt.dflash.drafter import TtDFlashDrafter

TRACE_REGION = 90_000_000


def _mesh_shape():
    import os

    name = (os.environ.get("MESH_DEVICE") or "").upper()
    return {"P150": (1, 1), "N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(name, (1, 8))


MESH_SHAPE = _mesh_shape()


@pytest.mark.timeout(0)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": TRACE_REGION}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_drafter_trace_probe(mesh_device, device_params, reset_seeds, ensure_gc):
    """Attempt a capture of one drafter step and report exactly where it stops."""
    del device_params
    try:
        path = resolve_drafter_path()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"drafter checkpoint unavailable ({type(e).__name__}: {e})")

    cfg = DFlashDrafterConfig.from_pretrained(path)
    drafter = TtDFlashDrafter(mesh_device, cfg, load_drafter_state_dict(path))
    gen = torch.Generator().manual_seed(0)
    q_len = n_new = cfg.block_size

    def inputs():
        taps = _upload_taps(mesh_device, torch.randn(1, n_new, cfg.target_feature_size, generator=gen) * 0.05, cfg)
        noise = _upload_noise(mesh_device, torch.randn(1, q_len, cfg.hidden_size, generator=gen) * 0.05)
        return taps, noise

    # Warm: compile every program and get past the first step's no-history special case.
    for _ in range(3):
        taps, noise = inputs()
        ttnn.deallocate(drafter.forward(drafter.project_taps(taps), noise, drafter.context_len + n_new))
        ttnn.deallocate(noise)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"warmed to context {drafter.context_len}; attempting capture")

    # Shape stability check FIRST -- if the step is not shape-stable, a trace is wrong even if the
    # capture succeeds, and that is the harder failure to notice.
    ctx_before = drafter.context_len
    shapes_a = [tuple(t.shape) for t in drafter._ctx_k if t is not None]
    taps, noise = inputs()
    ttnn.deallocate(drafter.forward(drafter.project_taps(taps), noise, drafter.context_len + n_new))
    ttnn.deallocate(noise)
    shapes_b = [tuple(t.shape) for t in drafter._ctx_k if t is not None]
    logger.info(f"KV history per layer: {shapes_a[0]} -> {shapes_b[0]} (context {ctx_before} -> {drafter.context_len})")
    shape_stable = shapes_a == shapes_b

    taps, noise = inputs()
    start = drafter.context_len + n_new
    err = None
    try:
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        out = drafter.forward(drafter.project_taps(taps), noise, start)
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(out)
    except Exception as e:  # noqa: BLE001 — the failure IS the result
        err = f"{type(e).__name__}: {str(e).splitlines()[0][:400]}"

    logger.info("=" * 100)
    logger.info(f"SHAPE-STABLE ACROSS STEPS: {shape_stable}  (a trace is only correct if this is True)")
    logger.info(f"CAPTURE: {'FAILED -> ' + err if err else 'succeeded'}")
    logger.info("=" * 100)
    print(f"\n>>> shape_stable={shape_stable}  capture_error={err}\n")
