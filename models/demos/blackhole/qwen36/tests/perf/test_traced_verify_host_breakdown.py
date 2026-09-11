# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where does a TRACED verify step's wall time go, piece by piece?

The verify forward was traced (1.96x end to end) and then profiled: **105.8 ms of device time**
(tests/perf/test_profile_verify_forward.py). But a forward costs ~460 ms of wall, so the traced
path is still only ~23 % device-utilized and ~350 ms per forward is NOT the device. That is now the
largest single lever in the loop, and there is no attribution for it -- tracy profiles device ops,
not host time.

So time the pieces directly. Every step of a traced verify, in order:

    restore_gdn_state      48 GDN layers x ttnn.copy, EAGER (outside the trace)
    setup                  _build_request_rope + _set_vision_merge
    stage_verify_inputs    6 copy_host_to_device_tensor
    execute_trace + sync   the part that is actually traced
    to_torch(logits)       reads [1, bucket, vocab] back -- the FULL bucket, ~39 MB, when only
                           valid_len rows are ever used
    take_taps              slices the 5 residual taps
    save_gdn_state         48 layers x ttnn.copy, EAGER

Two suspects going in, both cheap to fix if they are the answer: the logits readback moves the whole
128-row bucket when 16 rows are wanted, and the GDN state save/restore is ~96 eager dispatches per
step that the trace does not cover.

This does NOT assert a threshold -- host time is machine-dependent. It prints a breakdown.

Run::

    DFLASH_RUN_TARGET=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      TT_CACHE_PATH=$HOME/.cache/tt_cache/Qwen3.6-27B \\
      pytest -svq models/demos/blackhole/qwen36/tests/perf/test_traced_verify_host_breakdown.py
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

PAGED_BLOCK_SIZE = 64
NUM_BLOCKS = 64
TRACE_REGION = 200_000_000
BLOCK = 16
ITERS = 8


def _mesh_shape():
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
def test_traced_verify_host_breakdown(mesh_device, device_params, reset_seeds, ensure_gc):
    """Per-phase wall time of one traced verify step."""
    del device_params
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to run the full 27B")

    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * PAGED_BLOCK_SIZE)
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, PAGED_BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)
    bucket = model._mask_bucket_for(BLOCK)

    g = torch.Generator().manual_seed(5)
    token_buf = torch.zeros(1, bucket, dtype=torch.int32)
    token_buf[:, :BLOCK] = torch.randint(1000, 2000, (1, BLOCK), generator=g, dtype=torch.int32)

    model.set_residual_taps([1, 16, 31, 46, 61], keep_on_device=True)
    model.capture_verify_trace(page_table, bucket, warm_tokens=token_buf, valid_len=BLOCK)
    snap = model.save_gdn_state()

    phases = {k: 0.0 for k in ("restore", "setup", "stage", "replay", "logits", "taps", "save")}

    def _t(key, fn):
        t0 = time.perf_counter()
        r = fn()
        phases[key] += (time.perf_counter() - t0) * 1000
        return r

    for _ in range(ITERS):
        _t("restore", lambda: model.restore_gdn_state(snap))

        def _setup():
            model._build_request_rope(token_buf[:, :BLOCK], None)
            model._set_vision_merge(token_buf, None, 0)

        _t("setup", _setup)
        _t("stage", lambda: model.stage_verify_inputs(token_buf, BLOCK, 0, page_table, bucket))

        def _replay():
            ttnn.execute_trace(model.device, model._vt_trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(model.device)

        _t("replay", _replay)
        # Call the REAL readback, not a copy of it. An earlier version of this file inlined
        # verify_traced's body here, and when that body was optimised this test kept timing the old
        # one -- reporting 609 ms for a readback the product no longer performs. Time the shipping
        # code or the number is fiction.
        _t("logits", lambda: model._read_verify_logits(BLOCK))
        if getattr(model, "_vt_taps", None):
            model._taps = dict(model._vt_taps)
        _t("taps", lambda: model.take_taps(BLOCK))
        _t("save", lambda: model.save_gdn_state(into=snap))

    total = sum(phases.values()) / ITERS
    logger.info("=" * 78)
    for k, v in sorted(phases.items(), key=lambda kv: -kv[1]):
        logger.info(f"  {k:8} {v / ITERS:8.1f} ms  ({100 * v / sum(phases.values()):5.1f} %)")
    logger.info(f"  {'TOTAL':8} {total:8.1f} ms per traced verify step (device time is 105.8 ms)")
    logger.info("=" * 78)
    print(f"\n>>> traced verify step: {total:.1f} ms wall, device 105.8 ms\n")
    for k, v in sorted(phases.items(), key=lambda kv: -kv[1]):
        print(f"    {k:8} {v / ITERS:8.1f} ms")
