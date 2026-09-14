# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where does a TRACED verify step's wall time go, piece by piece?

MEASURED 2026-09-14 (T3K, block 16, bucket 128, after the readback narrowing):

    replay      107.1 ms  (52.5 %)   device time is 105.8 ms -- the replay IS the device
    save         30.7 ms  (15.0 %)   see below: ~1.7 ms amortized in the real loop
    logits       26.8 ms  (13.1 %)
    stage        19.4 ms  ( 9.5 %)
    restore      17.7 ms  ( 8.7 %)
    taps          2.3 ms
    setup         0.1 ms
    TOTAL       204.1 ms

THE VERIFY IS NOW DEVICE-BOUND. replay 107.1 ms against 105.8 ms of device time is ~99 %
utilization: tracing has extracted everything there is to extract here, and no further dispatch work
on the verify forward can help it. What is left is the ~68 ms of host work around it, plus the
device floor itself (25 % collectives, 17.6 % layout churn -- see test_profile_verify_forward.py).

TWO CORRECTIONS this measurement forced:

* ``save`` is overstated HERE. This harness calls save_gdn_state every iteration; the real loop
  saves only when the anchor advances, i.e. once per ~18 steps at acceptance 7. Amortized it is
  ~1.7 ms/step, not 30.7. Read this row as the cost of a save, not the cost of a step.
* ``restore`` is 17.7 ms, NOT the 60-85 ms estimated from its dispatch count (~144 ttnn.copy calls
  across 48 GDN layers at the ~0.6 ms/dispatch exchange rate). The estimate was 3-5x too high.
  Moving it inside the capture is still worth ~17 ms and is still trace-legal -- it copies between
  persistent, fixed-address buffers -- but it is a modest lever, not the dominant one.

AND ONE THING THAT DOES NOT WORK: taking the verify's argmax on device.

Greedy verification uses these logits for exactly one thing, ``posterior = argmax(logits)``, so the
drafter's 4.5x win (tests/unit/test_drafter_device_argmax.py) looks like it should transplant here.
It does not. Implemented and measured: tokens identical, acceptance identical, and the loop went
**22.19 -> 4.48 tok/s**. Reverted.

The difference is the SOURCE tensor. The drafter argmaxes ``[1, 15, vocab]`` straight off a fresh
lm_head; the verify's ``_vt_logits`` is ``[1, 1, 128, vocab]`` -- ~63 MB -- and slicing that to 16
rows and untilizing it costs vastly more than the 26.8 ms readback it replaces. Same op, same
width, opposite verdict, decided by the tensor it reads. Do not re-derive this from the drafter
result; the readback here is already close to the best available form.

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
