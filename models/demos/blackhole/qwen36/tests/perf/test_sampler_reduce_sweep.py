# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sweep the per-shard GREEDY SAMPLER reduction over its implementation choices.

WHAT THIS IS FOR
----------------
Greedy decode's served path (``demo/text_demo.py``, ``QWEN36_BATCHED_DECODE_MODE="shard"``, the
default) reduces each device's own vocab shard ``[1, 1, B, vocab/tp]`` to a per-user
``(argmax index, max value)`` pair on device, then reads back two tiny ``[num_devices, B]``
tensors. MEASURED (T3K TP=8, 27B, tests/perf/test_profile_model_tail_decode.py) that reduction is
198 us/token at B=1 and **1,192 us/step at B=32 -- more than the LM-head matmul (866 us)**, and
almost none of it is the reduction itself:

    op              B=1     B=32
    ReshapeView    32.2    553.9
    ArgMax         19.5    512.1
    Pad            54.8     54.9
    FillPad        57.8       --
    Reduce         18.8     39.0     <- the actual maxes
    Untilize       15.1     32.1

So it is reshape/pad overhead around a 39 us reduction, which makes it worth sweeping rather than
reasoning about. This file exists because iterating on it through the model is minutes per trial
(27B weight load) while the reduction depends on NOTHING but the shard shape -- a synthetic
``[1, 1, B, per_shard]`` tensor reproduces it exactly, in seconds.

WHY THE CURRENT SHAPE IS WHAT IT IS
----------------------------------
``ttnn`` reduces ``dim=-1`` in parallel over tile ROWS, and ``[1, 1, B, per_shard]`` has only
``B/32`` of them -- one core at B<=32. The demo therefore views the shard as a tall/narrow
``(B, R, C=32)`` grid so there are ``B*R/32`` tile rows, and reduces twice. ``R`` is rounded UP to
a multiple of 32 and the shard PADDED to ``R*C`` -- and the rounding is not for the first reduce,
it is so the SECOND reduce's ``[1, 1, B, R]`` input is tile-aligned. That is the detail that makes
"just drop the pad" wrong on its own, and it is why variant B moves the pad instead of deleting it.

VARIANTS
--------
``cur``      what ships: pad the shard to R*C (R rounded to 32), reshape, max, reshape, max.
``nopad``    per_shard is 970*32 EXACTLY, so the first stage needs no pad at all. Reshape the raw
             shard to (B, 970, 32), reduce, then pad the TINY [B, 970] intermediate to [B, 992]
             before the second reduce. Same arithmetic, but the pad moves off a 2 MB tensor onto a
             30 KB one.
``topk``     ``ttnn.topk(k=32, dim=-1)`` returns values AND indices in one op, which would delete
             the whole two-leg structure (no separate argmax, no max legs). k=32 because the kernel
             has a minimum; element 0 of each is the answer.
``direct``   ``ttnn.max(dim=-1)`` on the shard with no reshaping -- the one-tile-row case the
             reshape exists to avoid. Included as the control that justifies the complexity.
``argmax``   the argmax leg alone (untilize + argmax), so the two legs can be priced separately.

Every variant is checked against a torch reference on the SAME data before it is timed, so a fast
wrong answer cannot win the sweep.

Run (needs a device; skipped unless the env var is set)::

    QWEN_SAMPLER_SWEEP=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      pytest models/demos/blackhole/qwen36/tests/perf/test_sampler_reduce_sweep.py -v -s

For device kernel times rather than wall clock, run it under tracy and read
DEVICE KERNEL DURATION; the wall-clock column here is a screening tool (it includes host dispatch,
which at these op counts is not negligible).
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0_or_blackhole

# 27B at TP=8: vocab 248,320 / 8. Parametrized so the same sweep serves other TP factors.
PER_SHARD = int(os.environ.get("QWEN_SAMPLER_PER_SHARD", 31040))
BATCHES = [1, 8, 32]
ITERS = 3

_SKIP = os.environ.get("QWEN_SAMPLER_SWEEP") != "1"


def _mesh_shape():
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    explicit = {"P150": (1, 1), "N150": (1, 1), "P150X4": (1, 4), "N150X4": (1, 4), "N300": (1, 2), "T3K": (1, 8)}
    return explicit.get(name, (1, max(1, min(ttnn.get_num_devices(), 2))))


MESH_SHAPE = _mesh_shape()
_MULTI = MESH_SHAPE != (1, 1)
DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "num_command_queues": 2,
        **({"fabric_config": ttnn.FabricConfig.FABRIC_1D} if _MULTI else {}),
    }
]

NEG = -1e30

try:  # signposts let tracy slice each variant; absent outside a tracy run
    from tracy import signpost as _SP
except ImportError:  # pragma: no cover
    _SP = None


# ---------------------------------------------------------------- variants


def _v_cur(logits, B, per_shard):
    """Exactly what demo/text_demo.py runs today."""
    C = 32
    R = (((per_shard + C - 1) // C) + 31) // 32 * 32
    logits_rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
    idx = ttnn.argmax(logits_rm, dim=-1, keepdim=False)
    ttnn.deallocate(logits_rm)
    padded = ttnn.pad(logits, [(0, 0), (0, 0), (0, 0), (0, R * C - per_shard)], value=NEG)
    grid = ttnn.reshape(padded, (1, B, R, C))
    part = ttnn.max(grid, dim=-1)
    part_row = ttnn.reshape(part, (1, 1, B, R))
    val = ttnn.max(part_row, dim=-1)
    for t in (padded, grid, part, part_row):
        ttnn.deallocate(t)
    return idx, val


def _v_nopad(logits, B, per_shard):
    """First stage needs no pad when per_shard % C == 0; pad the tiny intermediate instead."""
    C = 32
    assert per_shard % C == 0, "nopad variant requires per_shard divisible by C"
    R1 = per_shard // C
    R2 = (R1 + 31) // 32 * 32  # second reduce wants a tile-aligned last dim
    logits_rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
    idx = ttnn.argmax(logits_rm, dim=-1, keepdim=False)
    ttnn.deallocate(logits_rm)
    grid = ttnn.reshape(logits, (1, B, R1, C))
    part = ttnn.max(grid, dim=-1)
    part_row = ttnn.reshape(part, (1, 1, B, R1))
    if R2 != R1:
        padded_small = ttnn.pad(part_row, [(0, 0), (0, 0), (0, 0), (0, R2 - R1)], value=NEG)
        ttnn.deallocate(part_row)
        part_row = padded_small
    val = ttnn.max(part_row, dim=-1)
    for t in (grid, part, part_row):
        ttnn.deallocate(t)
    return idx, val


def _v_topk(logits, B, per_shard):
    """One op for both legs. k=32 is the kernel minimum; element 0 is the winner."""
    k = 32
    vals, idxs = ttnn.topk(logits, k=k, dim=-1)
    return idxs, vals


def _v_direct(logits, B, per_shard):
    """No reshaping at all -- the control for whether the tall/narrow view earns its cost."""
    logits_rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
    idx = ttnn.argmax(logits_rm, dim=-1, keepdim=False)
    ttnn.deallocate(logits_rm)
    val = ttnn.max(logits, dim=-1)
    return idx, val


def _v_c64(logits, B, per_shard):
    """nopad with C=64 instead of 32. 31040 = 485*64, so this is the other exact factorization.

    Worth a shot because the ReshapeView -- not the reduce -- is what costs at B=32 (549us for a
    2MB re-tile, ~7 GB/s), and a 64-wide target tile column is a different re-tile pattern.
    """
    C = 64
    assert per_shard % C == 0
    R1 = per_shard // C
    R2 = (R1 + 31) // 32 * 32
    logits_rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
    idx = ttnn.argmax(logits_rm, dim=-1, keepdim=False)
    ttnn.deallocate(logits_rm)
    grid = ttnn.reshape(logits, (1, B, R1, C))
    part = ttnn.max(grid, dim=-1)
    part_row = ttnn.reshape(part, (1, 1, B, R1))
    if R2 != R1:
        pad_small = ttnn.pad(part_row, [(0, 0), (0, 0), (0, 0), (0, R2 - R1)], value=NEG)
        ttnn.deallocate(part_row)
        part_row = pad_small
    val = ttnn.max(part_row, dim=-1)
    for t in (grid, part, part_row):
        ttnn.deallocate(t)
    return idx, val


def _v_flat(logits, B, per_shard):
    """nopad, but fold the batch into the ROW axis: (1, 1, B*R1, C) rather than (1, B, R1, C).

    Same element count and same reduction, one fewer logical dim. If the re-tile cost is driven by
    the 4D indexing rather than the byte movement, this is where it shows.
    """
    C = 32
    assert per_shard % C == 0
    R1 = per_shard // C
    R2 = (R1 + 31) // 32 * 32
    logits_rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
    idx = ttnn.argmax(logits_rm, dim=-1, keepdim=False)
    ttnn.deallocate(logits_rm)
    grid = ttnn.reshape(logits, (1, 1, B * R1, C))
    part = ttnn.max(grid, dim=-1)  # (1,1,B*R1,1)
    part_row = ttnn.reshape(part, (1, 1, B, R1))
    if R2 != R1:
        pad_small = ttnn.pad(part_row, [(0, 0), (0, 0), (0, 0), (0, R2 - R1)], value=NEG)
        ttnn.deallocate(part_row)
        part_row = pad_small
    val = ttnn.max(part_row, dim=-1)
    for t in (grid, part, part_row):
        ttnn.deallocate(t)
    return idx, val


def _v_argmax(logits, B, per_shard):
    """The argmax leg alone, to price the two legs separately."""
    logits_rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
    idx = ttnn.argmax(logits_rm, dim=-1, keepdim=False)
    ttnn.deallocate(logits_rm)
    return idx, None


VARIANTS = {
    "cur": _v_cur,
    "nopad": _v_nopad,
    "c64": _v_c64,
    "flat": _v_flat,
    "direct": _v_direct,
    "argmax": _v_argmax,
}
# topk measured 17,292us of DEVICE time at both B=1 and B=32 -- ~90x the shipped reduction and
# batch-independent, so it is not a "needs tuning" result, it is the wrong kernel for k=1. Dropped
# from the default sweep; re-add if the kernel is ever reworked.
if os.environ.get("QWEN_SAMPLER_SWEEP_TOPK") == "1":
    VARIANTS["topk"] = _v_topk


def _check(name, idx_t, val_t, ref_idx, ref_val, mesh_device):
    """Verify a variant on device 0's replica before trusting its timing."""
    comp = ttnn.ConcatMeshToTensor(mesh_device, dim=0) if _MULTI else None
    got_i = (ttnn.to_torch(idx_t, mesh_composer=comp) if comp else ttnn.to_torch(idx_t)).reshape(-1)
    n = ref_idx.numel()
    # topk returns [.., B, k]; the winner is column 0 of each row.
    if name == "topk":
        got_i = (ttnn.to_torch(idx_t, mesh_composer=comp) if comp else ttnn.to_torch(idx_t)).reshape(-1, 32)[:, 0]
    got_i = got_i[:n].to(torch.int64)
    ok_i = bool((got_i == ref_idx).all())
    ok_v = True
    if val_t is not None and ref_val is not None:
        got_v = (ttnn.to_torch(val_t, mesh_composer=comp) if comp else ttnn.to_torch(val_t)).float()
        got_v = (got_v.reshape(-1, 32)[:, 0] if name == "topk" else got_v.reshape(-1))[:n]
        ok_v = bool(torch.allclose(got_v, ref_val, atol=2e-2, rtol=1e-3))
    return ok_i, ok_v


@pytest.mark.skipif(_SKIP, reason="set QWEN_SAMPLER_SWEEP=1 to run the sampler reduction sweep")
@pytest.mark.timeout(1800)
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("B", BATCHES, ids=[f"batch{b}" for b in BATCHES])
def test_sampler_reduce_sweep(mesh_device, device_params, B):
    """Correctness-checked wall-clock sweep of the greedy per-shard reduction."""
    del device_params
    mesh_device.enable_program_cache()
    per_shard = PER_SHARD

    torch.manual_seed(0)
    # Deliberately ALL-NEGATIVE logits in half the rows: a variant that reduces over zero padding
    # instead of -inf padding gets the right answer on positive data and the wrong one here.
    x = torch.randn(1, 1, B, per_shard, dtype=torch.float32) * 4.0
    x[:, :, : max(1, B // 2), :] -= 20.0
    ref_val, ref_idx = x.reshape(B, per_shard).max(dim=-1)
    xb = x.to(torch.bfloat16)
    ref_val = xb.reshape(B, per_shard).gather(1, ref_idx.view(-1, 1)).reshape(-1).float()

    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if _MULTI else None
    logits = ttnn.from_torch(
        xb,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        **({"mesh_mapper": mapper} if mapper else {}),
    )

    logger.info(f"sampler reduce sweep: B={B} per_shard={per_shard} mesh={MESH_SHAPE}")
    results = []
    for name, fn in VARIANTS.items():
        try:
            idx_t, val_t = fn(logits, B, per_shard)  # warmup + correctness
            ttnn.synchronize_device(mesh_device)
            ok_i, ok_v = _check(name, idx_t, val_t, ref_idx, ref_val, mesh_device)
            ttnn.deallocate(idx_t)
            if val_t is not None:
                ttnn.deallocate(val_t)
        except Exception as e:  # a variant the kernels reject is a result, not a failure
            logger.warning(f"  {name:8} UNSUPPORTED: {type(e).__name__}: {str(e)[:160]}")
            results.append((name, None, "unsupported", "unsupported"))
            continue

        # Signpost the timed loop so tracy can slice each variant out by name; divide the sliced
        # device total by ITERS. Wall clock is kept alongside because these variants differ in OP
        # COUNT (3 vs 7), so host dispatch is part of the real cost and device time alone would
        # flatter the chatty ones.
        if _SP is not None:
            _SP(f"{name}_start")
        t0 = time.time()
        for _ in range(ITERS):
            idx_t, val_t = fn(logits, B, per_shard)
            ttnn.deallocate(idx_t)
            if val_t is not None:
                ttnn.deallocate(val_t)
        ttnn.synchronize_device(mesh_device)
        us = (time.time() - t0) / ITERS * 1e6
        if _SP is not None:
            _SP(f"{name}_stop")
        results.append((name, us, "idx OK" if ok_i else "idx WRONG", "val OK" if ok_v else "val WRONG"))
        logger.info(f"  {name:8} {us:9.1f} us/call   {results[-1][2]:10} {results[-1][3]}")

    ttnn.deallocate(logits)
    logger.info(f"=== B={B} summary (wall clock, {ITERS} iters) ===")
    for name, us, oi, ov in results:
        logger.info(f"  {name:8} {'n/a' if us is None else f'{us:9.1f} us':>12}   {oi:12} {ov}")
    # The sweep reports; it does not gate. Only the shipped variant's correctness is asserted.
    cur = next(r for r in results if r[0] == "cur")
    assert cur[2] == "idx OK" and cur[3] == "val OK", f"reference variant itself is wrong: {cur}"
