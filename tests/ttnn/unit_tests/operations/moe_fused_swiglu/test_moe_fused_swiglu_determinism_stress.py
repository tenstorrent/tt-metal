# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""INTERLEAVED determinism stress for the SHIPPED moe_fused_swiglu configuration.

`test_moe_fused_swiglu_determinism.py` runs ONE shape repeatedly. That is the weaker half of the
question: a race whose outcome is fixed by a steady-state pipeline will reproduce bit-for-bit
forever, because every dispatch finds the grid in the same rhythm. This file attacks the other half
— it **interleaves many different shapes in a pseudo-random order** across thousands of dispatches,
so consecutive programs differ in `m_eff`, `m_blocks`, activation format, `emb` and capacity, and no
two dispatches see the same arrival timing twice.

WHY INTERLEAVING IS THE POINT. Every synchronisation object in this op is MONOTONE and never reset
inside a dispatch (`SEM_GO`, `SEM_DATA`, `SEM_HSLICE`, `SEM_H_FREE`, `SEM_H_RDY_BASE + s`,
`SEM_XSTAGED`, `SEM_WDSPLIT`). That is only safe because each launch re-zeroes them and because the
per-dispatch expectation arithmetic is a pure function of the mailbox words. Interleaving shapes is
what actually exercises that:

  * `m_eff` changes the slice plan (`slice_workers`), the `blocks_cap` used to clamp `HACK_AHEAD`,
    the `cb_h` slot stride and the h payload size. Running m_eff 1, 2, 4 and 8 back to back means a
    core's *previous* dispatch left CBs and flag cells at different offsets each time.
  * `m_blocks` 1 vs 2 vs 4 changes which M-block does the W_down DRAM read (`WD_RESIDENT` reads only
    at b == 0) and therefore when the writer's `SEM_WDSPLIT` publish lands relative to the reader.
  * Alternating `bf16_rm` / `bfp8_tile` alternates whether COMPUTE participates in x staging at all.
  * `count == 0` interleaves a dispatch that does NO work on any core, so any state a previous shape
    left behind has to survive being skipped over.

THE THREE ROUND-17 MECHANISMS THIS IS AIMED AT, all of which are new cross-agent protocols:
  1. `HPOSTED=1` — the h multicast payload is POSTED, so it generates no acks and its safety rests
     entirely on the non-posted VALID flag being LINKED behind it on the same VC. If that ordering
     ever fails, a receiver consumes a partially-written h block: wrong answer, no hang, and PCC
     would very likely still pass. Only a bitwise compare across many timings can see it.
  2. `WD_SPLIT=3` — the writer reads part of every W_down K-block on NOC_1 into a CB the READER owns,
     with an intra-core semaphore as the only completion proof. A missed handshake yields a
     half-written weight tile.
  3. `REDUCE_MECH=dest_acc` — the slice reduce accumulates in DEST across all KGROUPS contributors.
     The fold order is fixed by slot index (not arrival), which is exactly the property that must
     hold; the landing CB is still filled by 8 peers' NoC writes in arbitrary order.

THE COMPARE NEVER LEAVES THE DEVICE — same `ne`/`max`/`maximum` marker chain as the sibling file,
and the same tile-aligned slice to the rows the op provably writes (rows past `ceil_tile(count)` are
undefined by contract and hold stale DRAM). One readback per checkpoint, not per iteration.

FAIL-FAST WITH ATTRIBUTION. Markers are checked every `MOE_STRESS_CHECK` dispatches, so a
divergence names the shape and the dispatch WINDOW it appeared in rather than surfacing after the
whole run. On the green path the readback cost is one element per shape per checkpoint.

    # the default: ~5000 dispatches interleaved over 14 shapes
    scripts/run_safe_pytest.sh --run-all <this file>

    MOE_STRESS_ITERS=20000 MOE_STRESS_SEED=7 scripts/run_safe_pytest.sh --run-all <this file>
"""

import os
import random
import time

import pytest
import torch

import ttnn

from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu


from .test_moe_fused_swiglu_determinism import (
    LOCAL_EXPERT_ID,
    _build,
    assert_marker_can_fire,
    assert_reference_is_live,
    assert_shipped_configuration,
    defined_region,
    device_marker_is_set,
    device_tensors_mismatch_marker,
    merge_device_mismatch_markers,
    resolved_knobs,
    written_rows,
)


#: The op's worker grid is a PARAMETER now, not an environment knob. `MOE_GRID=11x8` selects the
#: 88-core configuration every graded number is quoted at; empty = the device's full grid. It is a
#: harness variable, passed through as `core_grid=`, so the op itself stays env-free.
def _core_grid():
    g = os.environ.get("MOE_GRID", "").strip().lower()
    if not g:
        return None
    x, y = g.split("x")
    return (int(x), int(y))


CORE_GRID = _core_grid()

#: Total op dispatches across the whole interleave (NOT per shape).
ITERS = int(os.environ.get("MOE_STRESS_ITERS", 5000))
#: Dispatches between marker checkpoints. Bounds how wide a failure window can be.
CHECK_EVERY = int(os.environ.get("MOE_STRESS_CHECK", 250))
SEED = int(os.environ.get("MOE_STRESS_SEED", 0))
#: Dispatches per shape BEFORE its reference is captured. 0 (the default) takes the reference from
#: the very first dispatch, which is the strictest thing to compare against and is what a correct op
#: must satisfy. A non-zero warmup exists for NEGATIVE CONTROLS: with a deliberate race injected, the
#: first dispatch reads uninitialised L1 and produces `inf`, which trips the liveness guard before
#: the bitwise marker ever gets a chance to speak. Warming up first gives the injected race finite,
#: plausible-but-varying values, so the control exercises the marker path itself.
WARMUP = int(os.environ.get("MOE_STRESS_WARMUP", 0))

#: (emb, capacity, count, format). Chosen so the interleave sweeps every runtime regime the
#: kernels branch on, not just a list of big shapes:
#:
#:   m_eff  — `m_tiles_eff` rounds the tail block UP to a power of two in [M_EFF_MIN, M_BLOCK],
#:            so counts 32/64/96/128/160 give m_eff 1/2/4/4/8. m_eff drives the slice plan, the
#:            h payload, the cb_h slot stride and the HACK_AHEAD clamp.
#:   m_blocks — 1 (<=256), 2 (257..512), 4 (1024). Only b == 0 reads W_down under WD_RESIDENT.
#:   ragged  — 255 / 257 / 513 straddle the tile and the M-block seam.
#:   format  — bf16_rm puts COMPUTE in the x staging path (fused tilize); bfp8_tile does not.
#:   emb     — 6144 and 7168 give different KR_PAD / EC_MAX and a different bank-run shape.
#:   count 0 — every core does zero M-blocks: no CB traffic, no collective, no semaphore.
#:
#: Capacity is mostly 1024 to keep ~14 resident input sets affordable; the two graded focus cells
#: at capacity 5120 are included so the stress covers the exact shapes the perf numbers quote.
_DEFAULT_SHAPES = (
    "7168,1024,32,bf16_rm;"
    "7168,1024,64,bfp8_tile;"
    "7168,1024,96,bf16_rm;"
    "7168,1024,128,bfp8_tile;"
    "7168,1024,160,bf16_rm;"
    "7168,1024,255,bfp8_tile;"
    "7168,1024,256,bf16_rm;"
    "7168,1024,257,bfp8_tile;"
    "7168,1024,512,bf16_rm;"
    "7168,1024,1024,bfp8_tile;"
    "6144,1024,256,bf16_rm;"
    "6144,1024,513,bfp8_tile;"
    "7168,5120,256,bf16_rm;"
    "7168,5120,512,bfp8_tile"
)

#: Interleaved alongside the compared shapes but never compared (its output rows are all
#: undefined). It exists to make a no-work dispatch land between two real ones.
_ZERO_SHAPE = (7168, 1024, 0, "bf16_rm")


def _shapes():
    spec = os.environ.get("MOE_STRESS_SHAPES", _DEFAULT_SHAPES)
    out = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        emb, capacity, count, fmt = part.split(",")
        out.append((int(emb), int(capacity), int(count), fmt.strip()))
    return out


class _Shape:
    """One resident input set plus its reference output and running mismatch marker."""

    def __init__(self, spec, device):
        self.emb, self.capacity, self.count, self.fmt = spec
        self.label = f"{self.fmt}_e{self.emb}_c{self.capacity}_n{self.count}"
        self.x, self.w, self.counts, self.idx = _build(self.emb, self.capacity, self.count, self.fmt, device)
        self.rows = written_rows(self.count)
        self.reference = None
        self.marker = None
        self.dispatches = 0
        self.window_start = 0

    def run(self, device):
        out = moe_fused_swiglu(
            self.x, self.w[0], self.w[1], self.w[2], self.counts, self.idx, LOCAL_EXPERT_ID, core_grid=CORE_GRID
        )
        assert list(out.shape) == [1, 1, self.capacity, self.emb]
        self.dispatches += 1
        if self.count == 0:
            ttnn.deallocate(out)  # every row undefined: nothing to compare, the point is the dispatch
            return
        region = defined_region(out, self.rows)
        if self.reference is None:
            # Keep the reference resident. `region` aliases `out` when count == capacity, so the
            # output must stay allocated too — do not deallocate either.
            self.reference = region
            self.peak = assert_reference_is_live(self.reference)
            assert_marker_can_fire(self.reference, self.peak, device)
            self._ref_out = out
            return
        self.marker = merge_device_mismatch_markers(self.marker, device_tensors_mismatch_marker(self.reference, region))
        if region is not out:
            ttnn.deallocate(region)
        ttnn.deallocate(out)

    def run_discard(self, device):
        """A dispatch whose output is thrown away — warmup only, never compared."""
        out = moe_fused_swiglu(
            self.x, self.w[0], self.w[1], self.w[2], self.counts, self.idx, LOCAL_EXPERT_ID, core_grid=CORE_GRID
        )
        self.dispatches += 1
        ttnn.deallocate(out)

    def diverged(self):
        return self.marker is not None and device_marker_is_set(self.marker)


def test_determinism_stress(device):
    """Thousands of interleaved dispatches; every shape must stay bitwise identical to its own first run."""
    overrides = assert_shipped_configuration()
    specs = _shapes()
    assert ITERS >= 2 * len(specs), f"MOE_STRESS_ITERS={ITERS} is too small to revisit {len(specs)} shapes"

    shapes = [_Shape(s, device) for s in specs]
    zero = _Shape(_ZERO_SHAPE, device)

    # Optional warmup (negative controls only — see WARMUP).
    for _ in range(WARMUP):
        for s in shapes:
            s.run_discard(device)

    # Reference pass, in declaration order: one dispatch per shape, each establishing its own
    # baseline and proving (via assert_marker_can_fire) that its compare is not vacuous.
    for s in shapes:
        s.run(device)

    # The interleave. A fresh pseudo-random permutation per round rather than one global shuffle,
    # so every shape is revisited at a bounded interval AND the order it is revisited in keeps
    # changing — a fixed cycle would settle into its own steady-state rhythm, which is exactly the
    # condition this file exists to avoid.
    rng = random.Random(SEED)
    order = shapes + [zero]
    done = len(shapes)
    checkpoint = done
    t0 = time.time()

    while done < ITERS:
        rng.shuffle(order)
        for s in order:
            if done >= ITERS:
                break
            s.run(device)
            done += 1

            if done - checkpoint >= CHECK_EVERY:
                failed = [s2 for s2 in shapes if s2.diverged()]
                if failed:
                    _fail(failed, checkpoint, done, overrides)
                for s2 in shapes:
                    s2.window_start = done
                checkpoint = done
                rate = done / max(time.time() - t0, 1e-9)
                print(f"[stress] {done}/{ITERS} dispatches, {rate:.1f}/s, all {len(shapes)} shapes bitwise stable")

    failed = [s for s in shapes if s.diverged()]
    if failed:
        _fail(failed, checkpoint, done, overrides)

    total = sum(s.dispatches for s in shapes) + zero.dispatches
    per = ", ".join(f"{s.label}:{s.dispatches}" for s in shapes)
    print(
        f"[stress] PASS — {total} dispatches over {len(shapes)} interleaved shapes "
        f"(+{zero.dispatches} zero-count), every shape bitwise identical to its own first run.\n"
        f"[stress] per-shape dispatches: {per}, zero-count: {zero.dispatches}\n"
        f"[stress] seed={SEED} overrides={overrides or 'none'} knobs={resolved_knobs()}"
    )


def _fail(failed, window_start, window_end, overrides):
    names = ", ".join(f"{s.label} (rows[0:{s.rows}], {s.dispatches} dispatches so far)" for s in failed)
    pytest.fail(
        f"moe_fused_swiglu is NOT deterministic under interleaved dispatch: {names} diverged from "
        f"its own first run somewhere in dispatch window [{window_start}, {window_end}]. "
        f"overrides={overrides or 'none'} knobs={resolved_knobs()}"
    )
