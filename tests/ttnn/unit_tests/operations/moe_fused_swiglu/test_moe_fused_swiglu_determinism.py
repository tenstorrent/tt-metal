# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bitwise determinism of the SHIPPED (most optimal) moe_fused_swiglu configuration.

Run N times with identical inputs; every output must be BITWISE identical to iteration 0.
This is not an accuracy test (the golden suite owns PCC) — it guards the one class of bug
PCC cannot see: a result that depends on *arrival order* rather than on the data. The op is
built out of exactly the machinery that produces that bug — an 88-core reduce-scatter
(`MOE_SWIGLU_REDUCE=scatter`) where every core folds KGROUPS peers' partials into one
accumulator, plus per-slot flags, a peer invite semaphore and a mailbox. Float addition is
not associative, so if the accumulation ever consumed contributors in the order they LANDED
instead of in slot order, the op would still pass every PCC gate and silently return a
different answer run to run.

THE COMPARE NEVER LEAVES THE DEVICE. Pattern taken from ring joint SDPA
(`tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py:463`, `device_tensors_mismatch_marker`):

    mismatch_marker = ttnn.max(ttnn.ne(reference, actual))      # -> one element, on device
    merged          = ttnn.maximum(merged, mismatch_marker)     # accumulate across iterations
    ...
    float(ttnn.to_torch(ttnn.from_device(merged)).item()) != 0  # ONE readback, at the very end

A host compare would have to `to_torch` a 5120x7168 bfp8 tensor per iteration — ~37 MB over
PCIe plus an untilize, ten times per cell. The device compare reads back 1 element per cell
on the green path, and only expands to per-iteration markers when something actually differs.

WHAT IS COMPARED — NOT THE WHOLE TENSOR. Rows `[count, capacity)` of the output are UNDEFINED
by the op's contract: the kernels write `ceil(count/32)` tile-rows (rounded up to the block's
`m_eff`) and never touch the rest, so those rows hold whatever the freshly allocated DRAM
buffer happened to contain. Each iteration allocates a DIFFERENT buffer, so comparing them
would fail on stale memory rather than on the op. The compare is therefore sliced to
`ceil(count/TILE)*TILE` rows — every one of which the op provably writes, tile-aligned so the
slice is a whole-tile copy with no requantization. The tile-padding rows inside that last tile
are included on purpose: they are computed from the input's phantom rows, which are fixed, so
they must be reproducible too.

WHAT "MOST OPTIMAL" MEANS HERE — two things, both asserted rather than assumed:
  1. Every `MOE_SWIGLU_*` knob at its shipped default (see the module docstring of
     `moe_fused_swiglu_program_descriptor.py`). A stale env var from a knob sweep would
     otherwise silently retarget this test at a configuration nobody ships;
     `MOE_DET_ALLOW_KNOBS=1` opts into testing a non-default configuration on purpose.
  2. Weights at the op's DESIGNED ND shard (`weight_memory_configs`), checked through the
     READER's own predicate `nd_shard_n_tiles()` — the same call site the graded perf harness
     measures, and worth up to 11 % (see `test_moe_fused_swiglu_perf.py`'s docstring).

NEGATIVE CONTROL. A determinism test that compares a tensor with itself, or whose `ne` never
fires, is vacuously green. Every cell therefore also (a) poisons ONE element of the reference
and asserts the marker fires, and (b) asserts the reference is finite and non-zero, so an op
that returned all zeros could not pass.

    scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/moe_fused_swiglu/test_moe_fused_swiglu_determinism.py

    MOE_DET_CASES=guard MOE_DET_ITERS=20 scripts/run_safe_pytest.sh --run-all <this file>
"""

import os

import pytest
import torch

import ttnn

from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu
from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu_geometry as pd
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_program_descriptor import (
    nd_shard_n_tiles,
    weight_memory_configs,
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

TILE = 32
HIDDEN = 2048
NUM_GLOBAL_EXPERTS, NUM_LOCAL_EXPERTS, LOCAL_EXPERT_ID, GLOBAL_EXPERT_ID = 256, 8, 3, 137

_FORMATS = {"bf16_rm": (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT), "bfp8_tile": (ttnn.bfloat8_b, ttnn.TILE_LAYOUT)}

#: Default cell list: the graded focus shape in both activation formats, plus the two shapes
#: whose WRITTEN EXTENT differs from `count` — the non-tile-aligned seam and count == capacity.
_DEFAULT = "7168,5120,256,bf16_rm;7168,5120,256,bfp8_tile;7168,1024,255,bf16_rm;6144,1024,1024,bfp8_tile"

#: `MOE_DET_CASES=guard` — the Perf-2 guard set: one representative per distinct kernel path x
#: layout x M regime. Kept identical to `test_moe_fused_swiglu_r2_perf.GUARD_SET` so "fastest"
#: and "deterministic" are asserted over the same cells.
_GUARD = (
    "7168,5120,128,bf16_rm;"
    "7168,5120,256,bf16_rm;"
    "7168,5120,512,bf16_rm;"
    "7168,1024,256,bf16_rm;"
    "6144,5120,256,bf16_rm;"
    "7168,5120,5120,bf16_rm;"
    "7168,5120,128,bfp8_tile;"
    "7168,5120,256,bfp8_tile;"
    "7168,5120,512,bfp8_tile;"
    "7168,1024,256,bfp8_tile;"
    "6144,5120,256,bfp8_tile;"
    "7168,5120,5120,bfp8_tile"
)

ITERS = int(os.environ.get("MOE_DET_ITERS", 10))

#: The knobs whose resolved values this test certifies as deterministic. Reported in the pass
#: line so a green run states WHICH configuration it verified, not just that one existed.
_REPORTED_KNOBS = (
    "M_BLOCK",
    "W_RESIDENT",
    "WD_RESIDENT",
    "WD_AHEAD",
    "GU_CHUNKS",
    "HACK_AHEAD",
    "XPRIO",
    "WD_SPLIT",
    "DEPTH_X",
    "DEPTH_H",
    "ABLATE",
)


def _cases():
    spec = os.environ.get("MOE_DET_CASES", _DEFAULT)
    if spec == "guard":
        spec = _GUARD
    out = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        emb, capacity, count, fmt = part.split(",")
        out.append((int(emb), int(capacity), int(count), fmt.strip()))
    return out


# ---------------------------------------------------------------------------
# "most optimal" == the shipped defaults. Assert it instead of assuming it.
# ---------------------------------------------------------------------------
#: The shipped value of every tuning constant this test certifies as deterministic. The op has no
#: environment knobs any more, so "shipped" is no longer "an empty environment" — it is these
#: values, compared against the live module. A source edit or an in-process rebind that moved one
#: would otherwise silently retarget the test at a configuration nobody ships.
_SHIPPED = {
    "M_BLOCK": 8,
    "DEPTH_X": 2,
    "DEPTH_H": 3,
    "DEPTH_OUT": 2,
    "WD_AHEAD": 1,
    "W_RESIDENT": True,
    "WD_RESIDENT": True,
    "GU_CHUNKS": 3,
    "XPRIO": True,
    "HACK_AHEAD": 2,
    "WD_SPLIT": 3,
    "ABLATE": "",
    "OUT_SUBBLOCK_H_GU": 1,
    "OUT_SUBBLOCK_H_DN_MAX": 1,
}


def assert_shipped_configuration():
    overrides = {k: getattr(pd, k) for k, v in _SHIPPED.items() if getattr(pd, k, v) != v}
    if os.environ.get("MOE_PERF_WPLACE", "nd_shard") != "nd_shard":
        overrides["MOE_PERF_WPLACE"] = os.environ["MOE_PERF_WPLACE"]
    if overrides and os.environ.get("MOE_DET_ALLOW_KNOBS") != "1":
        pytest.fail(
            "this test certifies the SHIPPED configuration, but these differ from it: "
            f"{overrides}. Restore them, or set MOE_DET_ALLOW_KNOBS=1 to certify a non-default "
            "configuration on purpose."
        )
    return overrides


def resolved_knobs():
    return {name: getattr(pd, name) for name in _REPORTED_KNOBS if hasattr(pd, name)}


# ---------------------------------------------------------------------------
# The device-side compare (ring joint SDPA's `device_tensors_mismatch_marker` et al.)
# ---------------------------------------------------------------------------
def device_tensors_mismatch_marker(reference_tensor, actual_tensor):
    """One-element device tensor, non-zero iff any element differs exactly."""
    return ttnn.max(ttnn.ne(reference_tensor, actual_tensor, dtype=ttnn.bfloat16))


def merge_device_mismatch_markers(current_marker, new_marker):
    if current_marker is None:
        return new_marker
    return ttnn.maximum(current_marker, new_marker)


def device_marker_value(marker):
    """The ONLY host readback on the green path: one element."""
    return float(ttnn.to_torch(ttnn.from_device(marker)).item())


def device_marker_is_set(marker):
    return device_marker_value(marker) != 0.0


def written_rows(count):
    """Rows the kernels provably write: `count` rounded UP to the tile row.

    Rows past this are never written by the op, so they hold stale DRAM from whatever
    previously owned the freshly allocated output buffer — different every iteration.
    """
    return ((count + TILE - 1) // TILE) * TILE


def defined_region(out, rows):
    """The written prefix, tile-aligned so the slice is a whole-tile copy (no requantization)."""
    if rows == int(out.shape[-2]):
        return out  # count == capacity: the whole tensor is written, skip the copy
    return ttnn.slice(out, [0, 0, 0, 0], [1, 1, rows, int(out.shape[-1])])


# ---------------------------------------------------------------------------
# Inputs — byte-identical to the graded perf harness's `_build`, so determinism is
# certified at the same call site perf is measured at.
# ---------------------------------------------------------------------------
def _build(emb, capacity, count, input_format, device):
    torch.manual_seed(42)
    x = torch.randn((1, 1, capacity, emb), dtype=torch.float32)
    if count < capacity:
        x[:, :, count:, :] = 100.0  # hostile sentinel: the over-computed tile padding is fixed too
    dt, lay = _FORMATS[input_format]
    tt_x = ttnn.from_torch(
        x.to(torch.bfloat16), dtype=dt, layout=lay, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    gate_up_mc, down_mc = weight_memory_configs(device, emb, HIDDEN, core_grid=CORE_GRID)
    tt_w = [
        ttnn.from_torch(
            torch.randn(s, dtype=torch.bfloat16),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=mc,
        )
        for s, mc in (((emb, HIDDEN), gate_up_mc), ((emb, HIDDEN), gate_up_mc), ((HIDDEN, emb), down_mc))
    ]
    # The READER's own predicate, not the memory config we asked for: an interleaved weight is
    # silently CORRECT and takes the slower uncoalesced stream, which is not what ships.
    widths = [nd_shard_n_tiles(w) for w in tt_w]
    assert all(w > 0 for w in widths), f"asked for the designed ND shard but the reader sees interleaved: {widths}"

    counts = torch.zeros(NUM_GLOBAL_EXPERTS, dtype=torch.int32)
    counts[GLOBAL_EXPERT_ID] = count
    idx = torch.tensor([(11 + 37 * i) % NUM_GLOBAL_EXPERTS for i in range(NUM_LOCAL_EXPERTS)], dtype=torch.int32)
    idx[LOCAL_EXPERT_ID] = GLOBAL_EXPERT_ID
    to_dev = lambda t: ttnn.from_torch(  # noqa: E731
        t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return tt_x, tt_w, to_dev(counts), to_dev(idx)


def assert_marker_can_fire(reference, peak, device):
    """NEGATIVE CONTROL. Poison ONE element of the reference and require the marker to fire.

    Without this, a broken `ne`, a self-compare, or a slice that dropped the compared region
    would all read as "deterministic". Run once per CORNER of the compared region — first
    element and last element, independently — so an off-by-one slice that lost either end
    fails here instead of reading as green.

    The delta is scaled to the tensor's own peak, NOT a fixed constant: bf16 carries 8 mantissa
    bits, and the over-computed tile-padding rows are derived from the input's 100.0 sentinel, so
    they reach ~1e9 and absorb any small absolute delta entirely. `peak` is >= |x| everywhere in
    the region, so `x + peak != x` at every element (including x == 0).
    """
    rows, emb = int(reference.shape[-2]), int(reference.shape[-1])
    delta = torch.zeros((1, 1, rows, emb), dtype=torch.bfloat16)
    for label, (r, c) in (("first", (0, 0)), ("last", (rows - 1, emb - 1))):
        delta[0, 0, r, c] = peak
        tt_delta = ttnn.from_torch(
            delta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        poisoned = ttnn.add(reference, tt_delta, dtype=ttnn.bfloat16)
        fired = device_marker_is_set(device_tensors_mismatch_marker(reference, poisoned))
        ttnn.deallocate(tt_delta)
        ttnn.deallocate(poisoned)
        delta[0, 0, r, c] = 0.0
        assert fired, (
            f"the mismatch marker did NOT fire with the {label} element of the compared region "
            f"([{r}, {c}]) poisoned by {peak:.4g} — the compare does not cover it, so it is vacuous"
        )


def assert_reference_is_live(reference):
    """A slice of zeros (or of NaNs) would compare equal to itself forever. It must not be either."""
    peak = device_marker_value(ttnn.max(ttnn.abs(reference)))
    assert peak == peak and peak not in (float("inf"), float("-inf")), f"reference is non-finite (max |x| = {peak})"
    assert peak > 0.0, "reference output is all zeros — the determinism compare would be vacuous"
    return peak


@pytest.mark.parametrize("case", _cases(), ids=lambda c: f"{c[3]}_e{c[0]}_c{c[1]}_n{c[2]}")
def test_determinism(device, case):
    emb, capacity, count, input_format = case
    overrides = assert_shipped_configuration()
    assert ITERS >= 2, f"MOE_DET_ITERS must be >= 2 to compare anything, got {ITERS}"
    assert count > 0, "count == 0 leaves every output row undefined: there is nothing to compare"

    tt_x, tt_w, tt_counts, tt_idx = _build(emb, capacity, count, input_format, device)
    rows = written_rows(count)

    reference = None
    merged_marker = None
    per_iteration_markers = []

    for i in range(ITERS):
        out = moe_fused_swiglu(tt_x, tt_w[0], tt_w[1], tt_w[2], tt_counts, tt_idx, LOCAL_EXPERT_ID, core_grid=CORE_GRID)
        assert list(out.shape) == [1, 1, capacity, emb]
        region = defined_region(out, rows)

        if reference is None:
            reference = region
            peak = assert_reference_is_live(reference)
            assert_marker_can_fire(reference, peak, device)
            continue

        marker = device_tensors_mismatch_marker(reference, region)
        per_iteration_markers.append((i, marker))
        merged_marker = merge_device_mismatch_markers(merged_marker, marker)
        if region is not out:
            ttnn.deallocate(region)
        ttnn.deallocate(out)

    # ONE readback on the green path. Only if it fires do we pay for per-iteration detail.
    if device_marker_is_set(merged_marker):
        diverged = [i for i, marker in per_iteration_markers if device_marker_is_set(marker)]
        pytest.fail(
            f"moe_fused_swiglu is NOT deterministic at {input_format} emb={emb} capacity={capacity} "
            f"count={count}: iteration(s) {diverged} of {ITERS} differ from iteration 0 over the "
            f"{rows} written rows (knobs: {resolved_knobs()})"
        )

    print(
        f"[det] {input_format} emb={emb} cap={capacity} count={count}: {ITERS} runs bitwise identical "
        f"over rows[0:{rows}] x {emb}, max|out|={peak:.4g}, wplace=nd_shard, overrides={overrides or 'none'}"
    )


def test_determinism_failure_path_is_wired(device):
    """The marker MERGE and the per-iteration attribution, exercised on a planted divergence.

    `assert_marker_can_fire` proves `ne`/`max` fire; this proves the rest of the reporting path
    — that a difference in ONE iteration survives `ttnn.maximum` across all the others, and that
    re-reading the per-iteration markers names that iteration and not its neighbours. Without it,
    a green run above could mean "deterministic" or "the merge silently swallowed it".
    """
    emb, capacity, count, poisoned_iteration = 7168, 1024, 256, 1
    tt_x, tt_w, tt_counts, tt_idx = _build(emb, capacity, count, "bf16_rm", device)
    rows = written_rows(count)

    reference = None
    merged_marker = None
    per_iteration_markers = []
    for i in range(4):
        out = moe_fused_swiglu(tt_x, tt_w[0], tt_w[1], tt_w[2], tt_counts, tt_idx, LOCAL_EXPERT_ID, core_grid=CORE_GRID)
        region = defined_region(out, rows)
        if reference is None:
            reference = region
            peak = assert_reference_is_live(reference)
            continue
        if i == poisoned_iteration:
            delta = torch.zeros((1, 1, rows, emb), dtype=torch.bfloat16)
            delta[0, 0, rows // 2, emb // 2] = peak
            region = ttnn.add(
                region,
                ttnn.from_torch(
                    delta,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
                dtype=ttnn.bfloat16,
            )
        marker = device_tensors_mismatch_marker(reference, region)
        per_iteration_markers.append((i, marker))
        merged_marker = merge_device_mismatch_markers(merged_marker, marker)

    assert device_marker_is_set(merged_marker), "a planted divergence did NOT survive the marker merge"
    diverged = [i for i, marker in per_iteration_markers if device_marker_is_set(marker)]
    assert diverged == [
        poisoned_iteration
    ], f"attribution named iteration(s) {diverged}, expected [{poisoned_iteration}]"
