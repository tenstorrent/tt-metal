# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""SEQUENCE-LENGTH SWEEP — one device-kernel duration per token count, 32 -> 5120 step 32.

    MOE_SWEEP_FORMATS=bf16_rm scripts/run_safe_pytest.sh --profile <this file>

The swept axis is `count`, the number of REAL tokens routed to the local expert. That is this op's
sequence length: `capacity` is only the allocated slot count (SUPPORTED[capacity] = {1024, 2048,
5120}), and `count` is what the kernels actually work on. `count` is DEVICE-resident, so every point
in the sweep runs the SAME compiled program — the whole curve is one profiled session with one JIT
build, and the only thing that changes between points is the content of the 256-entry `counts`
tensor.

Because the case list cannot be recovered from the profiler CSV (count lives in device memory, not
in a tensor shape), the test writes a MANIFEST listing every dispatch in issue order to
`MOE_SWEEP_MANIFEST`. `perf_experiments/parse_seqlen_sweep.py` zips that manifest against the CSV's
moe_fused_swiglu rows sorted by GLOBAL CALL COUNT, and refuses to report if the two lengths
disagree. Correctness is not asserted here beyond shape — that is the golden suite's job.
"""

import json
import os

import pytest
import torch

import ttnn

from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_helpers import (
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
#: N and the weight dtype are harness variables now — the op generalizes over both, and the
#: read-bytes denominator below must follow them or the reported utilisation silently lies.
HIDDEN = int(os.environ.get("MOE_HIDDEN", 2048))
WEIGHT_DTYPE = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}[
    os.environ.get("MOE_WDTYPE", "bfp4")
]
W_TILE = ttnn.tile_size(WEIGHT_DTYPE)
BFP4_TILE = 576
NUM_GLOBAL_EXPERTS, NUM_LOCAL_EXPERTS, LOCAL_EXPERT_ID, GLOBAL_EXPERT_ID = 256, 8, 3, 137

_FORMATS = {"bf16_rm": (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT), "bfp8_tile": (ttnn.bfloat8_b, ttnn.TILE_LAYOUT)}

EMB = int(os.environ.get("MOE_SWEEP_EMB", "7168"))
CAPACITY = int(os.environ.get("MOE_SWEEP_CAPACITY", "5120"))
#: 32 -> CAPACITY step 32: every tile-row count the capacity can hold.
STEP = int(os.environ.get("MOE_SWEEP_STEP", "32"))
#: Repeats per point. The parser reports the median and the min/max spread across them.
REPS = int(os.environ.get("MOE_SWEEP_REPS", "3"))
WARMUP = int(os.environ.get("MOE_SWEEP_WARMUP", "2"))
#: Dispatches between `ttnn.ReadDeviceProfiler` drains. LOAD-BEARING under `--profile`: the op emits
#: up to ~125 zone records per core per dispatch and the device-side profiler DRAM buffer holds 12000
#: per core, so a 480-dispatch session overflows it (~96 ops in) and tracy then ABORTS report
#: generation with "Device data missing: Op N not present in cpp_device_perf_report.csv" — no CSV at
#: all, not a partial one. Draining once per point keeps the high-water mark at ~3x125 records.
READ_PROFILER_EVERY = int(os.environ.get("MOE_SWEEP_PROFILER_READ_EVERY", "3"))
MANIFEST = os.environ.get("MOE_SWEEP_MANIFEST", "/tmp/moe_seqlen_sweep_manifest.json")

#: Inclusive count window, so the sweep can be CHUNKED across sessions. Load-bearing under
#: `--profile` for a reason that has nothing to do with the device: tracy's HOST-side
#: `process_ops_logs` holds the whole trace in pandas frames and needs ~50 MB of RSS per profiled
#: dispatch, so one session covering all 160 counts x 3 reps x 2 formats gets OOM-killed (measured:
#: 964 dispatches -> 50 GB -> SIGKILL) even though every dispatch ran clean on device. Keep a session
#: near ~150 dispatches and merge the chunks in the parser.
LO = int(os.environ.get("MOE_SWEEP_LO", "0"))
HI = int(os.environ.get("MOE_SWEEP_HI", str(CAPACITY)))

#: Explicit count list, overriding the LO..HI range — for a coarse re-measurement at a few M values.
_EXPLICIT = os.environ.get("MOE_SWEEP_COUNTS", "").strip()
COUNTS = (
    [int(c) for c in _EXPLICIT.split(",") if c.strip()]
    if _EXPLICIT
    else [c for c in range(STEP, CAPACITY + 1, STEP) if LO <= c <= HI]
)

#: WEIGHT PLACEMENT, the axis PERF 12 turns on. `interleaved` = plain DRAM_MEMORY_CONFIG, which makes
#: `nd_shard_n_tiles` return 0 and the reader emit the uncoalesced one-request-per-tile weight stream;
#: `nd_shard` = the DRAM ND shard `weight_memory_configs()` asks for, one K-row of a core's N slice per
#: shard, so the same bytes arrive as one request per K-row. This is the CALLER's choice, not a knob:
#: the op reads whatever width it is handed. The shard width depends on HGROUPS, so `MOE_GRID` is
#: threaded into BOTH the placement and the op call — a mismatch would silently build wrong shards.
WPLACES = [p.strip() for p in os.environ.get("MOE_SWEEP_WPLACE", "interleaved").split(",") if p.strip()]

#: Pass the TIGHT `input_m_tiles` bound (ceil(count/32)) rather than defaulting it to capacity//TILE.
TIGHT_MT = int(os.environ.get("MOE_SWEEP_MT", 0))

# Truncate at IMPORT (once per session), not per test: the two format tests share one session and
# must APPEND to one manifest, but a manifest left behind by a previous session must never survive
# into this one — stale entries would shift the whole CSV mapping by however many they are.
if os.path.exists(MANIFEST):
    os.remove(MANIFEST)


def _formats():
    return [f.strip() for f in os.environ.get("MOE_SWEEP_FORMATS", "bf16_rm,bfp8_tile").split(",") if f.strip()]


def read_bytes(count, emb, input_format):
    """DRAM bytes the op must read: three bfp4 weight sets + ONE read of the real tokens."""
    weights = 3 * (emb * HIDDEN // 1024) * W_TILE
    if input_format == "bf16_rm":
        return weights + count * emb * 2.0
    return weights + ((count + TILE - 1) // TILE) * TILE * emb * 1.0625


def _idx_table(device):
    idx = torch.tensor([(11 + 37 * i) % NUM_GLOBAL_EXPERTS for i in range(NUM_LOCAL_EXPERTS)], dtype=torch.int32)
    idx[LOCAL_EXPERT_ID] = GLOBAL_EXPERT_ID
    return ttnn.from_torch(
        idx, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _counts_tensor(count, device):
    counts = torch.zeros(NUM_GLOBAL_EXPERTS, dtype=torch.int32)
    counts[GLOBAL_EXPERT_ID] = count
    return ttnn.from_torch(
        counts, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _build_static(emb, capacity, input_format, wplace, device):
    """x and the three weight sets — built ONCE for the whole sweep.

    x is `randn` over ALL `capacity` rows, not zero-padded past a count: one tensor has to serve every
    point of the sweep, so it cannot carry a per-count sentinel the way the Perf-1/Perf-2 harnesses
    do. Nothing here depends on the phantom rows' contents — this file measures time and asserts only
    the output shape; whether the phantom rows are correctly ignored is the golden suite's job.
    """
    torch.manual_seed(42)
    x = torch.randn((1, 1, capacity, emb), dtype=torch.float32)
    dt, lay = _FORMATS[input_format]
    tt_x = ttnn.from_torch(
        x.to(torch.bfloat16), dtype=dt, layout=lay, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    del x
    if wplace == "nd_shard":
        gate_up_mc, down_mc = weight_memory_configs(device, emb, HIDDEN, core_grid=CORE_GRID)
    elif wplace == "shard_tall":
        # A NON-PREFERRED but still coalescible shard: same N extent, four tile-rows tall. Every
        # K-row read is still one transaction (consecutive n stay contiguous at any height), so
        # this isolates the bank-rotation half of the placement from the request-size half.
        gate_up_mc, down_mc = weight_memory_configs(device, emb, HIDDEN, core_grid=CORE_GRID, shard_height_tiles=4)
    elif wplace == "interleaved":
        gate_up_mc = down_mc = ttnn.DRAM_MEMORY_CONFIG
    else:
        raise ValueError(f"unknown MOE_SWEEP_WPLACE {wplace!r}")
    tt_w = [
        ttnn.from_torch(
            torch.randn(s, dtype=torch.bfloat16),
            dtype=WEIGHT_DTYPE,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=mc,
        )
        for s, mc in (((emb, HIDDEN), gate_up_mc), ((emb, HIDDEN), gate_up_mc), ((HIDDEN, emb), down_mc))
    ]
    # SELF-VERIFYING: the placement is the whole point of this axis, and an interleaved tensor is
    # silently CORRECT (just uncoalesced), so a config that failed to shard would produce a plausible
    # number attributed to the wrong path. `nd_shard_n_tiles` is the exact predicate the reader uses.
    widths = [nd_shard_n_tiles(w) for w in tt_w]
    if wplace.startswith("shard") or wplace == "nd_shard":
        assert all(w > 0 for w in widths), f"asked for nd_shard but reader sees interleaved: {widths}"
    else:
        assert all(w == 0 for w in widths), f"asked for interleaved but reader sees shards: {widths}"
    print(f"[sweep] wplace={wplace} reader shard widths (N tiles/shard, 0=interleaved) = {widths}", flush=True)
    return tt_x, tt_w


@pytest.mark.parametrize("wplace", WPLACES)
@pytest.mark.parametrize("input_format", _formats())
def test_seqlen_sweep(device, input_format, wplace):
    tt_x, tt_w = _build_static(EMB, CAPACITY, input_format, wplace, device)
    tt_idx = _idx_table(device)

    # One counts tensor per point, all uploaded up front: the sweep loop then contains nothing but
    # the op itself, so no host-side allocation lands between two dispatches of the same point.
    tt_counts = {c: _counts_tensor(c, device) for c in COUNTS}

    manifest = []
    if os.path.exists(MANIFEST):
        manifest = json.load(open(MANIFEST))

    def dispatch(count, rep, warmup):
        # `MOE_SWEEP_MT=1` passes the tight host-time M bound instead of letting it default to
        # `capacity // TILE`. This is the CALLER promising `count <= input_m_tiles * 32`, which is the
        # condition feature_spec records for the count-128/256 target bases ("had the count known at
        # COMPILE time"). It is a bound, not a branch on the counts' CONTENTS, so it does not break the
        # device-resident-count contract — and it is the only host-time handle on M the op offers.
        m_t = -(-count // TILE) if TIGHT_MT else None
        out = moe_fused_swiglu(
            tt_x,
            tt_w[0],
            tt_w[1],
            tt_w[2],
            tt_counts[count],
            tt_idx,
            LOCAL_EXPERT_ID,
            input_m_tiles=m_t,
            core_grid=CORE_GRID,
        )
        assert list(out.shape) == [1, 1, CAPACITY, EMB]
        ttnn.deallocate(out)
        manifest.append(
            {
                "format": input_format,
                "wplace": wplace,
                "grid": os.environ.get("MOE_GRID", "") or "full",
                "emb": EMB,
                "capacity": CAPACITY,
                "count": count,
                "rep": rep,
                "warmup": warmup,
                "read_bytes": read_bytes(count, EMB, input_format),
            }
        )

    for w in range(WARMUP):
        dispatch(COUNTS[len(COUNTS) // 2], w, True)
    ttnn.ReadDeviceProfiler(device)

    since_read = 0
    for count in COUNTS:
        for rep in range(REPS):
            dispatch(count, rep, False)
            since_read += 1
            if since_read >= READ_PROFILER_EVERY:
                ttnn.ReadDeviceProfiler(device)  # no-op when the profiler is off
                since_read = 0
        print(f"[sweep] {input_format} emb={EMB} cap={CAPACITY} count={count} done", flush=True)
    ttnn.ReadDeviceProfiler(device)

    json.dump(manifest, open(MANIFEST, "w"))
    print(f"[sweep] manifest: {MANIFEST} ({len(manifest)} dispatches)", flush=True)
