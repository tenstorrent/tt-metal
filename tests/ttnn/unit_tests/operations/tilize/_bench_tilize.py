# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""tilize perf bench — measurement only, NO correctness assertions.

Underscore-prefixed and deliberately NOT in `feature_spec.INPUTS`: the golden
cells are tiny (they are the correctness gate and must stay fast) and far too
small to be bandwidth-bound, so they cannot measure the thing Track A optimizes.
This file carries the grid-filling shapes instead.

    # everything (all shapes x all arms)
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/tilize/_bench_tilize.py

    # a subset
    TB_SHAPES=square,wide_short TB_ARMS=base,lever_b7_barrier_per_read \\
        scripts/run_safe_pytest.sh --run-all tests/.../_bench_tilize.py

Metric: `DEVICE KERNEL DURATION [ns]` read from the IN-PROCESS device profiler
(`ttnn.ReadDeviceProfiler` + `ttnn.get_latest_programs_perf_data`), so no Tracy
CSV parsing and no host wall-clock is involved. Results are also written to
`generated/tilize_bench/<name>.json` so the changelog table can be generated
rather than typed.

**Every arm is a `levers=dict(...)` forcing arm**, so each lever's counterfactual
stays re-runnable from here instead of being an ad-hoc kernel edit — see
`ttnn/ttnn/operations/tilize/lever_ledger.json`. The `ablate_*` arms are the
/perf-measure ablation variants: they stub ONE payload while keeping every CB
reserve/push/wait/pop and every loop trip count, so their output is wrong BY
DESIGN. That is why this file asserts nothing about values (correctness lives in
the golden suite and test_tilize_debug.py, which also proves every non-stub lever
arm still computes the right answer).

Shape regimes — both grid regimes are mandatory, because a bench that measures
only the square reports healthy while a height-only split strands the wide-short
case on one core:

  square      [1,1,2048,2048]  grid-filling, several blocks/core -> per-core DRAM efficiency
  wide_short  [1,1,32,16384]   nt_h == 1                         -> does the split FILL the grid
  tall_narrow [1,1,2048,64]    n_wchunks == 1                     -> pure-height-split degenerate
  smallest    [1,1,32,64]      1 core, 1 block                    -> master.md B0: every
                               per-core-overhead lever must be counterfactualed HERE too

(The exact block/core counts move with TARGET_READ_BYTES, which is the point of
the knob; the table the bench prints reports the live numbers per shape.)
"""

import json
import os
import pathlib

# The in-process device profiler must be enabled BEFORE the device opens.
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import torch

import ttnn
from ttnn.operations.tilize.tilize import _dispatch
from ttnn.operations.tilize.tilize_program_descriptor import blocking, plan_cores

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# Device kernel duration has no warm-up transient, so this is not a "trial loop":
# WARMUP launches exist only to get the program into the cache (so the measured
# launch is not the compiling one), and TRIALS is small on purpose.
N_WARMUP = int(os.environ.get("TB_WARMUP", "2"))
N_TRIALS = int(os.environ.get("TB_TRIALS", "3"))

SHAPES = {
    "square": (1, 1, 2048, 2048),
    "wide_short": (1, 1, 32, 16384),
    "tall_narrow": (1, 1, 2048, 64),
    "smallest": (1, 1, 32, 64),
    # The smallest shape the Phase-0 op can RUN (1 core, 1 block, 1 tile). The
    # smallest shape in feature_spec.INPUTS is [1,1,30,32], but that is a
    # pad_mode="auto" cell which Phase 0 refuses, so this is its tile-aligned
    # counterpart with the identical per-core geometry (nt_h=1, Wt=1, 1 block).
    # master.md B0's per-core-overhead levers are counterfactualed here.
    "smallest_aligned": (1, 1, 32, 32),
}

# arm name -> the kwargs handed to `_dispatch`. `base` is the shipped
# configuration (`levers=dict()` == DEFAULT_LEVERS); every other arm flips
# exactly ONE lever off (or stubs one payload for ablation). Keeping every arm in
# the `levers=dict(<knob>=<value>)` shape is what makes each counterfactual
# re-runnable — `eval/verify_levers.py` scans this file for exactly that form.
ARMS = {
    # ---- baseline -------------------------------------------------------
    "base": dict(levers=dict()),
    # ---- distribution levers (A0 / A1) ----------------------------------
    "base_singlecore": dict(levers=dict(multicore=0)),  # A0 off-arm
    "lever_a1_width_split_off": dict(levers=dict(width_split=0)),
    "lever_a1_row_wise_off": dict(levers=dict(row_wise=0)),
    # ---- transaction-shape levers (B5 / B6 / B7 / B9) -------------------
    "lever_b6_read_128": dict(levers=dict(target_read_bytes=128)),
    "lever_b6_read_256": dict(levers=dict(target_read_bytes=256)),
    "lever_b6_read_512": dict(levers=dict(target_read_bytes=512)),
    "lever_b6_read_2048": dict(levers=dict(target_read_bytes=2048)),
    "lever_b6_read_4096": dict(levers=dict(target_read_bytes=4096)),
    "lever_b7_barrier_per_read": dict(levers=dict(barrier_per_block=0)),
    "lever_b5_face_writes": dict(levers=dict(coalesce_writes=0)),
    "lever_b9_noc_swap": dict(levers=dict(noc_split=0)),
    # ---- buffering lever (C16) ------------------------------------------
    "lever_c16_depth1": dict(levers=dict(double_buffer=0)),
    # ---- ablation arms (classification; output wrong by design) ---------
    "ablate_compute": dict(levers=dict(stub_compute=1)),
    "ablate_read": dict(levers=dict(stub_read=1)),
    "ablate_write": dict(levers=dict(stub_write=1)),
    "ablate_read_compute": dict(levers=dict(stub_read=1, stub_compute=1)),
    "ablate_all": dict(levers=dict(stub_read=1, stub_compute=1, stub_write=1)),
}

_OUT_DIR = pathlib.Path("generated/tilize_bench")


def _selected(env_name, universe):
    raw = os.environ.get(env_name)
    if not raw:
        return list(universe)
    names = [n.strip() for n in raw.split(",") if n.strip()]
    for name in names:
        if name not in universe:
            raise ValueError(f"{env_name}: unknown entry {name!r}; known: {list(universe)}")
    return names


def _bench_input(shape, device, dtype=ttnn.bfloat16):
    torch.manual_seed(0)
    return ttnn.from_torch(
        torch.randn(shape).bfloat16(),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if entry is None:
                continue
            total += float(entry.duration)
            found = True
    return total if found else None


def _measure_ns(device, run_fn):
    for _ in range(N_WARMUP):
        run_fn()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # flush the warm-up window
    for _ in range(N_TRIALS):
        run_fn()
    ttnn.synchronize_device(device)
    total = _read_kernel_ns(device)
    return None if total is None else total / N_TRIALS


def test_bench(device):
    """Measure every selected (shape, arm) and print the table. Perf is evidence,
    never pass/fail — the only assertion is that the profiler produced numbers."""
    shape_names = _selected("TB_SHAPES", SHAPES)
    arm_names = _selected("TB_ARMS", ARMS)
    grid = device.compute_with_storage_grid_size()

    results = {}
    for shape_name in shape_names:
        shape = SHAPES[shape_name]
        tt_input = _bench_input(shape, device)
        for arm in arm_names:
            kwargs = ARMS[arm]
            run_fn = lambda kw=kwargs, t=tt_input: _dispatch(t, use_multicore=True, **kw)
            ns = _measure_ns(device, run_fn)
            assert ns is not None, f"profiler produced no data for {shape_name}/{arm}"
            results[(shape_name, arm)] = ns

    # ---- report ----
    lines = [
        "",
        "=== tilize bench — DEVICE KERNEL DURATION [ns], "
        f"grid={grid.x}x{grid.y}={grid.x * grid.y}, {N_TRIALS} launches averaged ===",
    ]
    payload = {}
    for shape_name in shape_names:
        shape = SHAPES[shape_name]
        blk = blocking(list(shape), 32, 2)
        cores, _all_cores, _per_core = plan_cores(device, blk["total_blocks"], use_multicore=True)
        elem_bytes = 2
        total_bytes = 2 * torch.tensor(shape).prod().item() * elem_bytes  # read + write
        base = results.get((shape_name, "base"))
        lines += [
            "",
            f"  {shape_name} {tuple(shape)}: nt_h={blk['nt_h']} Wt={blk['Wt']} "
            f"WT_BLOCK={blk['wt_block']} n_wchunks={blk['n_wchunks']} "
            f"blocks={blk['total_blocks']} cores={len(cores)} dram_bytes={total_bytes}",
            f"    {'arm':<28} {'ns':>12} {'vs base':>9} {'GB/s':>8}",
        ]
        for arm in arm_names:
            ns = results[(shape_name, arm)]
            ratio = f"{ns / base:.3f}x" if base else "-"
            gbps = total_bytes / ns if ns else 0.0
            lines.append(f"    {arm:<28} {ns:>12.1f} {ratio:>9} {gbps:>8.1f}")
            payload[f"{shape_name}/{arm}"] = {
                "ns": ns,
                "vs_base": (ns / base) if base else None,
                "gbps": gbps,
                "shape": list(shape),
                "cores": len(cores),
                "blocks": blk["total_blocks"],
                "wt_block": blk["wt_block"],
            }
    print("\n".join(lines))

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = os.environ.get("TB_OUT", "latest")
    (_OUT_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"    -> {_OUT_DIR / (name + '.json')}")
