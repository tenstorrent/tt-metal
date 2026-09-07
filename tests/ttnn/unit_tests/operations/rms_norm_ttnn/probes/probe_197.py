# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Driver for the isolated `bank_coalesced_txn` DM bake-off.  One fresh run per variant
# (device kernel time has no warm-up transient); correctness gated on every variant whose
# reader/writer permutations compose to the identity (all the payload-carrying ones).

import sys, os

# absolute: tt-probe.sh copies this script into tests/.../probes/, so __file__ is not here
sys.path.insert(
    0,
    "/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/rms_norm_ttnn_run1/tt-metal"
    "/ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/bank_coalesced_txn",
)

import ttnn
import bank_bench as B


def _nb(device):
    v = device.dram_grid_size()
    return v.x * v.y, f"dram_grid_size()={v.x}x{v.y}"


CASES = {
    # name: (shape, WT_CHUNK)  -- WT_CHUNK mirrors the op's plan for that shape
    "FOCUS_8192x2304": ((1, 1, 8192, 2304), 72),
    "w1024_8192": ((1, 1, 8192, 1024), 32),
    "w7168_8192": ((1, 1, 8192, 7168), 56),
    "w1024_32": ((1, 1, 32, 1024), 32),
    # crossover ladder: constant per-core work (ONE tile-row of 32 tiles), 1 -> 110
    # active cores.  Locates where the DRAM stops being the binding constraint.
    "w1024_128": ((1, 1, 128, 1024), 32),
    "w1024_352": ((1, 1, 352, 1024), 32),
    "w1024_1024": ((1, 1, 1024, 1024), 32),
    "w1024_3520": ((1, 1, 3520, 1024), 32),
    "w2304_2304": ((1, 1, 2304, 2304), 72),
}

# which variants to run: "sweep" (the PPT / rot menu, rw only) or "halves" (read/write split)
GROUPS = {
    "sweep": ["base_rw"] + [f"coal{p}{r}" for p in (1, 2, 3, 4, 6, 8) for r in ("", "_rot")],
    # attribute the regression to a half: same structure, one half's payload ablated
    "halves": (
        ["floor", "base_r", "base_w", "base_rw"] + [f"coal{p}_rot{h}" for p in (1, 2, 8) for h in ("_r", "_w", "")]
    ),
    # the domain sweep: one representative coalesce ladder per regime
    "domain": ["base_rw"] + [f"coal{p}_rot" for p in (1, 2, 4, 8)],
    "stagger": ["base_rw", "stag", "coal1_rot", "coal2_rot"],
    "stagger_halves": ["base_r", "base_w", "base_rw", "stag_r", "stag_w", "stag"],
}


def main():
    # tt-probe.sh feeds the script on stdin, so configuration comes from the environment
    argv = (os.environ.get("BC_ARGS") or " ".join(sys.argv[1:])).split()
    group = "sweep"
    if argv and argv[0] in GROUPS:
        group, argv = argv[0], argv[1:]
    extra = []
    if argv and argv[0].startswith("+"):
        extra = argv[0][1:].split(",")
        argv = argv[1:]
    case_names = argv or ["FOCUS_8192x2304"]

    device = ttnn.open_device(device_id=0)
    try:
        nb, how = _nb(device)
        cg = device.compute_with_storage_grid_size()
        print(f"RESULT DEVICE NUM_DRAM_BANKS={nb} ({how})  compute_grid={cg.x}x{cg.y}")
        grid = (cg.x, cg.y)
        variants = GROUPS[group] + extra
        for cn in case_names:
            shape, wtc = CASES[cn]
            WT = shape[-1] // 32
            print(f"RESULT === {cn} shape={shape} WT={WT} WT_CHUNK={wtc} NB={nb} grid={grid}")
            res = {}
            for v in variants:
                ns, ok = B.measure(device, shape, wtc, v, grid, nb)
                res[v] = ns
                _, _, _, ppt, _ = B.VARIANTS[v]
                q = (wtc + nb - 1) // nb
                per_row = wtc if B.VARIANTS[v][0] != 1 else min(nb, wtc) * ((q + (ppt or q) - 1) // (ppt or q))
                print(f"RESULT {cn:16s} {v:12s} {ns:12.0f} ns  txn/row={per_row:4d}  exact={ok}")
            if "base_rw" in res:
                b = res["base_rw"]
                for v in variants:
                    if v != "base_rw":
                        print(f"RESULT {cn:16s} SPEEDUP {v:12s} {b / res[v]:.3f}x")
    finally:
        ttnn.close_device(device)


main()
