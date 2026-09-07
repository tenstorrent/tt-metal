# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Driver for the isolated `bank_coalesced_txn` DM bake-off.  One fresh run per variant
# (device kernel time has no warm-up transient); correctness gated on the two variants
# whose reader/writer permutations compose to the identity.

import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ttnn
import bank_bench as B


def _nb(device):
    for name in ("num_dram_channels", "dram_grid_size"):
        f = getattr(device, name, None)
        if f is None:
            continue
        try:
            v = f()
        except Exception:
            continue
        if hasattr(v, "x"):
            return v.x * v.y, f"{name}()={v.x}x{v.y}"
        return int(v), f"{name}()={v}"
    return None, "unknown"


CASES = {
    # name: (shape, WT_CHUNK)  -- WT_CHUNK mirrors the op's plan for that shape
    "FOCUS_8192x2304": ((1, 1, 8192, 2304), 72),
    "w1024_8192": ((1, 1, 8192, 1024), 32),
    "w7168_8192": ((1, 1, 8192, 7168), 56),
    "w1024_32": ((1, 1, 32, 1024), 32),
}

ORDER = ["floor", "base_r", "coal_r", "base_w", "coal_w", "base_rw", "coal_rw"]


def main():
    case_names = sys.argv[1:] or ["FOCUS_8192x2304"]
    device = ttnn.open_device(device_id=0)
    try:
        nb, how = _nb(device)
        cg = device.compute_with_storage_grid_size()
        print(f"RESULT DEVICE NUM_DRAM_BANKS={nb} ({how})  compute_grid={cg.x}x{cg.y}")
        grid = (cg.x, cg.y)
        for cn in case_names:
            shape, wtc = CASES[cn]
            WT = shape[-1] // 32
            print(f"RESULT === {cn} shape={shape} WT={WT} WT_CHUNK={wtc} NB={nb} grid={grid}")
            print(f"RESULT     txn/tile-row: baseline={wtc}  coalesced={min(wtc, nb)}")
            res = {}
            for v in ORDER:
                ns, ok = B.measure(device, shape, wtc, v, grid, nb)
                res[v] = ns
                print(f"RESULT {cn:18s} {v:9s} {ns:12.0f} ns   exact={ok}")
            b, c = res["base_rw"], res["coal_rw"]
            print(
                f"RESULT {cn:18s} SPEEDUP rw={b / c:.3f}x  r={res['base_r'] / res['coal_r']:.3f}x  "
                f"w={res['base_w'] / res['coal_w']:.3f}x"
            )
    finally:
        ttnn.close_device(device)


main()
