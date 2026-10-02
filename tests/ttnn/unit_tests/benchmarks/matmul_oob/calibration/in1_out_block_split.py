# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Calibration sweep for the 1D in1-mcast output block split (HeuristicBlocking::Tuned).

In 1D in1-mcast each core reads its own rows of A and writes its own output. With a single output block per core,
every output tile is written after the last K step, so none of the writes overlap compute; splitting the block
lets the writes overlap the next block's compute, but every extra block streams all of B (Kt x Nt tiles) again.

The grid: rows per core P, K and N in tiles, and B's format, on a full-grid 1D in1 layout with interleaved DRAM
tensors (M = P x cores tiles, so the layout fills the grid on any architecture). For each grid point it times the
v2 default selection (a check on the sweep, not fitting data) and the 1D in1 config at every block height dividing P (1 to P output blocks per core), with
in0_block_w and the subblock chosen by v2's rules for each block height, so only the split varies.

  python tests/ttnn/unit_tests/benchmarks/matmul_oob/calibration/in1_out_block_split.py            # sweep
  python tests/ttnn/unit_tests/benchmarks/matmul_oob/calibration/in1_out_block_split.py --report   # fit

The sweep writes data/<arch>/in1_out_block_split.csv (resumable); --report prints the fit, and fit() is what the
calibration test checks against HeuristicBlocking::Tuned::in1_split_min_block_work.
"""

import argparse
import csv
import math
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

ROWS_PER_CORE = (2, 4, 8, 12, 16, 32, 64)
K_TILES = (1, 2, 3, 4, 6, 8, 12, 16, 32)
N_TILES = (1, 2, 4, 8, 12, 16)
B_DTYPES = ("bf16", "bfp8")
TILE_BYTES = {"bf16": 2048, "bfp8": 1088, "bfp4": 576, "fp32": 4096}
FIELDS = [
    "arch",
    "P",
    "Kt",
    "Nt",
    "b_dtype",
    "variant",
    "blocks",
    "out_block_h",
    "k",
    "subblock",
    "status",
    "device_ns",
]


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def subblock(bh, bw, max_area, two_wide):
    """v2's subblock rule (HeuristicSubblock): largest area, then two tiles or more on each side, then wider."""
    best = (1, 1)
    for h in range(1, min(bh, max_area) + 1):
        if bh % h:
            continue
        for w in range(min(bw, max_area // h), 0, -1):
            if bw % w:
                continue
            area, best_area = h * w, best[0] * best[1]
            tw, best_tw = min(h, w) >= 2, min(best) >= 2
            tie = tw if (two_wide and tw != best_tw) else w > best[1]
            if area > best_area or (area == best_area and tie):
                best = (h, w)
            break
    return best


def k_depth(Kt, bh):
    """v2's K depth rule for 1D in1 (HeuristicBlocking defaults): the deepest divisor of Kt within the caps."""
    cap = min(8, Kt // 2 if Kt >= 2 else Kt, max(2, 8 // bh))
    return max(d for d in divisors(Kt) if d <= max(1, cap))


def out_path(arch):
    return HERE / "data" / arch / "in1_out_block_split.csv"


def sweep(args):
    import ttnn
    from run_suite import CaseRun
    from suite import Case

    device = ttnn.open_device(device_id=args.device_id)
    arch = str(device.arch()).split(".")[-1].lower()
    grid = device.compute_with_storage_grid_size()
    cores = grid.x * grid.y
    path = out_path(arch)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Resume per variant: (P, Kt, Nt, b_dtype, variant, out_block_h) already timed
    done = set()
    if path.exists():
        done = {
            (int(r["P"]), int(r["Kt"]), int(r["Nt"]), r["b_dtype"], r["variant"], r["out_block_h"])
            for r in csv.DictReader(open(path))
        }
    run_args = types.SimpleNamespace(warmup=2, iters=10, pcc_threshold=0.99, pcc_max_flops=0)
    seen = set()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not done:
            writer.writeheader()
        for P in ROWS_PER_CORE:
            for Kt in K_TILES:
                for Nt in N_TILES:
                    for b_dtype in B_DTYPES:
                        point = (P, Kt, Nt, b_dtype)
                        heights = sorted(divisors(P), reverse=True)  # every block height the selector can pick
                        todo = [bh for bh in heights if (*point, "in1", str(bh)) not in done]
                        need_default = (*point, "default v2", "") not in done
                        if not todo and not need_default:
                            continue
                        case = Case(
                            name=f"in1_P{P}_K{Kt}_N{Nt}_{b_dtype}",
                            a_shape=(P * cores * 32, Kt * 32),
                            b_shape=(Kt * 32, Nt * 32),
                            tier="calibration",
                            b_dtype=b_dtype,
                        )
                        row = dict(arch=arch, P=P, Kt=Kt, Nt=Nt, b_dtype=b_dtype)
                        run = CaseRun(case, device, run_args, seen)
                        try:
                            if need_default:  # a check on the sweep's replay of the rule, not fitting data
                                r = run.measure("v2")
                                writer.writerow(
                                    {
                                        **row,
                                        "variant": "default v2",
                                        "status": r["status"],
                                        "device_ns": r.get("device_ns", ""),
                                    }
                                )
                            for bh in todo:
                                k = k_depth(Kt, bh)
                                sh, sw = subblock(bh, Nt, 8, TILE_BYTES[b_dtype] >= TILE_BYTES["bf16"])
                                config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                                    compute_with_storage_grid_size=grid,
                                    in0_block_w=k,
                                    out_subblock_h=sh,
                                    out_subblock_w=sw,
                                    out_block_h=bh,
                                    out_block_w=Nt,
                                    per_core_M=P,
                                    per_core_N=Nt,
                                    fuse_batch=True,
                                    fused_activation=None,
                                    mcast_in0=False,
                                )
                                r = run.measure("oob", config)
                                writer.writerow(
                                    {
                                        **row,
                                        "variant": "in1",
                                        "blocks": P // bh,
                                        "out_block_h": bh,
                                        "k": k,
                                        "subblock": f"{sh}x{sw}",
                                        "status": r["status"],
                                        "device_ns": r.get("device_ns", ""),
                                    }
                                )
                        finally:
                            run.close()
                        f.flush()
                        print(f"done P={P} Kt={Kt} Nt={Nt} {b_dtype}", flush=True)
    ttnn.close_device(device)


# The thresholds fit() considers: tile products of work per output block (out_block_h * Nt * Kt), powers of 2 and
# their midpoints from 1 to 4096, and "never split"
CANDIDATES = tuple(sorted({2**i for i in range(13)} | {3 * 2**i for i in range(11)})) + (math.inf,)


def load(path):
    """Grid points with more than one block count timed: {(P, Kt, Nt, b_dtype): {out_block_h: ns}}. The v2
    default selection's rows are a check on the sweep, not fitting data."""
    points = {}
    for r in csv.DictReader(open(path)):
        if r["variant"] == "in1" and r["status"] == "ok" and r["device_ns"]:
            key = (int(r["P"]), int(r["Kt"]), int(r["Nt"]), r["b_dtype"])
            points.setdefault(key, {})[int(r["out_block_h"])] = float(r["device_ns"])
    return {k: v for k, v in points.items() if len(v) > 1}


def choose(heights, Kt, Nt, w):
    """The split rule: the shortest block whose work out_block_h * Nt * Kt is at least w (one block when none is)"""
    ok = [bh for bh in heights if bh * Nt * Kt >= w]
    return min(ok) if ok else max(heights)


def score(points, w):
    """(geomean regret, worst regret) of threshold w over the grid; regret is the chosen variant's time over the
    fastest variant timed at that grid point"""
    regret = [
        times[choose(sorted(times), Kt, Nt, w)] / min(times.values()) for (P, Kt, Nt, dt), times in points.items()
    ]
    return math.exp(sum(map(math.log, regret)) / len(regret)), max(regret)


def fit(path):
    """HeuristicBlocking::Tuned::in1_split_min_block_work from the sweep data at `path`.

    Criterion: the lowest geomean regret against the fastest block count timed at each grid point, then the lowest
    worst-case regret, then the smallest threshold. Regrets are compared to 0.1%, below the timing noise. Returns
    the value and the range of thresholds within 0.3% of its geomean regret."""
    points = load(path)
    scored = {w: score(points, w) for w in CANDIDATES}
    best = min(CANDIDATES, key=lambda w: (round(scored[w][0], 3), round(scored[w][1], 3), w))
    near = [w for w in CANDIDATES if scored[w][0] <= scored[best][0] + 0.003]
    return {
        "value": best,
        "range": (min(near), max(near)),
        "points": len(points),
        "score": scored[best],
        "never_split": scored[math.inf],
    }


def report(args):
    path = out_path(args.arch)
    points = load(path)
    print(f"{args.arch}: {len(points)} grid points with more than one block count")
    for w in CANDIDATES:
        geo, worst = score(points, w)
        print(f"  work >= {w:<6} regret geomean {geo:.3f}  worst {worst:.3f}")
    result = fit(path)
    print(
        f"fit: {result['value']} tile products per block (within 0.3%: {result['range'][0]}..{result['range'][1]}); "
        f"regret geomean {result['score'][0]:.3f}, worst {result['score'][1]:.3f}; "
        f"never splitting: geomean {result['never_split'][0]:.3f}, worst {result['never_split'][1]:.3f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--arch", default="wormhole_b0", help="--report: which data to fit")
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()
    report(args) if args.report else sweep(args)


if __name__ == "__main__":
    main()
