# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TensorOcean horizontal tracer flux on one Blackhole chip: run a version, check it against the float64
reference, time one model time step.

    python models/experimental/tensorocean/demo/demo.py --n 100 --levels 100 --version both
"""
import argparse

import ttnn

from models.experimental.tensorocean.tests.common import VERSIONS, check, time_per_step


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100, help="mesh is n x n cells (even)")
    ap.add_argument("--levels", type=int, default=100, help="depth levels")
    ap.add_argument("--version", choices=("optimized", "baseline", "both"), default="both")
    ap.add_argument("--reps", type=int, default=20, help="steps per timing sample (baseline uses at most 2)")
    a = ap.parse_args()
    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 << 20)
    try:
        for name in ("baseline", "optimized") if a.version == "both" else (a.version,):
            v = VERSIONS[name]
            m, s = check(v, device, a.n, a.levels)
            t = time_per_step(v, device, s, reps=a.reps if name == "optimized" else min(a.reps, 2))
            acc = ", ".join(f"{p}: pcc {r['pcc']:.12f} rms_rel {r['rms_rel']:.1e}" for p, r in zip(("even", "odd"), m))
            print(f"{name:9s} N={a.n} L={a.levels}: {t * 1e3:10.3f} ms per step   ({acc})")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
