# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Driver for the I4 scaler bake-off. Fed to `scripts/tt-probe.sh rms_norm`.

    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \
    TT_METAL_PROFILER_CPP_POST_PROCESS=1 timeout 900 scripts/tt-probe.sh rms_norm <<'EOF'
    from ttnn.operations.rms_norm.perf_experiments.scaler_offpath.driver import main
    main(variants=[...], cores=110, c=2)
    EOF

ONE fresh dispatch per (variant, regime) — device kernel time has no warm-up
transient, so a trial loop would just re-measure the same number.
"""

from __future__ import annotations

import os
import shutil

import ttnn

from . import bench


def main(variants=None, cores=110, c=2, w_true=None, tag=None, keep_log=True, mask=False):
    variants = variants or list(bench.VARIANTS)
    device = ttnn.open_device(device_id=0)
    logdir = os.path.join(os.environ.get("TT_METAL_HOME", "."), "generated", "profiler", ".logs")
    rows = []
    try:
        for v in variants:
            ns, st = bench.run(device, v, n_cores=cores, c_tiles=c, w_true=w_true, mask=mask)
            rows.append((v, ns, st))
            print(
                f"I4 cores={cores} c={c} mask={int(mask)} variant={v:<17} ns={ns} "
                f"max_ratio_dev={st['max_ratio_dev']:.4g} pcc={st['pcc']:.6f} "
                f"cols1_31_absmax={st['cols_1_31_absmax']:.4g} nonfinite={st['cols_1_31_nonfinite']}",
                flush=True,
            )
        base = dict((v, ns) for v, ns, _ in rows).get(bench.BASELINE)
        if base:
            print("I4 SUMMARY " + f"cores={cores} c={c} mask={int(mask)}", flush=True)
            for v, ns, _ in rows:
                print(f"  {v:<17} {ns:>9.0f} ns   delta={ns - base:+8.0f}   x{base / ns:.3f}", flush=True)
        if keep_log:
            src = os.path.join(logdir, "profile_log_device.csv")
            if os.path.exists(src):
                name = tag or f"i4_c{c}_n{cores}"
                dst = os.path.join(logdir, f"zones_{name}.csv")
                shutil.copyfile(src, dst)
                print(f"I4 zones -> {dst} ({os.path.getsize(dst)} bytes)", flush=True)
    finally:
        ttnn.close_device(device)
    return rows
