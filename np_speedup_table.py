#!/usr/bin/env python3
"""Clean fused-vs-non-fused speedup table for neighbor_pad_conv3d.

Same table shape as cglagovich/fused_rms_norm `test_bench`/`_print_table`, but driven by
**device-FW MIN** (tracy) rather than trace-mode wall: the NP perf harness is untraced and
host-dispatch-bound, so the wall makes fusion look slower while the real device win is hidden
(NP_CONV3D_FUSED.md §4e). Device-FW MIN is the metric the wiki trusts and is host-independent.

  fused    = NpConv3d device-FW MIN (the single fused op)
  nonfused = NeighborPadAsync MIN + Conv3d MIN (the two sequential standalone ops)
  speedup  = nonfused / fused   (>1.0 = fusion is faster on device)

Run from the repo root inside python_env, with the device free:
    python np_speedup_table.py 2x4        # 2x4 production shapes (real NP topology)
    python np_speedup_table.py 4x8mock    # 4x8 conv sizes on a 2x4 box (NP topology still 2x4)
For a real 4x8 mesh the perf test needs a (4,8) mesh param (see NP_CONV3D_FUSED.md §10).
"""
import csv
import glob
import os
import subprocess
import sys

import pandas as pd

PERF = "models/tt_dit/tests/models/wan2_2/test_neighbor_pad_conv3d_fused_perf.py" "::test_fused_vs_standalone_perf"

# id -> (C_in, C_out, T, H_dev, W_dev, deployed_scheme). H_dev/W_dev are per-device. The ids must match
# the parametrize ids in test_neighbor_pad_conv3d_fused_perf.py (this drives that test via -k).
SHAPES = {
    "2x4": [
        ("ltx_s0_conv_in_2x4", 128, 1024, 21, 17, 15, "standalone"),
        ("ltx_s0_up_2x4", 1024, 4096, 21, 17, 15, "standalone"),
        ("ltx_s1_up_2x4", 512, 4096, 39, 34, 30, "standalone"),
        ("ltx_s0_res_2x4", 1024, 1024, 21, 17, 15, "standalone"),
        ("ltx_s1_res_2x4", 512, 512, 39, 34, 30, "standalone"),
        ("ltx_s2_res_2x4", 512, 512, 75, 68, 60, "standalone"),
        ("ltx_s3_res_2x4", 256, 256, 147, 68, 60, "halo_last"),
        ("ltx_s3_chg_2x4", 256, 512, 147, 68, 60, "halo_last"),
        ("ltx_s4_res_2x4", 128, 128, 147, 136, 120, "halo_last"),
        ("ltx_s4_out_2x4", 128, 48, 147, 136, 120, "force_spatial"),
    ],
    # 4x8 per-device conv sizes on a 2x4 box — NP topology is still 2x4 (a proxy; real NP differs).
    "4x8mock": [
        ("ltx_s0_conv_in_4x8mock", 128, 1024, 21, 9, 8, "standalone"),
        ("ltx_s0_up_4x8mock", 1024, 4096, 21, 9, 8, "standalone"),
        ("ltx_s1_up_4x8mock", 512, 4096, 39, 17, 15, "halo_last"),
        ("ltx_s1_res_4x8mock", 512, 512, 39, 17, 15, "standalone"),
        ("ltx_s2_res_4x8mock", 512, 512, 75, 34, 30, "halo_last"),
        ("ltx_s3_res_4x8mock", 256, 256, 147, 34, 30, "halo_last"),
        ("ltx_s3_chg_4x8mock", 256, 512, 147, 34, 30, "halo_last"),
        ("ltx_s4_res_4x8mock", 128, 128, 147, 68, 60, "force_spatial"),
        ("ltx_s4_out_4x8mock", 128, 48, 147, 68, 60, "standalone"),
        ("ltx_ups_post_res_4x8mock", 1024, 1024, 21, 10, 8, "standalone"),
        ("ltx_ups_final_4x8mock", 1024, 128, 21, 10, 8, "standalone"),
    ],
    # Real 4x8 (32-chip mesh): same per-device sizes on a real (4,8) mesh. Only runs on 32-chip hardware
    # (the perf test skips these on an 8-chip box); here they report n/a.
    "4x8": [
        ("ltx_s0_conv_in_4x8", 128, 1024, 21, 9, 8, "standalone"),
        ("ltx_s0_up_4x8", 1024, 4096, 21, 9, 8, "standalone"),
        ("ltx_s1_up_4x8", 512, 4096, 39, 17, 15, "halo_last"),
        ("ltx_s1_res_4x8", 512, 512, 39, 17, 15, "standalone"),
        ("ltx_s2_res_4x8", 512, 512, 75, 34, 30, "halo_last"),
        ("ltx_s3_res_4x8", 256, 256, 147, 34, 30, "halo_last"),
        ("ltx_s3_chg_4x8", 256, 512, 147, 34, 30, "halo_last"),
        ("ltx_s4_res_4x8", 128, 128, 147, 68, 60, "force_spatial"),
        ("ltx_s4_out_4x8", 128, 48, 147, 68, 60, "standalone"),
        ("ltx_ups_post_res_4x8", 1024, 1024, 21, 10, 8, "standalone"),
        ("ltx_ups_final_4x8", 1024, 128, 21, 10, 8, "standalone"),
    ],
}


def _min_us(df, namecol, fwcol, op):
    sub = df[df[namecol].astype(str).str.contains(op, case=False, na=False)]
    d = pd.to_numeric(sub[fwcol], errors="coerce").dropna()
    return None if d.empty else d.min() / 1000.0


def _measure(shape_id):
    """Run one shape under tracy, return (fused_us, np_us, conv_us) device-FW MIN."""
    subprocess.run(
        ["python", "-m", "tracy", "-p", "-r", "-m", "pytest", PERF, "-k", shape_id, "-s"],
        capture_output=True,
        timeout=900,
    )
    csvs = glob.glob("generated/profiler/reports/**/ops_perf_results_*.csv", recursive=True)
    if not csvs:
        return None, None, None
    df = pd.read_csv(max(csvs, key=os.path.getmtime))
    namecol = "OP CODE" if "OP CODE" in df.columns else "OP TYPE"
    fwcol = "DEVICE FW DURATION [ns]"
    return (
        _min_us(df, namecol, fwcol, "NpConv3d"),
        _min_us(df, namecol, fwcol, "NeighborPadAsync"),
        _min_us(df, namecol, fwcol, "Conv3d"),
    )


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "2x4"
    shapes = SHAPES[which]
    rows = []
    for shape_id, c_in, c_out, t, h, w, scheme in shapes:
        fused, npm, convm = _measure(shape_id)
        nonfused = (npm + convm) if (npm is not None and convm is not None) else None
        speedup = (nonfused / fused) if (fused and nonfused) else None
        rows.append(
            dict(
                cid=shape_id,
                c_in=c_in,
                c_out=c_out,
                t=t,
                hw=f"{h}x{w}",
                fused=fused,
                np=npm,
                conv=convm,
                nonfused=nonfused,
                speedup=speedup,
                scheme=scheme,
            )
        )
        print(f"  {shape_id}: fused={fused} NP={npm} conv={convm} nonfused={nonfused} speedup={speedup}")

    title = f"neighbor_pad_conv3d: fused vs non-fused — device-FW MIN, BH-LB {which}"
    cid_w = max(len("config_id"), max(len(r["cid"]) for r in rows))
    # non-fused is broken out as standalone NeighborPadAsync (NP) + full-grid Conv3d, plus their sum.
    header = (
        f"{'config_id':<{cid_w}}  {'C_in':>5} {'C_out':>5} {'T':>4} {'HxW(dev)':>9} "
        f"{'fused us':>10} | {'NP us':>9} {'conv us':>9} {'NP+conv us':>11} {'speedup':>8}  {'deployed':<13}"
    )
    box = "=" * max(len(header), len(title))
    print("\n" + box + f"\n{title}\n" + box + f"\n{header}\n" + "-" * len(header))

    def fmt(v, w):
        return f"{v:>{w}.1f}" if v is not None else f"{'n/a':>{w}}"

    for r in rows:
        sp_s = f"{r['speedup']:>7.2f}x" if r["speedup"] is not None else f"{'-':>8}"
        print(
            f"{r['cid']:<{cid_w}}  {r['c_in']:>5} {r['c_out']:>5} {r['t']:>4} {r['hw']:>9} "
            f"{fmt(r['fused'], 10)} | {fmt(r['np'], 9)} {fmt(r['conv'], 9)} {fmt(r['nonfused'], 11)} {sp_s}  "
            f"{r['scheme']:<13}"
        )
    print(box)
    print("non-fused = standalone NeighborPadAsync (NP) + Conv3d, both on the FULL grid; speedup = (NP+conv)/fused.")
    print("speedup > 1.0 ⇒ fusion faster on device; the deployed scheme follows it.")

    out = f"np_speedup_{which}.csv"
    with open(out, "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(f"CSV: {out}")


if __name__ == "__main__":
    main()
