# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-op device / op2op report for test_kimi_moe_layer_op2op, and a two-revision diff.

Reads the `ops_perf_results_*.csv` a tracy run of that test produces.  Segmentation follows the
rules the ops CSV actually obeys:

  * signpost rows have OP TYPE == "signpost"; their GLOBAL CALL COUNT is NaN, so they only mark host
    positions and must be dropped from op grouping.
  * GLOBAL CALL COUNT on a device row is a PER-DEVICE runtime id -- the 32 rows of one logical op do
    not share it, so it cannot be used to group them.  Group positionally instead: rows are in
    execution order, and one logical op is consecutive rows up to the point where OP CODE changes or
    a DEVICE ID repeats.
  * `iter_<n>_start` / `iter_<n>_end` bracket each iteration.  Iteration 0 is the JIT-compile one;
    default to the last iteration present.

Per op it reports the MAX across devices of device-kernel time and of op2op latency -- the max, not
the mean, because the layer's critical path is the slowest chip.

Usage:
    python tests/ttnn/unit_tests/kimi_op2op_report.py <ops_csv>                 # one run
    python tests/ttnn/unit_tests/kimi_op2op_report.py <ops_csv_a> <ops_csv_b>   # A/B, per-op
    python tests/ttnn/unit_tests/kimi_op2op_report.py <a> <b> --by-opcode       # A/B, per OP CODE
    optional: --iter N   (default: last iteration in the file)

Prefer --by-opcode across revisions: the same python call does not always lower to the same number
of device ops (all_gather_async is AllBroadcast+Concat on 2026-01-01, one fused AllGatherAsync on
current main), and once the op counts differ a positional diff is aligned wrong from there on.
"""

import sys

import pandas as pd


def _signpost_rows(df):
    out = {}
    for i, (t, c) in enumerate(zip(df["OP TYPE"], df["OP CODE"])):
        if str(t) == "signpost":
            out.setdefault(str(c), []).append(i)
    return out


def parse(csv_path, iteration=None):
    """Return (iteration, [ {idx, op, ndev, dev_ns, o2o_ns, cores, in0} ... ])."""
    df = pd.read_csv(csv_path, low_memory=False)
    sp = _signpost_rows(df)

    iters = sorted(int(k.split("_")[1]) for k in sp if k.startswith("iter_") and k.endswith("_start"))
    if not iters:
        raise SystemExit(f"{csv_path}: no iter_<n>_start signposts -- was this the op2op test?")
    if iteration is None:
        iteration = iters[-1]
    if iteration not in iters:
        raise SystemExit(f"{csv_path}: iteration {iteration} not present (have {iters})")

    lo = sp[f"iter_{iteration}_start"][0]
    hi = sp.get(f"iter_{iteration}_end", [len(df)])[0]

    op_code = df["OP CODE"].tolist()
    op_type = df["OP TYPE"].tolist()
    dev_id = df["DEVICE ID"].tolist()
    kern = df["DEVICE KERNEL DURATION [ns]"].tolist()
    o2o = df["OP TO OP LATENCY [ns]"].tolist()
    cores = df["CORE COUNT"].tolist()
    in0x = df.get("INPUT_0_X_PAD[LOGICAL]", pd.Series([None] * len(df))).tolist()
    in0y = df.get("INPUT_0_Y_PAD[LOGICAL]", pd.Series([None] * len(df))).tolist()

    groups, cur, seen = [], [], set()
    for i in range(lo + 1, hi):
        if str(op_type[i]) == "signpost":
            continue
        if cur and (op_code[i] != op_code[cur[0]] or dev_id[i] in seen):
            groups.append(cur)
            cur, seen = [], set()
        cur.append(i)
        seen.add(dev_id[i])
    if cur:
        groups.append(cur)

    ops = []
    for k, blk in enumerate(groups):
        i0 = blk[0]
        ops.append(
            {
                "idx": k,
                "op": op_code[i0],
                "ndev": len(blk),
                "dev_ns": max(kern[i] for i in blk),
                "o2o_ns": max(o2o[i] for i in blk),
                "cores": cores[i0],
                "in0": f"{in0y[i0]}x{in0x[i0]}",
            }
        )
    return iteration, ops


def report(csv_path, iteration=None):
    it, ops = parse(csv_path, iteration)
    print(f"# {csv_path}")
    print(f"# iteration {it}: {len(ops)} logical ops")
    print(f"{'idx':>4} {'op':46s} {'in0 (YxX)':>16} {'cores':>6} {'device us':>11} {'op2op us':>11}")
    dev = o2o = 0.0
    for o in ops:
        dev += o["dev_ns"]
        o2o += o["o2o_ns"]
        print(
            f"{o['idx']:4d} {o['op'][:46]:46s} {o['in0']:>16} {str(o['cores']):>6} "
            f"{o['dev_ns'] / 1e3:11.1f} {o['o2o_ns'] / 1e3:11.1f}"
        )
    total = dev + o2o
    pct = 100.0 * o2o / total if total else 0.0
    print(
        f"\n# TOTAL device {dev / 1e6:.3f} ms | op2op {o2o / 1e6:.3f} ms | sum {total / 1e6:.3f} ms | op2op {pct:.1f}%"
    )
    return ops


def _by_opcode(ops):
    agg = {}
    for o in ops:
        e = agg.setdefault(o["op"], [0, 0.0, 0.0])
        e[0] += 1
        e[1] += o["dev_ns"]
        e[2] += o["o2o_ns"]
    return agg


# Ops that belong to a gather.  A python-level all_gather call does not lower to the same number of
# device ops in every revision -- e.g. ttnn.experimental.all_gather_async is AllBroadcast + Concat
# (two ops) on 2026-01-01 and a single fused AllGatherAsync on current main -- so the gather family
# has to be totalled rather than matched op-for-op, and positional alignment across it is meaningless.
_GATHER_OPS = {
    "AllGatherAsyncDeviceOperation",
    "AllBroadcastDeviceOperation",
    "AllGatherDeviceOperation",
    "ConcatDeviceOperation",
}


def diff_by_opcode(csv_a, csv_b, iteration=None):
    """Totals per OP CODE, then a gather-family / everything-else split.

    This is the view to trust when the two revisions lower the same python call to different numbers
    of device ops: it compares like with like without assuming the op streams line up.
    """
    it_a, a = parse(csv_a, iteration)
    it_b, b = parse(csv_b, iteration)
    A, B = _by_opcode(a), _by_opcode(b)
    print(f"# A = {csv_a} (iteration {it_a}, {len(a)} device ops)")
    print(f"# B = {csv_b} (iteration {it_b}, {len(b)} device ops)")
    print(f"\n{'OP CODE':44s} {'nA':>3} {'nB':>3} {'devA ms':>8} {'devB ms':>8} {'o2oA ms':>8} {'o2oB ms':>8}")
    for k in sorted(set(A) | set(B)):
        ea, eb = A.get(k, [0, 0.0, 0.0]), B.get(k, [0, 0.0, 0.0])
        print(
            f"{k[:44]:44s} {ea[0]:3d} {eb[0]:3d} {ea[1] / 1e6:8.3f} {eb[1] / 1e6:8.3f} "
            f"{ea[2] / 1e6:8.3f} {eb[2] / 1e6:8.3f}"
        )
    allk = set(A) | set(B)
    for name, ks in [
        ("gather family (lowering may differ -- totals only)", allk & _GATHER_OPS),
        ("everything else (same op set both sides)", allk - _GATHER_OPS),
    ]:
        na = sum(A.get(k, [0, 0, 0])[0] for k in ks)
        nb = sum(B.get(k, [0, 0, 0])[0] for k in ks)
        da = sum(A.get(k, [0, 0, 0])[1] for k in ks)
        db = sum(B.get(k, [0, 0, 0])[1] for k in ks)
        ga = sum(A.get(k, [0, 0, 0])[2] for k in ks)
        gb = sum(B.get(k, [0, 0, 0])[2] for k in ks)
        print(f"\n{name}")
        print(f"  ops    {na} -> {nb}")
        if da:
            print(
                f"  device {da / 1e6:7.3f} -> {db / 1e6:7.3f} ms  ({(db - da) / 1e6:+.3f} ms, {100 * (db - da) / da:+.1f}%)"
            )
        if ga:
            print(
                f"  op2op  {ga / 1e6:7.3f} -> {gb / 1e6:7.3f} ms  ({(gb - ga) / 1e6:+.3f} ms, {100 * (gb - ga) / ga:+.1f}%)"
            )
        if na and nb:
            print(
                f"  op2op per op {ga / na / 1e3:6.1f} -> {gb / nb / 1e3:6.1f} us  ({(gb / nb - ga / na) / 1e3:+.1f} us/op)"
            )


def diff(csv_a, csv_b, iteration=None):
    it_a, a = parse(csv_a, iteration)
    it_b, b = parse(csv_b, iteration)
    print(f"# A = {csv_a} (iteration {it_a}, {len(a)} ops)")
    print(f"# B = {csv_b} (iteration {it_b}, {len(b)} ops)")
    if len(a) != len(b):
        print(
            f"# WARNING: op counts differ ({len(a)} vs {len(b)}).  The two runs did not execute the "
            f"same NUMBER of device ops -- most likely one revision lowers a gather to a different op "
            f"count (see _GATHER_OPS).  The rows below are aligned BY POSITION and are therefore "
            f"meaningless past the first divergence; use --by-opcode instead."
        )
    print(
        f"\n{'idx':>4} {'op':40s} {'devA us':>10} {'devB us':>10} {'d dev':>9} "
        f"{'o2oA us':>10} {'o2oB us':>10} {'d o2o':>9}"
    )
    da = db = ga = gb = 0.0
    for i in range(min(len(a), len(b))):
        oa, ob = a[i], b[i]
        tag = "" if oa["op"] == ob["op"] else "  <<< OP MISMATCH"
        da += oa["dev_ns"]
        db += ob["dev_ns"]
        ga += oa["o2o_ns"]
        gb += ob["o2o_ns"]
        print(
            f"{i:4d} {oa['op'][:40]:40s} {oa['dev_ns'] / 1e3:10.1f} {ob['dev_ns'] / 1e3:10.1f} "
            f"{(ob['dev_ns'] - oa['dev_ns']) / 1e3:+9.1f} "
            f"{oa['o2o_ns'] / 1e3:10.1f} {ob['o2o_ns'] / 1e3:10.1f} "
            f"{(ob['o2o_ns'] - oa['o2o_ns']) / 1e3:+9.1f}{tag}"
        )
    print(
        f"\n# device  A {da / 1e6:.3f} ms -> B {db / 1e6:.3f} ms  ({(db - da) / 1e6:+.3f} ms, "
        f"{100.0 * (db - da) / da if da else 0:+.1f}%)"
    )
    print(
        f"# op2op   A {ga / 1e6:.3f} ms -> B {gb / 1e6:.3f} ms  ({(gb - ga) / 1e6:+.3f} ms, "
        f"{100.0 * (gb - ga) / ga if ga else 0:+.1f}%)"
    )


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    it = None
    for a in sys.argv[1:]:
        if a.startswith("--iter"):
            it = int(a.split("=", 1)[1]) if "=" in a else int(sys.argv[sys.argv.index(a) + 1])
    by_opcode = any(a.startswith("--by-opcode") for a in sys.argv[1:])
    if len(args) == 1:
        report(args[0], it)
    elif len(args) >= 2:
        if by_opcode:
            diff_by_opcode(args[0], args[1], it)
        else:
            diff(args[0], args[1], it)
    else:
        print(__doc__)
        raise SystemExit(2)
