#!/usr/bin/env python
"""Per-call ring_mla perf (old=scalar vs new=metadata) across ALL ring_mla metadata equivalence tests
in tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py. Each test runs ONE scalar ring_mla call then
ONE metadata call (scalar first), so per device the RingJointSDPA rows are [scalar, metadata]. We run
each param under tracy, separate the two calls by per-device cumcount, and log worst-device device-kernel
time old vs new. Output -> models/demos/deepseek_v3_d_p/ring_mla_eq_perf.log."""
import pandas as pd
from loguru import logger
from tracy.process_model_log import get_latest_ops_log_filename
from models.perf.device_perf_utils import run_device_perf

TESTFILE = "tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py"
PARAMS = [
    ("test_ring_mla_metadata_matches_scalar_indexed", "slot0"),
    ("test_ring_mla_metadata_matches_scalar_indexed", "slot1"),
    ("test_ring_mla_metadata_matches_scalar_rotation", "kv64"),
    ("test_ring_mla_metadata_matches_scalar_rotation", "kv256"),
    ("test_ring_mla_metadata_matches_scalar_rotation", "kv320"),
]
LOGFILE = "models/demos/deepseek_v3_d_p/ring_mla_eq_perf.log"
DUR = "DEVICE KERNEL DURATION [ns]"


def parse(csv):
    df = pd.read_csv(csv)
    df = df[df["OP TYPE"] == "tt_dnn_device"]
    df = df[df["OP CODE"].str.contains("RingJointSDPA", na=False, regex=False)].copy()
    if df.empty:
        raise RuntimeError(f"no RingJointSDPA rows in {csv}")
    # per device, call 0 = scalar (run first), call 1 = metadata
    df["call"] = df.groupby("DEVICE ID").cumcount()
    ncalls = int(df["call"].max()) + 1
    scal = df[df["call"] == 0]
    meta = df[df["call"] == 1]
    scal_us = (scal.groupby("DEVICE ID")[DUR].mean() / 1000.0)
    meta_us = (meta.groupby("DEVICE ID")[DUR].mean() / 1000.0)
    return scal_us, meta_us, ncalls, df["DEVICE ID"].nunique()


def run_one(test, pid):
    import shutil
    sel = f"{test}[{pid}]"
    command = f'pytest "{TESTFILE}::{sel}"'
    subdir = f"ring_mla_eq_{test.split('_')[-1]}_{pid}"
    logger.info(f"[eqperf] running {sel}")
    run_device_perf(command, subdir=subdir, num_iterations=1, cols=["DEVICE KERNEL"], batch_size=1)
    csv = get_latest_ops_log_filename(subdir)
    try:  # retain a copy (run_device_perf cleans subdirs on the next run)
        shutil.copy(csv, f"/data/ppopovic/.claude-tmp/claude-4046/-home-ppopovic-tt-metal/ea247624-df70-4989-a0da-13dca0d9e850/scratchpad/eqcsv_{test.split('_')[-1]}_{pid}.csv")
    except Exception:
        pass
    return parse(csv)


def main():
    lines = [
        "# ring_mla per-call perf: OLD (scalar host args) vs NEW (metadata tensor)",
        "# source: ring_mla metadata equivalence tests (each runs 1 scalar call THEN 1 metadata call,",
        "# bit-exact). Numbers = worst-device RingJointSDPA device-kernel time (single call, us).",
        "# repro one functional test:",
        '#   KIMI_K2_6_HF_MODEL=/data/nbabin/Kimi-K2_6-dequantized python_env/bin/python -m pytest \\',
        f'#     "{TESTFILE}::test_ring_mla_metadata_matches_scalar_rotation[kv256]" -s',
        "# repro this perf log:  PYTHONPATH=. python_env/bin/python <this script>",
        "",
        "# NOTE single-call: worst-device is ONE outlier (dev ~16, ring coordinator) dominated by",
        "# dispatch/cold-start; the robust signal is the per-device PAIRWISE median delta (same device,",
        "# scalar-vs-metadata). Steady-state over many calls is in ring_mla_perf.log (Phase E, +0.7-1.2%).",
        "",
        f"{'test[param]':<26} {'worst_scal':>10} {'worst_meta':>10} {'wΔ':>7}  {'medianΔ(per-dev)':>16} {'meanΔ':>7}",
    ]
    for test, pid in PARAMS:
        name = f"{test.replace('test_ring_mla_metadata_matches_scalar_','')}[{pid}]"
        try:
            scal_us, meta_us, ncalls, ndev = run_one(test, pid)
            s_worst = scal_us.max(); m_worst = meta_us.max()
            wdelta = 100.0 * (m_worst - s_worst) / s_worst if s_worst else float("nan")
            # per-device pairwise delta (same device id, scalar vs metadata) — robust
            common = scal_us.index.intersection(meta_us.index)
            pair = (100.0 * (meta_us[common] - scal_us[common]) / scal_us[common])
            med = pair.median(); mean = pair.mean()
            lines.append(f"{name:<26} {s_worst:>10.2f} {m_worst:>10.2f} {wdelta:>+6.1f}%  {med:>+15.1f}% {mean:>+6.1f}%")
            lines.append(
                f"    scalar  worst-dev {scal_us.idxmax():.0f} {s_worst:.2f}us, all-dev range {scal_us.min():.2f}-{scal_us.max():.2f}"
            )
            lines.append(
                f"    metadata worst-dev {meta_us.idxmax():.0f} {m_worst:.2f}us, all-dev range {meta_us.min():.2f}-{meta_us.max():.2f}  ({ncalls} call/dev)"
            )
        except Exception as e:
            lines.append(f"{name:<26}  ERROR: {str(e)[:80]}")
        logger.info(f"[eqperf] {name} done")

    out = "\n".join(lines)
    print(out)
    with open(LOGFILE, "w") as f:
        f.write(out + "\n")
    logger.success(f"[eqperf] wrote {LOGFILE}")


if __name__ == "__main__":
    main()
