#!/usr/bin/env python
"""Run the ring_mla micro-bench under tracy R times; per kv size + mode, take the per-device median
RingJointSDPA device-kernel time (dropping a warmup prefix) and report scalar vs metadata. Aggregates
across R runs to show if the metadata overhead is genuine (consistent) or noise."""
import sys
import statistics
import pandas as pd
from loguru import logger
from tracy.process_model_log import get_latest_ops_log_filename
from models.perf.device_perf_utils import run_device_perf

TEST = "models/demos/deepseek_v3_d_p/tests/perf/test_ring_mla_microperf.py::test_ring_mla_microperf[auto]"
KV_SIZES = [256, 1024, 5120]
WARMUP = 6  # drop first N calls per (device,mode,kv) region
DUR = "DEVICE KERNEL DURATION [ns]"
R = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def region_rows(df, start, end):
    sp = df["OP TYPE"] == "signpost"
    isS = sp & (df["OP CODE"] == start)
    isE = sp & (df["OP CODE"] == end)
    depth = (isS.astype(int) - isE.astype(int)).cumsum()
    r = df[(depth > 0) & ~sp]
    r = r[r["OP TYPE"] == "tt_dnn_device"]
    return r[r["OP CODE"].str.contains("RingJointSDPA", na=False, regex=False)]


def per_dev_median(rows):
    # per device: drop WARMUP prefix calls, median the rest (us)
    out = {}
    for dev, g in rows.groupby("DEVICE ID"):
        vals = (g[DUR].values / 1000.0)[WARMUP:]
        if len(vals):
            out[dev] = statistics.median(vals)
    return out


def main():
    # results[kv]['scalar'|'meta'] = list over R runs of (worst_us, median_dev_us)
    results = {kv: {"scalar": [], "meta": []} for kv in KV_SIZES}
    for run in range(R):
        logger.info(f"[microperf] run {run+1}/{R}")
        run_device_perf(f'pytest "{TEST}"', subdir=f"ring_mla_micro_{run}", num_iterations=1,
                        cols=["DEVICE KERNEL"], batch_size=1)
        df = pd.read_csv(get_latest_ops_log_filename(f"ring_mla_micro_{run}"))
        for kv in KV_SIZES:
            for mode, tag in (("scalar", "SCALAR"), ("meta", "META")):
                pd_ = per_dev_median(region_rows(df, f"{tag}_kv{kv}_START", f"{tag}_kv{kv}_END"))
                if pd_:
                    worst = max(pd_.values())
                    med = statistics.median(pd_.values())
                    results[kv][mode].append((worst, med, pd_))

    lines = ["# ring_mla steady-state per-call perf (median over N calls/run, R runs): scalar vs metadata",
             f"# {WARMUP} warmup calls dropped per region; rotation inputs (kv = prior KV tokens).", ""]
    lines.append(f"{'kv':>5} {'scalar_med_us':>14} {'meta_med_us':>12} {'median-dev Δ':>13} {'worst-dev Δ':>12} {'runs':>5}")
    for kv in KV_SIZES:
        s = results[kv]["scalar"]; m = results[kv]["meta"]
        if not s or not m:
            lines.append(f"{kv:>5}  (no data)"); continue
        # median-device, averaged over runs
        s_med = statistics.mean(x[1] for x in s); m_med = statistics.mean(x[1] for x in m)
        s_worst = statistics.mean(x[0] for x in s); m_worst = statistics.mean(x[0] for x in m)
        # per-device pairwise delta averaged over devices, per run, then mean+std over runs
        pair_deltas = []
        for (_, _, ps), (_, _, pm) in zip(s, m):
            common = set(ps) & set(pm)
            if common:
                pair_deltas.append(statistics.mean(100.0 * (pm[d] - ps[d]) / ps[d] for d in common))
        dmed = 100.0 * (m_med - s_med) / s_med if s_med else float("nan")
        dworst = 100.0 * (m_worst - s_worst) / s_worst if s_worst else float("nan")
        pstr = f"{statistics.mean(pair_deltas):+.2f}%±{(statistics.pstdev(pair_deltas) if len(pair_deltas)>1 else 0):.2f}"
        lines.append(f"{kv:>5} {s_med:>14.2f} {m_med:>12.2f} {dmed:>+12.2f}% {dworst:>+11.2f}% {len(s):>5}")
        lines.append(f"      per-device-pairwise mean Δ over {len(pair_deltas)} runs: {pstr}%")
    out = "\n".join(lines)
    print(out)
    with open("models/demos/deepseek_v3_d_p/ring_mla_microperf.log", "w") as f:
        f.write(out + "\n")
    logger.success("[microperf] wrote models/demos/deepseek_v3_d_p/ring_mla_microperf.log")


if __name__ == "__main__":
    main()
