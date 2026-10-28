"""Markdown tables from the round-timeline JSONs of tt/round_timeline.py.   python3 -I round_tl_report.py <run_tl_dir> [warmup=10]  (one table per pass x k)."""
import glob
import json
import os
import sys

d = sys.argv[1]
warm = int(sys.argv[2]) if len(sys.argv) > 2 else 10
PH = [
    ("gap_before", "post-process of previous round (python loop, scheduler)"),
    ("ensure", "pool.ensure (pages)"),
    ("use", "bucket select"),
    ("force", "set_force (h2d copy)"),
    ("hash", "Engram n-gram hash (host)"),
    ("rows", "Engram table gather + dequant (host)"),
    ("pack_prep", "from_torch / cat of the packed inputs"),
    ("upload", "copy_host_to_device of ctrl + Engram rows"),
    ("feed_other", "rest of _feed (pos, reshapes)"),
    ("enqueue", "execute_trace (non-blocking enqueue)"),
    ("sync", "synchronize_device #1 (device wait)"),
    ("rb_issue", "read-back: from_device x rows (non-blocking)"),
    ("rb_sync", "read-back: synchronize #2"),
    ("rb_parse", "read-back: to_torch + cat + parse"),
]


def st(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return sum(xs) / len(xs), xs[min(len(xs) - 1, int(0.95 * len(xs)))], len(xs)


for f in sorted(glob.glob(os.path.join(d, "rounds_pass*.json"))):
    J = json.load(open(f))
    recs = [r for r in J["records"] if not r["calib"]]
    ks = sorted({(r["k"], r["draft"], r.get("rows", "g")) for r in recs})
    print(f"\n### {os.path.basename(f)}  ({J['label']}; {len(recs)} rounds, warm-up {warm} dropped per k)")
    for k, dr, rw in ks:
        rs = [r for r in recs if r["k"] == k and r["draft"] == dr and r.get("rows", "g") == rw][warm:]
        if len(rs) < 5:
            print(f"\nk={k} rows={rw}: only {len(rs)} rounds after warm-up, skipped")
            continue
        for r in rs:
            r["period"] = r["round"] + (r["gap_before"] or 0.0)
            r["pre"] = sum(r.get(x, 0.0) for x in ("ensure", "use", "force", "feed", "samp_params"))
            r["dev_wait"] = r.get("enqueue", 0.0) + r.get("sync", 0.0)
            r["post_dev"] = r["round"] - r["pre"] - r["dev_wait"]
            r["dev_idle"] = r["period"] - r["dev_wait"]
        P = st([r["period"] for r in rs])[0]
        print(f"\n**k={k} rows={'sampled' if rw == 's' else 'greedy'} (n={k+1}, T={rs[0]['T']} rows/mesh-row, B={rs[0]['B']}), {len(rs)} rounds, buckets {sorted({r['bucket'] for r in rs if r['bucket'] is not None})}**\n")
        print("| phase | mean ms | p95 ms | % of period |\n|---|---|---|---|")
        PH2 = PH + ([("samp_params", "host: sampling params (sampling_rows + upload)")] if rw == "s" else [])
        for key, desc in PH2:
            s = st([r.get(key) for r in rs])
            if s:
                print(f"| {key}: {desc} | {s[0]:.2f} | {s[1]:.2f} | {100 * s[0] / P:.1f} |")
        for key, desc in (
            ("pre", "host before enqueue (ensure+use+force+feed)"),
            ("dev_wait", "enqueue + sync = device-busy window seen by host"),
            ("post_dev", "host after the sync (read-back)"),
            ("round", "_round total (what the demo prints as round_ms)"),
            ("period", "start-to-start period = round + post-process gap"),
            ("dev_idle", "device idle per round (period - device window)"),
        ):
            s = st([r[key] for r in rs])
            print(f"| **{key}**: {desc} | {s[0]:.2f} | {s[1]:.2f} | {100 * s[0] / P:.1f} |")
        ex = J.get("extra", {}).get("dev_only", {})
        if f"k{k}" in ex and dr:
            e = ex[f"k{k}"]
            print(f"\ndevice-only back-to-back replays (bucket {e['bucket']}): " + ", ".join(f"{a} {e[a]:.2f}" for a in ("mono_ms", "A_verify_ms", "S_greedy_ms", "S_sampled_ms", "B_tail_ms", "ASB_ms") if a in e) + " ms")
