"""k=3 vs k=5 round decomposition per configuration (markdown).  python3 -I round_tl_summary.py <label>=<run_tl_dir> ...
Host phases: rounds of the position bucket that the device-only / truncated-trace block was measured at (the last bucket), greedy and sampled rows separately, first 10 per cell dropped.
Device phases (greedy monolithic trace): embed+head+argmax = L0_verify; accept = L0_accept - L0_verify; drafter = L0 - L0_accept; layers = L40_verify - L0_verify; commit = mono - L40_accept - drafter."""
import glob
import json
import os
import sys


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


for arg in sys.argv[1:]:
    label, d = arg.split("=", 1)
    J = json.load(open(glob.glob(os.path.join(d, "rounds_pass*.json"))[0]))
    ex = J["extra"].get("dev_only", {})
    print(f"\n#### {label}\n")
    print("| item (ms) | k=3 | k=5 | k5 - k3 |\n|---|---|---|---|")
    rows = {}
    for k in (3, 5):
        e = ex.get(f"k{k}", {})
        P = e.get("bucket")
        tr = e.get("trunc", {})
        c = {}
        c["rows/mesh-row T"] = None
        for rm in ("g", "s"):
            rs = [r for r in J["records"] if not r["calib"] and r["k"] == k and r.get("rmode", "g") == rm and r["bucket"] == P][10:]
            if not rs:
                continue
            for r in rs:
                r["pre"] = sum(r.get(x, 0.0) for x in ("ensure", "use", "force", "feed", "samp_params"))
                r["wait"] = r.get("enqueue", 0.0) + r.get("sync", 0.0)
                r["post"] = r["round"] - r["pre"] - r["wait"]
                r["period"] = r["round"] + (r["gap_before"] or 0.0)
            tag = "greedy" if rm == "g" else "sampled"
            c[f"[{tag}] rounds"] = len(rs)
            c[f"[{tag}] period (start-to-start)"] = mean([r["period"] for r in rs])
            c[f"[{tag}] device wait (enqueue+sync)"] = mean([r["wait"] for r in rs])
            c[f"[{tag}] host before enqueue"] = mean([r["pre"] for r in rs])
            c[f"[{tag}]   of which Engram hash+rows+pack+upload"] = mean([sum(r.get(x, 0.0) for x in ("hash", "rows", "pack_prep", "upload")) for r in rs])
            c[f"[{tag}] host after sync (read-back)"] = mean([r["post"] for r in rs])
            c[f"[{tag}] post-process gap"] = mean([r["gap_before"] or 0.0 for r in rs])
        c["dev: mono round trace"] = e.get("mono_ms")
        c["dev: A verify trace (sampled path)"] = e.get("A_verify_ms")
        c["dev: S sampler greedy params"] = e.get("S_greedy_ms")
        c["dev: S sampler T=1 top_p=.95 (full vocab)"] = e.get("S_sampled_ms")
        c["dev: S sampler top_k=20"] = e.get("S_topk20_ms")
        c["dev: B tail trace (accept+commit+drafter)"] = e.get("B_tail_ms")
        if "L0" in tr and "L40_accept" in tr and e.get("mono_ms"):
            drafter = tr["L0"] - tr["L0_accept"]
            c["dev: embed + head + argmax"] = tr["L0_verify"]
            c["dev: accept rule"] = tr["L0_accept"] - tr["L0_verify"]
            c["dev: drafter (write_main + 3 stages + pack)"] = drafter
            c["dev: 40 layers incl. Engram + taps"] = tr["L40_verify"] - tr["L0_verify"]
            c["dev:   per layer"] = (tr["L40_verify"] - tr["L0_verify"]) / 40
            c["dev: compressor commit + misc"] = e["mono_ms"] - tr["L40_accept"] - drafter
            c["dev: layers 0-9 per layer"] = (tr["L10"] - tr["L0"]) / 10
            c["dev: layers 10-19 per layer"] = (tr["L20"] - tr["L10"]) / 10
        rows[k] = c
    keys = list(rows[3].keys())
    for key in keys:
        a, b = rows[3].get(key), rows[5].get(key)
        if a is None and b is None:
            continue
        f = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
        diff = f"{b - a:+.3f}" if isinstance(a, float) and isinstance(b, float) else "-"
        print(f"| {key} | {f(a)} | {f(b)} | {diff} |")

    for k in (3, 5):
        sec = ex.get(f"k{k}", {}).get("sections")
        if sec:
            print(f"\neager per-section ms summed over the layers, k={k}: " + ", ".join(f"{a} {v:.1f}" for a, v in sorted(sec.items(), key=lambda kv: -kv[1])))
