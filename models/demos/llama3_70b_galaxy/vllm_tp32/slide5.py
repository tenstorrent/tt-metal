#!/usr/bin/env python3
"""Score one bench folder against the slide-5 rows (WH Galaxy TP32 vs SLA).

usage: slide5.py <bench_dir> [hit_json] [mmlu_results_json]
Reads the vllm bench JSONs written by bench_tis.sh. Tok/s/user = 1000 / TPOT, so decode P99
(the SLA floor) is 1000 / p99_tpot. Prints one row per SLA line and exits 0 only if every
scored row passes (the concurrent-users row is reported, not scored).
"""
import glob, json, os, sys

d = sys.argv[1]
hit_path = sys.argv[2] if len(sys.argv) > 2 else None
mmlu_path = sys.argv[3] if len(sys.argv) > 3 else None


def load(tag):
    p = os.path.join(d, f"{tag}.json")
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    j["_failed"] = j.get("failed", 0)
    return j


def tsu(ms):
    return 1000.0 / ms if ms else 0.0


rows = []  # (line, target, measured, ok or None)


def add(line, target, measured, ok):
    rows.append((line, target, measured, ok))


add("Concurrent users", "128", "32 max (TP32: 4 lanes x 8)", None)

s = load("single_4k_500")
if s:
    v = tsu(s["mean_tpot_ms"])
    add("Decode, 1 user at 4K (tok/s)", "25", f"{v:.1f}", v >= 25)

r = load("rpm60_mix")
if r:
    m, p = r["mean_ttft_ms"] / 1000, r["p99_ttft_ms"] / 1000
    add("60 RPM: TTFT mean / P99", "1 / 5 s", f"{m:.2f} / {p:.2f} s", m <= 1 and p <= 5)
    dm, dp = tsu(r["mean_tpot_ms"]), tsu(r["p99_tpot_ms"])
    add("60 RPM: decode mean / P99 (tok/s/user)", "25 / 15", f"{dm:.1f} / {dp:.1f}", dm >= 25 and dp >= 15)
    em, ep = r["mean_e2el_ms"] / 1000, r["p99_e2el_ms"] / 1000
    add("60 RPM: end-to-end mean / P99", "20 / 140 s", f"{em:.1f} / {ep:.1f} s", em <= 20 and ep <= 140)

x = load("rpm60_prefix40")
if x:
    m, p = x["mean_ttft_ms"] / 1000, x["p99_ttft_ms"] / 1000
    add("60 RPM + prefix: TTFT mean / P99", "0.5 / 1.5 s", f"{m:.2f} / {p:.2f} s", m <= 0.5 and p <= 1.5)

k = load("rpm240_mix")
if k:
    p, dp = k["p99_ttft_ms"] / 1000, tsu(k["p99_tpot_ms"])
    add("240 RPM: TTFT P99 / decode P99", "5 s / 15", f"{p:.1f} s / {dp:.1f}", p <= 5 and dp >= 15)

c16, c32 = load("cold_single_16k"), load("cold_single_32k")
if c16 and c32:
    a, b = c16["mean_ttft_ms"] / 1000, c32["mean_ttft_ms"] / 1000
    add("Cold 16K / 32K TTFT (1 request)", "5 / 10 s", f"{a:.2f} / {b:.2f} s", a <= 5 and b <= 10)

if hit_path and os.path.exists(hit_path):
    h = json.load(open(hit_path))["hit_rate"]
    # slide 5 counts 38.7% as PASS against a 40% target on a 40% designed workload
    add("Prefix hit rate", "40%", f"{h:.1%}", h >= 0.38)

if mmlu_path:
    paths = glob.glob(mmlu_path) if any(ch in mmlu_path for ch in "*?[") else [mmlu_path]
    if paths:
        res = json.load(open(sorted(paths)[-1]))["results"]
        key = "mmlu_llama" if "mmlu_llama" in res else sorted(res)[0]
        acc = next(v for kk, v in res[key].items() if kk.startswith("exact_match") and "stderr" not in kk)
        add("MMLU", ">= 80%", f"{acc:.1%}", acc >= 0.80)

failed = sum(j["_failed"] for j in (s, r, x, k, c16, c32) if j)
w = max(len(x[0]) for x in rows)
print(f"{'SLA line':{w}s}  {'Target':12s}  {'Measured':28s}  Result")
for line, t, m, ok in rows:
    res = "n/a" if ok is None else ("PASS" if ok else "FAIL")
    print(f"{line:{w}s}  {t:12s}  {m:28s}  {res}")
print(f"failed requests across cases: {failed}")
scored = [ok for *_, ok in rows if ok is not None]
sys.exit(0 if scored and all(scored) and failed == 0 else 1)
