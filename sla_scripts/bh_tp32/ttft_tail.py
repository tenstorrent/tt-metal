# Per-request view of a vllm bench --save-detailed result: where the TTFT tail comes from.
#   python ttft_tail.py <result.json> [...]
import json
import sys

import numpy as np

for path in sys.argv[1:]:
    d = json.load(open(path))
    ttft = np.array([t if t else np.nan for t in d["ttfts"]])
    start = np.array(d["start_times"]) - min(d["start_times"])
    inp = np.array(d["input_lens"])
    out = np.array(d["output_lens"])
    ok = ~np.isnan(ttft)
    print(f"== {path}: n={ok.sum()} ttft mean {np.nanmean(ttft):.2f} p50 {np.nanpercentile(ttft, 50):.2f} "
          f"p90 {np.nanpercentile(ttft, 90):.2f} p99 {np.nanpercentile(ttft, 99):.2f} s; tpot mean {d['mean_tpot_ms']:.1f} ms")
    # arrivals in the 5 s before each request was sent (burst size it joined)
    burst = np.array([((start > s - 5) & (start <= s)).sum() for s in start])
    order = np.argsort(-np.nan_to_num(ttft))
    print("  slowest: start_s  ttft_s  input  output  sent_in_prev_5s")
    for i in order[:12]:
        print(f"    {start[i]:7.1f} {ttft[i]:6.2f} {inp[i]:6d} {out[i]:6d} {burst[i]:4d}")
    for lo, hi in ((0, 4096), (4097, 99999)):
        m = ok & (inp >= lo) & (inp <= hi)
        if m.any():
            print(f"  input {lo}-{hi}: n={m.sum()} ttft mean {ttft[m].mean():.2f} p99 {np.percentile(ttft[m], 99):.2f}")
    for a, b in ((0, 15), (15, 60), (60, 1e9)):
        m = ok & (start >= a) & (start < b)
        if m.any():
            print(f"  sent {a:.0f}-{b:.0f}s: n={m.sum()} ttft mean {ttft[m].mean():.2f} max {ttft[m].max():.2f}")
    r = np.corrcoef(burst[ok], ttft[ok])[0, 1]
    print(f"  corr(ttft, sent_in_prev_5s) = {r:.2f}")
