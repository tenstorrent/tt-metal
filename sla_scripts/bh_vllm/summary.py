#!/usr/bin/env python3
# Prints TTFT and tok/s/user (1000 / TPOT) per case vs SLA, plus the measured prefix hit rate.
import json, os

out = os.environ["OUT"]
print(f"{'case':16s} {'ok':>8s} {'TTFT mean':>10s} {'TTFT P99':>9s} {'tsu mean':>9s} {'tsu P99':>8s}")
for tag in ["single_4k_500", "cold_single_16k", "cold_single_32k", "rpm60_mix", "rpm60_prefix40", "rpm240_mix"]:
    path = f"{out}/{tag}.json"
    if not os.path.exists(path):
        print(f"{tag:16s} missing")
        continue
    d = json.load(open(path))
    print(
        f"{tag:16s} {d['completed']:>4d}/{d['num_prompts']:<3d} {d['mean_ttft_ms']/1000:>9.2f}s {d['p99_ttft_ms']/1000:>8.2f}s "
        f"{1000/d['mean_tpot_ms']:>9.1f} {1000/d['p99_tpot_ms']:>8.1f}"
    )
hit = f"{out}/hit_rpm60_40pct.json"
if os.path.exists(hit):
    h = json.load(open(hit))
    print(f"prefix hit rate: {h['hit_rate']:.1%} measured, {h['designed_hit_rate']:.0%} designed")
print("SLA: TTFT mean 1 s, P99 5 s; cold 16K <= 5 s, 32K <= 10 s; tsu mean 25, P99 15")
