#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
# Inkling-2-Mini v1 baseline: golden check (forced + token-only), then traces + greedy + tok/s at max_ctx 4K/32K/128K.
# Usage: models/demos/inkling2_mini/demo/run_baseline.sh [out_dir]   (from the repo root, env set; takes ~6 min)
set -euo pipefail
OUT=${1:-/work/inkling-bringup/A1-scratch/baseline}
mkdir -p "$OUT"
L="flock /tmp/tt-card.lock"
M=models.demos.inkling2_mini.demo
filter() { grep -E '^prompt|logits|^\{"pos"|match_golden|Error|error' | cut -c1-300; }
$L python -m $M.golden_check --json "$OUT/golden_forced.json" 2>&1 | filter
$L python -m $M.golden_check --no-force --json "$OUT/golden_tokens.json" 2>&1 | filter
$L python -m $M.demo --max-ctx 131072 --json "$OUT/demo_128k.json" 2>&1 | grep '^{' | cut -c1-400
$L python -m $M.demo --max-ctx 4096 --skip-app-trace --prompts --perf-pos 2048 --json "$OUT/perf_4k.json" 2>&1 | filter
$L python -m $M.demo --max-ctx 34816 --skip-app-trace --prompts --perf-pos 32768 --json "$OUT/perf_32k.json" 2>&1 | filter
python - "$OUT" <<'EOF'
import json, sys
o = sys.argv[1]
d = json.load(open(f"{o}/demo_128k.json"))
print("traces bit-equal:", d["app_trace_bit_equal"], d["step_trace_bit_equal"], "eq eager:", d["step_trace_eq_eager"],
      f"trace {d['trace_used_mib']:.1f} MiB, DRAM free {d['dram_free_after_trace_gib']:.2f} GiB")
for g in d["gen"]:
    print("greedy prompt", g["prompt"], "golden16:", g.get("match_golden_first16"), repr(g.get("text", ""))[:160])
for name in ("perf_4k", "perf_32k"):
    for p in json.load(open(f"{o}/{name}.json"))["perf"]:
        print(f"{name}: pos {p['pos']} {p['ms']:.1f} ms {p['tok_s']:.2f} tok/s")
for p in d["perf"]:
    print(f"perf_128k: pos {p['pos']} {p['ms']:.1f} ms {p['tok_s']:.2f} tok/s")
EOF
