#!/usr/bin/env bash
set -euo pipefail
out=experiments/sdpa-l2/hybrid-mixed-v1
python_env/bin/python "$out/state_probe.py" > "$out/state-final.log" 2>&1
for mode in hybrid qk4_pv2 qk4_pv2_fullsub accurate fp32_hifi2_cheap; do
    for distribution in normal uniform common_v; do
        label=holdout-$mode-$distribution-1238
        python_env/bin/python "$out/run.py" --mode "$mode" --distinct-kv --q-repeats 1 \
            --k-chunks 512 --seed 1238 --distribution "$distribution" --iters 0 \
            --label "$label" > "$out/$label.log" 2>&1
    done
done
