#!/usr/bin/env bash
# Stage-07 review fixes, device phase 1.
#
#   1. tier A re-run, every row: the audit now carries lm_head_math_fidelity,
#      norm_math_fidelity and the two observed terminal dtypes, so the rows that
#      only move those fields stop being byte-identical to the baseline.
#      Also covers the two new stacked rows.
#   2. R26 -- R06 (attn bfp4) composed with R25 (widths 64/24): the orthogonal
#      stack the stage argued for but never measured.
#   3. R19 -- bfp8 KV, re-run now that prefill casts K/V to the cache dtype.
set -u
cd /home/raahem/tt-metal
source python_env/bin/activate
unset QWEN3_PRECISION_CONFIG
D=models/autoports/qwen_qwen3_coder_30b_a3b_instruct
L=$D/doc/datatype_sweep/logs
G="$(git rev-parse --short HEAD) + uncommitted stage-07 sweep + review fixes"
hdr () { printf '# cmd: %s\n# git: %s\n# date: %s\n# hw: 1x4 Blackhole P300_X2, FABRIC_1D_RING\n' "$1" "$G" "$(date -Is)"; }

C="python $D/doc/datatype_sweep/probes/structural_probe.py --layers 2"
hdr "$C" > $L/structural_probe.log; eval $C >> $L/structural_probe.log 2>&1
echo "=== structural exit $? ==="

C="python $D/doc/datatype_sweep/probes/sweep_runner.py --include-stacked --only R26_attn_bfp4_bw64_24"
hdr "$C" > $L/sweep_review_r26.log; eval $C >> $L/sweep_review_r26.log 2>&1
echo "=== R26 exit $? ==="

C="python $D/doc/datatype_sweep/probes/sweep_runner.py --force --only R19_kv_bfp8"
hdr "$C" > $L/sweep_review_r19.log; eval $C >> $L/sweep_review_r19.log 2>&1
echo "=== R19 exit $? ==="
