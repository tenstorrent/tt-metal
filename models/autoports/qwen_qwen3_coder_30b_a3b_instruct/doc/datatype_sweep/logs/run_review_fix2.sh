#!/usr/bin/env bash
# Stage-07 review fixes, device phase 2.
#
# Phase 1 exposed a baseline-drift problem of the stage's own making: the
# candidate set was expressed as deltas from DEFAULT_PRECISION, and stage 07
# MOVED DEFAULT_PRECISION, so re-running any probe rebuilt different configs
# than the 48-layer rows were measured at. candidates.BASELINE_PRECISION now
# pins the stage-06 policy. This re-runs everything phase 1 touched, against
# the pinned baseline:
#
#   tier A  -- all 29 rows, so every audit matches the config its tier-B row ran
#   R19     -- bfp8 KV as a delta from the stage-06 baseline (comparable with
#              every other row in the table)
#   R26     -- R06 composed with R25 (its phase-1 row was computed but lost to a
#              JSON write error; the log survived, the row did not)
#   R28     -- bfp8 KV on top of the selected widths, the row the context
#              contract actually needs
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

C="python $D/doc/datatype_sweep/probes/sweep_runner.py --include-stacked --force --only R19_kv_bfp8,R26_attn_bfp4_bw64_24,R28_kv_bfp8_bw64_24"
hdr "$C" > $L/sweep_review_rows.log; eval $C >> $L/sweep_review_rows.log 2>&1
echo "=== rows exit $? ==="
