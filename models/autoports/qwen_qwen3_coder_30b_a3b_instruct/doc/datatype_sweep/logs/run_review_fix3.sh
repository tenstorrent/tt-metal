#!/usr/bin/env bash
# Stage-07 review fixes, device phase 3.
#
# R26 (R06 composed with R25) came back at 43.58 t/s/u against R25's 43.54 --
# a 0.09% lead, a quarter of the measured 0.368% band, bought with a top-1
# point. The stated rule ranks eligible rows on the point estimate, so that lead
# would flip the selection on a difference the rule itself calls unmeasurable.
# Settle it the way R25's own lead was settled: repeat the row and look at the
# spread rather than at one number.
#
#   repeats R26 x3  -- the same treatment R00 and R25 got
#   R27             -- the REQUIRED BFP4+LoFi pair for R26, now that R26 is a
#                      live contender rather than a losing one
#
# The tier-B row log is preserved: probes/repeats.py re-uses run_row, which
# rewrites logs/rows/<id>.log, and sweep_results.json's R26 row points at it.
set -u
cd /home/raahem/tt-metal
source python_env/bin/activate
unset QWEN3_PRECISION_CONFIG
D=models/autoports/qwen_qwen3_coder_30b_a3b_instruct
L=$D/doc/datatype_sweep/logs
G="$(git rev-parse --short HEAD) + uncommitted stage-07 sweep + review fixes"
hdr () { printf '# cmd: %s\n# git: %s\n# date: %s\n# hw: 1x4 Blackhole P300_X2, FABRIC_1D_RING\n' "$1" "$G" "$(date -Is)"; }

cp $L/rows/R26_attn_bfp4_bw64_24.log $L/rows/R26_attn_bfp4_bw64_24.tierB.log

C="python $D/doc/datatype_sweep/probes/sweep_runner.py --include-stacked --only R27_attn_bfp4_lofi_bw64_24"
hdr "$C" > $L/sweep_review_r27.log; eval $C >> $L/sweep_review_r27.log 2>&1
echo "=== R27 exit $? ==="

C="python $D/doc/datatype_sweep/probes/repeats.py --ids R26_attn_bfp4_bw64_24 --n 3"
hdr "$C" > $L/repeats_r26.log; eval $C >> $L/repeats_r26.log 2>&1
echo "=== repeats exit $? ==="

# restore the row log the tier-B number was read from
mv $L/rows/R26_attn_bfp4_bw64_24.log $L/rows/R26_attn_bfp4_bw64_24.repeat3.log
mv $L/rows/R26_attn_bfp4_bw64_24.tierB.log $L/rows/R26_attn_bfp4_bw64_24.log
