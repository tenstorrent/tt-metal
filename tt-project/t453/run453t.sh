#!/bin/bash
# t453: one blx01 broker job on the full 4x8 mesh: one attention layer at the fl2va 10 s shape, dense ring SDPA vs
# VSA-0.9 (two-op, then vsa_ring_sdpa), host-timed with per-op synced breakdowns. Random weights.
set -o pipefail
F=/var/tmp/fasth3; T=$F/t409; W=$T/b; OUT=$T/out_t453; VENV=$F/t48/python_env
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le 88 ] || { echo "[t453] / at $use%"; exit 5; }
gb=$(timeout 120 du -sxBG $F | cut -f1 | tr -dc 0-9); [ "${gb:-0}" -le 450 ] || { echo "[t453] footprint ${gb}G"; exit 5; }
mkdir -p $OUT $F/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=$F/home/.cache/huggingface HF_HUB_OFFLINE=1
export TT_METAL_CACHE=$F/cache/tt-metal-cache-h3hf TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools:$T/plug
export T409_ALARM_S=$(( ${T409_PY_TIMEOUT:-540} - 35 ))
cd $W || exit 3
[ -f ttnn/ttnn/_ttnn.so ] || { echo "[t453] no build in $W"; exit 3; }
source $VENV/bin/activate || exit 3
CMD="timeout --foreground -k 10 ${T409_PY_TIMEOUT:-540} python -u -m pytest -sv -p no:cacheprovider -p t409_alarm models/tt_dit/tests/models/minimax_h3/test_vsa_timing_minimax_h3.py -k 4x8sp1tp0nl2"
echo "[t453] host=$(hostname) commit=$(git rev-parse HEAD) $(date -u '+%F %T') UTC" | tee $OUT/run.log
echo "[t453] cmd: $CMD" | tee -a $OUT/run.log
T0=$(date +%s)
bash -c "$CMD" 2>&1 | tee -a $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[t453] process wall $(( $(date +%s) - T0 )) s" | tee -a $OUT/run.log
echo "T409_EXIT=$rc" | tee -a $OUT/run.log
[ $rc = 0 ] && touch $OUT/PASS
exit $rc
