#!/bin/bash
# t409/t416: one blx01 broker job, VSA attention at sparsity 0 vs the dense ring path on the full 4x8 mesh
# (test_vsa_attention_minimax_h3.py::test_vsa_sparsity0_matches_ring_path, PCC >= 0.9995). No weights.
# pytest runs directly in the broker job's process group under timeout --foreground -k 10 (no setsid).
set -o pipefail
F=/var/tmp/fasth3; T=$F/t409; W=$T/b; OUT=$T/out_attn; VENV=$F/t48/python_env
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le 88 ] || { echo "[t409] / at $use%"; exit 5; }
gb=$(timeout 120 du -sxBG $F | cut -f1 | tr -dc 0-9); [ "${gb:-0}" -le 450 ] || { echo "[t409] footprint ${gb}G"; exit 5; }
mkdir -p $OUT $F/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=$F/home/.cache/huggingface HF_HUB_OFFLINE=1
export TT_METAL_CACHE=$F/cache/tt-metal-cache-h3hf TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools
cd $W || exit 3
[ -f ttnn/ttnn/_ttnn.so ] || { echo "[t409] no build in $W"; exit 3; }
source $VENV/bin/activate || exit 3
CMD="timeout --foreground -k 10 ${T409_PY_TIMEOUT:-540} python -u -m pytest -sv -p no:cacheprovider models/tt_dit/tests/models/minimax_h3/test_vsa_attention_minimax_h3.py::test_vsa_sparsity0_matches_ring_path -k 4x8sp1tp0"
echo "[t409] host=$(hostname) commit=$(git rev-parse HEAD) job=${TTP_RUNNER_JOB:-} $(date -u '+%F %T') UTC" | tee $OUT/run.log
echo "[t409] cmd: $CMD" | tee -a $OUT/run.log
T0=$(date +%s)
bash -c "$CMD" 2>&1 | tee -a $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[t409] process wall $(( $(date +%s) - T0 )) s" | tee -a $OUT/run.log
echo "T409_EXIT=$rc" | tee -a $OUT/run.log
[ $rc = 0 ] && touch $OUT/PASS
exit $rc
