#!/bin/bash
# t238 unit job (blx01 broker): the lean key-phase 2-D test at the t238 code head.
set -o pipefail
F=/var/tmp/fasth3; T=$F/t238; B=$T/b; O=$T/ut; mkdir -p $O; L=$O/run.log
mkdir -p $F/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=$F/home/.cache/huggingface
source $F/t48/python_env/bin/activate
export TT_METAL_HOME=$B PYTHONPATH=$B:$B/ttnn:$B/tools HF_HUB_OFFLINE=1 TT_METAL_CACHE=$F/cache/tt-metal-cache
unset DIFFVAE_S5_LEAN DIFFVAE_NA_KEY_PHASE
cd $B
echo "[t238-ut] host=$(hostname) $(date -u '+%F %T') UTC build=$(git -C $B rev-parse --short=11 HEAD)" | tee -a $L
timeout 540 pytest -p no:cacheprovider -q models/tt_dit/tests/unit/test_neighborhood_bricked_w_sharded.py -k "key_phase_lean" 2>&1 | tee -a $L
rc=${PIPESTATUS[0]}
echo "T238_EXIT=$rc" | tee -a $L
exit $rc
