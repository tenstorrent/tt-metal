#!/bin/bash
# t374: conv3d row ring A/B, bit-exact check + trace timing, one layer per job, 2x4 submesh of the 4x8 mesh,
# on ttp/t374 8a365da8129 (own Release build in /var/tmp/fasth3/t374/b). Usage: run374.sh <layer: s3_res|s2_res|s4_res>
# Broker: -e env374.yaml -t 600 (unmeasured).
if [ -z "$INNER" ]; then
  INNER=1 setsid bash "$0" "$@" & PG=$!
  trap 'kill -TERM -- -$PG 2>/dev/null; sleep 5; kill -KILL -- -$PG 2>/dev/null' EXIT
  trap 'exit 143' TERM; trap 'exit 130' INT
  wait $PG; exit $?
fi
set -o pipefail
LAYER=${1:?layer}
F=/var/tmp/fasth3; T=$F/t374; W=$T/b; WANT=8a365da8129; OUT=$T/out_$LAYER
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le 70 ] || { echo "[t374] df / $use% > 70%"; exit 5; }
gb=$(timeout 120 du -sxBG $F | cut -f1 | tr -dc 0-9); [ "${gb:-999}" -le 150 ] || { echo "[t374] $F ${gb}G > 150G"; exit 5; }
rm -rf $OUT; mkdir -p $OUT $F/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp HF_HUB_OFFLINE=1
source $F/t48/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools PYTHONDONTWRITEBYTECODE=1 TT_METAL_CACHE=$T/jit
unset TT_CONV3D_ROW_RING
cd $OUT || exit 3
HEAD=$(git -C $W rev-parse --short=11 HEAD 2>/dev/null)
echo "[t374] host=$(hostname) commit=$HEAD layer=$LAYER $(date -u '+%F %T') UTC" | tee run.log
[ "$HEAD" = $WANT ] || { echo "[t374] wrong commit $HEAD != $WANT" | tee -a run.log; exit 3; }
T0=$(date +%s)
python -u -m pytest -c $W/pytest.ini --rootdir=$W -sv -p no:cacheprovider --timeout=560 \
  "$W/models/tt_dit/tests/models/ltx/test_conv3d_row_ring.py::test_conv3d_row_ring_bit_exact" -k "$LAYER" 2>&1 | tee -a run.log
rc=${PIPESTATUS[0]}
echo "[t374] process wall $(( $(date +%s) - T0 )) s" | tee -a run.log
echo "[t374] AICLK clamp warnings: $(grep -c "AICLK failed to settle" run.log)" | tee -a run.log
echo "T374_EXIT=$rc" | tee -a run.log
exit $rc
