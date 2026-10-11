#!/bin/bash
# t380 broker job (blx01): test_conv3d_vol2col_ring.py on ttp/t380 (own build t380/b). Opens the 4x8 mesh, uses a
# create_submesh(2,4): a 5*4*4 blocking must fail on the host, 3*4*4 and 3*8*8 must still run (PCC >= 0.999).
# Broker: -e env380.yaml -t 600. pytest runs under timeout --foreground -k 10 (no setsid below the timeout).
if [ -z "$INNER" ]; then
  INNER=1 setsid bash "$0" "$@" & PG=$!
  trap 'kill -TERM -- -$PG 2>/dev/null; sleep 5; kill -KILL -- -$PG 2>/dev/null' EXIT
  trap 'exit 143' TERM; trap 'exit 130' INT
  wait $PG; exit $?
fi
set -o pipefail
F=/var/tmp/fasth3; T=$F/t380; W=$T/b; WANT=${1:?commit}; OUT=$T/out
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le 70 ] || { echo "[t380] df / $use% > 70%"; exit 5; }
gb=$(timeout 120 du -sxBG $F | cut -f1 | tr -dc 0-9); [ "${gb:-0}" -le 150 ] || { echo "[t380] $F ${gb}G > 150G"; exit 5; }
rm -rf $OUT; mkdir -p $OUT $F/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp
source $F/t48/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools PYTHONDONTWRITEBYTECODE=1 TT_METAL_CACHE=$T/jit
cd $OUT || exit 3
TF=$W/models/tt_dit/tests/unit/test_conv3d_vol2col_ring.py
HEAD=$(git -C $W rev-parse --short=11 HEAD 2>/dev/null)
echo "[t380] host=$(hostname) commit=$HEAD dirty=$(git -C $W status --porcelain -uno | wc -l) $(date -u '+%F %T') UTC" | tee run.log
[ "$HEAD" = "$WANT" ] || { echo "[t380] wrong commit $HEAD != $WANT" | tee -a run.log; exit 3; }
[ -r $W/ttnn/ttnn/_ttnn.so ] || { echo "[t380] no build" | tee -a run.log; exit 3; }
timeout --foreground -k 10 540 python -u -m pytest -c $W/pytest.ini --rootdir=$W -sv -p no:cacheprovider --timeout=520 \
  "$TF" 2>&1 | tee -a run.log
rc=${PIPESTATUS[0]}
echo "T380_EXIT=$rc $(date -u '+%F %T') UTC" | tee -a run.log
exit $rc
