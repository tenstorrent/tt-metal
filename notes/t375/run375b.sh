#!/bin/bash
# t375 job b: micro-benchmark of the conv VAE fused RMSNorm+SiLU variants (bench_norm.py) on the full 4x8 mesh,
# using the built t48 tree (bf7db12a149; same norm kernels as t48 head 90ed8257bac) read-only. Own JIT cache.
# Broker: -e env375.yaml -t 600 (unmeasured).
if [ -z "$INNER" ]; then
  INNER=1 setsid bash "$0" "$@" & PG=$!
  trap 'kill -TERM -- -$PG 2>/dev/null; sleep 5; kill -KILL -- -$PG 2>/dev/null' EXIT
  trap 'exit 143' TERM; trap 'exit 130' INT
  wait $PG; exit $?
fi
set -o pipefail
F=/var/tmp/fasth3; T=$F/t375; W=$F/t48; OUT=$T/out_b
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le 70 ] || { echo "[t375] df / $use% > 70%"; exit 5; }
gb=$(timeout 120 du -sxBG /var/tmp/fasth3 | cut -f1 | tr -dc 0-9); [ "${gb:-0}" -le 150 ] || { echo "[t375] /var/tmp/fasth3 ${gb}G > 150G"; exit 5; }
rm -rf $OUT; mkdir -p $OUT $F/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp
source $W/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools PYTHONDONTWRITEBYTECODE=1 TT_METAL_CACHE=$T/jit
export BENCH_SHAPES=${BENCH_SHAPES:-res128,res256,res512b}
cd $OUT || exit 3
echo "[t375] host=$(hostname) tree=$(git -C $W rev-parse --short=11 HEAD) $(date -u '+%F %T') UTC shapes=$BENCH_SHAPES" | tee run.log
python -u $T/bench_norm.py 2>&1 | tee -a run.log
rc=${PIPESTATUS[0]}
echo "T375_EXIT=$rc" | tee -a run.log
exit $rc
