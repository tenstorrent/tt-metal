#!/usr/bin/env bash
# Scratch, not for merge. topk_large_indices op level under the device profiler: three runs of the case set, ids per run printed.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 OPS_NMEAS=8 HWLOCK_HELD=ci
S=/work/ci_ab; OUT=/work/generated/glmg; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s')"
for r in 1 2 3; do
  export TT_METAL_CACHE=$OUT/cache_$r; o=$OUT/r$r; mkdir -p $o
  timeout -s INT -k 60 1800 python3 -m tracy -r -p --no-web-server -o $o $S/prof_ops_xl.py > $o.log 2>&1
  echo "== run $r rc=$? $(grep -c 'CASE .* ok' $o.log) ok, $(grep -c 'CASE .* FAILED' $o.log) failed"
  python3 $S/sum_ops.py $o X $r
done
echo "== done $(date -u +%FT%TZ)"
