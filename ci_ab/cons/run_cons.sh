#!/usr/bin/env bash
# Scratch (not for merge): consumers of the SCALAR MAX tile path that pack without the reduce mask, ttnn min over H and W
# (reduce_hw_neg and the HW program factories) and the reload test. Outputs hashed per test; compare two refs' logs.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1
S=/work/ci_ab/cons; OUT=/work/generated/cons; rm -rf $OUT; mkdir -p $OUT
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s' | cut -c1-70)"
export TT_METAL_CACHE=$OUT/cache; mkdir -p $TT_METAL_CACHE
for r in 1 2; do
  OUT_HASH=$OUT/hash_r$r.tsv AB_NODES_FILE=$S/nodes.txt timeout -s INT -k 60 3000 python3 $S/run_nodes.py > $OUT/run_r$r.log 2>&1; rc=$?
  echo "== run $r rc=$rc: $(grep -aE '[0-9]+ (passed|failed)' $OUT/run_r$r.log | tail -1)"
  grep -aE '^(FAILED|ERROR)' $OUT/run_r$r.log | cut -c1-200 | head -20
done
echo "== hashes r1 vs r2: $(paste <(sort $OUT/hash_r1.tsv) <(sort $OUT/hash_r2.tsv) | awk -F'\t' '{n++; if ($2==$4) s++} END {print s+0" identical of "n+0}')"
echo "== hash lines: $(wc -l < $OUT/hash_r1.tsv)"
echo "== HASHES BEGIN"; sort $OUT/hash_r1.tsv; echo "== HASHES END"
echo "== reload: $(grep -aE 'test_reduce_scalar_max_reload.*(PASSED|FAILED)|reload' $OUT/run_r1.log | tail -3)"
