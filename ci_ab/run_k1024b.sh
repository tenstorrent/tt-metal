#!/usr/bin/env bash
# Scratch, not for merge. topk_large_indices op level under the device profiler, one job, two arms interleaved A B B A B A A B
# (A = the op branch kernels, B = K 1024 split from two chunks); ids per run printed as "OPS OP <arm> <run> <case> ...".
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 OPS_NMEAS=8 HWLOCK_HELD=ci
export OPS_CASES="csa_r640_n3328_k1024 leaf_r100_n16384_k512 bounded_r2_n102400_k1536_v56320 r640_n4096_k1024 csa_r640_n8192_k1024 csa_r160_n32768_k1024 leaf_r40_n32768_k1024 k1024_r640_n2048 csa_r640_n2560_k1024_valid1280"; S=/work/ci_ab; K=/work/ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/kernels; OUT=/work/generated/k1024b; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s')"
r=0
for arm in A B B A B A A B B A A B; do
  r=$((r+1)); cp $S/variants/$arm/compute.cpp $S/variants/$arm/topk_large_indices_split.hpp $K/
  export TT_METAL_CACHE=$OUT/cache_$arm$r; o=$OUT/r$r; mkdir -p $o
  timeout -s INT -k 60 1800 python3 -m tracy -r -p --no-web-server -o $o $S/prof_ops_xl.py > $o.log 2>&1
  echo "== run $r arm $arm rc=$? $(grep -c 'CASE .* ok' $o.log) ok, $(grep -c 'CASE .* FAILED' $o.log) failed"
  python3 $S/sum_ops.py $o $arm $r
done
echo "== done $(date -u +%FT%TZ)"
