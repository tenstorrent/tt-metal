#!/usr/bin/env bash
# Scratch, not for merge. transpose_dest STALLWAIT without the MATH idle wait (arm F4) against the original (D0), ttnn level:
# the tests of the transpose_dest callers (layernorm welford, distributed layernorm pre allgather welford, dit rmsnorm fused, topk_large_indices)
# run on F4; failures rerun on D0 to tell a regression from a test that fails anyway. The compiled kernels' hashes show the swap reached the JIT.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work
S=/work/ci_ab; OUT=/work/generated/tb90d; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
HDR=tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_transpose_dest.h
TARGETS="/work/$HDR"
for d in $(python3 -c "import ttnn,os;print(os.path.dirname(ttnn.__file__))" 2>/dev/null) /work/runtime /work/ttnn; do
  f=$(find $d -maxdepth 7 -name llk_math_transpose_dest.h -path '*blackhole*' 2>/dev/null | head -3); TARGETS="$TARGETS $f"
done
TARGETS=$(printf '%s\n' $TARGETS | sort -u); echo "== targets: $TARGETS"
setarm() { for t in $TARGETS; do cp $S/var/$1/llk_math_transpose_dest.h $t; done; echo "== arm $1 $(md5sum /work/$HDR | cut -c1-8)"; }
TESTS=(
 "tests/ttnn/nightly/unit_tests/operations/fused/test_layernorm.py -k welford"
 "tests/ttnn/nightly/unit_tests/operations/fused/test_distributed_layernorm_pre_allgather.py"
 "tests/ttnn/nightly/unit_tests/operations/experimental/test_dit_rms_norm_unary_fused.py"
 "tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_large_indices.py"
)
n=0
for t in "${TESTS[@]}"; do
  n=$((n+1)); setarm F4; export TT_METAL_CACHE=$OUT/cache_F4_$n
  timeout -s INT -k 60 1500 python3 -m pytest -q -p no:cacheprovider -o timeout_method=thread --maxfail=30 $t > $OUT/F4_$n.log 2>&1
  echo "== F4 [$t]: $(grep -aE '[0-9]+ (passed|failed|skipped|error)' $OUT/F4_$n.log | tail -1)"
  grep -aE '^(FAILED|ERROR) ' $OUT/F4_$n.log | sed 's/ - .*//' | sort -u | head -40 > $OUT/fail_$n.txt; cat $OUT/fail_$n.txt | cut -c1-200
  python3 $S/elfhash.py F4_$n $TT_METAL_CACHE welford rmsnorm topk transpose 2>&1 | cut -c1-160 | head -4 | sed 's/^/ELFH /'
  if [[ -s $OUT/fail_$n.txt ]]; then
    setarm D0; export TT_METAL_CACHE=$OUT/cache_D0_$n
    timeout -s INT -k 60 1500 python3 -m pytest -q -p no:cacheprovider -o timeout_method=thread $(sed 's/^[A-Z]* //' $OUT/fail_$n.txt | tr '\n' ' ') > $OUT/D0_$n.log 2>&1
    echo "== rerun on D0: $(grep -aE '[0-9]+ (passed|failed|skipped|error)' $OUT/D0_$n.log | tail -1)"
    python3 $S/elfhash.py D0_$n $TT_METAL_CACHE welford rmsnorm topk transpose 2>&1 | cut -c1-160 | head -4 | sed 's/^/ELFH /'
  fi
done
echo "== done"
