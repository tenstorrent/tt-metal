#!/usr/bin/env bash
# Scratch, not for merge. Functional slice for _llk_unpack_AB_ callers: arm S (early poll) then the failures again on M.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1
S=/work/ci_ab; OUT=/work/generated/fn; mkdir -p $OUT
F=tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB.h
SP=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
PKS=$(find "$SP" -path "*/tt_llk_blackhole/llk_lib/llk_unpack_AB.h" -type f 2>/dev/null)
use() { cp $S/v/$1/$F /work/$F; for p in $PKS; do cp $S/v/$1/$F $p; done; echo "== use $1 $(md5sum < /work/$F | cut -c1-8) pkg $(echo $PKS | wc -w)"; export TT_METAL_CACHE=$OUT/cache_$1; mkdir -p $TT_METAL_CACHE; }
U=tests/ttnn/unit_tests/operations; L=/work/tt_metal/tt-llk/tests/python_tests
declare -A GRP
GRP[bng]="$U/eltwise/test_binary_ng_bcast_fp32_dest_acc.py $U/eltwise/test_binary_ng_sharded_fp32_batch.py $U/eltwise/test_binary_ng_program_cache.py $U/eltwise/test_binary_ng_typecast.py $U/eltwise/test_binary_ng_width_padded_stride.py $U/eltwise/test_binary_ng_activation_mixed_dtype.py"
GRP[bcast]="$U/eltwise/test_binary_bcast.py $U/eltwise/test_elt_binary.py $U/eltwise/test_binary_fp32.py $U/eltwise/test_binaryng_fp32.py $U/eltwise/test_binary_category1_bfloat16.py"
GRP[moe]="$U/experimental/deepseek/test_generalized_moe_gate_program_cache.py $U/experimental/deepseek/test_deepseek_moe_gate_program_cache.py"
GRP[sdpa]="$U/sdpa/test_sdpa_decode.py $U/sdpa/test_sdpa_prefill.py"
GRP[norm]="$U/fused/test_layer_norm.py $U/fused/test_rms_norm.py $U/fused/test_softmax.py"
g() { local arm=$1 name=$2; shift 2; local T0=$(date +%s)
  timeout -s INT -k 60 ${GT:-1500} python3 -m pytest -q -rfE -p no:cacheprovider -o timeout_method=thread "$@" > $OUT/${arm}_$name.log 2>&1
  echo "GRP $arm $name rc=$? $(( $(date +%s) - T0 ))s: $(grep -aE '[0-9]+ (passed|failed|error|skipped)' $OUT/${arm}_$name.log | tail -1 | cut -c1-160)"
  grep -a -E '^(FAILED|ERROR) ' $OUT/${arm}_$name.log | cut -d' ' -f2 | sort -u > $OUT/fail_${arm}_$name.txt
  head -8 $OUT/fail_${arm}_$name.txt | sed "s/^/FAILID $arm $name /"
  grep -a -E "TT_THROW|TENSIX TIMED OUT|hang" $OUT/${arm}_$name.log | head -2 | cut -c1-200 | sed "s/^/HANG $arm $name /"; }
use S
for n in moe bng bcast norm sdpa; do g S $n ${GRP[$n]}; done
use M
for n in moe bng bcast norm sdpa; do
  if [[ -s $OUT/fail_S_$n.txt ]]; then mapfile -t FF < <(head -60 $OUT/fail_S_$n.txt); g M $n "${FF[@]}"; fi
done
use S
# tt-llk tests on the device
python3 -m venv /tmp/llkvenv && . /tmp/llkvenv/bin/activate
pip install -q -r $L/../requirements.txt > /tmp/pip.log 2>&1; echo "== pip rc=$?"
(cd $L/.. && ./setup_testing_env.sh > /tmp/setup.log 2>&1); echo "== setup rc=$?"
export CHIP_ARCH=blackhole LLK_HOME=/work/tt_metal/tt-llk
cd $L
LK="test_generalized_moe_gate.py test_deepseek_moe_gate.py test_mul_reduce_scalar_chunked.py test_sdpa_reinits.py test_tilize_calculate_untilize_L1.py"
for arm in S M; do
  [[ $arm == M ]] && use M
  for t in $LK; do
    if [[ $arm == M && ! -s $OUT/lfail_S_$t.txt ]]; then continue; fi
    mapfile -t IDS < <([[ $arm == M ]] && head -60 $OUT/lfail_S_$t.txt || echo $t)
    mkdir -p /tmp/llkb_${arm}_$t
    T0=$(date +%s)
    export RUNNER_TEMP=/tmp/llkb_${arm}_$t TMPDIR=/tmp/llkb_${arm}_$t
    timeout -s INT -k 60 2400 python -m pytest --compile-producer -n ${NPROD:-8} -p no:cacheprovider --override-ini=log_cli=false -q "${IDS[@]}" > $OUT/lp_${arm}_$t.log 2>&1
    timeout -s INT -k 60 2400 python -m pytest --compile-consumer -p no:cacheprovider --override-ini=log_cli=false -o timeout_method=thread -q -rfE "${IDS[@]}" > $OUT/l_${arm}_$t.log 2>&1
    echo "LLK $arm $t rc=$? $(( $(date +%s) - T0 ))s: $(grep -aE '[0-9]+ (passed|failed|error|skipped)' $OUT/l_${arm}_$t.log | tail -1 | cut -c1-160)"
    grep -a -E '^(FAILED|ERROR) ' $OUT/l_${arm}_$t.log | cut -d' ' -f2 | sort -u > $OUT/lfail_${arm}_$t.txt
    head -6 $OUT/lfail_${arm}_$t.txt | sed "s/^/LFAILID $arm $t /"
  done
done
echo "== done"
