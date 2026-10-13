#!/usr/bin/env bash
# #58819 stacked on #58816 (17_CI_AB.md), measurement only. Side trees at one depth, each a copy of this checkout's
# tt_metal/tt-llk and tt_metal/hw with its variant's files on top (llkvar/<v>/tree), each build in its own dir:
#   H   the #58819 head 2920d57951f (per tile pack under the block rows)
#   P   the head merged with #58816 and the block rows on #58816's block pack
#   Z   P with the Z walk block unpack (this checkout), built twice (bZ, bZ2: the A/A at one tree path)
#   Zr  Z with the walk replayed from the recorded half instead of the MOP
#   mainf  main dc0b7bea9ad's LLK and tests (the bits reference)
# Parts: harness A/B of the 66 datacopy block rows (rotated, 4 runs a side), counters (one counter build and run a side),
# bits (mainf against Z and Zr), functional block tests with LLK asserts on (Z, Zr), then op level (P, Z, Zr, Z2: the
# device file swapped into /work and the runtime root, own kernel cache each, rotated).
# Each run's perf_data goes to the log as base64 tar.gz lines "LLKAB_DATA <tag> <chunk>"; counters as LLKAB_DATA cnt_<v>.
set +e +o pipefail
R=${STK_ROOT:-/work}; S=$R/ci_ab; T=${STK_TMP:-/tmp/stk}; rm -rf $T; mkdir -p $T; NP=${STK_NPROD:-16}
unset RUNNER_TEMP
export CHIP_ARCH=blackhole LLK_PERF_L1_MUX_GROUP=0 PERF_KEEP_RUNS=400
echo "== stk start $(date -u +%T) $(git -C $R rev-parse HEAD 2>/dev/null) $(nproc) cpus"

if [[ -n ${STK_PY:-} ]]; then PY=$STK_PY; else
  python3 -m venv $T/venv > $T/venv.log 2>&1
  $T/venv/bin/pip install -q --disable-pip-version-check -r $R/tt_metal/tt-llk/tests/requirements.txt > $T/pip.log 2>&1
  echo "== pip rc=$? $(tail -n 2 $T/pip.log | tr '\n' ' ')"
  PY=$T/venv/bin/python
fi

want=$(sed -n "s/^sfpi_version='\(.*\)'/\1/p" $R/tt_metal/tt-llk/tests/sfpi-version)
SF=${STK_SFPI:-}
for c in $R/runtime/sfpi /opt/tenstorrent/sfpi; do
  if [[ -x $c/compiler/bin/riscv-tt-elf-g++ ]]; then
    have=$(cat $c/sfpi.version 2>/dev/null)
    echo "== sfpi candidate $c version '$have' (want $want)"
    if [[ -z $SF && ( -z $have || $have == "$want" ) ]]; then SF=$c; fi
  fi
done
if [[ -z $SF ]]; then
  (cd $R/tt_metal/tt-llk/tests && bash ./setup_testing_env.sh > $T/sfpi.log 2>&1); echo "== sfpi download rc=$? $(tail -n 1 $T/sfpi.log)"
  SF=$T/sfpi; mv $R/tt_metal/tt-llk/tests/sfpi $SF
fi
echo "== sfpi $SF: $($SF/compiler/bin/riscv-tt-elf-g++ --version 2>&1 | sed -n 1p)"

mkside() {  # mkside <v>
  local v=$1 d=$T/side_$1
  mkdir -p $d/tt_metal
  for e in $R/*; do n=$(basename $e); [[ $n == tt_metal ]] || ln -s $e $d/$n; done
  for e in $R/tt_metal/*; do n=$(basename $e); [[ $n == tt-llk || $n == hw ]] || ln -s $e $d/tt_metal/$n; done
  cp -a $R/tt_metal/tt-llk $R/tt_metal/hw $d/tt_metal/
  [[ -d $S/llkvar/$v/tree ]] && cp -a $S/llkvar/$v/tree/. $d/
  cp $S/shadow/test_bitdump_block_x.py $d/tt_metal/tt-llk/tests/python_tests/
  rm -rf $d/tt_metal/tt-llk/tests/sfpi $d/tt_metal/tt-llk/perf_data; ln -s $SF $d/tt_metal/tt-llk/tests/sfpi
  echo "== side $v: $(cd $d/tt_metal && md5sum tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_A.h tt-llk/tt_llk_blackhole/llk_lib/llk_pack.h tt-llk/tests/sources/eltwise_unary_datacopy_test.cpp | cut -c1-10 | tr '\n' ' ')"
}

cat > $T/pyrun.py <<'PYEOF'
import sys, pytest
ids = [l.strip() for l in open(sys.argv[1]) if l.strip()]
sys.exit(pytest.main(["-p", "no:cacheprovider", "--override-ini=log_cli=false", "-q", *sys.argv[2:], *ids]))
PYEOF

harness() {  # harness <side> <build> <ids> <log> [pytest args...]
  local v=$1 b=$2 ids=$3 log=$4; shift 4
  (cd $T/side_$v/tt_metal/tt-llk/tests/python_tests && \
   LLK_HOME=$T/side_$v/tt_metal/tt-llk TMPDIR=$T/b_$b timeout -s INT -k 60 ${LIM:-1800} $PY $T/pyrun.py $ids "$@") > $log 2>&1
}
build() {  # build <side> <build> <ids> <run types|-> [asserts|counters]
  mkdir -p $T/b_$2
  if [[ $4 == - ]]; then unset LLK_PERF_RUN_TYPES; else export LLK_PERF_RUN_TYPES=$4; fi
  local extra=()
  [[ ${5:-} == counters ]] && extra=(--dump-perf-counters)
  if [[ ${5:-} == asserts ]]; then (unset TT_LLK_DISABLE_ASSERTS; harness $1 $2 $3 $T/prod_$2.log --compile-producer -n $NP)
  else TT_LLK_DISABLE_ASSERTS=1 harness $1 $2 $3 $T/prod_$2.log --compile-producer -n $NP "${extra[@]}"; fi
  echo "== build $2 ($1, $(basename $3)) rc=$? $(date -u +%T): $(grep -aE '[0-9]+ (passed|failed|error)' $T/prod_$2.log | tail -n 1)"
}
elfid() {  # elfid <build a> <build b> [same]: loaded-image identity; without "same" the job stops if no ELF differs
  local r; r=$($PY $S/elfload.py $T/b_$1 $T/b_$2 | tail -n 1)
  echo "== ELF identity $1 against $2: $r"
  [[ ${3:-} == same ]] && return 0
  if [[ $r != *" 0 differ"* ]] && [[ $r == *differ* ]]; then return 0; fi
  echo "== GUARD: $1 and $2 load identical images, the A/B would be void; stopping"; exit 3
}
ship() {  # ship <tag> <dir> <glob>
  local tag=$1 dir=$2 pat=$3
  (cd $dir && ls $pat > /dev/null 2>&1) || { echo "== data $tag: none"; return; }
  (cd $dir && tar czf $T/$tag.tgz $pat); echo "== data $tag $(wc -c < $T/$tag.tgz) bytes"
  base64 -w 20000 $T/$tag.tgz | sed "s/^/LLKAB_DATA $tag /"
}
run() {  # run <side> <build> <ids> <tag> <run types|-> <limit> [counters]
  local v=$1 b=$2 ids=$3 tag=$4 rts=$5 extra=()
  [[ ${7:-} == counters ]] && extra=(--dump-perf-counters)
  if [[ $rts == - ]]; then unset LLK_PERF_RUN_TYPES; else export LLK_PERF_RUN_TYPES=$rts; fi
  TT_LLK_DISABLE_ASSERTS=1 PERF_RUN_TAG=$tag LIM=$6 harness $v $b $ids $T/cons_$tag.log --compile-consumer -o timeout_method=thread --timeout 120 "${extra[@]}"
  echo "== run $tag ($v, $b) rc=$? $(date -u +%T): $(grep -aE '[0-9]+ (passed|failed|error)' $T/cons_$tag.log | tail -n 1)"
  grep -aE '^(FAILED|ERROR) ' $T/cons_$tag.log | cut -c1-300 | sed -n 1,10p
  local pd=$T/side_$v/tt_metal/tt-llk/perf_data
  if [[ ${7:-} == counters ]]; then
    mkdir -p $T/cnt_$tag; find $pd -name '*.counters.csv' -exec mv {} $T/cnt_$tag/ \;
    ship $tag $T/cnt_$tag '*.counters.csv'
  elif [[ -d $pd/runs/$tag ]]; then
    ship $tag $pd/runs $tag/$tag.parquet
  fi
}
seqrun() {  # seqrun <ids> <prefix> <rts> <limit> <side:build>...
  local ids=$1 p=$2 rts=$3 lim=$4 n=0; shift 4
  for sb in "$@"; do n=$((n + 1)); run ${sb%%:*} ${sb##*:} $ids ${p}_${sb%%:*}_${sb##*:}_$n $rts $lim; done
}
R5=L1_TO_L1,UNPACK_ISOLATE,MATH_ISOLATE,PACK_ISOLATE,L1_CONGESTION

for v in H P Z Zr mainf; do mkside $v; done

# Harness A/B
for v in H:bH P:bP Z:bZ Z:bZ2 Zr:bZr; do build ${v%%:*} ${v##*:} $S/blk.txt $R5; done
elfid bH bP; elfid bP bZ; elfid bZ bZr; elfid bZ bZ2 same
[[ -n ${STK_DRY:-} ]] && { echo "== dry: builds only"; exit 0; }
seqrun $S/blk.txt B $R5 600 H:bH P:bP Z:bZ Zr:bZr Z:bZ2 Z:bZ2 Zr:bZr Z:bZ P:bP H:bH \
  P:bP Zr:bZr H:bH Z:bZ2 Z:bZ Z:bZ H:bH Z:bZ2 Zr:bZr P:bP

# Counters: one counter build and two runs a side
for v in H:cH P:cP Z:cZ Zr:cZr; do build ${v%%:*} ${v##*:} $S/cnt.txt L1_TO_L1,UNPACK_ISOLATE,PACK_ISOLATE counters; done
for v in H:cH P:cP Z:cZ Zr:cZr Zr:cZr Z:cZ P:cP H:cH; do
  n=$((${n:-0} + 1)); run ${v%%:*} ${v##*:} $S/cnt.txt C_${v%%:*}_$n L1_TO_L1,UNPACK_ISOLATE,PACK_ISOLATE 600 counters
done

# Bits: main's per tile path against the block paths
unset LLK_PERF_RUN_TYPES
for v in mainf:dmf Z:dZ Zr:dZr; do build ${v%%:*} ${v##*:} $S/bits.txt -; done
for v in mainf:dmf Z:dZ Zr:dZr; do
  D=$T/bits_${v%%:*}; mkdir -p $D
  BITDUMP_DIR=$D TT_LLK_DISABLE_ASSERTS=1 LIM=900 harness ${v%%:*} ${v##*:} $S/bits.txt $T/cons_bits_${v%%:*}.log --compile-consumer -o timeout_method=thread --timeout 120
  echo "== bits ${v%%:*} rc=$? $(date -u +%T): $(grep -aE '[0-9]+ (passed|failed|error)' $T/cons_bits_${v%%:*}.log | tail -n 1), $(ls $D | wc -l) dumps"
  (cd $D && sha256sum * | sed "s/^/LLKAB_SHA ${v%%:*} /")
done
for v in Z Zr; do echo "== bits mainf against $v: $(cd $T/bits_mainf && for f in *; do cmp -s $f $T/bits_$v/$f && echo equal || echo differ; done | sort | uniq -c | tr '\n' ' ')"; done

# Functional block tests, LLK asserts on
for v in Z:fZ Zr:fZr; do
  build ${v%%:*} ${v##*:} $S/func.txt - asserts
  LIM=1500 harness ${v%%:*} ${v##*:} $S/func.txt $T/cons_func_${v%%:*}.log --compile-consumer -o timeout_method=thread --timeout 120
  echo "== functional ${v%%:*}, LLK asserts on, rc=$? $(date -u +%T): $(grep -aE '[0-9]+ (passed|failed|error)' $T/cons_func_${v%%:*}.log | tail -n 1)"
  grep -aE '^(FAILED|ERROR) ' $T/cons_func_${v%%:*}.log | cut -c1-300 | sed -n 1,20p
done
echo "== harness end $(date -u +%T)"

[[ -n ${NO_OL:-} ]] && exit 0
OL_VARS="P Z Zr Z2" bash $S/run_stack_ol.sh
echo "== stk end $(date -u +%T)"
