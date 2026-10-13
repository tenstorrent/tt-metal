#!/usr/bin/env bash
# #59135 stacked on #58819 (17_CI_AB.md), measurement only. Side trees at one depth, each a copy of this checkout's
# tt_metal/tt-llk and tt_metal/hw with its variant's files on top (llkvar/<v>/tree), each build in its own dir:
#   TH  the #59135 head 682177640f7 (block unpack with a CFGSHIFTMASK per tile, per tile pack, 4096 byte tile ring)
#   TP  the head merged with #58819 and #58816, the block rows' tiles back to back and on #58816's block pack
#   TW  TP with the W walk transpose block unpack (this checkout), built twice (bW, bW2: the A/A at one tree path)
#   TWr TW with the tile body replayed from a recorded copy instead of the MOP
# Parts: harness A/B of the transpose block rows (rotated, 4 runs a side), counters (one counter build, two runs a side),
# bits (block and alternating forms against the per tile form on every bf16 and fp16 pattern, seeded Bfp8_b and Int8),
# functional block tests with LLK asserts on (TW, TWr).
# Each run's perf_data goes to the log as base64 tar.gz lines "LLKAB_DATA <tag> <chunk>"; counters as LLKAB_DATA cnt_<v>.
set +e +o pipefail
R=${STK_ROOT:-/work}; S=$R/ci_ab; T=${STK_TMP:-/tmp/stktp}; rm -rf $T; mkdir -p $T; NP=${STK_NPROD:-16}
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
  cp $S/shadow/test_transpose_blk_bits_x.py $d/tt_metal/tt-llk/tests/python_tests/
  cp $S/shadow/transpose_blk_bits_x.cpp $d/tt_metal/tt-llk/tests/sources/
  rm -rf $d/tt_metal/tt-llk/tests/sfpi $d/tt_metal/tt-llk/perf_data; ln -s $SF $d/tt_metal/tt-llk/tests/sfpi
  echo "== side $v: $(cd $d/tt_metal && md5sum tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_A.h tt-llk/tt_llk_blackhole/common/inc/cunpack_common.h tt-llk/tests/sources/unpack_transpose_perf.cpp | cut -c1-10 | tr '\n' ' ')"
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
R2=L1_TO_L1,UNPACK_ISOLATE

for v in TH TP TW TWr; do mkside $v; done

# Harness A/B
for v in TH:bH TP:bP TW:bW TW:bW2 TWr:bWr; do build ${v%%:*} ${v##*:} $S/tblk.txt $R2; done
elfid bH bP; elfid bP bW; elfid bW bWr; elfid bW bW2 same
[[ -n ${STK_DRY:-} ]] && { echo "== dry: builds only"; exit 0; }
seqrun $S/tblk.txt B $R2 300 TH:bH TP:bP TW:bW TWr:bWr TW:bW2 TW:bW2 TWr:bWr TW:bW TP:bP TH:bH \
  TP:bP TWr:bWr TH:bH TW:bW2 TW:bW TW:bW TH:bH TW:bW2 TWr:bWr TP:bP

# Counters: one counter build and two runs a side
n=0
for v in TH:cH TP:cP TW:cW TWr:cWr; do build ${v%%:*} ${v##*:} $S/tblk.txt $R2 counters; done
for v in TH:cH TP:cP TW:cW TWr:cWr TWr:cWr TW:cW TP:cP TH:cH; do
  n=$((n + 1)); run ${v%%:*} ${v##*:} $S/tblk.txt C_${v%%:*}_$n $R2 300 counters
done

# Bits: block (1) and alternating (4) forms against the per tile form (0) of the same side
unset LLK_PERF_RUN_TYPES
for v in TW:dW TWr:dWr; do
  build ${v%%:*} ${v##*:} $S/bits.txt -
  D=$T/bits_${v%%:*}; mkdir -p $D
  BITDUMP_DIR=$D TT_LLK_DISABLE_ASSERTS=1 LIM=900 harness ${v%%:*} ${v##*:} $S/bits.txt $T/cons_bits_${v%%:*}.log --compile-consumer -o timeout_method=thread --timeout 120
  echo "== bits ${v%%:*} rc=$? $(date -u +%T): $(grep -aE '[0-9]+ (passed|failed|error)' $T/cons_bits_${v%%:*}.log | tail -n 1), $(ls $D | wc -l) dumps"
  (cd $D && sha256sum * | sed "s/^/LLKAB_SHA ${v%%:*} /")
  echo "== bits ${v%%:*} forms: $($PY $S/bits_cmp.py $D | tail -n 1)"
  $PY $S/bits_cmp.py $D | grep -v " 0 differ" | sed -n 1,20p
done
echo "== bits TW against TWr: $($PY $S/bits_cmp.py $T/bits_TW $T/bits_TWr | tail -n 1)"

# Functional block tests, LLK asserts on
for v in TW:fW TWr:fWr; do
  build ${v%%:*} ${v##*:} $S/func.txt - asserts
  LIM=900 harness ${v%%:*} ${v##*:} $S/func.txt $T/cons_func_${v%%:*}.log --compile-consumer -o timeout_method=thread --timeout 120
  echo "== functional ${v%%:*}, LLK asserts on, rc=$? $(date -u +%T): $(grep -aE '[0-9]+ (passed|failed|error)' $T/cons_func_${v%%:*}.log | tail -n 1)"
  grep -aE '^(FAILED|ERROR) ' $T/cons_func_${v%%:*}.log | cut -c1-300 | sed -n 1,20p
done
echo "== stk end $(date -u +%T)"
