#!/usr/bin/env bash
# #58819 regression audit: tt-llk harness A/B on one CI Blackhole machine (17_CI_AB.md). Side trees at the same depth, each a
# copy of this checkout's tt_metal/tt-llk and tt_metal/hw with its variant's files on top (llkvar/<v>/tree), each with its
# own build dir: head (the merged head), main (main f0832760bf5's LLK with the head's tests), mainf (main's full tree, for the
# bits), hn (head with two NOPs before the unpack INIT), hf1 (head with the tilize block body recorded at the first block
# call), pair (head merged with #58816 and a block pack under the datacopy block test; measurement only).
# JOB=A: candidates and gap families (c12) main against head, ABBA x2, and the hn probe against head, ABBA x2.
# JOB=B: hf1 against head and main on the tilize rows (f1), pair against head (pair), and unless PERF_ONLY=1 the bit dumps
# (mainf, head, hf1) and the functional tilize and datacopy tests with LLK asserts on for hf1.
# Each run's perf_data goes to the log as base64 tar.gz lines "LLKAB_DATA <tag> <chunk>"; bit dumps as sha256 lines.
# usage: JOB=A|B [PERF_ONLY=1] bash ci_ab/run_udc_ab.sh
set +e +o pipefail
R=${LLKAB_ROOT:-/work}; S=$R/ci_ab; T=${LLKAB_TMP:-/tmp/llkab}; rm -rf $T; mkdir -p $T
unset RUNNER_TEMP
export CHIP_ARCH=blackhole LLK_PERF_L1_MUX_GROUP=0 PERF_KEEP_RUNS=400
JOB=${JOB:-A}
echo "== llkab job $JOB start $(date -u +%T) $(git -C $R rev-parse HEAD 2>/dev/null) $(nproc) cpus"

if [[ -n ${LLKAB_PY:-} ]]; then PY=$LLKAB_PY; else
  python3 -m venv $T/venv > $T/venv.log 2>&1
  $T/venv/bin/pip install -q --disable-pip-version-check -r $R/tt_metal/tt-llk/tests/requirements.txt > $T/pip.log 2>&1
  echo "== pip rc=$? $(tail -n 2 $T/pip.log | tr '\n' ' ')"
  PY=$T/venv/bin/python
fi

want=$(sed -n "s/^sfpi_version='\(.*\)'/\1/p" $R/tt_metal/tt-llk/tests/sfpi-version)
SF=${LLKAB_SFPI:-}
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

mkside() {  # mkside <v>: root symlinks to /work, copies of tt_metal/tt-llk and tt_metal/hw, the variant's files on top
  local v=$1 d=$T/side_$1
  mkdir -p $d/tt_metal
  for e in $R/*; do n=$(basename $e); [[ $n == tt_metal ]] || ln -s $e $d/$n; done
  for e in $R/tt_metal/*; do n=$(basename $e); [[ $n == tt-llk || $n == hw ]] || ln -s $e $d/tt_metal/$n; done
  cp -a $R/tt_metal/tt-llk $R/tt_metal/hw $d/tt_metal/
  [[ -d $S/llkvar/$v/tree ]] && cp -a $S/llkvar/$v/tree/. $d/
  cp $S/shadow/test_bitdump_block_x.py $S/shadow/test_bitdump_bcast_x.py $d/tt_metal/tt-llk/tests/python_tests/
  cp $S/shadow/unpack_A_full_test.cpp $d/tt_metal/tt-llk/tests/sources/
  rm -rf $d/tt_metal/tt-llk/tests/sfpi $d/tt_metal/tt-llk/perf_data; ln -s $SF $d/tt_metal/tt-llk/tests/sfpi
  echo "== side $v: $(cd $d/tt_metal && md5sum tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_A.h tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_tilize.h tt-llk/tt_llk_blackhole/llk_lib/llk_pack.h tt-llk/tests/sources/eltwise_unary_datacopy_test.cpp | cut -c1-10 | tr '\n' ' ')"
}

cat > $T/pyrun.py <<'PYEOF'
import sys, pytest
ids = [l.strip() for l in open(sys.argv[1]) if l.strip()]
sys.exit(pytest.main(["-p", "no:cacheprovider", "--override-ini=log_cli=false", "-q", *sys.argv[2:], *ids]))
PYEOF

harness() {  # harness <side> <build> <ids> <log> [pytest args...]; env from the caller
  local v=$1 b=$2 ids=$3 log=$4; shift 4
  (cd $T/side_$v/tt_metal/tt-llk/tests/python_tests && \
   LLK_HOME=$T/side_$v/tt_metal/tt-llk TMPDIR=$T/b_$b timeout -s INT -k 60 ${LIM:-1800} $PY $T/pyrun.py $ids "$@") > $log 2>&1
}
build() {  # build <side> <build> <ids> <run types|-> [asserts]
  mkdir -p $T/b_$2
  if [[ $4 == - ]]; then unset LLK_PERF_RUN_TYPES; else export LLK_PERF_RUN_TYPES=$4; fi
  if [[ ${5:-} == asserts ]]; then (unset TT_LLK_DISABLE_ASSERTS; harness $1 $2 $3 $T/prod_$2.log --compile-producer -n ${LLKAB_NPROD:-16})
  else TT_LLK_DISABLE_ASSERTS=1 harness $1 $2 $3 $T/prod_$2.log --compile-producer -n ${LLKAB_NPROD:-16}; fi
  echo "== build $2 ($1, $(basename $3)) rc=$? $(date -u +%T): $(grep -aE '[0-9]+ (passed|failed|error)' $T/prod_$2.log | tail -n 1)"
}
elfid() {  # elfid <build a> <build b>: loaded-image identity of the ELFs paired by variant; the job stops if no ELF differs
  local r; r=$($PY $S/elfload.py $T/b_$1 $T/b_$2 | tail -n 1)
  echo "== ELF identity $1 against $2: $r"
  if [[ $r != *" 0 differ"* ]] && [[ $r == *differ* ]]; then return 0; fi
  echo "== GUARD: $1 and $2 load identical images, the A/B would be void; stopping"; exit 3
}
run() {  # run <side> <build> <ids> <tag> <run types|-> <limit>
  [[ -n ${LLKAB_DRY:-} ]] && return
  local v=$1 b=$2 ids=$3 tag=$4 rts=$5
  if [[ $rts == - ]]; then unset LLK_PERF_RUN_TYPES; else export LLK_PERF_RUN_TYPES=$rts; fi
  TT_LLK_DISABLE_ASSERTS=1 PERF_RUN_TAG=$tag LIM=$6 harness $v $b $ids $T/cons_$tag.log --compile-consumer -o timeout_method=thread --timeout 120
  echo "== run $tag ($v, $b) rc=$? $(date -u +%T): $(grep -aE '[0-9]+ (passed|failed|error)' $T/cons_$tag.log | tail -n 1)"
  grep -aE '^(FAILED|ERROR) ' $T/cons_$tag.log | cut -c1-300 | sed -n 1,10p
  local pd=$T/side_$v/tt_metal/tt-llk/perf_data/runs
  if [[ -d $pd/$tag ]]; then
    tar czf $T/$tag.tgz -C $pd $tag/$tag.parquet; echo "== data $tag $(wc -c < $T/$tag.tgz) bytes"
    base64 -w 20000 $T/$tag.tgz | sed "s/^/LLKAB_DATA $tag /"
  fi
}
seqrun() {  # seqrun <ids> <prefix> <rts> <limit> <side:build>...: runs numbered in the given order
  local ids=$1 p=$2 rts=$3 lim=$4 n=0; shift 4
  for sb in "$@"; do n=$((n + 1)); run ${sb%%:*} ${sb##*:} $ids ${p}_${sb%%:*}_$n $rts $lim; done
}
R4=L1_TO_L1,UNPACK_ISOLATE,MATH_ISOLATE,PACK_ISOLATE

if [[ $JOB == A ]]; then
  mkside head; mkside main; mkside hn
  build main c12m $S/c12.txt $R4; build head c12h $S/c12.txt $R4; build hn hnn $S/hn.txt $R4
  elfid c12h c12m; elfid c12h hnn
  seqrun $S/c12.txt A $R4 900 main:c12m head:c12h head:c12h main:c12m main:c12m head:c12h head:c12h main:c12m
  seqrun $S/hn.txt N $R4 300 head:c12h hn:hnn hn:hnn head:c12h head:c12h hn:hnn hn:hnn head:c12h
elif [[ $JOB == E ]]; then
  mkside main; mkside head; mkside hf3
  for v in main:f1m head:f1h hf3:f1x; do build ${v%%:*} ${v##*:} $S/f1.txt $R4; done
  elfid f1h f1m; elfid f1h f1x
  seqrun $S/f1.txt E $R4 600 main:f1m head:f1h hf3:f1x hf3:f1x head:f1h main:f1m hf3:f1x main:f1m head:f1h head:f1h main:f1m hf3:f1x
  if [[ -z ${PERF_ONLY:-} ]]; then
    mkside mainf
    for v in mainf:bdmf hf3:bdmx; do build ${v%%:*} ${v##*:} $S/bitdump_merge.txt -; done
    for v in mainf:bdmf hf3:bdmx; do
      D=$T/bits_${v%%:*}; mkdir -p $D
      BITDUMP_DIR=$D TT_LLK_DISABLE_ASSERTS=1 LIM=600 harness ${v%%:*} ${v##*:} $S/bitdump_merge.txt $T/cons_bits_${v%%:*}.log --compile-consumer -o timeout_method=thread --timeout 120
      echo "== bits ${v%%:*} rc=$? $(date -u +%T): $(grep -aE '[0-9]+ (passed|failed|error)' $T/cons_bits_${v%%:*}.log | tail -n 1), $(ls $D | wc -l) dumps"
      (cd $D && sha256sum * | sed "s/^/LLKAB_SHA ${v%%:*} /")
    done
    echo "== bits mainf against hf3: $(cd $T/bits_mainf && for f in *; do cmp -s $f $T/bits_hf3/$f && echo equal || echo differ; done | sort | uniq -c | tr '\n' ' ')"
    build hf3 fax $S/func_f1.txt - asserts
    unset LLK_PERF_RUN_TYPES
    LIM=1500 harness hf3 fax $S/func_f1.txt $T/cons_func.log --compile-consumer -o timeout_method=thread --timeout 120
    echo "== functional hf3, LLK asserts on, rc=$? $(date -u +%T): $(grep -aE '[0-9]+ (passed|failed|error)' $T/cons_func.log | tail -n 1)"
    grep -aE '^(FAILED|ERROR) ' $T/cons_func.log | cut -c1-400 | sed -n 1,40p
  fi
elif [[ $JOB == C ]]; then
  mkside main; mkside hf1; mkside ht0; mkside ht1
  build main dcm $S/dc.txt $R4; build hf1 dcf $S/dc.txt $R4
  for v in main:g2m hf1:g2f ht0:g2t0 ht1:g2t1; do build ${v%%:*} ${v##*:} $S/g2.txt $R4; done
  elfid dcf dcm; elfid g2f g2m; elfid g2f g2t0; elfid g2f g2t1
  seqrun $S/dc.txt D $R4 900 main:dcm hf1:dcf hf1:dcf main:dcm main:dcm hf1:dcf hf1:dcf main:dcm
  seqrun $S/g2.txt G $R4 600 main:g2m hf1:g2f ht0:g2t0 ht1:g2t1 ht1:g2t1 ht0:g2t0 hf1:g2f main:g2m hf1:g2f main:g2m ht1:g2t1 ht0:g2t0 ht0:g2t0 ht1:g2t1 main:g2m hf1:g2f
else
  mkside head; mkside main; mkside hf1; mkside pair
  build main f1m $S/f1.txt $R4; build head f1h $S/f1.txt $R4; build hf1 f1f $S/f1.txt $R4
  build head prh $S/pair.txt $R4; build pair prp $S/pair.txt $R4
  elfid f1h f1m; elfid f1h f1f; elfid prh prp
  seqrun $S/f1.txt F $R4 600 main:f1m head:f1h hf1:f1f hf1:f1f head:f1h main:f1m hf1:f1f main:f1m head:f1h head:f1h main:f1m hf1:f1f
  seqrun $S/pair.txt P $R4 300 head:prh pair:prp pair:prp head:prh head:prh pair:prp pair:prp head:prh
  if [[ -z ${PERF_ONLY:-} ]]; then
    mkside mainf
    for v in mainf:bdmf head:bdmh hf1:bdmx; do build ${v%%:*} ${v##*:} $S/bitdump_merge.txt -; done
    [[ -n ${LLKAB_DRY:-} ]] && { echo "== dry: no consumers"; exit 0; }
    for v in mainf:bdmf head:bdmh hf1:bdmx; do
      D=$T/bits_${v%%:*}; mkdir -p $D
      BITDUMP_DIR=$D TT_LLK_DISABLE_ASSERTS=1 LIM=600 harness ${v%%:*} ${v##*:} $S/bitdump_merge.txt $T/cons_bits_${v%%:*}.log --compile-consumer -o timeout_method=thread --timeout 120
      echo "== bits ${v%%:*} rc=$? $(date -u +%T): $(grep -aE '[0-9]+ (passed|failed|error)' $T/cons_bits_${v%%:*}.log | tail -n 1), $(ls $D | wc -l) dumps"
      (cd $D && sha256sum * | sed "s/^/LLKAB_SHA ${v%%:*} /")
    done
    for v in head hf1; do echo "== bits mainf against $v: $(cd $T/bits_mainf && for f in *; do cmp -s $f $T/bits_$v/$f && echo equal || echo differ; done | sort | uniq -c | tr '\n' ' ')"; done
    build hf1 fah $S/func_f1.txt - asserts
    unset LLK_PERF_RUN_TYPES
    LIM=1500 harness hf1 fah $S/func_f1.txt $T/cons_func.log --compile-consumer -o timeout_method=thread --timeout 120
    echo "== functional hf1, LLK asserts on, rc=$? $(date -u +%T): $(grep -aE '[0-9]+ (passed|failed|error)' $T/cons_func.log | tail -n 1)"
    grep -aE '^(FAILED|ERROR) ' $T/cons_func.log | cut -c1-400 | sed -n 1,40p
  fi
fi
echo "== llkab job $JOB end $(date -u +%T)"
