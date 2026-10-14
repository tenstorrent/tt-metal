#!/usr/bin/env bash
# #58819 ELWADD copy release (17_CI_AB.md; scratch, not for merge). This checkout is the #58819 head 279cad40361 (with the
# copy_tile fold) merged with main: G is the checkout itself; E is the ELWADD copy that releases its sources on each face's
# last ELWADD (built twice, bE and bE2: the A/A); M main's versions of every tt-llk and hw file the stack changes.
# MODE=ab: harness rows (ids/perf_c.txt on G, E, E2, M; ids/perf_h.txt on G, E, E2), rotated, 4 runs a side; then op level
# (run_e2_ol.sh: P the file before the fold, G, E, E2, M). MODE=bits: ids/bits.txt on G and E, ids/bits_m.txt on M.
set +e +o pipefail
R=${STK_ROOT:-/work}; S=$R/ci_ab; T=${STK_TMP:-/tmp/stk}; rm -rf $T; mkdir -p $T; NP=${STK_NPROD:-16}
unset RUNNER_TEMP
export CHIP_ARCH=blackhole LLK_PERF_L1_MUX_GROUP=0 PERF_KEEP_RUNS=400
echo "== stk start $(date -u +%T) $(git -C $R rev-parse HEAD 2>/dev/null) $(nproc) cpus $(hostname)"

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

DC=tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_datacopy.h
mkside() {  # mkside <v>
  local v=$1 d=$T/side_$1
  mkdir -p $d/tt_metal
  for e in $R/*; do n=$(basename $e); [[ $n == tt_metal ]] || ln -s $e $d/$n; done
  for e in $R/tt_metal/*; do n=$(basename $e); [[ $n == tt-llk || $n == hw ]] || ln -s $e $d/tt_metal/$n; done
  cp -a $R/tt_metal/tt-llk $R/tt_metal/hw $d/tt_metal/
  [[ -d $S/llkvar/$v/tree ]] && cp -a $S/llkvar/$v/tree/. $d/
  cp $S/shadow/bitdump_plugin.py $d/tt_metal/tt-llk/tests/python_tests/
  rm -rf $d/tt_metal/tt-llk/tests/sfpi $d/tt_metal/tt-llk/perf_data; ln -s $SF $d/tt_metal/tt-llk/tests/sfpi
  echo "== side $v: $(cd $d/tt_metal && md5sum $DC tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_A.h tt-llk/tt_llk_blackhole/llk_lib/llk_pack.h | cut -c1-10 | tr '\n' ' ')"
}

cat > $T/pyrun.py <<'PYEOF'
import sys, pytest
ids = [l.strip() for l in open(sys.argv[1]) if l.strip()]
sys.exit(pytest.main(["-p", "no:cacheprovider", "--override-ini=log_cli=false", "-q", *sys.argv[2:], *ids]))
PYEOF

harness() {  # harness <side> <build> <ids> <log> [pytest args...]
  local v=$1 b=$2 ids=$3 log=$4; shift 4
  (cd $T/side_$v/tt_metal/tt-llk/tests/python_tests && \
   LLK_HOME=$T/side_$v/tt_metal/tt-llk PYTHONPATH=$T/side_$v/tt_metal/tt-llk/tests/python_tests${PYTHONPATH:+:$PYTHONPATH} TMPDIR=$T/b_$b timeout -s INT -k 60 ${LIM:-1800} $PY $T/pyrun.py $ids "$@") > $log 2>&1
}
build() {  # build <side> <build> <ids> <run types|-> [asserts]
  mkdir -p $T/b_$2
  if [[ $4 == - ]]; then unset LLK_PERF_RUN_TYPES; else export LLK_PERF_RUN_TYPES=$4; fi
  if [[ ${5:-} == asserts ]]; then (unset TT_LLK_DISABLE_ASSERTS; LIM=5400 harness $1 $2 $3 $T/prod_$2.log --compile-producer -n $NP)
  else TT_LLK_DISABLE_ASSERTS=1 LIM=5400 harness $1 $2 $3 $T/prod_$2.log --compile-producer -n $NP; fi
  echo "== build $2 ($1, $(basename $3)) rc=$? $(date -u +%T): $(grep -aE '[0-9]+ (passed|failed|error)' $T/prod_$2.log | tail -n 1)"
  grep -aE '^(FAILED|ERROR) ' $T/prod_$2.log | cut -c1-300 | sed -n 1,10p
}
elfid() {  # elfid <build a> <build b> [same]: loaded-image identity; without "same" the job stops if no ELF differs
  local r; r=$($PY $S/tools/elfload.py $T/b_$1 $T/b_$2 | tail -n 1)
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
run() {  # run <side> <build> <ids> <tag> <run types|-> <limit>
  local v=$1 b=$2 ids=$3 tag=$4 rts=$5
  if [[ $rts == - ]]; then unset LLK_PERF_RUN_TYPES; else export LLK_PERF_RUN_TYPES=$rts; fi
  TT_LLK_DISABLE_ASSERTS=1 PERF_RUN_TAG=$tag LIM=$6 harness $v $b $ids $T/cons_$tag.log --compile-consumer -o timeout_method=thread --timeout 120
  echo "== run $tag ($v, $b) rc=$? $(date -u +%T): $(grep -aE '[0-9]+ (passed|failed|error)' $T/cons_$tag.log | tail -n 1)"
  grep -aE '^(FAILED|ERROR) ' $T/cons_$tag.log | cut -c1-300 | sed -n 1,10p
  local pd=$T/side_$v/tt_metal/tt-llk/perf_data
  [[ -d $pd/runs/$tag ]] && ship $tag $pd/runs $tag/$tag.parquet
}
R5=L1_TO_L1,UNPACK_ISOLATE,MATH_ISOLATE,PACK_ISOLATE,L1_CONGESTION

MODE=${MODE:-ab}; echo "== mode $MODE"
for v in G E M; do mkside $v; done

if [[ $MODE == ab ]]; then
PC=$S/ids/${IDS_C:-perf_c.txt}; PH=$S/ids/${IDS_H:-perf_h.txt}; echo "== rows $(wc -l < $PC) + $(wc -l < $PH)"
for v in G:bG E:bE E:bE2 M:bM; do build ${v%%:*} ${v##*:} $PC $R5; done
for v in G:hG E:hE E:hE2; do build ${v%%:*} ${v##*:} $PH $R5; done
elfid bG bE; elfid bE bE2 same; elfid bG bM same; elfid hG hE
[[ -n ${STK_DRY:-} ]] && { echo "== dry: builds only"; exit 0; }
n=0
for sb in G:bG:hG E:bE:hE M:bM:- E:bE2:hE2  E:bE2:hE2 M:bM:- E:bE:hE G:bG:hG \
          E:bE:hE G:bG:hG E:bE2:hE2 M:bM:-  M:bM:- E:bE2:hE2 G:bG:hG E:bE:hE; do
  IFS=: read -r v b h <<< "$sb"; n=$((n + 1))
  run $v $b $PC C_${v}_${b}_$n $R5 2400
  [[ $h != - ]] && run $v $h $PH D_${v}_${h}_$n $R5 900
done
echo "== harness end $(date -u +%T)"
[[ -n ${NO_OL:-} ]] || bash $S/run_e2_ol.sh
echo "== stk end $(date -u +%T)"
exit 0
fi

# Bits with LLK asserts on: G and E on every reached functional test, M on the modules main shares; E again (E2) on any
# test whose G and E dumps differ, to tell a nondeterministic test from a change
unset LLK_PERF_RUN_TYPES
for v in G:aG E:aE M:aM; do
  ids=$S/ids/bits.txt; [[ ${v%%:*} == M ]] && ids=$S/ids/bits_m.txt
  build ${v%%:*} ${v##*:} $ids - asserts
done
elfid aG aE
bitrun() {  # bitrun <side> <build> <tag> <ids>
  local D=$T/bits_$3; mkdir -p $D
  BITDUMP_DIR=$D LIM=7200 harness $1 $2 $4 $T/cons_bits_$3.log -p bitdump_plugin --compile-consumer -o timeout_method=thread --timeout 300
  echo "== bits $3 rc=$? $(date -u +%T): $(grep -aE '[0-9]+ (passed|failed|error)' $T/cons_bits_$3.log | tail -n 1), $(ls $D | wc -l) files"
  grep -aE '^(FAILED|ERROR) ' $T/cons_bits_$3.log | cut -c1-300 | sed -n 1,20p
}
bitrun G aG G $S/ids/bits.txt; bitrun E aE E $S/ids/bits.txt; bitrun M aM M $S/ids/bits_m.txt
for p in "G E" "M E" "M G"; do set -- $p; echo "== bits $1 against $2"; $PY $S/tools/bitcmp.py $T/bits_$1 $T/bits_$2; done
$PY $S/tools/bitcmp.py $T/bits_G $T/bits_E --list | sed -n 's/^BITS differ //p' > $T/redo.txt
if [[ -s $T/redo.txt ]]; then
  bitrun E aE E2 $T/redo.txt; bitrun G aG G2 $T/redo.txt
  for p in "E E2" "G G2"; do set -- $p; echo "== bits $1 against $2 (the differing tests)"; $PY $S/tools/bitcmp.py $T/bits_$1 $T/bits_$2; done
fi
echo "== bits end $(date -u +%T)"
