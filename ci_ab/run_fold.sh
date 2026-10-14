#!/usr/bin/env bash
# #59135 on the new #58819 head, proofs on its head merged with current main (17_CI_AB.md; scratch, not for merge). This checkout
# is that merge and carries the fold: G is the checkout itself, H puts back the file before the fold. Side trees at one depth, each a copy of this checkout's tt_metal/tt-llk and tt_metal/hw with its
# variant's files on top (llk_math_eltwise_unary_datacopy.h):
#   H  the head before the fold; G  the fold (the A2D MOP's last move clears the dest counter through ADDR_MOD_0), built twice
#   (bG, bG2: the A/A at one tree path); M  main's versions of every tt-llk and hw file the stack changes.
# MODE=ab: harness rows (ids/perf_c.txt on H, G, G2, M; ids/perf_h.txt, rows main lacks, on H, G, G2), rotated, 4 runs a
# side; then op level (run_fold_ol.sh). MODE=bits: bit dumps with LLK asserts on, ids/bits.txt on H and G, ids/bits_m.txt
# on M.
# Each run's perf_data goes to the log as base64 tar.gz lines "LLKAB_DATA <tag> <chunk>".
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

for v in H G M; do mkside $v; done
MODE=${MODE:-ab}; echo "== mode $MODE"

if [[ $MODE == ab ]]; then
# Harness A/B
for v in H:bH G:bG G:bG2 M:bM; do build ${v%%:*} ${v##*:} $S/ids/perf_c.txt $R5; done
for v in H:hH G:hG G:hG2; do build ${v%%:*} ${v##*:} $S/ids/perf_h.txt $R5; done
elfid bH bG; elfid bG bG2 same; elfid bH bM same; elfid hH hG
[[ -n ${STK_DRY:-} ]] && { echo "== dry: builds only"; exit 0; }
n=0
for sb in H:bH:hH G:bG:hG M:bM:- G:bG2:hG2  G:bG2:hG2 M:bM:- G:bG:hG H:bH:hH \
          G:bG:hG H:bH:hH G:bG2:hG2 M:bM:-  M:bM:- G:bG2:hG2 H:bH:hH G:bG:hG; do
  IFS=: read -r v b h <<< "$sb"; n=$((n + 1))
  run $v $b $S/ids/perf_c.txt C_${v}_${b}_$n $R5 2400
  [[ $h != - ]] && run $v $h $S/ids/perf_h.txt D_${v}_${h}_$n $R5 900
done
echo "== harness end $(date -u +%T)"
[[ -n ${NO_OL:-} ]] || bash $S/run_fold_ol.sh
echo "== stk end $(date -u +%T)"
exit 0
fi

# Bits with LLK asserts on: H and G on every reached functional test, M on the modules main shares; G again (G2) on any
# test whose H and G dumps differ, to tell a nondeterministic test from a change
unset LLK_PERF_RUN_TYPES
for v in H:aH G:aG M:aM; do
  ids=$S/ids/bits.txt; [[ ${v%%:*} == M ]] && ids=$S/ids/bits_m.txt
  build ${v%%:*} ${v##*:} $ids - asserts
done
elfid aH aG
bitrun() {  # bitrun <side> <build> <tag> <ids>
  local D=$T/bits_$3; mkdir -p $D
  BITDUMP_DIR=$D LIM=7200 harness $1 $2 $4 $T/cons_bits_$3.log -p bitdump_plugin --compile-consumer -o timeout_method=thread --timeout 300
  echo "== bits $3 rc=$? $(date -u +%T): $(grep -aE '[0-9]+ (passed|failed|error)' $T/cons_bits_$3.log | tail -n 1), $(ls $D | wc -l) files"
  grep -aE '^(FAILED|ERROR) ' $T/cons_bits_$3.log | cut -c1-300 | sed -n 1,20p
}
bitrun H aH H $S/ids/bits.txt; bitrun G aG G $S/ids/bits.txt; bitrun M aM M $S/ids/bits_m.txt
for p in "H G" "M G" "M H"; do set -- $p; echo "== bits $1 against $2"; $PY $S/tools/bitcmp.py $T/bits_$1 $T/bits_$2; done
$PY $S/tools/bitcmp.py $T/bits_H $T/bits_G --list | sed -n 's/^BITS differ //p' > $T/redo.txt
if [[ -s $T/redo.txt ]]; then
  bitrun G aG G2 $T/redo.txt; bitrun H aH H2 $T/redo.txt
  for p in "G G2" "H H2"; do set -- $p; echo "== bits $1 against $2 (the differing tests)"; $PY $S/tools/bitcmp.py $T/bits_$1 $T/bits_$2; done
fi
echo "== bits end $(date -u +%T)"
