#!/usr/bin/env bash
# Rope INIT audit (#60148): tt-llk harness A/B on one CI Blackhole runner (17_CI_AB.md). Every variant's files
# (ci_ab/variants/<v>/, the paths of ci_ab/files.txt) are copied into the one checkout before its builds and runs, so
# all variants share one tree path; each variant has its own fresh build directory. Variants: h = this head, m = main's
# ckernel_sfpu_rope.h with this head's perf kernel, o = this head's header with the perf kernel before the INIT fix.
# Plan lines: "<tag> <variant> <ids file> <run types|->". Then test_rope.py on h with LLK asserts on.
# Results go to the log as base64 of a tar.gz between @@RESULTS markers.
set +e +o pipefail
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; L=$R/tt_metal/tt-llk
PLAN=$S/${1:-plan.txt}
echo "== start $(date -u +%T) checkout $R $(git -C $R rev-parse HEAD) plan $PLAN host $(hostname)"
rm -rf /tmp/llkb_* /tmp/llkvenv
python3 -m venv /tmp/llkvenv && . /tmp/llkvenv/bin/activate
pip install -q -r $L/tests/requirements.txt > /tmp/pip.log 2>&1; echo "== pip rc=$? $(tail -1 /tmp/pip.log)"
(cd $L/tests && ./setup_testing_env.sh > /tmp/setup.log 2>&1); echo "== setup rc=$? $(tail -2 /tmp/setup.log | tr '\n' ' ')"
echo "== sfpi $($L/tests/sfpi/compiler/bin/riscv-tt-elf-g++ --version 2>&1 | head -1)"
export CHIP_ARCH=blackhole LLK_HOME=$L TT_LLK_DISABLE_ASSERTS=1 LLK_PERF_L1_MUX_GROUP=0 PERF_KEEP_RUNS=1000
place() {
  while read -r rel; do [[ -n $rel ]] && cp "$S/variants/$1/$rel" "$R/$rel"; done < $S/files.txt
  echo "   placed $1: $(while read -r rel; do [[ -n $rel ]] && md5sum < "$R/$rel" | cut -c1-8; done < $S/files.txt | paste -sd' ')"
}
OUT=/tmp/llkab_out; rm -rf $OUT; mkdir -p $OUT
cd $L/tests/python_tests
VARS=$(awk '!/^#/ && NF {print $2}' $PLAN | sort -u)
for v in $VARS; do
  place $v
  ids=$(awk -v v=$v '!/^#/ && NF && $2==v {print $3}' $PLAN | sort -u)
  cat $(for f in $ids; do echo $S/$f; done) | sort -u > /tmp/ids_$v.txt
  mapfile -t NODES < /tmp/ids_$v.txt
  mkdir -p /tmp/llkb_$v
  echo "== produce $v ${#NODES[@]} nodes $(date -u +%T)"
  RUNNER_TEMP=/tmp/llkb_$v TMPDIR=/tmp/llkb_$v timeout 3600 python -m pytest --compile-producer -n ${NPROD:-16} -p no:cacheprovider --override-ini=log_cli=false -q "${NODES[@]}" > $OUT/prod_$v.log 2>&1
  echo "== produce $v rc=$? $(grep -E 'passed|failed|error' $OUT/prod_$v.log | tail -1 | cut -c1-150) $(date -u +%T)"
done
# Guard (17_CI_AB.md): each side's ELF hashes as compiled (loaded sections only); pairs that must differ do, per thread
for v in $VARS; do
  python $S/tools/elfhash.py /tmp/llkb_$v > $OUT/elfs_$v.txt
  echo "== elfs $v: $(wc -l < $OUT/elfs_$v.txt) files, set $(md5sum < $OUT/elfs_$v.txt | cut -c1-12)"
done
cmp_pair() {  # cmp_pair <x> <y>: per thread, ELFs of the same variant dir that are equal / differ
  awk 'NR==FNR {h[$2]=$1; next} ($2 in h) {t=$2; sub(/.*\//, "", t); if (h[$2]==$1) s[t]++; else {d[t]++; nd++}; n++}
       END {for (t in s) o=o" "t" same "s[t]; for (t in d) o=o" "t" differ "d[t]; print n+0" paired:"o; exit (nd > 0 ? 0 : 1)}' $OUT/elfs_$1.txt $OUT/elfs_$2.txt
}
GFAIL=0
for a in ${GUARD_PAIRS:-m-h m-o o-h}; do
  x=${a%%-*}; y=${a##*-}; r=$(cmp_pair $x $y); rc=$?
  echo "== guard $y against $x: $r"; [[ $rc != 0 ]] && GFAIL=1
done
[[ $GFAIL == 1 ]] && { echo "== GUARD FAIL: two sides that should differ compiled to the same ELFs"; exit 1; }
i=0
while read -r tag v ids rts; do
  [[ -z ${tag:-} || $tag == \#* ]] && continue
  i=$((i + 1)); t=${tag}_${i}_$v
  place $v
  mapfile -t NODES < $S/$ids
  if [[ $rts != - ]]; then export LLK_PERF_RUN_TYPES=$rts; else unset LLK_PERF_RUN_TYPES; fi
  echo "== run $t ${#NODES[@]} nodes rts=$rts $(date -u +%T)"
  RUNNER_TEMP=/tmp/llkb_$v TMPDIR=/tmp/llkb_$v PERF_RUN_TAG=$t timeout 2400 python -m pytest --compile-consumer -p no:cacheprovider --override-ini=log_cli=false -o timeout_method=thread -q -rfE "${NODES[@]}" > $OUT/run_$t.log 2>&1
  rc=$?
  echo "== run $t rc=$rc $(grep -E 'passed|failed|error' $OUT/run_$t.log | tail -1 | cut -c1-150) $(date -u +%T)"
  if [[ $rc != 0 && -z ${SHOWN:-} ]]; then SHOWN=1; echo "== first failure of $t:"; grep -a -m1 -B2 -A40 -E "Error|Exception|Traceback|assert" $OUT/run_$t.log | cut -c1-240; fi
  if grep -q "TIMED OUT" $OUT/run_$t.log; then echo "== TENSIX TIMED OUT in $t, stopping"; break; fi
  [[ -d $L/perf_data/runs/$t ]] && mkdir -p $OUT/csv/$t && find $L/perf_data/runs/$t -name "*.csv" -exec cp {} $OUT/csv/$t/ \;
done < $PLAN
# functional: test_rope.py on h with the LLK asserts on (fresh build dir)
unset LLK_PERF_RUN_TYPES TT_LLK_DISABLE_ASSERTS
place h; mapfile -t NODES < $S/ids/func.txt; mkdir -p /tmp/llkb_hfunc
RUNNER_TEMP=/tmp/llkb_hfunc TMPDIR=/tmp/llkb_hfunc timeout 1800 python -m pytest --compile-producer -n ${NPROD:-16} -p no:cacheprovider --override-ini=log_cli=false -q "${NODES[@]}" > $OUT/prod_func.log 2>&1
echo "== produce func (asserts on) rc=$? $(grep -E 'passed|failed|error' $OUT/prod_func.log | tail -1 | cut -c1-150) $(date -u +%T)"
RUNNER_TEMP=/tmp/llkb_hfunc TMPDIR=/tmp/llkb_hfunc timeout 1800 python -m pytest --compile-consumer -p no:cacheprovider --override-ini=log_cli=false -o timeout_method=thread -q -rfE "${NODES[@]}" > $OUT/run_func.log 2>&1
echo "== functional test_rope.py on h, LLK asserts on, rc=$? $(grep -E 'passed|failed|error' $OUT/run_func.log | tail -1 | cut -c1-150) $(date -u +%T)"
grep -a -E "^(FAILED|ERROR)" $OUT/run_func.log | head -10 | cut -c1-240
for f in $OUT/prod_*.log $OUT/run_*.log; do grep -q -E "[0-9]+ (failed|error)" $f && { echo "== tail $f"; grep -a -E "^(FAILED|ERROR)" $f | head -3 | cut -c1-240; }; done
tar -czf /tmp/llkab.tgz -C $OUT csv $(cd $OUT && ls elfs_*.txt 2>/dev/null) 2>/dev/null
echo "@@RESULTS_BEGIN"; base64 -w 200 /tmp/llkab.tgz; echo "@@RESULTS_END"
echo "== done $(date -u +%T)"
