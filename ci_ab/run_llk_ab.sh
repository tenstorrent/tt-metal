#!/usr/bin/env bash
# tt-llk harness A/B on one CI Blackhole runner (test-dispatch). Every variant's tt-llk files (ci_ab/variants/<v>/, the
# paths of ci_ab/files.txt) are copied into the one checkout before its builds and runs, so all variants share one tree
# path; each variant has its own fresh build directory (RUNNER_TEMP, which the harness prefers to TMPDIR on CI). Plan lines: "<tag> <variant> <ids file> <run types|-> [rawz]".
# Results go to the log as base64 of a tar.gz between @@RESULTS markers.
set +e +o pipefail
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; L=$R/tt_metal/tt-llk
PLAN=$S/${1:-plan.txt}
echo "== start $(date -u +%T) checkout $R plan $PLAN"
python3 -m venv /tmp/llkvenv && . /tmp/llkvenv/bin/activate
pip install -q -r $L/tests/requirements.txt > /tmp/pip.log 2>&1; echo "== pip rc=$? $(tail -1 /tmp/pip.log)"
(cd $L/tests && ./setup_testing_env.sh > /tmp/setup.log 2>&1); echo "== setup rc=$? $(tail -2 /tmp/setup.log | tr '\n' ' ')"
export CHIP_ARCH=blackhole LLK_HOME=$L TT_LLK_DISABLE_ASSERTS=1 LLK_PERF_L1_MUX_GROUP=0 PERF_KEEP_RUNS=1000
export PYTHONPATH="$S/plug${PYTHONPATH:+:$PYTHONPATH}"
place() {
  while read -r rel; do [[ -n $rel ]] && cp "$S/variants/$1/$rel" "$R/$rel"; done < $S/files.txt
}
OUT=/tmp/llkab_out; rm -rf $OUT; mkdir -p $OUT
cp $S/shadow/*.py $L/tests/python_tests/ 2>/dev/null
cd $L/tests/python_tests
# one producer per variant over the union of its ids
for v in $(awk '!/^#/ && NF {print $2}' $PLAN | sort -u); do
  place $v
  ids=$(awk -v v=$v '!/^#/ && NF && $2==v {print $3}' $PLAN | sort -u)
  cat $(for f in $ids; do echo $S/$f; done) | sort -u > /tmp/ids_$v.txt
  mapfile -t NODES < /tmp/ids_$v.txt
  mkdir -p /tmp/llkb_$v
  echo "== produce $v ${#NODES[@]} nodes $(date -u +%T)"
  RUNNER_TEMP=/tmp/llkb_$v TMPDIR=/tmp/llkb_$v timeout 3600 python -m pytest --compile-producer -n ${NPROD:-16} -p no:cacheprovider --override-ini=log_cli=false -q "${NODES[@]}" > $OUT/prod_$v.log 2>&1
  echo "== produce $v rc=$? $(grep -E 'passed|failed|error' $OUT/prod_$v.log | tail -1 | cut -c1-150) $(date -u +%T)"
done
i=0
while read -r tag v ids rts rawz; do
  [[ -z ${tag:-} || $tag == \#* ]] && continue
  i=$((i + 1)); t=${tag}_${i}_$v
  place $v
  mapfile -t NODES < $S/$ids
  XA=(); [[ ${rawz:-} == rawz ]] && { XA=(-p rawzones); export RAWZONES_OUT=$OUT/rawz_$t.jsonl; }
  if [[ $rts != - ]]; then export LLK_PERF_RUN_TYPES=$rts; else unset LLK_PERF_RUN_TYPES; fi
  echo "== run $t ${#NODES[@]} nodes rts=$rts $(date -u +%T)"
  RUNNER_TEMP=/tmp/llkb_$v TMPDIR=/tmp/llkb_$v PERF_RUN_TAG=$t timeout 2400 python -m pytest --compile-consumer -p no:cacheprovider --override-ini=log_cli=false -o timeout_method=thread -q -rfE "${XA[@]}" "${NODES[@]}" > $OUT/run_$t.log 2>&1
  rc=$?
  echo "== run $t rc=$rc $(grep -E 'passed|failed|error' $OUT/run_$t.log | tail -1 | cut -c1-150) $(date -u +%T)"
  if [[ $rc != 0 && -z ${SHOWN:-} ]]; then SHOWN=1; echo "== first failure of $t:"; grep -a -m1 -B2 -A40 -E "Error|Exception|Traceback|assert" $OUT/run_$t.log | cut -c1-240; fi
  [[ -d $L/perf_data/runs/$t ]] && mkdir -p $OUT/csv/$t && find $L/perf_data/runs/$t -name "*.csv" ! -name "*.post.csv" -exec cp {} $OUT/csv/$t/ \;
  unset RAWZONES_OUT
done < $PLAN
for f in $OUT/prod_*.log $OUT/run_*.log; do grep -q -E "failed|error" $f && { echo "== tail $f"; grep -a -E "^(FAILED|ERROR)" $f | head -3 | cut -c1-240; }; done
tar -czf /tmp/llkab.tgz -C $OUT csv $(cd $OUT && ls rawz_*.jsonl 2>/dev/null) 2>/dev/null
echo "@@RESULTS_BEGIN"; base64 -w 200 /tmp/llkab.tgz; echo "@@RESULTS_END"
echo "== done $(date -u +%T)"
