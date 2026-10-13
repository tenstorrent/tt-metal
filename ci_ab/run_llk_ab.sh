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
for v in $(awk '!/^#/ && NF && $5 !~ /^bits/ {print $2}' $PLAN | sort -u); do
  place $v
  ids=$(awk -v v=$v '!/^#/ && NF && $2==v && $5 !~ /^bits/ {print $3}' $PLAN | sort -u)
  cat $(for f in $ids; do echo $S/$f; done) | sort -u > /tmp/ids_$v.txt
  mapfile -t NODES < /tmp/ids_$v.txt
  mkdir -p /tmp/llkb_$v
  echo "== produce $v ${#NODES[@]} nodes $(date -u +%T)"
  RUNNER_TEMP=/tmp/llkb_$v TMPDIR=/tmp/llkb_$v timeout 3600 python -m pytest --compile-producer -n ${NPROD:-16} -p no:cacheprovider --override-ini=log_cli=false -q "${NODES[@]}" > $OUT/prod_$v.log 2>&1
  echo "== produce $v rc=$? $(grep -E 'passed|failed|error' $OUT/prod_$v.log | tail -1 | cut -c1-150) $(date -u +%T)"
done
# Guard (17_CI_AB.md): each side's ELF hashes as compiled, and the sides must differ from main where their sources do
for v in $(awk '!/^#/ && NF && $5 !~ /^bits/ {print $2}' $PLAN | sort -u); do
  (cd /tmp/llkb_$v && find . -name "*.elf" -path "*/elf/*" | sort | while read -r e; do echo "$(md5sum < "$e" | cut -c1-12) $e"; done) > $OUT/elfs_$v.txt
  echo "== elfs $v: $(wc -l < $OUT/elfs_$v.txt) files, set $(md5sum < $OUT/elfs_$v.txt | cut -c1-12), dir $(ls -d /tmp/llkb_$v/tt-llk-build 2>/dev/null)"
done
GFAIL=0
for v in $(awk '!/^#/ && NF && $5 !~ /^bits/ {print $2}' $PLAN | sort -u); do
  [[ $v == ${GUARD_BASE:-main} || ! -s $OUT/elfs_${GUARD_BASE:-main}.txt ]] && continue
  d=$(awk 'NR==FNR {h[$1]=1; next} !($1 in h)' $OUT/elfs_${GUARD_BASE:-main}.txt $OUT/elfs_$v.txt | wc -l)
  echo "== guard $v against ${GUARD_BASE:-main}: $d of $(wc -l < $OUT/elfs_$v.txt) ELFs not among the base's"
  [[ $d == 0 ]] && GFAIL=1
done
[[ $GFAIL == 1 ]] && { echo "== GUARD FAIL: a side compiled to the base's ELFs"; exit 1; }
for a in ${GUARD_PAIRS:-}; do x=${a%%-*}; y=${a##*-}; d=$(awk 'NR==FNR {h[$1]=1; next} !($1 in h)' $OUT/elfs_$x.txt $OUT/elfs_$y.txt | wc -l); echo "== guard $y against $x: $d ELFs not among $x's"; [[ $d == 0 ]] && { echo "== GUARD FAIL $a"; exit 1; }; done
i=0
while read -r tag v ids rts rawz; do
  [[ -z ${tag:-} || $tag == \#* ]] && continue
  i=$((i + 1)); t=${tag}_${i}_$v
  place $v
  mapfile -t NODES < $S/$ids
  if [[ ${rawz:-} == bits* ]]; then
    # Functional runs with LLK asserts on, every result hashed bit for bit (resbits), own build directory.
    # bitsel: run the node ids' modules and keep the collected items whose id is in the list (ids absent on a side are skipped)
    SA=("${NODES[@]}"); SP=()
    if [[ $rawz == bitsel ]]; then SA=($(printf '%s\n' "${NODES[@]}" | sed 's/::.*//' | sort -u)); SP=(-p sel_plugin); export SEL_FILE=$S/$ids; fi
    rm -rf /tmp/llkbits_$v; mkdir -p /tmp/llkbits_$v
    echo "== bits $t ${#NODES[@]} nodes $(date -u +%T)"
    env -u TT_LLK_DISABLE_ASSERTS RUNNER_TEMP=/tmp/llkbits_$v TMPDIR=/tmp/llkbits_$v RESBITS_OUT=$OUT/bits_$t.jsonl timeout 5400 python -m pytest -n ${NBITS:-8} -p resbits "${SP[@]}" -p no:cacheprovider --override-ini=log_cli=false -o timeout_method=thread -q -rfE "${SA[@]}" > $OUT/run_$t.log 2>&1
    rc=$?; unset SEL_FILE; echo "== bits $t rc=$rc $(grep -E 'passed|failed|error' $OUT/run_$t.log | tail -1 | cut -c1-150) $(date -u +%T)"
    [[ $rc != 0 ]] && grep -a -E "(FAILED|ERROR) " $OUT/run_$t.log | head -10 | cut -c1-240
    (cd /tmp/llkbits_$v && find . -name "*.elf" -path "*/elf/*" | sort | while read -r e; do echo "$(md5sum < "$e" | cut -c1-12) $e"; done) > $OUT/elfs_$t.txt
    echo "== elfs $t: $(wc -l < $OUT/elfs_$t.txt) files"
    for o in $OUT/elfs_bits*.txt; do [[ $o == $OUT/elfs_$t.txt ]] && continue; echo "== guard $t against $(basename $o .txt): $(awk 'NR==FNR {h[$1]=1; next} !($1 in h)' $o $OUT/elfs_$t.txt | wc -l) ELFs not among its"; done
    continue
  fi
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
tar -czf /tmp/llkab.tgz -C $OUT csv $(cd $OUT && ls rawz_*.jsonl elfs_*.txt bits_*.jsonl 2>/dev/null) 2>/dev/null
echo "@@RESULTS_BEGIN"; base64 -w 200 /tmp/llkab.tgz; echo "@@RESULTS_END"
echo "== done $(date -u +%T)"
