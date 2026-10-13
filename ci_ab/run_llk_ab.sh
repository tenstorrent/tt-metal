#!/usr/bin/env bash
# Scratch, not for merge. tt-llk harness A/B on one CI Blackhole runner (test-dispatch). Every variant's files
# (ci_ab/variants/<v>/, the paths of ci_ab/files.txt) are copied into the one checkout before its builds and runs, so all
# variants share one tree path; each variant has its own fresh build directory (RUNNER_TEMP). A variant named <v>_wc is
# <v>'s files built with perf counters. Plan lines: "<tag> <variant> <ids file> <run types|-> [wc|bits|bitsel]".
# Guard: each variant's ELF .text hashes; GUARD_DIFF="x-y ..." pairs must differ, GUARD_SAME="x-y ..." pairs must match.
# Results go to the log as base64 of a tar.gz between @@RESULTS markers.
set +e +o pipefail
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; L=$R/tt_metal/tt-llk
PLAN=$S/${1:-plan.txt}
echo "== start $(date -u +%T) checkout $R $(git -C $R log -1 --format=%h) plan $PLAN"
python3 -m venv /tmp/llkvenv && . /tmp/llkvenv/bin/activate
pip install -q -r $L/tests/requirements.txt > /tmp/pip.log 2>&1; echo "== pip rc=$? $(tail -1 /tmp/pip.log)"
(cd $L/tests && ./setup_testing_env.sh > /tmp/setup.log 2>&1); echo "== setup rc=$? $(tail -2 /tmp/setup.log | tr '\n' ' ')"
export CHIP_ARCH=blackhole LLK_HOME=$L TT_LLK_DISABLE_ASSERTS=1 LLK_PERF_L1_MUX_GROUP=0 PERF_KEEP_RUNS=1000
export PYTHONPATH="$S/plug${PYTHONPATH:+:$PYTHONPATH}"
OBJ=$(find $L/tests/sfpi/ -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1); echo "== objcopy $OBJ"
place() {
  local d=${1%_wc}
  while read -r rel; do [[ -n $rel ]] && cp "$S/variants/$d/$rel" "$R/$rel"; done < $S/files.txt
  echo "== place $1: $(while read -r rel; do [[ -n $rel ]] && md5sum < "$R/$rel" | cut -c1-8; done < $S/files.txt | tr '\n' ' ')"
}
texthash() { (cd $1 && find . -name "*.elf" -path "*/elf/*" | sort | while read -r e; do echo "$($OBJ -O binary -j .text "$e" /dev/stdout 2>/dev/null | md5sum | cut -c1-12) ${e#./}"; done); }
OUT=/tmp/llkab_out; rm -rf $OUT; mkdir -p $OUT
cd $L/tests/python_tests
VARS=$(awk '!/^#/ && NF && $5 !~ /^bits/ {print $2}' $PLAN | sort -u)
for v in $VARS; do
  place $v
  ids=$(awk -v v=$v '!/^#/ && NF && $2==v && $5 !~ /^bits/ {print $3}' $PLAN | sort -u)
  cat $(for f in $ids; do echo $S/$f; done) | sort -u > /tmp/ids_$v.txt
  mapfile -t NODES < /tmp/ids_$v.txt
  mkdir -p /tmp/llkb_$v
  echo "== produce $v ${#NODES[@]} nodes $(date -u +%T)"
  PX=(); [[ $v == *_wc ]] && PX=(--dump-perf-counters)
  RUNNER_TEMP=/tmp/llkb_$v TMPDIR=/tmp/llkb_$v timeout 3600 python -m pytest "${PX[@]}" --compile-producer -n ${NPROD:-16} -p no:cacheprovider --override-ini=log_cli=false -q "${NODES[@]}" > $OUT/prod_$v.log 2>&1
  echo "== produce $v rc=$? $(grep -E 'passed|failed|error' $OUT/prod_$v.log | tail -1 | cut -c1-150) $(date -u +%T)"
  texthash /tmp/llkb_$v > $OUT/elfs_$v.txt
  echo "== elfs $v: $(wc -l < $OUT/elfs_$v.txt) files, .text set $(cut -d' ' -f1 $OUT/elfs_$v.txt | md5sum | cut -c1-12)"
done
GFAIL=0
for a in ${GUARD_DIFF:-}; do x=${a%%-*}; y=${a##*-}; d=$(join -j2 <(sort -k2 $OUT/elfs_$x.txt) <(sort -k2 $OUT/elfs_$y.txt) | awk '$2!=$3' | wc -l)
  echo "== guard $y against $x: $d ELFs differ (must differ)"; [[ $d == 0 ]] && GFAIL=1; done
for a in ${GUARD_SAME:-}; do x=${a%%-*}; y=${a##*-}; d=$(join -j2 <(sort -k2 $OUT/elfs_$x.txt) <(sort -k2 $OUT/elfs_$y.txt) | awk '$2!=$3' | wc -l)
  echo "== guard $y against $x: $d ELFs differ (must match)"; [[ $d != 0 ]] && GFAIL=1; done
[[ $GFAIL == 1 ]] && { echo "== GUARD FAIL"; exit 1; }
i=0
while read -r tag v ids rts mode; do
  [[ -z ${tag:-} || $tag == \#* ]] && continue
  i=$((i + 1)); t=${tag}_${i}_$v
  place $v
  mapfile -t NODES < $S/$ids
  if [[ ${mode:-} == bits* ]]; then
    # Functional runs with LLK asserts on, every L1 read hashed bit for bit (topkbits), own build directory.
    SA=("${NODES[@]}"); SP=()
    if [[ $mode == bitsel ]]; then SA=($(printf '%s\n' "${NODES[@]}" | sed 's/::.*//' | sort -u)); SP=(-p sel_plugin); export SEL_FILE=$S/$ids; fi
    rm -rf /tmp/llkbits_$v; mkdir -p /tmp/llkbits_$v
    echo "== bits $t ${#NODES[@]} entries $(date -u +%T)"
    env -u TT_LLK_DISABLE_ASSERTS RUNNER_TEMP=/tmp/llkbits_$v TMPDIR=/tmp/llkbits_$v TOPKBITS_OUT=$OUT/bits_$t.jsonl timeout 5400 python -m pytest -n ${NBITS:-8} -p topkbits "${SP[@]}" -p no:cacheprovider --override-ini=log_cli=false -o timeout_method=thread -q -rfE "${SA[@]}" > $OUT/run_$t.log 2>&1
    rc=$?; unset SEL_FILE; echo "== bits $t rc=$rc $(grep -E 'passed|failed|error' $OUT/run_$t.log | tail -1 | cut -c1-150) $(wc -l < $OUT/bits_$t.jsonl 2>/dev/null) reads $(date -u +%T)"
    [[ $rc != 0 ]] && grep -a -E "(FAILED|ERROR) " $OUT/run_$t.log | head -10 | cut -c1-240
    texthash /tmp/llkbits_$v > $OUT/elfs_$t.txt
    continue
  fi
  [[ $rts == none ]] && continue
  XA=(); [[ ${mode:-} == wc ]] && XA=(--dump-perf-counters)
  if [[ $rts != - ]]; then export LLK_PERF_RUN_TYPES=$rts; else unset LLK_PERF_RUN_TYPES; fi
  echo "== run $t ${#NODES[@]} nodes rts=$rts $(date -u +%T)"
  RUNNER_TEMP=/tmp/llkb_$v TMPDIR=/tmp/llkb_$v PERF_RUN_TAG=$t timeout 2400 python -m pytest --compile-consumer -p no:cacheprovider --override-ini=log_cli=false -o timeout_method=thread -q -rfE "${XA[@]}" "${NODES[@]}" > $OUT/run_$t.log 2>&1
  rc=$?
  echo "== run $t rc=$rc $(grep -E 'passed|failed|error' $OUT/run_$t.log | tail -1 | cut -c1-150) $(date -u +%T)"
  if [[ $rc != 0 && -z ${SHOWN:-} ]]; then SHOWN=1; echo "== first failure of $t:"; grep -a -m1 -B2 -A30 -E "Error|Exception|Traceback|assert" $OUT/run_$t.log | cut -c1-240; fi
  [[ -d $L/perf_data/runs/$t ]] && mkdir -p $OUT/csv/$t && find $L/perf_data/runs/$t -name "*.csv" ! -name "*.post.csv" -exec cp {} $OUT/csv/$t/ \;
done < $PLAN
tar -czf /tmp/llkab.tgz -C $OUT csv $(cd $OUT && ls elfs_*.txt bits_*.jsonl 2>/dev/null) 2>/dev/null
echo "@@RESULTS_BEGIN"; base64 -w 200 /tmp/llkab.tgz; echo "@@RESULTS_END"
echo "== done $(date -u +%T)"
