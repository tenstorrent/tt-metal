#!/usr/bin/env bash
# Scratch, not for merge. Round 5 (main M against the head H, A/A H2; bits lines run the functional
# tests with LLK asserts on and hash every L1 read, -p topkbits). Eltwise binary all-configuration sweep on one CI Blackhole runner (test-dispatch), from the
# matmul and moe_lever run_llk_ab.sh. Every variant's files (ci_ab/variants/<v>/, paths in ci_ab/files.txt) are copied into
# the one checkout before its build and runs, so all variants share one tree path; each variant has its own fresh build
# directory. <v>_wc is <v> built with perf counters. Plan lines: "<tag> <variant> <ids file> <run types|none> [wc]".
# Guard: ELF .text hashes per variant; M and H must differ, H and H2 must match; M and K (the pack-only stack) are printed.
# A pass that would start after DEADLINE seconds is skipped. Results: base64 of a tar.xz between @@RESULTS markers.
set +e +o pipefail
T0=$(date +%s); DEADLINE=${DEADLINE:-5400}
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; L=$R/tt_metal/tt-llk
PLAN=$S/${1:-plan.txt}; FL=$S/files.txt
echo "== start $(date -u +%T) checkout $R $(git -C $R log -1 --format=%h) plan $PLAN nproc $(nproc)"
python3 -m venv /tmp/llkvenv && . /tmp/llkvenv/bin/activate
pip install -q -r $L/tests/requirements.txt > /tmp/pip.log 2>&1; echo "== pip rc=$? $(tail -1 /tmp/pip.log)"
(cd $L/tests && ./setup_testing_env.sh > /tmp/setup.log 2>&1); echo "== setup rc=$? $(tail -2 /tmp/setup.log | tr '\n' ' ')"
export CHIP_ARCH=blackhole LLK_HOME=$L TT_LLK_DISABLE_ASSERTS=1 LLK_PERF_L1_MUX_GROUP=0 PERF_KEEP_RUNS=1000
export PYTHONPATH="$S/plug${PYTHONPATH:+:$PYTHONPATH}"
OBJ=$(find $L/tests/sfpi/ -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1); echo "== objcopy $OBJ"
place() {
  local d=${1%_wc}
  while read -r rel; do [[ -n $rel ]] && cp "$S/variants/$d/$rel" "$R/$rel"; done < $FL
  echo "== place $1: $(while read -r rel; do [[ -n $rel ]] && md5sum < "$R/$rel" | cut -c1-6; done < $FL | md5sum | cut -c1-12)"
}
texthash() { (cd $1 && find . -name "*.elf" -path "*/elf/*" | sort | while read -r e; do echo "$($OBJ -O binary -j .text "$e" /dev/stdout 2>/dev/null | md5sum | cut -c1-12) ${e#./}"; done); }
cmpelf() { join -j2 <(sort -k2 $OUT/elfs_$1.txt) <(sort -k2 $OUT/elfs_$2.txt) | awk '{n++; if ($2 != $3) d++} END {print d+0, n+0}'; }
OUT=/tmp/llkab_out; rm -rf $OUT; mkdir -p $OUT/csv
cd $L/tests/python_tests
VARS=$(awk '!/^#/ && NF && $1 != "bits" {print $2}' $PLAN | awk '!s[$0]++')
for v in $VARS; do
  place $v
  ids=$(awk -v v=$v '!/^#/ && NF && $2==v && $1 != "bits" {print $3}' $PLAN | sort -u)
  cat $(for f in $ids; do echo $S/$f; done) | sort -u > /tmp/ids_$v.txt
  mapfile -t NODES < /tmp/ids_$v.txt
  mkdir -p /tmp/llkb_$v
  PX=(); [[ $v == *_wc ]] && PX=(--dump-perf-counters)
  echo "== produce $v ${#NODES[@]} nodes $(date -u +%T)"
  RUNNER_TEMP=/tmp/llkb_$v TMPDIR=/tmp/llkb_$v timeout 2400 python -m pytest "${PX[@]}" --compile-producer -n ${NPROD:-$(nproc)} -p no:cacheprovider --override-ini=log_cli=false -q "${NODES[@]}" > $OUT/prod_$v.log 2>&1
  echo "== produce $v rc=$? $(grep -E 'passed|failed|error' $OUT/prod_$v.log | tail -1 | cut -c1-150) $(date -u +%T)"
  grep -a -E "^(FAILED|ERROR) " $OUT/prod_$v.log | head -3 | cut -c1-300
  texthash /tmp/llkb_$v > $OUT/elfs_$v.txt
  echo "== elfs $v: $(wc -l < $OUT/elfs_$v.txt) files"
done
GFAIL=0
d=$(comm -13 <(cut -d' ' -f1 $OUT/elfs_M.txt | sort -u) <(cut -d' ' -f1 $OUT/elfs_H.txt | sort -u) | wc -l)
echo "== guard H against M: $d distinct H .text hashes absent from M (must be > 0)"; [[ $d == 0 ]] && GFAIL=1
read d n <<< "$(cmpelf H H2)"; echo "== guard H2 against H: $d of $n ELFs differ (must match)"; [[ $d != 0 ]] && GFAIL=1
[[ $GFAIL == 1 ]] && { echo "== GUARD FAIL"; exit 1; }
i=0
while read -r tag v ids rts mode; do
  [[ -z ${tag:-} || $tag == \#* || $rts == none ]] && continue
  if [[ $tag == bits ]]; then
    place $v; mapfile -t NODES < $S/$ids; rm -rf /tmp/llkbits_$v; mkdir -p /tmp/llkbits_$v
    echo "== bits $v ${#NODES[@]} tests $(date -u +%T)"
    env -u TT_LLK_DISABLE_ASSERTS RUNNER_TEMP=/tmp/llkbits_$v TMPDIR=/tmp/llkbits_$v TOPKBITS_OUT=$OUT/bits_$v.jsonl timeout 3000 python -m pytest -n ${NBITS:-8} -p topkbits -p no:cacheprovider --override-ini=log_cli=false -o timeout_method=thread -q -rfE "${NODES[@]}" > $OUT/bits_$v.log 2>&1
    echo "== bits $v rc=$? $(grep -E 'passed|failed|error' $OUT/bits_$v.log | tail -1 | cut -c1-150) $(wc -l < $OUT/bits_$v.jsonl 2>/dev/null) reads $(date -u +%T)"
    grep -a -E "^(FAILED|ERROR) " $OUT/bits_$v.log | head -5 | cut -c1-300
    continue
  fi
  i=$((i + 1)); t=${tag}_${i}_$v
  if (( $(date +%s) - T0 > DEADLINE )); then echo "== skip $t: past the deadline"; continue; fi
  place $v
  XA=(); [[ ${mode:-} == wc ]] && XA=(--dump-perf-counters)
  export LLK_PERF_RUN_TYPES=$rts
  # The perf CSV key has no form column and a module CSV must not repeat a key, so each form runs as its own lane
  # (own run tag); the checked-in modules (ids without a form) run first as lane "base".
  for lane in base $(sed -n 's/.*form:\([a-z0-9_]*\)\]$/\1/p' $S/$ids | sort -u); do
    if [[ $lane == base ]]; then grep -v -E "form:[a-z0-9_]+\]$" $S/$ids > /tmp/lane.txt
    else grep -E "form:${lane}\]$" $S/$ids > /tmp/lane.txt; fi
    mapfile -t NODES < /tmp/lane.txt
    # main's isolate mocks have no unpack_A pairing: its ua rows run L1_TO_L1 only
    if [[ $v == M && $lane == *ua ]]; then export LLK_PERF_RUN_TYPES=L1_TO_L1; else export LLK_PERF_RUN_TYPES=$rts; fi
    [[ ${#NODES[@]} == 0 ]] && continue
    tl=${t}_L$lane
    echo "== run $tl ${#NODES[@]} nodes rts=$rts $(date -u +%T)"
    RUNNER_TEMP=/tmp/llkb_$v TMPDIR=/tmp/llkb_$v PERF_RUN_TAG=$tl timeout 2400 python -m pytest --compile-consumer -p no:cacheprovider --override-ini=log_cli=false --timeout=120 -o timeout_method=thread -q -rfE "${XA[@]}" "${NODES[@]}" > $OUT/run_$tl.log 2>&1
    rc=$?
    echo "== run $tl rc=$rc $(grep -E 'passed|failed|error' $OUT/run_$tl.log | tail -1 | cut -c1-150) $(date -u +%T)"
    if [[ $rc != 0 ]]; then grep -a -E "^(FAILED|ERROR) |Error:|rror\b" $OUT/run_$tl.log | head -4 | cut -c1-300; fi
    if [[ -d $L/perf_data/runs/$tl ]]; then
      mkdir -p $OUT/csv/$tl
      find $L/perf_data/runs/$tl -name "*.csv" ! -name "*.post.csv" -exec cp {} $OUT/csv/$tl/ \;
    fi
  done
done < $PLAN
python3 $S/trim_counters.py $OUT/csv
(cd $OUT && python3 -c "import tarfile,glob; t=tarfile.open('/tmp/llkab.txz','w:xz'); [t.add(f) for f in ['csv']+glob.glob('elfs_*.txt')+glob.glob('bits_*.jsonl')]; t.close()")
echo "== results $(du -k /tmp/llkab.txz | cut -f1) kB"
echo "@@RESULTS_BEGIN"; base64 -w 200 /tmp/llkab.txz; echo "@@RESULTS_END"
echo "== done $(date -u +%T)"
