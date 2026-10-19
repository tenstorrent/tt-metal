#!/usr/bin/env bash
# Scratch, not for merge. TopK all-configuration sweep on one CI Blackhole runner (test-dispatch), copied from the eltwise
# binary run_eb_allcfg.sh. Variants (ci_ab/variants/<v>/, paths in ci_ab/files.txt): M main's TopK LLK files, H and H2 the
# stack head's (#60291), Mnc and Hnc the same with thread-issued SFPSWAPs turned into non-SFPU NOPs (swap count probe).
# Every variant is placed into the one checkout before its build and runs; each has its own fresh build directory.
# <v>_wc is <v> built with perf counters. Plan lines: "<tag> <variant> <ids file> <run types|all|none> [wc]".
# Guard: ELF .text hashes per variant; M and H must differ, H and H2 must match, M and Mnc must differ.
set +e +o pipefail
T0=$(date +%s); DEADLINE=${DEADLINE:-5400}
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; L=$R/tt_metal/tt-llk
PLAN=$S/${1:-plan.txt}; FL=$S/files.txt
echo "== start $(date -u +%T) checkout $R $(git -C $R log -1 --format=%h) plan $PLAN nproc $(nproc)"
python3 -m venv /tmp/llkvenv && . /tmp/llkvenv/bin/activate
pip install -q -r $L/tests/requirements.txt > /tmp/pip.log 2>&1; echo "== pip rc=$? $(tail -1 /tmp/pip.log)"
(cd $L/tests && ./setup_testing_env.sh > /tmp/setup.log 2>&1); echo "== setup rc=$? $(tail -2 /tmp/setup.log | tr '\n' ' ')"
export CHIP_ARCH=blackhole LLK_HOME=$L TT_LLK_DISABLE_ASSERTS=1 LLK_PERF_L1_MUX_GROUP=0 PERF_KEEP_RUNS=1000
OBJ=$(find $L/tests/sfpi/ -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1); echo "== objcopy $OBJ"
place() {
  local d=${1%_wc}
  while read -r rel; do [[ -n $rel ]] && cp "$S/variants/$d/$rel" "$R/$rel"; done < $FL
  echo "== place $1: $(while read -r rel; do [[ -n $rel ]] && md5sum < "$R/$rel" | cut -c1-6; done < $FL | md5sum | cut -c1-12)"
}
texthash() { (cd $1 && find . -name "*.elf" -path "*/elf/*" | sort | while read -r e; do echo "$($OBJ -O binary -j .text "$e" /dev/stdout 2>/dev/null | md5sum | cut -c1-12) ${e#./}"; done); }
cmpelf() { join -j2 <(sort -k2 $OUT/elfs_$1.txt) <(sort -k2 $OUT/elfs_$2.txt) | awk '{n++; if ($2 != $3) d++} END {print d+0, n+0}'; }
OUT=/tmp/tkab_out; rm -rf $OUT; mkdir -p $OUT/csv
cd $L/tests/python_tests
VARS=$(awk '!/^#/ && NF {print $2}' $PLAN | awk '!s[$0]++')
for v in $VARS; do
  place $v
  ids=$(awk -v v=$v '!/^#/ && NF && $2==v {print $3}' $PLAN | sort -u)
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
GFAIL=0; d=$(comm -13 <(cut -d' ' -f1 $OUT/elfs_H.txt | sort -u) <(cut -d' ' -f1 $OUT/elfs_F8.txt | sort -u) | wc -l); echo "== guard F8 against H: $d distinct hashes absent (must be > 0)"; [[ $d == 0 ]] && GFAIL=1
d=$(comm -13 <(cut -d' ' -f1 $OUT/elfs_M.txt | sort -u) <(cut -d' ' -f1 $OUT/elfs_H.txt | sort -u) | wc -l)
echo "== guard H against M: $d of $(cut -d' ' -f1 $OUT/elfs_H.txt | sort -u | wc -l) distinct H .text hashes absent from M (must be > 0)"; [[ $d == 0 ]] && GFAIL=1
read d n <<< "$(cmpelf H H2)"; echo "== guard H2 against H: $d of $n ELFs differ (must match)"; [[ $d != 0 ]] && GFAIL=1
if [[ -f $OUT/elfs_Mnc_wc.txt ]]; then d=$(comm -13 <(cut -d" " -f1 $OUT/elfs_M_wc.txt | sort -u) <(cut -d" " -f1 $OUT/elfs_Mnc_wc.txt | sort -u) | wc -l); echo "== guard Mnc_wc against M_wc: $d distinct hashes absent (must be > 0)"; [[ $d == 0 ]] && GFAIL=1; fi
if [[ -f $OUT/elfs_Hnc_wc.txt ]]; then d=$(comm -13 <(cut -d" " -f1 $OUT/elfs_H_wc.txt | sort -u) <(cut -d" " -f1 $OUT/elfs_Hnc_wc.txt | sort -u) | wc -l); echo "== guard Hnc_wc against H_wc: $d distinct hashes absent (must be > 0)"; [[ $d == 0 ]] && GFAIL=1; fi
[[ $GFAIL == 1 ]] && { echo "== GUARD FAIL"; exit 1; }
i=0
while read -r tag v ids rts mode; do
  [[ -z ${tag:-} || $tag == \#* || $rts == none ]] && continue
  i=$((i + 1)); t=${tag}_${i}_$v
  if (( $(date +%s) - T0 > DEADLINE )); then echo "== skip $t: past the deadline"; continue; fi
  place $v
  mapfile -t NODES < $S/$ids
  XA=(); [[ ${mode:-} == wc ]] && XA=(--dump-perf-counters)
  [[ $rts == all ]] && unset LLK_PERF_RUN_TYPES || export LLK_PERF_RUN_TYPES=$rts
  echo "== run $t ${#NODES[@]} nodes rts=$rts $(date -u +%T)"
  RUNNER_TEMP=/tmp/llkb_$v TMPDIR=/tmp/llkb_$v PERF_RUN_TAG=$t timeout 2400 python -m pytest --compile-consumer -p no:cacheprovider --override-ini=log_cli=false --timeout=120 -o timeout_method=thread -q -rfE "${XA[@]}" "${NODES[@]}" > $OUT/run_$t.log 2>&1
  rc=$?
  echo "== run $t rc=$rc $(grep -E 'passed|failed|error' $OUT/run_$t.log | tail -1 | cut -c1-150) $(date -u +%T)"
  [[ $rc != 0 ]] && grep -a -E "^(FAILED|ERROR) " $OUT/run_$t.log | head -3 | cut -c1-300
  if [[ -d $L/perf_data/runs/$t ]]; then
    mkdir -p $OUT/csv/$t
    find $L/perf_data/runs/$t -name "*.csv" ! -name "*.post.csv" -exec cp {} $OUT/csv/$t/ \;
  fi
done < $PLAN
python3 $S/trim_counters.py $OUT/csv
(cd $OUT && python3 -c "import tarfile,glob; t=tarfile.open('/tmp/tkab.txz','w:xz'); [t.add(f) for f in ['csv']+glob.glob('elfs_*.txt')]; t.close()")
echo "== results $(du -k /tmp/tkab.txz | cut -f1) kB"
echo "@@RESULTS_BEGIN"; base64 -w 200 /tmp/tkab.txz; echo "@@RESULTS_END"
echo "== done $(date -u +%T)"
