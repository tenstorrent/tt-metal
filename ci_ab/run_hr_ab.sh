#!/usr/bin/env bash
# #59149 stack round, scratch: tt-llk harness A/B on one CI Blackhole runner (knowledge/process/17_CI_AB.md). Each label (ci_ab/variants/<label>/tree
# laid over the checkout's device-side paths) is built and run from one slot path (/tmp/hr_slots/slot), so profiler zone
# ids and file strings match across labels. Each line of ci_ab/lines.txt ("<ids> <A label> <B label> <run types>") runs
# ABBA twice (A1 B2 B3 A4 A5 B6 B7 A8); the repeats of one label are its A/A. Guard: each label's compiled variant files
# are hashed and the job fails when two labels that should differ compiled the same files.
set +e +o pipefail
unset RUNNER_TEMP
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; SL=${HR_SL:-/tmp/hr_slots}; cd $R
rm -rf $SL; mkdir -p $SL/out
echo "== runner $(hostname) $(nproc) cores $(date -u +%FT%TZ) head $(git rev-parse HEAD)"; ls /dev/tenstorrent 2>&1 | head -3
if [[ -n ${HR_PY:-} ]]; then PY=$HR_PY; else
  python3 -m venv $SL/venv && $SL/venv/bin/pip install -q -r tt_metal/tt-llk/tests/requirements.txt > $SL/pip.log 2>&1; echo "== pip rc=$? $(tail -1 $SL/pip.log)"
  PY=$SL/venv/bin/python3; [[ -x $PY ]] || PY=$(command -v python3)
fi
$PY -c "import tt_exalens, pytest, torch; print('== harness python ok', torch.__version__)"
SF=${HR_SF:-}
[[ -z $SF ]] && for c in $R/runtime/sfpi /opt/tenstorrent/sfpi; do [[ -x $c/compiler/bin/riscv-tt-elf-g++ ]] && { SF=$c; break; }; done
if [[ -z $SF ]]; then (cd tt_metal/tt-llk/tests && ./setup_testing_env.sh > $SL/sfpi.log 2>&1); SF=$R/tt_metal/tt-llk/tests/sfpi; fi
echo "== sfpi $SF: $($SF/compiler/bin/riscv-tt-elf-g++ --version | head -1)"
$SF/compiler/bin/riscv-tt-elf-g++ --version | grep -q '7\.87\.0' || { echo "== wrong sfpi, stopping"; exit 1; }
P="tt_metal/tt-llk tt_metal/hw tt_metal/hostdevcommon tt_metal/tools/profiler tt_metal/sfpi-version ttnn/cpp/ttnn/operations/experimental"
NP=$(( $(nproc) / 2 )); (( NP > 16 )) && NP=16; (( NP < 4 )) && NP=4
LABELS=$(awk '!/^#/ && NF {print $2; print $3}' $S/lines.txt | sort -u)
slot_in() { mv $SL/lab_$1 $SL/slot; }
slot_out() { mv $SL/slot $SL/lab_$1; }
for L in $LABELS; do
  D=$SL/lab_$L; mkdir -p $D/builds
  tar -C $R -cf - $P | tar -C $D -xf -
  (cd $S/variants/$L/tree && find . -type f) | while read -r f; do mkdir -p "$(dirname "$D/$f")"; cp "$S/variants/$L/tree/$f" "$D/$f"; done
  ln -sfn $SF $D/tt_metal/tt-llk/tests/sfpi; rm -rf $D/tt_metal/tt-llk/perf_data; mkdir -p $D/tt_metal/tt-llk/perf_data
  ( cd $D && md5sum tt_metal/hw/inc/api/compute/pack_untilize.h tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_A.h tt_metal/tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_unpack_fast_untilize.h tt_metal/tt-llk/tests/sources/pack_untilize_perf.cpp tt_metal/tt-llk/tests/sources/fast_untilize_test.cpp ) > $SL/out/src_$L.md5
  rts=$(awk -v l=$L '$2==l || $3==l {print $4}' $S/lines.txt | tr ',' '\n' | sort -u | paste -sd,)
  cat $(awk -v l=$L -v s=$S '$2==l || $3==l {print s"/ids/"$1".txt"}' $S/lines.txt) | sort -u > $SL/ids_$L.txt
  slot_in $L
  ( cd $SL/slot/tt_metal/tt-llk/tests/python_tests; mapfile -t N < $SL/ids_$L.txt
    env CHIP_ARCH=blackhole LLK_HOME=$SL/slot/tt_metal/tt-llk TT_LLK_DISABLE_ASSERTS=1 TMPDIR=$SL/slot/builds LLK_PERF_L1_MUX_GROUP=0 LLK_PERF_RUN_TYPES=$rts \
      RA_HASHLOG=$SL/out/$L.hash PYTHONPATH=$S/tools timeout -s INT -k 60 3600 $PY -m pytest --compile-producer -n $NP -p no:cacheprovider \
      --override-ini=log_cli=false -q -p ra_hashlog "${N[@]}" < /dev/null > $SL/prod_$L.log 2>&1 )
  echo "== built $L ($rts, $(wc -l < $SL/ids_$L.txt) nodes): $(grep -aE '[0-9]+ (passed|failed|error)' $SL/prod_$L.log | tail -1) $(date -u +%T)"
  grep -a -m3 'error:' $SL/prod_$L.log
  $PY $S/tools/ra_imgs.py $SL/slot/builds $SL/out/$L.hash > $SL/out/img_$L.tsv 2>/dev/null
  slot_out $L
done
echo "== GUARD: compiled variant files per label"; for L in $LABELS; do echo "-- $L"; cat $SL/out/src_$L.md5; done
guard_fail=0
while read -r tag A B rts; do
  [[ -z $tag || $tag == \#* ]] && continue
  if cmp -s $SL/out/src_$A.md5 $SL/out/src_$B.md5; then echo "GUARD FAIL: $A and $B compiled the same files"; guard_fail=1; else echo "GUARD ok: $A and $B compiled different files"; fi
done < $S/lines.txt
[[ $guard_fail == 1 ]] && { echo "== stopping on the guard"; exit 1; }
run() { # run <tag> <side> <k> <label> <rts>
  local tag=$1 s=$2 k=$3 L=$4 rts=$5 log=$SL/out/run_$1_$2$3.log
  slot_in $L
  ( cd $SL/slot/tt_metal/tt-llk/tests/python_tests; mapfile -t N < $S/ids/$tag.txt
    env CHIP_ARCH=blackhole LLK_HOME=$SL/slot/tt_metal/tt-llk TT_LLK_DISABLE_ASSERTS=1 TMPDIR=$SL/slot/builds LLK_PERF_L1_MUX_GROUP=0 LLK_PERF_RUN_TYPES=$rts \
      PERF_RUN_TAG=${tag}${LN}_${L}_$s$k PERF_KEEP_RUNS=4000 timeout -s INT -k 60 900 $PY -m pytest --compile-consumer -p no:cacheprovider \
      --override-ini=log_cli=false -q -o timeout_method=thread "${N[@]}" < /dev/null > $log 2>&1 )
  local rc=$?
  mkdir -p $SL/perf/$L; cp -r $SL/slot/tt_metal/tt-llk/perf_data/runs/${tag}${LN}_${L}_$s$k $SL/perf/$L/ 2>/dev/null
  slot_out $L
  echo "   $tag $L $s$k rc=$rc $(grep -aE '[0-9]+ (passed|failed|error)' $log | tail -1) $(date -u +%T)"
  if grep -q "TIMED OUT" $log; then echo "== TENSIX TIMED OUT in $tag $s$k, stopping"; return 1; fi
}
stop=0; [[ -n ${HR_DRY:-} ]] && stop=1; LN=0
while read -r tag A B rts; do
  [[ -z $tag || $tag == \#* || $stop == 1 ]] && continue
  LN=$((LN+1))
  echo "== $tag ($A against $B, $rts) $(date -u +%T)"; k=0
  for b in 1 2; do for s in A B B A; do k=$((k+1)); L=$A; [[ $s == B ]] && L=$B; run $tag $s $k $L $rts || { stop=1; break 2; }; done; done
done < $S/lines.txt
tar -czf $SL/hr_out.tgz -C $SL perf out
echo "== results $(stat -c %s $SL/hr_out.tgz) bytes"; echo "=====RA_BEGIN====="; base64 -w 76 $SL/hr_out.tgz; echo "=====RA_END====="
exit 0
