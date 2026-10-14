#!/usr/bin/env bash
# Pack-lever round (scratch, not for merge): tt-llk harness A/B, counters and functional runs on one CI Blackhole runner
# (knowledge/process/17_CI_AB.md). Each label's tree (ci_ab/variants/<label>/tree, laid over the checkout's device-side
# paths) is built and run from one slot path, so profiler zone ids and file strings match across labels.
#   ci_ab/lines.txt "<ids> <A> <B> <run types>": NC builds, ABBA twice (A1 B2 B3 A4 A5 B6 B7 A8); repeats of a label are its A/A.
#   ci_ab/wc.txt    "<ids> <label> <run types>": one counter (WC) build and run per line.
#   ci_ab/func.txt  "<ids> <label>": functional run with LLK asserts on.
#   ci_ab/guard.txt "<A> <B>": the two labels' compiled overlay files must differ (job stops otherwise); ELF images are
#   logged per label and node for the offline check.
set +e +o pipefail
unset RUNNER_TEMP
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; SL=${HR_SL:-/tmp/pl_slots}; cd $R
rm -rf $SL; mkdir -p $SL/out
echo "== runner $(hostname) $(nproc) cores $(date -u +%FT%TZ) head $(git rev-parse HEAD)"; ls /dev/tenstorrent 2>&1 | head -3
if [[ -n ${PL_PY:-} ]]; then PY=$PL_PY; else
  python3 -m venv $SL/venv && $SL/venv/bin/pip install -q -r tt_metal/tt-llk/tests/requirements.txt > $SL/pip.log 2>&1; echo "== pip rc=$? $(tail -1 $SL/pip.log)"
  PY=$SL/venv/bin/python3; [[ -x $PY ]] || PY=$(command -v python3)
fi
$PY -c "import tt_exalens, pytest, torch; print('== harness python ok', torch.__version__)"
SF=${PL_SF:-}
[[ -z $SF ]] && for c in $R/runtime/sfpi /opt/tenstorrent/sfpi; do [[ -x $c/compiler/bin/riscv-tt-elf-g++ ]] && { SF=$c; break; }; done
if [[ -z $SF ]]; then (cd tt_metal/tt-llk/tests && ./setup_testing_env.sh > $SL/sfpi.log 2>&1); SF=$R/tt_metal/tt-llk/tests/sfpi; fi
echo "== sfpi $SF: $($SF/compiler/bin/riscv-tt-elf-g++ --version | head -1)"
P="tt_metal/tt-llk tt_metal/hw tt_metal/hostdevcommon tt_metal/tools/profiler tt_metal/sfpi-version ttnn/cpp/ttnn/operations/experimental"
NP=${PL_NP:-$(( $(nproc) / 2 ))}; (( NP > 16 )) && NP=16; (( NP < 2 )) && NP=2
LABELS=$( { awk '!/^#/ && NF {print $2; print $3}' $S/lines.txt; cat $S/wc.txt $S/func.txt 2>/dev/null | awk '!/^#/ && NF {print $2}'; } | sort -u)
echo "== labels: $LABELS"
slot_in() { mv $SL/lab_$1 $SL/slot; }
slot_out() { mv $SL/slot $SL/lab_$1; }
for L in $LABELS; do
  D=$SL/lab_$L; mkdir -p $D/builds $D/builds_wc $D/builds_fn
  tar -C $R -cf - $P | tar -C $D -xf -
  if [[ -d $S/variants/$L/tree ]]; then
    (cd $S/variants/$L/tree && find . -type f ! -name .keep) | while read -r f; do mkdir -p "$(dirname "$D/$f")"; cp "$S/variants/$L/tree/$f" "$D/$f"; done
    (cd $S/variants/$L/tree && find . -type f ! -name .keep | sort) | (cd $D && xargs -r md5sum) > $SL/out/src_$L.md5
  else : > $SL/out/src_$L.md5; fi
  ln -sfn $SF $D/tt_metal/tt-llk/tests/sfpi; rm -rf $D/tt_metal/tt-llk/perf_data; mkdir -p $D/tt_metal/tt-llk/perf_data
done
# compiled overlay files: every label's copy of the union of overlay files
ALLF=$(cd $S/variants && find . -path '*/tree/*' -type f ! -name .keep | sed 's|^\./[^/]*/tree/||' | sort -u)
echo "== GUARD: compiled overlay files per label"
for L in $LABELS; do (cd $SL/lab_$L && echo "$ALLF" | xargs -r md5sum 2>/dev/null) > $SL/out/union_$L.md5; echo "-- $L $(md5sum < $SL/out/union_$L.md5 | cut -c1-12)"; done
guard_fail=0
while read -r A B; do
  [[ -z $A || $A == \#* ]] && continue
  if cmp -s $SL/out/union_$A.md5 $SL/out/union_$B.md5; then echo "GUARD FAIL: $A and $B compiled the same files"; guard_fail=1; else echo "GUARD ok: $A and $B differ"; fi
done < $S/guard.txt
[[ $guard_fail == 1 ]] && { echo "== stopping on the guard"; exit 1; }
build() { # build <label> <ids file> <run types> <builds subdir> <asserts 0|1> [extra]
  local L=$1 ids=$2 rts=$3 bd=$4 asr=$5; shift 5
  slot_in $L
  ( cd $SL/slot/tt_metal/tt-llk/tests/python_tests; mapfile -t N < $ids
    env CHIP_ARCH=blackhole LLK_HOME=$SL/slot/tt_metal/tt-llk TT_LLK_DISABLE_ASSERTS=$((1 - asr)) TMPDIR=$SL/slot/$bd LLK_PERF_L1_MUX_GROUP=0 ${rts:+LLK_PERF_RUN_TYPES=$rts} \
      RA_HASHLOG=$SL/out/${L}_$bd.hash PYTHONPATH=$S/tools timeout -s INT -k 60 3600 $PY -m pytest --compile-producer -n $NP -p no:cacheprovider \
      --override-ini=log_cli=false -q -p ra_hashlog "$@" "${N[@]}" < /dev/null > $SL/prod_${L}_$bd.log 2>&1 )
  echo "== built $L $bd ($rts, $(wc -l < $ids) nodes): $(grep -aE '[0-9]+ (passed|failed|error)' $SL/prod_${L}_$bd.log | tail -1) $(date -u +%T)"
  grep -a -m3 'error:' $SL/prod_${L}_$bd.log
  $PY $S/tools/ra_imgs.py $SL/slot/$bd $SL/out/${L}_$bd.hash > $SL/out/img_${L}_$bd.tsv 2>/dev/null
  slot_out $L
}
run() { # run <tag> <label> <ids file> <run types> <builds subdir> <asserts 0|1> [extra]
  local tag=$1 L=$2 ids=$3 rts=$4 bd=$5 asr=$6; shift 6; local log=$SL/out/run_$tag.log
  slot_in $L
  ( cd $SL/slot/tt_metal/tt-llk/tests/python_tests; mapfile -t N < $ids
    env CHIP_ARCH=blackhole LLK_HOME=$SL/slot/tt_metal/tt-llk TT_LLK_DISABLE_ASSERTS=$((1 - asr)) TMPDIR=$SL/slot/$bd LLK_PERF_L1_MUX_GROUP=0 ${rts:+LLK_PERF_RUN_TYPES=$rts} \
      PERF_RUN_TAG=$tag PERF_KEEP_RUNS=4000 RAWDUMP_DIR=$SL/out/raw/$tag PYTHONPATH=$S/tools timeout -s INT -k 60 1800 $PY -m pytest --compile-consumer -p no:cacheprovider -p rawdump \
      --override-ini=log_cli=false -q -rfE -o timeout_method=thread "$@" "${N[@]}" < /dev/null > $log 2>&1 )
  local rc=$?
  mkdir -p $SL/perf/$L; cp -r $SL/slot/tt_metal/tt-llk/perf_data/runs/$tag $SL/perf/$L/ 2>/dev/null
  slot_out $L
  echo "   $tag rc=$rc $(grep -aE '[0-9]+ (passed|failed|error)' $log | tail -1) $(date -u +%T)"
  grep -a -m5 -E '^FAILED|^ERROR' $log | cut -c1-220
  if grep -q "TIMED OUT" $log; then echo "== TENSIX TIMED OUT in $tag, stopping"; return 1; fi
}
# builds: NC per label over the union of its ids and run types
for L in $LABELS; do
  ids=$(awk -v l=$L -v s=$S '!/^#/ && NF && ($2==l || $3==l) {print s"/ids/"$1".txt"}' $S/lines.txt | sort -u)
  [[ -z $ids ]] && continue
  cat $ids | sort -u > $SL/ids_nc_$L.txt
  rts=$(awk -v l=$L '!/^#/ && NF && ($2==l || $3==l) {print $4}' $S/lines.txt | tr ',' '\n' | sort -u | paste -sd,)
  build $L $SL/ids_nc_$L.txt $rts builds 0
done
while read -r ids L rts; do
  [[ -z $ids || $ids == \#* ]] && continue
  build $L $S/ids/$ids.txt $rts builds_wc 0 --dump-perf-counters
done < <(cat $S/wc.txt 2>/dev/null)
stop=0; [[ -n ${PL_DRY:-} ]] && stop=1; LN=0
while read -r ids A B rts; do
  [[ -z $ids || $ids == \#* || $stop == 1 ]] && continue
  LN=$((LN+1)); echo "== $ids ($A against $B, $rts) $(date -u +%T)"; k=0
  for b in 1 2; do for s in A B B A; do k=$((k+1)); L=$A; [[ $s == B ]] && L=$B
    run ${ids}${LN}_${L}_$s$k $L $S/ids/$ids.txt $rts builds 0 || { stop=1; break 2; }; done; done
done < $S/lines.txt
while read -r ids L rts; do
  [[ -z $ids || $ids == \#* || $stop == 1 ]] && continue
  echo "== WC $ids $L ($rts) $(date -u +%T)"
  run wc_${ids}_$L $L $S/ids/$ids.txt $rts builds_wc 0 --dump-perf-counters || stop=1
done < <(cat $S/wc.txt 2>/dev/null)
while read -r ids L; do
  [[ -z $ids || $ids == \#* || $stop == 1 ]] && continue
  echo "== functional $ids $L (asserts on) $(date -u +%T)"
  build $L $S/ids/$ids.txt "" builds_fn 1
  run fn_${ids}_$L $L $S/ids/$ids.txt "" builds_fn 1 || stop=1
done < <(cat $S/func.txt 2>/dev/null)
tar -czf $SL/pl_out.tgz -C $SL perf out
echo "== results $(stat -c %s $SL/pl_out.tgz) bytes"; echo "=====RA_BEGIN====="; base64 -w 76 $SL/pl_out.tgz; echo "=====RA_END====="
exit 0
