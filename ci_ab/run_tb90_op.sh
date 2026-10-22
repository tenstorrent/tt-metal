#!/usr/bin/env bash
# Scratch, not for merge. #59135 candidate 7ff78ce4209 (early poll in transpose_tile) at the ttnn level, one job:
# 1 functional ttnn transpose/permute/sort tests on the candidate, failures rerun on the PR head file;
# 2 op level (tracy) ABBA of arms H (PR head transpose.h) and C (candidate) with A/A copies H2 C2, fresh caches;
# 3 bits: output sha1 per arm and shape (BITS lines), ELF guard on the transpose kernels.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 R3_REPS=${R3_REPS:-5} AB_NODES_FILE=/work/ci_ab/nodes_tb90.txt
S=/work/ci_ab; OUT=/work/generated/tb90; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
export PYTHONPATH=$S:${PYTHONPATH:-}
echo "== $(git log -1 --format='%h %s' | cut -c1-80)"
HDR=tt_metal/hw/inc/api/compute/transpose.h
TARGETS="/work/$HDR"
for d in $(python3 -c "import ttnn,os;print(os.path.dirname(ttnn.__file__))" 2>/dev/null) /work/runtime /work/ttnn; do
  [[ -f $d/$HDR ]] && TARGETS="$TARGETS $d/$HDR"
  f=$(find $d -maxdepth 6 -path '*api/compute/transpose.h' 2>/dev/null | head -1); [[ -n $f ]] && TARGETS="$TARGETS $f"
done
TARGETS=$(printf '%s\n' $TARGETS | sort -u); echo "== header targets: $TARGETS"
setarm() { for t in $TARGETS; do cp $S/var/$1/transpose.h $t; done; echo "== arm $1 $(md5sum /work/$HDR | cut -c1-8)"; }
FUNC="tests/tt_eager/python_api_testing/unit_testing/misc/test_transpose.py
tests/ttnn/unit_tests/operations/data_movement/test_permute.py
tests/ttnn/unit_tests/operations/data_movement/test_sort.py"
setarm C
for t in $FUNC; do
  timeout -s INT -k 60 1500 python3 -m pytest -q -p no:cacheprovider -o timeout_method=thread -x --maxfail=50 $t > $OUT/func_C.log 2>&1
  echo "== func C $t: $(grep -aE '[0-9]+ (passed|failed)' $OUT/func_C.log | tail -1)"
  grep -aE '^(FAILED|ERROR) ' $OUT/func_C.log | cut -c1-250 | sort -u | head -30 > $OUT/fail_C.txt; cat $OUT/fail_C.txt
  if [[ -s $OUT/fail_C.txt ]]; then
    setarm H
    sed 's/^[A-Z]* //; s/ - .*//' $OUT/fail_C.txt > $OUT/rerun.txt
    timeout -s INT -k 60 1500 python3 -m pytest -q -p no:cacheprovider -o timeout_method=thread $(cat $OUT/rerun.txt | tr '\n' ' ') > $OUT/func_H.log 2>&1
    echo "== rerun on head $t: $(grep -aE '[0-9]+ (passed|failed)' $OUT/func_H.log | tail -1)"
    setarm C
  fi
done
for a in H C C2 H2 H2 C2 C H; do
  k=$((k+1)); case $a in H|H2) v=H;; *) v=C;; esac
  setarm $v
  o=$OUT/r${k}_$a; mkdir -p $o; export TT_METAL_CACHE=$OUT/cache_${k}_$a EB_ARM=$a EB_BITS_OUT=$OUT/bits.txt
  timeout -s INT -k 60 2400 python3 -m tracy -r -p --no-web-server -o $o $S/run_nodes.py > $o.log 2>&1
  echo "== run $k $a rc=$?: $(grep -aE '[0-9]+ (passed|failed)' $o.log | tail -1)"
  python3 $S/sum_prof.py $o $a $k | sed 's/^/OPDATA /'
  python3 $S/elfhash.py $k$a $TT_METAL_CACHE transpose permute | cut -c1-200 | sed 's/^/ELFH /' | head -6
done
sort -u $OUT/bits.txt | sed 's/^/BITS /'
echo "== done"
