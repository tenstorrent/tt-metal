#!/usr/bin/env bash
# Scratch op-level job (not for merge): Fp8_e4m3 untilize kernels through ttnn.generic_op (tracy), arms A main, B PR head fd36d27b42b,
# C the odd-block commit, C2 A/A copy of C, order A B C C2 C2 C B A twice, R3_REPS calls per node (first is the warm-up), ELF guard,
# then the split test file on B and C (test count per arm).
source /work/ci_ab/common.sh
OUT=/work/generated/utp; rm -rf $OUT; mkdir -p $OUT; export AB_NODES_FILE=$S/nodes.txt
declare -A ES; r=0; fail=0
for blk in 1 2; do for a in A B C C2 C2 C B A; do
  r=$((r+1)); use $a ${a%2}
  o=$OUT/${a}_r$r; mkdir -p $o; t0=$(date +%s)
  timeout -s INT -k 60 2400 python3 -m tracy -r -p --no-web-server -o $o $S/run_nodes.py > $o.log 2>&1
  rc=$?
  echo "== run $r $a rc=$rc $(( $(date +%s) - t0 )) s: $(grep -aE '[0-9]+ (passed|failed)' $o.log | tail -1)"
  grep -a -E '^(FAILED|ERROR)' $o.log | cut -c1-200 | head -12
  python3 $S/sum_op19.py $o $a $r
  e=$(elfsum $TT_METAL_CACHE); ES[$a]+=" $e"
  echo "RAB ELF $a r$r $e $(ls -d $TT_METAL_CACHE/*/kernels/*/* 2>/dev/null | wc -l) builds"
done; done
uniq1() { echo $1 | tr ' ' '\n' | sort -u | wc -l; }
one() { echo $1 | tr ' ' '\n' | grep . | head -1; }
gs=ok; for a in A B C C2; do [[ $(uniq1 "${ES[$a]}") == 1 ]] || gs=UNSTABLE; done
[[ -n $OBJ ]] || gs=VOID
gc=$([[ $(one "${ES[C]}") == $(one "${ES[C2]}") ]] && echo ok || echo DIFF)
gd=$([[ $(one "${ES[A]}") != $(one "${ES[B]}") && $(one "${ES[B]}") != $(one "${ES[C]}") ]] && echo ok || echo VOID)
echo "RAB GUARD stable=$gs C_vs_C2=$gc A_B_C_distinct=$gd"
for a in B C; do
  use t_$a $a
  timeout -s INT -k 60 1500 python3 -m pytest -q -p no:cacheprovider -o timeout_method=thread -o addopts= -rfE tests/ttnn/unit_tests/operations/data_movement/test_untilize_fp8_row_split.py > $OUT/split_$a.log 2>&1
  echo "== split tests $a rc=$? $(grep -aE '[0-9]+ (passed|failed)' $OUT/split_$a.log | tail -1)"
done
use A A
echo "== done $(date -u +%FT%TZ)"
