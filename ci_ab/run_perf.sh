#!/usr/bin/env bash
# Scratch op-level job: ttnn.untilize (tracy) at production block-float shapes (ct 9 to 128) and bf16 / fp32 controls, arms A B C C2
# in A B C C2 C2 C B A order, twice, R3_REPS calls per node (the first a warm-up), per op code sums, ELF guard.
source /work/ci_ab/common.sh
OUT=/work/generated/utp; rm -rf $OUT; mkdir -p $OUT; export AB_NODES_FILE=$S/nodes.txt
declare -A ES; r=0; fail=0
for blk in 1 2; do for a in A B C C2 C2 C B A; do
  r=$((r+1)); use $a ${a%2}
  o=$OUT/${a}_r$r; mkdir -p $o; t0=$(date +%s)
  timeout -s INT -k 60 2400 python3 -m tracy -r -p --no-web-server -o $o $S/run_nodes.py > $o.log 2>&1
  rc=$?
  echo "== run $r $a rc=$rc $(( $(date +%s) - t0 )) s: $(grep -aE '[0-9]+ (passed|failed)' $o.log | tail -1)"
  [[ $rc -ne 0 ]] && { fail=1; grep -a -m5 -E '^(FAILED|ERROR)|Error' $o.log | cut -c1-200; }
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
echo "RAB GUARD stable=$gs C_vs_C2=$gc A_B_C_distinct=$gd"; [[ $gs == ok && $gc == ok && $gd == ok ]] || fail=1
for a in A B C; do python3 $S/elfhash.py $a $OUT/cache_$a untilize_metal2 untilize; done
use A A
echo "== done $(date -u +%FT%TZ) fail=$fail"
exit $fail
