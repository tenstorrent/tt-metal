#!/usr/bin/env bash
# Scratch (not for merge). Bisect of the untilize PR commits on the odd-ct dispatch/combine real-mode cases.
# Arms A main, A2 A/A, then the PR commits E ecf5f5e, D 5f42914, P 68668cf, Q cf26bde, R dadbe90, S 511764d, F fd36d27, H 779f541. 10 blocks, order alternates.
source /work/ci_ab/common.sh
OUT=/work/generated/utp; rm -rf $OUT; mkdir -p $OUT; export AB_NODES_FILE=$S/nodes_b.txt
declare -A ES; r=0
for blk in 1 2 3 4 5 6 7 8 9 10; do
  if (( blk % 2 )); then L="A E D P Q R S F H A2"; else L="A2 H F S R Q P D E A"; fi
  for a in $L; do
  r=$((r+1)); use $a ${a%2}
  o=$OUT/${a}_r$r; mkdir -p $o
  timeout -s INT -k 60 1500 python3 -m tracy -r -p --no-web-server -o $o $S/run_nodes.py > $o.log 2>&1
  rc=$?
  echo "== run $r $a rc=$rc: $(grep -aE '[0-9]+ (passed|failed)' $o.log | tail -1)"
  grep -a -E '^(FAILED|ERROR)' $o.log | cut -c1-200 | head -6
  grep -a "RAB BITS" $o.log | sed "s/^/RAB B $a $r /" | sort -u | head -20
  python3 $S/sum_op19.py $o $a $r
  e=$(elfsum $TT_METAL_CACHE); ES[$a]+=" $e"
  rm -rf $o
done; done
for a in A A2 E D P Q R S F H; do echo "RAB ELFS $a $(echo ${ES[$a]} | tr ' ' '\n' | sort | uniq -c | tr '\n' ' ')"; done
use A A
echo "== done $(date -u +%FT%TZ)"
