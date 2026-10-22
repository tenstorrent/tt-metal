#!/usr/bin/env bash
# Scratch (not for merge). Fp8 untilize dispatch/combine op level, ct 9..19 odd + 12, real and null, bits printed in real mode.
# Arms A main, A2 A/A of main, H PR head 779f5410a16, H2 A/A of H. 6 blocks of A H H2 A2 A2 H2 H A = 12 runs per arm. R3_REPS calls per node (first dropped).
source /work/ci_ab/common.sh
OUT=/work/generated/utp; rm -rf $OUT; mkdir -p $OUT; export AB_NODES_FILE=$S/nodes.txt
declare -A ES; r=0
for blk in 1 2 3 4 5 6; do for a in A H H2 A2 A2 H2 H A; do
  r=$((r+1)); use $a ${a%2}
  o=$OUT/${a}_r$r; mkdir -p $o; t0=$(date +%s)
  timeout -s INT -k 60 1500 python3 -m tracy -r -p --no-web-server -o $o $S/run_nodes.py > $o.log 2>&1
  rc=$?
  echo "== run $r $a rc=$rc $(( $(date +%s) - t0 )) s: $(grep -aE '[0-9]+ (passed|failed)' $o.log | tail -1)"
  grep -a -E '^(FAILED|ERROR)' $o.log | cut -c1-200 | head -6
  grep -a "RAB BITS" $o.log | sed "s/^/RAB B $a $r /" | sort -u | head -20
  python3 $S/sum_op19.py $o $a $r
  e=$(elfsum $TT_METAL_CACHE); ES[$a]+=" $e"
  echo "RAB ELF $a r$r $e $(ls -d $TT_METAL_CACHE/*/kernels/*/* 2>/dev/null | wc -l) builds"
  rm -rf $o
done; done
for a in A A2 H H2; do echo "RAB ELFS $a $(echo ${ES[$a]} | tr ' ' '\n' | sort | uniq -c | tr '\n' ' ')"; done
use A A
echo "== done $(date -u +%FT%TZ)"
