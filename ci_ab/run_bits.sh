#!/usr/bin/env bash
# Scratch (not for merge). Bits of the dispatch/combine real-mode cases per arm (A main, H PR head, F fd36d27), one pass each, no timing.
source /work/ci_ab/common.sh
OUT=/work/generated/utb; rm -rf $OUT; mkdir -p $OUT; export R3_REPS=1
grep real $S/nodes.txt > $S/nodes_r.txt; export AB_NODES_FILE=$S/nodes_r.txt
for a in A H F Q S; do
  use $a $a; export RAB_BITS_FILE=$OUT/bits_$a.txt; rm -f $RAB_BITS_FILE
  timeout -s INT -k 60 1200 python3 -m pytest -q -p r3_prof_plugin -p no:cacheprovider -o timeout_method=thread -o addopts= -rfE $(cat $S/nodes_r.txt) > $OUT/$a.log 2>&1
  echo "== bits arm $a rc=$? $(grep -aE '[0-9]+ (passed|failed)' $OUT/$a.log | tail -1)"
  sort $RAB_BITS_FILE | sed "s/^/RAB BITSARM $a /"
done
use A A
