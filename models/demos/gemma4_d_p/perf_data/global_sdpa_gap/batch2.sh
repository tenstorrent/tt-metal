#!/bin/bash
# Sequential experiments; each run capped at 15 min. A timeout means a likely device hang: stop the batch.
W=/data/mmanzoor/global_sdpa_gap_c04u02
run() { local tag=$1; shift; echo "== $(date +%T) $tag ($*)"
  env "$@" timeout -k 30 900 $W/run_layer.sh $tag 64 >/dev/null 2>&1; local rc=$?
  grep -h "RESULT\|^exit\|TT_THROW\|FAILED" $W/$tag.log | tail -3; cat $W/$tag.ops.txt 2>/dev/null
  if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then echo "TIMEOUT on $tag -> stopping batch (device may need reset)"; echo BATCH_STOPPED; exit 1; fi; }
run q64k512 GEMMA4_GLOBAL_SDPA_CFG=64,512,3,1
run pr TT_RING_SDPA_PARTIAL_READINESS=1
echo BATCH_DONE
