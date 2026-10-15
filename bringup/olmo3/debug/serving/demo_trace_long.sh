#!/bin/bash
# traced long prefill in the demo, 3 repeated batches with the allocator-shifting spacer; prints per-batch text
cd /home/ubuntu/ssinghal/olmo3; source tt-metal/python_env/bin/activate
for cfg in "16k:long-16k-b1:16384:1000000000" "32k:long-32k-b1:32768:2000000000" "64k:long-64k-b1:65536:3000000000"; do
  IFS=: read tag name sl region <<< "$cfg"
  ps -eo pid,args | awk '/pytest|run_vllm_api|EngineCore/ && !/awk/ {print $1}' | xargs -r kill -9; sleep 3; tt-smi -glx_reset > /dev/null 2>&1
  OLMO_TRACE_SEQLENS=$sl OLMO_TRACE_REGION=$region OLMO_PERTURB_ALLOC=1 OLMO_REPEAT_BATCHES=3 OLMO_MAX_NEW=24 timeout 1500 ./run_demo.sh tr_$tag $name > ~/ssinghal/olmo3-logs/tr_$tag.out 2>&1
  echo "== $tag: $(grep -E '^== long' ~/ssinghal/olmo3-logs/tr_$tag.out | cut -c1-120)"
  grep -h "TT_FATAL\|trace buffers of size" ~/ssinghal/olmo3-logs/*tr_${tag}_$name.log | head -2 | cut -c1-200
  python - <<PY
import json,glob
from transformers import AutoTokenizer
tk=AutoTokenizer.from_pretrained("/home/ubuntu/models/olmo31-32b-think")
for f in sorted(glob.glob("/home/ubuntu/ssinghal/olmo3-logs/*tr_${tag}_${name}.json.b*"))[-3:]:
    d=json.load(open(f)); print("  ", f[-3:], repr(tk.decode(d["generated"][0])[:80]))
PY
done
echo trace-long-done
