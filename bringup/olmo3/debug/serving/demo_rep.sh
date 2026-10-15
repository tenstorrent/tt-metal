#!/bin/bash
# repeat the 8K prefill 3x in the demo under different perturbations; print the decoded text of each batch
cd /home/ubuntu/ssinghal/olmo3; source tt-metal/python_env/bin/activate
pkill_devs() { ps -eo pid,args | awk '/pytest|run_vllm_api|EngineCore|run.py --model allenai/ && !/awk/ {print $1}' | xargs -r kill -9; }
for cfg in "perturb:OLMO_PERTURB_ALLOC=1" "nokvreset:OLMO_NO_KV_RESET=1" "both:OLMO_PERTURB_ALLOC=1 OLMO_NO_KV_RESET=1"; do
  tag=${cfg%%:*}; envs=${cfg#*:}
  pkill_devs; sleep 3; tt-smi -glx_reset > /dev/null 2>&1
  env $envs OLMO_REPEAT_BATCHES=3 OLMO_MAX_NEW=24 ./run_demo.sh rep_$tag long-8k-b1 > ~/ssinghal/olmo3-logs/rep_$tag.out 2>&1
  echo "== $tag ($envs): $(grep -E '^== long-8k' ~/ssinghal/olmo3-logs/rep_$tag.out | cut -c1-60)"
  python - <<PY
import json,glob
from transformers import AutoTokenizer
tk=AutoTokenizer.from_pretrained("/home/ubuntu/models/olmo31-32b-think")
for f in sorted(glob.glob("/home/ubuntu/ssinghal/olmo3-logs/*rep_${tag}_long-8k-b1.json.b*"))[-3:]:
    d=json.load(open(f)); print("  ", f[-3:], repr(tk.decode(d["generated"][0])[:90]))
PY
done
echo demo-rep-done
