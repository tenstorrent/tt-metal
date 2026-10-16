#!/bin/bash
# Demo regression with NO tt-smi reset. Stops at the first failure and never resets anything.
# Results: ~/ssinghal/olmo3-logs/demo_regress.out
OUT=/home/ubuntu/ssinghal/olmo3-logs/demo_regress.out
echo "start $(date -u +%H:%M:%S) uptime $(uptime -p)" >> $OUT
sleep 120   # let the boards settle after the power cycle
cd /home/ubuntu/ssinghal/olmo3; source tt-metal/python_env/bin/activate
for cfg in batch-32 batch-1 gate-b32 long-4k-b32 long-8k-b1 long-16k-b1 aime24-p1 long-32k-b1 long-64k-b1; do
  line=$(./run_demo.sh final $cfg 2>&1 | tail -1); echo "$line" | cut -c1-240 >> $OUT
  echo "$line" | grep -q "rc=0" || { echo "STOP: $cfg did not pass; not continuing, not resetting" >> $OUT; exit 1; }
done
line=$(OLMO_PREFILL_SAMPLING=1 OLMO_SAMPLE_T=0.6 OLMO_SAMPLE_SEED=none OLMO_MAX_NEW=1024 ./run_demo.sh finalps gate-b32 2>&1 | tail -1); echo "gate-b32 + device first-token sampling: $line" | cut -c1-240 >> $OUT
python - >> $OUT 2>&1 <<'PY'
import json,re,glob
from transformers import AutoTokenizer
tk=AutoTokenizer.from_pretrained("/home/ubuntu/models/olmo31-32b-think")
f=sorted(glob.glob("/home/ubuntu/ssinghal/olmo3-logs/*finalps_gate-b32.json"))[-1]
d=json.load(open(f)); t=[tk.decode(g) for g in d["generated"]]
w=sum(len(x.split()) for x in t); g=sum(len(re.findall(r"\b([A-Za-z]+) \1\b",x)) for x in t)
print(f"  device first-token sampling gate-b32: words={w} doubled={g}")
PY
echo demo-regress-done >> $OUT
