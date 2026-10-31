#!/bin/bash
# sweep hold: plain tok/s/user with S = 1, 2, 4 structured (or unstructured control) looping workers; then mask upload diff vs full at S=4; then canary in m2. STOP at the end.
H=$1
B=/mnt/tt-data/ssinghal/dsv4-logs/structured; M=/mnt/tt-data/ssinghal/dsv4-logs/structured_m2b; O=$M/res/$H; mkdir -p $O
PY=/mnt/tt-data/ssinghal/wt/v031/venv/bin/python; PY2=/mnt/tt-data/ssinghal/tests/tt-metal/python_env/bin/python
MODEF=/home/ttuser/ssinghal/tt-metal-cache/1017/structured_mode
UPF=/home/ttuser/ssinghal/tt-metal-cache/1017/mask_upload
echo "load start $(date -u +%T)" > $O/progress
echo m2 > $MODEF
curl -sf http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"deepseek-ai/DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"hi"}],"max_tokens":8}' -o /dev/null
mkdir -p $O/s1 $O/s2 $O/s4 $O/ufull $O/udiff; echo diff > $UPF
for MODE in m1 m0 m2; do
  echo $MODE > $MODEF; sleep 1
  echo "stream_check $MODE $(date -u +%T)" >> $O/progress
  timeout 900 $PY2 $M/stream_check.py --rounds 1 --par 3 --out $O/${MODE}_stream3.txt > $O/${MODE}_stream3.log 2>&1
done
echo m2 > $MODEF
for S in 1 2 4; do
  echo "sweep S=$S $(date -u +%T)" >> $O/progress
  timeout 2400 $PY $B/clients/mode_ab.py $MODEF $O/s$S none,m1,m2,ctl_tok,ctl_dur,m2,m1,none $S 8 256 2 > $O/sweep_S$S.log 2>&1
done
for U in full diff full diff; do
  echo $U > $UPF; echo "upload $U $(date -u +%T)" >> $O/progress
  timeout 1200 $PY $B/clients/mode_ab.py $MODEF $O/u$U m1,m2 4 8 256 2 > $O/upload_$U.$(date +%s).log 2>&1
done
echo diff > $UPF; echo m2 > $MODEF
echo "canary $(date -u +%T)" >> $O/progress
timeout 2400 $PY2 $B/clients/canary.py $O/canary.json 30 8000 8 > $O/canary.log 2>&1
echo "load done $(date -u +%T)" >> $O/progress
touch $M/srv/$H/STOP
