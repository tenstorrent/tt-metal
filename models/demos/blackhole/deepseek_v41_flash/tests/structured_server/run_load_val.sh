#!/bin/bash
# validation hold: for each of m1, m2 (live mode file): probe, matrix28, prestream, mixed8, stream_check; then the tool matrix (256) in m2 and m1. STOP at the end.
H=$1
B=/mnt/tt-data/ssinghal/dsv4-logs/structured; M=/mnt/tt-data/ssinghal/dsv4-logs/structured_m2b; O=$M/res/$H; mkdir -p $O
PY=/mnt/tt-data/ssinghal/tests/tt-metal/python_env/bin/python; C=$B/clients
PROBE=/mnt/tt-data/ssinghal/wt/b8_final/models/demos/blackhole/deepseek_v41_flash/tools/serving_tests/tool_protocol_probe.py
TC=/mnt/tt-data/ssinghal/dsv4-logs/cand_rel/tool_checks.py
MODEF=/home/ttuser/ssinghal/tt-metal-cache/1017/structured_mode
step() { echo "$1 start $(date -u +%T)" >> $O/progress; shift; "$@"; echo "rc=$? $(date -u +%T)" >> $O/progress; }
echo "load start $(date)" > $O/progress
curl -sf http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"deepseek-ai/DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"hi"}],"max_tokens":8}' -o /dev/null
for MODE in m1 m2; do
  echo $MODE > $MODEF; sleep 1
  step ${MODE}_probe timeout 600 $PY $PROBE --out $O/${MODE}_probe.json > $O/${MODE}_probe.log 2>&1
  step ${MODE}_matrix28 timeout 1200 $PY $C/matrix.py --workers 1 --out $O/${MODE}_matrix28.txt > $O/${MODE}_matrix28.stdout 2>&1
  step ${MODE}_prestream timeout 600 $PY $C/prestream.py > $O/${MODE}_prestream.log 2>&1
  step ${MODE}_stream timeout 900 $PY $M/stream_check.py --rounds 1 --par 1 --out $O/${MODE}_stream1.txt > $O/${MODE}_stream1.log 2>&1
  step ${MODE}_stream_par timeout 900 $PY $M/stream_check.py --rounds 1 --par 3 --out $O/${MODE}_stream3.txt > $O/${MODE}_stream3.log 2>&1
  step ${MODE}_mixed8 timeout 1500 $PY $C/matrix.py --workers 8 --filler 3 --rounds 2 --cases required,named,json_schema,regex,choice --out $O/${MODE}_mixed8.txt > $O/${MODE}_mixed8.stdout 2>&1
done
for MODE in m2 m1; do
  echo $MODE > $MODEF; sleep 1
  step ${MODE}_toolmatrix timeout 2400 $PY $TC $O/tool_matrix_$MODE matrix > $O/${MODE}_tool_matrix.log 2>&1
done
echo m2 > $MODEF
echo "load done $(date)" >> $O/progress
touch $M/srv/$H/STOP
