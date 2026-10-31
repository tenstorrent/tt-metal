#!/bin/bash
# M1-default boot (DSV41_STRUCTURED_SPEC=1 VLLM_TT_STRUCTURED_SPEC=1 DSV41_INTRACE_SAMPLE=1, M2 off both sides, no mode file):
# 1 plain-only decode c8 ISL1k x3, GSM8K 250 (no structured requests yet), 2 structured suite, 3 tool matrix x2, 4 four-worker sweep. STOP at the end.
H=$1
B=/mnt/tt-data/ssinghal/dsv4-logs/structured; M=/mnt/tt-data/ssinghal/dsv4-logs/structured_m2b; O=$M/res/$H; mkdir -p $O/plain $O/s4
PY=/mnt/tt-data/ssinghal/tests/tt-metal/python_env/bin/python; PYV=/mnt/tt-data/ssinghal/wt/v031/venv/bin/python; C=$B/clients
PROBE=/mnt/tt-data/ssinghal/wt/b8_final/models/demos/blackhole/deepseek_v41_flash/tools/serving_tests/tool_protocol_probe.py
TC=/mnt/tt-data/ssinghal/dsv4-logs/cand_rel/tool_checks.py
DEC=/mnt/tt-data/ssinghal/dsv4-logs/stg_ab/dec.py
step() { echo "$1 start $(date -u +%T)" >> $O/progress; shift; "$@"; echo "rc=$? $(date -u +%T)" >> $O/progress; }
echo "load start $(date)" > $O/progress
curl -sf http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"deepseek-ai/DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"hi"}],"max_tokens":8}' -o /dev/null
step plain_c8 timeout 1800 $PYV $DEC 127.0.0.1 1024 256 8 3 plain_m1def $O/plain > $O/plain.log 2>&1
step gsm250 timeout 3000 $PY $M/gsm_check.py 250 8 $O/gsm250.json > $O/gsm250.log 2>&1
step probe timeout 600 $PY $PROBE --out $O/probe.json > $O/probe.log 2>&1
step matrix28 timeout 1200 $PY $C/matrix.py --workers 1 --out $O/matrix28.txt > $O/matrix28.stdout 2>&1
step prestream timeout 600 $PY $C/prestream.py > $O/prestream.log 2>&1
step stream1 timeout 900 $PY $M/stream_check.py --rounds 1 --par 1 --out $O/stream1.txt > $O/stream1.log 2>&1
step stream3 timeout 900 $PY $M/stream_check.py --rounds 1 --par 3 --out $O/stream3.txt > $O/stream3.log 2>&1
step mixed8 timeout 1500 $PY $C/matrix.py --workers 8 --filler 3 --rounds 2 --cases required,named,json_schema,regex,choice --out $O/mixed8.txt > $O/mixed8.stdout 2>&1
step sweep4 timeout 2400 $PYV $C/mode_ab.py /tmp/m1def_modefile_unused $O/s4 none,m1,ctl_tok,ctl_dur,m1,none 4 8 256 2 > $O/sweep_S4.log 2>&1
step toolmatrix1 timeout 2400 $PY $TC $O/tool_matrix_1 matrix > $O/tool_matrix_1.log 2>&1
step toolmatrix2 timeout 2400 $PY $TC $O/tool_matrix_2 matrix > $O/tool_matrix_2.log 2>&1
echo "load done $(date)" >> $O/progress
touch $M/srv/$H/STOP
