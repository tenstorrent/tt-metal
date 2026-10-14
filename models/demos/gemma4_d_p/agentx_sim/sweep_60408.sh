#!/bin/bash
# The calibrated sweep behind the tt-metal#60408 results: the whole 256k AgentX corpus (393 sessions, 68,266 requests),
# every chunk design with its silicon calibration, 32 and 64 sessions, think cap 60 s (heavy) and 1800 s (light).
#   sweep_60408.sh <traces.jsonl> <out.jsonl>     (~2.5 min on 24 cores)
set -e
H=$(dirname "$(readlink -f "$0")")
T=${1:?traces.jsonl}
O=${2:?out.jsonl}
S="python3 $H/agentx_sweep.py --traces $T --out $O -j 6 --concurrency 32,64 --max-think-s 60,1800"
$S --config 8192x8192:fcfs --scale 1.125 --host-ms 3.3 --cache inf,16e6,4e6 &
$S --config 4096x4096:fcfs --scale 1.057 --host-ms 3.3 --cache inf &
$S --config 2048x2048:fcfs --scale 1.009 --host-ms 3.3 --cache inf &
$S --config 2048x8192:fcfs,2048x8192:pack --scale 1.054 --host-ms 6.3 --cache inf &
wait
$S --config 8192x8192:fcfs --scale 1.125 --host-ms 3.3 --cache inf --align exact
