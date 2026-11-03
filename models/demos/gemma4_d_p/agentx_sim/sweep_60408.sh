#!/bin/bash
# The calibrated sweep for the AgentX chunking study: the whole 256k AgentX corpus (393 sessions, 68,266 requests),
# every chunk design with its silicon calibration, 32 and 64 sessions, think cap 60 s (heavy) and 1800 s (light).
#   sweep_60408.sh <traces.jsonl> <out.jsonl>     (~1 min on 24 cores)
set -e
H=$(dirname "$(readlink -f "$0")")
T=${1:?traces.jsonl}
O=${2:?out.jsonl}
S=(python3 "$H/agentx_sim.py" --traces "$T" --json "$O" --concurrency 32,64 --max-think-s 60,1800)
"${S[@]}" -j 24 --config 8192x8192:fcfs,4096x4096:fcfs,2048x2048:fcfs,2048x8192:fcfs,2048x8192:pack
"${S[@]}" -j 8 --config 8192x8192:fcfs --cache 16e6,4e6
"${S[@]}" -j 4 --config 8192x8192:fcfs --align exact
