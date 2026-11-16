#!/bin/bash
# reduce-scatter (1x8 path) link/worker sweep; baseline = links 1, workers 2
run() { RS_LINKS=$1 RS_WORKERS=$2 bash ~/teja/tp8_step.sh "rs_l$1_w$2"; tt-smi -r >/dev/null 2>&1 || true; }
run 2 2
run 4 2
run 4 1
run 4 3
run 4 4
echo RSDONE
