#!/bin/bash
# usage: [tuning env vars] tp8_gate.sh <tag>  -- accuracy eval + 128-user greedy run, then PASS/FAIL gate
TAG="$1"
K="performance and ci-token-matching" bash ~/teja/run_tp8dp4.sh "gate_acc_${TAG}"
[ "$TAG" = baseline ] || bash ~/teja/tp8_step128.sh "$TAG"
python3 ~/teja/tp8_gate.py "$TAG"
