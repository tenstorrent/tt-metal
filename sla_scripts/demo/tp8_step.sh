#!/bin/bash
# usage: tp8_step.sh <tag>   -- greedy DP4 x 8 users/group (32 users), records perf + outputs
TAG="$1"
K="performance and batch-32 and not log-probs" bash ~/teja/run_tp8dp4.sh "step_${TAG}" --batch_size 8 \
  --sampling_params "$(python3 -c 'import json;print(json.dumps({"temperature":[0.0]*32,"top_p":[0.08]*32,"top_k":[1]*32}))')"
python3 ~/teja/tp8_record.py "$TAG"
