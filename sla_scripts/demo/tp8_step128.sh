#!/bin/bash
# usage: tp8_step128.sh <tag>   -- greedy DP4 x 32 users/group (128 users), compared to b128_baseline
TAG="b128_$1"
K="performance and batch-32 and not log-probs" bash ~/teja/run_tp8dp4.sh "step_${TAG}" --batch_size 32 \
  --sampling_params "$(python3 -c 'import json;n=128;print(json.dumps({"temperature":[0.0]*n,"top_p":[0.08]*n,"top_k":[1]*n,"frequency_penalty":[0.0]*n,"presence_penalty":[0.0]*n,"repetition_penalty":[1.0]*n}))')"
python3 ~/teja/tp8_record.py "$TAG" b128_baseline
