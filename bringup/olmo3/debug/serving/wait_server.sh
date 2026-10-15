#!/bin/bash
# wait until the vLLM server answers, or its newest log reports a failed start
d=/home/ubuntu/ssinghal/olmo3/tt-inference-server/workflow_logs/local_server
sleep 45
while ! curl -s -m 3 localhost:8000/v1/models >/dev/null; do
  f=$d/$(ls -t $d | head -1)
  if grep -q "Engine core initialization failed" $f; then echo "server failed to start"; exit 1; fi
  sleep 10
done
echo "server ready"
