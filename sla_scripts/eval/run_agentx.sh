#!/bin/bash
# agentx (inferencex-agentx-mvp, CI shape: 900 s profile, 32 traces) against the TP32 vLLM server.
# Client = InferenceX 8f12037 vendored aiperf + tt-inference-server patch, flags as in its aiperf_agentic_traces driver.
# usage: run_agentx.sh <max_ctx>   (traces with a recorded peak above max_ctx are dropped)
CTX=${1:-32768}
TAG=agentx_tp32_ctx${CTX}
OUT=~/teja/runs/$TAG; mkdir -p $OUT
pkill -f server_example_tt.py; sleep 20
tmux kill-window -t teja:vllmsrv 2>/dev/null; tmux kill-window -t teja:tp8srv 2>/dev/null
tt-smi -glx_reset_auto > $OUT/reset.log 2>&1
tmux new-window -d -t teja -n vllmsrv "bash ~/teja/run_vllm_tp32.sh $TAG --max-model-len $CTX --override-generation-config '{\"top_k\": 32, \"top_p\": 0.9, \"temperature\": 0.6}'"
until curl -sf localhost:8000/v1/models >/dev/null; do sleep 30; done
source ~/teja/agentx_venv/bin/activate
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
python -m aiperf profile --scenario inferencex-agentx-mvp --url http://localhost:8000 \
  --endpoint /v1/chat/completions --endpoint-type chat --model llama70b \
  --tokenizer /home/ubuntu/teja/weights/Llama-3.3-70B-Instruct \
  --concurrency 8 --benchmark-duration 900 --random-seed 42 --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane 3 --warmup-grace-period 600 --num-dataset-entries 32 --slice-duration 1.0 \
  --max-context-length $CTX --output-artifact-dir $OUT --public-dataset semianalysis_cc_traces_weka_062126 \
  --record-processor-service-count 8 --server-metrics http://localhost:8000/metrics \
  --streaming --use-server-token-count --no-gpu-telemetry --trace-idle-gap-cap-seconds 300 \
  > $OUT/client.log 2>&1
echo "AGENTX EXIT $?" >> $OUT/client.log
