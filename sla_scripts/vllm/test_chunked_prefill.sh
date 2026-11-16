#!/bin/bash
# TP32 chunked-prefill check on one WH Galaxy: correctness first, then the SLA load cases and MMLU.
# Needs tt-metal branch teja/tp32-chunked-prefill in ~/teja/tt-metal.
# usage: test_chunked_prefill.sh [budget]   (max-num-batched-tokens, default 2048)
BUDGET=${1:-2048}
TAG=chunk${BUDGET}
R=~/teja/runs; S=$R/${TAG}_suite.log
log() { echo "$(date -u +%H:%M) $*" >> $S; }

tt-smi -glx_reset_auto > $R/reset_${TAG}.log 2>&1
tmux new-session -d -s chunk "LLAMA_GALAXY_CHUNKED_PREFILL=1 ASYNC= bash ~/teja/run_vllm_tp32.sh $TAG \
  --no-async-scheduling --enable-chunked-prefill --max-num-batched-tokens $BUDGET --max-model-len 36864 \
  --enable-prompt-tokens-details --override-generation-config '{\"top_k\": 32}'"
log "wait for server"
until curl -sf localhost:8000/v1/models > /dev/null; do sleep 30; done
grep -m1 "Chunked prefill enabled" $R/vllm_server_${TAG}.log >> $S || { log "chunked prefill NOT enabled, stop"; exit 1; }

cd ~/teja/vllm-tt-plugin && source ~/teja/tt-metal/python_env/bin/activate
python -m pytest tests/tt/test_chunked_prefill.py --tt-server-url http://localhost:8000 --tt-model-name llama70b \
  --tt-max-num-seqs 32 --tt-chunked-prefill-budget $BUDGET -q > $R/${TAG}_plugin_test.log 2>&1
log "plugin chunked test exit $?"
python3 ~/teja/garbage_probe.py http://localhost:8000/v1/completions llama70b > $R/${TAG}_garbage_probe.log 2>&1
log "garbage probe exit $?"

OUT=$R/bench_$TAG HIT_TAG=${TAG}_rpm60_40pct bash ~/teja/bench_tis.sh
log "bench done"
NC=8 TMO=600 bash ~/teja/run_mmlu.sh $TAG
log "mmlu: $(grep 'MMLU EXIT' $R/mmlu_$TAG.log)"

pkill -f server_example_tt.py; sleep 20; tmux kill-session -t chunk 2>/dev/null
tt-smi -glx_reset_auto >> $R/reset_${TAG}.log 2>&1
log "DONE"
