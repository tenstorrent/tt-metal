#!/bin/bash
# TP32 SLA suite on tt-metal gtobar/prefill-levers + plugin gtobar/prefill-arrival-wait.
# usage: run_prefill_levers_suite.sh A|B|C
#   A: model prefill levers, our server settings (async off)
#   B: A + async scheduling + plugin prefill flush, admission cap, arrival wait
#   C: A + async scheduling only
# bfp4 weights and HiFi2 SDPA stay off: they change precision.
CFG=${1:-A}
TAG=gt${CFG}
R=~/teja/runs; S=$R/${TAG}_suite.log
log() { echo "$(date -u +%H:%M) $*" >> $S; }

export LLAMA_GALAXY_PREFILL_PAD=1024 LLAMA_GALAXY_WO_AG_MAX=2048 LLAMA_GALAXY_PREFILL_SLAB1024=64
export LLAMA_GALAXY_PREFILL_SEQSPLIT=1 LLAMA_GALAXY_PREFILL_SDPA_Q64_2048=1 LLAMA_GALAXY_PREFILL_QKV_SEQSPLIT=1
export LLAMA_GALAXY_PREFILL_MINMM=qkv,mlp2048,wo2048,mlp3072 LLAMA_GALAXY_PREFILL_CCL_BUFS=16
export LLAMA_GALAXY_PREFILL_CPF_RING=1 LLAMA_GALAXY_PREFIX_SKIP=ring LLAMA_GALAXY_PREFILL_CPF_QKV_SEQSPLIT=1
export LLAMA_GALAXY_PREFILL_CPF_REPLICA=1

TT='"dispatch_core_axis": "col", "sample_on_device_mode": "all", "fabric_config": "FABRIC_1D_RING", "worker_l1_size": 1344544, "trace_region_size": 640000000'
if [ "$CFG" = B ]; then
  export TT_PLUGIN_EXP_ARRIVAL_WAIT_MS=2
  SCHED="--async-scheduling"
  TT="$TT, \"flush_prefill_output\": true, \"max_prefill_requests_per_step\": 1"
elif [ "$CFG" = C ]; then
  SCHED="--async-scheduling"
else
  SCHED="--no-async-scheduling"
fi

cd ~/teja/tt-metal && source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export HF_MODEL=/home/ubuntu/teja/weights/Llama-3.3-70B-Instruct TT_CACHE_PATH=/home/ubuntu/teja/cache/llama70b
export MESH_DEVICE=TG TT_LLAMA_TEXT_VER=llama3_70b_galaxy VLLM_RPC_TIMEOUT=900000
LOG=$R/vllm_server_${TAG}.log

log "reset"; tt-smi -glx_reset_auto > $R/reset_${TAG}.log 2>&1
log "start server $SCHED"; env | grep -E "^(LLAMA_GALAXY|TT_PLUGIN)" | sort >> $S
cd ~/teja/vllm-tt-plugin
python examples/server_example_tt.py --model "$HF_MODEL" --served-model-name llama70b \
  --data_parallel_size 4 --max_num_seqs 8 $SCHED --port 8000 --max-model-len 36864 \
  --enable-prompt-tokens-details --override-generation-config '{"top_k": 32}' \
  --additional-config "{\"tt\": {$TT}}" > $LOG 2>&1 &
SRV=$!
until curl -sf localhost:8000/v1/models > /dev/null; do
  kill -0 $SRV 2>/dev/null || { log "server died"; exit 1; }
  sleep 30
done
log "server up"

python3 ~/teja/needle_probe.py > $R/${TAG}_needle.log 2>&1; log "needle exit $? $(tail -1 $R/${TAG}_needle.log)"
python3 ~/teja/garbage_probe.py http://localhost:8000/v1/completions llama70b > $R/${TAG}_garbage_probe.log 2>&1
log "garbage probe exit $?"

OUT=$R/bench_$TAG HIT_TAG=${TAG}_rpm60_40pct bash ~/teja/bench_tis.sh > $R/${TAG}_bench.log 2>&1
log "bench done"
NC=8 TMO=600 bash ~/teja/run_mmlu.sh $TAG
log "mmlu: $(grep 'MMLU EXIT' $R/mmlu_$TAG.log)"

kill $SRV; sleep 20; pkill -f server_example_tt.py
tt-smi -glx_reset_auto >> $R/reset_${TAG}.log 2>&1
log "DONE"
