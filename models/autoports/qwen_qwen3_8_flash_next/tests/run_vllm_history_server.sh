#!/usr/bin/env bash
set -euo pipefail
# Explicit test-only factory; never use this launcher for acceptance/performance.
experiment_dir=${1:?supply a fresh absolute experiment directory}
[[ "$experiment_dir" = /* ]] || { echo "Use an absolute experiment directory" >&2; exit 2; }
[[ ! -e "$experiment_dir" ]] || { echo "Preserve existing experiment evidence" >&2; exit 2; }
mkdir -p "$experiment_dir"
export QWEN_VLLM_HISTORY_DIR="$experiment_dir/recording"
cd /home/ttuser/qwen38-flash/tt-metal
source /home/ttuser/qwen38-flash/vllm-env/bin/activate
source /home/ttuser/qwen38-flash/lane-env.sh
export VLLM_TT_PLUGIN_ROOT=/home/ttuser/qwen38-flash/vllm-tt-plugin
export TT_MODEL_CLASS_OVERRIDES='TTQwen4ExpForConditionalGeneration=models.autoports.qwen_qwen3_8_flash_next.tests.vllm_history_record:RecordedQwenFlashNextForCausalLM'
export TT_VISIBLE_DEVICES=0,1,2,3
export TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto
export QWEN_VLLM_LAYER_INDICES=0,1,3
export TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
export VLLM_SERVER_DEV_MODE=1 OMP_NUM_THREADS=8
unset TT_METAL_DEVICE_PROFILER TT_METAL_WATCHER
exec timeout 3600 python -m readiness_check.run_vllm_server \
 --model-dir models/autoports/qwen_qwen3_8_flash_next \
 --hf-model /home/ttuser/qwen38-flash/hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/812d96e10a0dd82b69d749e68693698cf6c36daa \
 --mesh-device P300x2 --max-num-seqs 2 --max-model-len 1024 --block-size 32 \
 --sampling-profile smoke --stages serve --server-timeout 1200 \
 --tt-config '{"trace_region_size":0,"fabric_config":"FABRIC_1D_RING","trace_mode":"all"}' \
 --additional-server-args "--max-num-batched-tokens 256 --no-enable-prefix-caching --language-model-only --async-scheduling" \
 --output-dir "$experiment_dir/server"
