#!/bin/bash
# Galaxy model tracing harness — invoked once per model.
#
# Usage:
#   bash trace_runner.sh <slug>
# where <slug> is one of:
#   motif, wan22, mochi, qwenimage, flux1, deepseek_v3,
#   qwen3-32b-galaxy-unit, qwen3-32b-galaxy-e2e,
#   gpt-oss-120b-galaxy-unit, gpt-oss-120b-galaxy-e2e
#
# Logs: model_tracer/galaxy_trace_logs/trace_<slug>.log
# Each model gets the env overrides discovered via the runnability scan.

set -uo pipefail

cd /data/stevenlee/tt-metal
source python_env/bin/activate
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export ARCH_NAME=wormhole_b0

# Common HF setup (token + cache live in user's ml_cache)
export HF_HOME=/home/stevenlee/ml_cache/hf
export HF_TOKEN=$(cat /home/stevenlee/ml_cache/hf/token)
export HF_HUB_TOKEN="$HF_TOKEN"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

SLUG=${1:?must pass model slug}
LOG=model_tracer/galaxy_trace_logs/trace_${SLUG}.log

trace() {
  local test_path="$1"
  shift
  echo "==========================================" | tee -a "$LOG"
  echo "[$SLUG] tracing: $test_path" | tee -a "$LOG"
  echo "extra args: $*" | tee -a "$LOG"
  echo "==========================================" | tee -a "$LOG"
  python model_tracer/generic_ops_tracer.py "$test_path" "$@" 2>&1 | tee -a "$LOG"
}

case "$SLUG" in
  motif)
    export TT_CACHE_HOME=/data/MLPerf/huggingface/tt_cache
    export NO_PROMPT=1
    trace "models/tt_dit/tests/models/motif/test_pipeline_motif.py" -- -k "4x8cfg1sp0tp1"
    ;;

  wan22)
    export TT_DIT_CACHE_DIR=/tmp/TT_DIT_CACHE
    export NO_PROMPT=1
    trace "models/tt_dit/tests/models/wan2_2/test_pipeline_wan.py" -- -k "wh_4x8sp1tp0 and resolution_720p"
    ;;

  mochi)
    export TT_DIT_CACHE_DIR=/tmp/TT_DIT_CACHE
    export NO_PROMPT=1
    trace "models/tt_dit/tests/models/mochi/test_pipeline_mochi.py" -- -k "wh_4x8sp1tp0"
    ;;

  qwenimage)
    export TT_DIT_CACHE_DIR=/tmp/TT_DIT_CACHE
    export NO_PROMPT=1
    trace "models/tt_dit/tests/models/qwenimage/test_pipeline_qwenimage.py" -- -k "4x8"
    ;;

  flux1)
    export TT_CACHE_HOME=/data/MLPerf/huggingface/tt_cache
    export NO_PROMPT=1
    export TT_MM_THROTTLE_PERF=5
    trace "models/tt_dit/tests/models/flux1/test_pipeline_flux1.py" -- -k "4x8sp0tp1-dev"
    ;;

  deepseek_v3)
    export DEEPSEEK_V3_HF_MODEL=/data/deepseek/DeepSeek-R1-0528-dequantized-stacked
    export DEEPSEEK_V3_CACHE=/data/deepseek/DeepSeek-R1-0528-Cache/CI
    export MESH_DEVICE=TG
    trace "models/demos/deepseek_v3/demo/test_demo.py" -- -k "tg_stress or tg_upr8"
    ;;

  qwen3-32b-galaxy-unit)
    export HF_MODEL=Qwen/Qwen3-32B
    export TT_CACHE_PATH=/data/MLPerf/huggingface/tt_cache/Qwen/Qwen3-32B
    trace "models/demos/llama3_70b_galaxy/tests/unit_tests/" -- \
      -k "test_qwen_mlp or test_qwen_mlp_prefill or test_qwen_attention or test_qwen_attention_prefill or test_qwen_decoder or test_qwen_decoder_prefill"
    ;;

  qwen3-32b-galaxy-e2e)
    export HF_MODEL=Qwen/Qwen3-32B
    export TT_CACHE_PATH=/data/MLPerf/huggingface/tt_cache/Qwen/Qwen3-32B
    trace "models/demos/llama3_70b_galaxy/tests/test_qwen_accuracy.py"
    ;;

  gpt-oss-120b-galaxy-unit)
    uv pip install -r models/demos/gpt_oss/requirements.txt 2>&1 | tail -3 | tee -a "$LOG"
    export HF_MODEL=openai/gpt-oss-120b
    export TT_CACHE_PATH=/data/MLPerf/huggingface/tt_cache/openai--gpt-oss-120b/
    trace "models/demos/gpt_oss/tests/unit" -- -k "4x8 and decode_high_throughput"
    ;;

  gpt-oss-120b-galaxy-e2e)
    uv pip install -r models/demos/gpt_oss/requirements.txt 2>&1 | tail -3 | tee -a "$LOG"
    export HF_MODEL=openai/gpt-oss-120b
    export TT_CACHE_PATH=/data/MLPerf/huggingface/tt_cache/openai--gpt-oss-120b/
    trace "models/demos/gpt_oss/demo/text_demo.py" -- -k "mesh_4x8 and batch128"
    ;;

  *)
    echo "Unknown slug: $SLUG" >&2
    exit 2 ;;
esac
