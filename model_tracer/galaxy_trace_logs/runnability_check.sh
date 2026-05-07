#!/bin/bash
# Runnability check for each wormhole galaxy nightly model.
# For each entry:
#   1) pytest --collect-only with the exact -k filter and env  -> catches import/config errors
#   2) Print PASS/FAIL with a short reason
#
# Does NOT run the full test (no weight loading, no device-perf workloads).
# A "FAIL" here means the test cannot even be collected; "PASS" means it imports
# and at least one parametrization matches the -k filter.

set -uo pipefail

cd /data/stevenlee/tt-metal
source python_env/bin/activate
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export ARCH_NAME=wormhole_b0

LOGDIR=model_tracer/galaxy_trace_logs/runnability
mkdir -p "$LOGDIR"

run_collect() {
  local label="$1"
  local cmd="$2"
  local logfile="$LOGDIR/${label}.collect.log"

  echo "=========================================="
  echo "[$label]"
  echo "  cmd: $cmd"
  bash -c "$cmd --collect-only -q" >"$logfile" 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    local n
    n=$(grep -cE "::test_" "$logfile" || true)
    echo "  result: PASS  ($n parametrizations collected)"
  else
    echo "  result: FAIL  (exit=$rc)"
    echo "  --- last 15 lines of $logfile ---"
    tail -15 "$logfile" | sed 's/^/    /'
  fi
}

############### 1. sentence_bert (Galaxy) ###############
run_collect "01_sentence_bert" \
  'pytest models/demos/tg/sentence_bert/tests/test_sentence_bert_e2e_performant.py'

############### 2. sd35 (Galaxy) ###############
run_collect "02_sd35" \
  'TT_CACHE_HOME=/mnt/MLPerf/huggingface/tt_cache NO_PROMPT=1 TT_MM_THROTTLE_PERF=5 \
   pytest models/tt_dit/tests/models/sd35/test_pipeline_sd35.py -k "4x8cfg1sp0tp1"'

############### 3. flux1 (Galaxy) ###############
run_collect "03_flux1" \
  'TT_CACHE_HOME=/mnt/MLPerf/huggingface/tt_cache NO_PROMPT=1 TT_MM_THROTTLE_PERF=5 \
   pytest models/tt_dit/tests/models/flux1/test_pipeline_flux1.py -k "4x8sp0tp1-dev"'

############### 4. motif (Galaxy) ###############
run_collect "04_motif" \
  'TT_CACHE_HOME=/mnt/MLPerf/huggingface/tt_cache NO_PROMPT=1 \
   pytest models/tt_dit/tests/models/motif/test_pipeline_motif.py -k "4x8cfg1sp0tp1"'

############### 5. wan22 (Galaxy) ###############
run_collect "05_wan22" \
  'TT_DIT_CACHE_DIR=/tmp/TT_DIT_CACHE NO_PROMPT=1 \
   pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan.py -k "wh_4x8sp1tp0 and resolution_720p"'

############### 6. mochi (Galaxy) ###############
run_collect "06_mochi" \
  'TT_DIT_CACHE_DIR=/tmp/TT_DIT_CACHE NO_PROMPT=1 \
   pytest models/tt_dit/tests/models/mochi/test_pipeline_mochi.py -k "wh_4x8sp1tp0"'

############### 7. deepseek_v3 (Galaxy) ###############
run_collect "07_deepseek_v3" \
  'DEEPSEEK_V3_HF_MODEL=/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-dequantized-stacked \
   DEEPSEEK_V3_CACHE=/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/CI \
   MESH_DEVICE=TG \
   pytest models/demos/deepseek_v3/demo/test_demo.py -k "tg_stress or tg_upr8"'

############### 8. qwenimage (Galaxy) ###############
run_collect "08_qwenimage" \
  'TT_DIT_CACHE_DIR=/tmp/TT_DIT_CACHE NO_PROMPT=1 \
   pytest models/tt_dit/tests/models/qwenimage/test_pipeline_qwenimage.py -k "4x8"'

############### 9a. qwen3-32b-galaxy unit ###############
run_collect "09a_qwen3_32b_galaxy_unit" \
  'HF_MODEL=Qwen/Qwen3-32B TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/Qwen/Qwen3-32B \
   pytest models/demos/llama3_70b_galaxy/tests/unit_tests/test_qwen_mlp.py \
          models/demos/llama3_70b_galaxy/tests/unit_tests/test_qwen_mlp_prefill.py \
          models/demos/llama3_70b_galaxy/tests/unit_tests/test_qwen_attention.py \
          models/demos/llama3_70b_galaxy/tests/unit_tests/test_qwen_attention_prefill.py \
          models/demos/llama3_70b_galaxy/tests/unit_tests/test_qwen_decoder.py \
          models/demos/llama3_70b_galaxy/tests/unit_tests/test_qwen_decoder_prefill.py'

############### 9b. qwen3-32b-galaxy e2e ###############
run_collect "09b_qwen3_32b_galaxy_e2e" \
  'HF_MODEL=Qwen/Qwen3-32B TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/Qwen/Qwen3-32B \
   pytest models/demos/llama3_70b_galaxy/tests/test_qwen_accuracy.py'

############### 10a. gpt-oss-120b-galaxy unit (high batch) ###############
run_collect "10a_gpt_oss_120b_galaxy_unit" \
  'HF_MODEL=openai/gpt-oss-120b TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/openai--gpt-oss-120b/ \
   pytest models/demos/gpt_oss/tests/unit -k "4x8 and decode_high_throughput"'

############### 10b. gpt-oss-120b-galaxy e2e ###############
run_collect "10b_gpt_oss_120b_galaxy_e2e" \
  'HF_MODEL=openai/gpt-oss-120b TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/openai--gpt-oss-120b/ \
   pytest models/demos/gpt_oss/demo/text_demo.py -k "mesh_4x8 and batch128"'

############### 11. llama3.3-70b-galaxy e2e ###############
run_collect "11_llama33_70b_galaxy_e2e" \
  'TT_CACHE_HOME=/mnt/MLPerf/huggingface/tt_cache \
   LLAMA_DIR=/mnt/MLPerf/tt_dnn-models/llama/Llama3.3-70B-Instruct/ \
   FAKE_DEVICE=TG \
   pytest models/demos/llama3_70b_galaxy/demo/text_demo.py -k "pcc-80L"'

echo
echo "=========================================="
echo "Done. Detailed logs in $LOGDIR/*.collect.log"
