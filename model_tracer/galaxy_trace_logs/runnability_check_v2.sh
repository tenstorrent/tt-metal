#!/bin/bash
# Stronger runnability check: actually start each test and see whether it
# fails fast (broken) or is still running at the wallclock cap (runnable).
#
# Verdict logic:
#   exit 124 (timeout cmd killed it)   -> RUNNABLE   (test body was active)
#   exit 0  (passed within window)     -> RUNNABLE   (and even completed)
#   exit !=0 with duration < 30s       -> BROKEN     (fail-fast)
#   exit !=0 with duration >= 30s      -> SUSPECT    (real device error)

set -u

cd /data/stevenlee/tt-metal
source python_env/bin/activate
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export ARCH_NAME=wormhole_b0

WALLCLOCK=120   # seconds per test

# Install model-specific requirements once up front (gpt_oss has its own deps)
if [[ -f models/demos/gpt_oss/requirements.txt ]]; then
  echo "==== installing gpt_oss requirements ===="
  uv pip install -r models/demos/gpt_oss/requirements.txt 2>&1 | tail -5
fi
LOGDIR=model_tracer/galaxy_trace_logs/runnability_v2
mkdir -p "$LOGDIR"

run_one() {
  local label="$1"
  local cmd="$2"
  local logfile="$LOGDIR/${label}.run.log"

  echo "=========================================="
  echo "[$label]  (max ${WALLCLOCK}s)"
  local start_ts end_ts dur rc
  start_ts=$(date +%s)
  timeout --signal=TERM --kill-after=10s "${WALLCLOCK}s" \
    bash -c "$cmd" >"$logfile" 2>&1
  rc=$?
  end_ts=$(date +%s)
  dur=$((end_ts - start_ts))

  case $rc in
    0)
      echo "  RUNNABLE  (passed in ${dur}s)" ;;
    124|137|143)
      echo "  RUNNABLE  (still running at ${dur}s timeout, rc=$rc)" ;;
    *)
      if [[ $dur -lt 30 ]]; then
        echo "  BROKEN    (failed in ${dur}s, rc=$rc)"
      else
        echo "  SUSPECT   (failed in ${dur}s, rc=$rc — could be runner/device, inspect log)"
      fi
      echo "  --- failure tail of $logfile ---"
      tail -25 "$logfile" | grep -E "Error|FAILED|Traceback|TT_FATAL|TT_THROW|KeyError|ImportError|assert|RuntimeError|^E " | head -15 | sed 's/^/    /'
      ;;
  esac
}

# pytest --timeout below is per-test internal; outer `timeout` provides the wallclock cap

# 1
run_one "01_sentence_bert" \
  'pytest models/demos/tg/sentence_bert/tests/test_sentence_bert_e2e_performant.py --timeout=300 -x'

# 2
run_one "02_sd35" \
  'TT_CACHE_HOME=/mnt/MLPerf/huggingface/tt_cache NO_PROMPT=1 TT_MM_THROTTLE_PERF=5 \
   pytest models/tt_dit/tests/models/sd35/test_pipeline_sd35.py -k "4x8cfg1sp0tp1" --timeout=300 -x'

# 3
run_one "03_flux1" \
  'TT_CACHE_HOME=/mnt/MLPerf/huggingface/tt_cache NO_PROMPT=1 TT_MM_THROTTLE_PERF=5 \
   pytest models/tt_dit/tests/models/flux1/test_pipeline_flux1.py -k "4x8sp0tp1-dev" --timeout=300 -x'

# 4
run_one "04_motif" \
  'TT_CACHE_HOME=/mnt/MLPerf/huggingface/tt_cache NO_PROMPT=1 \
   pytest models/tt_dit/tests/models/motif/test_pipeline_motif.py -k "4x8cfg1sp0tp1" --timeout=300 -x'

# 5
run_one "05_wan22" \
  'TT_DIT_CACHE_DIR=/tmp/TT_DIT_CACHE NO_PROMPT=1 \
   pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan.py -k "wh_4x8sp1tp0 and resolution_720p" --timeout=300 -x'

# 6
run_one "06_mochi" \
  'TT_DIT_CACHE_DIR=/tmp/TT_DIT_CACHE NO_PROMPT=1 \
   pytest models/tt_dit/tests/models/mochi/test_pipeline_mochi.py -k "wh_4x8sp1tp0" --timeout=300 -x'

# 7
run_one "07_deepseek_v3" \
  'DEEPSEEK_V3_HF_MODEL=/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-dequantized-stacked \
   DEEPSEEK_V3_CACHE=/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/CI \
   MESH_DEVICE=TG \
   pytest models/demos/deepseek_v3/demo/test_demo.py -k "tg_stress or tg_upr8" --timeout=300 -x'

# 8
run_one "08_qwenimage" \
  'TT_DIT_CACHE_DIR=/tmp/TT_DIT_CACHE NO_PROMPT=1 \
   pytest models/tt_dit/tests/models/qwenimage/test_pipeline_qwenimage.py -k "4x8" --timeout=300 -x'

# 9a — pick one fast unit test for collection
run_one "09a_qwen3_32b_galaxy_unit" \
  'HF_MODEL=Qwen/Qwen3-32B TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/Qwen/Qwen3-32B \
   pytest models/demos/llama3_70b_galaxy/tests/unit_tests/test_qwen_mlp.py --timeout=300 -x'

# 9b
run_one "09b_qwen3_32b_galaxy_e2e" \
  'HF_MODEL=Qwen/Qwen3-32B TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/Qwen/Qwen3-32B \
   pytest models/demos/llama3_70b_galaxy/tests/test_qwen_accuracy.py --timeout=300 -x'

# 10a
run_one "10a_gpt_oss_120b_galaxy_unit" \
  'HF_MODEL=openai/gpt-oss-120b TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/openai--gpt-oss-120b/ \
   pytest models/demos/gpt_oss/tests/unit -k "4x8 and decode_high_throughput" --timeout=300 -x'

# 10b
run_one "10b_gpt_oss_120b_galaxy_e2e" \
  'HF_MODEL=openai/gpt-oss-120b TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/openai--gpt-oss-120b/ \
   pytest models/demos/gpt_oss/demo/text_demo.py -k "mesh_4x8 and batch128" --timeout=300 -x'

# 11
run_one "11_llama33_70b_galaxy_e2e" \
  'TT_CACHE_HOME=/mnt/MLPerf/huggingface/tt_cache \
   LLAMA_DIR=/mnt/MLPerf/tt_dnn-models/llama/Llama3.3-70B-Instruct/ \
   FAKE_DEVICE=TG \
   pytest models/demos/llama3_70b_galaxy/demo/text_demo.py -k "pcc-80L" --timeout=300 -x'

echo
echo "=========================================="
echo "Done. Detailed logs in $LOGDIR/*.run.log"
