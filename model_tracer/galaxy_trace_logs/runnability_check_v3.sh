#!/bin/bash
# Re-run only the BROKEN entries from v2 with class-1 env fixes:
#   - HF_HOME and HF_TOKEN from ~/ml_cache/hf/
#   - deepseek path redirects to /data/deepseek
#   - gpt-oss-120b unit re-run in isolation (suspected device pollution previously)
#
# sentence_bert (transformers downgrade) and llama3.3-70b-galaxy (weights gap)
# are NOT retested here — they need user input.

set -u

cd /data/stevenlee/tt-metal
source python_env/bin/activate
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export ARCH_NAME=wormhole_b0

# --- class-1 env fixes ---
export HF_HOME=/home/stevenlee/ml_cache/hf
export HF_TOKEN=$(cat /home/stevenlee/ml_cache/hf/token)
export HF_HUB_TOKEN="$HF_TOKEN"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
echo "HF_HOME=$HF_HOME, HF_TOKEN=${HF_TOKEN:0:6}…"

WALLCLOCK=120
LOGDIR=model_tracer/galaxy_trace_logs/runnability_v3
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
        echo "  SUSPECT   (failed in ${dur}s, rc=$rc)"
      fi
      echo "  --- failure tail of $logfile ---"
      grep -E "Error|FAILED|Traceback|TT_FATAL|^E " "$logfile" \
        | grep -v "TT_FATAL: Read unexpected" | head -10 | sed 's/^/    /'
      ;;
  esac
}

# 02 sd35 — gated repo, retry with HF_TOKEN
run_one "02_sd35" \
  'TT_CACHE_HOME=/mnt/MLPerf/huggingface/tt_cache NO_PROMPT=1 TT_MM_THROTTLE_PERF=5 \
   pytest models/tt_dit/tests/models/sd35/test_pipeline_sd35.py -k "4x8cfg1sp0tp1" --timeout=300 -x'

# 03 flux1 — gated repo
run_one "03_flux1" \
  'TT_CACHE_HOME=/mnt/MLPerf/huggingface/tt_cache NO_PROMPT=1 TT_MM_THROTTLE_PERF=5 \
   pytest models/tt_dit/tests/models/flux1/test_pipeline_flux1.py -k "4x8sp0tp1-dev" --timeout=300 -x'

# 07 deepseek_v3 — redirect path env vars to /data/deepseek
run_one "07_deepseek_v3" \
  'DEEPSEEK_V3_HF_MODEL=/data/deepseek/DeepSeek-R1-0528-dequantized-stacked \
   DEEPSEEK_V3_CACHE=/data/deepseek/DeepSeek-R1-0528-Cache/CI \
   MESH_DEVICE=TG \
   pytest models/demos/deepseek_v3/demo/test_demo.py -k "tg_stress or tg_upr8" --timeout=300 -x'

# 10a gpt-oss-120b unit — re-run in isolation (device pollution previously)
run_one "10a_gpt_oss_120b_galaxy_unit" \
  'HF_MODEL=openai/gpt-oss-120b TT_CACHE_PATH=/data/MLPerf/huggingface/tt_cache/openai--gpt-oss-120b/ \
   pytest models/demos/gpt_oss/tests/unit -k "4x8 and decode_high_throughput" --timeout=300 -x'

echo
echo "=========================================="
echo "Done. Detailed logs in $LOGDIR/*.run.log"
