#!/bin/bash
# 20x outer loop. Each iteration: tt-smi -glx_reset && pytest with 200 inner iterations.
# Per-run log: /workspace/tt-metal/stress_x20_iter200/log_<NN>
# pytest runs in foreground (no timeout) — wrapper blocks on each run for the user to debug if hang.

set -u
ENV_VARS='NORM_OUTPUT_DIR=/tmp/NORM_MAIN TT_DS_PREFILL_HOST_REF_CACHE=/DeepSeek-R1-0528-Cache/DeepSeek-R1-0528-prefill-golden-32a3e8b6 TT_DS_PREFILL_TTNN_CACHE=/DeepSeek-R1-0528-Cache/DeepSeek-R1-0528-Cache-prefill-fresh HF_HOME=/huggingface/hub'
KFILTER='pretrained and e256_device_fp32 and mesh-8x4 and 61_layers and regular and right_pad and smoke and iter200 and not iter2000 and 25600_cf8 and pie960'
DIR=/workspace/tt-metal/stress_x20_iter200
mkdir -p "$DIR"

for i in $(seq 1 20); do
  N=$(printf "%02d" "$i")
  LOG=$DIR/log_$N
  echo ""
  echo "############################################################"
  echo "###  Run $N / 20  (200 inner iter)  @ $(date)"
  echo "###  log: $LOG"
  echo "############################################################"

  source /workspace/tt-metal/python_env/bin/activate
  tt-smi -glx_reset 2>&1 | tail -3

  cd /workspace/tt-metal
  bash -c "$ENV_VARS pytest -vs models/demos/deepseek_v3_d_p/tests/test_prefill_transformer.py -k \"$KFILTER\" |& tee $LOG; echo TEST_DONE_EXIT=\$?"

  pkill -9 -f pytest 2>/dev/null || true
  pkill -9 -f test_prefill 2>/dev/null || true
  sleep 2
done

echo ""
echo "############################################################"
echo "###  ALL 20 DONE @ $(date)"
echo "############################################################"
