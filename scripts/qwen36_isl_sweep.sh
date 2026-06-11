#!/usr/bin/env bash
# Qwen3.6-27B BH-Galaxy ISL sweep — inline demo (prefill-CCL + trace), coherent path.
# Runs each ISL, retrying on the Galaxy fabric flake (topology_mapper.cpp:527).
# Per-ISL JSON lands at /tmp/qwen36_demo_isl_<ISL>.json; logs at /tmp/qwen36_sweep_isl_<ISL>.log
set -u
cd /home/tt-admin/ssinghal/qwen36/new/tt-metal
source python_env/bin/activate 2>/dev/null
SMI=/home/tt-admin/.tenstorrent-venv/bin/tt-smi
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) ARCH_NAME=wormhole_b0 MESH_DEVICE=BH-Galaxy
export QWEN36_SEQ_CORES_PER_HEAD=4 QWEN36_DECODE_L1_RESIDUAL=1 QWEN36_FORCE_SWITCH_DECODE=1
export QWEN36_LM_HEAD_PLAIN_DECODE=1 QWEN36_FULLATTN_WO_TUNED=1 QWEN36_DELTA_OP_TUNED=1
export QWEN36_CCL_NUM_LINKS_DELTA=1 QWEN36_DECODE_STEPS=32

ISLS="128 4096 8192 16384 32768 65536 131072 262144"
SUMMARY=/tmp/qwen36_isl_sweep_summary.txt
echo "ISL sweep started $(date)" > "$SUMMARY"

for ISL in $ISLS; do
  LOG=/tmp/qwen36_sweep_isl_${ISL}.log
  ok=0
  for attempt in 1 2 3; do
    $SMI -r >/dev/null 2>&1; sleep 12; $SMI -r >/dev/null 2>&1; sleep 12
    QWEN36_PERF_T_PREFILL=$ISL python -m pytest -q -s \
      models/demos/qwen3_6_galaxy_v2/demo/text_demo_qwen36.py::test_qwen36_demo_batch1 \
      > "$LOG" 2>&1
    if grep -q "topology_mapper.cpp:527" "$LOG"; then
      echo "ISL=$ISL attempt=$attempt: fabric flake, retrying" | tee -a "$SUMMARY"
      continue
    fi
    if grep -q "1 passed" "$LOG"; then ok=1; fi
    break
  done
  TPUT=$(grep "decode throughput / user" "$LOG" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
  TTFT=$(grep "TTFT (warm" "$LOG" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
  GEN=$(grep "GENERATED" "$LOG" | tail -1 | cut -c1-90)
  echo "ISL=$ISL ok=$ok decode_tok_s=${TPUT:-NA} ttft_warm_ms=${TTFT:-NA} | ${GEN:-no-output}" | tee -a "$SUMMARY"
done
echo "ISL sweep done $(date)" | tee -a "$SUMMARY"
