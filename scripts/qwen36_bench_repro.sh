#!/usr/bin/env bash
# Reproduce the benchmark engine HANG (dispatch fetch-queue timeout) on the SHARED
# generator path — ONE model build, NO reset between requests, NO HTTP/vLLM.
# Runs an ISL schedule (default 12x ISL-128 then ISL-4096) = the benchmark's 3rd-combo
# trigger. Decisive: if it hangs at the first big request -> generator/model path bug
# (iterate the fix here); if it runs clean -> bug is vLLM/chat-endpoint path.
#
#   scripts/qwen36_bench_repro.sh                       # default schedule
#   QWEN36_ISL_SCHEDULE="128,128,128,4096" \
#   QWEN36_SCHED_DECODE_STEPS=32 scripts/qwen36_bench_repro.sh
#
# Log: /tmp/qwen36_bench_repro.log
set -u
cd /home/tt-admin/ssinghal/qwen36/new/tt-metal
source python_env/bin/activate 2>/dev/null
SMI=/home/tt-admin/.tenstorrent-venv/bin/tt-smi
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) ARCH_NAME=wormhole_b0 MESH_DEVICE=BH-Galaxy
TEST=models/demos/qwen3_6_galaxy_v2/demo/text_demo_qwen36.py::test_qwen36_isl_schedule_repro
LOG=/tmp/qwen36_bench_repro.log

# NOTE: `tt-smi -r` on this box can trigger a PCIe AER storm that crash-reboots the
# HOST (observed 2026-06-07). Default OFF — run on freshly-booted/clean device state.
# Set QWEN36_REPRO_RESET=1 to restore the double-reset (only on a CPLD-fixed board).
for attempt in 1 2 3; do
  echo "==== attempt $attempt $(date) ====" | tee "$LOG"
  if [ "${QWEN36_REPRO_RESET:-0}" = "1" ]; then
    $SMI -r >/dev/null 2>&1; sleep 12; $SMI -r >/dev/null 2>&1; sleep 12
  fi
  python -m pytest -q -s "$TEST" >>"$LOG" 2>&1
  rc=$?
  if grep -q "topology_mapper.cpp:527\|Timed out while waiting for active ethernet" "$LOG"; then
    echo "attempt=$attempt: fabric flake at mesh-open, retrying"; continue
  fi
  break
done

echo "==== exit=$rc ===="
echo "--- last request markers / crash ---"
grep -E "\[sched\]|CRASH|hang detected|fetch queue|TT_THROW|TT_FATAL|num_blocks_x|COMPLETED CLEAN" "$LOG" | tail -40
