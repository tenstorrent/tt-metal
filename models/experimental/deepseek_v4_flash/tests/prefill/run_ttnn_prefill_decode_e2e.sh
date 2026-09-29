#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# One-galaxy e2e: the ttnn prefill (models/demos/deepseek_v3_d_p/tt/v4) on the rows 4-7 4x4 submesh -> hand-off ->
# the traced decode (tt/model.py) on rows 0-1. See TTNN_PREFILL_E2E.md next to this file for the prerequisites.
#
#   TAG=run4 NEW=64 PLEN=1000 COMPARE=decode ./run_ttnn_prefill_decode_e2e.sh
#
# Every path is an env override; the defaults are host 41's.
set -u
REPO=${REPO:-$(cd "$(dirname "$0")/../../../../.." && pwd)}
LOGDIR=${E2E_LOG_DIR:-$HOME/sdawle/e2e_runs}
TAG=${TAG:-run}
LOG=$LOGDIR/$TAG.log
mkdir -p "$LOGDIR"
cd "$REPO"
source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD

# After a reset a galaxy sometimes leaves an eth link untrained and the TORUS_XY (8,4) mesh cannot map ("Graph specified in
# MGD could not fit"). Gate on a quick open of exactly the mesh the test opens; reset again until it maps. No reset after it.
for a in 1 2 3 4; do
  echo "[e2e] $(date +%T) reset (attempt $a)"
  sudo -n tt-smi -glx_reset_auto > "$LOGDIR/$TAG.reset.log" 2>&1
  echo "[e2e] reset rc=$? devs=$(ls /dev/tenstorrent | wc -l)"
  if timeout 180 python - > "$LOGDIR/$TAG.open$a.log" 2>&1 <<'EOF'
import time, ttnn
t0 = time.time()
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D_TORUS_XY)
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4), num_command_queues=2, l1_small_size=1152)
print(f"[open] ok {mesh.get_num_devices()} devices in {time.time()-t0:.1f}s", flush=True)
ttnn.close_mesh_device(mesh)
print("[open] PASS", flush=True)
EOF
  then
    grep -q "\[open\] PASS" "$LOGDIR/$TAG.open$a.log" && { echo "[e2e] mesh open gate PASS (attempt $a)"; break; }
  fi
  echo "[e2e] mesh open gate FAILED (attempt $a): $(grep -a -m1 TT_FATAL "$LOGDIR/$TAG.open$a.log" | cut -c1-160)"
done

export DEEPSEEK_V4_CACHE_DIR=${DEEPSEEK_V4_CACHE_DIR:-/mnt/tt-data/sdawle/dsv4_sankar_cache}      # decode bf4 tile cache
export PREFILL_TTNN_CACHE=${PREFILL_TTNN_CACHE:-/mnt/tt-data/sdawle/dsv4_flash_ttnn_cache}       # prefill tile cache
export PREFILL_HF_MODEL=${PREFILL_HF_MODEL:-/mnt/tt-data/sdawle/models/DeepSeek-V4-Flash-0731}   # HF checkpoint
export DEEPSEEK_V4_MAX_NEW_TOKENS=${NEW:-64} DEEPSEEK_V4_E2E_PROMPT_LEN=${PLEN:-1000}
[ -n "${COMPARE:-}" ] && export DEEPSEEK_V4_E2E_COMPARE=$COMPARE
echo "[e2e] $(date +%T) pytest -> $LOG"
timeout "${BUDGET:-14000}" python -m pytest -s -x -p no:cacheprovider \
  models/experimental/deepseek_v4_flash/tests/prefill/test_ttnn_prefill_decode_e2e.py > "$LOG" 2>&1
echo "[e2e] $(date +%T) END rc=$?"
grep -a -E "E2E SUMMARY|GENERATED|hand-off|ttnn prefill:|first token|decode:|passed|failed|Error|TT_FATAL|TT_THROW" "$LOG" \
  | tail -20 | cut -c1-300
