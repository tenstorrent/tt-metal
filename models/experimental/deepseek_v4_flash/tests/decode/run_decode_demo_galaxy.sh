#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# The traced decode ALONE on a galaxy (tests/decode/test_full_model_decode_demo.py -k tp4_32chip: two 1x4 TP4 stages on
# rows 0-1), in two stages so a failure costs minutes, not the first full weight conversion:
#   A: DEEPSEEK_V4_DECODE_LAYERS=4 (layers 0-3 = SWA, SWA, CSA, HCA: every attention kind), 32 new tokens
#   B: only if A passes -- all 43 layers, 64 new tokens (the first run converts every weight into DEEPSEEK_V4_CACHE_DIR)
# See tests/prefill/TTNN_PREFILL_E2E.md for the prerequisites.
#
#   TAG=demo STAGES="A B" ./run_decode_demo_galaxy.sh      # STAGES="A" = the 4-layer check only
#
# Every path is an env override; the defaults are host 41's.
set -u
REPO=${REPO:-$(cd "$(dirname "$0")/../../../../.." && pwd)}
LOGDIR=${E2E_LOG_DIR:-$HOME/sdawle/e2e_runs}
TAG=${TAG:-demo}
STAGES=${STAGES:-A B}
mkdir -p "$LOGDIR"
cd "$REPO"
source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export DEEPSEEK_V4_CACHE_DIR=${DEEPSEEK_V4_CACHE_DIR:-/mnt/tt-data/sdawle/dsv4_sankar_cache}
export DEEPSEEK_V4_DSPARK=0   # the MTP link; the standalone drafter (tt/decode/dspark.py) is hard-wired to the prefetcher
T=models/experimental/deepseek_v4_flash/tests/decode/test_full_model_decode_demo.py

gate() {  # reset, then open the (8,4) TORUS_XY mesh the test opens; reset again until it maps
  for a in 1 2 3 4; do
    echo "[demo] $(date +%T) $1 reset (attempt $a)"
    sudo -n tt-smi -glx_reset_auto > "$LOGDIR/$1.reset.log" 2>&1
    echo "[demo] reset rc=$? devs=$(ls /dev/tenstorrent | wc -l)"
    if timeout 180 python - > "$LOGDIR/$1.open$a.log" 2>&1 <<'EOF'
import time, ttnn
t0 = time.time()
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D_TORUS_XY)
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4), num_command_queues=2, l1_small_size=1152)
print(f"[open] ok {mesh.get_num_devices()} devices in {time.time()-t0:.1f}s", flush=True)
ttnn.close_mesh_device(mesh)
print("[open] PASS", flush=True)
EOF
    then
      grep -q "\[open\] PASS" "$LOGDIR/$1.open$a.log" && { echo "[demo] $1 mesh open gate PASS (attempt $a)"; return 0; }
    fi
    echo "[demo] $1 mesh open gate FAILED (attempt $a): $(grep -a -m1 TT_FATAL "$LOGDIR/$1.open$a.log" | cut -c1-160)"
  done
  return 1
}

stage() {  # tag layers new_tokens budget_s
  gate "$1" || { echo "[demo] $1 ABORT: mesh never mapped"; return 1; }
  if [ "$2" = all ]; then unset DEEPSEEK_V4_DECODE_LAYERS; else export DEEPSEEK_V4_DECODE_LAYERS=$2; fi
  export DEEPSEEK_V4_MAX_NEW_TOKENS=$3
  echo "[demo] $(date +%T) $1 pytest layers=$2 new=$3 -> $LOGDIR/$1.log"
  timeout "$4" python -m pytest -s -x -p no:cacheprovider "$T" -k tp4_32chip > "$LOGDIR/$1.log" 2>&1
  rc=$?
  echo "[demo] $(date +%T) $1 END rc=$rc"
  grep -a -E "GENERATED|decode throughput|prefetcher OFF|passed|failed|TT_FATAL|TT_THROW|Error" "$LOGDIR/$1.log" \
    | tail -8 | cut -c1-300
  return $rc
}

for s in $STAGES; do
  case $s in
    A) stage "${TAG}A" 4 32 3600 || break ;;
    B) stage "${TAG}B" all 64 16000 || break ;;
  esac
done
echo "[demo] $(date +%T) ALL DONE"
