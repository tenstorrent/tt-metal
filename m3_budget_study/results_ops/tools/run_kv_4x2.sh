#!/bin/bash
# One profile_4x2.py KV run (no tracy) under a watchdog: RUN_TIMEOUT s total, STALL_TIMEOUT s without log output.
#   RUN_ID=kv_4x2_L0-6 PROFILE_MESH=4x2 PROFILE_KV_DUMP=... tools/run_kv_4x2.sh
# Resets the galaxy first (env -u TT_VISIBLE_DEVICES), refuses to start unless .lock names owner=vmelnykov-ops-agent
# or while another M3 harness runs. Log: logs/$RUN_ID.log, env: logs/$RUN_ID.env.
set -uo pipefail
RES="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$RES/../.." && pwd)}"
: "${RUN_ID:?set RUN_ID}"
RUN_TIMEOUT="${RUN_TIMEOUT:-1200}" STALL_TIMEOUT="${STALL_TIMEOUT:-300}"
LOG="$RES/logs/$RUN_ID.log" ENVF="$RES/logs/$RUN_ID.env"
grep -q "owner=vmelnykov-ops-agent" "$RES/.lock" || { echo "lock not ours"; exit 2; }
busy="$(pgrep -u "$(id -u)" -af 'minimax_m3/tests/perf/[a-z_]+\.py|tools/profile_4x2\.py|tracy-capture' | grep -v "^$$ ")"
[ -z "$busy" ] || { echo "another harness is running: $busy"; exit 2; }
cd "$TT_METAL_HOME"; source python_env/bin/activate
export PYTHONPATH="$TT_METAL_HOME" LOGURU_LEVEL=INFO
export HF_MODEL="${HF_MODEL:-/mnt/weka/model-weights/llm/minimax/MiniMax-M3}"
# CFG_* / CFG_DESC override the transport (e.g. the 4x4 sub-torus: pass TT_VISIBLE_DEVICES and PROFILE_PARENT_MESH too).
export TT_MESH_GRAPH_DESC_PATH="${CFG_DESC:-$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto}"
export TT_CACHE_PATH="${TT_CACHE_PATH:-/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill}"
export M3_FABRIC="${CFG_FABRIC:-1d}" EXPERT_DTYPE=bf4 M3_PROFILE_ZONES=0 TT_METAL_DEVICE_PROFILER=0
export M3_MOE_W_NDSHARD=1 M3_MOE_HYBRID_THRESHOLD=128 M3_MOE_DISPATCH="${CFG_DISPATCH:-v1}" M3_MOE_COMBINE="${CFG_COMBINE:-v1}"
export M3_MOE_TOPOLOGY="${CFG_MOE_TOPOLOGY:-linear}" M3_CCL_TOPOLOGY="${CFG_CCL_TOPOLOGY:-linear}"
ulimit -Su "$(ulimit -Hu)"
{ echo "run_id=$RUN_ID"; echo "git_sha=$(git rev-parse HEAD)"; echo "dirty=$(git status --porcelain -uno | tr '\n' ' ')"
  echo "date=$(date -Is)"; env | grep -E '^(M3_|PROFILE_|TT_|EXPERT_|HF_|PREFILL_)' | sort; } > "$ENVF"
env -u TT_VISIBLE_DEVICES tt-smi -glx_reset > "$RES/logs/$RUN_ID.reset" 2>&1 || { echo "reset failed"; exit 3; }
t0=$(date +%s)
setsid python3 "$RES/tools/profile_4x2.py" "$@" > "$LOG" 2>&1 &
pid=$! status=""
while kill -0 "$pid" 2>/dev/null; do
  sleep 15; now=$(date +%s); age=$(( now - $(stat -c %Y "$LOG") ))
  [ $(( now - t0 )) -gt "$RUN_TIMEOUT" ] && { status=TIMEOUT; break; }
  [ "$age" -gt "$STALL_TIMEOUT" ] && { status=HANG; break; }
done
if [ -n "$status" ]; then
  kill -TERM -- "-$pid" 2>/dev/null; sleep 20; kill -KILL -- "-$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  env -u TT_VISIBLE_DEVICES tt-smi -glx_reset >> "$RES/logs/$RUN_ID.reset" 2>&1
else
  wait "$pid"; rc=$?; status=$([ $rc -eq 0 ] && echo OK || echo "ERROR(rc=$rc)")
fi
echo "[run_kv] $RUN_ID status=$status elapsed=$(( $(date +%s) - t0 ))s" | tee -a "$LOG"
