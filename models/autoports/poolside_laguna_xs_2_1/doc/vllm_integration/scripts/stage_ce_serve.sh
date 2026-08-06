#!/bin/bash
# Stage C+E — boot the TOOL-CALLING server (stays up; agents + concurrency probe run against it).
# Canonical smoke_test.md §0 launch + native tool-calling (glm47) + reasoning split (deepseek_r1) + APC on
# (agentic loops get ~95% prefix-cache hit). Detached so a reaped shell can't kill it.
set +e
LOCAL=/home/ttuser/.local/lib/model-bringup/tt-metal
BASE=/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1
SLOG=$BASE/doc/vllm_integration/stage_ce/serve.log
mkdir -p "$(dirname "$SLOG")"
# free any stale holders (never self-match pkill -f) + dead API/port remnants
for p in $(lsof -t /dev/tenstorrent/* 2>/dev/null | sort -u; lsof -t -i:8000 2>/dev/null; pgrep -f "run_vllm_server|api_server" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
tt-smi -r all >/dev/null 2>&1; sleep 8
# CRITICAL: truncate the readiness log — the runner aborts if it finds a STALE 'EngineDeadError'/fatal
# marker from a prior crash in this file (it is not truncated on boot).
: > "$BASE/readiness_vllm/server.log" 2>/dev/null || true
nohup env \
  TT_METAL_HOME=$LOCAL \
  PYTHONPATH=/home/ttuser/dev/tt-metal:$LOCAL/vllm:$LOCAL/vllm/plugins/vllm-tt-plugin/src \
  TT_LAGUNA_PIPE_CHUNK=2048 TT_LAGUNA_PREFIX_CACHE=1 \
  bash -c "cd /tmp && stdbuf -oL -eL /home/ttuser/.tenstorrent-venv/bin/python -u \
     -m models.common.readiness_check.run_vllm_server \
     --model-dir $BASE --hf-model poolside/Laguna-XS-2.1 --mesh-device P150x4 --stages serve \
     --max-num-seqs 32 --block-size 64 --max-model-len 262144 \
     --tt-config '{\"trace_region_size\": 1500000000, \"fabric_config\": \"FABRIC_1D_RING\", \"env_passthrough\": [\"VLLM_*\", \"MESH_DEVICE\", \"TT_LAGUNA_*\", \"TT_METAL_*\", \"PYTHONPATH\"]}' \
     --additional-server-args='--trust-remote-code --max-num-batched-tokens 131072 --enable-prefix-caching --reasoning-parser deepseek_r1 --enable-auto-tool-choice --tool-call-parser glm47'" \
  > "$SLOG" 2>&1 &
echo $! > /tmp/laguna_srv_pgid_ce
echo "server launching (pgid $(cat /tmp/laguna_srv_pgid_ce)); tail: $SLOG and $BASE/readiness_vllm/server.log"
# wait for health
for i in $(seq 1 360); do sleep 5
  curl -sf -m3 http://localhost:8000/health >/dev/null 2>&1 && { echo "HEALTHY ~$((i*5))s"; break; }
done
curl -s http://localhost:8000/v1/models | python3 -c 'import sys,json;print("model:",json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || echo "NOT READY"
