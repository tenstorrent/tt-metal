#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Launch Molmo2-8B OpenAI-compatible servers on the 1x8 slices ("trays") of a 32-chip Galaxy, one
# vllm-tt-plugin server per slice, ports PORT_BASE .. PORT_BASE+N-1 (default 8000..8003).
#
#   ./launch_4x_galaxy_t3k.sh                 # slices 0 1 2 3 -> ports 8000 8001 8002 8003
#   SLICES="1 2 3" ./launch_4x_galaxy_t3k.sh  # only some slices (port = PORT_BASE + slice)
#   ./launch_4x_galaxy_t3k.sh stop            # kill the servers
#
# Prerequisites (see models/demos/molmo2/README.md "Server"):
#   * tt-metal built, TT_METAL_HOME set (default: repo root); `tt-smi -r && tt-smi -glx_reset` done.
#   * A python env (VLLM_ENV) = a copy of tt-metal's python_env with vLLM 0.30.0 (VLLM_TARGET_DEVICE=empty)
#     and vllm-tt-plugin installed, with vllm-tt-plugin-molmo2-multimodal.patch applied to the plugin
#     (lets the model declare video + extra multimodal fields; other models are unaffected).
#   * HF weights for allenai/Molmo2-8B in the local HF cache (or HF_HOME).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export TT_METAL_HOME="${TT_METAL_HOME:-$REPO_ROOT}"
VLLM_ENV="${VLLM_ENV:?set VLLM_ENV to the python venv with vllm 0.30.0 + vllm-tt-plugin}"
SLICES="${SLICES:-0 1 2 3}"
PORT_BASE="${PORT_BASE:-8000}"
STAGGER_S="${STAGGER_S:-30}"          # stagger starts so tt-metal init phases don't collide
LOG_DIR="${LOG_DIR:-/tmp/molmo2_servers}"
MODEL_DIR="${MODEL_DIR:-$LOG_DIR/molmo2_ckpt}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-36864}"   # model card: max_position_embeddings
mkdir -p "$LOG_DIR"

if [[ "${1:-}" == "stop" ]]; then
    pkill -f "vllm.entrypoints.openai.api_server --model $MODEL_DIR" || true
    exit 0
fi

# transformers 5 cannot hash HF-cache snapshot symlinks for remote code: serve from a directory with real
# copies of the small code/config files and symlinks only for the safetensors.
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    SNAP="$(ls -d "${HF_HOME:-$HOME/.cache/huggingface}"/hub/models--allenai--Molmo2-8B/snapshots/* | head -1)"
    mkdir -p "$MODEL_DIR"
    for f in "$SNAP"/*; do
        b="$(basename "$f")"
        if [[ "$b" == *.safetensors ]]; then ln -sf "$(readlink -f "$f")" "$MODEL_DIR/$b"; else cp -L "$f" "$MODEL_DIR/$b"; fi
    done
fi

for sl in $SLICES; do
    port=$((PORT_BASE + sl))
    devices="$(seq -s, $((sl * 8)) $((sl * 8 + 7)))"
    echo "slice $sl: devices $devices -> port $port (log $LOG_DIR/server_$sl.log)"
    (
        export VIRTUAL_ENV="$VLLM_ENV" PATH="$VLLM_ENV/bin:$PATH" VLLM_USE_V1=1
        export PYTHONPATH="$TT_METAL_HOME" ARCH_NAME=wormhole_b0 MESH_DEVICE=T3K HF_HUB_OFFLINE=1
        export TT_VISIBLE_DEVICES="$devices"
        # One Inspector RPC port per server so `tools/tt-triage.py` can be pointed at the right one:
        #   TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS=127.0.0.1:$((INSPECTOR_PORT_BASE + slice)) python tools/tt-triage.py
        export TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS="127.0.0.1:$((${INSPECTOR_PORT_BASE:-50050} + sl))"
        export TT_MESH_GRAPH_DESC_PATH="$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto"
        export TT_MODEL_CLASS_OVERRIDES="TTMolmo2ForConditionalGeneration=models.demos.molmo2.tt.generator_vllm:Molmo2ForConditionalGeneration"
        cd "$TT_METAL_HOME"
        nohup python -m vllm.entrypoints.openai.api_server --model "$MODEL_DIR" --served-model-name allenai/Molmo2-8B \
            --port "$port" --max-model-len "$MAX_MODEL_LEN" --max-num-seqs 1 --block-size 64 --trust-remote-code \
            --allowed-local-media-path "${ALLOWED_MEDIA_PATH:-/}" \
            --additional-config '{"tt": {"fabric_config": "FABRIC_1D"}}' \
            > "$LOG_DIR/server_$sl.log" 2>&1 &
    )
    sleep "$STAGGER_S"
done
echo "waiting for servers..."
for sl in $SLICES; do
    port=$((PORT_BASE + sl))
    until curl -sf "http://localhost:$port/v1/models" > /dev/null; do sleep 5; done
    echo "port $port ready"
done
