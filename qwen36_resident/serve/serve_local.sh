#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# vllm serve the resident Qwen3.6/3.8-27B on one 1 x 4 Blackhole mesh (QuietBox 2: 2x P300c).
#
# usage: HF_MODEL=<checkpoint snapshot dir> qwen36_resident/serve/serve_local.sh plain|mtp|multi [extra vllm serve args...]
#   plain: one user, any sampling; greedy and top_k <= 32 are drawn on device (full speed), others on the host.
#   mtp:   one user, greedy only (a request with temperature > 0 is served greedily; the server warns once), MTP
#          speculative decode (4 tokens per step) through the plugin's adaptive block output.
#   multi: MAX_NUM_SEQS users (default 8) on the rows-mode build (launches of 8, paged KV pool), chunked prefill of
#          PREFILL_CHUNK_TOKENS (default 1024) between decode steps. Needs vllm-tt-plugin branch resident-mu (TT chunk
#          core). Host sampling unless SAMPLE_ON_DEVICE=1 (needs the rows-mode hub feed, WP7).
#
# env: HF_MODEL (required; a local checkpoint dir, or a repo id the offline HF cache resolves), PORT (8000),
#   SERVED_MODEL_NAME (Qwen/Qwen3.8-27B), MAX_MODEL_LEN (32768), VLLM_VENV (a venv with vLLM 0.26 + vllm-tt-plugin),
#   VLLM_TT_PLUGIN_SRC (put a plugin source tree first on PYTHONPATH), TT_VISIBLE_DEVICES (0,1,2,3);
#   multi: MAX_NUM_SEQS, PREFILL_CHUNK_TOKENS, SAMPLE_ON_DEVICE, RESIDENT_MAX_TOKENS_ALL_USERS (KV pool tokens, default
#   MAX_MODEL_LEN per user).
set -euo pipefail
PROFILE=${1:?usage: serve_local.sh plain|mtp|multi [vllm serve args...]}
shift
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${HF_MODEL:?set HF_MODEL to the checkpoint snapshot dir}"

export TT_METAL_HOME=$WT
export PYTHONPATH=${VLLM_TT_PLUGIN_SRC:+$VLLM_TT_PLUGIN_SRC:}$WT:$WT/ttnn:$WT/tools${PYTHONPATH:+:$PYTHONPATH}
export ARCH_NAME=blackhole
export MESH_DEVICE=P150x4
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-0,1,2,3}
export TT_MESH_GRAPH_DESC_PATH=${TT_MESH_GRAPH_DESC_PATH:-$WT/tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto}
# the pinned host-buffer cache stalls a large host -> device upload for 10-40 min in ioctl_pin_pages
export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0
export HF_MODEL HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
# the adapter bundle: Qwen3_5ForConditionalGeneration -> Qwen36ResidentForCausalLM
export EXTRA_MODELS_DIR=$WT/qwen36_resident/vllm_models
# building the model (weights, prefill and decode programs, traces) takes ~6 min before the engine reports ready
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-3600}
# only the plugin's two entry points (another vllm-tt-plugin install in the same venv may register more)
export VLLM_PLUGINS=${VLLM_PLUGINS:-tt,tt_model_registry}
SEQS=1
TT_CONFIG='{"tt": {"trace_region_size": 134217728, "sample_on_device_mode": "decode_only"}}'
case $PROFILE in
  plain) ;;
  mtp) export TT_MODEL_CLASS_OVERRIDES=TTQwen3_5ForConditionalGeneration=qwen36_resident.generator_vllm:Qwen36ResidentMTPForCausalLM ;;
  multi)
    SEQS=${MAX_NUM_SEQS:-8}
    SAMPLING=''
    [ "${SAMPLE_ON_DEVICE:-0}" = 1 ] && SAMPLING=', "sample_on_device_mode": "decode_only"'
    TT_CONFIG="{\"tt\": {\"trace_region_size\": 134217728, \"prefill_chunk_tokens\": ${PREFILL_CHUNK_TOKENS:-1024}$SAMPLING}}"
    ;;
  *) echo "unknown profile $PROFILE (plain|mtp|multi)" >&2; exit 2 ;;
esac

VLLM=${VLLM_VENV:+$VLLM_VENV/bin/}vllm
# kernel sources resolve from the working directory first
cd "$WT"
# log which plugin, ttnn and adapter this server runs
"${VLLM_VENV:+$VLLM_VENV/bin/}python" -c "import vllm_tt_plugin, ttnn, qwen36_resident
for m in (vllm_tt_plugin, ttnn, qwen36_resident): print('serving', m.__name__, 'from', m.__file__)" 2>&1 | grep '^serving'
exec "$VLLM" serve "$HF_MODEL" \
  --served-model-name "${SERVED_MODEL_NAME:-Qwen/Qwen3.8-27B}" \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --max-num-seqs "$SEQS" \
  --block-size 64 \
  --additional-config "$TT_CONFIG" \
  --async-scheduling \
  --port "${PORT:-8000}" \
  "$@"
