#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# vllm serve the resident Qwen3.6/3.8-27B on one 1 x 4 Blackhole mesh (QuietBox 2: 2x P300c).
#
# usage: HF_MODEL=<checkpoint snapshot dir> qwen36_resident/serve/serve_local.sh plain|mtp|dflash2|multi|multi-long [args]
#   plain: one user, any sampling; greedy and top_k <= 32 are drawn on device (full speed), others on the host.
#   mtp:   one user, greedy only (a request with temperature > 0 is served greedily; the server warns once), MTP
#          speculative decode (4 tokens per step) through the plugin's adaptive block output.
#   dflash2: one user, greedy only (as mtp), DFlash2 speculative decode (8 tokens per step; lane dfl's
#          qwen36_resident.dflash.ResidentDFlash) through the plugin's adaptive block output.
#   multi: MAX_NUM_SEQS users (default 16) on the rows-mode build (launches of 8, paged KV pool), chunked prefill of
#          PREFILL_CHUNK_TOKENS (default 1024) between decode steps. Needs vllm-tt-plugin branch resident-mu (TT chunk
#          core). Host sampling unless SAMPLE_ON_DEVICE=1 (needs the rows-mode hub feed, WP7).
#   multi-long: multi without the 128 PP chunk (RESIDENT_PREFILL_CHUNKS 256,512,1024) and with a 29,440-block pool
#          (RESIDENT_MAX_TOKENS_ALL_USERS 1,884,160; the plugin adds one block per slot, so the engine logs 29,456):
#          7 users at 256K, 8 at 230K, 16 at 115K; TTFT +49 ms at 128-token prompts.
#
# env: HF_MODEL (required; a local checkpoint dir, or a repo id the offline HF cache resolves), PORT (8000),
#   SERVED_MODEL_NAME (Qwen/Qwen3.8-27B), MAX_MODEL_LEN (32768), VLLM_VENV (a venv with vLLM 0.26 + vllm-tt-plugin),
#   VLLM_TT_PLUGIN_SRC (put a plugin source tree first on PYTHONPATH), TT_VISIBLE_DEVICES (0,1,2,3);
#   multi: MAX_NUM_SEQS, PREFILL_CHUNK_TOKENS, SAMPLE_ON_DEVICE, RESIDENT_MAX_TOKENS_ALL_USERS (KV pool tokens, default
#   1,540,096 = 24,064 blocks), RESIDENT_ATTN_WORKERS / RESIDENT_HEAD_GROUPS (default 28 / 4 for this profile).
set -euo pipefail
PROFILE=${1:?usage: serve_local.sh plain|mtp|dflash2|multi|multi-long [vllm serve args...]}
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
  dflash2) export TT_MODEL_CLASS_OVERRIDES=TTQwen3_5ForConditionalGeneration=qwen36_resident.generator_vllm:Qwen36ResidentDFlashForCausalLM ;;
  multi | multi-long)
    if [ "$PROFILE" = multi-long ]; then
      # long contexts: no 128 PP chunk (its small-matmul weight copies, 3.21 GB/chip) buys a 29,440-block pool of
      # 64 positions (TTFT +49 ms at 128-token prompts, +20 ms at 512, none from 2K; accuracy-gated). The plugin adds
      # max_num_seqs blocks of output reservation (16 -> 29,456 blocks), and 30,016 + 16 OOMs at build: launch 1's
      # 288 MiB GDN state finds 37.06 MB/bank free of the 37.75 MB it needs; 29,440 passed the card churn (queue_wp8_r3)
      export RESIDENT_PREFILL_CHUNKS=${RESIDENT_PREFILL_CHUNKS:-256,512,1024}
      export RESIDENT_MAX_TOKENS_ALL_USERS=${RESIDENT_MAX_TOKENS_ALL_USERS:-1921024}
    fi
    SEQS=${MAX_NUM_SEQS:-16}  # the shipped multi profiles: two launches
    # the rows build's attention / GDN core split: 28 attention workers, 4 GDN head groups (accuracy-gated, -6 to -9 %
    # step time at <= 32K); the single-user profiles keep the defaults (32 workers)
    export RESIDENT_ATTN_WORKERS=${RESIDENT_ATTN_WORKERS:-28} RESIDENT_HEAD_GROUPS=${RESIDENT_HEAD_GROUPS:-4}
    # the shipped KV pool: 24,064 blocks of 64 positions (16 slots leave ~0.66 GB/chip; 24,640 fails at build)
    export RESIDENT_MAX_TOKENS_ALL_USERS=${RESIDENT_MAX_TOKENS_ALL_USERS:-1540096}
    SAMPLING=''
    [ "${SAMPLE_ON_DEVICE:-0}" = 1 ] && SAMPLING=', "sample_on_device_mode": "decode_only"'
    TT_CONFIG="{\"tt\": {\"trace_region_size\": 134217728, \"prefill_chunk_tokens\": ${PREFILL_CHUNK_TOKENS:-1024}$SAMPLING}}"
    ;;
  *) echo "unknown profile $PROFILE (plain|mtp|dflash2|multi|multi-long)" >&2; exit 2 ;;
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
