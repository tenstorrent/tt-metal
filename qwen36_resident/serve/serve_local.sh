#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# vllm serve the resident Qwen3.6/3.8-27B on one 1 x 4 Blackhole mesh (QuietBox 2: 2x P300c).
#
# usage: HF_MODEL=<checkpoint snapshot dir> qwen36_resident/serve/serve_local.sh plain|mtp|dflash2|multi|multi-long|multi-ttft
#   [args]
#   plain: one user, any sampling; greedy and top_k <= 32 are drawn on device (full speed), others on the host.
#   mtp:   one user, greedy only (a request with temperature > 0 is served greedily; the server warns once), MTP
#          speculative decode (4 tokens per step) through the plugin's adaptive block output.
#   dflash2: one user, greedy only (as mtp), DFlash2 speculative decode (8 tokens per step; lane dfl's
#          qwen36_resident.dflash.ResidentDFlash) through the plugin's adaptive block output.
#   multi: MAX_NUM_SEQS users (default 16) on the rows-mode build (launches of 8, paged KV pool). Prefill gangs: the
#          waiting prompts prefill whole, as one step, before their decodes start (a burst is one wave); a prompt over
#          PREFILL_GANG_TOKENS (2048) arriving while others decode is split into PREFILL_CHUNK_TOKENS (1024) spans
#          between decode steps. Needs vllm-tt-plugin branch resident-mu-sched (TT chunk core + gangs). Host sampling
#          unless SAMPLE_ON_DEVICE=1 (needs the rows-mode hub feed, WP7).
#          No 128 PP chunk (RESIDENT_PREFILL_CHUNKS 256,512,1024) and a 29,440-block pool (RESIDENT_MAX_TOKENS_ALL_USERS
#          1,884,160; the plugin adds one block per slot, so the engine logs 29,456): 7 users at 256K, 8 at 230K, 16 at
#          115K; TTFT +49 ms at 128-token prompts.
#   multi-ttft: multi with the interleaved prefill policy instead of prefill gangs (= multi with PREFILL_GANG_TOKENS=0,
#          the package's TTFT-first option): a long prompt is split into PREFILL_CHUNK_TOKENS spans between decode
#          steps, so first tokens come sooner and the running users' decode is slower
#   multi-long: the same as multi (its sizing became multi's, 2026-11-02)
#
# env: HF_MODEL (required; a local checkpoint dir, or a repo id the offline HF cache resolves), PORT (8000),
#   SERVED_MODEL_NAME (Qwen/Qwen3.8-27B), MAX_MODEL_LEN (32768), VLLM_VENV (a venv with vLLM 0.26 + vllm-tt-plugin),
#   VLLM_TT_PLUGIN_SRC (put a plugin source tree first on PYTHONPATH), TT_VISIBLE_DEVICES (0,1,2,3);
#   multi: MAX_NUM_SEQS, PREFILL_CHUNK_TOKENS, PREFILL_GANG_TOKENS / PREFILL_GANG_IDLE_TOKENS (prefill gangs, default
#   2048 / 0; PREFILL_GANG_TOKENS=0 = off),
#   SAMPLE_ON_DEVICE, RESIDENT_MAX_TOKENS_ALL_USERS (KV pool tokens, default
#   1,540,096 = 24,064 blocks), RESIDENT_ATTN_WORKERS / RESIDENT_HEAD_GROUPS (default 28 / 4 for this profile).
set -euo pipefail
PROFILE=${1:?usage: serve_local.sh plain|mtp|dflash2|multi|multi-long|multi-ttft [vllm serve args...]}
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
  multi | multi-long | multi-ttft)
    # The package's multi-user profile (user decision 2026-11-02: the former multi-long sizing for every multi
    # profile; multi-long stays as an alias). No 128 PP chunk (its small-matmul weight copies, 3.21 GB/chip) buys a
    # 29,440-block pool of 64 positions (TTFT +49 ms at 128-token prompts, +20 ms at 512, none from 2K;
    # accuracy-gated). The plugin adds max_num_seqs blocks of output reservation (16 -> 29,456 blocks), and 30,016 + 16
    # OOMs at build: launch 1's 288 MiB GDN state finds 37.06 MB/bank free of the 37.75 MB it needs; 29,440 passed the
    # card churn (queue_wp8_r3) and, with PPPrefill.run_batch's staging, a 250K-token prompt (lane sched rb1).
    # The former multi sizing: RESIDENT_PREFILL_CHUNKS=128,256,512,1024 RESIDENT_MAX_TOKENS_ALL_USERS=1540096.
    export RESIDENT_PREFILL_CHUNKS=${RESIDENT_PREFILL_CHUNKS:-256,512,1024}
    export RESIDENT_MAX_TOKENS_ALL_USERS=${RESIDENT_MAX_TOKENS_ALL_USERS:-1884160}
    SEQS=${MAX_NUM_SEQS:-16}  # the shipped multi profiles: two launches
    # the rows build's attention / GDN core split: 28 attention workers, 4 GDN head groups (accuracy-gated, -6 to -9 %
    # step time at <= 32K); the single-user profiles keep the defaults (32 workers)
    export RESIDENT_ATTN_WORKERS=${RESIDENT_ATTN_WORKERS:-28} RESIDENT_HEAD_GROUPS=${RESIDENT_HEAD_GROUPS:-4}
    SAMPLING=''
    [ "${SAMPLE_ON_DEVICE:-0}" = 1 ] && SAMPLING=', "sample_on_device_mode": "decode_only"'
    # prefill gangs (plugin resident-mu-sched): waiting prompts run whole, several per prefill step, capped at
    # PREFILL_GANG_TOKENS while others decode and PREFILL_GANG_IDLE_TOKENS (0: the step's budget) while none do.
    # Default 2048 / 0 (lane sched, 2026-10-30): a burst prefills as one wave before its decodes start (c=16 TPOT at
    # the decode step, TTFT ~3x the interleaved policy's); a prompt arriving while others decode stalls them about as
    # the interleave did. PREFILL_GANG_TOKENS=0: the interleaved policy (one long prompt split between decode steps)
    # A step's whole prompts of one PP chunk stream through one pipelined PP pass (PPPrefill.run_batch, fill / drain
    # once per step: c=16 TTFT -16..-31 %, output tok/s +14..+20 %, lane sched rb1); RESIDENT_PP_BATCH=0 turns it off
    GANG=''
    [ "$PROFILE" = multi-ttft ] && : "${PREFILL_GANG_TOKENS:=0}"
    [ "${PREFILL_GANG_TOKENS:=2048}" != 0 ] && GANG=", \"prefill_gang_tokens\": $PREFILL_GANG_TOKENS"
    [ "${PREFILL_GANG_IDLE_TOKENS:-0}" != 0 ] && GANG="$GANG, \"prefill_gang_idle_tokens\": $PREFILL_GANG_IDLE_TOKENS"
    TT_CONFIG="{\"tt\": {\"trace_region_size\": 134217728, \"prefill_chunk_tokens\": ${PREFILL_CHUNK_TOKENS:-1024}$SAMPLING$GANG}}"
    ;;
  *) echo "unknown profile $PROFILE (plain|mtp|dflash2|multi|multi-long|multi-ttft)" >&2; exit 2 ;;
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
