#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# Sourced by the common launcher; runs all 36 layers and both 1K cache slots.

[ "${CONFIG}" = sc1 ] || { echo "GPT-OSS 1K acceptance supports sc1 only" >&2; exit 2; }
[ "${PREFILL_PRODUCER_NUM_USERS:-2}" = 2 ] || { echo "GPT-OSS acceptance verifies both slots" >&2; exit 2; }
export PIPELINE_DIR="${PREFILL_SUMMARIES}/gptoss120b_prefill_runner_kv"
MANIFEST="${TT_METAL_HOME}/models/demos/gpt_oss_d_p/tt/runners/manifests/gpt_oss_120b_1k.json"
MGD="${TT_METAL_HOME}/models/demos/gpt_oss_d_p/tt/runners/gptoss_sc1_mgd.textproto"
CHUNK_SIZE=1024
SC1_MAX_SEQ_LEN=1024
SC1_NUM_LAYERS=36
SC1_NUM_USERS=2
PRODUCER_USERS=2
WARMUP_CHUNKS=0
PCC_THRESHOLD="${PREFILL_STANDALONE_CHUNKED_PCC:-0.85}"
PROBE_CHUNKS=0
REQUIRE_CLEAN_SHUTDOWN=1
RUNNER_SHUTDOWN_TIMEOUT_S=120

# Validate worker-visible references and the fixed shape before opening the mesh.
printf -v TRACE_VALIDATION_CMD '%q ' env \
    "PREFILL_STANDALONE_CHUNKED_PCC=${PCC_THRESHOLD}" \
    "PREFILL_NUM_USERS=${PREFILL_NUM_USERS:-2}" \
    "PREFILL_NUM_LAYERS=${PREFILL_NUM_LAYERS:-36}" \
    "GPT_OSS_BOUNDED_SLIDING_KV=${GPT_OSS_BOUNDED_SLIDING_KV:-0}" \
    python3 -c '
import sys
from models.demos.gpt_oss_d_p.tt.runners.acceptance import prefill_runner_scenario, validate_prefill_slot_traces
validate_prefill_slot_traces(sys.argv[1], prefill_runner_scenario())
print(sys.argv[1])
' "${PREFILL_PRODUCER_SLOT_TRACES:-${PREFILL_TRACE_DIR:-}}"
MPIRUN=$(command -v mpirun-ulfm || command -v mpirun)
PREFILL_PRODUCER_SLOT_TRACES=$("${MPIRUN}" --bind-to none --pernode --allow-run-as-root \
    --hostfile "${TTRUN_DIR:-/etc/ttop}/hostfile" \
    -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH bash -lc "${TRACE_VALIDATION_CMD}")
export PREFILL_PRODUCER_SLOT_TRACES

printf -v SLOT_TRACES '%q' "${PREFILL_PRODUCER_SLOT_TRACES}"
printf -v CHECKPOINT '%q' "${PREFILL_HF_MODEL:?Set PREFILL_HF_MODEL to the compatible GPT-OSS checkpoint}"
printf -v WEIGHT_CACHE '%q' "${PREFILL_TTNN_CACHE:-}"
printf -v CACHE_ONLY '%q' "${GPT_OSS_WEIGHTS_FROM_CACHE:-0}"
RUNNER_ENV="export PREFILL_LAYER_ACK_D2H=0; export PREFILL_USE_TRACE=0; \
    export PREFILL_KV_ONLY_LAST_LAYER=0; export GPT_OSS_BOUNDED_SLIDING_KV=0; \
    export PREFILL_HF_MODEL=${CHECKPOINT}; export HF_MODEL=${CHECKPOINT}; \
    export PREFILL_TTNN_CACHE=${WEIGHT_CACHE}; export GPT_OSS_WEIGHTS_FROM_CACHE=${CACHE_ONLY};"
PRODUCER_ENV="export PREFILL_PRODUCER_MANIFEST='${MANIFEST}'; \
    export PREFILL_PRODUCER_SLOT_TRACES=${SLOT_TRACES}; \
    export PREFILL_PRODUCER_MAX_REQUESTS=2; export PREFILL_PRODUCER_DURATION_S=inf; \
    export PREFILL_PRODUCER_INTERLEAVE=round_robin; export PREFILL_PRODUCER_MULTI_TURN_PROB=0; \
    export PREFILL_PRODUCER_P_GAP=0; export PREFILL_PRODUCER_P_BURST=0;"
