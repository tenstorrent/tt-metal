#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# Sourced by the common launcher; only the Llama acceptance settings live here.

[ "${CONFIG}" = sc1 ] || { echo "Llama prefill currently supports sc1 only" >&2; exit 2; }
export PIPELINE_DIR="${PREFILL_SUMMARIES}/llama31_prefill_runner_kv"
MANIFEST="${TT_METAL_HOME}/models/demos/llama_3p1_8b_d_p/tt/runners/manifests/llama_3p1_8b.json"
MGD="${MGD_DIR}/llama31_sc1_mgd.textproto"
CHUNK_SIZE=$(manifest_env PREFILL_CHUNK_SIZE)
GOLDEN_LEN="${PREFILL_MAX_SEQ_LEN:-$(manifest_env PREFILL_MAX_SEQ_LEN)}"
SC1_MAX_SEQ_LEN=${GOLDEN_LEN}
SC1_NUM_LAYERS=$(manifest_env PREFILL_NUM_LAYERS)
SC1_NUM_USERS=$(manifest_env PREFILL_NUM_USERS)
PRODUCER_USERS=${SC1_NUM_USERS}
WARMUP_CHUNKS=0
PCC_THRESHOLD=0.99
PROBE_CHUNKS="0,$((GOLDEN_LEN / CHUNK_SIZE - 1))"

# Validate references before reserving the mesh. This emits no shell code.
python3 - <<'PYTHON'
import os
from models.demos.llama_3p1_8b_d_p.tests.utils import prefill_runner_scenario, validate_prefill_slot_traces
validate_prefill_slot_traces(os.environ.get("PREFILL_PRODUCER_SLOT_TRACES", ""), prefill_runner_scenario())
PYTHON

printf -v SLOT_TRACES '%q' "${PREFILL_PRODUCER_SLOT_TRACES}"
printf -v CHECKPOINT '%q' "${PREFILL_HF_MODEL:-/mnt/models/meta-llama/Llama-3.1-8B-Instruct}"
RUNNER_ENV="export PREFILL_LAYER_ACK_D2H=0; export PREFILL_USE_TRACE=0; \
    export PREFILL_KV_ONLY_LAST_LAYER=0; export PREFILL_HF_MODEL=${CHECKPOINT};"
PRODUCER_ENV="export PREFILL_PRODUCER_MANIFEST='${MANIFEST}'; \
    export PREFILL_PRODUCER_SLOT_TRACES=${SLOT_TRACES}; \
    export PREFILL_PRODUCER_MAX_REQUESTS=${PRODUCER_USERS}; \
    export PREFILL_PRODUCER_DURATION_S=inf; export PREFILL_PRODUCER_INTERLEAVE=round_robin; \
    export PREFILL_PRODUCER_MULTI_TURN_PROB=0; export PREFILL_PRODUCER_P_GAP=0; \
    export PREFILL_PRODUCER_P_BURST=0;"
