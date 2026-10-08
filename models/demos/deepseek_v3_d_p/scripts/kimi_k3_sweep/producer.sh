#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# One request set against the live runner: producer.sh <rundir> <tag> <users> <isl> [prefix]
# Single process, no read-back: throughput comes from the runner's timing CSVs.
set -uo pipefail
D=${1:?rundir}; TAG=${2:?tag}; USERS=${3:?users}; ISL=${4:?isl}; PREFIX=${5:-0}
HOME_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
cd "$HOME_DIR"
source python_env/bin/activate
export TT_METAL_HOME=$HOME_DIR PYTHONPATH=$HOME_DIR:$HOME_DIR/ttnn:$HOME_DIR/tools
export PREFILL_MODEL=kimi_k3 PREFILL_MANIFEST=models/demos/deepseek_v3_d_p/tt/runners/manifests/kimi_k3.json
export PREFILL_H2D_SERVICE_ID=ds_prefill PREFILL_NUM_LAYERS=93 PREFILL_MAX_SEQ_LEN=1049600 PREFILL_SP=8 PREFILL_TP=4
export PREFILL_NUM_USERS=$USERS PREFILL_PRODUCER_MAX_REQUESTS=$USERS PREFILL_PRODUCER_INTERLEAVE=round_robin
export PREFILL_PRODUCER_ISL=$ISL PREFILL_PRODUCER_PREFIX_LEN=$PREFIX PREFILL_PRODUCER_CHUNKS=205
export PREFILL_PRODUCER_CHECK_PCC=0 PREFILL_SEND_SHUTDOWN=0 PREFILL_H2D_CONNECT_TIMEOUT=900
export PREFILL_TRACE_DIR=${PREFILL_TRACE_DIR:-/mnt/weka/model-cache/stable/deepseek-prefill-cache/golden/k3_vllm_code_debug_1M}
exec python3 -m models.demos.common.prefill.runners.prefill_producer > "$D/producer_$TAG.log" 2>&1
