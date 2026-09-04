#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# P3 serving gate: prefill_runner serves chunks pushed by prefill_producer, and the producer reads
# the KV back device-lessly over UMD through the published address table and PCCs it against the
# golden trace.
#
#   HF_MODEL=/path/to/Meta-Llama-3.1-8B-Instruct ./run_serving_pcc.sh
#
# Both processes are local; no inference server and no decode side are involved. The runner runs with
# PREFILL_MOCK_MIGRATION=1 so it publishes the KV chunk table + device map without needing the
# external migration endpoint (that is Gate 2, tracked separately).
#
# The two processes MUST agree on PREFILL_{SP,TP,CHUNK_SIZE,NUM_USERS,MAX_SEQ_LEN,H2D_SERVICE_ID} or
# the byte layout on the socket disagrees; the manifest is the single place they are set.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
PY="${PYTHON:-/opt/venv/bin/python}"
PKG=models/demos/llama3_1_8b_d_p

: "${HF_MODEL:?set HF_MODEL to a Llama-3.1-8B-Instruct checkpoint directory}"
SEQ_LEN="${SEQ_LEN:-2048}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
TRACE_DIR="${PREFILL_TRACE_DIR:-/tmp/llama31_8b_trace_${SEQ_LEN}}"
OUT="${OUT_DIR:-$HERE/../.bringup_runs}"
mkdir -p "$OUT"
cd "$ROOT"

# 1. Golden trace (CPU, cached on disk — regenerate only if absent).
if [ ! -f "$TRACE_DIR/metadata.json" ]; then
    echo "=== generating golden trace at $TRACE_DIR (CPU reference, slow) ==="
    "$PY" -m models.demos.llama3_1_8b_d_p.scripts.generate_golden_trace \
        --out "$TRACE_DIR" --seq-len "$SEQ_LEN" || exit 1
else
    echo "=== reusing golden trace at $TRACE_DIR ==="
fi

export PREFILL_MANIFEST="$PKG/tt/runners/manifests/llama3_1_8b.json"
export PREFILL_MODEL=llama3_1_8b_d_p
# PREFILL_NUM_LAYERS must be EXPORTED, not just set in the manifest: the runner applies the manifest
# at import, but the producer only honours a --manifest CLI argument and reads NUM_LAYERS from the
# environment at import time. Left unset it defaults to 61 (DeepSeek's depth) on the producer side,
# which then waits for 61*chunks layer acks that a 32-layer model never sends, and gives up after
# 600s with "timed out at 128/244 acks".
export PREFILL_NUM_LAYERS=32
export PREFILL_HF_MODEL="$HF_MODEL"
export PREFILL_TRACE_DIR="$TRACE_DIR"
export PREFILL_SP=8 PREFILL_TP=4
export PREFILL_CHUNK_SIZE="$CHUNK_SIZE"
export PREFILL_NUM_USERS="${PREFILL_NUM_USERS:-1}"
export PREFILL_MAX_SEQ_LEN="$SEQ_LEN"
export PREFILL_H2D_SERVICE_ID="${PREFILL_H2D_SERVICE_ID:-llama31_8b_prefill}"
export PREFILL_TOPOLOGY=linear

# 2. Runner: publishes the H2D service descriptor, the KV chunk table and the device map.
# PREFILL_ENABLE_LAYER_ACK=1 is REQUIRED with PREFILL_PRODUCER_CHECK_PCC=1: without it the producer's
# device-less UMD read races the runner's prefill (an H2D push returning does not mean the layers are
# done), and the producer refuses to run rather than report a meaningless PCC.
echo "=== starting runner (log: $OUT/p3_runner.log) ==="
PREFILL_MOCK_MIGRATION=1 PREFILL_ENABLE_LAYER_ACK=1 "$PY" -m models.demos.common.prefill.runners.prefill_runner \
    > "$OUT/p3_runner.log" 2>&1 &
RUNNER_PID=$!
trap 'kill -9 $RUNNER_PID 2>/dev/null' EXIT

# Wait for the runner to advertise its service before the producer tries to connect.
for _ in $(seq 1 "${READY_TIMEOUT_S:-1800}"); do
    grep -qE "descriptor service_id=|request .* loop start|WORKER_READY" "$OUT/p3_runner.log" 2>/dev/null && break
    kill -0 $RUNNER_PID 2>/dev/null || { echo "runner died early:"; tail -30 "$OUT/p3_runner.log"; exit 1; }
    sleep 1
done

# 3. Producer: pushes chunks, then reads the KV back over UMD and PCCs it against the trace.
echo "=== starting producer (log: $OUT/p3_producer.log) ==="
PREFILL_PRODUCER_CHECK_PCC=1 \
PREFILL_PRODUCER_CHUNKS="${PREFILL_PRODUCER_CHUNKS:-$((SEQ_LEN / CHUNK_SIZE))}" \
    "$PY" -m models.demos.common.prefill.runners.prefill_producer \
    > "$OUT/p3_producer.log" 2>&1
RC=$?
echo "producer exit=$RC"
grep -E "PCC|slot [0-9]" "$OUT/p3_producer.log" | tail -20
exit $RC
