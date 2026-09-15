#!/usr/bin/env bash
# Mistral Small 4, 36 layers, real PP=4 pipeline over device-to-device fabric sockets:
# 4 ranks x [8,1] column sub-meshes of one Blackhole galaxy, driven by prefill_producer.
#
# This is the measurement the PP_HANDOFF none/host bracket was standing in for. Unlike that bracket
# (one process, 4 submeshes, activation relayed through the host) every hop here is a real ttnn
# MeshSocket transfer over fabric, and every rank is its own process -- which is also why this does
# not need ttnn traces to pipeline: each rank has its own host thread issuing only its 9 layers,
# so eager dispatch is no longer the serialising bottleneck it is in the single-process test.
#
# No KV PCC gate here: prefill_runner rejects PREFILL_MOCK_MIGRATION for num_ranks>1 (each rank would
# publish a table covering only its own layer slice). Correctness was established single-rank; see
# logs/01 and logs/02. This run is functional + throughput.
set -euo pipefail
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$S/env.sh"
cd "$TT_METAL_HOME"
source "$TT_METAL_HOME/python_env/bin/activate"

RUN_TAG="${RUN_TAG:-pp4}"
OUT="$S/logs/${RUN_TAG}"
mkdir -p "$OUT"

export TT_METAL_CACHE="${TT_METAL_CACHE:-/tmp/tt-metal-cache-pp}"
export PREFILL_MANIFEST=$TT_METAL_HOME/models/demos/deepseek_v3_d_p/tt/runners/manifests/mistral4.json
# Per-stage weights: byte-identical to the 32dev/8x1 cache (the device-count component of the cache
# path is namespacing only), hardlinked into the 8dev namespace each rank resolves.
# Weight cache: resolved as {name}_{arch}_{get_num_devices()}dev/{sp}x{tp}. Each PP rank sees 8
# devices -> 8dev/8x1; a single-rank run sees 32 -> 32dev/8x4. Both trees exist on shared /data.
export PREFILL_TTNN_CACHE="${PP_TTNN_CACHE:-${M4_CACHE_8x1}}"
export PREFILL_CHUNK_SIZE="${PP_CHUNK_SIZE:-5120}"
# Runner AND producer must agree on the slot count: the runner asserts 0 <= slot_id < num_users,
# so a producer driving more slots than the runner allocated dies mid-run on slot_id out of range.
export PREFILL_NUM_USERS="${PP_USERS:-2}"
# TTFT mode: PP_KV_ONLY_LAST_LAYER=0 makes the last rank run its final layer in full and build the
# final norm + LM head, so a first token is actually produced. The runner DEFAULTS this to 1 (last
# layer kv-only, no norm/LM head, no token) because for KV-migration throughput the token is waste —
# which means a throughput run measures prefill-completion latency, NOT TTFT. Pair with PP_REQUESTS=1.
export PREFILL_KV_ONLY_LAST_LAYER="${PP_KV_ONLY_LAST_LAYER:-1}"
# Layer count: PREFILL_* is not auto-propagated by ttrun (only TT_/ARCH_/TTNN_/... are), so this
# MUST be in the -x list below or the ranks fall back to the manifest's 36.
export PREFILL_NUM_LAYERS="${PREFILL_NUM_LAYERS:-36}"
export LOGURU_LEVEL=INFO

SERVICE_ID=ds_prefill
DESCRIPTOR=/dev/shm/tt_h2d_stream_service_${SERVICE_ID}.bin
BINDING="${PP_BINDING:-models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_request_intragalaxy_4rank_8x1.yaml}"
# PP_MAX_SEQ_LEN / PP_CHUNKS drive the window. PREFILL_MAX_SEQ_LEN lives in the binding's
# global_env (which ttrun applies to the child env, beating -x), so vary it by rewriting a copy of
# the binding rather than exporting. The copy keeps mesh_graph_desc_path relative to TT_METAL_HOME.
if [ -n "${PP_MAX_SEQ_LEN:-}" ]; then
  cp "$BINDING" "$OUT/binding.yaml"
  sed -i "s|PREFILL_MAX_SEQ_LEN: \"[0-9]*\"|PREFILL_MAX_SEQ_LEN: \"$PP_MAX_SEQ_LEN\"|" "$OUT/binding.yaml"
  BINDING="$OUT/binding.yaml"
  echo "[driver] window override: PREFILL_MAX_SEQ_LEN=$PP_MAX_SEQ_LEN via $BINDING"
fi

# A stale descriptor from a prior run would make the readiness poll pass before this runner is up.
rm -f "$DESCRIPTOR"

echo "[driver] launching 4-rank runner under tt-run ($(date -Is))"
setsid python3 ttnn/ttnn/distributed/ttrun.py \
  --rank-binding "$BINDING" \
  --mpi-args "--host $(hostname):${PP_RANKS:-4} --map-by slot --bind-to none --tag-output --allow-run-as-root \
              -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x MISTRAL4_HF_MODEL -x PREFILL_HF_MODEL \
              -x PREFILL_MANIFEST -x PREFILL_TTNN_CACHE -x PREFILL_CHUNK_SIZE -x PREFILL_NUM_USERS -x PREFILL_USE_TRACE -x PREFILL_TRACE_REGION_SIZE -x PREFILL_KV_ONLY_LAST_LAYER -x PREFILL_NUM_LAYERS -x PROF_ROOT -x PROF_NAME -x PROF_PORT_BASE" \
  -- ${PP_TARGET:-python3 -m models.demos.common.prefill.runners.prefill_runner} \
  > "$OUT/runner.log" 2>&1 &
RUNNER_PGID=$!
echo "$RUNNER_PGID" > "$OUT/runner.pgid"
echo "[driver] runner pgid=$RUNNER_PGID; waiting for H2D descriptor $DESCRIPTOR"

# Readiness: rank 0 publishes the H2D descriptor once it is serving. With no migration there is no
# KV table / device map to wait on, so the descriptor is the only gate.
READY_TIMEOUT_S="${READY_TIMEOUT_S:-3000}"
deadline=$(( $(date +%s) + READY_TIMEOUT_S ))
while [ ! -e "$DESCRIPTOR" ]; do
  if ! kill -0 "$RUNNER_PGID" 2>/dev/null; then
    echo "[driver] FAIL: runner exited during startup; tail:"; tail -40 "$OUT/runner.log"; exit 1
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "[driver] FAIL: runner not ready within ${READY_TIMEOUT_S}s; tail:"; tail -40 "$OUT/runner.log"; exit 1
  fi
  sleep 5
done
echo "[driver] runner ready ($(date -Is)); starting producer"

# Independent single-chunk requests back to back -- the same workload shape the single-process
# concurrent test measures ("one request retires per iteration"), so the throughput is comparable.
PREFILL_MAX_SEQ_LEN="${PP_MAX_SEQ_LEN:-10240}" \
PREFILL_CHUNK_SIZE="${PP_CHUNK_SIZE:-5120}" \
PREFILL_H2D_SERVICE_ID=$SERVICE_ID \
PREFILL_PRODUCER_CHECK_PCC=0 \
PREFILL_PRODUCER_CHUNKS="${PP_CHUNKS:-1}" \
PREFILL_PRODUCER_MAX_REQUESTS="${PP_REQUESTS:-24}" \
PREFILL_PRODUCER_INTERLEAVE=round_robin \
PREFILL_PRODUCER_P_GAP=0 \
PREFILL_PRODUCER_P_BURST=0 \
PREFILL_SEND_SHUTDOWN=1 \
  timeout "${PRODUCER_TIMEOUT_S:-1800}" python3 -m models.demos.common.prefill.runners.prefill_producer \
  > "$OUT/producer.log" 2>&1
PROD_RC=$?
echo "[driver] producer rc=$PROD_RC ($(date -Is))"

# PREFILL_SEND_SHUTDOWN=1 means the sentinel drains the pipeline and every rank exits 0 on its own.
for i in $(seq 1 60); do kill -0 "$RUNNER_PGID" 2>/dev/null || break; sleep 5; done
if kill -0 "$RUNNER_PGID" 2>/dev/null; then
  echo "[driver] runner still up after sentinel; terminating"
  kill -INT -"$RUNNER_PGID" 2>/dev/null || kill -INT "$RUNNER_PGID" 2>/dev/null || true
  sleep 20
  kill -9 -"$RUNNER_PGID" 2>/dev/null || kill -9 "$RUNNER_PGID" 2>/dev/null || true
fi
echo "[driver] done; logs in $OUT"
exit $PROD_RC
