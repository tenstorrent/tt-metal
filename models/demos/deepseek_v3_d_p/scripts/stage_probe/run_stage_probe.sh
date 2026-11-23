#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Time one GLM-5.3 pipeline stage alone on this galaxy as a single 8x4 rank.
#
#   run_stage_probe.sh <name> <first_layer> <num_layers> [request_tokens]
#
# The stock prefill runner (glm53.json: traced, scaled-FP8 KV, D2H layer acks) builds layers
# [first_layer, first_layer + num_layers) through PREFILL_PP_LAYER_OFFSET at one user, and
# stage_probe_producer.py feeds it 10 warmup chunks and then one request of request_tokens (default the
# CI's 1,044,480) from the 1M golden trace's tokens, in 5120-token chunks. The stage writes its own
# KV and index caches chunk by chunk, so a chunk at depth d attends to d model-written rows. Every chunk
# logs one CHUNK_RETIRE line with its per-layer ack times (producer clock); stage_probe_summary.py reads them.
#
# Device access follows scripts/run_safe_pytest.sh's protocol: the global flock /tmp/tt-device.lock for
# the whole job, /tmp/tt-device.dirty while it runs (removed only on a clean exit), and a galaxy reset
# first if the previous holder left it dirty. Logs and stage lines go to $STAGE_PROBE_LOG_DIR.
set -euo pipefail

NAME="${1:?usage: run_stage_probe.sh <name> <first_layer> <num_layers> [request_tokens]}"
FIRST_LAYER="${2:?first layer (0-indexed)}"
NUM_LAYERS="${3:?layer count}"
REQUEST_TOKENS="${4:-1044480}"
CHUNK_SIZE=5120
WARMUP_CHUNKS="${STAGE_PROBE_WARMUP_CHUNKS:-10}"
# Extra runner exports, e.g. "export PREFILL_USE_TRACE=0;" for a debug run. tt-run does not forward PREFILL_*,
# so they go into the rank's own shell line.
RUNNER_EXTRA_ENV="${STAGE_PROBE_RUNNER_ENV:-}"

TT_METAL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
LOG_DIR="${STAGE_PROBE_LOG_DIR:?set STAGE_PROBE_LOG_DIR}"
TCP_INTERFACE="${PREFILL_TCP_INTERFACE:-ens5f0np0}"
RUNNER_READY_SECS="${STAGE_PROBE_READY_SECS:-1800}"
[ $((REQUEST_TOKENS % CHUNK_SIZE)) -eq 0 ] || { echo "request_tokens must be whole ${CHUNK_SIZE}-token chunks" >&2; exit 2; }

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SHA="$(git -C "${TT_METAL_HOME}" rev-parse --short=11 HEAD)"
RUN_DIR="${LOG_DIR}/stage-probe-${NAME}-${SHA}-${STAMP}"
mkdir -p "${RUN_DIR}"
STAGE_LOG="${RUN_DIR}/stages.log"
RUNNER_LOG="${RUN_DIR}/runner.log"
PRODUCER_LOG="${RUN_DIR}/producer.log"
# Unique per run, so no other agent's prefill shm segments are touched or reused.
SERVICE_ID="stage_probe_${NAME}_${STAMP}"

stage() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "${STAGE_LOG}"; }

stage "probe ${NAME}: layers [${FIRST_LAYER}, $((FIRST_LAYER + NUM_LAYERS))) request=${REQUEST_TOKENS} tok " \
  "warmup=${WARMUP_CHUNKS} chunks, ${TT_METAL_HOME} @ ${SHA} dirty=$(git -C "${TT_METAL_HOME}" status --porcelain --untracked-files=no | wc -l)"
cat > "${RUN_DIR}/command.txt" <<EOF
$0 $*
TT_METAL_HOME=${TT_METAL_HOME} sha=$(git -C "${TT_METAL_HOME}" rev-parse HEAD) host=$(hostname) start=${STAMP}
warmup_chunks=${WARMUP_CHUNKS} runner_extra_env=${RUNNER_EXTRA_ENV:-<none>}
EOF

cd "${TT_METAL_HOME}"
source python_env/bin/activate
export TT_METAL_HOME PYTHONPATH="${TT_METAL_HOME}"

exec 9>/tmp/tt-device.lock
stage "waiting for device lock"
flock 9
stage "device lock acquired"
if [ -f /tmp/tt-device.dirty ]; then
  stage "device dirty from a previous holder: tt-smi -glx_reset"
  tt-smi -glx_reset > "${RUN_DIR}/reset.log" 2>&1 || { stage "reset FAILED (see reset.log)"; exit 3; }
  rm -f /tmp/tt-device.dirty
  stage "reset done"
fi
touch /tmp/tt-device.dirty
# Power and clock state, read inside the lock window (tt-smi opens the chips): per-chip tdp_limit and aiclk.
if timeout 120 tt-smi -s --snapshot_no_tty --offline > "${RUN_DIR}/tt_smi_snapshot.json" 2> "${RUN_DIR}/tt_smi_snapshot.err"; then
  POWER="$(python3 - "${RUN_DIR}/tt_smi_snapshot.json" <<'PY'
import collections, json, sys
text = open(sys.argv[1]).read()
snap = json.loads(text[text.index("{"):])
vals = collections.defaultdict(collections.Counter)
for dev in snap.get("device_info", []):
    for section in dev.values():
        if isinstance(section, dict):
            for key in ("tdp_limit", "aiclk", "aiclk_limit_max"):
                if key in section and section[key] not in (None, "N/A"):
                    vals[key][str(section[key]).strip()] += 1
print(" ".join(f"{k}={dict(c)}" for k, c in sorted(vals.items())) or "no tdp/aiclk fields")
PY
)"
else
  POWER="tt-smi snapshot failed (see tt_smi_snapshot.err)"
fi
echo "power: ${POWER}" >> "${RUN_DIR}/command.txt"
stage "power: ${POWER}"

RUNNER_PID=""
CLEAN=0
cleanup() {
  if [ -n "${RUNNER_PID}" ] && kill -0 "${RUNNER_PID}" 2>/dev/null; then
    stage "stopping runner process group ${RUNNER_PID}"
    kill -- "-${RUNNER_PID}" 2>/dev/null || true
    timeout 30 tail --pid="${RUNNER_PID}" -f /dev/null 2>/dev/null || kill -KILL -- "-${RUNNER_PID}" 2>/dev/null || true
  fi
  rm -f /dev/shm/*"${SERVICE_ID}"* 2>/dev/null || true
  # The runner prints its "[pp rank" layout line just before it opens the mesh; without it no chip was touched.
  if [ "${CLEAN}" != 1 ] && ! grep -q "\[pp rank" "${RUNNER_LOG}" 2>/dev/null; then
    stage "runner never reached mesh open; device untouched"
    CLEAN=1
  fi
  if [ "${CLEAN}" = 1 ]; then
    rm -f /tmp/tt-device.dirty
    stage "done, device clean"
  else
    stage "FAILED: device left marked dirty (/tmp/tt-device.dirty); the next lock holder resets it"
  fi
}
trap cleanup EXIT

mkdir -p "${RUN_DIR}/ttrun-cwd"
# A static one-rank binding on the single-galaxy torus descriptor (run_pipeline_prefill.sh's one-rank default):
# tt-run discovery needs build/tools/scaleout/generate_rank_bindings, which a default build does not produce.
BINDING="${RUN_DIR}/rank_binding.yaml"
cat > "${BINDING}" <<YAML
rank_bindings:
  - {rank: 0, mesh_id: 0, mesh_host_rank: 0, env_overrides: {}}
mesh_graph_desc_path: ${TT_METAL_HOME}/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto
YAML
stage "launching runner (tt-run, 1 rank, ${BINDING##*/})"
(
  cd "${RUN_DIR}/ttrun-cwd"
  exec setsid python3 "${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py" \
    --skip-executable-check --tcp-interface "${TCP_INTERFACE}" --rank-binding "${BINDING}" \
    --mpi-args "--host $(hostname):1 --map-by slot --bind-to none --tag-output --wdir ${TT_METAL_HOME} -x PATH -x LD_LIBRARY_PATH" \
    bash -lc "cd '${TT_METAL_HOME}'; source python_env/bin/activate; export TT_METAL_HOME='${TT_METAL_HOME}'; \
      export PYTHONPATH='${TT_METAL_HOME}'; \
      export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0; export PYTHONUNBUFFERED=1; \
      export PREFILL_MANIFEST='${TT_METAL_HOME}/models/demos/deepseek_v3_d_p/tt/runners/manifests/glm53.json'; \
      export PREFILL_CHUNK_SIZE=${CHUNK_SIZE}; export PREFILL_NUM_USERS=1; \
      export PREFILL_NUM_LAYERS=${NUM_LAYERS}; export PREFILL_PP_LAYER_OFFSET=${FIRST_LAYER}; \
      export PREFILL_ENABLE_MIGRATION=0; \
      export PREFILL_MIGRATION_TABLE_PATH='${RUN_DIR}/kv_chunk_table.pb'; \
      export PREFILL_MIGRATION_DEVICE_MAP_PATH='${RUN_DIR}/device_map.json'; \
      export PREFILL_H2D_SERVICE_ID='${SERVICE_ID}'; \
      export PREFILL_LAYER_COMPLETION_RING='/${SERVICE_ID}_ring'; \
      export LOGURU_LEVEL=INFO; ${RUNNER_EXTRA_ENV} \
      exec python3 -m models.demos.common.prefill.runners.prefill_runner"
) > "${RUNNER_LOG}" 2>&1 &
RUNNER_PID=$!

waited=0
until grep -q "setup complete, entering request loop" "${RUNNER_LOG}" 2>/dev/null; do
  kill -0 "${RUNNER_PID}" 2>/dev/null || { stage "runner exited during setup (see runner.log)"; tail -30 "${RUNNER_LOG}"; exit 1; }
  [ "${waited}" -lt "${RUNNER_READY_SECS}" ] || { stage "runner not ready after ${waited}s"; exit 1; }
  sleep 10
  waited=$((waited + 10))
  [ $((waited % 60)) -ne 0 ] || stage "runner setup ${waited}s: $(grep -v '^\s*$' "${RUNNER_LOG}" | tail -1 | cut -c1-160)"
done
stage "runner ready after ~${waited}s; starting producer"

(
  export PREFILL_PRODUCER_MANIFEST="${TT_METAL_HOME}/models/demos/deepseek_v3_d_p/tt/runners/manifests/glm53.json"
  export PREFILL_MODEL=glm_5_3 PREFILL_CHUNK_SIZE=${CHUNK_SIZE} PREFILL_MAX_SEQ_LEN=1064960
  export PREFILL_NUM_USERS=1 PREFILL_NUM_LAYERS=${NUM_LAYERS}
  export PREFILL_PRODUCER_CHUNKS=$((REQUEST_TOKENS / CHUNK_SIZE)) PREFILL_PRODUCER_WARMUP_CHUNKS=${WARMUP_CHUNKS}
  export PREFILL_PRODUCER_MAX_REQUESTS=1 PREFILL_PRODUCER_MULTI_TURN_PROB=0 PREFILL_PRODUCER_CHECK_PCC=0
  export PREFILL_TRACE_DIR=/mnt/weka/model-cache/scratch/zai-org/GLM-5.3-Cache/golden_traces/glm53-1020k-last5120
  export PREFILL_H2D_SERVICE_ID="${SERVICE_ID}" PREFILL_H2D_CONNECT_TIMEOUT=120
  export PYTHONUNBUFFERED=1 LOGURU_LEVEL=INFO
  exec python3 -m models.demos.deepseek_v3_d_p.scripts.stage_probe.stage_probe_producer
) > "${PRODUCER_LOG}" 2>&1 &
PRODUCER_PID=$!
while kill -0 "${PRODUCER_PID}" 2>/dev/null; do
  sleep 20
  stage "request: $(grep -c 'CHUNK_RETIRE' "${PRODUCER_LOG}" 2>/dev/null || true) chunks retired"
done
set +e
wait "${PRODUCER_PID}"
PROD_RC=$?
set -e
stage "producer rc=${PROD_RC}; $(grep -c 'CHUNK_RETIRE.*warmup=0' "${PRODUCER_LOG}" || true) measured chunks"

# The SHUTDOWN sentinel makes the runner drain, release its trace and close the mesh.
if timeout 600 tail --pid="${RUNNER_PID}" -f /dev/null; then
  set +e
  wait "${RUNNER_PID}"
  RUNNER_RC=$?
  set -e
  RUNNER_PID=""
else
  RUNNER_RC=124
fi
stage "runner rc=${RUNNER_RC}"
[ "${PROD_RC}" = 0 ] && [ "${RUNNER_RC}" = 0 ] && CLEAN=1
echo "STAGE_PROBE_RESULT name=${NAME} first_layer=${FIRST_LAYER} num_layers=${NUM_LAYERS} producer_rc=${PROD_RC} runner_rc=${RUNNER_RC} dir=${RUN_DIR}" | tee -a "${STAGE_LOG}"
[ "${CLEAN}" = 1 ]
