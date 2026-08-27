#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Multi-galaxy disaggregated prefill cache-accuracy leg: a background N-rank runner opens the mesh, loads
# the model, warmup-compiles and publishes the merged KV chunk table to shared NFS; a device-less producer
# then feeds it, reads EVERY device cache the model published back over UMD (a sparse model's indexer key
# cache as well as its KVPE cache) and PCCs each against the golden trace. The producer's
# exit code is the verdict, so this script's exit code is the producer's.
#
# Usage: run_multirank_pcc.sh <model-key>       # model-key selects a block in the case below
#
# Everything outside that case block is model-independent launcher plumbing (host-order derivation, table
# poll, runner reaping, durable-verdict dump), so a new model is a case entry, not another copy.
set -euo pipefail

MODEL="${1:?usage: run_multirank_pcc.sh <model-key>}"

: "${TT_METAL_HOME:?TT_METAL_HOME must be set}"
: "${PREFILL_SUMMARIES:?PREFILL_SUMMARIES must be set by the blaze impl (shared /ci scratch for the KV table)}"
export PYTHONPATH="${TT_METAL_HOME}"
MANIFEST_DIR="${TT_METAL_HOME}/models/demos/deepseek_v3_d_p/tt/runners/manifests"
MGD_DIR="${TT_METAL_HOME}/models/demos/common/prefill/runners/topology_configuration"

# Shared defaults; a case block below overrides what its model needs. Each branch also names its own
# scratch dir by swapping the summaries component out of PREFILL_SUMMARIES: that keeps the run-scoped leaf
# and the shared-NFS root the blaze impl chose, while leaving the summaries dir itself alone.
MAX_SEQ_LEN=56320
PCC_THRESHOLD=0.85
RUNNER_ENV=""
PRODUCER_ENV=""

case "${MODEL}" in
  kimi27)
    export PIPELINE_DIR="${PREFILL_SUMMARIES/prefill_summaries/prefill_runner_kv}"
    MGD="${MGD_DIR}/pipeline_prefill_4galaxy_connected_mesh_graph_descriptor.textproto"
    MANIFEST="${MANIFEST_DIR}/kimi27.json"
    # Dense MLA: one device cache, so the manifest's model + depth are all the producer needs. The runner
    # needs the weight path on top: the K2.7 adapter inherits K2.6's reference default.
    RUNNER_ENV="export PREFILL_HF_MODEL=/mnt/models/moonshotai/Kimi-K2_7-Code-dequantized;"
    PRODUCER_ENV="export PREFILL_PRODUCER_MANIFEST='${MANIFEST}';"
    ;;
  glm52)
    export PIPELINE_DIR="${PREFILL_SUMMARIES/prefill_summaries/glm52_prefill_runner_kv}"
    # LINE/LINE variant: GLM-5.2's MoE all_to_all deadlocks in warmup under the torus fabric modes on
    # multi-galaxy pipeline prefill, so the descriptor declares no wrap and the fabric mode stays 2d.
    MGD="${MGD_DIR}/pipeline_prefill_4galaxy_connected_fabric2d_mesh_graph_descriptor.textproto"
    MANIFEST="${MANIFEST_DIR}/glm52.json"
    # Sparse DSA: TWO device caches (MLA KVPE over all 78 layers + the lightning-indexer KEY cache over the
    # 21 `full` layers), both PCC'd. The trace must be the indexer-K dump -- the adapter's default golden
    # carries no dsa/indexer_k_layer_*, which would silently downgrade this leg to a KVPE-only check.
    PRODUCER_ENV="export PREFILL_PRODUCER_MANIFEST='${MANIFEST}'; \
        export PREFILL_TRACE_DIR=/mnt/models/deepseek-prefill-cache/glm-traces/vllm-glm52-indexer-kcache-55k;"
    ;;
  *)
    echo "unknown model key '${MODEL}'" >&2
    exit 2
    ;;
esac

mkdir -p "${PIPELINE_DIR}"
TTRUN_DIR="${TTRUN_DIR:-/etc/ttop}"
TTRUN_PY="${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py"

# The runner goes through tt-run automapper, which discovers rank->mesh placement from the MGD and takes
# the plain comma-separated host list. The producer (raw mpirun) needs a slotted host list in the SAME rank
# order tt-run's discovery chose (parsed from the generated rankfile below) so its master co-locates with
# runner rank 0 -- the H2D stream service is host-local /dev/shm IPC.
RESOLVED_HOSTS=$(awk 'NF {printf "%s,", $1}' "${TTRUN_DIR}/hostfile" | sed 's/,$//')
# tt-run Phase 1 (generate_rank_bindings) writes generated/ttrun/ under the launch cwd; keep it on shared
# NFS so the runner pod can read rank-0 outputs.
TTRUN_CWD="${PIPELINE_DIR}/ttrun-cwd"
mkdir -p "${TTRUN_CWD}"

MR_DIR=$(mktemp -d "${PIPELINE_DIR}/${MODEL}_prefill_ci_mr.XXXXXX")
export TABLE_PATH="${MR_DIR}/kv_chunk_table.pb"
# Each producer rank drops its PCC verdict here before the shutdown sentinel; mpirun drops buffered stdout
# when the runner tears down, so this file is the only durable record of the measured per-cache PCC.
PCC_DIR="${MR_DIR}/pcc_verdict"
# Per-rank stdout/stderr files (mpirun --output-filename). These survive the teardown race that truncates
# forwarded stdout, so their tails are the authoritative debug log.
RANKLOGS="${MR_DIR}/ranklogs"

# The runner idles owning the multi-galaxy allocation until the producer's shutdown sentinel; on any early
# exit (table-publish timeout, producer failure) it must be reaped or it strands the hardware until the
# step timeout. Killing ttrun.py tears down the remote ranks with it.
cleanup() {
  if [ -n "${RUNNER_PID:-}" ] && kill -0 "${RUNNER_PID}" 2>/dev/null; then
    kill "${RUNNER_PID}" 2>/dev/null || true
    wait "${RUNNER_PID}" 2>/dev/null || true
  fi
  # Dump diagnostics before rm: on every exit path (success, producer failure, table timeout, step-timeout
  # kill) the live mpirun stdout may be truncated at teardown, so the durable verdict JSON and per-rank
  # ranklog tails are the authoritative record.
  echo "==================== per-cache PCC verdicts (PROD_RC=${PROD_RC:-<unset>}) ===================="
  for f in "${PCC_DIR}"/rank*.json; do
    [ -e "$f" ] || { echo "no PCC verdict files under ${PCC_DIR}"; break; }
    echo "$(basename "$f"): $(cat "$f")"
  done
  if [ -d "${RANKLOGS}" ]; then
    echo "==================== ranklog tails ===================="
    find "${RANKLOGS}" -type f | sort | while read -r f; do
      echo "---- ${f#"${RANKLOGS}"/} ----"
      tail -n 40 "$f" 2>/dev/null || true
    done
  fi
  rm -rf "${MR_DIR}"
}
trap cleanup EXIT

# tt-run automapper: --mesh-graph-descriptor + --hosts generate the rank binding by discovery, so no static
# binding is committed. tt-run forwards only the device vars it owns (TT_MESH_*) plus its
# ENV_PASSTHROUGH_PREFIXES; PREFILL_* fall outside that set, so -- exactly as the producer launch below does
# -- export them inside a per-rank bash -lc rather than relying on tt-run to carry them. Launch from
# TTRUN_CWD so Phase 1 caches on shared NFS; RUNNER_PID stays ttrun.py's pid so cleanup()'s kill still tears
# the remote ranks down.
cd "${TTRUN_CWD}"
python3 "${TTRUN_PY}" \
  --skip-executable-check \
  --force-rediscovery \
  --tcp-interface ens5f0np0 \
  --mesh-graph-descriptor "${MGD}" \
  --hosts "${RESOLVED_HOSTS}" \
  --mpi-args "--bind-to none --tag-output --allow-run-as-root --wdir ${TT_METAL_HOME} --output-filename ${RANKLOGS}/runner -x PATH -x LD_LIBRARY_PATH" \
  bash -lc "cd '${TT_METAL_HOME}'; \
    export PYTHONPATH='${TT_METAL_HOME}'; \
    export PYTHONUNBUFFERED=1; \
    export PREFILL_MANIFEST='${MANIFEST}'; \
    export PREFILL_FABRIC_MODE=2d; \
    export PREFILL_MAX_SEQ_LEN=${MAX_SEQ_LEN}; \
    export PREFILL_ENABLE_MIGRATION=1; \
    export PREFILL_MOCK_MIGRATION=1; \
    export PREFILL_MIGRATION_TABLE_PATH='${TABLE_PATH}'; \
    ${RUNNER_ENV} \
    export LOGURU_LEVEL=INFO; \
    exec python3 -m models.demos.common.prefill.runners.prefill_runner" &
RUNNER_PID=$!
cd "${TT_METAL_HOME}"

for _ in $(seq 1 360); do
  [ -f "${TABLE_PATH}" ] && break
  kill -0 "${RUNNER_PID}" 2>/dev/null || { echo "runner exited before publishing the KV table"; wait "${RUNNER_PID}"; exit 1; }
  sleep 5
done
[ -f "${TABLE_PATH}" ] || { echo "KV table not published within timeout"; exit 1; }

# Producer rank i must land on the host running runner rank i (host-local /dev/shm H2D descriptor), and
# tt-run discovery -- not hostfile order -- decides that mapping, so derive the host order from the rankfile
# --force-rediscovery just wrote. Using raw hostfile order races discovery and intermittently strands the
# master on the wrong host -- H2DStreamService.connect then times out and TT_THROWs.
RANKFILE=$(ls -t "${TTRUN_CWD}"/generated/ttrun/*/rankfile 2>/dev/null | head -1)
[ -f "${RANKFILE}" ] || { echo "tt-run rankfile not found under ${TTRUN_CWD}/generated/ttrun/*/rankfile"; exit 1; }
# rankfile lines are "rank N=hostname slots=X"; emit "N hostname", sort by rank, join as host:1.
HOSTS=$(awk '/^rank[[:space:]]+[0-9]+=/ {n=$2; sub(/=.*/,"",n); h=$2; sub(/^[0-9]+=/,"",h); print n" "h}' "${RANKFILE}" | sort -n | awk '{printf "%s%s:1", (NR>1?",":""), $2}')
[ -n "${HOSTS}" ] || { echo "failed to parse producer host order from ${RANKFILE}"; exit 1; }
echo "producer host order from tt-run discovery: ${HOSTS}"

MPIRUN=$(command -v mpirun-ulfm || command -v mpirun)
set +e
"${MPIRUN}" \
  --host "${HOSTS}" --map-by slot --bind-to none --tag-output --allow-run-as-root \
  --output-filename "${RANKLOGS}/producer" \
  --mca btl self,tcp --mca btl_tcp_if_include ens5f0np0 \
  bash -lc "cd '${TT_METAL_HOME}'; \
    export PYTHONPATH='${TT_METAL_HOME}'; \
    export PYTHONUNBUFFERED=1; \
    export PREFILL_MAX_SEQ_LEN=${MAX_SEQ_LEN}; \
    export PREFILL_MIGRATION_TABLE_PATH='${TABLE_PATH}'; \
    export PREFILL_PCC_SUMMARY_DIR='${PCC_DIR}'; \
    export PREFILL_PRODUCER_CHECK_PCC=1; \
    export PREFILL_SEND_SHUTDOWN=1; \
    export PREFILL_STANDALONE_CHUNKED_PCC=${PCC_THRESHOLD}; \
    export PREFILL_H2D_CONNECT_TIMEOUT=120; \
    ${PRODUCER_ENV} \
    export LOGURU_LEVEL=INFO; \
    exec python3 -m models.demos.common.prefill.runners.prefill_producer"
PROD_RC=$?
set -e

# The cleanup() EXIT trap dumps the durable PCC verdicts and ranklog tails on every exit path, so no inline
# diagnostics are needed here. On success the producer already sent the shutdown sentinel; wait for the
# runner to finish teardown so a hung shutdown surfaces here instead of being orphaned.
if [ "${PROD_RC}" -eq 0 ]; then
  wait "${RUNNER_PID}" || echo "runner exited non-zero after producer success (rc=$?)"
fi

exit ${PROD_RC}
