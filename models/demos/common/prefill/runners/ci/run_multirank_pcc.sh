#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
set -euo pipefail

MODEL="${1:?usage: run_multirank_pcc.sh <model-key> [config]}"
CONFIG="${2:-sc4}"

: "${TT_METAL_HOME:?TT_METAL_HOME must be set}"
: "${PREFILL_SUMMARIES:?PREFILL_SUMMARIES must be set by the blaze impl (shared /ci scratch for the KV table)}"
export PYTHONPATH="${TT_METAL_HOME}"
MANIFEST_DIR="${TT_METAL_HOME}/models/demos/deepseek_v3_d_p/tt/runners/manifests"
MGD_DIR="${TT_METAL_HOME}/models/demos/common/prefill/runners/topology_configuration/ci"

CHUNK_SIZE=5120
GOLDEN_LEN=56320
WARMUP_CHUNKS=10
PCC_THRESHOLD=0.85
RUNNER_ENV=""
PRODUCER_ENV=""
TP_SHARD_KV_DEFAULT=0
FABRIC_MODE=2d
# sc1 runs a single galaxy, so both of these exist to shrink the sc4 model down to what one fits.
# Defaults keep every model that does fit unchanged: full 256k context, full manifest depth.
SC1_MAX_SEQ_LEN=256000
SC1_NUM_LAYERS=""

case "${CONFIG}" in
  sc1|sc4) ;;
  *)
    echo "unknown config '${CONFIG}' (expected sc1 or sc4)" >&2
    exit 2
    ;;
esac

case "${MODEL}" in
  kimi27)
    export PIPELINE_DIR="${PREFILL_SUMMARIES/prefill_summaries/prefill_runner_kv}"
    MANIFEST="${MANIFEST_DIR}/kimi27.json"
    MAX_SEQ_LEN=256000
    # Users are bounded by per-bank KV capacity, and that bound has to be bisected, not computed --
    # the arithmetic bound overshoots ~20% once weights and transients are counted. The OOM edge sits
    # just above this and wanders between ranks, so re-bisect before raising it.
    NUM_USERS_DEFAULT=86
    RUNNER_ENV="export PREFILL_HF_MODEL=/mnt/models/moonshotai/Kimi-K2_7-Code-dequantized; export PREFILL_USE_TRACE=1; export PREFILL_LAYER_ACK_D2H=1;"
    PRODUCER_ENV="export PREFILL_PRODUCER_MANIFEST='${MANIFEST}';"
    ;;
  glm52)
    export PIPELINE_DIR="${PREFILL_SUMMARIES/prefill_summaries/glm52_prefill_runner_kv}"
    MANIFEST="${MANIFEST_DIR}/glm52.json"
    MAX_SEQ_LEN=1049600
    # Same per-bank capacity bound, relaxed by the TP KV dedup below. The sparse KV format moves it
    # a long way (SP x TP fits 34 at bf16, 56 at fp8), so this sits well under the edge, not on it.
    NUM_USERS_DEFAULT=28
    TP_SHARD_KV_DEFAULT=1
    # UNTRACED, as of now
    RUNNER_ENV="export PREFILL_LAYER_ACK_D2H=1;"
    PRODUCER_ENV="export PREFILL_PRODUCER_MANIFEST='${MANIFEST}'; \
        export PREFILL_TRACE_DIR=/mnt/models/deepseek-prefill-cache/glm-traces/vllm-glm52-indexer-kcache-55k;"
    ;;
  kimi_k3)
    export PIPELINE_DIR="${PREFILL_SUMMARIES/prefill_summaries/kimi_k3_prefill_runner_kv}"
    MANIFEST="${MANIFEST_DIR}/kimi_k3.json"
    MAX_SEQ_LEN=56320
    # Bisected, not computed: the arithmetic bound overshoots ~20% once weights and transients are
    # counted, and the OOM edge wanders between ranks. 1 is the measured 93-layer configuration.
    NUM_USERS_DEFAULT=1
    # 93 layers do not fit one galaxy -- MLA's static CBs become unplaceable past ~36 layers on a
    # rank (#54876) and a 48-layer single rank OOMs at 2 users. 24 fits, ends on an MLA layer, and
    # is the deepest depth the golden's decoder-output stream covers, so sc1 is a real accuracy gate
    # rather than a smaller copy of sc4.
    SC1_NUM_LAYERS=24
    SC1_MAX_SEQ_LEN=${MAX_SEQ_LEN}
    RUNNER_ENV="export PREFILL_HF_MODEL=/mnt/models/blaze/moonshotai/Kimi-K3-dequantized; export PREFILL_LAYER_ACK_D2H=1;"
    PRODUCER_ENV="export PREFILL_PRODUCER_MANIFEST='${MANIFEST}'; \
        export PREFILL_TRACE_DIR=/mnt/models/deepseek-prefill-cache/golden/k3_vllm_code_debug_1M;"
    ;;
  *)
    echo "unknown model key '${MODEL}'" >&2
    exit 2
    ;;
esac

MGD="${MGD_DIR}/${MODEL}_${CONFIG}_mgd.textproto"
[ -f "${MGD}" ] || { echo "no mesh-graph descriptor for ${MODEL}/${CONFIG} at ${MGD}" >&2; exit 2; }

SC4_MAX_SEQ_LEN=${MAX_SEQ_LEN}
NUM_LAYERS_ENV=""
if [ "${CONFIG}" = sc1 ]; then
  MAX_SEQ_LEN=${SC1_MAX_SEQ_LEN}
  NUM_USERS_DEFAULT=1
  # Exported to BOTH runner and producer, and only when the model asked for it: the manifest's depth
  # is applied with setdefault, so an explicit export is what shrinks it.
  [ -n "${SC1_NUM_LAYERS}" ] && NUM_LAYERS_ENV="export PREFILL_NUM_LAYERS=${SC1_NUM_LAYERS};"
fi

REAL_CHUNKS=$((MAX_SEQ_LEN / CHUNK_SIZE))

SC1_CHUNKS=$((SC1_MAX_SEQ_LEN / CHUNK_SIZE))
SC4_CHUNKS=$((SC4_MAX_SEQ_LEN / CHUNK_SIZE))
PROBE_CHUNKS="0,$((50000 / CHUNK_SIZE)),$((SC1_CHUNKS / 2 - 1)),$((SC1_CHUNKS - 1)),$((SC4_CHUNKS / 2 - 1)),$((SC4_CHUNKS - 1))"

mkdir -p "${PIPELINE_DIR}"
TTRUN_DIR="${TTRUN_DIR:-/etc/ttop}"
TTRUN_PY="${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py"

RESOLVED_HOSTS=$(awk 'NF {printf "%s,", $1}' "${TTRUN_DIR}/hostfile" | sed 's/,$//')
TTRUN_CWD="${PIPELINE_DIR}/ttrun-cwd"
mkdir -p "${TTRUN_CWD}"

MR_DIR=$(mktemp -d "${PIPELINE_DIR}/${MODEL}_prefill_ci_mr.XXXXXX")
export TABLE_PATH="${MR_DIR}/kv_chunk_table.pb"
PCC_DIR="${MR_DIR}/pcc_verdict"
RANKLOGS="${MR_DIR}/ranklogs"
TIMING_DIR="${MR_DIR}/timing"
mkdir -p "${TIMING_DIR}"

cleanup() {
  if [ -n "${RUNNER_PID:-}" ] && kill -0 "${RUNNER_PID}" 2>/dev/null; then
    kill "${RUNNER_PID}" 2>/dev/null || true
    wait "${RUNNER_PID}" 2>/dev/null || true
  fi
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
    python3 "${TT_METAL_HOME}/models/demos/common/prefill/runners/ci/summarize_ci_run.py" \
      --ranklogs "${RANKLOGS}" --timing-dir "${TIMING_DIR}" --real-chunks "${REAL_CHUNKS}" \
      --chunk-size "${CHUNK_SIZE}" \
      --probe-chunks "${PROBE_CHUNKS}" \
      --summary-name "${MODEL}_${CONFIG}" \
      || echo "summary generation failed (non-fatal)"
    if [ "$(find "${TIMING_DIR}" -name '*.csv' 2>/dev/null | wc -l)" -ge 2 ]; then
      GANTT_DIR="${PREFILL_SUMMARIES}/plots"
      mkdir -p "${GANTT_DIR}"
      python3 -c "import matplotlib" 2>/dev/null \
        || timeout 90 uv pip install --quiet matplotlib 2>/dev/null \
        || timeout 90 python3 -m pip install --quiet matplotlib 2>/dev/null \
        || echo "matplotlib install failed (gantt skipped, non-fatal)"
      python3 "${TT_METAL_HOME}/models/demos/deepseek_v3_d_p/scripts/plot_pipeline_trace.py" \
        --timing-dir "${TIMING_DIR}" --real-chunks "${REAL_CHUNKS}" \
        -o "${GANTT_DIR}/${MODEL}_pipeline_gantt.png" \
        || echo "gantt render failed (non-fatal)"
    fi
  fi
  rm -rf "${MR_DIR}"
}
trap cleanup EXIT

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
    export PREFILL_FABRIC_MODE=${FABRIC_MODE}; \
    export PREFILL_MAX_SEQ_LEN=${MAX_SEQ_LEN}; \
    export PREFILL_NUM_USERS=${PREFILL_NUM_USERS:-${NUM_USERS_DEFAULT}}; \
    export PREFILL_TP_SHARD_KV=${PREFILL_TP_SHARD_KV:-${TP_SHARD_KV_DEFAULT}}; \
    export PREFILL_SYNC_PER_CHUNK=1; \
    export PREFILL_TIMING_DIR='${TIMING_DIR}'; \
    export PREFILL_ENABLE_MIGRATION=1; \
    export PREFILL_MOCK_MIGRATION=1; \
    export PREFILL_MIGRATION_TABLE_PATH='${TABLE_PATH}'; \
    ${NUM_LAYERS_ENV} \
    ${RUNNER_ENV} \
    export LOGURU_LEVEL=INFO; \
    exec python3 -m models.demos.common.prefill.runners.prefill_runner" &
RUNNER_PID=$!
cd "${TT_METAL_HOME}"

# Bounds mesh bringup + weight load + warmup compile, not the chunk loop -- the table is published
# once the runner starts serving. Scales with model depth, so it is sized for the deepest model on
# the rig: Kimi-K3 at 93 layers over 4 ranks measured 42.6 min from step start to _serve_request
# (run 34648535999), which overran the previous 30 min and made the script exit 1 while the ranks
# were still compiling. The loop breaks as soon as the table appears, so a larger bound costs the
# shallower models nothing. Keep it below the leg's job timeout, or the container is killed first
# and this diagnostic never prints.
TABLE_WAIT_SECS="${TABLE_WAIT_SECS:-3600}"
for _ in $(seq 1 $((TABLE_WAIT_SECS / 5))); do
  [ -f "${TABLE_PATH}" ] && break
  kill -0 "${RUNNER_PID}" 2>/dev/null || { echo "runner exited before publishing the KV table"; wait "${RUNNER_PID}"; exit 1; }
  sleep 5
done
[ -f "${TABLE_PATH}" ] || { echo "KV table not published within ${TABLE_WAIT_SECS}s"; exit 1; }

RANKFILE=$(ls -t "${TTRUN_CWD}"/generated/ttrun/*/rankfile 2>/dev/null | head -1)
[ -f "${RANKFILE}" ] || { echo "tt-run rankfile not found under ${TTRUN_CWD}/generated/ttrun/*/rankfile"; exit 1; }
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
    export PREFILL_PRODUCER_CHUNKS=${REAL_CHUNKS}; \
    export PREFILL_PRODUCER_WARMUP_CHUNKS=${WARMUP_CHUNKS}; \
    export PREFILL_PCC_GOLDEN_LEN=${GOLDEN_LEN}; \
    export PREFILL_MIGRATION_TABLE_PATH='${TABLE_PATH}'; \
    export PREFILL_PCC_SUMMARY_DIR='${PCC_DIR}'; \
    export PREFILL_PRODUCER_CHECK_PCC=1; \
    export PREFILL_SEND_SHUTDOWN=1; \
    export PREFILL_STANDALONE_CHUNKED_PCC=${PCC_THRESHOLD}; \
    export PREFILL_H2D_CONNECT_TIMEOUT=120; \
    ${NUM_LAYERS_ENV} \
    ${PRODUCER_ENV} \
    export LOGURU_LEVEL=INFO; \
    exec python3 -m models.demos.common.prefill.runners.prefill_producer"
PROD_RC=$?
set -e

if [ "${PROD_RC}" -eq 0 ]; then
  wait "${RUNNER_PID}" || echo "runner exited non-zero after producer success (rc=$?)"
fi

EXPECTED_RANKS=$(printf '%s' "${HOSTS}" | tr ',' '\n' | grep -c .)
PCC_GATE_RC=0
python3 - "${PCC_DIR}" "${EXPECTED_RANKS}" <<'PY' || PCC_GATE_RC=$?
import glob, json, os, sys

pcc_dir, expected = sys.argv[1], int(sys.argv[2])
files = sorted(glob.glob(os.path.join(pcc_dir, "rank*.json")))
if len(files) < expected:
    print(f"PCC GATE FAIL: {len(files)}/{expected} producer verdict file(s) present", file=sys.stderr)
    sys.exit(1)
bad = 0
for f in files:
    name = os.path.basename(f)
    try:
        v = json.load(open(f))
    except Exception as e:
        print(f"PCC GATE FAIL: {name} unreadable: {e}", file=sys.stderr)
        bad += 1
        continue
    status = "ok" if v.get("ok") else "FAIL"
    print(f"  {name}: {status} min_pcc={v.get('min_pcc')} threshold={v.get('threshold')} per_cache={v.get('per_cache')}")
    if not v.get("ok"):
        bad += 1
if bad:
    print(f"PCC GATE FAIL: {bad}/{len(files)} rank(s) below threshold or unvalidated", file=sys.stderr)
    sys.exit(1)
print(f"PCC GATE PASS: {len(files)}/{expected} ranks ok, all caches >= threshold")
PY

if [ "${PROD_RC}" -ne 0 ]; then
  exit "${PROD_RC}"
fi
exit "${PCC_GATE_RC}"
