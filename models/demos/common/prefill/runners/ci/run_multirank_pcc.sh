#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
set -euo pipefail

MODEL="${1:?usage: run_multirank_pcc.sh <model-key> [config]}"
CONFIG="${2:-sc4}"

: "${TT_METAL_HOME:?TT_METAL_HOME must be set}"
: "${PREFILL_SUMMARIES:?PREFILL_SUMMARIES must be set by the blaze impl (shared /ci scratch for the KV table)}"
export PYTHONPATH="${TT_METAL_HOME}:${TT_METAL_HOME}/ttnn"
MANIFEST_DIR="${TT_METAL_HOME}/models/demos/deepseek_v3_d_p/tt/runners/manifests"
MGD_DIR="${TT_METAL_HOME}/models/demos/common/prefill/runners/topology_configuration/ci"

CHUNK_SIZE=5120
GOLDEN_LEN=56320
WARMUP_CHUNKS=10
PCC_THRESHOLD=0.85
RUNNER_ENV=""
PRODUCER_ENV=""
PRODUCER_USERS=1
TCP_INTERFACE="${PREFILL_TCP_INTERFACE:-ens5f0np0}"

case "${CONFIG}" in
  sc1|sc4) ;;
  *)
    echo "unknown config '${CONFIG}' (expected sc1 or sc4)" >&2
    exit 2
    ;;
esac

case "${MODEL}" in
  llama31)
    [ "${CONFIG}" = sc1 ] || { echo "Llama prefill currently supports sc1 only" >&2; exit 2; }
    export PIPELINE_DIR="${PREFILL_SUMMARIES}/llama31_prefill_runner_kv"
    MANIFEST="${TT_METAL_HOME}/models/demos/llama_3p1_8b_d_p/tt/runners/manifests/llama_3p1_8b.json"
    CHUNK_SIZE=1024
    GOLDEN_LEN=2048
    WARMUP_CHUNKS=0
    PCC_THRESHOLD=0.99
    PRODUCER_USERS=2
    # Separate passages make a slot-address mixup visible to the PCC check.
    : "${PREFILL_PRODUCER_SLOT_TRACES:?set two comma-separated Llama golden trace directories}"
    python3 - "${PREFILL_PRODUCER_SLOT_TRACES}" <<'PY'
import json, pathlib, sys
paths = [pathlib.Path(p.strip()) for p in sys.argv[1].split(",")]
if len(paths) != 2:
    sys.exit("Llama requires exactly two slot traces")
ids = [json.loads((p / "metadata.json").read_text())["token_ids"] for p in paths]
if any(len(tokens) != 2048 for tokens in ids) or ids[0] == ids[1]:
    sys.exit("Llama requires two distinct, complete 2048-token traces")
for path in paths:
    for layer in range(32):
        if not (path / "kv_cache" / f"layer_{layer}.safetensors").is_file():
            sys.exit(f"missing layer {layer} under {path}")
PY
    printf -v LLAMA_CHECKPOINT '%q' "${PREFILL_HF_MODEL:-/mnt/models/meta-llama/Llama-3.1-8B-Instruct}"
    RUNNER_ENV="export PREFILL_LAYER_ACK_D2H=0; export PREFILL_USE_TRACE=0; export PREFILL_KV_ONLY_LAST_LAYER=0; \
        export PREFILL_HF_MODEL=${LLAMA_CHECKPOINT};"
    PRODUCER_ENV="export PREFILL_PRODUCER_MANIFEST='${MANIFEST}'; \
        export PREFILL_PRODUCER_SLOT_TRACES='${PREFILL_PRODUCER_SLOT_TRACES}'; \
        export PREFILL_PRODUCER_INTERLEAVE=round_robin; \
        export PREFILL_PRODUCER_MAX_REQUESTS=2; \
        export PREFILL_PRODUCER_DURATION_S=inf; export PREFILL_PRODUCER_MULTI_TURN_PROB=0; \
        export PREFILL_PRODUCER_P_GAP=0; export PREFILL_PRODUCER_P_BURST=0;"
    ;;
  kimi27)
    export PIPELINE_DIR="${PREFILL_SUMMARIES/prefill_summaries/prefill_runner_kv}"
    MANIFEST="${MANIFEST_DIR}/kimi27.json"
    PRODUCER_ENV="export PREFILL_PRODUCER_MANIFEST='${MANIFEST}';"
    ;;
  glm52)
    export PIPELINE_DIR="${PREFILL_SUMMARIES/prefill_summaries/glm52_prefill_runner_kv}"
    MANIFEST="${MANIFEST_DIR}/glm52.json"
    RUNNER_ENV="export TT_METAL_SHM_TRACKING_DISABLED=1; export LOGURU_LEVEL=ERROR;"
    PRODUCER_ENV="export PREFILL_PRODUCER_MANIFEST='${MANIFEST}'; \
        export PREFILL_TRACE_DIR=/mnt/models/deepseek-prefill-cache/glm-traces/vllm-glm52-indexer-kcache-55k;"
    ;;
  *)
    echo "unknown model key '${MODEL}'" >&2
    exit 2
    ;;
esac

MGD="${MGD_DIR}/${MODEL}_${CONFIG}_mgd.textproto"
[ -f "${MGD}" ] || { echo "no mesh-graph descriptor for ${MODEL}/${CONFIG} at ${MGD}" >&2; exit 2; }

manifest_env() {
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["env"][sys.argv[2]])' "${MANIFEST}" "$1"
}
MAX_SEQ_LEN=$(manifest_env PREFILL_MAX_SEQ_LEN)
NUM_USERS=$(manifest_env PREFILL_NUM_USERS)

RUNNER_OVERRIDES=""
SC4_MAX_SEQ_LEN=${MAX_SEQ_LEN}
SC1_MAX_SEQ_LEN=256000
if [ "${CONFIG}" = sc1 ] && [ "${MODEL}" != llama31 ]; then
  MAX_SEQ_LEN=${SC1_MAX_SEQ_LEN}
  NUM_USERS=1
  RUNNER_OVERRIDES="export PREFILL_MAX_SEQ_LEN=${MAX_SEQ_LEN}; export PREFILL_NUM_USERS=${NUM_USERS};"
fi
if [ -n "${PREFILL_NUM_USERS:-}" ]; then
  NUM_USERS=${PREFILL_NUM_USERS}
  RUNNER_OVERRIDES="${RUNNER_OVERRIDES} export PREFILL_NUM_USERS=${NUM_USERS};"
fi
if [ -n "${PREFILL_TP_SHARD_KV:-}" ]; then
  RUNNER_OVERRIDES="${RUNNER_OVERRIDES} export PREFILL_TP_SHARD_KV=${PREFILL_TP_SHARD_KV};"
fi
echo "resolved shape for ${MODEL}/${CONFIG}: max_seq_len=${MAX_SEQ_LEN} num_users=${NUM_USERS}"

REAL_CHUNKS=$((MAX_SEQ_LEN / CHUNK_SIZE))

SC1_CHUNKS=$((SC1_MAX_SEQ_LEN / CHUNK_SIZE))
SC4_CHUNKS=$((SC4_MAX_SEQ_LEN / CHUNK_SIZE))
PROBE_CHUNKS="0,$((50000 / CHUNK_SIZE)),$((SC1_CHUNKS / 2 - 1)),$((SC1_CHUNKS - 1)),$((SC4_CHUNKS / 2 - 1)),$((SC4_CHUNKS - 1))"
if [ "${MODEL}" = llama31 ]; then
  [ "${NUM_USERS}" = 2 ] || { echo "Llama acceptance requires two slots" >&2; exit 2; }
  PROBE_CHUNKS="0,1"
fi

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
  # Keep the table, verdicts and logs so a failure is reproducible after exit.
  echo "Run evidence: ${MR_DIR}"
}
trap cleanup EXIT

cd "${TTRUN_CWD}"
python3 "${TTRUN_PY}" \
  --skip-executable-check \
  --force-rediscovery \
  --tcp-interface "${TCP_INTERFACE}" \
  --mesh-graph-descriptor "${MGD}" \
  --hosts "${RESOLVED_HOSTS}" \
  --mpi-args "--bind-to none --tag-output --allow-run-as-root --wdir ${TT_METAL_HOME} --output-filename ${RANKLOGS}/runner -x PATH -x LD_LIBRARY_PATH" \
  bash -lc "cd '${TT_METAL_HOME}'; \
    export PYTHONPATH='${TT_METAL_HOME}:${TT_METAL_HOME}/ttnn'; \
    export PYTHONUNBUFFERED=1; \
    export PREFILL_MANIFEST='${MANIFEST}'; \
    export PREFILL_SYNC_PER_CHUNK=1; \
    export PREFILL_TIMING_DIR='${TIMING_DIR}'; \
    export PREFILL_ENABLE_MIGRATION=1; \
    export PREFILL_MOCK_MIGRATION=1; \
    export PREFILL_MIGRATION_TABLE_PATH='${TABLE_PATH}'; \
    ${RUNNER_OVERRIDES} \
    export LOGURU_LEVEL=INFO; \
    ${RUNNER_ENV} \
    exec python3 -m models.demos.common.prefill.runners.prefill_runner" &
RUNNER_PID=$!
cd "${TT_METAL_HOME}"

for _ in $(seq 1 360); do
  [ -f "${TABLE_PATH}" ] && break
  kill -0 "${RUNNER_PID}" 2>/dev/null || { echo "runner exited before publishing the KV table"; wait "${RUNNER_PID}"; exit 1; }
  sleep 5
done
[ -f "${TABLE_PATH}" ] || { echo "KV table not published within timeout"; exit 1; }

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
  --mca btl self,tcp --mca btl_tcp_if_include "${TCP_INTERFACE}" \
  bash -lc "cd '${TT_METAL_HOME}'; \
    export PYTHONPATH='${TT_METAL_HOME}:${TT_METAL_HOME}/ttnn'; \
    export PYTHONUNBUFFERED=1; \
    export PREFILL_MAX_SEQ_LEN=${MAX_SEQ_LEN}; \
    export PREFILL_NUM_USERS=${PRODUCER_USERS}; \
    export PREFILL_PRODUCER_CHUNKS=${REAL_CHUNKS}; \
    export PREFILL_PRODUCER_WARMUP_CHUNKS=${WARMUP_CHUNKS}; \
    export PREFILL_PCC_GOLDEN_LEN=${GOLDEN_LEN}; \
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

if [ "${PROD_RC}" -eq 0 ]; then
  # The shutdown sentinel must let the runner drain and exit successfully.
  wait "${RUNNER_PID}"
fi

EXPECTED_RANKS=$(printf '%s' "${HOSTS}" | tr ',' '\n' | grep -c .)
PCC_GATE_RC=0
python3 - "${PCC_DIR}" "${EXPECTED_RANKS}" "${MODEL}" <<'PY' || PCC_GATE_RC=$?
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
    if not v.get("ok") or (sys.argv[3] == "llama31" and v.get("slots_checked") != 2):
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
