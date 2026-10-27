#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
set -euo pipefail

USAGE="usage: run_multirank_pcc.sh <model-key> [config] [--variant <manifest-suffix>]"
VARIANT=""
POSITIONAL=()
while [ $# -gt 0 ]; do
  case "$1" in
    --variant) VARIANT="_${2:?--variant needs a manifest suffix, e.g. mtp_kv_share or dflash}"; shift 2 ;;
    -*) echo "unknown option '$1'; ${USAGE}" >&2; exit 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done
set -- "${POSITIONAL[@]+"${POSITIONAL[@]}"}"

MODEL="${1:?${USAGE}}"
RUN_NAME="${MODEL}${VARIANT}"
# First entry is the default; membership is the validation. These models support sc1 only.
case "${MODEL}" in
  mistral4|gptoss120b) MODEL_CONFIGS="sc1" ;;
  *)                   MODEL_CONFIGS="sc4 sc1" ;;
esac
CONFIG="${2:-${MODEL_CONFIGS%% *}}"
case " ${MODEL_CONFIGS} " in
  *" ${CONFIG} "*) ;;
  *) echo "${MODEL} does not support config '${CONFIG}' (supported: ${MODEL_CONFIGS})" >&2; exit 2 ;;
esac

: "${TT_METAL_HOME:?TT_METAL_HOME must be set}"
: "${PREFILL_SUMMARIES:?PREFILL_SUMMARIES must be set by the blaze impl (shared /ci scratch for the KV table)}"
export PYTHONPATH="${TT_METAL_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
printf -v CHILD_PYTHONPATH '%q' "${PYTHONPATH}"
MANIFEST_DIR="${TT_METAL_HOME}/models/demos/deepseek_v3_d_p/tt/runners/manifests"
MGD_DIR="${TT_METAL_HOME}/models/demos/common/prefill/runners/topology_configuration/ci"

manifest_env() {
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["env"][sys.argv[2]])' "${MANIFEST}" "$1"
}

MGD="${MGD_DIR}/${CONFIG}_mgd.textproto"

CHUNK_SIZE=5120
WARMUP_CHUNKS=10
PCC_THRESHOLD=0.85
GOLDEN_LEN=56320
EXPECTED_TPS="${PREFILL_EXPECTED_TPS:-}"
PERF_BASELINES="${TT_METAL_HOME}/models/demos/common/prefill/runners/ci/perf_baselines.json"
PERF_GATE="${TT_METAL_HOME}/models/demos/common/prefill/runners/ci/perf_gate.py"
# Demand a measurement even before a baseline exists, so a leg being calibrated cannot pass green
# on nothing.
REQUIRE_TPS="${PREFILL_REQUIRE_TPS:-0}"
RUNNER_ENV=""
PRODUCER_ENV=""
PRODUCER_USERS="${PREFILL_PRODUCER_NUM_USERS:-1}"
TCP_INTERFACE="${PREFILL_TCP_INTERFACE:-ens5f0np0}"
# Non-zero for a pipeline-parallel model: the rank count its binding fixes. Set in the case block.
PP_RANKS=0
# PCC verdicts, rank logs and any rank binding, kept for the run's artifact upload.
EVIDENCE_DIR="${PREFILL_SUMMARIES}/pcc/${RUN_NAME}"
MTP_PCC_THRESHOLD=""
DFLASH_PCC_THRESHOLD=""
TABLE_WAIT_DEFAULT=3600
VARIANT_MAX_SEQ_LEN=""
MTP=0
DFLASH=0
# sc1 runs a single galaxy, so both of these exist to shrink the sc4 model down to what one fits.
SC1_MAX_SEQ_LEN=""
SC1_NUM_LAYERS=""
SC1_NUM_USERS=1

case "${MODEL}" in
  gptoss120b)
    source "${TT_METAL_HOME}/models/demos/gpt_oss_d_p/scripts/ci/runner_config.sh"
    ;;
  llama31)
    source "${TT_METAL_HOME}/models/demos/llama_3p1_8b_d_p/scripts/ci/runner_config.sh"
    ;;
  kimi27)
    export PIPELINE_DIR="${PREFILL_SUMMARIES/prefill_summaries/prefill_runner_kv}"
    MANIFEST="${MANIFEST_DIR}/kimi27.json"
    PRODUCER_ENV="export PREFILL_TRACE_DIR=/mnt/weka/model-cache/stable/deepseek-prefill-cache/golden/structured_traces/vllm-kimi-k27-codedebug-256000-last5120;"
    if [ "${VARIANT}" = _dflash ]; then
      DFLASH=1
      MANIFEST="${MANIFEST_DIR}/kimi27_dflash.json"
      VARIANT_MAX_SEQ_LEN=256000
      DFLASH_PCC_THRESHOLD="${PREFILL_DFLASH_PCC:-0.85}"
      PRODUCER_ENV="export PREFILL_DFLASH_PCC=${DFLASH_PCC_THRESHOLD};"
      TABLE_WAIT_DEFAULT=7200
    fi
    ;;
  glm53)
    export PIPELINE_DIR="${PREFILL_SUMMARIES/prefill_summaries/glm53_prefill_runner_kv}"
    MANIFEST="${MANIFEST_DIR}/glm53.json"
    RUNNER_ENV="export TT_METAL_SHM_TRACKING_DISABLED=1; export LOGURU_LEVEL=ERROR;"
    PCC_THRESHOLD=0.83
    PRODUCER_ENV="export PREFILL_TRACE_DIR=/mnt/weka/model-cache/scratch/zai-org/GLM-5.3-Cache/golden_traces/glm53-1020k-last5120;"
    if [ "${VARIANT#_mtp}" != "${VARIANT}" ]; then
      MTP=1
      MANIFEST="${MANIFEST_DIR}/glm53${VARIANT}.json"
      SC1_MAX_SEQ_LEN=${GOLDEN_LEN}
      MTP_PCC_THRESHOLD=0.80
      MTP_TRACE="${PREFILL_MTP_TRACE_DIR:-/mnt/weka/model-cache/scratch/zai-org/GLM-5.3-Cache/golden_traces/mtp-tail-802cad1b-56320tok-L7-nozeropos0}"
      PRODUCER_ENV="export PREFILL_TRACE_DIR=/mnt/weka/model-cache/scratch/zai-org/GLM-5.3-Cache/golden_traces/vllm-glm53-indexer-kcache-55k; \
          export PREFILL_MTP_TRACE_DIR='${MTP_TRACE}'; export PREFILL_MTP_PCC=${MTP_PCC_THRESHOLD};"
      [ "${CONFIG}" = sc1 ] && TABLE_WAIT_DEFAULT=5400
    fi
    ;;
  kimi_k3)
    export PIPELINE_DIR="${PREFILL_SUMMARIES/prefill_summaries/kimi_k3_prefill_runner_kv}"
    MANIFEST="${MANIFEST_DIR}/kimi_k3.json"
    # 93 layers do not fit one galaxy -- MLA's static CBs become unplaceable past ~36 layers on a
    # rank (#54876) and a 48-layer single rank OOMs at 2 users. 24 fits, ends on an MLA layer, and
    # is the deepest depth the golden's decoder-output stream covers, so sc1 is a real accuracy gate
    # rather than a smaller copy of sc4. sc1 keeps the default 256k context, so its perf probes are
    # the 5k..250k ones GLM-5.3 reports and sc4's 1M request adds 510k and 1M on top. Not a
    # baseline for sc4 -- 24 layers against 93 -- which is why the summary job reports sc4 alone.
    SC1_NUM_LAYERS=24
    # sc1 is sc4's stage 0, so it opens sc4's slots.
    SC1_NUM_USERS=$(manifest_env PREFILL_NUM_USERS)
    RUNNER_ENV="export PREFILL_HF_MODEL=/mnt/weka/model-weights/llm/moonshotai/Kimi-K3-dequantized;"
    # The sc4 93-layer leg's KV PCC
    # collapsed to ~0 past layer ~24 on run 36717057668 with the Weka trace (it reads 0.900..0.994 on
    # run 36524165128), and the two candidates -- the Weka golden copy and the Weka TTNN cache -- give
    # the same symptom.
    PRODUCER_ENV="export PREFILL_TRACE_DIR=/mnt/weka/model-cache/stable/deepseek-prefill-cache/golden/k3_vllm_code_debug_1M;"
    # Deliberately only the first 55k: the K3 golden trace is currently unreliable.
    PRODUCER_ENV+=" export PREFILL_PCC_GOLDEN_LEN=${GOLDEN_LEN};"
    if [ "${VARIANT}" = _dspark ]; then
      [ "${CONFIG}" = sc4 ] || { echo "${MODEL} runs sc4 only" >&2; exit 2; }
      DFLASH=1
      MANIFEST="${MANIFEST_DIR}/kimi_k3_dspark.json"
      DFLASH_PCC_THRESHOLD="${PREFILL_DFLASH_PCC:-0.85}"
      PRODUCER_ENV="export PREFILL_DFLASH_PCC=${DFLASH_PCC_THRESHOLD};"
      TABLE_WAIT_DEFAULT=7200
    fi
    ;;
  mistral4)
    export PIPELINE_DIR="${PREFILL_SUMMARIES/prefill_summaries/mistral4_pp4_kv}"
    MANIFEST="${MANIFEST_DIR}/mistral4.json"
    # Pre-generated 4-rank binding on one host, not tt-run discovery. Everything below keys off
    # PP_RANKS, so the next pipeline-parallel model sets only that.
    PP_RANKS=4
    # Validates the entire 36-layer golden, including positions >8192.
    MAX_SEQ_LEN=${GOLDEN_LEN}
    # sc1 is the only config, so the sc1 window below must not widen it to the 256k default.
    SC1_MAX_SEQ_LEN=${GOLDEN_LEN}
    MISTRAL_MODEL="${PREFILL_HF_MODEL:-/mnt/models/blaze/mistralai/Mistral-Small-4-119B-2603}"
    MISTRAL_CACHE="${PREFILL_TTNN_CACHE:-/mnt/models/blaze/mistralai/Mistral-Small-4-Cache/CI}"
    MISTRAL_GOLDEN="${PREFILL_TRACE_DIR:-/mnt/models/blaze/mistralai/Mistral-Small-4-Cache/golden/mistral4_56320_36L}"
    # torus_y, not the runner's 2d_torus_xy default: a [8,1] stage wraps only the SP axis, and a Ring
    # collective on an unwrapped axis hangs at bring-up.
    RUNNER_ENV="export PREFILL_HF_MODEL='${MISTRAL_MODEL}'; export PREFILL_TTNN_CACHE='${MISTRAL_CACHE}'; \
        export PREFILL_FABRIC_MODE=2d_torus_y; \
        export PREFILL_USE_TRACE=1; export PREFILL_LAYER_ACK_D2H=1; \
        export TT_METAL_OPERATION_TIMEOUT_SECONDS=0;"
    PRODUCER_ENV="export PREFILL_TRACE_DIR='${MISTRAL_GOLDEN}'; export PREFILL_SP=8; export PREFILL_TP=1;"
    ;;
  *)
    echo "unknown model key '${MODEL}'" >&2
    exit 2
    ;;
esac

if [ -n "${VARIANT}" ]; then
  [ "${MANIFEST##*/}" = "${RUN_NAME}.json" ] || { echo "${MODEL} has no ${VARIANT#_} variant" >&2; exit 2; }
  [ -f "${MANIFEST}" ] || { echo "no manifest for ${RUN_NAME} at ${MANIFEST}" >&2; exit 2; }
  export PIPELINE_DIR="${PIPELINE_DIR}${VARIANT}"
  PRODUCER_ENV+=" export PREFILL_PCC_GOLDEN_LEN=${GOLDEN_LEN}; \
      export PREFILL_PRODUCER_MAX_REQUESTS=1; export PREFILL_PRODUCER_MULTI_TURN_PROB=0;"
  PERF_BASELINES=""
  REQUIRE_TPS=0
  EXPECTED_TPS=""
fi
PRODUCER_ENV="export PREFILL_PRODUCER_MANIFEST='${MANIFEST}'; ${PRODUCER_ENV}"

# A PP model carries its descriptor in its rank binding, so only the discovery path needs one.
if [ "${PP_RANKS}" -eq 0 ]; then
  [ -f "${MGD}" ] || { echo "no mesh-graph descriptor for ${CONFIG} at ${MGD}" >&2; exit 2; }
fi

MAX_SEQ_LEN=${VARIANT_MAX_SEQ_LEN:-$(manifest_env PREFILL_MAX_SEQ_LEN)}
NUM_USERS=$(manifest_env PREFILL_NUM_USERS)

RUNNER_OVERRIDES="${VARIANT_MAX_SEQ_LEN:+export PREFILL_MAX_SEQ_LEN=${MAX_SEQ_LEN};}"
SC4_MAX_SEQ_LEN=${MAX_SEQ_LEN}
SC1_MAX_SEQ_LEN=${SC1_MAX_SEQ_LEN:-${MAX_SEQ_LEN}}
NUM_LAYERS_ENV=""
if [ "${CONFIG}" = sc1 ]; then
  MAX_SEQ_LEN=${SC1_MAX_SEQ_LEN}
  NUM_USERS=${SC1_NUM_USERS}
  RUNNER_OVERRIDES="export PREFILL_MAX_SEQ_LEN=${MAX_SEQ_LEN}; export PREFILL_NUM_USERS=${NUM_USERS};"
  # Exported to BOTH runner and producer, and only when the model asked for it: the manifest's depth
  # is applied with setdefault, so an explicit export is what shrinks it.
  [ -n "${SC1_NUM_LAYERS}" ] && NUM_LAYERS_ENV="export PREFILL_NUM_LAYERS=${SC1_NUM_LAYERS};"
fi
if [ -n "${PREFILL_NUM_USERS:-}" ]; then
  NUM_USERS=${PREFILL_NUM_USERS}
  RUNNER_OVERRIDES="${RUNNER_OVERRIDES} export PREFILL_NUM_USERS=${NUM_USERS};"
fi
# tt-run does not forward PREFILL_*: pass a cache-format override to every rank.
if [ -n "${PREFILL_SCALED_FP8_KV_CACHE:-}" ]; then
  RUNNER_OVERRIDES="${RUNNER_OVERRIDES} export PREFILL_SCALED_FP8_KV_CACHE=${PREFILL_SCALED_FP8_KV_CACHE};"
fi
echo "resolved shape for ${RUN_NAME}/${CONFIG}: max_seq_len=${MAX_SEQ_LEN} num_users=${NUM_USERS}"

REAL_CHUNKS=$((MAX_SEQ_LEN / CHUNK_SIZE))

SC1_CHUNKS=$((SC1_MAX_SEQ_LEN / CHUNK_SIZE))
SC4_CHUNKS=$((SC4_MAX_SEQ_LEN / CHUNK_SIZE))
# Start, ~50k, ~100k, ~255k, and the midpoint + tail of each config's window. Deduped: a model with
# one config yields the same chunk twice, and the summary drops chunks past a short window.
PROBE_CHUNKS="${PROBE_CHUNKS:-$(printf '%s\n' 0 "$((50000 / CHUNK_SIZE))" "$((100000 / CHUNK_SIZE))" \
  "$((256000 / CHUNK_SIZE))" "$((SC1_CHUNKS / 2 - 1))" "$((SC1_CHUNKS - 1))" "$((SC4_CHUNKS / 2 - 1))" \
  "$((SC4_CHUNKS - 1))" | sort -n -u | paste -sd,)}"


mkdir -p "${PIPELINE_DIR}"
TTRUN_DIR="${TTRUN_DIR:-/etc/ttop}"
TTRUN_PY="${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py"
MPIRUN=$(command -v mpirun-ulfm || command -v mpirun)

RESOLVED_HOSTS=$(awk 'NF && $1 !~ /^#/ {printf "%s,", $1}' "${TTRUN_DIR}/hostfile" | sed 's/,$//')
TTRUN_CWD="${PIPELINE_DIR}/ttrun-cwd"
mkdir -p "${TTRUN_CWD}"

MR_DIR=$(mktemp -d "${PIPELINE_DIR}/${RUN_NAME}_prefill_ci_mr.XXXXXX")
GATE_LOG="${MR_DIR}/gates.log"
export TABLE_PATH="${MR_DIR}/kv_chunk_table.pb"
PCC_DIR="${MR_DIR}/pcc_verdict"
RANKLOGS="${MR_DIR}/ranklogs"
TIMING_DIR="${MR_DIR}/timing"
PRODUCER_LOG="${MR_DIR}/producer.log"
mkdir -p "${TIMING_DIR}"

SUMMARY_DONE=0
emit_summary() {
  [ "${SUMMARY_DONE}" = 1 ] && return 0
  SUMMARY_DONE=1
  python3 "${TT_METAL_HOME}/models/demos/common/prefill/runners/ci/summarize_ci_run.py" \
    --ranklogs "${RANKLOGS}" --timing-dir "${TIMING_DIR}" --real-chunks "${REAL_CHUNKS}" \
    --chunk-size "${CHUNK_SIZE}" \
    --probe-chunks "${PROBE_CHUNKS}" \
    --summary-name "${RUN_NAME}_${CONFIG}" \
    || echo "summary generation failed (non-fatal)"
}

cleanup() {
  if [ -n "${RUNNER_PID:-}" ] && kill -0 "${RUNNER_PID}" 2>/dev/null; then
    kill -- "-${RUNNER_PID}" 2>/dev/null || true
    timeout 10 tail --pid="${RUNNER_PID}" -f /dev/null 2>/dev/null || kill -KILL -- "-${RUNNER_PID}" 2>/dev/null || true
    wait "${RUNNER_PID}" 2>/dev/null || true
  fi
  echo "==================== per-cache PCC verdicts (PROD_RC=${PROD_RC:-<unset>}) ===================="
  for f in "${PCC_DIR}"/rank*.json; do
    [ -e "$f" ] || { echo "no PCC verdict files under ${PCC_DIR}"; break; }
    echo "$(basename "$f"): $(cat "$f")"
  done
  if [ "${MTP}" = 1 ]; then
    echo "==================== MTP per-level KV PCC ===================="
    find "${RANKLOGS}" -type f 2>/dev/null -exec grep -h -E "MTP level|MTP KV PCC|MTP: golden|MTP:.*level slots" {} + \
      | tail -40 || echo "no MTP PCC lines in the rank logs"
  elif [ "${DFLASH}" = 1 ]; then
    echo "==================== drafter KV PCC ===================="
    grep -h -E "drafter KV PCC|min over all|kv_cache_pcc_complete" "${PRODUCER_LOG}" 2>/dev/null | tail -20 \
      || echo "no drafter PCC lines -- were the dflash_* configs in the table?"
  fi
  mkdir -p "${EVIDENCE_DIR}"
  for path in "${PCC_DIR}" "${RANKLOGS}" "${PRODUCER_LOG}" "${BINDING:-}"; do
    [ -n "${path}" ] && [ -e "${path}" ] && cp -r "${path}" "${EVIDENCE_DIR}/" || true
  done
  if [ -d "${RANKLOGS}" ]; then
    echo "==================== ranklog tails ===================="
    find "${RANKLOGS}" -type f | sort | while read -r f; do
      echo "---- ${f#"${RANKLOGS}"/} ----"
      tail -n 40 "$f" 2>/dev/null || true
    done
    emit_summary
    if [ "$(find "${TIMING_DIR}" -name '*.csv' 2>/dev/null | wc -l)" -ge 2 ]; then
      GANTT_DIR="${PREFILL_SUMMARIES}/plots"
      mkdir -p "${GANTT_DIR}"
      GANTT_PY=python3
      GANTT_TMP=""
      if ! python3 -c "import matplotlib" 2>/dev/null; then
        GANTT_TMP=$(mktemp -d)
        if UV_CACHE_DIR="${GANTT_TMP}/uvcache" timeout 120 uv venv --quiet "${GANTT_TMP}/venv" \
           && UV_CACHE_DIR="${GANTT_TMP}/uvcache" timeout 180 uv pip install --quiet \
                --python "${GANTT_TMP}/venv/bin/python" matplotlib; then
          GANTT_PY="${GANTT_TMP}/venv/bin/python"
        else
          echo "matplotlib install failed (gantt skipped, non-fatal)"
        fi
      fi
      "${GANTT_PY}" "${TT_METAL_HOME}/models/demos/deepseek_v3_d_p/scripts/plot_pipeline_trace.py" \
        --timing-dir "${TIMING_DIR}" --real-chunks "${REAL_CHUNKS}" \
        -o "${GANTT_DIR}/${RUN_NAME}_pipeline_gantt.png" \
        || echo "gantt render failed (non-fatal)"
      if [ -n "${GANTT_TMP}" ]; then
        rm -rf "${GANTT_TMP}"
      fi
    fi
  fi
  # Last in the log, after the rank-log tails, so a failed gate is what a tail reader sees.
  echo "==================== gate verdicts ===================="
  echo "producer rc=${PROD_RC:-<unset>} runner rc=${RUNNER_RC:-<unset>} perf gate rc=${PERF_GATE_RC:-<unset>} pcc gate rc=${PCC_GATE_RC:-<unset>}"
  grep -h "GATE" "${GATE_LOG}" 2>/dev/null || true
  if [ -n "${PREFILL_KEEP_RUN_DIR:-}" ]; then
    echo "keeping run dir (PREFILL_KEEP_RUN_DIR set): ${MR_DIR}"
  else
    rm -rf "${MR_DIR}"
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [ "${PP_RANKS}" -ne 0 ]; then
  # The CI controller has no chips: probe the allocated worker, and never reuse the template's
  # device IDs, since column enumeration is host-specific.
  [ -n "${RESOLVED_HOSTS}" ] && [[ "${RESOLVED_HOSTS}" != *,* ]] || {
    echo "mistral4 PP4 requires exactly one Galaxy host in ${TTRUN_DIR}/hostfile" >&2; exit 2;
  }
  HOSTS="${RESOLVED_HOSTS}:${PP_RANKS}"
  EXPECTED_RANKS=${PP_RANKS}
  BINDING="${MR_DIR}/mistral4_pp4_binding.yaml"
  "${MPIRUN}" -np 1 --host "${RESOLVED_HOSTS}:1" --bind-to none --allow-run-as-root \
    bash -lc "set -e; cd '${TT_METAL_HOME}'; \
      test -r '${MISTRAL_MODEL}/config.json' || { echo 'Mistral checkpoint missing: ${MISTRAL_MODEL}' >&2; exit 1; }; \
      test -d '${MISTRAL_CACHE}/mistral_small_4_bh_8dev/8x1' || { echo 'Stage the PP4 8dev/8x1 weight cache under ${MISTRAL_CACHE} (or set PREFILL_TTNN_CACHE)' >&2; exit 1; }; \
      for layer in {0..35}; do \
        test -r '${MISTRAL_GOLDEN}/kv_cache/layer_'\${layer}'.safetensors' || { echo \"Missing golden KV layer \${layer} in ${MISTRAL_GOLDEN}\" >&2; exit 1; }; \
      done; \
      export PYTHONPATH='${TT_METAL_HOME}'; \
      export TT_METAL_OPERATION_TIMEOUT_SECONDS=0; \
      export TT_MESH_GRAPH_DESC_PATH='${TT_METAL_HOME}/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto'; \
      exec python3 models/demos/deepseek_v3_d_p/utils/gen_pipeline_binding.py --out '${BINDING}'"
  TTRUN_ARGS=(--rank-binding "${BINDING}")
  RUNNER_PLACEMENT="--host ${HOSTS} --map-by slot"
  # Each producer reads its own stage's nine layers; merging the four maps would read the whole
  # galaxy four times.
  RUNNER_ENV+=" export PREFILL_MIGRATION_DEVICE_MAP_PATH='${MR_DIR}/device_map.json';"
  PRODUCER_ENV+=" export PREFILL_MIGRATION_DEVICE_MAP_PATH='${MR_DIR}/device_map.json';"
else
  TTRUN_ARGS=(--force-rediscovery --mesh-graph-descriptor "${MGD}" --hosts "${RESOLVED_HOSTS}")
  RUNNER_PLACEMENT=""
fi

cd "${TTRUN_CWD}"
setsid python3 "${TTRUN_PY}" \
  --skip-executable-check \
  --tcp-interface "${TCP_INTERFACE}" \
  "${TTRUN_ARGS[@]}" \
  --mpi-args "${RUNNER_PLACEMENT} --bind-to none --tag-output --allow-run-as-root --wdir ${TT_METAL_HOME} --output-filename ${RANKLOGS}/runner -x PATH -x LD_LIBRARY_PATH" \
  bash -lc "cd '${TT_METAL_HOME}'; \
    export PYTHONPATH=${CHILD_PYTHONPATH}; \
    export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0; \
    export PYTHONUNBUFFERED=1; \
    export PREFILL_MANIFEST='${MANIFEST}'; \
    export PREFILL_CHUNK_SIZE=${CHUNK_SIZE}; \
    export PREFILL_SYNC_PER_CHUNK=1; \
    export PREFILL_TIMING_DIR='${TIMING_DIR}'; \
    export PREFILL_ENABLE_MIGRATION=0; \
    export PREFILL_MIGRATION_TABLE_PATH='${TABLE_PATH}'; \
    ${RUNNER_OVERRIDES} \
    ${NUM_LAYERS_ENV} \
    export LOGURU_LEVEL=INFO; \
    ${RUNNER_ENV} \
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
TABLE_WAIT_SECS="${TABLE_WAIT_SECS:-${TABLE_WAIT_DEFAULT}}"
WAITED=0
while [ "${WAITED}" -lt "${TABLE_WAIT_SECS}" ]; do
  [ -f "${TABLE_PATH}" ] && break
  kill -0 "${RUNNER_PID}" 2>/dev/null || { echo "runner exited before publishing the KV table"; wait "${RUNNER_PID}"; exit 1; }
  if [ -n "${VARIANT}" ] && [ $((WAITED % 60)) -eq 0 ] && [ "${WAITED}" -gt 0 ]; then
    CRASHED=$(find "${RANKLOGS}" -type f -name '*.err' 2>/dev/null \
      -exec grep -l -F "Traceback (most recent call last)" {} + 2>/dev/null || true)
    [ -z "${CRASHED}" ] || { echo "a rank crashed ${WAITED}s in, before the KV table was published: ${CRASHED}"; exit 1; }
  fi
  sleep 5
  WAITED=$((WAITED + 5))
done
[ -f "${TABLE_PATH}" ] || { echo "KV table not published within ${TABLE_WAIT_SECS}s"; exit 1; }

# A PP binding already fixed the rank count and host; only discovery has to parse them back out.
if [ "${PP_RANKS}" -eq 0 ]; then
  RANKFILE=$(ls -t "${TTRUN_CWD}"/generated/ttrun/*/rankfile 2>/dev/null | head -1)
  [ -f "${RANKFILE}" ] || { echo "tt-run rankfile not found under ${TTRUN_CWD}/generated/ttrun/*/rankfile"; exit 1; }
  HOSTS=$(awk '/^rank[[:space:]]+[0-9]+=/ {n=$2; sub(/=.*/,"",n); h=$2; sub(/^[0-9]+=/,"",h); print n" "h}' "${RANKFILE}" | sort -n | awk '{printf "%s%s:1", (NR>1?",":""), $2}')
  [ -n "${HOSTS}" ] || { echo "failed to parse producer host order from ${RANKFILE}"; exit 1; }
  echo "producer host order from tt-run discovery: ${HOSTS}"
  EXPECTED_RANKS=$(printf '%s' "${HOSTS}" | tr ',' '\n' | grep -c .)
fi

PRODUCER_PLACEMENT=(-np "${EXPECTED_RANKS}" --map-by slot)
if [ "${DFLASH}" = 1 ]; then
  mkdir -p "${TT_METAL_HOME}/generated/ttrun"
  cp "${RANKFILE}" "${TT_METAL_HOME}/generated/ttrun/producer_rankfile"
  PRODUCER_PLACEMENT=(--map-by rankfile:file=generated/ttrun/producer_rankfile --wdir "${TT_METAL_HOME}")
fi

if [ "${DFLASH}" = 1 ]; then
  CHECK_PY="${MR_DIR}/check_table_configs.py"
  cat > "${CHECK_PY}" <<'PY'
import sys, ttnn

table = ttnn.experimental.disaggregation.import_from_protobuf_file(sys.argv[1])
names = [table.config_name(i) for i in range(table.num_configs())]
dflash = [x for x in names if x.startswith("dflash_")]
print(f"[dflash-ci] table configs: {len(names)} total, {len(dflash)} drafter -> {dflash[:4]}...")
if not dflash:
    sys.exit("[dflash-ci] FATAL: merged table carries no dflash_* configs; the drafter half would be skipped")
PY
  "${MPIRUN}" -np 1 --host "${HOSTS%%,*}" --bind-to none --tag-output --allow-run-as-root \
    -x PATH -x LD_LIBRARY_PATH \
    bash -lc "cd '${TT_METAL_HOME}'; export PYTHONPATH='${TT_METAL_HOME}'; exec python3 '${CHECK_PY}' '${TABLE_PATH}'"
fi

set +e
"${MPIRUN}" \
  --host "${HOSTS}" "${PRODUCER_PLACEMENT[@]}" --bind-to none --tag-output --allow-run-as-root \
  --output-filename "${RANKLOGS}/producer" \
  --mca btl self,tcp --mca btl_tcp_if_include "${TCP_INTERFACE}" \
  -x PATH -x LD_LIBRARY_PATH \
  bash -lc "cd '${TT_METAL_HOME}'; \
    export PYTHONPATH=${CHILD_PYTHONPATH}; \
    export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0; \
    export PYTHONUNBUFFERED=1; \
    export PREFILL_CHUNK_SIZE=${CHUNK_SIZE}; \
    export PREFILL_MAX_SEQ_LEN=${MAX_SEQ_LEN}; \
    export PREFILL_NUM_USERS=${PRODUCER_USERS}; \
    export PREFILL_PRODUCER_CHUNKS=${REAL_CHUNKS}; \
    export PREFILL_PRODUCER_WARMUP_CHUNKS=${WARMUP_CHUNKS}; \
    export PREFILL_MIGRATION_TABLE_PATH='${TABLE_PATH}'; \
    export PREFILL_PCC_SUMMARY_DIR='${PCC_DIR}'; \
    export PREFILL_PRODUCER_CHECK_PCC=1; \
    export PREFILL_SEND_SHUTDOWN=1; \
    export PREFILL_STANDALONE_CHUNKED_PCC=${PCC_THRESHOLD}; \
    export PREFILL_H2D_CONNECT_TIMEOUT=120; \
    ${NUM_LAYERS_ENV} \
    ${PRODUCER_ENV} \
    export LOGURU_LEVEL=INFO; \
    exec python3 -m models.demos.common.prefill.runners.prefill_producer" 2>&1 | tee "${PRODUCER_LOG}"
PROD_RC=${PIPESTATUS[0]}
set -e

if [ "${DFLASH}" = 1 ] && [ "${PROD_RC}" -eq 0 ] && ! grep -q "min_dflash_pcc=" "${PRODUCER_LOG}"; then
  echo "no rank reported a drafter KV PCC: the dflash caches were never checked" >&2
  exit 1
fi

RUNNER_RC=0
if [ "${PROD_RC}" -eq 0 ]; then
  # A producer can finish successfully while a runner hangs during shutdown.
  if timeout 120 tail --pid="${RUNNER_PID}" -f /dev/null; then
    wait "${RUNNER_PID}" || RUNNER_RC=$?
  else
    echo "runner did not shut down within 120 seconds" >&2
    RUNNER_RC=124
  fi
fi

PERF_GATE_RC=0
if [ "${REQUIRE_TPS}" = 1 ] || [ -n "${EXPECTED_TPS}" ] || [ -f "${PERF_BASELINES}" ]; then
  # The summary normally runs in the EXIT trap, after this point; force it early so the gate has
  # a file to read. The trap's own call is a no-op once this one has run.
  emit_summary
  PERF_POWER=""
  if [ -z "${EXPECTED_TPS}" ] && [ -f "${PERF_BASELINES}" ]; then
    PERF_POWER=$("${MPIRUN}" -np 1 --host "${RESOLVED_HOSTS%%,*}:1" --bind-to none --allow-run-as-root \
      bash -lc "cd '${TT_METAL_HOME}' && PYTHONPATH='${TT_METAL_HOME}' python3 ${PERF_GATE} --print-power" | tail -n 1) || true
    case "${PERF_POWER}" in
      8kw|14kw) ;;
      *) echo "perf baseline WARN: power probe on ${RESOLVED_HOSTS%%,*} returned '${PERF_POWER}'" >&2; PERF_POWER="" ;;
    esac
  fi
  python3 "${PERF_GATE}" \
    "${PREFILL_SUMMARIES}/perf_json/${RUN_NAME}_${CONFIG}.json" \
    --table "${PERF_BASELINES}" --model "${MODEL}" --config "${CONFIG}" --power "${PERF_POWER}" \
    --expected-tps "${EXPECTED_TPS}" --margin "${PREFILL_PERF_MARGIN:-}" \
    --require "${REQUIRE_TPS}" 2>&1 | tee -a "${GATE_LOG}" || PERF_GATE_RC=$?
fi

PCC_GATE_RC=0
python3 - "${PCC_DIR}" "${EXPECTED_RANKS}" "${PCC_THRESHOLD}" "${PRODUCER_USERS}" "${MTP_PCC_THRESHOLD}" <<'PY' 2>&1 | tee -a "${GATE_LOG}" || PCC_GATE_RC=$?
import glob, json, math, os, sys

pcc_dir, expected, threshold = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
expected_slots = int(sys.argv[4])
mtp_threshold = float(sys.argv[5]) if sys.argv[5] else None


def floor(cache):
    return mtp_threshold if mtp_threshold is not None and cache.startswith("mtp") else threshold


files = sorted(glob.glob(os.path.join(pcc_dir, "rank*.json")))
if {os.path.basename(f) for f in files} != {f"rank{r}.json" for r in range(expected)}:
    print(f"PCC GATE FAIL: expected exactly ranks 0..{expected - 1}, found {files}", file=sys.stderr)
    sys.exit(1)
bad = 0
mtp_ranks = []
for f in files:
    name = os.path.basename(f)
    try:
        v = json.load(open(f))
    except Exception as e:
        print(f"PCC GATE FAIL: {name} unreadable: {e}", file=sys.stderr)
        bad += 1
        continue
    per_cache = v.get("per_cache") or {}
    if any(c.startswith("mtp") for c in per_cache):
        mtp_ranks.append(name)
    scores = [(v.get("min_pcc"), min([threshold, *map(floor, per_cache)])), *((s, floor(c)) for c, s in per_cache.items())]

    def unusable(s, lo):
        return not isinstance(s, (int, float)) or isinstance(s, bool) or not math.isfinite(s) or not lo <= s <= 1

    # A verdict is trusted only if it says so AND proves it did the work. Checked as a list so the
    # log names the clause that tripped -- "unvalidated" and "below threshold" need different fixes.
    reason = next(
        (m for ok, m in (
            (v.get("ok") is True, "ok is not literally true"),
            (v.get("rank") == int(name[4:-5]), "rank field does not match the filename"),
            (v.get("slots_checked") == expected_slots, f"slots_checked is not the {expected_slots} slots fed"),
            (bool(per_cache), "per_cache is empty, so no cache was compared"),
            (v.get("threshold") == threshold, f"threshold is not the {threshold} this gate demands"),
            (not any(unusable(s, lo) for s, lo in scores), "a score is missing, non-finite, or below its floor"),
        ) if not ok),
        None,
    )
    print(
        f"  {name}: {'ok' if reason is None else 'FAIL'} min_pcc={v.get('min_pcc')} "
        f"threshold={v.get('threshold')} per_cache={per_cache}"
        + (f"  <-- {reason}" if reason else "")
    )
    if reason is not None:
        bad += 1
if mtp_threshold is not None and len(mtp_ranks) != 1:
    print(f"PCC GATE FAIL: expected exactly one rank with mtp caches, found {mtp_ranks}", file=sys.stderr)
    bad += 1
if bad:
    print(f"PCC GATE FAIL: {bad}/{len(files)} rank(s) below threshold or unvalidated", file=sys.stderr)
    sys.exit(1)
print(f"PCC GATE PASS: {len(files)}/{expected} ranks ok, all caches >= threshold")
PY

if [ "${PROD_RC}" -ne 0 ]; then
  exit "${PROD_RC}"
fi
if [ "${RUNNER_RC}" -ne 0 ]; then
  echo "runner exited non-zero after producer success (rc=${RUNNER_RC})" >&2
  exit "${RUNNER_RC}"
fi
if [ "${PERF_GATE_RC}" -ne 0 ]; then
  exit "${PERF_GATE_RC}"
fi
exit "${PCC_GATE_RC}"
