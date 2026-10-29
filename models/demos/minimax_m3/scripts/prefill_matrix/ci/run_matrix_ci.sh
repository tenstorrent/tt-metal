#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# MiniMax-M3 pipeline-prefill PERF MATRIX, CI flavour (Blaze prefill pipeline, bh_sc4 = 4 galaxies, 16 stages of [2,4]).
# Same measurement as run_matrix.sh (matrix_producer.py, unchanged), but launched the way the other SC4 runner legs are
# (models/demos/common/prefill/runners/ci/run_multirank_pcc.sh): hosts from /etc/ttop/hostfile, ranks through tt-run,
# no Slurm, no galaxy reset (the job's setup step already reset and validated the cluster). Perf only: the numbers are
# reported, not gated; the script fails only when the run itself breaks.
#
# Unlike run_matrix.sh (one launch per cached row, KV capacity cached + MAX_NEW), ONE runner launch serves every row,
# with the KV capacity at the model's max context (KV_CAPACITY, default 1048576), as a production server allocates it.
# This loads the weights once. Capacity barely costs time on current main: in run 39751806622 the single-dense-layer
# stages took 9-12 ms per chunk at the same KV length with capacity 194560 and 604160. Each row is one producer session
# against the live runner; the runner is stopped once, after the last row.
#
# Env (all optional): CACHED=0  NEW=5120,32768  ITERS=2  USERS=2  TARGET_CHUNKS=64  REQS  SKIP_IDLE=1
#   CHUNK=5120  LAYER_COUNTS=<per-stage comma list>  MAX_NEW=51200  KV_CAPACITY=1048576 (rounded down to a CHUNK
#   multiple)  EXPERT_DTYPE=bf4
#   HF_MODEL / TT_CACHE_PATH / PREFILL_TRACE_DIR (weka defaults)  MATRIX_CI_LAUNCH=discovery|binding
#   TABLE_WAIT_SECS=2400 (bring-up bound)  PRODUCER_TIMEOUT_SECS=900 (per cached row; a hung row fails before the job timeout)
#   SHUTDOWN_WAIT_SECS=240 (how long a runner gets to exit after the SHUTDOWN sentinel before it is killed)
#   MATRIX_CI_DRY_RUN=1: print the launch commands and exit (no MPI, no devices; TTRUN_DIR may point at a fake
#   hostfile/rankfile).
#
# MATRIX_CI_LAUNCH:
#   discovery (default): tt-run new mode on the 16-stage z-chain mesh graph; generate_rank_bindings places the 16
#     meshes on whatever quad the job got (trays, TT_VISIBLE_DEVICES and host order included; MPI rank == mesh_id).
#   binding: the hand-written exabox "straight chain" binding (binding_16stage_quad.yaml.in), hosts in the ring order of
#     ${TTRUN_DIR}/rankfile. Fallback for a quad on which the discovery mapping does not fit.
set -euo pipefail

PKG_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${TT_METAL_HOME:?TT_METAL_HOME must be set}"
: "${PREFILL_SUMMARIES:?PREFILL_SUMMARIES must be set by the blaze impl (shared /ci scratch)}"
export TT_METAL_HOME
export PYTHONPATH="${TT_METAL_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
source "${PKG_DIR}/matrix_common.sh"   # MATRIX_CHUNK, MATRIX_MAX_NEW, MATRIX_LAYER_COUNTS, matrix_producer_env

STAGES=16
LAST=$((STAGES - 1))
CACHED=${CACHED:-0}
NEW=${NEW:-5120,32768}
ITERS=${ITERS:-2}
USERS=${USERS:-2}
TARGET_CHUNKS=${TARGET_CHUNKS:-64}
MAX_NEW=${MAX_NEW:-$MATRIX_MAX_NEW}
KV_CAPACITY=${KV_CAPACITY:-$((1048576 / MATRIX_CHUNK * MATRIX_CHUNK))}   # MiniMax-M3 max_position_embeddings, in whole chunks
LAUNCH=${MATRIX_CI_LAUNCH:-discovery}
DRY_RUN=${MATRIX_CI_DRY_RUN:-0}
TABLE_WAIT_SECS=${TABLE_WAIT_SECS:-2400}
PRODUCER_TIMEOUT_SECS=${PRODUCER_TIMEOUT_SECS:-900}
SHUTDOWN_WAIT_SECS=${SHUTDOWN_WAIT_SECS:-240}
TCP_INTERFACE="${PREFILL_TCP_INTERFACE:-ens5f0np0}"
export HF_MODEL=${HF_MODEL:-/mnt/weka/model-weights/llm/minimax/MiniMax-M3}
export TT_CACHE_PATH=${TT_CACHE_PATH:-/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill}
export PREFILL_TRACE_DIR=${PREFILL_TRACE_DIR:-/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill/golden/longbook_56320}

# ---- validate the knobs before touching the cluster --------------------------------------------------------------
case "${LAUNCH}" in discovery | binding) ;; *) echo "MATRIX_CI_LAUNCH must be discovery or binding, got '${LAUNCH}'" >&2; exit 2 ;; esac
[ "${MATRIX_CHUNK}" -ge 1024 ] && [ $((MATRIX_CHUNK % 32)) -eq 0 ] || { echo "CHUNK=${MATRIX_CHUNK} must be a multiple of 32 and >= 1024" >&2; exit 2; }
[ $((MAX_NEW % MATRIX_CHUNK)) -eq 0 ] || { echo "MAX_NEW=${MAX_NEW} is not a multiple of CHUNK=${MATRIX_CHUNK}" >&2; exit 2; }
for n in ${NEW//,/ }; do
  [ "${n}" -le "${MAX_NEW}" ] || { echo "NEW value ${n} exceeds MAX_NEW=${MAX_NEW}" >&2; exit 2; }
done
CACHED_LIST=()
for c in ${CACHED//,/ }; do
  c_rounded=$((c / MATRIX_CHUNK * MATRIX_CHUNK))
  [ "${c_rounded}" -eq "${c}" ] || echo "[matrix-ci] rounding cached ${c} down to ${c_rounded} (multiple of CHUNK=${MATRIX_CHUNK})"
  CACHED_LIST+=("${c_rounded}")
done
[ "${#CACHED_LIST[@]}" -gt 0 ] || { echo "CACHED names no rows" >&2; exit 2; }
[ $((KV_CAPACITY % MATRIX_CHUNK)) -eq 0 ] || { echo "KV_CAPACITY=${KV_CAPACITY} is not a multiple of CHUNK=${MATRIX_CHUNK}" >&2; exit 2; }
for c in "${CACHED_LIST[@]}"; do
  [ $((c + MAX_NEW)) -le "${KV_CAPACITY}" ] || { echo "cached ${c} + MAX_NEW ${MAX_NEW} exceeds KV_CAPACITY=${KV_CAPACITY}" >&2; exit 2; }
done
RUNNER_USERS=$((USERS > 0 ? USERS : 1))   # USERS=0 is idle-only; the runner still needs slot 0

# ---- hosts -------------------------------------------------------------------------------------------------------
TTRUN_DIR="${TTRUN_DIR:-/etc/ttop}"
TTRUN_PY="${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py"
MPIRUN=$(command -v mpirun-ulfm || command -v mpirun || echo mpirun)
RESOLVED_HOSTS=$(awk 'NF && $1 !~ /^#/ {printf "%s,", $1}' "${TTRUN_DIR}/hostfile" | sed 's/,$//')
NUM_HOSTS=$(tr ',' '\n' <<< "${RESOLVED_HOSTS}" | grep -c . || true)
[ "${NUM_HOSTS}" -eq 4 ] || { echo "expected 4 hosts in ${TTRUN_DIR}/hostfile for 16 stages, got ${NUM_HOSTS}: ${RESOLVED_HOSTS}" >&2; exit 2; }
if [ "${LAUNCH}" = binding ]; then
  # Ring order from the CI rankfile (first occurrence of each host), as the SC4 DC-validation leg derives it.
  RING_HOSTS=$(sed -n 's/^rank[[:space:]]*[0-9]*=\([^[:space:]]*\).*/\1/p' "${TTRUN_DIR}/rankfile" 2>/dev/null | awk '!seen[$0]++' | paste -sd, -)
  [ "$(tr ',' '\n' <<< "${RING_HOSTS}" | grep -c . || true)" -eq 4 ] || {
    echo "MATRIX_CI_LAUNCH=binding: ${TTRUN_DIR}/rankfile does not name 4 hosts (got '${RING_HOSTS}')" >&2; exit 2;
  }
fi

# ---- work dir on the shared /ci volume: every host writes its rank's timing CSV, the producer reads rank 0 and 15 ---
# The launch: ${WORK}/{runner.log, ranklogs/, timing/, shutdown.log[, binding.yaml]}; per row: ${WORK}/c<cached>/producer.log.
OUT_DIR="${PREFILL_SUMMARIES}/m3_prefill_matrix"
WORK="${OUT_DIR}/work"
TTRUN_CWD="${WORK}/ttrun-cwd"   # tt-run keeps its discovered placement here
TIMING="${WORK}/timing"          # one CSV per rank for the whole launch; each producer session continues the chunk index
RANKLOGS="${WORK}/ranklogs"
RUNNER_LOG="${WORK}/runner.log"
OUT="${OUT_DIR}/results_${STAGES}stage.jsonl"
mkdir -p "${TTRUN_CWD}" "${PREFILL_SUMMARIES}/perf"
[ "${DRY_RUN}" != 0 ] || mkdir -p "${TIMING}" "${RANKLOGS}"

COMMIT=$(git -C "${TT_METAL_HOME}" rev-parse --short HEAD 2>/dev/null || echo unknown)
echo "[matrix-ci] $(date) commit=${COMMIT} launch=${LAUNCH} hosts=${RESOLVED_HOSTS} stages=${STAGES}"
echo "[matrix-ci] cached=${CACHED_LIST[*]} new=${NEW} iters=${ITERS} users=${USERS} target_chunks=${TARGET_CHUNKS}" \
  "chunk=${MATRIX_CHUNK} layer_counts=${MATRIX_LAYER_COUNTS:-even} capacity=${KV_CAPACITY} expert_dtype=${EXPERT_DTYPE:-bf4}"
echo "[matrix-ci] work=${WORK} out=${OUT}"

printf -v CHILD_PYTHONPATH '%q' "${PYTHONPATH}"
MPI_COMMON="--bind-to none --tag-output --allow-run-as-root --wdir ${TT_METAL_HOME} -x PATH -x LD_LIBRARY_PATH"

# ---- the runner command (one launch for all rows) -----------------------------------------------------------------
# Runner env: what the Slurm flow splits across binding global_env + manifest, exported on EVERY rank (the bash -lc
# below runs on each rank, so these reach all 16; see the config-fingerprint check in prefill_runner.py).
RUNNER_ENV="export PREFILL_MODEL=minimax_m3; export PREFILL_NUM_LAYERS=${MATRIX_NUM_LAYERS};"
RUNNER_ENV+=" export PREFILL_FABRIC_MODE=2d; export PREFILL_SP=2; export PREFILL_TP=4;"
RUNNER_ENV+=" export PREFILL_MAX_SEQ_LEN=${KV_CAPACITY}; export PREFILL_NUM_USERS=${RUNNER_USERS};"
RUNNER_ENV+=" export PREFILL_CHUNK_SIZE=${MATRIX_CHUNK}; export PREFILL_H2D_SERVICE_ID=ds_prefill;"
RUNNER_ENV+=" export PREFILL_PP_D2D_FIFO_BYTES=32768; export M3_INDEX_CACHE_BF16=0; export EXPERT_DTYPE=${EXPERT_DTYPE:-bf4};"
RUNNER_ENV+=" export PREFILL_SYNC_PER_CHUNK=1; export PREFILL_TIMING_DIR=$(printf '%q' "${TIMING}");"
RUNNER_ENV+=" export HF_MODEL=$(printf '%q' "${HF_MODEL}"); export TT_CACHE_PATH=$(printf '%q' "${TT_CACHE_PATH}");"
[ -z "${MATRIX_LAYER_COUNTS}" ] || RUNNER_ENV+=" export PREFILL_PP_LAYER_COUNTS=${MATRIX_LAYER_COUNTS};"

PLACEMENT=""
if [ "${LAUNCH}" = discovery ]; then
  TTRUN_ARGS=(--force-rediscovery --mesh-graph-descriptor "${PKG_DIR}/quad_bh_galaxy_16stage_2x4_z_chain.textproto"
    --hosts "${RESOLVED_HOSTS}")
else
  BINDING="${WORK}/binding_16stage_quad.yaml"
  esc() { printf '%s' "$1" | sed 's/[&#\\]/\\&/g'; }
  if [ "${DRY_RUN}" = 0 ]; then   # the binding pins the KV capacity
    sed -e "s#@MAX_SEQ_LEN@#${KV_CAPACITY}#" -e "s#@PKG_DIR@#$(esc "${PKG_DIR}")#g" -e "s#@MANIFEST@##" \
      -e "s#@TT_CACHE_PATH@#$(esc "${TT_CACHE_PATH}")#" "${PKG_DIR}/binding_16stage_quad.yaml.in" \
      | grep -v 'PREFILL_MANIFEST: ""' > "${BINDING}"
  fi
  TTRUN_ARGS=(--rank-binding "${BINDING}")
  PLACEMENT="--host $(sed 's/\(,\|$\)/:4\1/g' <<< "${RING_HOSTS}") --map-by slot"
fi

RUNNER_CMD=(python3 "${TTRUN_PY}" --skip-executable-check --tcp-interface "${TCP_INTERFACE}" "${TTRUN_ARGS[@]}"
  --mpi-args "${PLACEMENT} ${MPI_COMMON} --output-filename ${RANKLOGS}/runner"
  bash -lc "cd '${TT_METAL_HOME}'; export PYTHONPATH=${CHILD_PYTHONPATH}; export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0; \
    export PYTHONUNBUFFERED=1; export LOGURU_LEVEL=INFO; ${RUNNER_ENV} \
    exec python3 -m models.demos.common.prefill.runners.prefill_runner")
RUNNER_PID=""
R0_HOST=""
ROW_C=""; ROW_DIR=""   # the current row, set by set_row

set_row() {   # set_row <cached>
  ROW_C=$1
  ROW_DIR="${WORK}/c${ROW_C}"
  [ "${DRY_RUN}" != 0 ] || mkdir -p "${ROW_DIR}"
}

# Run one command string on rank 0's host (the producer and the shutdown sentinel attach to rank 0's H2D service).
on_rank0() {
  "${MPIRUN}" -np 1 --host "${R0_HOST}:1" --bind-to none --allow-run-as-root -x PATH -x LD_LIBRARY_PATH \
    bash -lc "cd '${TT_METAL_HOME}'; export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0; export PYTHONUNBUFFERED=1; $1"
}
producer_cmd() {   # for the current row
  echo "env $(matrix_producer_env "${KV_CAPACITY}") PREFILL_H2D_CONNECT_TIMEOUT=120 \
    timeout ${PRODUCER_TIMEOUT_SECS} python3 $(printf '%q' "${PKG_DIR}/matrix_producer.py") --cached ${ROW_C} --new $(printf '%q' "${NEW}") \
    --iters ${ITERS} --timing-dir $(printf '%q' "${TIMING}") --out $(printf '%q' "${OUT}") --last-rank ${LAST} \
    --label ci_${STAGES}stage_c${MATRIX_CHUNK}${MATRIX_LAYER_COUNTS:+_split} --users ${USERS} \
    ${REQS:+--reqs ${REQS}} --target-chunks ${TARGET_CHUNKS} ${SKIP_IDLE:+--skip-idle}"
}
# Per-rank files of --output-filename: OpenMPI 5 writes <prefix>.<job>@<n>.<rank>.out/.err, OpenMPI 4 <prefix>/.../stdout.
rank_log_files() { find "$1" -type f \( -name '*.out' -o -name '*.err' -o -name 'std*' \) 2>/dev/null | sort; }
all_runner_logs() {   # tt-run's own stdout plus the per-rank files
  { cat "${RUNNER_LOG}"; rank_log_files "${RANKLOGS}" | xargs -r cat; } 2>/dev/null || true
}

if [ "${DRY_RUN}" != 0 ]; then
  R0_HOST="<rank0 host from the tt-run rankfile>"
  echo "[matrix-ci] DRY RUN -- runner, KV capacity ${KV_CAPACITY}:"; printf '  %q' "${RUNNER_CMD[@]}"; echo
  for c in "${CACHED_LIST[@]}"; do
    set_row "${c}"
    echo "[matrix-ci] DRY RUN -- row cached=${c} producer on ${R0_HOST}:"; echo "  $(producer_cmd)"
  done
  exit 0
fi

# ---- runner lifecycle --------------------------------------------------------------------------------------------
start_runner() {   # launch the runner, wait until all 16 ranks serve, set R0_HOST
  echo "[matrix-ci] $(date) launching runner, KV capacity ${KV_CAPACITY} -> ${RUNNER_LOG}"
  cd "${TTRUN_CWD}"
  setsid "${RUNNER_CMD[@]}" > "${RUNNER_LOG}" 2>&1 &
  RUNNER_PID=$!
  cd "${TT_METAL_HOME}"

  local fatal='could not fit|Timed out while waiting for active ethernet' n=0 logs
  for _ in $(seq 1 $((TABLE_WAIT_SECS / 5))); do
    logs=$(all_runner_logs)
    n=$(grep -oE '\[pp rank [0-9]+\] setup complete' <<< "${logs}" | sort -u | wc -l || true)
    [ "${n}" -ge "${STAGES}" ] && break
    if grep -qE "${fatal}" <<< "${logs}"; then
      echo "[matrix-ci] runner FAILED during bring-up:"; grep -E "${fatal}" <<< "${logs}" | head -5 || true
      exit 1
    fi
    kill -0 "${RUNNER_PID}" 2>/dev/null || { echo "[matrix-ci] runner exited during bring-up (${n}/${STAGES} ready)"; tail -n 60 "${RUNNER_LOG}"; exit 1; }
    sleep 5
  done
  [ "${n}" -ge "${STAGES}" ] || { echo "[matrix-ci] runner not ready after ${TABLE_WAIT_SECS}s (${n}/${STAGES} ranks)"; exit 1; }
  echo "[matrix-ci] $(date) runner ready (${n}/${STAGES} ranks)"

  if [ "${LAUNCH}" = discovery ]; then
    local rankfile
    rankfile=$(ls -t "${TTRUN_CWD}"/generated/ttrun/*/rankfile 2>/dev/null | head -1)
    [ -f "${rankfile}" ] || { echo "[matrix-ci] tt-run rankfile not found under ${TTRUN_CWD}/generated/ttrun/*/rankfile"; exit 1; }
    R0_HOST=$(awk '/^rank[[:space:]]+0=/ {h=$2; sub(/^0=/,"",h); print h; exit}' "${rankfile}")
    echo "[matrix-ci] tt-run placement (${rankfile}):"; cat "${rankfile}"
  else
    R0_HOST=${RING_HOSTS%%,*}
  fi
  [ -n "${R0_HOST}" ] || { echo "[matrix-ci] could not determine rank 0's host"; exit 1; }
  echo "[matrix-ci] rank 0 host: ${R0_HOST}"
}

stop_runner() {   # SHUTDOWN sentinel -> wait SHUTDOWN_WAIT_SECS -> kill the launcher's process group. 0 = clean exit, 1 = killed.
  [ -n "${RUNNER_PID}" ] && kill -0 "${RUNNER_PID}" 2>/dev/null || { RUNNER_PID=""; return 0; }
  local clean=0 slog="${WORK}/shutdown.log"
  if [ -n "${R0_HOST}" ]; then
    # The helper opens the devices, so it prints ~100 lines of UMD bring-up chatter: keep that in a file, not the job log.
    if on_rank0 "timeout 300 env $(matrix_producer_env "${KV_CAPACITY}") python3 $(printf '%q' "${PKG_DIR}/matrix_shutdown.py")" \
      > "${slog}" 2>&1; then
      echo "[matrix-ci] SHUTDOWN sent to the runner (log: ${slog})"
    else
      echo "[matrix-ci] SHUTDOWN helper failed; tail of ${slog}:"; tail -n 20 "${slog}" || true
    fi
    timeout "${SHUTDOWN_WAIT_SECS}" tail --pid="${RUNNER_PID}" -f /dev/null 2>/dev/null || true
  fi
  if kill -0 "${RUNNER_PID}" 2>/dev/null; then
    echo "[matrix-ci] runner still alive; killing its process group"
    clean=1
    kill -- "-${RUNNER_PID}" 2>/dev/null || true
    timeout 10 tail --pid="${RUNNER_PID}" -f /dev/null 2>/dev/null || kill -KILL -- "-${RUNNER_PID}" 2>/dev/null || true
  fi
  wait "${RUNNER_PID}" 2>/dev/null || true
  RUNNER_PID=""
  echo "[matrix-ci] $(date) runner stopped"
  return "${clean}"
}

cleanup() {
  local rc=$?
  stop_runner || true
  if [ -s "${OUT}" ]; then
    # Markdown in the layout of the tt-metal#57827 comments, so a run can be pasted there as is.
    python3 "${PKG_DIR}/matrix_table.py" "${OUT}" --markdown --csv "${OUT_DIR}/results_${STAGES}stage.csv" \
      > "${OUT_DIR}/tables.md" 2>&1 || true
    {
      echo "### MiniMax-M3 pipeline prefill perf matrix (16 stages, SC4)"
      echo
      echo "**Setup:** commit \`${COMMIT}\`, hosts \`${RESOLVED_HOSTS}\` (launch \`${LAUNCH}\`). ${STAGES} stages x [2,4]," \
        "chunk ${MATRIX_CHUNK}, layers per stage \`${MATRIX_LAYER_COUNTS:-even}\`, ${EXPERT_DTYPE:-bf4} experts, ${USERS} users," \
        "${ITERS} idle iterations, ${TARGET_CHUNKS} chunks per loaded cell, KV capacity ${KV_CAPACITY} for every row" \
        "(one runner launch). Cached rows \`${CACHED_LIST[*]}\`. Baseline and history: tt-metal#57827."
      cat "${OUT_DIR}/tables.md"
    } > "${PREFILL_SUMMARIES}/perf/m3_prefill_matrix_sc4.md"
    cat "${PREFILL_SUMMARIES}/perf/m3_prefill_matrix_sc4.md"
  else
    echo "[matrix-ci] no results in ${OUT}"
  fi
  if [ "${rc}" -ne 0 ] && [ -d "${RANKLOGS}" ]; then
    echo "==================== runner rank log tails (last row cached=${ROW_C:-none}) ===================="
    rank_log_files "${RANKLOGS}" | while read -r f; do
      echo "---- ${f#"${RANKLOGS}"/} ----"; tail -n 20 "$f" 2>/dev/null || true
    done
  fi
  exit "${rc}"
}
trap cleanup EXIT

# ---- one runner launch, one producer session per cached row ---------------------------------------------------------
start_runner
for c in "${CACHED_LIST[@]}"; do
  set_row "${c}"
  PLOG="${ROW_DIR}/producer.log"
  echo "[matrix-ci] $(date) row cached=${ROW_C}: producer -> ${PLOG}"
  on_rank0 "$(producer_cmd)" > "${PLOG}" 2>&1 || {
    rc=$?; echo "[matrix-ci] producer for cached=${ROW_C} FAILED (rc=${rc}); tail:"; tail -n 60 "${PLOG}"; exit "${rc}";
  }
  grep -E '\[matrix\] (CELL|LOADED cached|cached prefix)' "${PLOG}" | sed 's/.*\[matrix\]/[matrix]/' || true
done
echo "[matrix-ci] $(date) all rows done"
stop_runner || echo "[matrix-ci] the runner had to be killed after the last row (the results are complete)"
