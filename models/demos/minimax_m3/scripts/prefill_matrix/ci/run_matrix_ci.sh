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
# Like run_matrix.sh, every cached row gets its OWN runner launch with the KV capacity fitted to that row
# (cached + MAX_NEW): the dense layers gather the whole KV shard per chunk, so an over-sized cache would slow the
# smaller rows and make them incomparable with the #57827 baseline. A row whose runner does not exit on the SHUTDOWN
# sentinel is killed, and the remaining rows are skipped (the devices may be left wedged; there is no reset in here).
#
# Env (all optional): CACHED=0  NEW=5120,32768  ITERS=2  USERS=2  TARGET_CHUNKS=64  REQS  SKIP_IDLE=1
#   CHUNK=5120  LAYER_COUNTS=<per-stage comma list>  MAX_NEW=51200  EXPERT_DTYPE=bf4
#   HF_MODEL / TT_CACHE_PATH / PREFILL_TRACE_DIR (weka defaults)  MATRIX_CI_LAUNCH=discovery|binding
#   TABLE_WAIT_SECS=2400 (bring-up bound per launch)  PRODUCER_TIMEOUT_SECS=900 (per cached row; a hung row fails before the job timeout)
#   SHUTDOWN_WAIT_SECS=240 (how long a runner gets to exit after the SHUTDOWN sentinel before it is killed)
#   MATRIX_CI_DRY_RUN=1: print the launch commands and exit (no MPI, no devices; TTRUN_DIR may point at a fake
#   hostfile/rankfile).
#
# MATRIX_CI_LAUNCH:
#   discovery (default): tt-run new mode on the 16-stage z-chain mesh graph; generate_rank_bindings places the 16
#     meshes on whatever quad the job got (trays, TT_VISIBLE_DEVICES and host order included; MPI rank == mesh_id).
#     The first launch forces the discovery, later rows reuse tt-run's cached placement (same hosts, same graph).
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
# Per row: ${WORK}/c<cached>/{runner.log, ranklogs/, timing/, producer.log, shutdown.log[, binding.yaml]}.
OUT_DIR="${PREFILL_SUMMARIES}/m3_prefill_matrix"
WORK="${OUT_DIR}/work"
TTRUN_CWD="${WORK}/ttrun-cwd"   # shared by all launches: tt-run keeps its discovered placement here
OUT="${OUT_DIR}/results_${STAGES}stage.jsonl"
mkdir -p "${TTRUN_CWD}" "${PREFILL_SUMMARIES}/perf"

COMMIT=$(git -C "${TT_METAL_HOME}" rev-parse --short HEAD 2>/dev/null || echo unknown)
echo "[matrix-ci] $(date) commit=${COMMIT} launch=${LAUNCH} hosts=${RESOLVED_HOSTS} stages=${STAGES}"
echo "[matrix-ci] cached=${CACHED_LIST[*]} new=${NEW} iters=${ITERS} users=${USERS} target_chunks=${TARGET_CHUNKS}" \
  "chunk=${MATRIX_CHUNK} layer_counts=${MATRIX_LAYER_COUNTS:-even} capacity=cached+${MAX_NEW} per row expert_dtype=${EXPERT_DTYPE:-bf4}"
echo "[matrix-ci] work=${WORK} out=${OUT}"

printf -v CHILD_PYTHONPATH '%q' "${PYTHONPATH}"
MPI_COMMON="--bind-to none --tag-output --allow-run-as-root --wdir ${TT_METAL_HOME} -x PATH -x LD_LIBRARY_PATH"

# ---- per-row state, set by set_row ---------------------------------------------------------------------------------
ROW_C=""; ROW_CAP=""; ROW_DIR=""; ROW_TIMING=""; ROW_RANKLOGS=""; ROW_RUNNER_LOG=""; RUNNER_CMD=()
LAUNCHES=0       # runner launches so far (the first one forces tt-run discovery)
RUNNER_PID=""
R0_HOST=""

set_row() {   # set_row <cached>: row paths, KV capacity and the runner command for this row
  ROW_C=$1
  ROW_CAP=$((ROW_C + MAX_NEW))
  ROW_DIR="${WORK}/c${ROW_C}"
  ROW_TIMING="${ROW_DIR}/timing"
  ROW_RANKLOGS="${ROW_DIR}/ranklogs"
  ROW_RUNNER_LOG="${ROW_DIR}/runner.log"
  [ "${DRY_RUN}" != 0 ] || mkdir -p "${ROW_TIMING}" "${ROW_RANKLOGS}"

  # Runner env: what the Slurm flow splits across binding global_env + manifest, exported on EVERY rank (the bash -lc
  # below runs on each rank, so these reach all 16; see the config-fingerprint check in prefill_runner.py).
  local env="export PREFILL_MODEL=minimax_m3; export PREFILL_NUM_LAYERS=${MATRIX_NUM_LAYERS};"
  env+=" export PREFILL_FABRIC_MODE=2d; export PREFILL_SP=2; export PREFILL_TP=4;"
  env+=" export PREFILL_MAX_SEQ_LEN=${ROW_CAP}; export PREFILL_NUM_USERS=${RUNNER_USERS};"
  env+=" export PREFILL_CHUNK_SIZE=${MATRIX_CHUNK}; export PREFILL_H2D_SERVICE_ID=ds_prefill;"
  env+=" export PREFILL_PP_D2D_FIFO_BYTES=32768; export M3_INDEX_CACHE_BF16=0; export EXPERT_DTYPE=${EXPERT_DTYPE:-bf4};"
  env+=" export PREFILL_SYNC_PER_CHUNK=1; export PREFILL_TIMING_DIR=$(printf '%q' "${ROW_TIMING}");"
  env+=" export HF_MODEL=$(printf '%q' "${HF_MODEL}"); export TT_CACHE_PATH=$(printf '%q' "${TT_CACHE_PATH}");"
  [ -z "${MATRIX_LAYER_COUNTS}" ] || env+=" export PREFILL_PP_LAYER_COUNTS=${MATRIX_LAYER_COUNTS};"

  local ttrun_args=() placement=""
  if [ "${LAUNCH}" = discovery ]; then
    ttrun_args=(--mesh-graph-descriptor "${PKG_DIR}/quad_bh_galaxy_16stage_2x4_z_chain.textproto" --hosts "${RESOLVED_HOSTS}")
    [ "${LAUNCHES}" -gt 0 ] || ttrun_args=(--force-rediscovery "${ttrun_args[@]}")
  else
    local binding="${ROW_DIR}/binding_16stage_quad.yaml"
    esc() { printf '%s' "$1" | sed 's/[&#\\]/\\&/g'; }
    if [ "${DRY_RUN}" = 0 ]; then   # the binding pins this row's KV capacity
      sed -e "s#@MAX_SEQ_LEN@#${ROW_CAP}#" -e "s#@PKG_DIR@#$(esc "${PKG_DIR}")#g" -e "s#@MANIFEST@##" \
        -e "s#@TT_CACHE_PATH@#$(esc "${TT_CACHE_PATH}")#" "${PKG_DIR}/binding_16stage_quad.yaml.in" \
        | grep -v 'PREFILL_MANIFEST: ""' > "${binding}"
    fi
    ttrun_args=(--rank-binding "${binding}")
    placement="--host $(sed 's/\(,\|$\)/:4\1/g' <<< "${RING_HOSTS}") --map-by slot"
  fi

  RUNNER_CMD=(python3 "${TTRUN_PY}" --skip-executable-check --tcp-interface "${TCP_INTERFACE}" "${ttrun_args[@]}"
    --mpi-args "${placement} ${MPI_COMMON} --output-filename ${ROW_RANKLOGS}/runner"
    bash -lc "cd '${TT_METAL_HOME}'; export PYTHONPATH=${CHILD_PYTHONPATH}; export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0; \
      export PYTHONUNBUFFERED=1; export LOGURU_LEVEL=INFO; ${env} \
      exec python3 -m models.demos.common.prefill.runners.prefill_runner")
}

# Run one command string on rank 0's host (the producer and the shutdown sentinel attach to rank 0's H2D service).
on_rank0() {
  "${MPIRUN}" -np 1 --host "${R0_HOST}:1" --bind-to none --allow-run-as-root -x PATH -x LD_LIBRARY_PATH \
    bash -lc "cd '${TT_METAL_HOME}'; export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0; export PYTHONUNBUFFERED=1; $1"
}
producer_cmd() {   # for the current row
  echo "env $(matrix_producer_env "${ROW_CAP}") PREFILL_H2D_CONNECT_TIMEOUT=120 \
    timeout ${PRODUCER_TIMEOUT_SECS} python3 $(printf '%q' "${PKG_DIR}/matrix_producer.py") --cached ${ROW_C} --new $(printf '%q' "${NEW}") \
    --iters ${ITERS} --timing-dir $(printf '%q' "${ROW_TIMING}") --out $(printf '%q' "${OUT}") --last-rank ${LAST} \
    --label ci_${STAGES}stage_c${MATRIX_CHUNK}${MATRIX_LAYER_COUNTS:+_split} --users ${USERS} \
    ${REQS:+--reqs ${REQS}} --target-chunks ${TARGET_CHUNKS} ${SKIP_IDLE:+--skip-idle}"
}
# Per-rank files of --output-filename: OpenMPI 5 writes <prefix>.<job>@<n>.<rank>.out/.err, OpenMPI 4 <prefix>/.../stdout.
rank_log_files() { find "$1" -type f \( -name '*.out' -o -name '*.err' -o -name 'std*' \) 2>/dev/null | sort; }
all_runner_logs() {   # tt-run's own stdout plus the per-rank files of the current row
  { cat "${ROW_RUNNER_LOG}"; rank_log_files "${ROW_RANKLOGS}" | xargs -r cat; } 2>/dev/null || true
}

if [ "${DRY_RUN}" != 0 ]; then
  R0_HOST="<rank0 host from the tt-run rankfile>"
  for c in "${CACHED_LIST[@]}"; do
    set_row "${c}"
    echo "[matrix-ci] DRY RUN -- row cached=${c} capacity=${ROW_CAP} runner:"; printf '  %q' "${RUNNER_CMD[@]}"; echo
    echo "[matrix-ci] DRY RUN -- row cached=${c} producer on ${R0_HOST}:"; echo "  $(producer_cmd)"
    LAUNCHES=$((LAUNCHES + 1))
  done
  exit 0
fi

# ---- runner lifecycle --------------------------------------------------------------------------------------------
start_runner() {   # launch the current row's runner, wait until all 16 ranks serve, set R0_HOST
  echo "[matrix-ci] $(date) row cached=${ROW_C}: launching runner, KV capacity ${ROW_CAP} -> ${ROW_RUNNER_LOG}"
  cd "${TTRUN_CWD}"
  setsid "${RUNNER_CMD[@]}" > "${ROW_RUNNER_LOG}" 2>&1 &
  RUNNER_PID=$!
  cd "${TT_METAL_HOME}"
  LAUNCHES=$((LAUNCHES + 1))

  local fatal='could not fit|Timed out while waiting for active ethernet' n=0 logs
  for _ in $(seq 1 $((TABLE_WAIT_SECS / 5))); do
    logs=$(all_runner_logs)
    n=$(grep -oE '\[pp rank [0-9]+\] setup complete' <<< "${logs}" | sort -u | wc -l || true)
    [ "${n}" -ge "${STAGES}" ] && break
    if grep -qE "${fatal}" <<< "${logs}"; then
      echo "[matrix-ci] runner FAILED during bring-up:"; grep -E "${fatal}" <<< "${logs}" | head -5 || true
      exit 1
    fi
    kill -0 "${RUNNER_PID}" 2>/dev/null || { echo "[matrix-ci] runner exited during bring-up (${n}/${STAGES} ready)"; tail -n 60 "${ROW_RUNNER_LOG}"; exit 1; }
    sleep 5
  done
  [ "${n}" -ge "${STAGES}" ] || { echo "[matrix-ci] runner not ready after ${TABLE_WAIT_SECS}s (${n}/${STAGES} ranks)"; exit 1; }
  echo "[matrix-ci] $(date) runner ready (${n}/${STAGES} ranks)"

  if [ "${LAUNCH}" = discovery ]; then
    local rankfile
    rankfile=$(ls -t "${TTRUN_CWD}"/generated/ttrun/*/rankfile 2>/dev/null | head -1)
    [ -f "${rankfile}" ] || { echo "[matrix-ci] tt-run rankfile not found under ${TTRUN_CWD}/generated/ttrun/*/rankfile"; exit 1; }
    R0_HOST=$(awk '/^rank[[:space:]]+0=/ {h=$2; sub(/^0=/,"",h); print h; exit}' "${rankfile}")
    [ "${LAUNCHES}" -gt 1 ] || { echo "[matrix-ci] tt-run placement (${rankfile}):"; cat "${rankfile}"; }
  else
    R0_HOST=${RING_HOSTS%%,*}
  fi
  [ -n "${R0_HOST}" ] || { echo "[matrix-ci] could not determine rank 0's host"; exit 1; }
  echo "[matrix-ci] rank 0 host: ${R0_HOST}"
}

stop_runner() {   # SHUTDOWN sentinel -> wait SHUTDOWN_WAIT_SECS -> kill the launcher's process group. 0 = clean exit, 1 = killed.
  [ -n "${RUNNER_PID}" ] && kill -0 "${RUNNER_PID}" 2>/dev/null || { RUNNER_PID=""; return 0; }
  local clean=0 slog="${ROW_DIR}/shutdown.log"
  if [ -n "${R0_HOST}" ]; then
    # The helper opens the devices, so it prints ~100 lines of UMD bring-up chatter: keep that in a file, not the job log.
    if on_rank0 "timeout 300 env $(matrix_producer_env "${ROW_CAP}") python3 $(printf '%q' "${PKG_DIR}/matrix_shutdown.py")" \
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
  echo "[matrix-ci] $(date) row cached=${ROW_C}: runner stopped"
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
        "${ITERS} idle iterations, ${TARGET_CHUNKS} chunks per loaded cell, KV capacity cached + ${MAX_NEW} (one runner" \
        "launch per row). Cached rows \`${CACHED_LIST[*]}\`. Baseline and history: tt-metal#57827."
      cat "${OUT_DIR}/tables.md"
    } > "${PREFILL_SUMMARIES}/perf/m3_prefill_matrix_sc4.md"
    cat "${PREFILL_SUMMARIES}/perf/m3_prefill_matrix_sc4.md"
  else
    echo "[matrix-ci] no results in ${OUT}"
  fi
  if [ "${rc}" -ne 0 ] && [ -n "${ROW_RANKLOGS}" ] && [ -d "${ROW_RANKLOGS}" ]; then
    echo "==================== runner rank log tails (row cached=${ROW_C}) ===================="
    rank_log_files "${ROW_RANKLOGS}" | while read -r f; do
      echo "---- ${f#"${ROW_RANKLOGS}"/} ----"; tail -n 20 "$f" 2>/dev/null || true
    done
  fi
  exit "${rc}"
}
trap cleanup EXIT

# ---- one runner launch + one producer session per cached row -------------------------------------------------------
for i in "${!CACHED_LIST[@]}"; do
  set_row "${CACHED_LIST[$i]}"
  start_runner
  PLOG="${ROW_DIR}/producer.log"
  echo "[matrix-ci] $(date) row cached=${ROW_C}: producer -> ${PLOG}"
  on_rank0 "$(producer_cmd)" > "${PLOG}" 2>&1 || {
    rc=$?; echo "[matrix-ci] producer for cached=${ROW_C} FAILED (rc=${rc}); tail:"; tail -n 60 "${PLOG}"; exit "${rc}";
  }
  grep -E '\[matrix\] (CELL|LOADED cached|cached prefix)' "${PLOG}" | sed 's/.*\[matrix\]/[matrix]/' || true
  if ! stop_runner; then
    if [ "$((i + 1))" -lt "${#CACHED_LIST[@]}" ]; then
      echo "[matrix-ci] the runner had to be killed, so the devices may be wedged; skipping the remaining rows"
      exit 1
    fi
    echo "[matrix-ci] the last row's runner had to be killed (its results are complete)"
  fi
done
echo "[matrix-ci] $(date) all rows done"
