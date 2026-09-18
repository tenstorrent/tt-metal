#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Multi-galaxy DFlash drafter KV accuracy. Sibling of run_multirank_pcc.sh, which covers the VERIFIER's
# KVPE cache only -- that leg sets no PREFILL_DFLASH and cannot be extended in place, because it runs
# PREFILL_USE_TRACE=1 and the drafter path is not trace-captured (prefill_runner asserts on the pair).
#
# What this proves that the single-galaxy leg cannot: the drafter is built on the LAST pipeline rank while
# rank 0 builds and serializes the KV chunk table, so every drafter address in that table is a remote
# host's DRAM, reached through the all-gathered stage layout. The producer then reads those addresses back
# over UMD from a third process with no device of its own. Single-rank exercises none of that.
#
# Requires the drafter caches to be IN the merged table under pipeline parallelism -- see
# populate_kv_chunk_address_table_dflash's stage_layout path. Against a build without it the runner logs
# "DFlash drafter caches are NOT in the KV chunk table" and this script fails on the missing configs
# rather than silently reporting the verifier number alone.
set -euo pipefail

MODEL="${1:-kimi27}"
CONFIG="${2:-sc4}"

: "${TT_METAL_HOME:?TT_METAL_HOME must be set}"
export PYTHONPATH="${TT_METAL_HOME}"
MANIFEST_DIR="${TT_METAL_HOME}/models/demos/deepseek_v3_d_p/tt/runners/manifests"
MGD_DIR="${TT_METAL_HOME}/models/demos/common/prefill/runners/topology_configuration/ci"

# Two lengths, not one. The request runs to MAX_SEQ_LEN so the drafter is timed at the depth the
# verifier legs are timed at, while PCC is gated on GOLDEN_LEN: the drafter golden spans 55k and every
# position past it is unverifiable. Past the trace's own tokens the producer's pool repeats them, which
# makes the tail a throughput workload and nothing else -- it must never be PCC'd.
CHUNK_SIZE=5120
GOLDEN_LEN=56320
MAX_SEQ_LEN=256000
REAL_CHUNKS=$((MAX_SEQ_LEN / CHUNK_SIZE))
# Start, the last chunk wholly inside the golden, midpoint, end. The window that the throughput probe
# averages over runs forward from each of these, so the first three carry a rate and the last a latency.
PROBE_CHUNKS="0,$((GOLDEN_LEN / CHUNK_SIZE - 1)),$((REAL_CHUNKS / 2 - 1)),$((REAL_CHUNKS - 1))"
# Verifier threshold is the sibling KV leg's value for this trace and depth -- run_multirank_pcc.sh gates
# the same cache on the same golden, so the two legs must not disagree on what a passing verifier is.
# The drafter gate is held at that same value and never above it: an unnormalized V carries the verifier's
# own error, so a tighter drafter gate fails whenever the verifier merely sits near its own floor rather
# than when the drafter regresses. K is normalized by RMSNorm+RoPE and clears both by a wide margin, so
# this number is effectively a V gate. The drafter module default (0.88, dflash_kv_validation.DEFAULT_PCC)
# is a K2.6 bring-up value that does not hold for K2.7 at full depth; override with PREFILL_DFLASH_PCC
# rather than editing the module default.
PCC_THRESHOLD=0.85
DFLASH_PCC_THRESHOLD="${PREFILL_DFLASH_PCC:-0.85}"

# The dflash manifest differs from the plain one by the knobs a drafter run needs, so the two legs cannot
# share a file. What it pins, and why, since JSON cannot say it:
#   PREFILL_USE_TRACE=0    -- the drafter path is not trace-captured (prefill_runner asserts the pair).
#   PREFILL_LAYER_ACK_D2H=1 -- each rank stands up its own LayerAckService from D2H device records.
#      Without it the non-first ranks take the host-ring branch and connect() to
#      /tt_prefill_layer_completion_ring_N with a HARD-CODED 30 s timeout, which a rank that finished
#      weight load minutes earlier blows through whenever the ranks skew (cold vs warm page cache differs
#      by 10+ min here). The sibling KV leg sets this for the same reason.
#   PREFILL_PP_D2D_FIFO_BYTES=4096 -- DFlash packs [hidden || drafter-partial] into a 2*H-wide
#      activation, so the pipeline handoff costs twice the L1 of a plain run and the FIFO has to shrink to
#      pay for it. Measured on 2 galaxies: 32768 overflows rank 1 by 27136 B ("Statically allocated
#      circular buffers in program N clash with L1 buffers ... L1 buffer allocated at 1536128 and static
#      circular buffer region ends at 1563264"), which caps the FIFO at 5632; 4096 is the next power of
#      two under it and is also the value a plain Kimi 2-galaxy run settled on. It fails on rank 1 at
#      MLA_START of layer 0, i.e. only once tokens flow -- never at init -- so a too-large value survives
#      weight load and warmup before killing the run.
#   PREFILL_NUM_USERS=1    -- unlike the verifier leg's 86 on sc4: the drafter cache is allocated at
#      max_seq_len x num_users and slot 0 is the only slot with a golden behind it, so extra slots buy no
#      coverage. Every one of these stays overridable -- the manifest is applied with setdefault.
case "${MODEL}" in
  kimi27) MANIFEST="${MANIFEST_DIR}/kimi27_dflash.json" ;;
  *)
    echo "unknown model key '${MODEL}' (expected kimi27 -- only Kimi-K2.7 ships a drafter)" >&2
    exit 2
    ;;
esac

MPIRUN=$(command -v mpirun-ulfm || command -v mpirun)
# ttrun writes the allocated hosts here in CI, rank 0 first -- the same file the sibling KV leg reads.
if [ -z "${PREFILL_HOSTS:-}" ] && [ -f "${TTRUN_DIR:-/etc/ttop}/hostfile" ]; then
  PREFILL_HOSTS=$(awk 'NF {printf "%s,", $1}' "${TTRUN_DIR:-/etc/ttop}/hostfile" | sed 's/,$//')
fi
# No apostrophe in this message: bash treats a single quote inside ${var:?word} as opening a quote
# context even within double quotes, and the script fails to parse rather than to run.
HOSTS="${PREFILL_HOSTS:?PREFILL_HOSTS must list the rank hosts, rank 0 first (e.g. hostA,hostB)}"

# sc1 is single-galaxy. It takes the same STAGED code path as sc4 -- PREFILL_MOCK_MIGRATION all-gathers
# the stage layouts at any rank count -- so what it does not cover is the remote HOST, not the branch: one
# rank builds the table over its own DRAM and the producer reads it back over local PCIe. The rank count
# itself comes from the mesh-graph descriptor below, so this only validates the key.
case "${CONFIG}" in
  sc1|sc2|sc4) ;;
  *) echo "unknown config '${CONFIG}' (expected sc1, sc2 or sc4)" >&2; exit 2 ;;
esac

# sc1/sc4 have model-specific CI descriptors; sc2 has none, so fall back to the shared 2-galaxy one the
# manual pipeline bindings already use (it is model-agnostic -- an 8x4 RING mesh per galaxy).
MGD="${MGD_DIR}/${MODEL}_${CONFIG}_mgd.textproto"
if [ ! -f "${MGD}" ] && [ "${CONFIG}" = sc2 ]; then
  MGD="${TT_METAL_HOME}/models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_2galaxy_connected_mesh_graph_descriptor.textproto"
fi
if [ ! -f "${MGD}" ]; then
  echo "missing mesh-graph descriptor for ${MODEL}/${CONFIG}: ${MGD}" >&2
  exit 2
fi

# The table must live on storage every rank AND the device-less producer can read. prefill_producer
# rejects a per-host path outright for world_size > 1 (_require_shared_table_path), so fail here with a
# clearer message than "validators on other hosts cannot read rank 0's table".
# In CI the pipeline supplies PREFILL_SUMMARIES; derive the scratch root from it the way the sibling KV
# leg derives PIPELINE_DIR, so the workflow needs no dflash-specific variable. Interactive runs set
# PREFILL_SHARED_DIR directly.
if [ -z "${PREFILL_SHARED_DIR:-}" ] && [ -n "${PREFILL_SUMMARIES:-}" ]; then
  PREFILL_SHARED_DIR="${PREFILL_SUMMARIES/prefill_summaries/prefill_dflash_kv}"
fi
: "${PREFILL_SHARED_DIR:?PREFILL_SHARED_DIR must be set to shared/NFS scratch (e.g. /data/\$USER/dflash_ci)}"
case "${PREFILL_SHARED_DIR}" in
  /tmp/*|/dev/shm/*|/run/*|/var/tmp/*)
    echo "PREFILL_SHARED_DIR=${PREFILL_SHARED_DIR} is per-host storage; point it at NFS" >&2; exit 2 ;;
esac
mkdir -p "${PREFILL_SHARED_DIR}"
MR_DIR=$(mktemp -d "${PREFILL_SHARED_DIR}/${MODEL}_dflash_XXXXXX")
TABLE_PATH="${MR_DIR}/kv_chunk_table.pb"
PCC_DIR="${MR_DIR}/pcc_verdict"
RANKLOGS="${MR_DIR}/ranklogs"
TIMING_DIR="${MR_DIR}/timing"
PRODUCER_LOG="${MR_DIR}/producer.log"
mkdir -p "${PCC_DIR}" "${RANKLOGS}" "${TIMING_DIR}"
# ttrun writes generated/ttrun/<id>/ relative to its own CWD and hands the ranks that path as an
# absolute one. Launched from TT_METAL_HOME, rank 0 lands rank_bindings.yaml on its own node, where
# the launcher -- a different machine in CI -- cannot see it and calls Phase 1 silently failed.
# MR_DIR is the shared scratch both sides mount, so ttrun bookkeeping belongs under it.
TTRUN_CWD="${MR_DIR}/ttrun-cwd"
mkdir -p "${TTRUN_CWD}"

TTRUN_PY="${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py"
TCP_IFACE="${PREFILL_TCP_IFACE:-ens5f0np0}"

cleanup() {
  if [ -n "${RUNNER_PID:-}" ] && kill -0 "${RUNNER_PID}" 2>/dev/null; then
    kill "${RUNNER_PID}" 2>/dev/null || true
    wait "${RUNNER_PID}" 2>/dev/null || true
  fi
  # ttrun spawns prterun, which is NOT reaped by killing ttrun: it reparents to init and its ranks sit in
  # the unbounded request loop holding the mesh, so a failed producer leaks the box to the next CI job.
  # MR_DIR is mktemp-unique and appears in prterun's --output-filename, so this matches only this run.
  if pgrep -f "prterun.*${MR_DIR}" >/dev/null 2>&1; then
    pkill -TERM -f "prterun.*${MR_DIR}" 2>/dev/null || true
    for _ in $(seq 1 30); do pgrep -f "prterun.*${MR_DIR}" >/dev/null 2>&1 || break; sleep 1; done
    pkill -KILL -f "prterun.*${MR_DIR}" 2>/dev/null || true
  fi
  echo "==================== PCC verdicts (PROD_RC=${PROD_RC:-<unset>}) ===================="
  for f in "${PCC_DIR}"/rank*.json; do
    [ -e "$f" ] || { echo "no verdict files under ${PCC_DIR}"; break; }
    echo "$(basename "$f"): $(cat "$f")"
  done
  # The drafter minimum is printed but NOT recorded in rank*.json (_write_pcc_verdict carries only
  # per_cache kvpe), so scrape it from the logs or a pass/fail here is unattributable. The producer runs
  # under its own mpirun, not ttrun, so its output is in PRODUCER_LOG; RANKLOGS holds runner ranks only.
  echo "==================== drafter KV PCC ===================="
  grep -h -E "drafter KV PCC|min over all|kv_cache_pcc_complete" "${PRODUCER_LOG}" 2>/dev/null | tail -20 \
    || echo "no drafter PCC lines -- were the dflash_* configs in the table?"
  # Timing is the other half of this leg and it is only in the rank logs, so summarize before MR_DIR
  # goes away. Non-fatal: a perf summary that cannot be built must not turn a passing PCC run red.
  if [ -d "${RANKLOGS}" ]; then
    python3 "${TT_METAL_HOME}/models/demos/common/prefill/runners/ci/summarize_ci_run.py" \
      --ranklogs "${RANKLOGS}" --timing-dir "${TIMING_DIR}" --real-chunks "${REAL_CHUNKS}" \
      --chunk-size "${CHUNK_SIZE}" \
      --probe-chunks "${PROBE_CHUNKS}" \
      --summary-name "${MODEL}_dflash_${CONFIG}" \
      || echo "summary generation failed (non-fatal)"
  fi
  if [ -n "${PREFILL_KEEP_RUN_DIR:-}" ]; then
    echo "keeping run dir (PREFILL_KEEP_RUN_DIR set): ${MR_DIR}"
  else
    rm -rf "${MR_DIR}"
  fi
}
# TERM and INT as well as EXIT: bash runs an EXIT-only trap when the script returns, but a signal that
# kills the shell outright skips it, so a CI cancel or a stray kill leaves the whole ttrun tree holding
# every reserved box. Killing prterun cascades to the remote prted daemons and their ranks; nothing in
# cleanup can reach the other hosts directly.
trap cleanup EXIT INT TERM

cd "${TTRUN_CWD}"
python3 "${TTRUN_PY}" \
  --skip-executable-check \
  --tcp-interface "${TCP_IFACE}" \
  --mesh-graph-descriptor "${MGD}" \
  --hosts "${HOSTS}" \
  --mpi-args "--bind-to none --tag-output --wdir ${TT_METAL_HOME} --output-filename ${RANKLOGS}/runner -x PATH -x LD_LIBRARY_PATH" \
  bash -lc "cd '${TT_METAL_HOME}'; \
    export PYTHONPATH='${TT_METAL_HOME}'; \
    export PYTHONUNBUFFERED=1; \
    export PREFILL_MANIFEST='${MANIFEST}'; \
    export PREFILL_CHUNK_SIZE=${CHUNK_SIZE}; \
    export PREFILL_MAX_SEQ_LEN=${MAX_SEQ_LEN}; \
    export PREFILL_TIMING_DIR='${TIMING_DIR}'; \
    export PREFILL_ENABLE_MIGRATION=1; \
    export PREFILL_MOCK_MIGRATION=1; \
    export PREFILL_MIGRATION_TABLE_PATH='${TABLE_PATH}'; \
    export PREFILL_MIGRATION_DEVICE_MAP_PATH=/tmp/dflash_kv_device_map.json; \
    export PREFILL_SYNC_PER_CHUNK=1; \
    export LOGURU_LEVEL=INFO; \
    exec python3 -m models.demos.common.prefill.runners.prefill_runner" &
RUNNER_PID=$!
cd "${TT_METAL_HOME}"

# Wait for the table. The REAL guard is runner liveness below -- a dead runner is detected in seconds --
# so the deadline only bounds a silent hang and must not be mistaken for a load-time budget. Sizing it
# to an expected weight-load time is how this fails: multi-host load is far slower than single-host
# (every rank pulls the same TTNN cache over the same NFS mount concurrently -- measured ~83 s/layer on
# two galaxies vs ~53 s/layer on one), so a 40-minute ceiling killed a run two minutes short of
# publishing. Default 2 h; override with PREFILL_TABLE_WAIT_S.
TABLE_WAIT_S="${PREFILL_TABLE_WAIT_S:-7200}"
table_deadline=$(( $(date +%s) + TABLE_WAIT_S ))
while [ ! -f "${TABLE_PATH}" ]; do
  kill -0 "${RUNNER_PID}" 2>/dev/null || { echo "runner exited before publishing the KV table"; wait "${RUNNER_PID}"; exit 1; }
  if [ "$(date +%s)" -ge "${table_deadline}" ]; then
    echo "KV table not published within ${TABLE_WAIT_S}s (runner still alive -- raise PREFILL_TABLE_WAIT_S if it was merely slow)"
    exit 1
  fi
  sleep 5
done

# Fail loudly on a build whose cross-stage merge still drops the drafter: without the dflash_* configs the
# producer reports only the verifier number and this leg would "pass" while testing nothing new.
#
# On a rank host, not here: the ttnn wheel is installed per-node over MPI, so the launcher has no bindings.
# It does not fail closed either -- PYTHONPATH's ttnn/ source directory carries no __init__.py, so an import
# there resolves to an empty namespace package and succeeds into a ttnn with no .experimental. The table
# lives on shared scratch, so any rank can read it.
CHECK_PY="${TTRUN_CWD}/check_table_configs.py"
cat > "${CHECK_PY}" <<'PY'
import sys, ttnn

table = ttnn.experimental.disaggregation.import_from_protobuf_file(sys.argv[1])
names = [table.config_name(i) for i in range(table.num_configs())]
dflash = [x for x in names if x.startswith("dflash_")]
print(f"[dflash-ci] table configs: {len(names)} total, {len(dflash)} drafter -> {dflash[:4]}...")
if not dflash:
    sys.exit("[dflash-ci] FATAL: merged table carries no dflash_* configs; the drafter half would be skipped")
PY
"${MPIRUN}" \
  --host "${HOSTS%%,*}:1" -n 1 --bind-to none --tag-output --allow-run-as-root \
  -x PATH -x LD_LIBRARY_PATH \
  bash -lc "cd '${TT_METAL_HOME}'; \
    export PYTHONPATH='${TT_METAL_HOME}'; \
    exec python3 '${CHECK_PY}' '${TABLE_PATH}'"

# The producer's rank order MUST match ttrun's actual placement, not PREFILL_HOSTS. ttrun assigns ranks
# from its own topology discovery and freely reorders the host list, so rank 0 of the RUNNER (the only
# rank that exports the H2D descriptor into its host-local /dev/shm) may not be the first host given.
# Producer rank 0 connects to that descriptor, so a mismatched order fails as
# "Timeout waiting for service descriptor file: /dev/shm/tt_h2d_stream_service_<id>.bin" -- which reads
# like a dead runner rather than a placement bug. Read the true mapping out of the generated rankfile.
#
# Ordering the --host list is NOT sufficient: OpenMPI pins rank 0 to the host mpirun was launched from
# and ignores the list order (verified -- "--host c10u20:1,c10u14:1" still put rank 0 on the local
# c10u14). So the rankfile is replayed through --map-by rankfile:file= to pin placement explicitly,
# with --host kept alongside because a rankfile alone is rejected as "host not allocated". A CI head
# node that happens to be ttrun's rank 0 masks this; a reversed assignment does not.
RANKFILE=$(ls -t "${TTRUN_CWD}"/generated/ttrun/*/rankfile 2>/dev/null | head -1)
[ -f "${RANKFILE}" ] || { echo "tt-run rankfile not found under ${TTRUN_CWD}/generated/ttrun/*/rankfile" >&2; exit 1; }
PRODUCER_HOSTS=$(awk '/^rank[[:space:]]+[0-9]+=/ {n=$2; sub(/=.*/,"",n); h=$2; sub(/^[0-9]+=/,"",h); sub(/[[:space:]].*/,"",h); print n" "h}' "${RANKFILE}" \
  | sort -n | awk '{printf "%s%s:1", (NR>1?",":""), $2}')
[ -n "${PRODUCER_HOSTS}" ] || { echo "failed to parse producer host order from ${RANKFILE}" >&2; exit 1; }
# RELATIVE path, deliberately: PRRTE's --map-by qualifier parser rejects an absolute one with
# "The map-by directive contains an unrecognized qualifier: file=/..." (while listing file= as valid,
# so the message points nowhere). It resolves against TT_METAL_HOME, which is both the launch CWD here
# and the --wdir below; the rankfile itself lives on shared scratch outside it, so copy it in.
mkdir -p "${TT_METAL_HOME}/generated/ttrun"
cp "${RANKFILE}" "${TT_METAL_HOME}/generated/ttrun/producer_rankfile"
RANKFILE_REL="generated/ttrun/producer_rankfile"
echo "producer host order from tt-run discovery: ${PRODUCER_HOSTS} (PREFILL_HOSTS was ${HOSTS})"

set +e
# --mca btl_tcp_if_include is REQUIRED, not tuning: without it MPI_Init never completes and every rank
# logs "applied manifest" then goes silent forever (no error). ttrun passes the same transport args to
# the runner above, so the producer must match them or only this leg hangs.
"${MPIRUN}" \
  --host "${PRODUCER_HOSTS}" --map-by "rankfile:file=${RANKFILE_REL}" --bind-to none --tag-output \
  --wdir "${TT_METAL_HOME}" \
  --mca btl self,tcp --mca btl_tcp_if_include "${TCP_IFACE}" \
  -x PATH -x LD_LIBRARY_PATH \
  bash -lc "cd '${TT_METAL_HOME}'; \
    export PYTHONPATH='${TT_METAL_HOME}'; \
    export PREFILL_PRODUCER_MANIFEST='${MANIFEST}'; \
    export PREFILL_CHUNK_SIZE=${CHUNK_SIZE}; \
    export PREFILL_MAX_SEQ_LEN=${MAX_SEQ_LEN}; \
    export PREFILL_PRODUCER_CHECK_PCC=1; \
    export PREFILL_PRODUCER_CHUNKS=${REAL_CHUNKS}; \
    export PREFILL_PCC_GOLDEN_LEN=${GOLDEN_LEN}; \
    export PREFILL_PRODUCER_MAX_REQUESTS=1; \
    export PREFILL_PRODUCER_MULTI_TURN_PROB=0; \
    export PREFILL_STANDALONE_CHUNKED_PCC=${PCC_THRESHOLD}; \
    export PREFILL_DFLASH_PCC=${DFLASH_PCC_THRESHOLD}; \
    export PREFILL_MIGRATION_TABLE_PATH='${TABLE_PATH}'; \
    export PREFILL_MIGRATION_DEVICE_MAP_PATH=/tmp/dflash_kv_device_map.json; \
    export PREFILL_PCC_SUMMARY_DIR='${PCC_DIR}'; \
    export LOGURU_LEVEL=INFO; \
    exec python3 -m models.demos.common.prefill.runners.prefill_producer" 2>&1 | tee "${PRODUCER_LOG}"
PROD_RC=${PIPESTATUS[0]}
set -e

# A rank whose drafter caches are not local skips the drafter check entirely, so a zero exit code only
# proves the verifier caches were compared. The completion line carries the drafter minimum iff it ran.
if [ "${PROD_RC}" -eq 0 ] && ! grep -q "min_dflash_pcc=" "${PRODUCER_LOG}"; then
  echo "no rank reported a drafter KV PCC: the dflash caches were never checked" >&2
  exit 1
fi

exit "${PROD_RC}"
