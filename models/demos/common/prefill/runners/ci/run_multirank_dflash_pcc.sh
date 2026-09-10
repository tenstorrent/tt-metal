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

# The drafter golden spans 55k, so the request is sized to it rather than to a perf depth: the drafter
# cache is allocated at max_seq_len x num_users and every position outside the golden is unverifiable.
CHUNK_SIZE=5120
GOLDEN_LEN=56320
MAX_SEQ_LEN=56320
NUM_USERS_DEFAULT=1

# D2D FIFO. DFlash packs [hidden || drafter-partial] into a 2*H-wide activation, so the pipeline handoff
# costs twice the L1 of a plain run and the FIFO has to shrink to pay for it. Measured on 2 galaxies:
# 32768 overflows rank 1 by 27136 B ("Statically allocated circular buffers in program N clash with L1
# buffers ... L1 buffer allocated at 1536128 and static circular buffer region ends at 1563264"), which
# caps the FIFO at 5632; 4096 is the next power of two under it and is also the value a plain Kimi
# 2-galaxy run settled on. It fails on rank 1 at MLA_START of layer 0, i.e. only once tokens flow --
# never at init -- so a too-large value survives weight load and warmup before killing the run.
# Verifier threshold matches the sibling KV leg for this trace/depth. The drafter's own default (0.88,
# dflash_kv_validation.DEFAULT_PCC) was calibrated during K2.6 bring-up and is NOT re-derived for K2.7 at
# full depth -- an unnormalized V tracks the verifier's own accuracy, which is ~0.879 here. Override with
# PREFILL_DFLASH_PCC once a K2.7 number is agreed rather than editing the module default.
PCC_THRESHOLD=0.877
DFLASH_PCC_THRESHOLD="${PREFILL_DFLASH_PCC:-0.86}"

case "${MODEL}" in
  kimi27)
    MANIFEST="${MANIFEST_DIR}/kimi27.json"
    HF_MODEL=/mnt/models/moonshotai/Kimi-K2_7-Code-dequantized
    DFLASH_MODEL=/mnt/models/blaze/closed_do_not_share/Kimi-K2.7-Code-DFlash
    # Prompt trace and drafter golden MUST come from the same tap: dflash_27_context_kv_55k's metadata
    # records tap_source=vllm-kimi-k27-codedebug-56320. Pairing a golden with a different prompt yields a
    # plausible-looking PCC in the 0.2-0.6 range rather than an error.
    TRACE_DIR=/mnt/models/deepseek-prefill-cache/golden/structured_traces/vllm-kimi-k27-codedebug-56320
    GOLDEN_KV_DIR=/mnt/models/deepseek-prefill-cache/golden/dflash_27_context_kv_55k
    ;;
  kimi26)
    MANIFEST="${MANIFEST_DIR}/kimi26.json"
    HF_MODEL=models/demos/deepseek_v3_d_p/reference/kimi_k2_6
    DFLASH_MODEL=/mnt/models/Kimi-K2.6-DFlash
    TRACE_DIR=/mnt/models/deepseek-prefill-cache/golden/structured_traces/kimi_debug_55k_vllm
    GOLDEN_KV_DIR=/mnt/models/deepseek-prefill-cache/golden/dflash_context_kv_55k_v3
    ;;
  *)
    echo "unknown model key '${MODEL}' (expected kimi26 or kimi27 -- only Kimi-K2.x ships a drafter)" >&2
    exit 2
    ;;
esac

# sc1 is single-galaxy: it still runs, but it exercises the local-tensor path, not the staged one.
case "${CONFIG}" in
  sc1) NUM_RANKS=1 ;;
  sc2) NUM_RANKS=2 ;;
  sc4) NUM_RANKS=4 ;;
  *) echo "unknown config '${CONFIG}' (expected sc1, sc2 or sc4)" >&2; exit 2 ;;
esac

# sc1/sc4 have model-specific CI descriptors; sc2 has none, so fall back to the shared 2-galaxy one the
# manual pipeline bindings already use (it is model-agnostic -- an 8x4 RING mesh per galaxy).
MGD="${MGD_DIR}/${MODEL}_${CONFIG}_mgd.textproto"
if [ ! -f "${MGD}" ] && [ "${CONFIG}" = sc2 ]; then
  MGD="${TT_METAL_HOME}/models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_2galaxy_connected_mesh_graph_descriptor.textproto"
fi
[ -f "${MGD}" ] || { echo "no mesh-graph descriptor for ${MODEL}/${CONFIG} at ${MGD}" >&2; exit 2; }
[ -d "${DFLASH_MODEL}" ] || { echo "drafter checkpoint not found: ${DFLASH_MODEL}" >&2; exit 2; }
[ -d "${GOLDEN_KV_DIR}" ] || { echo "drafter golden not found: ${GOLDEN_KV_DIR}" >&2; exit 2; }

# The table must live on storage every rank AND the device-less producer can read. prefill_producer
# rejects a per-host path outright for world_size > 1 (_require_shared_table_path), so fail here with a
# clearer message than "validators on other hosts cannot read rank 0's table".
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
mkdir -p "${PCC_DIR}" "${RANKLOGS}"

TTRUN_PY="${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py"
TCP_IFACE="${PREFILL_TCP_IFACE:-ens5f0np0}"
# No apostrophe in this message: bash treats a single quote inside ${var:?word} as opening a quote
# context even within double quotes, and the script fails to parse rather than to run.
HOSTS="${PREFILL_HOSTS:?PREFILL_HOSTS must list the rank hosts, rank 0 first (e.g. hostA,hostB)}"

cleanup() {
  if [ -n "${RUNNER_PID:-}" ] && kill -0 "${RUNNER_PID}" 2>/dev/null; then
    kill "${RUNNER_PID}" 2>/dev/null || true
    wait "${RUNNER_PID}" 2>/dev/null || true
  fi
  echo "==================== PCC verdicts (PROD_RC=${PROD_RC:-<unset>}) ===================="
  for f in "${PCC_DIR}"/rank*.json; do
    [ -e "$f" ] || { echo "no verdict files under ${PCC_DIR}"; break; }
    echo "$(basename "$f"): $(cat "$f")"
  done
  # The drafter minimum is printed but NOT recorded in rank*.json (_write_pcc_verdict carries only
  # per_cache kvpe), so scrape it from the logs or a pass/fail here is unattributable.
  echo "==================== drafter KV PCC ===================="
  grep -h -E "drafter KV PCC|min over all|kv_cache_pcc_complete" "${RANKLOGS}"/* 2>/dev/null | tail -20 \
    || echo "no drafter PCC lines -- were the dflash_* configs in the table?"
  rm -rf "${MR_DIR}"
}
trap cleanup EXIT

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
    export PREFILL_HF_MODEL='${HF_MODEL}'; \
    export PREFILL_FABRIC_MODE=2d_torus_xy; \
    export PREFILL_CHUNK_SIZE=${CHUNK_SIZE}; \
    export PREFILL_MAX_SEQ_LEN=${MAX_SEQ_LEN}; \
    export PREFILL_NUM_USERS=${PREFILL_NUM_USERS:-${NUM_USERS_DEFAULT}}; \
    export PREFILL_TRACE_DIR='${TRACE_DIR}'; \
    export PREFILL_DFLASH=1; \
    export DFLASH_HF_MODEL='${DFLASH_MODEL}'; \
    export PREFILL_DFLASH_GOLDEN_KV_DIR='${GOLDEN_KV_DIR}'; \
    export PREFILL_USE_TRACE=0; \
    # Each rank stands up its own LayerAckService from D2H device records. Without this the non-first
    # ranks take the host-ring branch and connect() to /tt_prefill_layer_completion_ring_N with a
    # HARD-CODED 30 s timeout -- which a rank that finished weight load minutes earlier blows through
    # whenever the ranks skew (cold vs warm page cache differs by 10+ min here). The sibling KV leg
    # sets this for the same reason.
    export PREFILL_LAYER_ACK_D2H=1; \
    export PREFILL_ENABLE_MIGRATION=1; \
    export PREFILL_MOCK_MIGRATION=1; \
    export PREFILL_MIGRATION_TABLE_PATH='${TABLE_PATH}'; \
    export PREFILL_MIGRATION_DEVICE_MAP_PATH=/tmp/dflash_kv_device_map.json; \
    export PREFILL_PP_D2D_FIFO_BYTES=${PREFILL_PP_D2D_FIFO_BYTES:-4096}; \
    export PREFILL_SYNC_PER_CHUNK=1; \
    export LOGURU_LEVEL=INFO; \
    exec python3 -m models.demos.common.prefill.runners.prefill_runner" &
RUNNER_PID=$!

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
python3 - "${TABLE_PATH}" <<'PY'
import sys, ttnn
t = ttnn.experimental.disaggregation.import_from_protobuf_file(sys.argv[1])
names = [t.config_name(i) for i in range(t.num_configs())]
n = sum(1 for x in names if x.startswith("dflash_"))
print(f"[dflash-ci] table configs: {len(names)} total, {n} drafter -> {[x for x in names if x.startswith('dflash_')][:4]}...")
if n == 0:
    sys.exit("[dflash-ci] FATAL: merged table carries no dflash_* configs; the drafter half would be skipped")
PY

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
RANKFILE=$(ls -t "${TT_METAL_HOME}"/generated/ttrun/*/rankfile 2>/dev/null | head -1)
[ -f "${RANKFILE}" ] || { echo "tt-run rankfile not found under ${TT_METAL_HOME}/generated/ttrun/*/rankfile" >&2; exit 1; }
PRODUCER_HOSTS=$(awk '/^rank[[:space:]]+[0-9]+=/ {n=$2; sub(/=.*/,"",n); h=$2; sub(/^[0-9]+=/,"",h); sub(/[[:space:]].*/,"",h); print n" "h}' "${RANKFILE}" \
  | sort -n | awk '{printf "%s%s:1", (NR>1?",":""), $2}')
[ -n "${PRODUCER_HOSTS}" ] || { echo "failed to parse producer host order from ${RANKFILE}" >&2; exit 1; }
# RELATIVE path, deliberately: PRRTE's --map-by qualifier parser rejects an absolute one with
# "The map-by directive contains an unrecognized qualifier: file=/..." (while listing file= as valid,
# so the message points nowhere). Relative to --wdir, which is TT_METAL_HOME for both launches.
RANKFILE_REL="${RANKFILE#"${TT_METAL_HOME}"/}"
echo "producer host order from tt-run discovery: ${PRODUCER_HOSTS} (PREFILL_HOSTS was ${HOSTS})"

MPIRUN=$(command -v mpirun-ulfm || command -v mpirun)
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
    export PREFILL_HF_MODEL='${HF_MODEL}'; \
    export PREFILL_CHUNK_SIZE=${CHUNK_SIZE}; \
    export PREFILL_MAX_SEQ_LEN=${MAX_SEQ_LEN}; \
    export PREFILL_NUM_USERS=${PREFILL_NUM_USERS:-${NUM_USERS_DEFAULT}}; \
    export PREFILL_TRACE_DIR='${TRACE_DIR}'; \
    export PREFILL_DFLASH_GOLDEN_KV_DIR='${GOLDEN_KV_DIR}'; \
    export PREFILL_PRODUCER_CHECK_PCC=1; \
    export PREFILL_PRODUCER_CHUNKS=$((GOLDEN_LEN / CHUNK_SIZE)); \
    export PREFILL_PRODUCER_MAX_REQUESTS=1; \
    export PREFILL_PRODUCER_MULTI_TURN_PROB=0; \
    export PREFILL_STANDALONE_CHUNKED_PCC=${PCC_THRESHOLD}; \
    export PREFILL_DFLASH_PCC=${DFLASH_PCC_THRESHOLD}; \
    export PREFILL_MIGRATION_TABLE_PATH='${TABLE_PATH}'; \
    export PREFILL_MIGRATION_DEVICE_MAP_PATH=/tmp/dflash_kv_device_map.json; \
    export PREFILL_PCC_SUMMARY_DIR='${PCC_DIR}'; \
    export LOGURU_LEVEL=INFO; \
    exec python3 -m models.demos.common.prefill.runners.prefill_producer"
PROD_RC=$?
set -e
exit "${PROD_RC}"
