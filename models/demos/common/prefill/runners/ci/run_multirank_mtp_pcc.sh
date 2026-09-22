#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Multi-galaxy GLM-5.2 MTP prefill KV accuracy. Sibling of run_multirank_pcc.sh, which gates the TRUNK's
# two caches only -- that leg sets no PREFILL_MTP_LEVELS and cannot be extended in place:
#   * its shape comes from the manifest (manifest_env PREFILL_MAX_SEQ_LEN / PREFILL_NUM_USERS) and the MTP
#     manifests deliberately carry neither, so they stay usable by the pytest e2e scenarios unchanged;
#   * it runs the release-gating glm52 entry, whose env must not grow a branch on a new feature;
#   * MTP PCC has to stop at the TAIL trace's 56320 rows, a different golden from the trunk's.
# MODEL stays `glm52` so both existing CI mesh-graph descriptors are reused as-is; the MTP-ness of the run
# is this script plus the manifest it picks, not a new model key. The descriptors are per-SKU since #57093,
# so the MGD path is ${CONFIG} alone and MODEL does not appear in it.
#
# What only the 4-rank shape covers -- none of it has ever executed, on any machine:
#   * the MTP layer re-split. compute_layer_split balances num_layers+K EQUIVALENTS, so MTP4 on 4 ranks is
#     a 22/20/20/16 trunk at starts 0/22/42/62, not the 18/20/20/20 the trunk-only legs run.
#   * the D2D union-embedding transport. The last rank has no embedding table, so the chunk's union
#     embedding rides under the hidden: d2d_activation_rows goes 5120 -> 10496 for chunk 5120 at sp=8,
#     K=4 (+105%). If rank 1 dies at MLA_START of layer 0 with a static-CB/L1 clash, the knob is
#     PREFILL_PP_D2D_FIFO_BYTES (default 256; DFlash's 2x-wide activation had to cap it at 5632).
#   * TtPrefillRuntime._mtp_prepare_input's else-branch -- _mtp_unpack_activation and
#     MTPUnionEmbedding.from_embedding are unreachable on one galaxy, where every rank is the first rank.
#   * the merged KV chunk table at NUM_LAYERS+K = 82 slots, and the index stage's own MTP slot
#     (full_indexer_rank over first_layer_idx + num_my_layers + mtp_tail).
# Everything above is what the producer's `config 0 declares N layers, not 78+4` assert is guarding.
set -euo pipefail

MODEL="${1:-glm52}"
CONFIG="${2:-sc4}"
LEVELS="${3:-4}"

: "${TT_METAL_HOME:?TT_METAL_HOME must be set}"
: "${PREFILL_SUMMARIES:?PREFILL_SUMMARIES must be set by the blaze impl (shared /ci scratch for the KV table)}"
export PYTHONPATH="${TT_METAL_HOME}"
MANIFEST_DIR="${TT_METAL_HOME}/models/demos/deepseek_v3_d_p/tt/runners/manifests"
MGD_DIR="${TT_METAL_HOME}/models/demos/common/prefill/runners/topology_configuration/ci"

# One length, not two. The MTP tail golden is exactly [0, 56320) -- kv_cache/layer_78/ holds the single
# shard rows_00000000_00056320.safetensors -- so running past it would add only unverifiable rows. This
# leg carries no throughput number: MTP runs untraced (PREFILL_USE_TRACE=0 in the manifest, because the
# MTP path is not trace-captured), so its wall clock is not comparable to the traced trunk legs anyway.
CHUNK_SIZE=5120
GOLDEN_LEN=56320
MAX_SEQ_LEN=${GOLDEN_LEN}
# Only slot 0 has a golden behind it, and the MTP tail costs K more cache layers per user, so extra users
# buy no coverage. The trunk legs pin the producer to 1 user for the same reason.
NUM_USERS=1
# The measured floor for this configuration, not a chosen target: the trunk reads 0.859015 at full depth
# and the MTP levels 0.882565, both re-reproduced 2026-09-16. An MTP level is a layer of this same cache,
# so _check_mtp_kv_slots folds its minimum into this one gate rather than inventing a second threshold.
PCC_THRESHOLD=0.85

case "${MODEL}" in
  glm52) ;;
  *) echo "unknown model key '${MODEL}' (expected glm52 -- only GLM-5.2 ships MTP weights)" >&2; exit 2 ;;
esac
case "${CONFIG}" in
  sc1|sc4) ;;
  *) echo "unknown config '${CONFIG}' (expected sc1 or sc4)" >&2; exit 2 ;;
esac
MANIFEST="${MANIFEST_DIR}/${MODEL}_mtp${LEVELS}.json"
[ -f "${MANIFEST}" ] || { echo "no manifest for ${MODEL} MTP${LEVELS} at ${MANIFEST}" >&2; exit 2; }
MGD="${MGD_DIR}/${CONFIG}_mgd.textproto"
[ -f "${MGD}" ] || { echo "no mesh-graph descriptor for ${CONFIG} at ${MGD}" >&2; exit 2; }

NUM_LAYERS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["env"]["PREFILL_NUM_LAYERS"])' "${MANIFEST}")
MTP_LEVELS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["env"]["PREFILL_MTP_LEVELS"])' "${MANIFEST}")
[ "${MTP_LEVELS}" = "${LEVELS}" ] || { echo "${MANIFEST} declares PREFILL_MTP_LEVELS=${MTP_LEVELS}, not ${LEVELS}" >&2; exit 2; }

TRUNK_TRACE=/mnt/models/deepseek-prefill-cache/glm-traces/vllm-glm52-indexer-kcache-55k
MTP_TRACE="${PREFILL_MTP_TRACE_DIR:-/mnt/models/deepseek-prefill-cache/glm-traces/mtp-glm52-55k}"

# Fail here, not after 25 minutes of quad time. The MTP golden lives in its OWN trace: unset
# PREFILL_MTP_TRACE_DIR and _mtp_golden_source silently falls back to the trunk trace, which carries no
# kv_post_transform for layers 78+. The producer then logs "MTP: golden KV comparison SKIPPED", returns
# None, and this leg goes GREEN having gated the MTP tail on plumbing alone (finite, non-zero, pairwise
# distinct slots). Check the exact files kv_golden_present() reads, for every level this run will write.
for k in $(seq 0 $((MTP_LEVELS - 1))); do
  layer=$((NUM_LAYERS + k))
  if [ ! -f "${MTP_TRACE}/kv_cache/layer_${layer}.safetensors" ] \
     && ! compgen -G "${MTP_TRACE}/kv_cache/layer_${layer}/rows_*.safetensors" >/dev/null; then
    echo "MTP golden missing for level ${k} (layer ${layer}) under ${MTP_TRACE}/kv_cache -- the producer" >&2
    echo "would skip the MTP KV comparison and this leg would pass on plumbing alone" >&2
    exit 2
  fi
done
echo "MTP golden present for layers ${NUM_LAYERS}..$((NUM_LAYERS + MTP_LEVELS - 1)) under ${MTP_TRACE}"

export PIPELINE_DIR="${PREFILL_SUMMARIES/prefill_summaries/glm52_mtp_prefill_runner_kv}"
mkdir -p "${PIPELINE_DIR}"
TTRUN_DIR="${TTRUN_DIR:-/etc/ttop}"
TTRUN_PY="${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py"

RESOLVED_HOSTS=$(awk 'NF {printf "%s,", $1}' "${TTRUN_DIR}/hostfile" | sed 's/,$//')
TTRUN_CWD="${PIPELINE_DIR}/ttrun-cwd"
mkdir -p "${TTRUN_CWD}"

MR_DIR=$(mktemp -d "${PIPELINE_DIR}/${MODEL}_mtp${LEVELS}_ci_mr.XXXXXX")
export TABLE_PATH="${MR_DIR}/kv_chunk_table.pb"
PCC_DIR="${MR_DIR}/pcc_verdict"
RANKLOGS="${MR_DIR}/ranklogs"
TIMING_DIR="${MR_DIR}/timing"
mkdir -p "${TIMING_DIR}"

REAL_CHUNKS=$((MAX_SEQ_LEN / CHUNK_SIZE))
# First and last chunk only: 11 chunks is too few for a rate/latency split, and the indexer's cost grows
# with the prefix, so the pair brackets the run instead of pretending to sample it.
PROBE_CHUNKS="0,$((REAL_CHUNKS - 1))"

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
  # The per-level numbers are logged, not recorded in rank*.json (_write_pcc_verdict carries per_cache
  # kvpe/index and the folded "mtp" minimum), so scrape them or a regression is unattributable to a level.
  echo "==================== MTP per-level KV PCC ===================="
  find "${RANKLOGS}" -type f 2>/dev/null -exec grep -h -E "MTP level|MTP KV PCC|MTP: golden|MTP:.*level slots" {} + \
    | tail -40 || echo "no MTP PCC lines in the rank logs"
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
      --summary-name "${MODEL}_mtp${LEVELS}_${CONFIG}" \
      || echo "summary generation failed (non-fatal)"
  fi
  if [ -n "${PREFILL_KEEP_RUN_DIR:-}" ]; then
    echo "keeping run dir (PREFILL_KEEP_RUN_DIR set): ${MR_DIR}"
  else
    rm -rf "${MR_DIR}"
  fi
}
trap cleanup EXIT

echo "resolved shape for ${MODEL} MTP${MTP_LEVELS}/${CONFIG}: layers=${NUM_LAYERS} max_seq_len=${MAX_SEQ_LEN} users=${NUM_USERS} chunks=${REAL_CHUNKS}"

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
    export PREFILL_MAX_SEQ_LEN=${MAX_SEQ_LEN}; \
    export PREFILL_NUM_USERS=${NUM_USERS}; \
    export PREFILL_SYNC_PER_CHUNK=1; \
    export PREFILL_TIMING_DIR='${TIMING_DIR}'; \
    export PREFILL_ENABLE_MIGRATION=1; \
    export PREFILL_MOCK_MIGRATION=1; \
    export PREFILL_MIGRATION_TABLE_PATH='${TABLE_PATH}'; \
    export PREFILL_LAYER_ACK_D2H=1; \
    export TT_METAL_SHM_TRACKING_DISABLED=1; \
    export LOGURU_LEVEL=INFO; \
    exec python3 -m models.demos.common.prefill.runners.prefill_runner" &
RUNNER_PID=$!
cd "${TT_METAL_HOME}"

# Bounds mesh bringup + weight load, not the chunk loop: the table is published once the runner serves.
# 3600s follows the trunk launcher, which was raised from 30 min after Kimi-K3 at 93 layers over 4 ranks
# measured 42.6 min to first serve and exited 1 while the ranks were still compiling. MTP needs the room
# for its own reason: it loads one module per level beyond the trunk and pulls them from a SECOND ttnn
# cache dir (TT_GLM52_MTP_TTNN_CACHE) that every rank hits over the same NFS mount. The real liveness
# guard is the kill -0 below, which catches a dead runner in seconds, so a larger bound costs nothing.
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
  -x PATH -x LD_LIBRARY_PATH \
  bash -lc "cd '${TT_METAL_HOME}'; \
    export PYTHONPATH='${TT_METAL_HOME}'; \
    export PYTHONUNBUFFERED=1; \
    export PREFILL_PRODUCER_MANIFEST='${MANIFEST}'; \
    export PREFILL_MAX_SEQ_LEN=${MAX_SEQ_LEN}; \
    export PREFILL_NUM_USERS=${NUM_USERS}; \
    export PREFILL_PRODUCER_CHUNKS=${REAL_CHUNKS}; \
    export PREFILL_PRODUCER_MAX_REQUESTS=1; \
    export PREFILL_PRODUCER_MULTI_TURN_PROB=0; \
    export PREFILL_PCC_GOLDEN_LEN=${GOLDEN_LEN}; \
    export PREFILL_MIGRATION_TABLE_PATH='${TABLE_PATH}'; \
    export PREFILL_PCC_SUMMARY_DIR='${PCC_DIR}'; \
    export PREFILL_PRODUCER_CHECK_PCC=1; \
    export PREFILL_SEND_SHUTDOWN=1; \
    export PREFILL_STANDALONE_CHUNKED_PCC=${PCC_THRESHOLD}; \
    export PREFILL_H2D_CONNECT_TIMEOUT=120; \
    export PREFILL_TRACE_DIR='${TRUNK_TRACE}'; \
    export PREFILL_MTP_TRACE_DIR='${MTP_TRACE}'; \
    export LOGURU_LEVEL=INFO; \
    exec python3 -m models.demos.common.prefill.runners.prefill_producer"
PROD_RC=$?
set -e

if [ "${PROD_RC}" -eq 0 ]; then
  wait "${RUNNER_PID}" || echo "runner exited non-zero after producer success (rc=$?)"
fi

# Same gate as the trunk legs, with one addition: exactly one rank must have reported an "mtp" entry. The
# MTP levels run on the LAST rank only, so _check_mtp_kv_slots returns None everywhere else -- and it also
# returns None when the golden is absent, which the preflight above rules out. Without this count a run
# that placed the levels on no rank at all, or on several, reads as a clean pass.
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
    status = "ok" if v.get("ok") else "FAIL"
    print(f"  {name}: {status} min_pcc={v.get('min_pcc')} threshold={v.get('threshold')} per_cache={per_cache}")
    if not v.get("ok"):
        bad += 1
    if "mtp" in per_cache:
        mtp_ranks.append((name, per_cache["mtp"]))
if bad:
    print(f"PCC GATE FAIL: {bad}/{len(files)} rank(s) below threshold or unvalidated", file=sys.stderr)
    sys.exit(1)
if len(mtp_ranks) != 1:
    print(
        f"PCC GATE FAIL: {len(mtp_ranks)} rank(s) reported an MTP KV PCC, expected exactly 1 (the last "
        f"pipeline rank): {mtp_ranks}",
        file=sys.stderr,
    )
    sys.exit(1)
print(f"PCC GATE PASS: {len(files)}/{expected} ranks ok, MTP tail gated on {mtp_ranks[0][0]} at {mtp_ranks[0][1]}")
PY

if [ "${PROD_RC}" -ne 0 ]; then
  exit "${PROD_RC}"
fi
exit "${PCC_GATE_RC}"
