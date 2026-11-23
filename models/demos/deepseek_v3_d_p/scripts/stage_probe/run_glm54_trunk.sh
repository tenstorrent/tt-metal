#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# The whole GLM-5.4 trunk (all 106 layer passes) on one 8x4 rank, as one run_stage_probe.sh job (q171.2.9):
#
#   run_glm54_trunk.sh <name> [request_tokens]
#
# (TRUNK_PASSES / TRUNK_MTP below: one stage alone, and MTP on, q171.2.10.)
#
# Same request, chunking and warm-up as the GLM-5.3 reference (q171.3.4, full78): the CI 1M prompt in 5120-token
# chunks after 10 warm-up chunks (request_tokens defaults to 1,044,480). trunk_probe_runner adds the pass-2 push-reuse
# check and the end-of-request KV digest. Stage times are cut afterwards from the layer acks (make_glm54_trunk_stages.sh).
#
# Device priority: with TT_DEVICE_PRIORITY_DIR set, it waits while a marker whose priority sorts before
# TRUNK_PRIORITY (default P0a) exists, and holds <priority>-<TRUNK_MARKER_NAME> only while its job runs.
# Env: STAGE_PROBE_LOG_DIR (required); STAGE_PROBE_WARMUP_CHUNKS (default 10); TRUNK_RUNNER_ENV (extra runner exports);
# TRUNK_PASSES="<first_pass> <num_passes>" (default "0 106"; "82 24" = G3 alone); TRUNK_MTP=1 runs GLM-5.3's MTP on the
# last rank with GLM-5.3's production MTP settings, read from manifests/glm53_mtp_kv_share.json (PREFILL_MTP_LEVELS,
# PREFILL_MTP_DECODE_LEVELS, TT_GLM53_MTP_TTNN_CACHE), exported to the runner and the producer.
set -euo pipefail

NAME="${1:?usage: run_glm54_trunk.sh <name> [request_tokens]}"
TOKENS="${2:-1044480}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIORITY="${TRUNK_PRIORITY:-P0a}"
MARK="${TT_DEVICE_PRIORITY_DIR:-}/${PRIORITY}-${TRUNK_MARKER_NAME:-glm54-trunk}"

read -r FIRST_PASS NUM_PASSES <<< "${TRUNK_PASSES:-0 106}"
MTP_ENV=""
if [ "${TRUNK_MTP:-0}" = 1 ]; then
  MTP_ENV="$(python3 - "${HERE}/../../tt/runners/manifests/glm53_mtp_kv_share.json" <<'PY'
import json, sys
env = json.load(open(sys.argv[1]))["env"]
print(" ".join(f"export {k}={env[k]};" for k in ("PREFILL_MTP_LEVELS", "PREFILL_MTP_DECODE_LEVELS", "TT_GLM53_MTP_TTNN_CACHE")))
PY
)"
fi

say() { echo "[$(date -u +%FT%TZ)] trunk ${NAME}: $*"; }

if [ -n "${TT_DEVICE_PRIORITY_DIR:-}" ]; then
  waited=0
  while true; do
    ahead=""
    for f in $(ls "${TT_DEVICE_PRIORITY_DIR}" 2>/dev/null); do
      [[ "${f%%-*}" < "${PRIORITY}" ]] && ahead="${ahead} ${f}"
    done
    [ -n "${ahead}" ] || break
    [ $((waited % 60)) -eq 0 ] && say "waiting for higher-priority device work:${ahead}"
    sleep 15
    waited=$((waited + 15))
  done
  trap 'rm -f "${MARK}"' EXIT
  touch "${MARK}"
  say "priority marker ${MARK##*/} set"
fi

say "launching: passes [${FIRST_PASS}, $((FIRST_PASS + NUM_PASSES))), request ${TOKENS} tokens, warmup ${STAGE_PROBE_WARMUP_CHUNKS:-10} chunks, mtp: ${MTP_ENV:-off}"
STAGE_PROBE_MODEL=glm54 \
  STAGE_PROBE_RUNNER_MODULE=models.demos.deepseek_v3_d_p.scripts.stage_probe.trunk_probe_runner \
  STAGE_PROBE_RUNNER_ENV="${MTP_ENV} ${TRUNK_RUNNER_ENV:-}" \
  STAGE_PROBE_PRODUCER_ENV="${MTP_ENV}" \
  "${HERE}/run_stage_probe.sh" "${NAME}" "${FIRST_PASS}" "${NUM_PASSES}" "${TOKENS}"
