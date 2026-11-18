#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Manual Kimi-K3.1 runs on top of run_multirank_pcc.sh's kimi_k3_1 sc4 leg; CI runs neither. They check K3.1's
# prefill -> decode cache migration through tt-d-gen's KV Manager over HTTP, until d-gen's engine path runs it.
# PREFILL_MIGRATION_TESTING.md (Gate 3) has the decode side and the byte check.
#
#   run_k31_pd.sh smoke  sc4, all 93 layers, with a single-rank producer that only feeds tokens (no read-back,
#                        no PCC gate): the four-stage pipeline runs.
#   run_k31_pd.sh pd     smoke, plus the table and device maps exported to files, and migration_driver in place
#                        of the producer: it feeds the prompt, POSTs /migrate to the prefill leader KVM, then
#                        writes the handoff and the DONE sentinel for decode.
#
# Both need the environment run_multirank_pcc.sh needs. pd also needs PREFILL_PD_TABLE_PATH (shared storage),
# PREFILL_MIGRATION_KVM_URL, MIGRATION_DONE_FILE and PREFILL_MIGRATION_HANDOFF_PATH.
set -euo pipefail

USAGE="usage: run_k31_pd.sh smoke|pd"
MODE="${1:?${USAGE}}"
case "${MODE}" in
  smoke|pd) ;;
  *) echo "${USAGE}" >&2; exit 2 ;;
esac

export PREFILL_FEED_ONLY=1
# The CI leg's golden length.
export PREFILL_REQUEST_LEN_OVERRIDE="${PREFILL_SC4_REQUEST_LEN:-56320}"
# At the manifest's 1M-context shape the expert weights no longer fit in DRAM beside K3.1's indexer cache
# (bank_manager OOM on load). 245,760 is sc1's shape, 24 layers per galaxy as here.
export PREFILL_MAX_SEQ_LEN_OVERRIDE="${PREFILL_SC4_MAX_SEQ_LEN:-245760}"
RUNNER_ENV="export LOGURU_LEVEL=INFO;"
PRODUCER_ENV=""

if [ "${MODE}" = pd ]; then
  export PREFILL_TABLE_PATH="${PREFILL_PD_TABLE_PATH:?pd needs PREFILL_PD_TABLE_PATH on shared storage}"
  : "${PREFILL_MIGRATION_KVM_URL:?pd needs PREFILL_MIGRATION_KVM_URL}"
  : "${MIGRATION_DONE_FILE:?pd needs MIGRATION_DONE_FILE}"
  : "${PREFILL_MIGRATION_HANDOFF_PATH:?pd needs PREFILL_MIGRATION_HANDOFF_PATH}"
  RUNNER_ENV+=" export PREFILL_MIGRATION_EXPORT_TO_FILE=1; \
    export PREFILL_MIGRATION_DEVICE_MAP_PATH=${PREFILL_PD_DEVICE_MAP:-/tmp/prefill_device_map.txt};"
  export PREFILL_PRODUCER_MODULE=migration_driver
  # KDA state is recurrent and prefill samples no first token, so the handoff leaves the next prompt token for
  # decode to run (tt-blaze kimi_k3/entry.py MIGRATION_RESUME_NEEDS_FIRST_TOKEN).
  PRODUCER_ENV+=" export PREFILL_MIGRATION_KVM_URL='${PREFILL_MIGRATION_KVM_URL}'; \
    export PREFILL_MIGRATION_KVM_RECEIVER='${PREFILL_MIGRATION_KVM_RECEIVER:-decode-0}'; \
    export PREFILL_MIGRATION_PAIRS='${PREFILL_MIGRATION_PAIRS:-0:1}'; \
    export PREFILL_MIGRATION_SRC_ENDPOINT_ID=1; export PREFILL_MIGRATION_DEST_ENDPOINT_ID=0; \
    export MIGRATION_DONE_FILE='${MIGRATION_DONE_FILE}'; \
    export PREFILL_MIGRATION_HANDOFF_PATH='${PREFILL_MIGRATION_HANDOFF_PATH}'; \
    export PREFILL_MIGRATION_HANDOFF_TAIL=${PREFILL_MIGRATION_HANDOFF_TAIL:-1}; \
    export PREFILL_VERIFY_MIGRATION=off; \
    ${PREFILL_MIGRATION_DUMP_SRC_KV:+export PREFILL_MIGRATION_DUMP_SRC_KV='${PREFILL_MIGRATION_DUMP_SRC_KV}';}"
fi

export PREFILL_EXTRA_RUNNER_ENV="${RUNNER_ENV}"
export PREFILL_EXTRA_PRODUCER_ENV="${PRODUCER_ENV}"
exec "$(dirname "${BASH_SOURCE[0]}")/ci/run_multirank_pcc.sh" kimi_k3_1 sc4
