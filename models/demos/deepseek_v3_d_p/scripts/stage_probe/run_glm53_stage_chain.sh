#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# GLM-5.3 stage references on one galaxy (run_stage_probe.sh), in order:
#   full78        all 78 layers on one rank; THE REFERENCE: stage_probe_summary.py cuts each stage out of it by layer
#                 acks (<run>@<first>-<last>), so every stage sees real hidden states
#   prod_G0..G3   GLM-5.3's default 4-rank split (compute_layer_split: 19/19/20/20), each stage alone
#   g54_G0/G1/G3  GLM-5.3 at GLM-5.4's stage ranges L1-26, L27-54, L55-78 (G2 = G1's layers), each stage alone
# A stage run alone past layer 0 (PREFILL_PP_LAYER_OFFSET) feeds token embeddings, not the previous stage's
# hidden states, into its first layer; its MoE layers then measure ~1.5-2 ms slower than in full78. Those runs
# are kept only as evidence. (prod_G1 runs from layer 18, the full layer whose top-k layers 19-21 reuse.)
#
#   run_glm53_stage_chain.sh [stage_name ...]     (default: all; e.g. "prod_G0" for one)
#
# Holds the shared-device priority marker $TT_DEVICE_PRIORITY_DIR/P0a-glm54-perf across the chain, as
# ai-workspace instructions/tt-metal/shared-device.md asks; each stage takes the device lock on its own.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
declare -A STAGES=(
  [full78]="0 78"
  [prod_G0]="0 19" [prod_G1]="18 20" [prod_G2]="38 20" [prod_G3]="58 20"
  [g54_G0]="0 26" [g54_G1]="26 28" [g54_G3]="54 24"
)
ORDER=(full78 prod_G0 prod_G1 prod_G2 prod_G3 g54_G0 g54_G1 g54_G3)
[ $# -gt 0 ] && ORDER=("$@")

MARKER="${TT_DEVICE_PRIORITY_DIR:?set TT_DEVICE_PRIORITY_DIR}/P0a-glm54-perf"
mkdir -p "${TT_DEVICE_PRIORITY_DIR}"
# A marker that already exists belongs to a longer chain (several invocations); leave it to its owner.
if [ -e "${MARKER}" ]; then
  echo "[$(date -u +%FT%TZ)] chain: ${MARKER} already held; stages: ${ORDER[*]}"
else
  touch "${MARKER}"
  trap 'rm -f "${MARKER}"; echo "[$(date -u +%FT%TZ)] chain: removed ${MARKER}"' EXIT
  echo "[$(date -u +%FT%TZ)] chain: created ${MARKER}; stages: ${ORDER[*]}"
fi

rc=0
for s in "${ORDER[@]}"; do
  read -r first count <<< "${STAGES[$s]:?unknown stage $s}"
  echo "[$(date -u +%FT%TZ)] chain: ${s} layers [${first}, $((first + count)))"
  if ! bash "${HERE}/run_stage_probe.sh" "${s}" "${first}" "${count}" ${STAGE_PROBE_REQUEST_TOKENS:-}; then
    echo "[$(date -u +%FT%TZ)] chain: ${s} FAILED; stopping"
    rc=1
    break
  fi
done
exit "${rc}"
