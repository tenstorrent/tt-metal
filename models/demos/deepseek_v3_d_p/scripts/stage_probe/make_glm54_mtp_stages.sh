#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Write the GLM-5.4 whole-trunk + MTP stage JSON (CPU only, q171.2.10):
#
#   make_glm54_mtp_stages.sh <out.json> <glm53_stage_ref.json> <no_mtp.json> <mtp_run_dir> [<mtp_repeat_run_dir>]
#
# The runs are run_glm54_trunk.sh with TRUNK_MTP=1: all 106 passes and MTP on one rank. Stages are cut as in
# make_glm54_trunk_stages.sh, except that G3 runs from pass 82 to the run's last ack, its MTP rows included (MTP runs
# on G3 after the coda); g3_trunk_passes_in_mtp_run is G3's passes 82-105 alone. glm54_trunk_result.py adds the checks,
# determinism, repeat spread and the 1.54x verdict (with G3 + MTP); glm54_mtp_result.py compares G3 with the no-MTP
# run's (<no_mtp.json>, make_glm54_trunk_stages.sh output, same head). Extra text fields: STAGE_REF_NOTES
# (newline-separated key=text lines); TRUNK_PROVISIONAL as make_glm54_trunk_stages.sh.
set -euo pipefail
OUT="${1:?usage: make_glm54_mtp_stages.sh <out.json> <glm53_stage_ref.json> <no_mtp.json> <mtp_run_dir> [<repeat>]}"
REF="${2:?glm53 stage reference json}"
BASE="${3:?no-MTP trunk json}"
R1="${4:?mtp run dir}"
R2="${5:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARGS=(--split at_glm54_ranges G0="$R1@0-25" G1="$R1@26-53" G2="$R1@54-81" G3="$R1@82")
[ -z "${R2}" ] || ARGS+=(--split repeat/at_glm54_ranges G0="$R2@0-25" G1="$R2@26-53" G2="$R2@54-81" G3="$R2@82")
ARGS+=(--split g3_trunk_passes_in_mtp_run G3="$R1@82-105")
ARGS+=(--split full106_single_rank_with_mtp P0_end="$R1")
while IFS= read -r note; do
  [ -z "${note}" ] || ARGS+=(--note "${note}")
done <<< "${STAGE_REF_NOTES:-}"
ARGS+=(--note "generated_by=$(basename "$0") $(git -C "${HERE}" rev-parse --short=11 HEAD) $(date -u +%FT%TZ)")

python3 "${HERE}/stage_probe_summary.py" --out "${OUT}" "${ARGS[@]}" > /dev/null
python3 "${HERE}/glm54_trunk_result.py" "${OUT}" "${REF}" "${R1}" ${R2:+"${R2}"}
python3 "${HERE}/glm54_mtp_result.py" "${OUT}" "${BASE}"
