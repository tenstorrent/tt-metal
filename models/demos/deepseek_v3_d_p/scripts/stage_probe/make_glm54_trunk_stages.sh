#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Write the GLM-5.4 whole-trunk stage JSON from run_glm54_trunk.sh run dirs (CPU only):
#
#   make_glm54_trunk_stages.sh <out.json> <glm53_stage_ref.json> <trunk_run_dir> [<repeat_run_dir>]
#
# Stages are cut out of the 106-pass single-rank run by layer acks in pass space (stage_probe_summary.py), exactly as
# the GLM-5.3 reference cut its 78-layer run: G0 passes 0-25, G1 26-53 (loop pass 1), G2 54-81 (loop pass 2), G3
# 82-105. at_glm54_ranges has the shape of glm53_stage_ref.json's glm53_at_glm54_ranges; the repeat run, if given,
# goes under repeat/. glm54_trunk_result.py then adds the per-run checks, determinism, repeat spread and the per-probe
# met/unmet verdict against the reference. Extra text fields: STAGE_REF_NOTES (newline-separated key=text lines);
# TRUNK_PROVISIONAL (newline-separated <probe>=<why> lines) marks those probes' verdicts provisional.
set -euo pipefail
OUT="${1:?usage: make_glm54_trunk_stages.sh <out.json> <glm53_stage_ref.json> <trunk_run_dir> [<repeat_run_dir>]}"
REF="${2:?glm53 stage reference json}"
R1="${3:?trunk run dir}"
R2="${4:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARGS=(--split at_glm54_ranges G0="$R1@0-25" G1="$R1@26-53" G2="$R1@54-81" G3="$R1@82-105")
[ -z "${R2}" ] || ARGS+=(--split repeat/at_glm54_ranges G0="$R2@0-25" G1="$R2@26-53" G2="$R2@54-81" G3="$R2@82-105")
ARGS+=(--split full106_single_rank P0_105="$R1")
while IFS= read -r note; do
  [ -z "${note}" ] || ARGS+=(--note "${note}")
done <<< "${STAGE_REF_NOTES:-}"
ARGS+=(--note "generated_by=$(basename "$0") $(git -C "${HERE}" rev-parse --short=11 HEAD) $(date -u +%FT%TZ)")

python3 "${HERE}/stage_probe_summary.py" --out "${OUT}" "${ARGS[@]}" > /dev/null
python3 "${HERE}/glm54_trunk_result.py" "${OUT}" "${REF}" "${R1}" ${R2:+"${R2}"}
