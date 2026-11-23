#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Write the GLM-5.3 stage reference JSON from run_stage_probe.sh run dirs (CPU only):
#
#   make_glm53_stage_ref.sh <out.json> <full78_run_dir> [<stage_alone_run_dir> ...]
#
# The reference: GLM-5.3's default split 19/19/20/20, each stage cut out of the 78-layer single-rank run by layer
# acks, slowest-stage period_ms per probe. Also: GLM-5.3 at GLM-5.4's stage ranges, the whole 78-layer run, the
# CI comparison and, from the stage-alone runs (dir names stage-probe-<prod_G0|...|g54_G3>-*), the evidence
# section and the G0 entry overhead. Extra text fields: STAGE_REF_PROVISIONAL (sets the reference's status),
# STAGE_REF_NOTES (newline-separated key=text lines).
set -euo pipefail
OUT="${1:?usage: make_glm53_stage_ref.sh <out.json> <full78_run_dir> [<stage_alone_run_dir> ...]}"
F="${2:?full78 run dir}"
shift 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
declare -A ALONE
for d in "$@"; do
  n="$(basename "$d")"; n="${n#stage-probe-}"; ALONE["${n%%-*}"]="$d"
done

ARGS=(
  --split glm53_default_split G0="$F@0-18" G1="$F@19-37" G2="$F@38-57" G3="$F@58-77"
  --split glm53_at_glm54_ranges G0="$F@0-25" G1="$F@26-53" G2="$F@26-53" G3="$F@54-77"
  --split full78_single_rank L0_77="$F"
  --reference glm53_default_split
  # SC4 nightly 45115566753 job 137500206236 (main e39e42718ca), the epic's reference.pipeline tok/s.
  --ci glm53_default_split 5k=54757 50k=53445 100k=48938 255k=45738 510k=38616 1020k=31722
)
if [ "${#ALONE[@]}" -eq 7 ]; then
  ARGS+=(
    --split evidence_stage_alone/glm53_default_split G0="${ALONE[prod_G0]}" G1="${ALONE[prod_G1]}@19"
    G2="${ALONE[prod_G2]}" G3="${ALONE[prod_G3]}"
    --split evidence_stage_alone/glm53_at_glm54_ranges G0="${ALONE[g54_G0]}" G1="${ALONE[g54_G1]}"
    G2="${ALONE[g54_G1]}" G3="${ALONE[g54_G3]}"
    --entry-overhead evidence_stage_alone/glm53_default_split/G0 glm53_default_split/G0
  )
elif [ "${#ALONE[@]}" -ne 0 ]; then
  echo "need all 7 stage-alone runs (prod_G0..G3, g54_G0/G1/G3), got: ${!ALONE[*]}" >&2
  exit 2
fi
[ -z "${STAGE_REF_PROVISIONAL:-}" ] || ARGS+=(--provisional "${STAGE_REF_PROVISIONAL}")
while IFS= read -r note; do
  [ -z "${note}" ] || ARGS+=(--note "${note}")
done <<< "${STAGE_REF_NOTES:-}"
ARGS+=(--note "generated_by=$(basename "$0") $(git -C "${HERE}" rev-parse --short=11 HEAD) $(date -u +%FT%TZ)")

python3 "${HERE}/stage_probe_summary.py" --out "${OUT}" "${ARGS[@]}"
