#!/usr/bin/env bash
# laneMQ per-op runner for the TWO-operand 2^32 sem-vs-hand sweep: ONE op, run to
# completion, then exit (which frees the Slurm node). Mirrors lanemk_run_op.sh:
# object-identity gate -> stream the full 2^32 joint bf16^2 sweep (resume-safe from
# cached band SHAs) -> write VERDICT -> exit. No claims, no work-stealing, no
# supervisor: a dead job only ever affects its own op and is simply resubmitted.
#
# Config via env: OPS_TSV IDMAP BUILD VENV LLK_HOME PYDIR OUT.
#   OPS_TSV : "op<TAB>sem_node<TAB>hand_node" per line
#   IDMAP   : "op<TAB>sem_variant<TAB>sem_text<TAB>hand_variant<TAB>hand_text" (optional gate)
#   BUILD   : a dir containing tt-llk-build/ with both legs' prebuilt ELFs
set -uo pipefail
op="${1:?usage: lanemq_run_op.sh <op>}"
: "${OPS_TSV:?} ${BUILD:?} ${VENV:?} ${LLK_HOME:?} ${PYDIR:?} ${OUT:?}"
[ -s "$OUT/$op/$op-VERDICT.txt" ] && { echo "$op already has a verdict"; exit 0; }

sem=$(awk -F'\t' -v o="$op" '$1==o{print $2}' "$OPS_TSV")
hand=$(awk -F'\t' -v o="$op" '$1==o{print $3}' "$OPS_TSV")
[ -n "$sem" ] && [ -n "$hand" ] || { echo "no nodes for $op in $OPS_TSV"; exit 1; }

# node-local RUNNER_TEMP holding a private copy of the prebuilt ELFs (consume-only).
RT="/tmp/lanemq-rt-$(hostname -s)"
[ -d "$RT/tt-llk-build/sources" ] || { mkdir -p "$RT"; cp -a "$BUILD/tt-llk-build" "$RT/"; }
ulimit -u "$(ulimit -Hu)" 2>/dev/null || true

idmap_args=()
[ -n "${IDMAP:-}" ] && [ -s "${IDMAP:-}" ] && idmap_args=(--idmap "$IDMAP")

LANEMK_WAIT_TIMEOUT="${LANEMK_WAIT_TIMEOUT:-600}" \
"$VENV" "$(dirname "$0")/binary_stream_sweep.py" \
  --op "$op" --sem-node "$sem" --hand-node "$hand" \
  --farm "$PYDIR" --venv "$VENV" --llk-home "$LLK_HOME" --runner-temp "$RT" \
  --band-bits "${BAND_BITS:-26}" --chip "${CHIP:-0}" --out "$OUT/$op" "${idmap_args[@]}"
