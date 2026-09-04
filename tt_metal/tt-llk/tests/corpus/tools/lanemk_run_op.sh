#!/usr/bin/env bash
# laneMK per-op runner: ONE op, run to completion, then exit (which frees the Slurm node).
# object-identity gate -> stream the full 2^32 sem-vs-hand sweep (resume-safe from cached
# band SHAs) -> write VERDICT -> exit. No claims, no work-stealing, no supervisor: a dead
# job only ever affects its own op and is simply resubmitted.
#
# Config via env (set by lanemk_submit.sh): OPS_TSV IDMAP BUILD VENV LLK_HOME PYDIR OUT.
set -uo pipefail
op="${1:?usage: lanemk_run_op.sh <op>}"
: "${OPS_TSV:?} ${IDMAP:?} ${BUILD:?} ${VENV:?} ${LLK_HOME:?} ${PYDIR:?} ${OUT:?}"
[ -s "$OUT/$op/$op-VERDICT.txt" ] && { echo "$op already has a verdict"; exit 0; }

sem=$(awk -F'\t' -v o="$op" '$1==o{print $2}' "$OPS_TSV")
hand=$(awk -F'\t' -v o="$op" '$1==o{print $3}' "$OPS_TSV")
[ -n "$sem" ] && [ -n "$hand" ] || { echo "no nodes for $op in $OPS_TSV"; exit 1; }

# node-local RUNNER_TEMP holding a private copy of the prebuilt ELFs (consume-only; keeps
# conftest's order_records off shared NFS and is faster than NFS reads).
RT="/tmp/lanemk-rt-$(hostname -s)"
[ -d "$RT/tt-llk-build/sources" ] || { mkdir -p "$RT"; cp -a "$BUILD/tt-llk-build" "$RT/"; }
ulimit -u "$(ulimit -Hu)" 2>/dev/null || true

LANEMK_WAIT_TIMEOUT="${LANEMK_WAIT_TIMEOUT:-600}" \
"$VENV" "$(dirname "$0")/fp32_stream_sweep.py" \
  --op "$op" --sem-node "$sem" --hand-node "$hand" \
  --farm "$PYDIR" --venv "$VENV" --llk-home "$LLK_HOME" --runner-temp "$RT" \
  --idmap "$IDMAP" --tile-dim 256,256 --band-bits 28 --chip 0 --out "$OUT/$op"
