#!/usr/bin/env bash
# run_op.sh — per-op runner: ONE op, run to completion, then exit (which frees
# the Slurm node).  Object-identity gate -> stream the full 2^32 sem-vs-hand
# sweep (resume-safe from cached band SHAs) -> write VERDICT -> exit.  No
# claims, no work-stealing, no supervisor: a dead job only ever affects its own
# op and is simply resubmitted.
#
# This is the BREADTH leg: one chip per op, many ops in parallel across nodes.
# For one op across all 32 chips of a galaxy, use galaxy_shard.sh instead.
#
#   usage: [SWEEP=fp32|binary] OPS_TSV=... BUILD=... VENV=... LLK_HOME=... \
#          PYDIR=... OUT=... bash run_op.sh <op>
#
#   SWEEP   fp32   one-operand space,  fp32_stream_sweep.py   (default)
#           binary two-operand space,  binary_stream_sweep.py
#   OPS_TSV op<TAB>sem_node<TAB>hand_node
#   IDMAP   op<TAB>sem_variant<TAB>sem_text<TAB>hand_variant<TAB>hand_text
#           required for SWEEP=fp32, optional for SWEEP=binary
#   BUILD   a dir containing tt-llk-build/ with both legs' prebuilt ELFs
#   GOLDEN  0 turns OFF the ULP/golden leg (default on).  The leg RIDES this
#           pass -- the streamer exports LANEMR_GOLDEN, the device test folds
#           every streamed chunk through threeway_golden.py and writes a .corr
#           sidecar, and the sweep turns those into <op>-CORRECTNESS-LEDGER.tsv.
#           No extra device time, no extra retention; never affects the verdict.
set -uo pipefail
op="${1:?usage: run_op.sh <op>}"
SWEEP="${SWEEP:-fp32}"
: "${OPS_TSV:?} ${BUILD:?} ${VENV:?} ${LLK_HOME:?} ${PYDIR:?} ${OUT:?}"

case "$SWEEP" in
  fp32)   STREAMER=fp32_stream_sweep.py;   : "${IDMAP:?SWEEP=fp32 requires IDMAP}" ;;
  binary) STREAMER=binary_stream_sweep.py ;;
  *) echo "FATAL: SWEEP must be fp32 or binary, got '$SWEEP'" >&2; exit 2 ;;
esac

[ -s "$OUT/$op/$op-VERDICT.txt" ] && { echo "$op already has a verdict"; exit 0; }

sem=$(awk -F'\t' -v o="$op" '$1==o{print $2}' "$OPS_TSV")
hand=$(awk -F'\t' -v o="$op" '$1==o{print $3}' "$OPS_TSV")
[ -n "$sem" ] && [ -n "$hand" ] || { echo "no nodes for $op in $OPS_TSV"; exit 1; }

# node-local RUNNER_TEMP holding a private copy of the prebuilt ELFs
# (consume-only; keeps conftest's order_records off shared storage and is
# faster than reading them over NFS).
RT="/tmp/run-op-rt-$(hostname -s)"
[ -d "$RT/tt-llk-build/sources" ] || { mkdir -p "$RT"; cp -a "$BUILD/tt-llk-build" "$RT/"; }
ulimit -u "$(ulimit -Hu)" 2>/dev/null || true

idmap_args=()
[ -n "${IDMAP:-}" ] && [ -s "${IDMAP:-}" ] && idmap_args=(--idmap "$IDMAP")
golden_args=()
[ "${GOLDEN:-1}" = 1 ] && golden_args=(--golden "$op")

LANEMK_WAIT_TIMEOUT="${LANEMK_WAIT_TIMEOUT:-600}" \
"$VENV" "$(dirname "$0")/$STREAMER" \
  --op "$op" --sem-node "$sem" --hand-node "$hand" \
  --farm "$PYDIR" --venv "$VENV" --llk-home "$LLK_HOME" --runner-temp "$RT" \
  --tile-dim 256,256 --band-bits 28 --chip 0 --out "$OUT/$op" \
  ${idmap_args[@]+"${idmap_args[@]}"} \
  ${golden_args[@]+"${golden_args[@]}"}
