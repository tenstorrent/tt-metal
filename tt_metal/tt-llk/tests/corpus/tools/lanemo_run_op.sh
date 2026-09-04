#!/usr/bin/env bash
# laneMO per-op sampling runner: ONE op, sample to completion, then exit (which
# frees the Slurm node). Object-identity gate is enforced by lanemo_sample_sweep
# (same certified ELF + optional .text idmap) -> stream N stratified operand-A
# sample tiles through sem + hand -> compare output SHA -> write VERDICT -> exit.
# No claims, no work-stealing, no supervisor: a dead job only affects its own op
# and is simply resubmitted (--requeue).
#
# A verdict here is SAMPLED-CONSISTENT / SAMPLED-DIVERGENT / SAMPLE-REFUSED — a
# DISTINCT, WEAKER class than the proven-equal ops. Never "verified"/"proven".
#
# Config via env (set by the sbatch --export): OPS_TSV IDMAP BUILD VENV LLK_HOME
# PYDIR OUT [N_SAMPLES] [SEED] [CKPT].
set -uo pipefail
op="${1:?usage: lanemo_run_op.sh <op>}"
: "${OPS_TSV:?} ${BUILD:?} ${VENV:?} ${LLK_HOME:?} ${PYDIR:?} ${OUT:?}"
[ -s "$OUT/$op/$op-VERDICT.txt" ] && { echo "$op already has a verdict"; exit 0; }

sem=$(awk -F'\t' -v o="$op" '$1==o{print $2}' "$OPS_TSV")
hand=$(awk -F'\t' -v o="$op" '$1==o{print $3}' "$OPS_TSV")
[ -n "$sem" ] && [ -n "$hand" ] || { echo "no nodes for $op in $OPS_TSV"; exit 1; }

# node-local RUNNER_TEMP holding a private copy of the prebuilt ELFs (consume-only;
# keeps conftest's order_records off shared NFS and is faster than NFS reads).
RT="/tmp/lanemo-rt-$(hostname -s)"
[ -d "$RT/tt-llk-build/sources" ] || { mkdir -p "$RT"; cp -a "$BUILD/tt-llk-build" "$RT/"; }
ulimit -u "$(ulimit -Hu)" 2>/dev/null || true

LANEMK_WAIT_TIMEOUT="${LANEMK_WAIT_TIMEOUT:-600}" \
"$VENV" "$(dirname "$0")/lanemo_sample_sweep.py" \
  --op "$op" --sem-node "$sem" --hand-node "$hand" \
  --farm "$PYDIR" --venv "$VENV" --llk-home "$LLK_HOME" --runner-temp "$RT" \
  --n-samples "${N_SAMPLES:-1000000}" --seed "${SEED:-0x1}" --ckpt "${CKPT:-4096}" \
  --chip "${CHIP:-0}" --out "$OUT/$op"
