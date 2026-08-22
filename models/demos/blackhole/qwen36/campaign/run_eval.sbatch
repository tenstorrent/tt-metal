#!/usr/bin/env bash
# Accuracy-gate job for the qwen38 precision ladder. One iteration is:
#   push branch -> sbatch --export=ALL,REF=<sha-or-branch> run_eval.sbatch -> read one log.
# Shares run_bench.sbatch's conventions (env.sh contract, prologue reset, NFS
# budget guard) but its own singleton name, so an eval can queue behind a
# measurement without blocking it.
#
# Env passthrough knobs: REF, SKIP_BUILD=1, SKIP_GATE=1, plus every
# QWEN38_EVAL_* / QWEN38_GPQA_PATH knob from eval_gate.py. Precision-rung flow:
#   reference:  sbatch --export=ALL,REF=<ref>,QWEN38_EVAL_DUMP_REF=/data/ayerofieiev/qwen38/eval_data/tf_ref.json run_eval.sbatch
#   candidate:  sbatch --export=ALL,REF=<rung>,QWEN38_EVAL_REF=/data/ayerofieiev/qwen38/eval_data/tf_ref.json run_eval.sbatch
#
#SBATCH --job-name=qwen38-eval
#SBATCH --dependency=singleton
#SBATCH --nice=10000
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --output=/data/ayerofieiev/qwen38/logs/%x-%j.log
set -euo pipefail

source "${QWEN38_ENV:-/data/ayerofieiev/qwen38/env.sh}"
: "${QWEN38_WORKSPACE:?env.sh must export QWEN38_WORKSPACE}"
: "${QWEN38_PYTHON:?env.sh must export QWEN38_PYTHON}"
: "${HF_MODEL:?env.sh must export HF_MODEL (qwen38 weights path)}"
: "${MESH_DEVICE:?env.sh must export MESH_DEVICE (P150x4 or P150x8)}"

export PYTHONUNBUFFERED=1
export TT_METAL_HOME="${TT_METAL_HOME:-$QWEN38_WORKSPACE}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$QWEN38_WORKSPACE}"
# Venv cmake/ninja first: loudbox system cmake (3.22) would kill an incremental build.
export PATH="$(dirname "$QWEN38_PYTHON"):$PATH"
cd "$QWEN38_WORKSPACE"

# Cluster policy: nothing persists on node-local disks. Home is node-local here.
sweep_local_caches() {
    rm -rf "$HOME/.cache/tt-metal-cache" "$HOME/.cache/huggingface" /tmp/"${USER}"-* 2>/dev/null || true
}
# Same teardown-wedge handling as run_bench.sbatch (tt-metal#53281).
trap 'sweep_local_caches; tt-smi -r >/dev/null 2>&1 || true' EXIT
fuser -k /dev/tenstorrent/* 2>/dev/null || true
sleep 2
tt-smi -r >/dev/null 2>&1 || echo "WARN: prologue tt-smi -r failed"

# NFS budget guard: refuse a first-time weight-cache generation when /data is scarce.
FREE_G=$(df -BG --output=avail /data 2>/dev/null | tail -1 | tr -dc '0-9')
if [[ -n "$FREE_G" && "$FREE_G" -lt 500 ]]; then
    echo "WARN: /data has only ${FREE_G}G free (<500G) — NFS budget policy active"
    CACHE_DIR="${TT_CACHE_PATH:-}/${MESH_DEVICE:-}"
    if [[ -z "${TT_CACHE_PATH:-}" || ! -d "$CACHE_DIR" || -z "$(ls -A "$CACHE_DIR" 2>/dev/null)" ]]; then
        echo "ABORT: weight cache for $MESH_DEVICE not present and NFS nearly full — skipping cache-generating run"
        exit 1
    fi
fi

if [[ -n "${REF:-}" ]]; then
    git checkout --detach "$REF" 2>/dev/null || {
        git fetch origin "$REF"
        git checkout --detach FETCH_HEAD
    }
fi
SHA=$(git rev-parse --short HEAD)
echo "== qwen38 eval gate: $SHA on $(hostname) mesh=$MESH_DEVICE =="
git log -1 --oneline

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    cmake --build build_Release -- -j4 || cmake --build build_Release -- -j4
fi

if [[ "${SKIP_GATE:-0}" != "1" ]]; then
    GATELOG="$(mktemp)"
    set +e
    "$QWEN38_PYTHON" -m pytest models/demos/blackhole/qwen36/tests/test_mlp_tp.py::test_mlp_tp -q -x 2>&1 | tee "$GATELOG"
    set -e
    grep -qE "[0-9]+ passed" "$GATELOG" || { echo "GATE_FAILED"; exit 1; }
    rm -f "$GATELOG"
fi

LOGDIR=/data/ayerofieiev/qwen38/logs
mkdir -p "$LOGDIR" /data/ayerofieiev/qwen38/eval_data
EVALLOG="$LOGDIR/eval-${SLURM_JOB_ID:-manual}.log"

# Judge the run by its EVAL_JSON lines, not the process exit code (teardown
# SIGABRT after a green suite must not fail the job) — but a FAILED assert
# inside a stage (empty responses / nondeterminism) must: keep pytest's summary.
set +e
REF="$SHA" "$QWEN38_PYTHON" -m pytest models/demos/blackhole/qwen36/campaign/eval_gate.py -v -s 2>&1 | tee "$EVALLOG"
set -e
grep -q "EVAL_JSON" "$EVALLOG" || { echo "EVAL_PRODUCED_NO_JSON"; exit 1; }
if grep -qE "[0-9]+ failed" "$EVALLOG"; then
    echo "EVAL_STAGE_FAILED (see asserts above; EVAL_JSON lines still recorded)"
    STATUS=1
else
    STATUS=0
fi

grep "EVAL_JSON" "$EVALLOG" | sed 's/^.*EVAL_JSON //' \
    >> /data/ayerofieiev/qwen38/eval_results.jsonl
exit "$STATUS"
