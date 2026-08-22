#!/usr/bin/env bash
# Canonical qwen38 measurement job. One iteration is:
#   push branch -> sbatch --export=ALL,REF=<sha-or-branch> run_bench.sbatch -> read one log.
# Serialized per user by job-name + singleton so exactly one job owns the device;
# the profile_*.sbatch jobs share the same name for the same reason.
#
# Env passthrough knobs: REF (git ref to bench; default = workspace HEAD),
# SKIP_BUILD=1 (kernel-only change, JIT recompiles), SKIP_GATE=1, RUN_PREFILL=1,
# RUN_ASYNC_GATE=1 (stock-vs-async token-equality gate before the bench),
# plus every QWEN38_BENCH_* / QWEN38_PREFILL_* knob from the bench scripts.
#
#SBATCH --job-name=qwen38-meas
#SBATCH --dependency=singleton
#SBATCH --nice=10000
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --output=/data/ayerofieiev/qwen38/logs/%x-%j.log
set -euo pipefail

# Cluster recipe: must export QWEN38_WORKSPACE (NFS tt-metal checkout with a warm
# build_Release), QWEN38_PYTHON (venv interpreter), HF_MODEL (qwen38 weights dir),
# MESH_DEVICE (P150x4 / P150x8); may export HF_HOME etc. to keep caches on NFS.
source "${QWEN38_ENV:-/data/ayerofieiev/qwen38/env.sh}"
: "${QWEN38_WORKSPACE:?env.sh must export QWEN38_WORKSPACE}"
: "${QWEN38_PYTHON:?env.sh must export QWEN38_PYTHON}"
: "${HF_MODEL:?env.sh must export HF_MODEL (qwen38 weights path)}"
: "${MESH_DEVICE:?env.sh must export MESH_DEVICE (P150x4 or P150x8)}"

export PYTHONUNBUFFERED=1
export TT_METAL_HOME="${TT_METAL_HOME:-$QWEN38_WORKSPACE}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$QWEN38_WORKSPACE}"
# Loudbox system cmake is 3.22 (< the repo's required 3.24); a ninja-triggered
# reconfigure of a tree configured elsewhere then kills the job. The venv carries
# pip cmake/ninja — put its bin dir first so the incremental build always uses them.
export PATH="$(dirname "$QWEN38_PYTHON"):$PATH"
cd "$QWEN38_WORKSPACE"

# Cluster policy: nothing persists on node-local disks. Home is node-local here.
sweep_local_caches() {
    rm -rf "$HOME/.cache/tt-metal-cache" "$HOME/.cache/huggingface" /tmp/"${USER}"-* 2>/dev/null || true
}
# tt-metal#53281: a torn-down process can leave BH eth cores unresettable and the next
# init times out ("waiting for active ethernet core XX-25"). Prologue: kill stale
# /dev/tenstorrent holders + board reset; exit trap: reset again so the node is left
# clean for whoever comes next. TT_METAL_DISABLE_MULTI_AERISC=1 (set in env.sh)
# prevents the wedge and fixes the fabric local handshake on this cluster's eth fw.
trap 'sweep_local_caches; tt-smi -r >/dev/null 2>&1 || true' EXIT
fuser -k /dev/tenstorrent/* 2>/dev/null || true
sleep 2
tt-smi -r >/dev/null 2>&1 || echo "WARN: prologue tt-smi -r failed"

# NFS budget guard: /data is a shared, nearly-full volume. A first-time weight-cache
# generation writes tens of GB — refuse it when free space is scarce.
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
echo "== qwen38 bench: $SHA on $(hostname) mesh=$MESH_DEVICE =="
git log -1 --oneline

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    # Host-code changes need the incremental build; the parallel-lld link crash
    # (thread exhaustion) is retried once — -j4 is the known-safe width.
    cmake --build build_Release -- -j4 || cmake --build build_Release -- -j4
fi

if [[ "${SKIP_GATE:-0}" != "1" ]]; then
    # Fast PCC gate on the TP mesh: loads one layer's MLP weights only. Gate on the
    # pytest summary text, not the exit code — a teardown SIGABRT after a green suite
    # (the #53281 signature) must not fail the job.
    GATELOG="$(mktemp)"
    set +e
    "$QWEN38_PYTHON" -m pytest models/demos/blackhole/qwen36/tests/test_mlp_tp.py::test_mlp_tp -q -x 2>&1 | tee "$GATELOG"
    set -e
    grep -qE "[0-9]+ passed" "$GATELOG" || { echo "GATE_FAILED"; exit 1; }
    rm -f "$GATELOG"
fi

LOGDIR=/data/ayerofieiev/qwen38/logs
mkdir -p "$LOGDIR"
BENCHLOG="$LOGDIR/bench-${SLURM_JOB_ID:-manual}.log"

if [[ "${RUN_ASYNC_GATE:-0}" == "1" ]]; then
    # Token-equality gate for QWEN36_ASYNC_DECODE_STEP (stock vs device-resident
    # stepping). This one is exit-code strict: a divergence must fail the job.
    GATELOG2="$(mktemp)"
    set +e
    "$QWEN38_PYTHON" -m pytest models/demos/blackhole/qwen36/tests/test_async_decode_step.py -v -s 2>&1 | tee "$GATELOG2"
    set -e
    grep -qE "[0-9]+ passed" "$GATELOG2" || { echo "ASYNC_GATE_FAILED"; exit 1; }
    rm -f "$GATELOG2"
fi

# Same rationale: judge the bench by its BENCH_JSON line, not the process exit code.
set +e
REF="$SHA" "$QWEN38_PYTHON" -m pytest models/demos/blackhole/qwen36/campaign/bench_decode.py -v -s 2>&1 | tee "$BENCHLOG"
if [[ "${RUN_PREFILL:-0}" == "1" ]]; then
    REF="$SHA" "$QWEN38_PYTHON" -m pytest models/demos/blackhole/qwen36/campaign/bench_prefill.py -v -s 2>&1 | tee -a "$BENCHLOG"
fi
set -e
grep -q "BENCH_JSON" "$BENCHLOG" || { echo "BENCH_PRODUCED_NO_JSON"; exit 1; }

"$QWEN38_PYTHON" models/demos/blackhole/qwen36/campaign/parse_bench.py "$BENCHLOG" \
    --ref "$SHA" --node "${SLURMD_NODENAME:-$(hostname)}" \
    --append /data/ayerofieiev/qwen38/results.jsonl
