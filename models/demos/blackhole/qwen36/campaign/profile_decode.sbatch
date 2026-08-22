#!/usr/bin/env bash
# Tracy device-profile capture of decode steps (eager, per-op rows), tarred to
# /data/ayerofieiev/qwen38/profiles/<timestamp>/.
#
# Requires a Tracy build. ENABLE_TRACY goes sticky-OFF in CMakeCache after any
# --disable-profiler build, so keep a SEPARATE profiler tree and never mix:
#   ./build_metal.sh --enable-profiler --build-dir build_Release_tracy
# (or cmake -B build_Release_tracy -DENABLE_TRACY=ON ... && cmake --build ...),
# then export QWEN38_TRACY_BUILD_DIR=build_Release_tracy in env.sh.
#
# Same job name as run_bench.sbatch: singleton serializes all device owners.
#
#SBATCH --job-name=qwen38-meas
#SBATCH --dependency=singleton
#SBATCH --nice=10000
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --output=/data/ayerofieiev/qwen38/logs/%x-prof-%j.log
set -euo pipefail

source "${QWEN38_ENV:-/data/ayerofieiev/qwen38/env.sh}"
: "${QWEN38_WORKSPACE:?env.sh must export QWEN38_WORKSPACE}"
: "${QWEN38_PYTHON:?env.sh must export QWEN38_PYTHON}"

export PYTHONUNBUFFERED=1
export TT_METAL_HOME="${TT_METAL_HOME:-$QWEN38_WORKSPACE}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$QWEN38_WORKSPACE}"
cd "$QWEN38_WORKSPACE"

BUILD_DIR="${QWEN38_TRACY_BUILD_DIR:-build_Release}"
grep -q "ENABLE_TRACY:BOOL=ON" "$BUILD_DIR/CMakeCache.txt" || {
    echo "FATAL: $BUILD_DIR has ENABLE_TRACY OFF (sticky after a --disable-profiler build)."
    echo "Build a separate profiler tree and set QWEN38_TRACY_BUILD_DIR — see header."
    exit 2
}

# Point the runtime at the profiler build; restore the measurement build on exit
# so a following qwen38-meas job never silently runs a profiler binary.
restore() {
    ln -sfn build_Release build
    ln -sf "$QWEN38_WORKSPACE/build/ttnn/_ttnn.so" ttnn/ttnn/_ttnn.so
    ln -sf "$QWEN38_WORKSPACE/build/ttnn/_ttnncpp.so" ttnn/ttnn/_ttnncpp.so
    rm -rf "$HOME/.cache/tt-metal-cache" "$HOME/.cache/huggingface" /tmp/"${USER}"-* 2>/dev/null || true
}
trap restore EXIT
ln -sfn "$BUILD_DIR" build
ln -sf "$QWEN38_WORKSPACE/build/ttnn/_ttnn.so" ttnn/ttnn/_ttnn.so
ln -sf "$QWEN38_WORKSPACE/build/ttnn/_ttnncpp.so" ttnn/ttnn/_ttnncpp.so

export TT_METAL_DEVICE_PROFILER=1
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=/data/ayerofieiev/qwen38/profiles/$STAMP
mkdir -p "$OUT"
rm -rf generated/profiler

# Eager mode gives per-op rows for the decode-step attribution (which of GDN
# decode / full-attn decode / MLP / CCL / sampling dominates). Synth state skips
# the prefill: op timing is content-independent, only positions matter.
QWEN38_BENCH_SYNTH_STATE="${QWEN38_BENCH_SYNTH_STATE:-1}" \
QWEN38_BENCH_MODE="${QWEN38_BENCH_MODE:-eager}" \
QWEN38_BENCH_STEPS="${QWEN38_BENCH_STEPS:-4}" \
QWEN38_BENCH_WARMUP="${QWEN38_BENCH_WARMUP:-2}" \
    "$QWEN38_PYTHON" -m pytest models/demos/blackhole/qwen36/campaign/bench_decode.py -v -s \
    2>&1 | tee "$OUT/profile-decode.log"

tar czf "$OUT/profiler-decode.tar.gz" -C "$QWEN38_WORKSPACE" generated/profiler
echo "PROFILE_OUT $OUT"
echo "analyze offline: extract and run tt-perf-report on generated/profiler/reports/**/ops_perf_results_*.csv"
