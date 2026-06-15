#!/bin/bash
# Cold-cache audio kernel-compile measurement harness.
#
# Runs test_audio_decode_girl -k bh_4x8sp1tp0 against a FRESH (wiped) TT_METAL_CACHE
# so every run pays the full JIT kernel compile. Emits, to stdout and to a per-tag
# log, the headline metrics:
#   - JIT cache stats: compiled (= total - hits) unique kernels, build-once dedup
#   - AUDIO_GIRL cold=<ms> warm=<ms> + conv1d-vs-torch PCC (the test's built-in gate)
#
# Usage: audio_compile_bench.sh <tag>
# Run through the broker (TT_METAL_HOME = this worktree). Cold cache = reproducible.
set -uo pipefail

TAG="${1:-baseline}"
WT=/home/smarton/tt-metal/.claude/worktrees/audio-compile-opt
CACHE_DIR="/home/smarton/.cache/audio-compile-bench-${TAG}"
LOG="/home/smarton/.cache/audio-compile-bench-${TAG}.log"

cd "$WT"
# Wipe the kernel cache so the run is fully cold (every kernel recompiles).
rm -rf "$CACHE_DIR"
mkdir -p "$CACHE_DIR"

export TT_METAL_HOME="$WT"
export PYTHONPATH="$WT"
export TT_METAL_CACHE="$CACHE_DIR"
export LTX_CHECKPOINT=/home/kevinmi/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/76730e634e70a28f4e8d51f5e29c08e40e2d8e74/ltx-2.3-22b-distilled-1.1.safetensors
export GEMMA_PATH=/home/kevinmi/.cache/huggingface/hub/models--google--gemma-3-12b-it-qat-q4_0-unquantized/snapshots/68f7ee4fbd59087436ada77ed2d62f373fdd4482/
export TT_DIT_CACHE_DIR=/home/smarton/.cache/tt-dit
export LTX_TRACED=0   # eager path runs the torch-oracle PCC gate

echo "=== audio_compile_bench tag=${TAG} cache=${CACHE_DIR} $(date -u +%Y-%m-%dT%H:%M:%SZ)UTC ===" | tee "$LOG"

"$WT/python_env/bin/python" -m pytest \
  models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_audio_decode_girl \
  -k bh_4x8sp1tp0 -s -q --timeout=0 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== METRICS tag=${TAG} ===" | tee -a "$LOG"
grep -E "JIT cache stats" "$LOG" | tee -a "$LOG.metrics" || echo "NO JIT cache stats line found" | tee -a "$LOG"
grep -E "AUDIO_GIRL" "$LOG" | tee -a "$LOG.metrics" || echo "NO AUDIO_GIRL line found" | tee -a "$LOG"
