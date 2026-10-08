#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Run from the repository root on Blackhole 8x4 with a Tracy-enabled build.
# tt-perf-report 1.4.0 must be importable by the active Python environment.
export HF_MODEL=google/gemma-4-31B-it
export HF_HOME=/mnt/models/huggingface
export TT_CACHE_PATH=/mnt/models/huggingface/tt_cache/gemma4_d_p/google--gemma-4-31B-it
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=16
export GEMMA4_PREFILL_STABLE_REDUCTIONS=0 GEMMA4_ACTIVATIONS_DRAM_ONLY=0
export GEMMA4_LAYER_PERF_CHUNKS=0,31 GEMMA4_LAYER_PERF_SLOTS=4
# This existing harness retains prepared tensors between two traces. The
# allocation tracker rejects that setup; these profiles used its default of 0.
export TT_METAL_TRACE_ALLOC_TRACKING=0
export GEMMA4_LAYER_PROFILE_ROOT="${GEMMA4_LAYER_PROFILE_ROOT:-/tmp/gemma4-layer-profile}"
mkdir -p "$GEMMA4_LAYER_PROFILE_ROOT"

# Global layers first, then sliding-window layers. Never overlap hardware runs.
for layer in global local; do
    for mode in chunked ragged; do
        export GEMMA4_LAYER_PERF_RAGGED_LENGTHS=
        if [ "$mode" = ragged ]; then
            export GEMMA4_LAYER_PERF_RAGGED_LENGTHS=2048,2048,2048,2048
        fi
        case_dir="$GEMMA4_LAYER_PROFILE_ROOT/$layer-$mode"
        export PREFILL_SUMMARIES="$case_dir/summaries"
        python -m tracy -p -r -v --no-web-server -o "$case_dir/profiler" --op-support-count 4000 \
            -m "pytest models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_layer_perf_chunk_n[blackhole-chunkall-$layer-sz8192-ctx_256k-8x4] -sv" \
            > "$GEMMA4_LAYER_PROFILE_ROOT/$layer-$mode.log" 2>&1
        python models/demos/gemma4_d_p/scripts/layer_perf_report.py \
            --profiler-dir "$case_dir/profiler" --root "$PREFILL_SUMMARIES" \
            > "$GEMMA4_LAYER_PROFILE_ROOT/$layer-$mode-report.log" 2>&1
    done
done

python -m models.demos.gemma4_d_p.scripts.ragged_layer_profile_report \
    --input-root "$GEMMA4_LAYER_PROFILE_ROOT" \
    --full-model models/demos/gemma4_d_p/docs/perf/ragged_8k_reduce_scatter_2026_10_08/measurements.json \
    --output "$GEMMA4_LAYER_PROFILE_ROOT/report"

# To redraw the committed figures without hardware, use:
# python -m models.demos.gemma4_d_p.scripts.ragged_layer_profile_report \
#   --measurements models/demos/gemma4_d_p/docs/perf/ragged_8k_layer_profile_2026_10_08/measurements.json \
#   --output /tmp/gemma4-layer-profile-redrawn
