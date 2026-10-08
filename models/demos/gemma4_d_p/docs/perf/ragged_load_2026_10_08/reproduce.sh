#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Run from the repository root with python_env activated and exclusive device access.
# Usage: bash models/demos/gemma4_d_p/docs/perf/ragged_load_2026_10_08/reproduce.sh [log-directory]
set -euo pipefail

study_logs="${1:-/tmp/gemma4-ragged-load-study}"
study_output="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$study_logs"
export HF_MODEL=google/gemma-4-31B-it
export HF_HOME=/mnt/models/huggingface
export TT_CACHE_PATH=/mnt/models/huggingface/tt_cache/gemma4_d_p/google--gemma-4-31B-it
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=16 GEMMA4_PREFILL_STABLE_REDUCTIONS=0
export GEMMA4_RAGGED_TEST_LAYERS=60 GEMMA4_LOAD_REPEATS=5
unset GEMMA4_LOAD_CASES

run_logged() {
    local study_log="$1"
    shift
    echo "Running: $* (log: $study_log)"
    if ! "$@" > "$study_log" 2>&1; then
        tail -n 60 "$study_log"
        return 1
    fi
}

for chunk in 2048 4096 8192; do
    run_logged "$study_logs/canonical-$chunk.log" env GEMMA4_ACTIVATIONS_DRAM_ONLY=0 TT_METAL_TRACE_ALLOC_TRACKING=0 \
        pytest "models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_long_context_traced[blackhole-ctx_256k-chunk${chunk}-text-8x4]" -sv --timeout=3600
done

export TT_METAL_TRACE_ALLOC_TRACKING=1
for mode in regular8192 regular4096 regular2048 stable8192 ragged; do
    # The override only affects the packed four-full-request scenarios. Each result
    # records its activation storage. Without it, that shape fails the L1 allocator.
    run_logged "$study_logs/loaded-$mode.log" env GEMMA4_LOAD_MODE="$mode" \
        GEMMA4_ACTIVATIONS_DRAM_ONLY=0 GEMMA4_LOAD_FULL_DRAM=1 \
        GEMMA4_LOAD_OUTPUT="$study_logs/loaded-$mode.json" \
        pytest models/demos/gemma4_d_p/tests/test_ragged_prefill_load.py::test_ragged_prefill_under_load -sv --timeout=7200
done

for mode in regular8192 ragged; do
    dram=0
    suffix=regular8192
    if [ "$mode" = ragged ]; then dram=1; suffix=ragged-dram; fi
    run_logged "$study_logs/stream-$suffix.log" env GEMMA4_LOAD_MODE="$mode" \
        GEMMA4_LOAD_STREAM=1 GEMMA4_ACTIVATIONS_DRAM_ONLY="$dram" \
        GEMMA4_LOAD_OUTPUT="$study_logs/stream-$suffix.json" \
        pytest models/demos/gemma4_d_p/tests/test_ragged_prefill_load.py::test_full_context_loaded_stream -sv --timeout=7200
done

for mode in regular8192 stable8192; do
    # Same activation placement as the packed stream, to separate storage and
    # fixed-FP32 reduction costs from the remaining packing machinery.
    run_logged "$study_logs/stream-$mode-dram.log" env GEMMA4_LOAD_MODE="$mode" \
        GEMMA4_LOAD_STREAM=1 GEMMA4_ACTIVATIONS_DRAM_ONLY=1 \
        GEMMA4_LOAD_OUTPUT="$study_logs/stream-$mode-dram.json" \
        pytest models/demos/gemma4_d_p/tests/test_ragged_prefill_load.py::test_full_context_loaded_stream -sv --timeout=7200
done

python -m models.demos.gemma4_d_p.scripts.ragged_load_report --input "$study_logs" --output "$study_output"
