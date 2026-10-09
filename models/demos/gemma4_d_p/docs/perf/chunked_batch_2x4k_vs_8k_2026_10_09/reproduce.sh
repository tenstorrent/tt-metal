#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
repo_root="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$repo_root"
export HF_MODEL="${HF_MODEL:-google/gemma-4-31B-it}"
export HF_HOME="${HF_HOME:-/mnt/models/huggingface}"
export TT_CACHE_PATH="${TT_CACHE_PATH:-/mnt/models/huggingface/tt_cache/gemma4_d_p/google--gemma-4-31B-it}"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=16 TT_METAL_TRACE_ALLOC_TRACKING=1
export GEMMA4_CANONICAL_CHUNK_SIZE=8192 GEMMA4_BATCH_TEST_SHAPE=2x4k
export GEMMA4_BATCH_PERF_CONTEXT=262144 GEMMA4_BATCH_PERF_REPEATS=5
study_output="${GEMMA4_STUDY_OUTPUT:-/tmp/gemma4-chunked-batching-2x4k-reproduction}"
mkdir -p "$study_output"
GEMMA4_BATCH_TEST_LAYERS=6 pytest models/demos/gemma4_d_p/tests/test_chunked_batch.py -sv --timeout=7200 |& tee "$study_output/correctness.log"
pytest 'models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_long_context_traced[blackhole-ctx_256k-chunk8192-text-8x4]' -sv --timeout=7200 |& tee "$study_output/canonical-official.log"
for mode in canonical chunked2; do
    GEMMA4_BATCH_TEST_LAYERS=60 GEMMA4_BATCH_PERF_MODE="$mode" GEMMA4_BATCH_PERF_OUTPUT="$study_output/$mode.json" \
        pytest models/demos/gemma4_d_p/tests/test_chunked_batch_perf.py -sv --timeout=7200 |& tee "$study_output/$mode.log"
done
# Isolated layers use synthetic KV, matching test_prefill_layer_perf_chunk_n.
export TT_METAL_TRACE_ALLOC_TRACKING=0 TRACY_NO_WEB_SERVER=1 GEMMA4_LAYER_PERF_REPEATS=5
for mode in canonical chunked2; do
    export GEMMA4_LAYER_BATCH_MODE="$mode"
    export GEMMA4_LAYER_PERF_CHUNKS=0,31
    if [[ "$mode" == chunked2 ]]; then
        export GEMMA4_LAYER_PERF_CHUNKS=0,62,63
    fi
    export PREFILL_SUMMARIES="$study_output/layer-profile/$mode/summaries"
    python -m tracy -p -r -v --no-web-server --disable-device-data-push-to-tracy --check-exit-code --op-support-count 20000 \
        -o "$study_output/layer-profile/$mode/profiler" \
        -m 'pytest models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_layer_perf_chunk_n[blackhole-chunkall-both-sz8192-ctx_256k-8x4] -sv --timeout=7200'
done
perf_report_repo="${TT_PERF_REPORT_REPO:-$study_output/tt-perf-report-main}"
if [[ ! -d "$perf_report_repo/.git" ]]; then
    git clone --depth 1 --branch main https://github.com/tenstorrent/tt-perf-report.git "$perf_report_repo"
fi
python models/demos/gemma4_d_p/scripts/chunked_batch_layer_report.py \
    --batch-mode chunked2 --input "$study_output/layer-profile" --output "$study_output/report" --report-repo "$perf_report_repo"
python models/demos/gemma4_d_p/scripts/chunked_batch_report.py \
    --batch-mode chunked2 --input "$study_output" --output "$study_output/report" \
    --official-log "$study_output/canonical-official.log" --source-commit "$(git rev-parse HEAD)" \
    --layers "$study_output/report/layer_measurements.json"
