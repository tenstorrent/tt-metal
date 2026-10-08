#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Run from the repository root on Blackhole 8x4. Device tests run sequentially.
export HF_MODEL=google/gemma-4-31B-it
export HF_HOME=/mnt/models/huggingface
export TT_CACHE_PATH=/mnt/models/huggingface/tt_cache/gemma4_d_p/google--gemma-4-31B-it
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=16
export GEMMA4_PREFILL_STABLE_REDUCTIONS=0 GEMMA4_ACTIVATIONS_DRAM_ONLY=0
export GEMMA4_LOAD_FULL_DRAM=0 TT_METAL_TRACE_ALLOC_TRACKING=1
export GEMMA4_LOAD_REPEATS=5 GEMMA4_RAGGED_TEST_LAYERS=60
export GEMMA4_LOAD_CASES=single_early,tails2_early,tails4_early,single_late,tails2_late,tails4_late,tails4_one_late,tails4_two_late,tails4_three_late,tails2_mixed,tails4_mixed
export GEMMA4_REMEASURE_DIR="${GEMMA4_REMEASURE_DIR:-/tmp/gemma4-ragged-reduce-scatter}"
mkdir -p "$GEMMA4_REMEASURE_DIR"

for mode in regular8192 ragged; do
    GEMMA4_LOAD_MODE="$mode" GEMMA4_LOAD_OUTPUT="$GEMMA4_REMEASURE_DIR/loaded-$mode.json" \
        pytest models/demos/gemma4_d_p/tests/test_ragged_prefill_load.py::test_ragged_prefill_under_load \
        -sv --timeout=7200 > "$GEMMA4_REMEASURE_DIR/loaded-$mode.log" 2>&1
done

python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["GEMMA4_REMEASURE_DIR"])
loaded = {mode: json.loads((root / f"loaded-{mode}.json").read_text()) for mode in ("regular8192", "ragged")}
(root / "input.json").write_text(json.dumps({"loaded": loaded}, indent=2) + "\n")
PY

python -m models.demos.gemma4_d_p.scripts.ragged_8k_report \
    --input "$GEMMA4_REMEASURE_DIR/input.json" \
    --before models/demos/gemma4_d_p/docs/perf/ragged_8k_2026_10_08/measurements.json \
    --output "$GEMMA4_REMEASURE_DIR/report"

# Check the changed arithmetic using the existing numerical thresholds.
GEMMA4_RAGGED_TEST_LAYERS=6 pytest \
    models/demos/gemma4_d_p/tests/test_ragged_prefill.py::test_packed_requests_outputs_cache_replay_and_timing \
    -sv --timeout=3600 > "$GEMMA4_REMEASURE_DIR/correctness-6-layer.log" 2>&1

GEMMA4_RAGGED_TEST_LAYERS=60 pytest \
    models/demos/gemma4_d_p/tests/test_ragged_prefill.py::test_packed_requests_outputs_cache_replay_and_timing \
    -sv --timeout=3600 > "$GEMMA4_REMEASURE_DIR/correctness-60-layer.log" 2>&1
