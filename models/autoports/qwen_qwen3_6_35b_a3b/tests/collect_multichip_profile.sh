#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Narrow B32 capture: unprofiled paired seeds avoid Tracy's source-location cap.
set -euo pipefail
layer=${1:?layer}
artifact=${2:?fresh artifact directory}
doc=models/autoports/qwen_qwen3_6_35b_a3b/doc/multichip_decoder
seed=${3:-"$doc/seed_l${layer}_b32.pt"}
[[ ! -e "$artifact" ]] || { echo 'Use a fresh capture directory' >&2; exit 2; }
[[ -f "$seed" ]] || { echo 'Missing paired prefill-state seed' >&2; exit 2; }
if fuser /dev/tenstorrent/*; then echo 'Target mesh is owned' >&2; exit 2; fi
mkdir -p "$artifact"
unset TT_VISIBLE_DEVICES TT_METAL_WATCHER TT_METAL_TRACE_ALLOC_TRACKING TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE
unset TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES TT_METAL_PROFILER_NO_CACHE_OP_INFO
export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1
export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=100000
export TTNN_OP_PROFILER=1 TT_METAL_PROFILER_TRACE_TRACKING=1
timeout 1800 python -m tracy -p -v --no-web-server -o "$artifact/raw" \
    -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_multichip_profile \
    --layer "$layer" --seed "$seed" --output "$artifact/correctness.json" \
    > "$artifact/capture.log" 2>&1
timeout 1800 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.postprocess_multichip_profile \
    --artifact "$artifact" > "$artifact/postprocess.log" 2>&1
python - "$artifact" <<'PY'
import json
import pathlib
import shutil
import sys
p = pathlib.Path(sys.argv[1])
files = list((p / 'postprocess/reports').rglob('ops_perf_results_*.csv'))
assert len(files) == 1, files
shutil.copy2(files[0], p / 'ops.csv')
(p / 'ops_source.txt').write_text(str(files[0]) + '\n')
result = json.loads((p / 'correctness.json').read_text())
assert result['passed'] and result['multichip']['deterministic']
assert result['batch'] == 32 and result['profile_fixture'] == 'seeded_b32_v1'
PY
for phase in prefill decode; do
    signpost="PERF_${phase^^}"
    tt-perf-report "$artifact/ops.csv" --active-experts 8 --start-signpost "$signpost" --end-signpost "${signpost}_END" --csv "$artifact/${phase}_perf_report.csv" > "$artifact/${phase}_perf_report.console.log" 2>&1
    tt-perf-report "$artifact/ops.csv" --active-experts 8 --start-signpost "$signpost" --end-signpost "${signpost}_END" --no-summary --no-color > "$artifact/${phase}_perf_report.txt" 2>&1
done
