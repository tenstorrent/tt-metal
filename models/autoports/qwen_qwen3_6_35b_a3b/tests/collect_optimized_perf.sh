#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
layer=${1:?layer}
artifact=${2:?fresh artifact directory}
policy=${3:-'{}'}
[[ ! -e "$artifact" ]] || { echo 'Use a fresh capture directory' >&2; exit 2; }
if fuser /dev/tenstorrent/0; then
    echo 'Chip 0 is owned' >&2
    exit 2
fi
mkdir -p "$artifact"
export QWEN_DECODER=optimized
export QWEN_PROFILE_POLICY="$policy"
unset TT_VISIBLE_DEVICES TT_METAL_WATCHER TT_METAL_TRACE_ALLOC_TRACKING TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES
export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1
export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=100000 TT_METAL_PROFILER_NO_CACHE_OP_INFO=1
timeout 1800 python -m tracy -r -p -v --no-web-server --no-op-info-cache -o "$artifact/raw" \
    -m models.autoports.qwen_qwen3_6_35b_a3b.tests.profile_optimized_decoder \
    --layer "$layer" --output "$artifact/correctness.json" > "$artifact/capture.log" 2>&1
python - "$artifact" <<'PY'
import json
import pathlib
import shutil
import sys
p = pathlib.Path(sys.argv[1])
files = list((p / 'raw').rglob('ops_perf_results_*.csv'))
assert len(files) == 1, files
shutil.copy2(files[0], p / 'ops.csv')
(p / 'ops_source.txt').write_text(str(files[0]) + '\n')
result = json.loads((p / 'correctness.json').read_text())
assert result['decoder'] == 'OptimizedDecoder'
assert result['prefill_pcc'] >= 0.995 and result['decode_traced_pcc'] >= 0.995
assert result['deterministic'] and result['profile_decode_replays'] == 16
PY
for phase in prefill decode; do
    signpost="PERF_${phase^^}"
    tt-perf-report "$artifact/ops.csv" --active-experts 8 --start-signpost "$signpost" --end-signpost "${signpost}_END" --csv "$artifact/${phase}_perf_report.csv" > "$artifact/${phase}_perf_report.console.log" 2>&1
    tt-perf-report "$artifact/ops.csv" --active-experts 8 --start-signpost "$signpost" --end-signpost "${signpost}_END" --no-summary --no-color > "$artifact/${phase}_perf_report.txt" 2>&1
done
