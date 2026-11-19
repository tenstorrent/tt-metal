#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
layer=${1:?layer}
artifact=${2:?fresh artifact directory}
policy=${3:-'{}'}
batch=${4:-1}
doc=models/autoports/qwen_qwen3_6_35b_a3b/doc/multichip_decoder
case $batch in
    1)
        steps=2
        baseline="$doc/baseline_l${layer}_s32.pt"
        activation_args=()
        ;;
    32)
        echo 'Use collect_multichip_profile.sh for B32; this broad fixture exceeds Tracy source-location capacity' >&2
        exit 2 ;;
    *) echo 'Profiler batch must be 1 or 32' >&2; exit 2 ;;
esac
[[ ! -e "$artifact" ]] || { echo 'Use a fresh capture directory' >&2; exit 2; }
if fuser /dev/tenstorrent/*; then
    echo 'Target mesh is owned' >&2
    exit 2
fi
mkdir -p "$artifact"
unset TT_VISIBLE_DEVICES TT_METAL_WATCHER TT_METAL_TRACE_ALLOC_TRACKING TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES
export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1
export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=100000
cache_args=(--no-op-info-cache)
export TT_METAL_PROFILER_NO_CACHE_OP_INFO=1
timeout 1800 python -m tracy -r -p -v --no-web-server "${cache_args[@]}" -o "$artifact/raw" \
    -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_multichip_decoder \
    --layer "$layer" --length 32 --batch "$batch" --decode-steps "$steps" --repeats 16 --prefill-repeats 1 --stress-repeats 2 \
    --baseline-load "$baseline" "${activation_args[@]}" \
    --mesh-build-kwargs "$policy" --profile --output "$artifact/correctness.json" > "$artifact/capture.log" 2>&1
python - "$artifact" "$batch" <<'PY'
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
assert result['passed']
assert result['multichip']['deterministic']
assert result['batch'] == int(sys.argv[2])
PY
for phase in prefill decode; do
    signpost="PERF_${phase^^}"
    tt-perf-report "$artifact/ops.csv" --active-experts 8 --start-signpost "$signpost" --end-signpost "${signpost}_END" --csv "$artifact/${phase}_perf_report.csv" > "$artifact/${phase}_perf_report.console.log" 2>&1
    tt-perf-report "$artifact/ops.csv" --active-experts 8 --start-signpost "$signpost" --end-signpost "${signpost}_END" --no-summary --no-color > "$artifact/${phase}_perf_report.txt" 2>&1
done
