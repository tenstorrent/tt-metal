#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Run from the configured model-bringup environment. No background device jobs.
# Usage: bash collect_optimized_multichip_profiles.sh [STAGE] [B1_BASELINE_ROOT]
# Requires current passing STAGE/final_l{0,3}_b32.{pt,json,json.provenance.json}.
# Create STAGE/STOP_PROFILES to stop between jobs; never kill an active capture.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$script_dir" rev-parse --show-toplevel)
cd "$repo"
model=models/autoports/qwen_qwen3_6_35b_a3b
stage=$(realpath -m "${1:-$model/doc/optimized_multichip_decoder}")
baseline_root=$(realpath -m "${2:-$model/doc/multichip_decoder}")
module=models.autoports.qwen_qwen3_6_35b_a3b.tests

# Validate all destinations and seeds before the first hardware-facing command.
for layer in 0 3; do
    for artifact in "$stage/tracy/layer_$layer" "$stage/perf_b32/tracy/layer_$layer"; do
        [[ ! -e "$artifact" ]] || { echo "Use fresh profile roots; already exists: $artifact" >&2; exit 2; }
    done
    for input in "$baseline_root/baseline_l${layer}_s32.pt" \
        "$stage/final_l${layer}_b32.pt" "$stage/final_l${layer}_b32.json" \
        "$stage/final_l${layer}_b32.json.provenance.json"; do
        [[ -f "$input" ]] || { echo "Missing required profile input: $input" >&2; exit 2; }
    done
done
mkdir -p "$stage/profile_logs"
commands="$stage/profile_commands.log"
[[ ! -e "$commands" ]] || { echo "Preserve the existing command log; use a fresh stage root" >&2; exit 2; }

unset TT_VISIBLE_DEVICES TT_METAL_WATCHER TT_METAL_TRACE_ALLOC_TRACKING TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE
unset TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES TT_METAL_PROFILER_NO_CACHE_OP_INFO
export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1
export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=100000
export TTNN_OP_PROFILER=1 TT_METAL_PROFILER_TRACE_TRACKING=1

run_logged() {
    local log=$1 status=0
    shift
    { date -u '+%Y-%m-%dT%H:%M:%SZ'; printf '%q ' "$@"; printf '\nlog=%q\n' "$log"; } >> "$commands"
    "$@" > "$log" 2>&1 || status=$?
    printf 'exit_status=%s\n' "$status" >> "$commands"
    if (( status != 0 )); then
        echo "Profile job failed ($status); inspect $log. No retry/reset was attempted." >&2
        return "$status"
    fi
}

mesh_available() {
    [[ ! -e "$stage/STOP_PROFILES" ]] || { echo 'Stopped between profile jobs' >&2; exit 75; }
    local devices=(/dev/tenstorrent/*) status=0
    [[ -e "${devices[0]}" ]] || { echo 'No Tenstorrent devices present' >&2; exit 2; }
    timeout 30 fuser "${devices[@]}" > "$stage/profile_logs/fuser_$1.log" 2>&1 || status=$?
    printf 'fuser_%s_exit_status=%s\n' "$1" "$status" >> "$commands"
    if (( status != 1 )); then
        echo "Mesh is owned or ownership check failed ($status); inspect fuser_$1.log" >&2
        exit 2
    fi
}

run_logged "$stage/profile_logs/manifest.log" timeout 60 python - "$stage" "$baseline_root" <<'PY'
import json
import os
import pathlib
import subprocess
import sys
import time
stage, baseline = map(pathlib.Path, sys.argv[1:])
data = dict(
    timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    git_branch=subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
    git_status=subprocess.check_output(['git', 'status', '--short'], text=True).splitlines(),
    regime='serial; target Blackhole 1x4 mesh; no watcher',
    roots={'batch1': str(stage), 'batch32': str(stage / 'perf_b32')},
    b1_baselines=[str(baseline / f'baseline_l{layer}_s32.pt') for layer in (0, 3)],
    b32_seeds=[str(stage / f'final_l{layer}_b32.pt') for layer in (0, 3)],
    environment={k: v for k, v in os.environ.items() if k.startswith(('TT_METAL_', 'TTNN_', 'TT_VISIBLE_'))},
    provenance='Each runner writes correctness.json.provenance.json; commands are in profile_commands.log',
    b1_fixture='functional_debug command: length32, B1, 2 changing steps, 16 profiled replays, unchanged defaults',
    b32_fixture='existing collect_multichip_profile.sh; current passing final_lX_b32 paired seeds',
)
(stage / 'profile_run_manifest.json').write_text(json.dumps(data, indent=2) + '\n')
PY

for layer in 0 3; do
    artifact="$stage/tracy/layer_$layer"
    mesh_available "b1_l$layer"
    mkdir -p "$artifact"
    run_logged "$artifact/capture.log" timeout 1800 env TT_METAL_PROFILER_NO_CACHE_OP_INFO=1 \
        python -m tracy -p -v --no-web-server -o "$artifact/raw" \
        -m "$module.run_multichip_decoder" --layer "$layer" --length 32 --batch 1 \
        --decode-steps 2 --repeats 16 --prefill-repeats 1 --stress-repeats 2 \
        --baseline-load "$baseline_root/baseline_l${layer}_s32.pt" \
        --mesh-build-kwargs '{}' --profile --output "$artifact/correctness.json"
    run_logged "$artifact/postprocess.log" timeout 1800 python -m "$module.postprocess_multichip_profile" \
        --artifact "$artifact"
    run_logged "$artifact/validate.log" timeout 60 python - "$artifact" <<'PY'
import json
import pathlib
import shutil
import sys
p = pathlib.Path(sys.argv[1])
files = list((p / 'postprocess/reports').rglob('ops_perf_results_*.csv'))
assert len(files) == 1, files
result = json.loads((p / 'correctness.json').read_text())
assert result['passed'] and result['multichip']['deterministic']
assert result['batch'] == 1 and result['length'] == 32 and result['decode_steps'] == 2
shutil.copy2(files[0], p / 'ops.csv')
(p / 'ops_source.txt').write_text(str(files[0]) + '\n')
PY
    for phase in prefill decode; do
        signpost="PERF_${phase^^}"
        run_logged "$artifact/${phase}_perf_report.console.log" timeout 300 tt-perf-report "$artifact/ops.csv" \
            --active-experts 8 --start-signpost "$signpost" --end-signpost "${signpost}_END" \
            --csv "$artifact/${phase}_perf_report.csv"
        run_logged "$artifact/${phase}_perf_report.txt" timeout 300 tt-perf-report "$artifact/ops.csv" \
            --active-experts 8 --start-signpost "$signpost" --end-signpost "${signpost}_END" --no-summary --no-color
    done
done

for layer in 0 3; do
    mesh_available "b32_l$layer"
    # The child enforces its own 1800s capture and 1800s processing bounds.
    # The outer bound also covers report commands; -x preserves expanded commands.
    run_logged "$stage/profile_logs/b32_l${layer}.log" timeout 4500 bash -x \
        "$script_dir/collect_multichip_profile.sh" "$layer" "$stage/perf_b32/tracy/layer_$layer" \
        "$stage/final_l${layer}_b32.pt"
done

run_logged "$stage/profile_logs/summary_b1.log" timeout 300 python -m "$module.summarize_multichip_perf" \
    --doc "$stage" --batch 1
run_logged "$stage/profile_logs/summary_b32.log" timeout 300 python -m "$module.summarize_multichip_perf" \
    --doc "$stage/perf_b32" --batch 32
run_logged "$stage/profile_logs/moe_share.log" timeout 300 python -m "$module.summarize_multichip_moe_share" --doc "$stage"
echo "Serial final profiles and MoE attribution complete: $stage"
