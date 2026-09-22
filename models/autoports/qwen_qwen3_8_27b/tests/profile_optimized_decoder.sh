#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
export TT_MODEL_BRINGUP_ROOT=/home/mvasiljevic/qwen38-full-rerun/codex-home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.4
export TT_AUTODEBUG_ROOT=/home/mvasiljevic/qwen38-full-rerun/codex-home/plugins/cache/tenstorrent-skills/tt-autodebug/0.1.5
BRINGUP_EXPORTS=$(python_env/bin/python "$TT_MODEL_BRINGUP_ROOT/scripts/environment.py")
eval "$BRINGUP_EXPORTS"
export PYTHONPATH=.:$PYTHONPATH
profile_name=$1
shift
profile_evidence=models/autoports/qwen_qwen3_8_27b/doc/optimized_decoder
printf '%q ' "$0" "$profile_name" "$@" >> "$profile_evidence/commands.log"
printf '\n' >> "$profile_evidence/commands.log"
sha256sum models/autoports/qwen_qwen3_8_27b/tt/optimized_decoder.py > "$profile_evidence/$profile_name.source.sha256"
timeout -k 10 900 python_env/bin/python -m tracy -r -p -v \
 -o "$profile_evidence/tracy/$profile_name" \
 -m models.autoports.qwen_qwen3_8_27b.tests.run_optimized_decoder \
 --snapshot /home/mvasiljevic/hf-cache/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
 --activations /home/mvasiljevic/qwen38-full-rerun/tt-metal-cache/optimized_decoder_activations \
 --profile --output "$profile_evidence/$profile_name.json" "$@" > "$profile_evidence/$profile_name.log" 2>&1
test -s "$profile_evidence/$profile_name.json"
profile_csv=$(find "$profile_evidence/tracy/$profile_name" -name 'ops_perf_results*.csv' | head -1)
for phase in prefill decode; do
 signpost="PERF_${phase^^}"
 python_env/bin/tt-perf-report "$profile_csv" --start-signpost "$signpost" --end-signpost "${signpost}_END" --csv "$profile_evidence/tracy/$profile_name/${phase}_perf_report.csv" > "$profile_evidence/tracy/$profile_name/${phase}_perf_report.console.log"
 python_env/bin/tt-perf-report "$profile_csv" --start-signpost "$signpost" --end-signpost "${signpost}_END" --no-summary > "$profile_evidence/tracy/$profile_name/${phase}_perf_report.txt"
done
