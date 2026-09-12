#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
export TT_MODEL_BRINGUP_ROOT=/home/mvasiljevic/qwen38-full-rerun/codex-home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.4
export TT_AUTODEBUG_ROOT=/home/mvasiljevic/qwen38-full-rerun/codex-home/plugins/cache/tenstorrent-skills/tt-autodebug/0.1.5
BRINGUP_EXPORTS=$(python_env/bin/python "$TT_MODEL_BRINGUP_ROOT/scripts/environment.py")
eval "$BRINGUP_EXPORTS"
export PYTHONPATH=.:$PYTHONPATH
optimization_name=$1
shift
optimization_evidence=models/autoports/qwen_qwen3_8_27b/doc/optimized_decoder
printf '%q ' "$0" "$optimization_name" "$@" >> "$optimization_evidence/commands.log"
printf '\n' >> "$optimization_evidence/commands.log"
sha256sum models/autoports/qwen_qwen3_8_27b/tt/optimized_decoder.py > "$optimization_evidence/$optimization_name.source.sha256"
mkdir -p "$optimization_evidence/sources"
optimization_hash=$(sha256sum models/autoports/qwen_qwen3_8_27b/tt/optimized_decoder.py | cut -d ' ' -f 1)
cp models/autoports/qwen_qwen3_8_27b/tt/optimized_decoder.py "$optimization_evidence/sources/$optimization_hash.py.txt"
timeout -k 10 900 python_env/bin/python models/autoports/qwen_qwen3_8_27b/tests/run_optimized_decoder.py \
 --snapshot /home/mvasiljevic/hf-cache/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
 --output "$optimization_evidence/$optimization_name.json" "$@" > "$optimization_evidence/$optimization_name.log" 2>&1
