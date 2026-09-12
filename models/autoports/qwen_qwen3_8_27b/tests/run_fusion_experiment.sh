#!/usr/bin/env bash
set -euo pipefail
export TT_MODEL_BRINGUP_ROOT=/home/mvasiljevic/qwen38-full-rerun/codex-home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.4
export TT_AUTODEBUG_ROOT=/home/mvasiljevic/qwen38-full-rerun/codex-home/plugins/cache/tenstorrent-skills/tt-autodebug/0.1.5
BRINGUP_EXPORTS=$(python_env/bin/python "$TT_MODEL_BRINGUP_ROOT/scripts/environment.py")
eval "$BRINGUP_EXPORTS"
export PYTHONPATH=.:$PYTHONPATH
fusion_name=$1
shift
fusion_evidence=models/autoports/qwen_qwen3_8_27b/doc/fused_decoder
printf '%q ' "$0" "$fusion_name" "$@" >> "$fusion_evidence/commands.log"
printf '\n' >> "$fusion_evidence/commands.log"
sha256sum models/autoports/qwen_qwen3_8_27b/tt/fused_decoder.py > "$fusion_evidence/$fusion_name.source.sha256"
timeout -k 10 900 python_env/bin/python models/autoports/qwen_qwen3_8_27b/tests/run_fused_decoder.py \
 --snapshot /home/mvasiljevic/hf-cache/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
 --output "$fusion_evidence/$fusion_name.json" "$@" > "$fusion_evidence/$fusion_name.log" 2>&1
