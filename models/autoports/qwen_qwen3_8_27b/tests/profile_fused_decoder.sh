#!/usr/bin/env bash
set -euo pipefail
export TT_MODEL_BRINGUP_ROOT=/home/mvasiljevic/qwen38-full-rerun/codex-home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.4
export TT_AUTODEBUG_ROOT=/home/mvasiljevic/qwen38-full-rerun/codex-home/plugins/cache/tenstorrent-skills/tt-autodebug/0.1.5
BRINGUP_EXPORTS=$(python_env/bin/python "$TT_MODEL_BRINGUP_ROOT/scripts/environment.py")
eval "$BRINGUP_EXPORTS"
export PYTHONPATH=.:$PYTHONPATH
fusion_evidence=models/autoports/qwen_qwen3_8_27b/doc/fused_decoder
fusion_profile_root=${1:-tracy_final}
for fusion_variant in baseline fused; do
 for fusion_layer in 0 3; do
  fusion_args=()
  if [[ "$fusion_variant" == baseline ]]; then fusion_args+=(--baseline); fi
  fusion_name=profile_${fusion_profile_root}_${fusion_variant}_l${fusion_layer}
  fusion_source=fused_decoder
  if [[ "$fusion_variant" == baseline ]]; then fusion_source=functional_decoder; fi
  sha256sum "models/autoports/qwen_qwen3_8_27b/tt/$fusion_source.py" > "$fusion_evidence/$fusion_name.source.sha256"
  timeout -k 10 900 python_env/bin/python -m tracy -r -p -v \
   -o "$fusion_evidence/$fusion_profile_root/${fusion_variant}_l${fusion_layer}" \
   -m models.autoports.qwen_qwen3_8_27b.tests.run_fused_decoder \
   --snapshot /home/mvasiljevic/hf-cache/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
   --layer "$fusion_layer" --length 128 --profile "${fusion_args[@]}" \
   --output "$fusion_evidence/$fusion_name.json" > "$fusion_evidence/$fusion_name.log" 2>&1
 done
done
