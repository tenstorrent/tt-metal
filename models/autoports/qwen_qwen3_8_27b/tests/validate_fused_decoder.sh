#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# Serial stage validation. Requires exclusive use of the local TT devices.
set -euo pipefail
fusion_model=models/autoports/qwen_qwen3_8_27b
fusion_evidence=$fusion_model/doc/fused_decoder
fusion_snapshot=/home/mvasiljevic/hf-cache/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
PYTHONPATH=. python_env/bin/python -m pytest -q "$fusion_model/tests/test_fused_decoder.py" \
 --basetemp /tmp/qwen_fused_verified_pytest \
 --junitxml "$fusion_evidence/verified_synthetic.xml" > "$fusion_evidence/verified_synthetic.log" 2>&1
for fusion_layer in 0 3; do
 for fusion_batch in 1 32; do
  fusion_name=watcher_final_l${fusion_layer}_b${fusion_batch}
  TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH="$PWD/$fusion_evidence/$fusion_name" \
   bash "$fusion_model/tests/run_fusion_experiment.sh" "$fusion_name" \
   --layer "$fusion_layer" --batch "$fusion_batch" --length 257 --continuation
 done
done
bash "$fusion_model/tests/profile_fused_decoder.sh" tracy_final
for fusion_kind in linear full; do
 fusion_layer=0
 if [[ "$fusion_kind" == full ]]; then fusion_layer=3; fi
 PYTHONPATH=. timeout -k 10 7200 python_env/bin/python "$fusion_model/tests/run_fused_context.py" \
  --snapshot "$fusion_snapshot" --layer "$fusion_layer" \
  --baseline "$fusion_model/doc/functional_decoder/${fusion_kind}_context.json" \
  --output "$fusion_evidence/verified_${fusion_kind}_context.json" \
  > "$fusion_evidence/verified_${fusion_kind}_context.log" 2>&1
done
