#!/usr/bin/env bash
set -euo pipefail
export USER=mvasiljev TT_METAL_HOME=/workspace TT_METAL_CACHE=/workspace/.cache/tt-metal-cache PYTHONPATH=/workspace
export TT_MODEL_BRINGUP_ROOT=/workspace/.codex/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.21
export TT_AUTODEBUG_ROOT=/workspace/.codex/plugins/cache/tenstorrent-skills/tt-autodebug/0.1.8
eval "$(python "$TT_MODEL_BRINGUP_ROOT/scripts/environment.py")"
base=models/autoports/qwen_qwen3_8_27b
python -m models.autoports.qwen_qwen3_8_27b.tests.probe_prefill_norm > "$base/doc/optimized_decoder/prefill_norm_probe.log" 2>&1
bash "$base/tests/validate_optimized.sh" > "$base/doc/optimized_decoder/final_validation.log" 2>&1
export QWEN_OPT_POLICY='{}'
for layer in 0 3; do
 bash "$base/tests/optimized_run.sh" profile "final_accepted_l${layer}" --layer "$layer" --recorded --trace-prefill > "$base/doc/optimized_decoder/final_accepted_l${layer}.log" 2>&1
done
python -m models.autoports.qwen_qwen3_8_27b.tests.report_optimized
