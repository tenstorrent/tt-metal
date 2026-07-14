#!/usr/bin/env bash
# Production-scale slow PCC gate for HunyuanImage-3.0.
#
# Smoke (fast):  pytest tests/pcc/ -m "not slow and not unit_host and not e2e_random_inputs"
# Production (slow): this script — 32L, GRID=64, S=4160, S=22784 max-context, submodule gates,
# plus full-dim support: vision S=1024/27L, on-device AR full vocab, scheduler 64×64, timestep×50.
#
# Usage (from tt-metal repo root):
#   bash models/experimental/hunyuan_image_3_0/tests/run_pcc_production_slow.sh
#
# Backbone-only (skip vision/AR/scheduler full-dim section):
#   HY_SKIP_FULL_DIM_SUPPORT=1 bash models/experimental/hunyuan_image_3_0/tests/run_pcc_production_slow.sh
#
# Opt-in E2E with random inputs (not part of production gate):
#   HY_RUN_E2E_RANDOM=1 python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_pipeline.py -k e2e_pipeline -v -s
#
# Requires: checkpoint at HUNYUAN_MODEL_DIR (or HF cache auto-download).
# Instruct checkpoint for recaption / generate_device (HUNYUAN_INSTRUCT_MODEL_DIR).
# Run one device pytest job at a time — concurrent runs cause device timeout.

set -euo pipefail

cd "$(dirname "$0")/../../../.."

PY="${PY:-python_env/bin/python}"
export HY_NUM_LAYERS="${HY_NUM_LAYERS:-32}"

K_FILTER="\
production_32l or production_64 or \
logit_stack_production or logit_stack_max_context or \
moe_module_production or moe_router_production or \
moe_module_max_context or moe_router_max_context or \
denoise_loop_production or \
all_layers_production or final_production or denoise_step_production_32l or \
rms_norm_production or rms_norm_max_context or \
rope_2d_production or mask_production or \
attention_production or attention_max_context or \
decoder_layer_production or decoder_layer_max_context or \
wte_production or recaption_production or \
lm_head_production or i2i_denoise_step_production"

echo "=== HunyuanImage-3.0 production slow PCC (HY_NUM_LAYERS=${HY_NUM_LAYERS}) ==="

"${PY}" -m pytest \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_teacher_forced.py \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_pipeline.py \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_logit_stack.py \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_moe.py \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_full_dim_moe_denoise.py \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_embeddings.py \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_attention_modules.py \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_transformer.py \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_recaption.py \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_lm_head.py \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_denoise.py \
  -m slow \
  -k "${K_FILTER}" \
  -v -s \
  --timeout=43200

if [[ "${HY_SKIP_FULL_DIM_SUPPORT:-0}" != "1" ]]; then
  echo ""
  echo "=== Full-dimension support gates (vision / AR / scheduler / timestep) ==="
  K_FULL_DIM="full_dim or generate_device_production or full_latent or full_schedule"
  "${PY}" -m pytest \
    models/experimental/hunyuan_image_3_0/tests/vision/test_siglip2_full_dim.py \
    models/experimental/hunyuan_image_3_0/tests/pcc/test_generate_device.py \
    models/experimental/hunyuan_image_3_0/tests/pcc/test_scheduler.py \
    models/experimental/hunyuan_image_3_0/tests/pcc/test_embeddings.py \
    -m slow \
    -k "${K_FULL_DIM}" \
    -v -s \
    --timeout=10800
fi

echo ""
echo "Production slow PCC gate passed."
