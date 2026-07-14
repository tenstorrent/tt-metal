#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Run encoder conv3d blocking sweeps for ResolutionGroup presets.
#
# Usage:
#   ./scripts/run_encoder_preset_sweeps.sh priority   # 7 common aspect ratios (~6-8h)
#   ./scripts/run_encoder_preset_sweeps.sh all        # all 37 presets (~50-70h)
#   ./scripts/run_encoder_preset_sweeps.sh 1216x832,1280x720
#
# After sweeps finish:
#   python_env/bin/python models/experimental/hunyuan_image_3_0/scripts/apply_encoder_sweep_blockings.py

set -euo pipefail
cd "$(dirname "$0")/../../.."

PRESETS="${1:-priority}"
LOG_DIR="models/experimental/hunyuan_image_3_0/sweep_results"
mkdir -p "$LOG_DIR"

export CONV3D_SWEEP_HW_PRODUCT=none
export CONV3D_SWEEP_MAX_COMBOS=800
export HY_CONV3D_SWEEP_SKIP_EXISTING=1
export HY_CONV3D_SWEEP_PRESETS="$PRESETS"

LOG="$LOG_DIR/sweep_${PRESETS//,/_}_$(date +%Y%m%d_%H%M%S).log"
echo "HY_CONV3D_SWEEP_PRESETS=$PRESETS"
echo "Log: $LOG"

python_env/bin/python -m pytest \
  models/experimental/hunyuan_image_3_0/tests/vae/test_encoder_conv3d_sweep.py \
  -k "bh_2x2" -s --timeout=0 \
  2>&1 | tee "$LOG"

python_env/bin/python models/experimental/hunyuan_image_3_0/scripts/apply_encoder_sweep_blockings.py
