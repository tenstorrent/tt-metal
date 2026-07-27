#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# ============================================================================
# Depth Anything V2 - N300 Hardware Validation Script
# Run this on a Koyeb N300s instance or any machine with Wormhole B0 hardware.
# ============================================================================

set -euo pipefail

echo "============================================"
echo " Depth Anything V2 - N300 Hardware Validation"
echo "============================================"
echo "Started at: $(date -u)"
echo ""

# 1. Setup tt-metal environment
echo "[1/6] Setting up tt-metal environment..."
if [ -d "/opt/tt_metal_infra" ]; then
    echo "  Using pre-installed tt-metal from Docker image"
    source /opt/tt_metal_infra/env/activate 2>/dev/null || true
elif [ -f "build/python_env/bin/activate" ]; then
    echo "  Using local tt-metal build"
    source build/python_env/bin/activate
fi

# Set environment variables
export TT_METAL_HOME="${TT_METAL_HOME:-$(pwd)}"
export PYTHONPATH="${TT_METAL_HOME}:${PYTHONPATH:-}"
export ARCH_NAME=wormhole_b0

echo "  TT_METAL_HOME=$TT_METAL_HOME"
echo "  ARCH_NAME=$ARCH_NAME"

# 2. Install Python dependencies
echo ""
echo "[2/6] Checking Python dependencies..."
if python3 -c "import torch, transformers, PIL" >/dev/null 2>&1; then
    echo "  Required Python packages already available; skipping install."
else
    echo "  Missing Python packages detected; installing dependencies..."
    pip install --quiet torch transformers==5.14.1 pillow 2>&1 | tail -3
fi

# 3. Clone the branch if not already present
echo ""
echo "[3/6] Checking model files..."
MODEL_DIR="models/experimental/depth_anything_v2"
if [ ! -f "${MODEL_DIR}/tt/model_def.py" ]; then
    echo "  ERROR: Model files not found at ${MODEL_DIR}/tt/model_def.py"
    echo "  Please run this script from a checked-out tt-metal repo with the"
    echo "  depth_anything_v2 model files present."
    exit 1
else
    echo "  Model files found at ${MODEL_DIR}"
fi

# 4. Verify device access
echo ""
echo "[4/6] Verifying Tenstorrent device access..."
python3 -c "
import ttnn
device = ttnn.open_device(device_id=0, l1_small_size=32768)
grid = device.compute_with_storage_grid_size()
print(f'  Device: {device}')
print(f'  Compute grid: {grid.x}x{grid.y} = {grid.x * grid.y} cores')
ttnn.close_device(device)
print('  Device access OK!')
" || { echo "ERROR: Cannot access Tenstorrent device!"; exit 1; }

# 5. Run PCC accuracy test
echo ""
echo "[5/6] Running PCC accuracy test..."
echo "  pytest ${MODEL_DIR}/tests/test_depth_anything_v2_pcc.py"
pytest ${MODEL_DIR}/tests/test_depth_anything_v2_pcc.py -v --tb=short 2>&1 | tee /tmp/pcc_results.txt
PCC_EXIT=${PIPESTATUS[0]}

# 6. Run validation suite (FPS + PCC + visualization)
echo ""
echo "[6/6] Running full validation suite..."
echo "  python ${MODEL_DIR}/demo/validate.py"
python3 ${MODEL_DIR}/demo/validate.py 2>&1 | tee /tmp/validate_results.txt
VAL_EXIT=${PIPESTATUS[0]}

# Summary
echo ""
echo "============================================"
echo " RESULTS SUMMARY"
echo "============================================"
echo "  PCC test:    $([ $PCC_EXIT -eq 0 ] && echo 'PASSED ✓' || echo 'FAILED ✗')"
echo "  Validation:  $([ $VAL_EXIT -eq 0 ] && echo 'PASSED ✓' || echo 'FAILED ✗')"
echo ""
echo "  PCC results: /tmp/pcc_results.txt"
echo "  Validation:  /tmp/validate_results.txt"

if [ -d "validation" ]; then
    echo ""
    echo "  Validation artifacts:"
    ls -la validation/
fi

echo ""
echo "Completed at: $(date -u)"
echo "============================================"
