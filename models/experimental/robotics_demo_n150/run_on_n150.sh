#!/bin/bash
# ======================================================================
# N150 Single-Chip Robotics Demo -- Ready-to-Run Launcher
#
# Usage:
#   ./run_on_n150.sh smolvla           # Run SmolVLA on N150
#   ./run_on_n150.sh pi0               # Run PI0 on N150 (needs weights)
#   ./run_on_n150.sh smolvla --record   # Run + record video
#   ./run_on_n150.sh demo              # Scripted IK demo (no TT needed)
#   ./run_on_n150.sh setup             # Download PI0 weights
# ======================================================================

set -euo pipefail

# Paths
PYBIN=/home/ubuntu/backup/tt-metal/python_env/bin/python3
export TT_METAL_HOME=/home/ubuntu/agent/agentic/tt-metal
export PYTHONPATH="$TT_METAL_HOME:/home/ubuntu/robotics/tt-metal"
export ARCH_NAME=wormhole_b0
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml

DEMO_SCRIPT=/home/ubuntu/robotics/tt-metal/models/experimental/robotics_demo_n150/run_demo.py
PI0_WEIGHTS="$TT_METAL_HOME/models/experimental/pi0/weights/pi0_base"

echo "======================================================================"
echo "  N150 Robotics Demo"
echo "  ARCH: $ARCH_NAME | Device: Wormhole N150"
echo "======================================================================"

MODEL="${1:-smolvla}"
shift || true

case "$MODEL" in
    setup)
        echo ""
        echo "Downloading PI0 weights..."
        mkdir -p "$PI0_WEIGHTS"
        $PYBIN -c "
import sys
sys.path.insert(0, '$TT_METAL_HOME')
from models.experimental.pi0.tests.download_pretrained_weights import *
" 2>&1 || echo "Auto-download failed. Manual download:"
        echo "  1. Visit: https://drive.google.com/drive/folders/1qfY0EBGh_-6Zz-omKPQW6nBcc1Cp2_WN"
        echo "  2. Download model.safetensors and config.json"
        echo "  3. Place in: $PI0_WEIGHTS/"
        ;;

    demo)
        echo ""
        echo "Running scripted IK demo (no TT hardware)..."
        $PYBIN "$DEMO_SCRIPT" --demo-mode --steps 200 --record-video "$@"
        ;;

    smolvla)
        echo ""
        echo "Running SmolVLA on N150..."
        echo "  (Model auto-downloads from HuggingFace on first run)"
        echo ""
        EXTRA=""
        for arg in "$@"; do
            if [ "$arg" = "--record" ]; then EXTRA="--record-video"; fi
        done
        $PYBIN "$DEMO_SCRIPT" \
            --model smolvla \
            --task "pick up the cube" \
            --steps 300 \
            --replan-interval 5 \
            $EXTRA
        ;;

    pi0)
        if [ ! -f "$PI0_WEIGHTS/model.safetensors" ]; then
            echo ""
            echo "ERROR: PI0 weights not found at $PI0_WEIGHTS"
            echo "Run: ./run_on_n150.sh setup"
            exit 1
        fi
        echo ""
        echo "Running PI0 on N150..."
        EXTRA=""
        for arg in "$@"; do
            if [ "$arg" = "--record" ]; then EXTRA="--record-video"; fi
        done
        $PYBIN "$DEMO_SCRIPT" \
            --model pi0 \
            --task "pick up the cube" \
            --steps 300 \
            --checkpoint "$PI0_WEIGHTS" \
            --replan-interval 5 \
            $EXTRA
        ;;

    *)
        echo "Usage: $0 <smolvla|pi0|demo|setup> [--record]"
        exit 1
        ;;
esac
