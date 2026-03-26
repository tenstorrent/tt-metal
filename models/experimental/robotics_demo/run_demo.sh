#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# ======================================================================
# Tenstorrent Robotics Intelligence Demo Suite -- Launch Script
#
# Usage:
#   ./run_demo.sh                  # Launch Streamlit dashboard (default)
#   ./run_demo.sh --cli 1          # Run Scenario 1 headless via CLI
#   ./run_demo.sh --cli 2          # Run Scenario 2 headless via CLI
#   ./run_demo.sh --cli 3          # Run Scenario 3 headless via CLI
#   ./run_demo.sh --cli 4          # Run Scenario 4 headless via CLI
#   ./run_demo.sh --setup          # First-time setup (deps + tokenizer)
#   ./run_demo.sh --test           # Quick smoke test (no TT hardware)
# ======================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export TT_METAL_HOME
export PYTHONPATH="${TT_METAL_HOME}:${PYTHONPATH:-}"

PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"

echo "======================================================================"
echo "   Tenstorrent Robotics Intelligence Demo Suite"
echo "   TT_METAL_HOME: $TT_METAL_HOME"
echo "======================================================================"

# Parse args
MODE="dashboard"
SCENARIO=""
for arg in "$@"; do
    case "$arg" in
        --setup) MODE="setup" ;;
        --test)  MODE="test" ;;
        --cli)   MODE="cli" ;;
        1|2|3|4) SCENARIO="$arg" ;;
    esac
done

case "$MODE" in
    setup)
        echo ""
        echo "--- First-time Setup ---"
        echo ""
        echo "[1/3] Installing Python dependencies..."
        pip install pybullet numpy torch imageio[ffmpeg] opencv-python-headless \
                    streamlit matplotlib Pillow safetensors transformers 2>/dev/null || true

        echo ""
        echo "[2/3] Setting up Gemma tokenizer (optional, requires HF auth)..."
        $PYTHON "$SCRIPT_DIR/tokenizer_setup.py" || true

        echo ""
        echo "[3/3] Downloading PI0 weights (if not present)..."
        WEIGHTS_DIR="$TT_METAL_HOME/models/experimental/pi0/weights/pi0_base"
        if [ -d "$WEIGHTS_DIR" ] && [ -f "$WEIGHTS_DIR/model.safetensors" ]; then
            echo "  Weights already present at $WEIGHTS_DIR"
        else
            echo "  Weights not found. Run:"
            echo "    python $TT_METAL_HOME/models/experimental/pi0/tests/download_pretrained_weights.py"
        fi

        echo ""
        echo "Setup complete. Run:  ./run_demo.sh"
        ;;

    test)
        echo ""
        echo "--- Smoke Test (no TT hardware) ---"
        echo ""
        echo "[1/3] Testing multi_env.py..."
        $PYTHON -c "
import sys; sys.path.insert(0, '$TT_METAL_HOME')
from models.experimental.robotics_demo.multi_env import MultiEnvironment
env = MultiEnvironment(num_envs=2, image_size=64)
obs = env.capture_all_observations()
print(f'  Envs: {env.num_envs}, Images: {len(obs[0][0])}, State: {obs[0][1].shape}')
frames = env.capture_all_display_frames(320, 240)
print(f'  Display frames: {len(frames)}, Shape: {frames[0].shape}')
env.close()
print('  multi_env OK')
"
        echo ""
        echo "[2/3] Testing video_composer.py..."
        $PYTHON -c "
import sys; sys.path.insert(0, '$TT_METAL_HOME')
import numpy as np
from models.experimental.robotics_demo.video_composer import compose_quad_view, compose_side_by_side
frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(4)]
q = compose_quad_view(frames, labels=['A','B','C','D'])
print(f'  Quad view: {q.shape}')
s = compose_side_by_side(frames[:1], frames[1:2])
print(f'  Side-by-side: {s.shape}')
print('  video_composer OK')
"
        echo ""
        echo "[3/3] Testing metrics.py..."
        $PYTHON -c "
import sys; sys.path.insert(0, '$TT_METAL_HOME')
from models.experimental.robotics_demo.metrics import MetricsCollector, EnvironmentMetrics
mc = MetricsCollector(num_envs=4)
for i in range(4):
    mc.record(EnvironmentMetrics(env_id=i, model_name='PI0', step=0,
              inference_time_ms=330.0, loop_time_ms=85.0,
              distance_to_target=0.5, is_inference_step=True))
s = mc.get_global_summary()
print(f'  Global: {s[\"num_envs\"]} envs, {s[\"total_inferences\"]} inferences')
print('  metrics OK')
"
        echo ""
        echo "All smoke tests passed."
        ;;

    cli)
        if [ -z "$SCENARIO" ]; then
            echo "Usage: ./run_demo.sh --cli <1|2|3|4>"
            exit 1
        fi
        echo ""
        echo "--- Running Scenario $SCENARIO via CLI ---"
        echo ""
        xvfb-run -a $PYTHON "$SCRIPT_DIR/demo_orchestrator.py" --scenario "$SCENARIO" "$@"
        ;;

    dashboard)
        echo ""
        echo "--- Launching Streamlit Dashboard ---"
        echo ""
        echo "Open your browser to the URL shown below."
        echo "Press Ctrl+C to stop."
        echo ""
        streamlit run "$SCRIPT_DIR/streamlit_app.py" \
            --server.port 8501 \
            --server.address 0.0.0.0 \
            --browser.gatherUsageStats false
        ;;
esac
