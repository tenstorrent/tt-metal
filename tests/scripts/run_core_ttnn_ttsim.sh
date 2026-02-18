#!/usr/bin/env bash
# =============================================================================
# Run "TTNN core ttnn unit test group" on TTSim locally (same as CI job
# "TTNN core ttnn unit test group - wormhole_b0" / "... - blackhole" in
# .github/workflows/ttsim.yaml).
#
# Usage:
#   ./tests/scripts/run_core_ttnn_ttsim.sh [arch]
#
# Examples:
#   ./tests/scripts/run_core_ttnn_ttsim.sh              # wormhole_b0
#   ./tests/scripts/run_core_ttnn_ttsim.sh wormhole_b0
#   ./tests/scripts/run_core_ttnn_ttsim.sh blackhole
#
# Prerequisites:
#   - Run from repo root: ./build_metal.sh and ./create_venv.sh
#   - Activate venv: source python_env/bin/activate
#   - Optional: yq (for CI-matching skip list)
# =============================================================================

set -euo pipefail

ARCH="${1:-wormhole_b0}"
if [[ "$ARCH" != "wormhole_b0" && "$ARCH" != "blackhole" ]]; then
  echo "Usage: $0 [wormhole_b0|blackhole]" >&2
  exit 1
fi

# Repo root (script lives in tests/scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TT_METAL_SIMULATOR_HOME="${TT_METAL_SIMULATOR_HOME:-$REPO_ROOT/sim}"
TT_SIM_VERSION="${TT_SIM_VERSION:-v1.3.4}"
TTSIM_SKIP_LIST_YAML="$REPO_ROOT/tests/pipeline_reorg/ttsim-skip-list.yaml"

echo "[run_core_ttnn_ttsim] ARCH=$ARCH"
echo "[run_core_ttnn_ttsim] TT_METAL_SIMULATOR_HOME=$TT_METAL_SIMULATOR_HOME"

mkdir -p "$TT_METAL_SIMULATOR_HOME"

# Download ttsim if libttsim.so is missing
if [[ ! -f "$TT_METAL_SIMULATOR_HOME/libttsim.so" ]]; then
  echo "[run_core_ttnn_ttsim] Downloading ttsim $TT_SIM_VERSION..."
  if [[ "$ARCH" == "wormhole_b0" ]]; then
    curl -sL "https://github.com/tenstorrent/ttsim/releases/download/${TT_SIM_VERSION}/libttsim_wh.so" -o "$TT_METAL_SIMULATOR_HOME/libttsim.so"
    cp "$REPO_ROOT/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml" "$TT_METAL_SIMULATOR_HOME/soc_descriptor.yaml"
  else
    curl -sL "https://github.com/tenstorrent/ttsim/releases/download/${TT_SIM_VERSION}/libttsim_bh.so" -o "$TT_METAL_SIMULATOR_HOME/libttsim.so"
    cp "$REPO_ROOT/tt_metal/soc_descriptors/blackhole_140_arch.yaml" "$TT_METAL_SIMULATOR_HOME/soc_descriptor.yaml"
  fi
else
  # Ensure soc_descriptor.yaml is present/updated
  if [[ "$ARCH" == "wormhole_b0" ]]; then
    cp "$REPO_ROOT/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml" "$TT_METAL_SIMULATOR_HOME/soc_descriptor.yaml"
  else
    cp "$REPO_ROOT/tt_metal/soc_descriptors/blackhole_140_arch.yaml" "$TT_METAL_SIMULATOR_HOME/soc_descriptor.yaml"
  fi
fi

export ARCH_NAME="$ARCH"
export TT_METAL_SIMULATOR_HOME
export TT_METAL_SIMULATOR="$TT_METAL_SIMULATOR_HOME/libttsim.so"
export TT_METAL_SLOW_DISPATCH_MODE=1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

# Build skip list args like CI (optional; requires yq)
SKIP_ARGS=""
if command -v yq &>/dev/null && [[ -f "$TTSIM_SKIP_LIST_YAML" ]]; then
  SKIP_ARGS=$(yq -r ".${ARCH_NAME}[]" "$TTSIM_SKIP_LIST_YAML" 2>/dev/null | sed 's/^/--deselect=/' | tr '\n' ' ' | sed 's/[[:space:]]*$//' || true)
  echo "[run_core_ttnn_ttsim] Using skip list from $TTSIM_SKIP_LIST_YAML"
fi

# Run core ttnn unit test group (same cmd as ttnn-tests.yaml + CI -n 4 and skip list)
echo "[run_core_ttnn_ttsim] Running: pytest ... tests/ttnn/unit_tests/base_functionality tests/ttnn/unit_tests/benchmarks tests/ttnn/unit_tests/tensor"
exec pytest --timeout 300 \
  tests/ttnn/unit_tests/base_functionality \
  tests/ttnn/unit_tests/benchmarks \
  tests/ttnn/unit_tests/tensor \
  -xv -m "not disable_fast_runtime_mode" \
  -n 4 \
  $SKIP_ARGS
