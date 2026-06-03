#!/usr/bin/env bash
# Run tests/ttnn/unit_tests/base_functionality/test_d2h_stream_service.py on hardware.
#
# Prerequisites:
#   cmake --build build_Release --target ttnn
#   Run on a node with Tenstorrent devices (e.g. via srun on a Galaxy allocation).
#
# Usage (from repo root):
#   tt-smi -r   # optional but recommended before first run
#   ./scripts/run_d2h_stream_service_pytest.sh
#   ./scripts/run_d2h_stream_service_pytest.sh -k 'bytes and 16'

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${D2H_PYTEST_VENV:-$REPO_ROOT/.venv_galaxy}"
PYTHON="${VENV_DIR}/bin/python3"

if [[ ! -x "$PYTHON" ]]; then
  echo "Creating Galaxy-compatible venv at $VENV_DIR (uses /usr/bin/python3 on this node)..."
  uv venv "$VENV_DIR" --python /usr/bin/python3
  uv pip install --python "$PYTHON" \
    loguru pytest torch "numpy>=1.24.4,<2" \
    pyyaml networkx graphviz click pandas seaborn ml_dtypes
fi

export TT_METAL_HOME="$REPO_ROOT"
export TT_METAL_RUNTIME_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/tools:$REPO_ROOT/build_Release/ttnn:$REPO_ROOT/ttnn:$REPO_ROOT"
export LD_LIBRARY_PATH="$REPO_ROOT/build_Release/ttnn:$REPO_ROOT/build_Release/lib:$REPO_ROOT/build_Release/tt_metal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Default --tt-arch so pytest does not skip when the prior test left devices busy
# during conftest option parsing.
TT_ARCH="${D2H_PYTEST_TT_ARCH:-$("$PYTHON" -c 'import ttnn; print(ttnn.get_arch_name())' 2>/dev/null || echo blackhole)}"
PYTEST_EXTRA=()
if ! echo " $* " | grep -q ' --tt-arch'; then
  PYTEST_EXTRA=(--tt-arch="$TT_ARCH")
fi

exec "$PYTHON" -m pytest -q tests/ttnn/unit_tests/base_functionality/test_d2h_stream_service.py "${PYTEST_EXTRA[@]}" "$@"
