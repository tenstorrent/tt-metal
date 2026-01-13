#!/bin/bash
set -eo pipefail

# Allow overriding Python command via environment variable
if [ -z "$PYTHON_CMD" ]; then
    PYTHON_CMD="python3"
else
    echo "Using user-specified Python: $PYTHON_CMD"
fi

# Verify Python command exists
if ! command -v $PYTHON_CMD &>/dev/null; then
    echo "Python command not found: $PYTHON_CMD"
    exit 1
fi

# Install uv if not already available
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    ${PYTHON_CMD} -m pip install uv
    # Add pip's bin directory to PATH for the current session
    USER_BIN="${HOME}/.local/bin"
    if [[ -d "$USER_BIN" ]] && [[ ":$PATH:" != *":$USER_BIN:"* ]]; then
        export PATH="$USER_BIN:$PATH"
    fi
fi

# Verify uv is available
if ! command -v uv &>/dev/null; then
    echo "Error: uv not found in PATH after installation"
    echo "Please ensure ~/.local/bin is in your PATH and try again"
    exit 1
fi

# Set Python environment directory
if [ -z "$PYTHON_ENV_DIR" ]; then
    PYTHON_ENV_DIR=$(pwd)/python_env
fi
echo "Creating virtual env in: $PYTHON_ENV_DIR"

# Install Python via uv and create virtual environment
echo "Installing Python ${VENV_PYTHON_VERSION} via uv..."
uv python install ${VENV_PYTHON_VERSION}
uv venv $PYTHON_ENV_DIR --python ${VENV_PYTHON_VERSION}
source $PYTHON_ENV_DIR/bin/activate

# Import functions for detecting OS
. ./install_dependencies.sh --source-only
detect_os

# PyTorch CPU index URL for all uv pip commands
PYTORCH_INDEX="https://download.pytorch.org/whl/cpu"

if [ "$OS_ID" = "ubuntu" ] && [ "$OS_VERSION" = "22.04" ]; then
    echo "Ubuntu 22.04 detected: force pip/setuptools/wheel versions"
    uv pip install --extra-index-url "$PYTORCH_INDEX" setuptools wheel==0.45.1
else
    echo "$OS_ID $OS_VERSION detected: updating wheel and setuptools to latest"
    uv pip install --upgrade wheel setuptools
fi

echo "Installing dev dependencies"
# Use --extra-index-url for PyTorch CPU wheels and index-strategy for transitive deps
uv pip install --extra-index-url "$PYTORCH_INDEX" --index-strategy unsafe-best-match -r $(pwd)/tt_metal/python_env/requirements-dev.txt

echo "Installing tt-metal"
uv pip install -e .

# Do not install hooks when this is a worktree
if [ $(git rev-parse --git-dir) = $(git rev-parse --git-common-dir) ]; then
    echo "Generating git hooks"
    pre-commit install
    pre-commit install --hook-type commit-msg
else
    echo "In worktree: not generating git hooks"
fi

echo "If you want stubs, run ./scripts/build_scripts/create_stubs.sh"
