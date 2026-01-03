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

echo "Getting uv and python 3.12"
${PYTHON_CMD} -m pip install uv

# Set Python environment directory
if [ -z "$PYTHON_ENV_DIR" ]; then
    PYTHON_ENV_DIR=$(pwd)/python_env
fi
echo "Creating virtual env in: $PYTHON_ENV_DIR"

# Create and activate virtual environment
uv python install 3.12
uv venv $PYTHON_ENV_DIR --python 3.12
source $PYTHON_ENV_DIR/bin/activate

# Import functions for detecting OS
. ./install_dependencies.sh --source-only
detect_os


if [ "$OS_ID" = "ubuntu" ] && [ "$OS_VERSION" = "22.04" ]; then
    echo "Ubuntu 22.04 detected: force pip/setuptools/wheel versions"
    uv pip config set global.extra-index-url https://download.pytorch.org/whl/cpu
    uv pip install setuptools wheel==0.45.1
else
    echo "$OS_ID $OS_VERSION detected: updating wheel and setuptools to latest"
    uv pip install --upgrade wheel setuptools
fi

echo "Installing dev dependencies"
# need the index strategy to fallback to specified torch version buried in reqs
uv pip install --index-strategy unsafe-best-match -r $(pwd)/tt_metal/python_env/requirements-dev.txt

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
