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

# Set Python environment directory
if [ -z "$PYTHON_ENV_DIR" ]; then
    PYTHON_ENV_DIR=$(pwd)/python_env_gstreamer
fi
echo "Creating virtual env in: $PYTHON_ENV_DIR"

# Create and activate virtual environment
$PYTHON_CMD -m venv $PYTHON_ENV_DIR --system-site-packages
source $PYTHON_ENV_DIR/bin/activate

# Import functions for detecting OS
. ./install_dependencies.sh --source-only
detect_os


if [ "$OS_ID" = "ubuntu" ] && [ "$OS_VERSION" = "22.04" ]; then
    echo "Ubuntu 22.04 detected: force pip/setuptools/wheel versions"
    pip install --force-reinstall pip==25.1.1
    python3 -m pip config set global.extra-index-url https://download.pytorch.org/whl/cpu
    python3 -m pip install setuptools wheel==0.45.1
else
    echo "$OS_ID $OS_VERSION detected: updating wheel and setuptools to latest"
    python3 -m pip install --upgrade wheel setuptools
fi

echo "Installing dev dependencies"
python3 -m pip install -r $(pwd)/tt_metal/python_env/requirements-dev.txt

echo "Installing tt-metal"
pip install -e .

# Do not install hooks when this is a worktree
if [ $(git rev-parse --git-dir) = $(git rev-parse --git-common-dir) ]; then
    echo "Generating git hooks"
    pre-commit install
    pre-commit install --hook-type commit-msg
else
    echo "In worktree: not generating git hooks"
fi

echo "If you want stubs, run ./scripts/build_scripts/create_stubs.sh"
