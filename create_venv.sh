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
    PYTHON_ENV_DIR=$(pwd)/python_env
fi
echo "Creating virtual env in: $PYTHON_ENV_DIR"

# Create and activate virtual environment
$PYTHON_CMD -m venv $PYTHON_ENV_DIR

# Add custom environment variables to the activation script
echo "Adding custom environment variables to activation script"

# Capture current working directory at creation time
# Assuming script is running from TT_METAL_HOME location
CREATION_DIR=$(pwd)

cat >> $PYTHON_ENV_DIR/bin/activate << EOF

# Custom TT-Metal environment variables
export PYTHONPATH="$CREATION_DIR"
export TT_METAL_HOME="$CREATION_DIR"
export TT_METAL_RUNTIME_ROOT="$CREATION_DIR"

# Store original values for deactivation
if [ -n "\${PYTHONPATH:-}" ]; then
    _OLD_VIRTUAL_PYTHONPATH="\$PYTHONPATH"
fi
if [ -n "\${TT_METAL_HOME:-}" ]; then
    _OLD_VIRTUAL_TT_METAL_HOME="\$TT_METAL_HOME"
fi
if [ -n "\${TT_METAL_RUNTIME_ROOT:-}" ]; then
    _OLD_VIRTUAL_TT_METAL_RUNTIME_ROOT="\$TT_METAL_RUNTIME_ROOT"
fi
EOF

# Also modify the deactivate function to restore original values
sed -i '/unset VIRTUAL_ENV/i\
    # Restore original environment variables\
    if [ -n "${_OLD_VIRTUAL_PYTHONPATH:-}" ]; then\
        PYTHONPATH="$_OLD_VIRTUAL_PYTHONPATH"\
        export PYTHONPATH\
        unset _OLD_VIRTUAL_PYTHONPATH\
    else\
        unset PYTHONPATH\
    fi\
    if [ -n "${_OLD_VIRTUAL_TT_METAL_HOME:-}" ]; then\
        TT_METAL_HOME="$_OLD_VIRTUAL_TT_METAL_HOME"\
        export TT_METAL_HOME\
        unset _OLD_VIRTUAL_TT_METAL_HOME\
    else\
        unset TT_METAL_HOME\
    fi\
    if [ -n "${_OLD_VIRTUAL_TT_METAL_RUNTIME_ROOT:-}" ]; then\
        TT_METAL_RUNTIME_ROOT="$_OLD_VIRTUAL_TT_METAL_RUNTIME_ROOT"\
        export TT_METAL_RUNTIME_ROOT\
        unset _OLD_VIRTUAL_TT_METAL_RUNTIME_ROOT\
    else\
        unset TT_METAL_RUNTIME_ROOT\
    fi' $PYTHON_ENV_DIR/bin/activate

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

echo "Environment setup complete!"
echo "If you want stubs, run ./scripts/build_scripts/create_stubs.sh"
echo ""
echo "The following environment variables will be automatically set when you activate the virtual environment:"
echo "  PYTHONPATH=$CREATION_DIR"
echo "  TT_METAL_HOME=$CREATION_DIR"
echo "  TT_METAL_RUNTIME_ROOT=$CREATION_DIR"
