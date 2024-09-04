#!/bin/bash
set -eo pipefail


# Allow overriding Python command via environment variable
if [ -z "$PYTHON_CMD" ]; then
    PYTHON_CMD="python3"
else
    echo "Using user-specified Python: $PYTHON_CMD"
fi

# Verify Python command exists
if ! command -v $PYTHON_CMD &> /dev/null; then
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
source $PYTHON_ENV_DIR/bin/activate

echo "Forcefully using a version of pip that will work with our view of editable installs"
pip install --force-reinstall pip==20.1.1

echo "Setting up virtual env"
python3 -m pip config set global.extra-index-url https://download.pytorch.org/whl/cpu
python3 -m pip install setuptools wheel

echo "Installing dev dependencies"
python3 -m pip install -r $(pwd)/tt_metal/python_env/requirements-dev.txt

echo "Installing tt-metal"
pip install -e .

echo "Generating git hooks"
pre-commit install
pre-commit install --hook-type commit-msg

echo "If you want stubs, run ./scripts/build_scripts/create_stubs.sh"
