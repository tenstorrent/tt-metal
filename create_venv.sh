
#!/bin/bash
set -eo pipefail

if [ -z "$PYTHON_ENV_DIR" ]; then
    PYTHON_ENV_DIR=$(pwd)/python_env
fi

echo "Creating virtual env in: $PYTHON_ENV_DIR"
python3 -m venv $PYTHON_ENV_DIR

source $PYTHON_ENV_DIR/bin/activate

echo "Setting up virtual env"
python3 -m pip config set global.extra-index-url https://download.pytorch.org/whl/cpu
python3 -m pip install setuptools wheel

echo "Installing dev dependencies"
python3 -m pip install -r $(pwd)/tt_metal/python_env/requirements-dev.txt

echo "Installing tt-metal"
pip install -e .
pip install -e ttnn/setup_ttnn.py
pip install -e ttnn/setup_tt_lib.py

echo "Generating git hooks"
pre-commit install
pre-commit install --hook-type commit-msg

echo "If you want stubs, run ./scripts/build_scripts/create_stubs.sh"
