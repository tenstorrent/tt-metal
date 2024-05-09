#!/bin/bash
set -eo pipefail

if [ -z "$PYTHON_ENV_DIR" ]; then
    PYTHON_ENV_DIR=$(pwd)/python_env
fi

if [ -z "$CONFIG" ]; then
    echo "Build type defaulted to RelWithDebInfo (assert)"
else
    VALID_CONFIGS="RelWithDebInfo Debug Release ci"
    if [[ $VALID_CONFIGS =~ (^|[[:space:]])"$CONFIG"($|[[:space:]]) ]]; then
        echo "CONFIG set to $CONFIG"
    else
        echo "Invalid config "$CONFIG" given.. Valid configs are: $VALID_CONFIGS"
        exit 1
    fi
fi

echo "Building tt-metal"
cmake -B build -G Ninja
cmake --build build
cmake --build build --target metal-install

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
pip install -e ttnn

if [ "$CONFIG" != "ci" ]; then
    echo "Building cpp tests"
    cmake --build build --target tests -- -j`nproc`

    echo "Generating stubs"
    stubgen -m tt_lib -m tt_lib.device -m tt_lib.profiler -m tt_lib.tensor -m tt_lib.operations -m tt_lib.operations.primary -m tt_lib.operations.primary.transformers -o tt_eager
    stubgen -p ttnn._ttnn -o ttnn
    sed -i 's/\._C/tt_lib/g' tt_eager/tt_lib/__init__.pyi

    echo "Generating git hooks"
    pre-commit install
    pre-commit install --hook-type commit-msg
fi
