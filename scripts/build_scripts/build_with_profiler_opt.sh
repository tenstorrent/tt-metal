#! /usr/bin/env bash

source scripts/tools_setup_common.sh

set -eo pipefail

cd $TT_METAL_HOME

if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
    exit 1
fi

remove_default_log_locations

ENABLE_TRACY=1 ENABLE_PROFILER=1 cmake -B build -G Ninja && cmake --build build --target clean
cmake --build build --target install
cmake --build build --target programming_examples
PYTHON_ENV_DIR=$(pwd)/build/python_env ./scripts/build_scripts/create_venv.sh
