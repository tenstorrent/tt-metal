#! /usr/bin/env bash

source scripts/tools_setup_common.sh

set -eo pipefail

cd $TT_METAL_HOME

if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
    exit 1
fi

remove_default_log_locations

cmake -B build && cmake --build build --target clean && rm -rf build
PYTHON_ENV_DIR=$(pwd)/build/python_env ENABLE_TRACY=1 ENABLE_PROFILER=1 ./build_metal.sh
cmake --build build --target programming_examples -- -j`nproc`
