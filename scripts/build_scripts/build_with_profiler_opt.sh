#! /usr/bin/env bash

source scripts/tools_setup_common.sh

set -eo pipefail

cd $TT_METAL_HOME

if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
    exit 1
fi

ENABLE_TRACY=1 ENABLE_PROFILER=1 cmake -B build -G Ninja

if [[ $1 == "NO_CLEAN" ]]; then
    cmake --build build
else
    remove_default_log_locations
    cmake --build build --target clean
fi

cmake --build build --target install
cmake --build build --target programming_examples
./create_venv.sh
