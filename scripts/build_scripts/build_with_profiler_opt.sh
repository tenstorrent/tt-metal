#! /usr/bin/env bash

source scripts/tools_setup_common.sh

set -eo pipefail

cd $TT_METAL_HOME

if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
    exit 1
fi

cmake -B build -G Ninja -DENABLE_TRACY=ON

if [[ $1 == "NO_CLEAN" ]]; then
    cmake --build build
else
    remove_default_log_locations
    cmake --build build --target clean
fi

cmake --build build --target install
cmake --build build --target programming_examples
