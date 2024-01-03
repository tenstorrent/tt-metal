#! /usr/bin/env bash

source scripts/tools_setup_common.sh

set -eo pipefail

cd $TT_METAL_HOME

if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
    exit 1
fi

remove_default_log_locations
make clean
make build ENABLE_PROFILER=1 ENABLE_TRACY=1
make programming_examples
