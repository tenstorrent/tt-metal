#! /usr/bin/env bash

set -eo pipefail

remove_default_log_locations(){
    rm -rf $TT_METAL_HOME/tt_metal/tools/profiler/logs/ops
    rm -rf $TT_METAL_HOME/tt_metal/tools/profiler/logs/ops_device
}

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

cd $TT_METAL_HOME

if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
    exit 1
fi

if [ "$ARCH_NAME" == "grayskull" ]; then
    remove_default_log_locations
    make clean
    make build ENABLE_PROFILER=1
fi
