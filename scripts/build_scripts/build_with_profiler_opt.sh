#! /usr/bin/env bash

set -eo pipefail

remove_default_log_locations(){
    rm -rf $TT_METAL_HOME/tt_metal/tools/profiler/logs/ops
    rm -rf $TT_METAL_HOME/tt_metal/tools/profiler/logs/ops_device
    rm -rf $TT_METAL_HOME/profiler_stats
    rm -rf $TT_METAL_HOME/profiler_data.csv
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
# Check if profile_unary.py exists
if [[ ! -f "tests/scripts/profile_unary.py" ]]; then
    echo "profile_unary.py does not exist in the TT_METAL_HOME directory." 1>&2
    exit 1
fi

# Check if postproc_unary.py exists
if [[ ! -f "tests/scripts/postproc_unary.py" ]]; then
    echo "postproc_unary.py does not exist in the TT_METAL_HOME directory." 1>&2
    exit 1
fi

remove_default_log_locations
make clean
make build ENABLE_PROFILER=1
