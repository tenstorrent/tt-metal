#! /usr/bin/env bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

PROFILER_SCRIPTS_ROOT=$TT_METAL_HOME/tt_metal/tools/profiler
PROFILER_TEST_SCRIPTS_ROOT=$TT_METAL_HOME/tests/tt_metal/tools/profiler
PROFILER_ARTIFACTS_DIR=$TT_METAL_HOME/generated/profiler
PROFILER_OUTPUT_DIR=$PROFILER_ARTIFACTS_DIR/reports

remove_default_log_locations(){
    rm -rf $PROFILER_ARTIFACTS_DIR
    echo "Removed all profiler artifacts"
}
