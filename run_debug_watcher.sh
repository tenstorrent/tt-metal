#!/bin/bash
cd "$(dirname "$0")"
export TT_METAL_HOME="$PWD"
export PYTHONPATH="$PWD"
export TT_METAL_WATCHER=5
export TT_METAL_WATCHER_APPEND=0
export TT_METAL_OPERATION_TIMEOUT_SECONDS=20
export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="$PWD/python_env/bin/python $PWD/tools/tt-triage.py --disable-progress --skip-version-check > $PWD/generated/triage_dbg.log 2>&1"
timeout --signal=KILL "${2:-240}" ./python_env/bin/python "$1"
echo "EXIT $?"
