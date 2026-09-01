#!/bin/bash
cd "$(dirname "$0")"
export TT_METAL_HOME="$PWD"
export PYTHONPATH="$PWD"
export TT_METAL_WATCHER=5
export TT_METAL_WATCHER_APPEND=0
timeout --signal=KILL "${2:-240}" ./python_env/bin/python "$1"
echo "EXIT $?"
