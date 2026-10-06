#!/bin/bash
set -eo pipefail
cd /home/smarton/fasth3/tt-metal/tt-project/worktrees/t158
git submodule update --init --recursive
source /home/smarton/fasth3/tt-metal/python_env/bin/activate
./build_metal.sh
echo BUILD_DONE
