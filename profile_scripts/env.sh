#!/bin/bash

# Setup environmental variables and compile TT-Metal
# If executing the scripts does not work, need to copy the scripts and run in the command line manually

export TT_METAL_HOME=$(pwd)
export PYTHONPATH=${TT_METAL_HOME}
export TT_METAL_ENV=dev
export ARCH_NAME=grayskull
export TT_METAL_ENV_IS_DEV
make build
source build/python_env/bin/activate
