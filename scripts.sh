#!/bin/bash
# cd tt-metal-user
# export TT_METAL_HOME=$(pwd)
# export PYTHONPATH=${TT_METAL_HOME}
# export TT_METAL_ENV=dev
# export ARCH_NAME=grayskull
# make build
# source build/python_env/bin/activate

# # Small test
# make programming_examples/loopback
# ./build/programming_examples/loopback

make programming_examples/eltwise_binary
./build/programming_examples/eltwise_binary

# # All tests
# make tests -j
# python3 -m tests.scripts.run_tt_metal
