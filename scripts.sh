#!/bin/bash
# export TT_METAL_HOME=$(pwd)
# export PYTHONPATH=${TT_METAL_HOME}
# export TT_METAL_ENV=dev
# export ARCH_NAME=grayskull
# make build
# source build/python_env/bin/activate

# # Small test
# for TILE in 1 2 4 8 16 32 64 128 256
# for TILE in {1..16}
# do
# make programming_examples/loopback
# ./build/programming_examples/loopback $TILE
# cp tt_metal/tools/profiler/logs/profile_log_device.csv tt_metal/tools/profiler/logs/profile_log_device_loopback_${TILE}_tile_2048_size.csv
# python3 tt_metal/tools/profiler/custom_profile.py tt_metal/tools/profiler/logs/profile_log_device.csv
# done

# for TILE in 1 2 4 8 16 32 64 128 256 512 1024
# for TILE in {9..16}
# do
# make programming_examples/eltwise_binary
# ./build/programming_examples/eltwise_binary $TILE
# cp tt_metal/tools/profiler/logs/profile_log_device.csv tt_metal/tools/profiler/logs/profile_log_device_eltwise_binary_${TILE}_tile_2048_size.csv
# python3 tt_metal/tools/profiler/custom_profile.py tt_metal/tools/profiler/logs/profile_log_device.csv
# done

make programming_examples/eltwise_binary
# ./build/programming_examples/eltwise_binary 1
LOGGER_LEVEL=Debug ./build/programming_examples/eltwise_binary 1

# # All tests
# make tests -j
# python3 -m tests.scripts.run_tt_metal
