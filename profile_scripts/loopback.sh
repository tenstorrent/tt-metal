#!/bin/bash

# Small test
# for TILE in 1 2 4 8 16 32 64 128 256
for TILE in {1..16}
do
make programming_examples/loopback
./build/programming_examples/loopback $TILE
cp tt_metal/tools/profiler/logs/profile_log_device.csv tt_metal/tools/profiler/logs/profile_log_device_loopback_${TILE}_tile_2048_size.csv
python3 profile_scripts/custom_profile.py tt_metal/tools/profiler/logs/profile_log_device.csv
done
