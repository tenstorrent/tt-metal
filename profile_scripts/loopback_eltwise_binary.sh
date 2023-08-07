#!/bin/bash

make programming_examples/loopback_eltwise_binary
./build/programming_examples/loopback_eltwise_binary 1 64

python3 profile_scripts/custom_profile.py --file-name tt_metal/tools/profiler/logs/profile_log_device.csv --profile-target profile_elewise_binary
