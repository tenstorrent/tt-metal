
#!/bin/bash

rm ./build/programming_examples/eltwise_binary
make programming_examples/eltwise_binary
./build/programming_examples/eltwise_binary 1

python3 profile_scripts/custom_profile.py --file-name tt_metal/tools/profiler/logs/profile_log_device.csv --profile-target profile_RISC_overhead --log-prefix "0, 0, 0, NCRISC"
python3 profile_scripts/custom_profile.py --file-name tt_metal/tools/profiler/logs/profile_log_device.csv --profile-target profile_RISC_overhead --log-prefix "0, 0, 0, BRISC"
python3 profile_scripts/custom_profile.py --file-name tt_metal/tools/profiler/logs/profile_log_device.csv --profile-target profile_RISC_overhead --log-prefix "0, 0, 0, TRISC_0"
python3 profile_scripts/custom_profile.py --file-name tt_metal/tools/profiler/logs/profile_log_device.csv --profile-target profile_RISC_overhead --log-prefix "0, 0, 0, TRISC_1"
python3 profile_scripts/custom_profile.py --file-name tt_metal/tools/profiler/logs/profile_log_device.csv --profile-target profile_RISC_overhead --log-prefix "0, 0, 0, TRISC_2"
