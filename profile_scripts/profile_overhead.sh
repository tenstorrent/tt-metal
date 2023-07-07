#!/bin/bash

rm ./build/test/build_kernels_for_riscv/test_build_kernel_blank
rm ./build/test/build_kernels_for_riscv/test_build_kernel_risc_read_speed
rm ./build/test/build_kernels_for_riscv/test_build_kernel_risc_write_speed
make tests/build_kernels_for_riscv/test_build_kernel_blank
make tests/build_kernels_for_riscv/test_build_kernel_risc_read_speed
make tests/build_kernels_for_riscv/test_build_kernel_risc_write_speed
./build/test/build_kernels_for_riscv/test_build_kernel_blank
./build/test/build_kernels_for_riscv/test_build_kernel_risc_read_speed --profile 1
./build/test/build_kernels_for_riscv/test_build_kernel_risc_write_speed --profile 1

rm tests/llrt/test_run_risc_read_speed
rm tests/llrt/test_run_risc_write_speed
make tests/llrt/test_run_risc_read_speed
make tests/llrt/test_run_risc_write_speed
mkdir -p log
rm log/profile_overhead_read.log
rm log/profile_overhead_write.log

for i in {1..100}
do
./build/test/llrt/test_run_risc_read_speed --buffer-size 64 --transaction-size 64 --num-repetitions 1 --profile 1
python3 profile_scripts/custom_profile.py --file-name tt_metal/tools/profiler/logs/profile_log_device.csv --profile-target profile_overhead --read-or-write read >> log/profile_overhead_read.log
./build/test/llrt/test_run_risc_write_speed --buffer-size 64 --transaction-size 64 --num-repetitions 1 --profile 1
python3 profile_scripts/custom_profile.py --file-name tt_metal/tools/profiler/logs/profile_log_device.csv --profile-target profile_overhead --read-or-write write >> log/profile_overhead_write.log
done
