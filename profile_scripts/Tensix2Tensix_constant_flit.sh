#!/bin/bash

# This script looks for the max Buffer size (= Transaction size) where filt latency is constant

rm ./build/test/build_kernels_for_riscv/test_build_kernel_blank
rm ./build/test/build_kernels_for_riscv/test_build_kernel_risc_read_speed
rm ./build/test/build_kernels_for_riscv/test_build_kernel_risc_write_speed
make tests/build_kernels_for_riscv/test_build_kernel_blank
make tests/build_kernels_for_riscv/test_build_kernel_risc_read_speed
make tests/build_kernels_for_riscv/test_build_kernel_risc_write_speed
./build/test/build_kernels_for_riscv/test_build_kernel_blank
./build/test/build_kernels_for_riscv/test_build_kernel_risc_read_speed --profile 1
./build/test/build_kernels_for_riscv/test_build_kernel_risc_write_speed --profile 1

rm ./build/test/llrt/test_run_risc_read_speed
rm ./build/test/llrt/test_run_risc_write_speed
make tests/llrt/test_run_risc_read_speed
make tests/llrt/test_run_risc_write_speed
mkdir -p log
rm log/Tensix2Tensix_read_speed_constant_flit.log
rm log/Tensix2Tensix_write_speed_constant_flit.log
rm log/Tensix2Tensix_constant_flit.log

for transaction_tmp in {32..64}
do
for num_transaction in 1 2 4 8 16 32 64
do

transaction=$((transaction_tmp * 32))
buffer=$((transaction * num_transaction))
echo "Buffer: "$buffer" Transaction: "$transaction >> log/Tensix2Tensix_read_speed_constant_flit.log
echo "Buffer: "$buffer" Transaction: "$transaction >> log/Tensix2Tensix_write_speed_constant_flit.log

for i in {1..250}
do
./build/test/llrt/test_run_risc_read_speed --buffer-size $buffer --transaction-size $transaction --num-repetitions 4 --profile 1
python3 profile_scripts/custom_profile.py --file-name tt_metal/tools/profiler/logs/profile_log_device.csv --profile-target profile_Tensix2Tensix_issue_barrier --read-or-write read >> log/Tensix2Tensix_read_speed_constant_flit.log
./build/test/llrt/test_run_risc_write_speed --buffer-size $buffer --transaction-size $transaction --num-repetitions 4 --profile 1
python3 profile_scripts/custom_profile.py --file-name tt_metal/tools/profiler/logs/profile_log_device.csv --profile-target profile_Tensix2Tensix_issue_barrier --read-or-write write >> log/Tensix2Tensix_write_speed_constant_flit.log
done

done
done

echo "read" >> log/Tensix2Tensix_constant_flit.log
python3 profile_scripts/script.py --file-name log/Tensix2Tensix_read_speed_constant_flit.log --profile-target Tensix2Tensix_Issue_Barrier >> log/Tensix2Tensix_constant_flit.log
echo "write" >> log/Tensix2Tensix_constant_flit.log
python3 profile_scripts/script.py --file-name log/Tensix2Tensix_write_speed_constant_flit.log --profile-target Tensix2Tensix_Issue_Barrier >> log/Tensix2Tensix_constant_flit.log

python3 profile_scripts/script.py --file-name log/Tensix2Tensix_constant_flit.log --profile-target Profile_Tensix2Tensix_Constant_Flit
