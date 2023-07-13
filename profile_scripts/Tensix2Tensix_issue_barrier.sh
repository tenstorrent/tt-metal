#!/bin/bash

power_of_2() {
    local n=$1
    local result=1

    for ((i=1; i<=n; i++)); do
        result=$((result * 2))
    done

    echo $result
}

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
rm log/Tensix2Tensix_read_speed_issue_barrier.log
rm log/Tensix2Tensix_write_speed_issue_barrier.log
rm log/Tensix2Tensix_issue_barrier.log

for buffer_pow in {6..18}
do
for transaction_pow in {6..18}
do
if (($buffer_pow >= $transaction_pow))
then

buffer=$(power_of_2 $buffer_pow)
transaction=$(power_of_2 $transaction_pow)
echo "Buffer: "$buffer" Transaction: "$transaction >> log/Tensix2Tensix_read_speed_issue_barrier.log
echo "Buffer: "$buffer" Transaction: "$transaction >> log/Tensix2Tensix_write_speed_issue_barrier.log

for i in {1..250}
do
./build/test/llrt/test_run_risc_read_speed --buffer-size $buffer --transaction-size $transaction --num-repetitions 4 --profile 1
python3 profile_scripts/custom_profile.py --file-name tt_metal/tools/profiler/logs/profile_log_device.csv --profile-target profile_Tensix2Tensix_issue_barrier --read-or-write read >> log/Tensix2Tensix_read_speed_issue_barrier.log
./build/test/llrt/test_run_risc_write_speed --buffer-size $buffer --transaction-size $transaction --num-repetitions 4 --profile 1
python3 profile_scripts/custom_profile.py --file-name tt_metal/tools/profiler/logs/profile_log_device.csv --profile-target profile_Tensix2Tensix_issue_barrier --read-or-write write >> log/Tensix2Tensix_write_speed_issue_barrier.log
done

fi
done
done

echo "read" >> log/Tensix2Tensix_issue_barrier.log
python3 profile_scripts/script.py --file-name log/Tensix2Tensix_read_speed_issue_barrier.log --profile-target Tensix2Tensix_Issue_Barrier >> log/Tensix2Tensix_issue_barrier.log
echo "write" >> log/Tensix2Tensix_issue_barrier.log
python3 profile_scripts/script.py --file-name log/Tensix2Tensix_write_speed_issue_barrier.log --profile-target Tensix2Tensix_Issue_Barrier >> log/Tensix2Tensix_issue_barrier.log

python3 profile_scripts/script.py --file-name log/Tensix2Tensix_issue_barrier.log --profile-target Print_Tensix2Tensix_Issue_Barrier
