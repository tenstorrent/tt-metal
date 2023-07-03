#!/bin/bash

power_of_2() {
    local n=$1
    local result=1

    for ((i=1; i<=n; i++)); do
        result=$((result * 2))
    done

    echo $result
}

export TT_METAL_ENV_IS_DEV
make tests -j
./build/test/build_kernels_for_riscv/test_build_kernel_blank
./build/test/build_kernels_for_riscv/test_build_kernel_risc_read_speed
./build/test/build_kernels_for_riscv/test_build_kernel_risc_write_speed
mkdir -p log
rm log/Tensix2Tensix_read_speed.log
rm log/Tensix2Tensix_write_speed.log

for buffer_pow in {6..18}
do
for transaction_pow in {6..18}
do
if (($buffer_pow >= $transaction_pow))
then
buffer=$(power_of_2 $buffer_pow)
transaction=$(power_of_2 $transaction_pow)
./build/test/llrt/test_run_risc_read_speed --buffer-size $buffer --transaction-size $transaction >> log/Tensix2Tensix_read_speed.log
./build/test/llrt/test_run_risc_write_speed --buffer-size $buffer --transaction-size $transaction >> log/Tensix2Tensix_write_speed.log
fi
done
done

python3 profile_scripts/script.py --file-name log/Tensix2Tensix_read_speed.log --profile-target Tensix2Tensix --read-or-write read
python3 profile_scripts/script.py --file-name log/Tensix2Tensix_write_speed.log --profile-target Tensix2Tensix --read-or-write write
