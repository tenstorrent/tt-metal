#!/bin/bash

DRAM_channel_pow_2=0

power_of_2() {
    local n=$1
    local result=1

    for ((i=1; i<=n; i++)); do
        result=$((result * 2))
    done

    echo $result
}

make tests -j
./build/test/build_kernels_for_riscv/test_build_kernel_risc_rw_speed_banked_dram
mkdir -p log
rm log/DRAM2Tensix_read_write_speed.log

for buffer_pow in {13..19}
do
for transaction_pow in {6..13}
do
buffer=$(power_of_2 $buffer_pow)
transaction=$(power_of_2 $transaction_pow)
./build/test/llrt/test_run_risc_rw_speed_banked_dram $buffer 10000 $transaction $DRAM_channel_pow_2 1 1 >> log/DRAM2Tensix_read_write_speed.log
done
done


python3 profile_scripts/script.py --file-name log/DRAM2Tensix_read_write_speed.log --profile-target DRAM2Tensix
