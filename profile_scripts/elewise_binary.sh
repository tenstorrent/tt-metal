#!/bin/bash

power_of_2() {
    local n=$1
    local result=1

    for ((i=1; i<=n; i++)); do
        result=$((result * 2))
    done

    echo $result
}

rm ./build/programming_examples/eltwise_binary
make programming_examples/eltwise_binary
rm log/Elewise_binary_read_write.log
rm log/Elewise_binary_multi_tile.log

# for transaction_pow in {6..11}
# do
# ./build/programming_examples/eltwise_binary 1 $(power_of_2 $transaction_pow)
# echo $(power_of_2 $transaction_pow) >> log/Elewise_binary_read_write.log
# python3 profile_scripts/custom_profile.py --file-name tt_metal/tools/profiler/logs/profile_log_device.csv --profile-target profile_Elewise_binary_read_write >> log/Elewise_binary_read_write.log
# done

for i in {1..10}
do
# for num_tile_pow in {0..13}
for num_tile in {33..64}
do
# ./build/programming_examples/eltwise_binary $(power_of_2 $num_tile_pow) 2048
./build/programming_examples/eltwise_binary $num_tile 2048
python3 profile_scripts/custom_profile.py --file-name tt_metal/tools/profiler/logs/profile_log_device.csv --profile-target profile_Elewise_binary_multi_tile >> log/Elewise_binary_multi_tile.log
done
done

# python3 profile_scripts/script.py --file-name log/Elewise_binary_multi_tile_pow_2_1000x.log --profile-target Print_Elewise_Binary_Multi_Tile --num-repetitions 1000 --range 14
python3 profile_scripts/script.py --file-name log/Elewise_binary_multi_tile.log --profile-target Print_Elewise_Binary_Multi_Tile --num-repetitions 10 --range 32

# ./build/programming_examples/eltwise_binary 1 2048
# # python3 profile_scripts/custom_profile.py --file-name tt_metal/tools/profiler/logs/profile_log_device.csv --profile-target profile_Elewise_binary_fine_grain

# python3 profile_scripts/custom_profile.py --file-name tt_metal/tools/profiler/logs/profile_log_device.csv --profile-target profile_elewise_binary



# ~/tt-metal/tt_metal/src/ckernels/sfpi/compiler/bin/riscv32-unknown-elf-objdump -S ~/tt-metal/built_kernels/eltwise_binary_writer_unary_reader_binary_diff_lengths/15724778220131602173/tensix_thread0/tensix_thread0.elf > ~/tt-metal/built_kernels/eltwise_binary_writer_unary_reader_binary_diff_lengths/15724778220131602173/tensix_thread0/tensix_thread0.asm

# ~/tt-metal/tt_metal/src/ckernels/sfpi/compiler/bin/riscv32-unknown-elf-objdump -S ~/tt-metal/built_kernels/eltwise_binary_writer_unary_reader_binary_diff_lengths/15724778220131602173/tensix_thread1/tensix_thread1.elf > ~/tt-metal/built_kernels/eltwise_binary_writer_unary_reader_binary_diff_lengths/15724778220131602173/tensix_thread1/tensix_thread1.asm

# ~/tt-metal/tt_metal/src/ckernels/sfpi/compiler/bin/riscv32-unknown-elf-objdump -S ~/tt-metal/built_kernels/eltwise_binary_writer_unary_reader_binary_diff_lengths/15724778220131602173/tensix_thread2/tensix_thread2.elf > ~/tt-metal/built_kernels/eltwise_binary_writer_unary_reader_binary_diff_lengths/15724778220131602173/tensix_thread2/tensix_thread2.asm
