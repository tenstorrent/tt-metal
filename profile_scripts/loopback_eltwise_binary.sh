#!/bin/bash

power_of_2() {
    local n=$1
    local result=1

    for ((i=1; i<=n; i++)); do
        result=$((result * 2))
    done

    echo $result
}

make programming_examples/loopback_eltwise_binary
rm log/loopback_eltwise_binary_multi_tile.log

for num_repetitions in {1..100}
do
for num_tiles in 3 5 6 7
do
# ./build/programming_examples/loopback_eltwise_binary $(power_of_2 $num_tiles) 2048
./build/programming_examples/loopback_eltwise_binary $num_tiles 2048
python3 profile_scripts/custom_profile.py --file-name tt_metal/tools/profiler/logs/profile_log_device.csv --profile-target profile_Elewise_binary_multi_tile >> log/loopback_eltwise_binary_multi_tile.log
done
done

python3 profile_scripts/script.py --file-name log/loopback_eltwise_binary_multi_tile.log --profile-target Print_Elewise_Binary_Multi_Tile --num-repetitions 100 --range 13
