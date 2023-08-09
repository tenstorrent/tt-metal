#!/bin/bash

power_of_2() {
    local n=$1
    local result=1

    for ((i=1; i<=n; i++)); do
        result=$((result * 2))
    done

    echo $result
}

# make programming_examples/loopback_bmm_tilize_untilize
rm log/loopback_bmm_tilize_untilze_multi_tile.log


# for num_repetitions in 1
# do
# for blocks_pow in 0
# do
# ./build/programming_examples/loopback_bmm_tilize_untilize $(power_of_2 $blocks_pow) 2048
# python3 profile_scripts/custom_profile.py --file-name tt_metal/tools/profiler/logs/profile_log_device.csv --profile-target profile_Elewise_binary_multi_tile >> log/loopback_bmm_tilize_untilze_multi_tile.log
# done
# done

# # python3 profile_scripts/script.py --file-name log/loopback_bmm_tilize_untilze_multi_tile.log --profile-target Print_Elewise_Binary_Multi_Tile --num-repetitions 10 --range 2


make build

for num_repetitions in {1..100}
do
for a_height_nblocks in 1
do
for a_width_nblocks_pow in 3 5 6 7
do
for b_width_nblocks in 1
do
for a_block_height_ntiles in 1
do
for a_block_width_ntiles in 1
do
for b_block_width_ntiles in 1
do
for out_subblock_height_ntiles in 1
do
for out_subblock_width_ntiles in 1
do

python3 tests/python_api_testing/unit_testing/test_loopback_bmm_tilize_untilize.py --a-height-nblocks $a_height_nblocks --a-width-nblocks $a_width_nblocks_pow --b-width-nblocks $b_width_nblocks --a-block-height-ntiles $a_block_height_ntiles --a-block-width-ntiles $a_block_width_ntiles --b-block-width-ntiles $b_block_width_ntiles --out-subblock-height-ntiles $out_subblock_height_ntiles --out-subblock-width-ntiles $out_subblock_width_ntiles

python3 profile_scripts/custom_profile.py --file-name tt_metal/tools/profiler/logs/profile_log_device.csv --profile-target profile_Elewise_binary_multi_tile >> log/loopback_bmm_tilize_untilze_multi_tile.log

done
done
done
done
done
done
done
done
done

python3 profile_scripts/script.py --file-name log/loopback_bmm_tilize_untilze_multi_tile.log --profile-target Print_Elewise_Binary_Multi_Tile --num-repetitions 100 --range 4
