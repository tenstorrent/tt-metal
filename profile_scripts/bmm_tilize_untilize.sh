#!/bin/bash

power_of_2() {
    local n=$1
    local result=1

    for ((i=1; i<=n; i++)); do
        result=$((result * 2))
    done

    echo $result
}

rm log/bmm_tilize_untilze_multi_tile.log

# pytest tests/python_api_testing/unit_testing/test_bmm_tilize_untilize.py
for num_repetitions in {1..10}
do
for a_height_nblocks in 1
do
for a_width_nblocks in {16..18}
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
# python3 tests/python_api_testing/unit_testing/test_bmm_tilize_untilize.py --a-height-nblocks $a_height_nblocks --a-width-nblocks $a_width_nblocks --b-width-nblocks $b_width_nblocks --a-block-height-ntiles $a_block_height_ntiles --a-block-width-ntiles $a_block_width_ntiles --b-block-width-ntiles $b_block_width_ntiles --out-subblock-height-ntiles $out_subblock_height_ntiles --out-subblock-width-ntiles $out_subblock_width_ntiles
python3 tests/python_api_testing/unit_testing/test_bmm_tilize_untilize.py --a-height-nblocks $a_height_nblocks --a-width-nblocks $(power_of_2 $a_width_nblocks) --b-width-nblocks $b_width_nblocks --a-block-height-ntiles $a_block_height_ntiles --a-block-width-ntiles $a_block_width_ntiles --b-block-width-ntiles $b_block_width_ntiles --out-subblock-height-ntiles $out_subblock_height_ntiles --out-subblock-width-ntiles $out_subblock_width_ntiles
python3 profile_scripts/custom_profile.py --file-name tt_metal/tools/profiler/logs/profile_log_device.csv --profile-target profile_Elewise_binary_multi_tile >> log/bmm_tilize_untilze_multi_tile.log
done
done
done
done
done
done
done
done
done

# # range of a_height_nblocks e.g. {0..13} range is 14
python3 profile_scripts/script.py --file-name log/bmm_tilize_untilze_multi_tile.log --profile-target Print_Elewise_Binary_Multi_Tile --num-repetitions 10 --range 3

# python3 profile_scripts/custom_profile.py --file-name tt_metal/tools/profiler/logs/profile_log_device.csv --profile-target profile_bmm_tilize_untilize_read_write
