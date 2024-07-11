// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"


TEST(AllGatherSharded_WidthShardedIndexing_FullWorkerGridVariant, AdvanceFullTileRow_ClockWise_In3x5_NumShards3) {
    bool is_clockwise = true;
    uint16_t const num_shards_x = 3;
    uint16_t const input_shard_num_tiles_x = 5;
    uint16_t const input_shard_num_tiles_y = 3;
    uint16_t const total_num_tiles_x = input_shard_num_tiles_x * num_shards_x;
    uint16_t const total_num_cores = 3;

    { // Advance to end from start of row
        uint16_t curr_core_index = 0;
        uint16_t curr_shard_tile_x = 0;
        uint16_t curr_shard_tile_y = 0;
        uint16_t curr_tile_index = (total_num_tiles_x * curr_shard_tile_y) + curr_shard_tile_x;
        uint16_t curr_shard = 0;
        uint16_t shard_offset = 0;
        uint16_t old_curr_shard = curr_shard;

       ttnn::ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
            curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_core_index, total_num_cores, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, curr_shard, is_clockwise);
        ASSERT_EQ(curr_shard_tile_x, 0);
        ASSERT_EQ(curr_shard_tile_y, 1);
        ASSERT_EQ(curr_tile_index, total_num_tiles_x);
        ASSERT_EQ(curr_shard, old_curr_shard);
        ASSERT_EQ(curr_core_index, 0);
    }
    { // Advance to end from "middle" of row
        uint16_t curr_core_index = 0;
        uint16_t curr_shard_tile_x = 3;
        uint16_t curr_shard_tile_y = 0;
        uint16_t curr_tile_index = (total_num_tiles_x * curr_shard_tile_y) + curr_shard_tile_x;
        uint16_t curr_shard = 0;
        uint16_t old_curr_shard = curr_shard;

        ttnn::ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
            curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_core_index, total_num_cores, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, curr_shard, is_clockwise);
        ASSERT_EQ(curr_shard_tile_x, 0);
        ASSERT_EQ(curr_shard_tile_y, 1);
        ASSERT_EQ(curr_tile_index, total_num_tiles_x);
        ASSERT_EQ(curr_shard, old_curr_shard);
        ASSERT_EQ(curr_core_index, 0);
    }

    { // Advance to end from start of "middle" row
        uint16_t curr_core_index = 0;
        uint16_t curr_shard_tile_x = 0;
        uint16_t curr_shard_tile_y = 1;
        uint16_t curr_tile_index = (total_num_tiles_x * curr_shard_tile_y) + curr_shard_tile_x;
        uint16_t curr_shard = 0;
        uint16_t old_curr_shard = curr_shard;
        ASSERT_EQ(curr_core_index, 0);

        ttnn::ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
            curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_core_index, total_num_cores, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, curr_shard, is_clockwise);
        ASSERT_EQ(curr_shard_tile_x, 0);
        ASSERT_EQ(curr_shard_tile_y, 2);
        ASSERT_EQ(curr_tile_index, total_num_tiles_x * 2);
        ASSERT_EQ(curr_shard, old_curr_shard);
        ASSERT_EQ(curr_core_index, 0);
    }
    { // Advance to end from "middle" of "middle" row
        uint16_t curr_core_index = 0;
        uint16_t curr_shard_tile_x = 3;
        uint16_t curr_shard_tile_y = 1;
        uint16_t curr_tile_index = (total_num_tiles_x * curr_shard_tile_y) + curr_shard_tile_x;
        uint16_t curr_shard = 0;
        uint16_t old_curr_shard = curr_shard;
        ASSERT_EQ(curr_core_index, 0);

        ttnn::ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
            curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_core_index, total_num_cores, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, curr_shard, is_clockwise);
        ASSERT_EQ(curr_shard_tile_x, 0);
        ASSERT_EQ(curr_shard_tile_y, 2);
        ASSERT_EQ(curr_tile_index, total_num_tiles_x * 2);
        ASSERT_EQ(curr_shard, old_curr_shard);
        ASSERT_EQ(curr_core_index, 0);
    }

    { // Advance to end from "start" of row, and from last row
        uint16_t curr_core_index = 0;
        uint16_t curr_shard_tile_x = 0;
        uint16_t curr_shard_tile_y = input_shard_num_tiles_y - 1;
        uint16_t curr_tile_index = (total_num_tiles_x * curr_shard_tile_y) + curr_shard_tile_x;
        uint16_t curr_shard = 0;
        uint16_t old_curr_shard = curr_shard;


        ttnn::ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
            curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_core_index, total_num_cores, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, curr_shard, is_clockwise);
        ASSERT_EQ(curr_shard_tile_x, 0);
        ASSERT_EQ(curr_shard_tile_y, 0);
        ASSERT_EQ(curr_tile_index, input_shard_num_tiles_x * old_curr_shard);
        ASSERT_EQ(curr_shard, old_curr_shard);
        ASSERT_EQ(curr_core_index, total_num_cores - 1);
    }
    { // Advance to end from "middle" of row, and from last row, first shard
        uint16_t curr_core_index = 0;
        uint16_t curr_shard_tile_x = 2;
        uint16_t curr_shard_tile_y = input_shard_num_tiles_y - 1;
        uint16_t curr_tile_index = (total_num_tiles_x * curr_shard_tile_y) + curr_shard_tile_x;
        uint16_t curr_shard = 0;
        uint16_t old_curr_shard = curr_shard;

        ttnn::ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
            curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_core_index, total_num_cores, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, curr_shard, is_clockwise);
        ASSERT_EQ(curr_shard_tile_x, 0);
        ASSERT_EQ(curr_shard_tile_y, 0);
        ASSERT_EQ(curr_tile_index, input_shard_num_tiles_x * old_curr_shard);
        ASSERT_EQ(curr_shard, old_curr_shard);
        ASSERT_EQ(curr_core_index, total_num_cores - 1);
    }
}


static uint16_t advance_core(uint16_t curr_core_index, bool is_clockwise, uint16_t total_num_cores) {
    return is_clockwise ?
        (curr_core_index == 0 ? total_num_cores - 1 : curr_core_index - 1) :
        (curr_core_index == total_num_cores - 1 ? 0 : curr_core_index + 1);
};

TEST(AllGatherSharded_WidthShardedIndexing_FullWorkerGridVariant, AdvanceFullTileRow_ClockWise_In3x5_NumShards3_Sequence) {
    bool is_clockwise = true;
    uint16_t const num_shards_x = 3;
    uint16_t const input_shard_num_tiles_x = 5;
    uint16_t const input_shard_num_tiles_y = 3;
    uint16_t const total_num_tiles_x = input_shard_num_tiles_x * num_shards_x;
    uint16_t total_num_tiles = total_num_tiles_x * input_shard_num_tiles_y;
    uint16_t const total_num_cores = 3;

    uint16_t curr_shard_tile_x = 0;
    uint16_t curr_shard_tile_y = 0;
    uint16_t curr_tile_index = 0;
    uint16_t curr_shard = 0;
    uint16_t old_curr_shard = 0;
    uint16_t curr_core_index = 0;

    uint16_t num_core_iterations = total_num_cores * 2;
    uint16_t expected_core_index = curr_core_index;
    for (uint16_t i = 0; i < num_core_iterations; i++) {

        for (uint16_t tile_row = curr_shard_tile_y; tile_row < input_shard_num_tiles_y; tile_row++) {
            ttnn::ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
                curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_core_index, total_num_cores, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, curr_shard, is_clockwise);
            uint16_t next_tile_row = tile_row + 1;
            if (next_tile_row == input_shard_num_tiles_y) {
                next_tile_row = 0;
                expected_core_index = advance_core(expected_core_index, is_clockwise, total_num_cores);
            }
            ASSERT_EQ(curr_shard_tile_x, 0);
            ASSERT_EQ(curr_shard_tile_y, next_tile_row);
            ASSERT_EQ(curr_tile_index, (next_tile_row * total_num_tiles_x) + (old_curr_shard * input_shard_num_tiles_x));
            ASSERT_EQ(curr_shard, old_curr_shard);
            ASSERT_EQ(curr_core_index, expected_core_index);
        }
    }
}


TEST(AllGatherSharded_WidthShardedIndexing_FullWorkerGridVariant, AdvanceFullTileRow_CounterClockWise_In3x5_NumShards3_Sequence) {
    bool is_clockwise = false;
    uint16_t const num_shards_x = 3;
    uint16_t const input_shard_num_tiles_x = 5;
    uint16_t const input_shard_num_tiles_y = 3;
    uint16_t const total_num_tiles_x = input_shard_num_tiles_x * num_shards_x;
    uint16_t total_num_tiles = total_num_tiles_x * input_shard_num_tiles_y;

    uint16_t curr_shard_tile_x = 0;
    uint16_t curr_shard_tile_y = 0;
    uint16_t curr_tile_index = 0;
    uint16_t curr_shard = 0;
    uint16_t old_curr_shard = curr_shard;
    uint16_t curr_core_index = 0;
    uint16_t total_num_cores = 0;

    uint16_t num_core_iterations = total_num_cores * 2;
    uint16_t expected_core_index = curr_core_index;
    for (uint16_t i = 0; i < num_core_iterations; i++) {

        for (uint16_t tile_row = curr_shard_tile_y; tile_row < input_shard_num_tiles_y; tile_row++) {
            ttnn::ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
                curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_core_index, total_num_cores, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, curr_shard, is_clockwise);
                uint16_t next_tile_row = tile_row + 1;
                if (next_tile_row == input_shard_num_tiles_y) {
                    next_tile_row = 0;
                    expected_core_index = advance_core(expected_core_index, is_clockwise, total_num_cores);
                }
                ASSERT_EQ(curr_shard_tile_x, 0);
                ASSERT_EQ(curr_shard_tile_y, next_tile_row);
                ASSERT_EQ(curr_tile_index, (next_tile_row * total_num_tiles_x) + (old_curr_shard * input_shard_num_tiles_x));
                ASSERT_EQ(curr_shard, old_curr_shard);
                ASSERT_EQ(curr_core_index, expected_core_index);
        }
    }
}



TEST(AllGatherSharded_WidthShardedIndexing_FullWorkerGridVariant, AdvanceSingleTile_ClockWise_In3x5_NumShards3_Sequence) {
    bool is_clockwise = true;
    uint16_t const num_shards_x = 3;
    uint16_t const input_shard_num_tiles_x = 5;
    uint16_t const input_shard_num_tiles_y = 3;
    uint16_t const total_num_tiles_x = input_shard_num_tiles_x * num_shards_x;
    uint16_t total_num_tiles = total_num_tiles_x * input_shard_num_tiles_y;

    uint16_t curr_shard_tile_x = 0;
    uint16_t curr_shard_tile_y = 0;
    uint16_t curr_tile_index = 0;
    uint16_t curr_shard = 0;
    uint16_t old_curr_shard = curr_shard;
    uint16_t curr_core_index = 0;
    uint16_t total_num_cores = 3;

    uint16_t num_core_iterations = total_num_cores * 2;
    uint16_t expected_core_index = curr_core_index;
    for (uint16_t i = 0; i < num_core_iterations; i++) {

        for (uint16_t tile_row = curr_shard_tile_y; tile_row < input_shard_num_tiles_y; tile_row++) {
            for (uint16_t tile_col = curr_shard_tile_x; tile_col < input_shard_num_tiles_x; tile_col++) {
                ttnn::ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance (
                    curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_core_index, total_num_cores, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, curr_shard, is_clockwise);
                uint16_t next_tile_row = tile_row;
                uint16_t next_tile_col = tile_col + 1;
                if (next_tile_col == input_shard_num_tiles_x) {
                    next_tile_col = 0;
                    next_tile_row = tile_row + 1;
                    if (next_tile_row == input_shard_num_tiles_y) {
                        next_tile_row = 0;
                        expected_core_index = advance_core(expected_core_index, is_clockwise, total_num_cores);
                    }
                }
                ASSERT_EQ(curr_shard_tile_x, next_tile_col);
                ASSERT_EQ(curr_shard_tile_y, next_tile_row);
                ASSERT_EQ(curr_tile_index, (next_tile_row * total_num_tiles_x) + (old_curr_shard * input_shard_num_tiles_x) + next_tile_col);
                ASSERT_EQ(curr_shard, old_curr_shard);
                ASSERT_EQ(curr_core_index, expected_core_index);
            }
        }
    }

}

TEST(AllGatherSharded_WidthShardedIndexing_FullWorkerGridVariant, AdvanceSingleTile_CounterClockWise_In3x5_NumShards3_Sequence) {
    bool is_clockwise = false;
    uint16_t const num_shards_x = 3;
    uint16_t const input_shard_num_tiles_x = 5;
    uint16_t const input_shard_num_tiles_y = 3;
    uint16_t const total_num_tiles_x = input_shard_num_tiles_x * num_shards_x;
    uint16_t total_num_tiles = total_num_tiles_x * input_shard_num_tiles_y;

    uint16_t curr_shard_tile_x = 0;
    uint16_t curr_shard_tile_y = 0;
    uint16_t curr_tile_index = 0;
    uint16_t curr_shard = 0;
    uint16_t const old_curr_shard = curr_shard;
    uint16_t curr_core_index = 0;
    uint16_t total_num_cores = 3;

    uint16_t num_core_iterations = total_num_cores * 2;
    uint16_t expected_core_index = curr_core_index;
    for (uint16_t i = 0; i < num_core_iterations; i++) {

        for (uint16_t tile_row = curr_shard_tile_y; tile_row < input_shard_num_tiles_y; tile_row++) {
            for (uint16_t tile_col = curr_shard_tile_x; tile_col < input_shard_num_tiles_x; tile_col++) {
                ttnn::ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance (
                    curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_core_index, total_num_cores, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, curr_shard, is_clockwise);
                uint16_t next_tile_row = tile_row;
                uint16_t next_tile_col = tile_col + 1;
                if (next_tile_col == input_shard_num_tiles_x) {
                    next_tile_col = 0;
                    next_tile_row = tile_row + 1;
                    if (next_tile_row == input_shard_num_tiles_y) {
                        next_tile_row = 0;
                        expected_core_index = advance_core(expected_core_index, is_clockwise, total_num_cores);
                    }
                }
                ASSERT_EQ(curr_shard_tile_x, next_tile_col);
                ASSERT_EQ(curr_shard_tile_y, next_tile_row);
                ASSERT_EQ(curr_tile_index, (next_tile_row * total_num_tiles_x) + (old_curr_shard * input_shard_num_tiles_x) + next_tile_col);
                ASSERT_EQ(curr_shard, old_curr_shard);
                ASSERT_EQ(curr_core_index, expected_core_index);
            }
        }
    }
}