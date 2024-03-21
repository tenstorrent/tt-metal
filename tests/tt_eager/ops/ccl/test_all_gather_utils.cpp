// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "tt_dnn/op_library/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_eager/tt_dnn/op_library/all_gather/all_gather_op.hpp"

// Col major orientation not supported yet
TEST(AllGatherUtils, OutputTensorShardAddrGenArgGenerator_GetFirstOutputShardStartingLocation_RowMajorOrientation) {
    // tt::tt_metal::OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
    //     num_workers, input_tensor_shard_grid_size, ring_index, serving_worker_index);

    {
        uint32_t ring_size = 8;
        auto const [dest_worker_index, offset_chunk_in_worker] =
            tt::tt_metal::OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
                1, 1, 0, ring_size, 0);
        ASSERT_EQ(dest_worker_index, 0);
        ASSERT_EQ(offset_chunk_in_worker, 0);
    }
    {
        uint32_t ring_size = 8;
        auto const [dest_worker_index, offset_chunk_in_worker] =
            tt::tt_metal::OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
                1, 1, 1, ring_size, 0);
        ASSERT_EQ(dest_worker_index, 0);
        ASSERT_EQ(offset_chunk_in_worker, 1);
    }
    {
        uint32_t ring_size = 2;
        auto const [dest_worker_index, offset_chunk_in_worker] =
            tt::tt_metal::OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
                4, 16, 0, ring_size, 0);
        ASSERT_EQ(dest_worker_index, 0);
        ASSERT_EQ(offset_chunk_in_worker, 0);
    }
    {
        uint32_t ring_size = 2;
        auto const [dest_worker_index, offset_chunk_in_worker] =
            tt::tt_metal::OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
                4, 16, 0, ring_size, 1);
        ASSERT_EQ(dest_worker_index, 2);
        ASSERT_EQ(offset_chunk_in_worker, 0);
    }
    {
        uint32_t ring_size = 2;
        auto const [dest_worker_index, offset_chunk_in_worker] =
            tt::tt_metal::OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
                4, 16, 1, ring_size, 0);
        ASSERT_EQ(dest_worker_index, 8);
        ASSERT_EQ(offset_chunk_in_worker, 0);
    }
    {
        uint32_t ring_size = 2;
        auto const [dest_worker_index, offset_chunk_in_worker] =
            tt::tt_metal::OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
                4, 16, 1, ring_size, 1);
        ASSERT_EQ(dest_worker_index, 10);
        ASSERT_EQ(offset_chunk_in_worker, 0);
    }
    {
        uint32_t ring_size = 8;
        uint32_t num_workers = 1;
        auto const [dest_worker_index, offset_chunk_in_worker] =
            tt::tt_metal::OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
            // num_workers, input_tensor_shard_grid_size, ring_index, ring_size, serving_worker_index);
                num_workers, 2, 0, ring_size, 0);
        ASSERT_EQ(dest_worker_index, 0);
        ASSERT_EQ(offset_chunk_in_worker, 0);
    }
    {
        uint32_t ring_size = 8;
        uint32_t num_workers = 2;
        auto const [dest_worker_index, offset_chunk_in_worker] =
            tt::tt_metal::OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
            // num_workers, input_tensor_shard_grid_size, ring_index, ring_size, serving_worker_index);
                num_workers, 2, 0, ring_size, 0);
        ASSERT_EQ(dest_worker_index, 0);
        ASSERT_EQ(offset_chunk_in_worker, 0);
    }
    {
        uint32_t ring_size = 8;
        uint32_t num_workers = 2;
        auto const [dest_worker_index, offset_chunk_in_worker] =
            tt::tt_metal::OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
            // num_workers, input_tensor_shard_grid_size, ring_index, ring_size, serving_worker_index);
                num_workers, 2, 0, ring_size, 1);
        ASSERT_EQ(dest_worker_index, 0);
        ASSERT_EQ(offset_chunk_in_worker, 1);
    }
    {
        uint32_t ring_size = 8;
        uint32_t num_workers = 2;
        auto const [dest_worker_index, offset_chunk_in_worker] =
            tt::tt_metal::OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
            // num_workers, input_tensor_shard_grid_size, ring_index, ring_size, serving_worker_index);
                num_workers, 2, 1, ring_size, 1);
        ASSERT_EQ(dest_worker_index, 0);
        ASSERT_EQ(offset_chunk_in_worker, 3);
    }
}


TEST(AllGatherUtils, OutputTensorShardAddrGenArgGenerator_ComputeWorkerDestCores_WidthSharding) {
    // tt::tt_metal::OutputTensorShardAddrGenArgGenerator::compute_worker_coord_worker_dest_cores (
    //         ccl::ShardType shard_type,
    //         std::vector<CoreCoord> const& global_shard_dest_cores,
    //         uint32_t input_num_shards,
    //         uint32_t output_num_shards,
    //         uint32_t num_workers,
    //         uint32_t worker_index,
    //         bool is_shard_orientation_row_major)

    bool is_shard_orientation_row_major = true;
    ccl::ShardType shard_type = ccl::ShardType::Width;
    {
        auto const& dest_cores = tt::tt_metal::OutputTensorShardAddrGenArgGenerator::compute_worker_coord_worker_dest_cores (
            shard_type, {CoreCoord(0,0)}, 1, 8, 1, 0, is_shard_orientation_row_major);
        ASSERT_EQ(dest_cores.size(), 1);
        ASSERT_EQ(dest_cores.at(0), CoreCoord(0,0));
    }
    {
        std::vector<CoreCoord> global_shard_dest_cores = {CoreCoord(0,0),CoreCoord(1,0)};
        uint32_t ring_size = 8;
        uint32_t num_workers = 1;
        auto const& dest_cores = tt::tt_metal::OutputTensorShardAddrGenArgGenerator::compute_worker_coord_worker_dest_cores (
            shard_type, global_shard_dest_cores, global_shard_dest_cores.size(), global_shard_dest_cores.size() * ring_size, num_workers, 0, is_shard_orientation_row_major);
        ASSERT_EQ(dest_cores.size(), 2);
        ASSERT_EQ(dest_cores.at(0), CoreCoord(0,0));
        ASSERT_EQ(dest_cores.at(1), CoreCoord(1,0));
    }
    {
        std::vector<CoreCoord> global_shard_dest_cores = {CoreCoord(0,0),CoreCoord(1,0)};
        uint32_t ring_size = 8;
        uint32_t num_workers = 2;
        auto const& dest_cores = tt::tt_metal::OutputTensorShardAddrGenArgGenerator::compute_worker_coord_worker_dest_cores (
            shard_type, global_shard_dest_cores, global_shard_dest_cores.size(), global_shard_dest_cores.size() * ring_size, num_workers, 0, is_shard_orientation_row_major);
        ASSERT_EQ(dest_cores.size(), 2);
        ASSERT_EQ(dest_cores.at(0), CoreCoord(0,0));
        ASSERT_EQ(dest_cores.at(1), CoreCoord(1,0));
        // ASSERT_EQ(dest_cores.size(), 1);
        // ASSERT_EQ(dest_cores.at(0), CoreCoord(0,0));
    }
    {
        std::vector<CoreCoord> global_shard_dest_cores = {CoreCoord(0,0),CoreCoord(1,0)};
        uint32_t ring_size = 8;
        uint32_t num_workers = 2;
        auto const& dest_cores = tt::tt_metal::OutputTensorShardAddrGenArgGenerator::compute_worker_coord_worker_dest_cores (
            shard_type, global_shard_dest_cores, global_shard_dest_cores.size(), global_shard_dest_cores.size() * ring_size, num_workers, 1, is_shard_orientation_row_major);
        ASSERT_EQ(dest_cores.size(), 2);
        ASSERT_EQ(dest_cores.at(0), CoreCoord(0,0));
        ASSERT_EQ(dest_cores.at(1), CoreCoord(1,0));
        // ASSERT_EQ(dest_cores.size(), 1);
        // ASSERT_EQ(dest_cores.at(0), CoreCoord(1,0));
    }

    // { // Unsupported For Now
    //     std::vector<CoreCoord> global_shard_dest_cores = {CoreCoord(0,0),CoreCoord(1,0),CoreCoord(2,0),CoreCoord(3,0)};
    //     uint32_t ring_size = 8;
    //     uint32_t num_workers = 2;
    //     auto const& dest_cores = tt::tt_metal::OutputTensorShardAddrGenArgGenerator::compute_worker_coord_worker_dest_cores (
    //         shard_type, global_shard_dest_cores, global_shard_dest_cores.size(), global_shard_dest_cores.size() * ring_size, num_workers, 0, is_shard_orientation_row_major);
    //     ASSERT_EQ(dest_cores.size(), 4);
    //     ASSERT_EQ(dest_cores.at(0), CoreCoord(0,0));
    //     ASSERT_EQ(dest_cores.at(1), CoreCoord(1,0));
    //     ASSERT_EQ(dest_cores.at(1), CoreCoord(2,0));
    //     ASSERT_EQ(dest_cores.at(2), CoreCoord(3,0));
    // }
    // { // Unsupported For Now
    //     std::vector<CoreCoord> global_shard_dest_cores = {CoreCoord(0,0),CoreCoord(1,0),CoreCoord(2,0),CoreCoord(3,0)};
    //     uint32_t ring_size = 8;
    //     uint32_t num_workers = 2;
    //     auto const& dest_cores = tt::tt_metal::OutputTensorShardAddrGenArgGenerator::compute_worker_coord_worker_dest_cores (
    //         shard_type, global_shard_dest_cores, global_shard_dest_cores.size(), global_shard_dest_cores.size() * ring_size, num_workers, 1, is_shard_orientation_row_major);
    //     ASSERT_EQ(dest_cores.size(), 4);
    //     ASSERT_EQ(dest_cores.at(0), CoreCoord(0,0));
    //     ASSERT_EQ(dest_cores.at(1), CoreCoord(1,0));
    //     ASSERT_EQ(dest_cores.at(1), CoreCoord(2,0));
    //     ASSERT_EQ(dest_cores.at(2), CoreCoord(3,0));
    // }

    { // shard grid size = 16, ring size = 8, num_workers = 1
        std::vector<CoreCoord> global_shard_dest_cores =
            {CoreCoord(0,0),CoreCoord(1,0)};
        uint32_t ring_size = 8;
        uint32_t num_workers = 2;
        {
            auto const& dest_cores = tt::tt_metal::OutputTensorShardAddrGenArgGenerator::compute_worker_coord_worker_dest_cores (
                shard_type, global_shard_dest_cores, global_shard_dest_cores.size(), global_shard_dest_cores.size() * ring_size, num_workers, 0, is_shard_orientation_row_major);
            ASSERT_EQ(dest_cores.size(), 2);
            ASSERT_EQ(dest_cores.at(0), CoreCoord(0,0));
            ASSERT_EQ(dest_cores.at(1), CoreCoord(1,0));
        }
        {
            auto const& dest_cores = tt::tt_metal::OutputTensorShardAddrGenArgGenerator::compute_worker_coord_worker_dest_cores (
                shard_type, global_shard_dest_cores, global_shard_dest_cores.size(), global_shard_dest_cores.size() * ring_size, num_workers, 1, is_shard_orientation_row_major);
            ASSERT_EQ(dest_cores.size(), 2);
            ASSERT_EQ(dest_cores.at(0), CoreCoord(0,0));
            ASSERT_EQ(dest_cores.at(1), CoreCoord(1,0));
        }
    }
    { // shard grid size = 16, ring size = 8, num_workers = 1
        std::vector<CoreCoord> global_shard_dest_cores =
            {CoreCoord(0,0),CoreCoord(1,0)};
        uint32_t ring_size = 8;
        uint32_t num_workers = 1;
        {
            auto const& dest_cores = tt::tt_metal::OutputTensorShardAddrGenArgGenerator::compute_worker_coord_worker_dest_cores (
                shard_type, global_shard_dest_cores, global_shard_dest_cores.size(), global_shard_dest_cores.size() * ring_size, num_workers, 0, is_shard_orientation_row_major);
            ASSERT_EQ(dest_cores.size(), 2);
            ASSERT_EQ(dest_cores.at(0), CoreCoord(0,0));
            ASSERT_EQ(dest_cores.at(1), CoreCoord(1,0));
        }
    }

    { // shard grid size = 16, ring size = 8, num_workers = 1
        std::vector<CoreCoord> global_shard_dest_cores =
            {CoreCoord(0,0),CoreCoord(1,0),CoreCoord(2,0),CoreCoord(3,0),CoreCoord(4,0),CoreCoord(5,0),CoreCoord(6,0),CoreCoord(7,0),
                CoreCoord(0,1),CoreCoord(1,1),CoreCoord(2,1),CoreCoord(3,1),CoreCoord(4,1),CoreCoord(5,1),CoreCoord(6,1),CoreCoord(7,1)};
        uint32_t ring_size = 2;
        uint32_t num_workers = 4;
        {
            auto const& dest_cores = tt::tt_metal::OutputTensorShardAddrGenArgGenerator::compute_worker_coord_worker_dest_cores (
                shard_type, global_shard_dest_cores, global_shard_dest_cores.size(), global_shard_dest_cores.size() * ring_size, num_workers, 0, is_shard_orientation_row_major);
            ASSERT_EQ(dest_cores.size(), 4);
            ASSERT_EQ(dest_cores.at(0), CoreCoord(0,0));
            ASSERT_EQ(dest_cores.at(1), CoreCoord(1,0));
            ASSERT_EQ(dest_cores.at(2), CoreCoord(0,1));
            ASSERT_EQ(dest_cores.at(3), CoreCoord(1,1));
        }
        {
            auto const& dest_cores = tt::tt_metal::OutputTensorShardAddrGenArgGenerator::compute_worker_coord_worker_dest_cores (
                shard_type, global_shard_dest_cores, global_shard_dest_cores.size(), global_shard_dest_cores.size() * ring_size, num_workers, 1, is_shard_orientation_row_major);
            ASSERT_EQ(dest_cores.size(), 4);
            ASSERT_EQ(dest_cores.at(0), CoreCoord(2,0));
            ASSERT_EQ(dest_cores.at(1), CoreCoord(3,0));
            ASSERT_EQ(dest_cores.at(2), CoreCoord(2,1));
            ASSERT_EQ(dest_cores.at(3), CoreCoord(3,1));
        }
        {
            auto const& dest_cores = tt::tt_metal::OutputTensorShardAddrGenArgGenerator::compute_worker_coord_worker_dest_cores (
                shard_type, global_shard_dest_cores, global_shard_dest_cores.size(), global_shard_dest_cores.size() * ring_size, num_workers, 2, is_shard_orientation_row_major);
            ASSERT_EQ(dest_cores.size(), 4);
            ASSERT_EQ(dest_cores.at(0), CoreCoord(4,0));
            ASSERT_EQ(dest_cores.at(1), CoreCoord(5,0));
            ASSERT_EQ(dest_cores.at(2), CoreCoord(4,1));
            ASSERT_EQ(dest_cores.at(3), CoreCoord(5,1));
        }
        {
            auto const& dest_cores = tt::tt_metal::OutputTensorShardAddrGenArgGenerator::compute_worker_coord_worker_dest_cores (
                shard_type, global_shard_dest_cores, global_shard_dest_cores.size(), global_shard_dest_cores.size() * ring_size, num_workers, 3, is_shard_orientation_row_major);
            ASSERT_EQ(dest_cores.size(), 4);
            ASSERT_EQ(dest_cores.at(0), CoreCoord(6,0));
            ASSERT_EQ(dest_cores.at(1), CoreCoord(7,0));
            ASSERT_EQ(dest_cores.at(2), CoreCoord(6,1));
            ASSERT_EQ(dest_cores.at(3), CoreCoord(7,1));
        }
    }

    { // shard grid size = 64, ring size = 8, num_workers = 4
        std::vector<CoreCoord> global_shard_dest_cores =
            {CoreCoord(0,0),CoreCoord(1,0),CoreCoord(2,0),CoreCoord(3,0),CoreCoord(4,0),CoreCoord(5,0),CoreCoord(6,0),CoreCoord(7,0),
                CoreCoord(0,1),CoreCoord(1,1),CoreCoord(2,1),CoreCoord(3,1),CoreCoord(4,1),CoreCoord(5,1),CoreCoord(6,1),CoreCoord(7,1),
                CoreCoord(0,2),CoreCoord(1,2),CoreCoord(2,2),CoreCoord(3,2),CoreCoord(4,2),CoreCoord(5,2),CoreCoord(6,2),CoreCoord(7,2),
                CoreCoord(0,3),CoreCoord(1,3),CoreCoord(2,3),CoreCoord(3,3),CoreCoord(4,3),CoreCoord(5,3),CoreCoord(6,3),CoreCoord(7,3),
                CoreCoord(0,4),CoreCoord(1,4),CoreCoord(2,4),CoreCoord(3,4),CoreCoord(4,4),CoreCoord(5,4),CoreCoord(6,4),CoreCoord(7,4),
                CoreCoord(0,5),CoreCoord(1,5),CoreCoord(2,5),CoreCoord(3,5),CoreCoord(4,5),CoreCoord(5,5),CoreCoord(6,5),CoreCoord(7,5),
                CoreCoord(0,6),CoreCoord(1,6),CoreCoord(2,6),CoreCoord(3,6),CoreCoord(4,6),CoreCoord(5,6),CoreCoord(6,6),CoreCoord(7,6),
                CoreCoord(0,7),CoreCoord(1,7),CoreCoord(2,7),CoreCoord(3,7),CoreCoord(4,7),CoreCoord(5,7),CoreCoord(6,7),CoreCoord(7,7)};
        EXPECT_EQ(global_shard_dest_cores.size(), 64); // otherwise I set it up wrong D:
        uint32_t ring_size = 8;
        uint32_t num_workers = 4;
        {
            auto const& dest_cores = tt::tt_metal::OutputTensorShardAddrGenArgGenerator::compute_worker_coord_worker_dest_cores (
                shard_type, global_shard_dest_cores, global_shard_dest_cores.size(), global_shard_dest_cores.size() * ring_size, num_workers, 0, is_shard_orientation_row_major);
            ASSERT_EQ(dest_cores.size(), 16);
            ASSERT_EQ(dest_cores.at(0), CoreCoord(0,0));   ASSERT_EQ(dest_cores.at(1), CoreCoord(1,0));
            ASSERT_EQ(dest_cores.at(2), CoreCoord(0,1));   ASSERT_EQ(dest_cores.at(3), CoreCoord(1,1));
            ASSERT_EQ(dest_cores.at(4), CoreCoord(0,2));   ASSERT_EQ(dest_cores.at(5), CoreCoord(1,2));
            ASSERT_EQ(dest_cores.at(6), CoreCoord(0,3));   ASSERT_EQ(dest_cores.at(7), CoreCoord(1,3));
            ASSERT_EQ(dest_cores.at(8), CoreCoord(0,4));   ASSERT_EQ(dest_cores.at(9), CoreCoord(1,4));
            ASSERT_EQ(dest_cores.at(10), CoreCoord(0,5));  ASSERT_EQ(dest_cores.at(11), CoreCoord(1,5));
            ASSERT_EQ(dest_cores.at(12), CoreCoord(0,6));  ASSERT_EQ(dest_cores.at(13), CoreCoord(1,6));
            ASSERT_EQ(dest_cores.at(14), CoreCoord(0,7));  ASSERT_EQ(dest_cores.at(15), CoreCoord(1,7));
        }
        {
            auto const& dest_cores = tt::tt_metal::OutputTensorShardAddrGenArgGenerator::compute_worker_coord_worker_dest_cores (
                shard_type, global_shard_dest_cores, global_shard_dest_cores.size(), global_shard_dest_cores.size() * ring_size, num_workers, 1, is_shard_orientation_row_major);
            ASSERT_EQ(dest_cores.size(), 16);
            ASSERT_EQ(dest_cores.at(0), CoreCoord(2,0));   ASSERT_EQ(dest_cores.at(1), CoreCoord(3,0));
            ASSERT_EQ(dest_cores.at(2), CoreCoord(2,1));   ASSERT_EQ(dest_cores.at(3), CoreCoord(3,1));
            ASSERT_EQ(dest_cores.at(4), CoreCoord(2,2));   ASSERT_EQ(dest_cores.at(5), CoreCoord(3,2));
            ASSERT_EQ(dest_cores.at(6), CoreCoord(2,3));   ASSERT_EQ(dest_cores.at(7), CoreCoord(3,3));
            ASSERT_EQ(dest_cores.at(8), CoreCoord(2,4));   ASSERT_EQ(dest_cores.at(9), CoreCoord(3,4));
            ASSERT_EQ(dest_cores.at(10), CoreCoord(2,5));  ASSERT_EQ(dest_cores.at(11), CoreCoord(3,5));
            ASSERT_EQ(dest_cores.at(12), CoreCoord(2,6));  ASSERT_EQ(dest_cores.at(13), CoreCoord(3,6));
            ASSERT_EQ(dest_cores.at(14), CoreCoord(2,7));  ASSERT_EQ(dest_cores.at(15), CoreCoord(3,7));
        }
        {
            auto const& dest_cores = tt::tt_metal::OutputTensorShardAddrGenArgGenerator::compute_worker_coord_worker_dest_cores (
                shard_type, global_shard_dest_cores, global_shard_dest_cores.size(), global_shard_dest_cores.size() * ring_size, num_workers, 2, is_shard_orientation_row_major);
            ASSERT_EQ(dest_cores.size(), 16);
            ASSERT_EQ(dest_cores.at(0), CoreCoord(4,0));   ASSERT_EQ(dest_cores.at(1), CoreCoord(5,0));
            ASSERT_EQ(dest_cores.at(2), CoreCoord(4,1));   ASSERT_EQ(dest_cores.at(3), CoreCoord(5,1));
            ASSERT_EQ(dest_cores.at(4), CoreCoord(4,2));   ASSERT_EQ(dest_cores.at(5), CoreCoord(5,2));
            ASSERT_EQ(dest_cores.at(6), CoreCoord(4,3));   ASSERT_EQ(dest_cores.at(7), CoreCoord(5,3));
            ASSERT_EQ(dest_cores.at(8), CoreCoord(4,4));   ASSERT_EQ(dest_cores.at(9), CoreCoord(5,4));
            ASSERT_EQ(dest_cores.at(10), CoreCoord(4,5));  ASSERT_EQ(dest_cores.at(11), CoreCoord(5,5));
            ASSERT_EQ(dest_cores.at(12), CoreCoord(4,6));  ASSERT_EQ(dest_cores.at(13), CoreCoord(5,6));
            ASSERT_EQ(dest_cores.at(14), CoreCoord(4,7));  ASSERT_EQ(dest_cores.at(15), CoreCoord(5,7));
        }
        {
            auto const& dest_cores = tt::tt_metal::OutputTensorShardAddrGenArgGenerator::compute_worker_coord_worker_dest_cores (
                shard_type, global_shard_dest_cores, global_shard_dest_cores.size(), global_shard_dest_cores.size() * ring_size, num_workers, 3, is_shard_orientation_row_major);
            ASSERT_EQ(dest_cores.size(), 16);
            ASSERT_EQ(dest_cores.at(0), CoreCoord(6,0));   ASSERT_EQ(dest_cores.at(1), CoreCoord(7,0));
            ASSERT_EQ(dest_cores.at(2), CoreCoord(6,1));   ASSERT_EQ(dest_cores.at(3), CoreCoord(7,1));
            ASSERT_EQ(dest_cores.at(4), CoreCoord(6,2));   ASSERT_EQ(dest_cores.at(5), CoreCoord(7,2));
            ASSERT_EQ(dest_cores.at(6), CoreCoord(6,3));   ASSERT_EQ(dest_cores.at(7), CoreCoord(7,3));
            ASSERT_EQ(dest_cores.at(8), CoreCoord(6,4));   ASSERT_EQ(dest_cores.at(9), CoreCoord(7,4));
            ASSERT_EQ(dest_cores.at(10), CoreCoord(6,5));  ASSERT_EQ(dest_cores.at(11), CoreCoord(7,5));
            ASSERT_EQ(dest_cores.at(12), CoreCoord(6,6));  ASSERT_EQ(dest_cores.at(13), CoreCoord(7,6));
            ASSERT_EQ(dest_cores.at(14), CoreCoord(6,7));  ASSERT_EQ(dest_cores.at(15), CoreCoord(7,7));
        }
    }

    // TODO: Add a 64 core, 8 ring size test
}
