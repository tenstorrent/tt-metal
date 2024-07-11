// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/all_gather/device/all_gather_op.hpp"


//////////////////////////////////////////////////////
/// ttnn::utils::InputTensorShardAddrGenArgGenerator TESTS
//////////////////////////////////////////////////////

// Col major orientation not supported yet
TEST(AllGatherUtils, OutputTensorShardAddrGenArgGenerator_GetFirstOutputShardStartingLocation_RowMajorOrientation) {
    // ttnn::utils::OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
    //     num_workers, input_tensor_shard_grid_size, ring_index, serving_worker_index);
    {
        uint32_t ring_size = 8;
        auto const [dest_worker_index, offset_chunk_in_worker] =
            ttnn::utils::OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
                1, 1, 0, ring_size, 0);
        ASSERT_EQ(dest_worker_index, 0);
        ASSERT_EQ(offset_chunk_in_worker, 0);
    }
    {
        uint32_t ring_size = 8;
        auto const [dest_worker_index, offset_chunk_in_worker] =
            ttnn::utils::OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
                1, 1, 1, ring_size, 0);
        ASSERT_EQ(dest_worker_index, 0);
        ASSERT_EQ(offset_chunk_in_worker, 1);
    }
    {
        uint32_t ring_size = 2;
        auto const [dest_worker_index, offset_chunk_in_worker] =
            ttnn::utils::OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
                4, 16, 0, ring_size, 0);
        ASSERT_EQ(dest_worker_index, 0);
        ASSERT_EQ(offset_chunk_in_worker, 0);
    }
    {
        uint32_t ring_size = 2;
        auto const [dest_worker_index, offset_chunk_in_worker] =
            ttnn::utils::OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
                4, 16, 0, ring_size, 1);
        ASSERT_EQ(dest_worker_index, 2);
        ASSERT_EQ(offset_chunk_in_worker, 0);
    }
    {
        uint32_t ring_size = 2;
        auto const [dest_worker_index, offset_chunk_in_worker] =
            ttnn::utils::OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
                4, 16, 1, ring_size, 0);
        ASSERT_EQ(dest_worker_index, 8);
        ASSERT_EQ(offset_chunk_in_worker, 0);
    }
    {
        uint32_t ring_size = 2;
        auto const [dest_worker_index, offset_chunk_in_worker] =
            ttnn::utils::OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
                4, 16, 1, ring_size, 1);
        ASSERT_EQ(dest_worker_index, 10);
        ASSERT_EQ(offset_chunk_in_worker, 0);
    }
    {
        uint32_t ring_size = 8;
        uint32_t num_workers = 1;
        auto const [dest_worker_index, offset_chunk_in_worker] =
            ttnn::utils::OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
                num_workers, 2, 0, ring_size, 0);
        ASSERT_EQ(dest_worker_index, 0);
        ASSERT_EQ(offset_chunk_in_worker, 0);
    }
    {
        uint32_t ring_size = 8;
        uint32_t num_workers = 2;
        auto const [dest_worker_index, offset_chunk_in_worker] =
            ttnn::utils::OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
                num_workers, 2, 0, ring_size, 0);
        ASSERT_EQ(dest_worker_index, 0);
        ASSERT_EQ(offset_chunk_in_worker, 0);
    }
    {
        uint32_t ring_size = 8;
        uint32_t num_workers = 2;
        auto const [dest_worker_index, offset_chunk_in_worker] =
            ttnn::utils::OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
                num_workers, 2, 0, ring_size, 1);
        ASSERT_EQ(dest_worker_index, 0);
        ASSERT_EQ(offset_chunk_in_worker, 1);
    }
    {
        uint32_t ring_size = 8;
        uint32_t num_workers = 2;
        auto const [dest_worker_index, offset_chunk_in_worker] =
            ttnn::utils::OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
                num_workers, 2, 1, ring_size, 1);
        ASSERT_EQ(dest_worker_index, 0);
        ASSERT_EQ(offset_chunk_in_worker, 3);
    }
    {
        uint32_t ring_size = 8;
        uint32_t num_workers = 8;
        auto const [dest_worker_index, offset_chunk_in_worker] =
            ttnn::utils::OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
                num_workers, 32, 1, ring_size, 0);
        ASSERT_EQ(dest_worker_index, 4);
        ASSERT_EQ(offset_chunk_in_worker, 0);
    }
}


TEST(AllGatherUtils, OutputTensorShardAddrGenArgGenerator_ComputeWorkerDestCores_WidthSharding) {
    bool is_shard_orientation_row_major = true;
    ttnn::utils::ccl::ShardType shard_type = ttnn::utils::ccl::ShardType::Width;

    { // shard grid size = 32, ring size = 8, num_workers = 8
        std::vector<CoreCoord> global_shard_dest_cores =
            {CoreCoord(0,0),CoreCoord(1,0),CoreCoord(2,0),CoreCoord(3,0),CoreCoord(4,0),CoreCoord(5,0),CoreCoord(6,0),CoreCoord(7,0),
                CoreCoord(0,1),CoreCoord(1,1),CoreCoord(2,1),CoreCoord(3,1),CoreCoord(4,1),CoreCoord(5,1),CoreCoord(6,1),CoreCoord(7,1),
                CoreCoord(0,2),CoreCoord(1,2),CoreCoord(2,2),CoreCoord(3,2),CoreCoord(4,2),CoreCoord(5,2),CoreCoord(6,2),CoreCoord(7,2),
                CoreCoord(0,3),CoreCoord(1,3),CoreCoord(2,3),CoreCoord(3,3),CoreCoord(4,3),CoreCoord(5,3),CoreCoord(6,3),CoreCoord(7,3)};
        EXPECT_EQ(global_shard_dest_cores.size(), 32); // otherwise I set it up wrong
        uint32_t ring_size = 8;
        uint32_t num_workers = 8;
        {
            std::cout << "HERE" << std::endl;
            auto const& dest_cores = ttnn::utils::OutputTensorShardAddrGenArgGenerator::compute_worker_coord_worker_dest_cores (
                shard_type, global_shard_dest_cores, global_shard_dest_cores.size(), global_shard_dest_cores.size() * ring_size, num_workers, 0, is_shard_orientation_row_major);
            ASSERT_EQ(dest_cores.size(), 8);
            ASSERT_EQ(dest_cores.at(0), CoreCoord(0,0));   ASSERT_EQ(dest_cores.at(1), CoreCoord(4,0));
            ASSERT_EQ(dest_cores.at(2), CoreCoord(0,1));   ASSERT_EQ(dest_cores.at(3), CoreCoord(4,1));
            ASSERT_EQ(dest_cores.at(4), CoreCoord(0,2));   ASSERT_EQ(dest_cores.at(5), CoreCoord(4,2));
            ASSERT_EQ(dest_cores.at(6), CoreCoord(0,3));   ASSERT_EQ(dest_cores.at(7), CoreCoord(4,3));

        }
        {
            auto const& dest_cores = ttnn::utils::OutputTensorShardAddrGenArgGenerator::compute_worker_coord_worker_dest_cores (
                shard_type, global_shard_dest_cores, global_shard_dest_cores.size(), global_shard_dest_cores.size() * ring_size, num_workers, 1, is_shard_orientation_row_major);
            ASSERT_EQ(dest_cores.size(), 8);
            ASSERT_EQ(dest_cores.at(0), CoreCoord(0,0));   ASSERT_EQ(dest_cores.at(1), CoreCoord(4,0));
            ASSERT_EQ(dest_cores.at(2), CoreCoord(0,1));   ASSERT_EQ(dest_cores.at(3), CoreCoord(4,1));
            ASSERT_EQ(dest_cores.at(4), CoreCoord(0,2));   ASSERT_EQ(dest_cores.at(5), CoreCoord(4,2));
            ASSERT_EQ(dest_cores.at(6), CoreCoord(0,3));   ASSERT_EQ(dest_cores.at(7), CoreCoord(4,3));

        }
        {
            auto const& dest_cores = ttnn::utils::OutputTensorShardAddrGenArgGenerator::compute_worker_coord_worker_dest_cores (
                shard_type, global_shard_dest_cores, global_shard_dest_cores.size(), global_shard_dest_cores.size() * ring_size, num_workers, 2, is_shard_orientation_row_major);
            ASSERT_EQ(dest_cores.size(), 8);
            ASSERT_EQ(dest_cores.at(0), CoreCoord(1,0));   ASSERT_EQ(dest_cores.at(1), CoreCoord(5,0));
            ASSERT_EQ(dest_cores.at(2), CoreCoord(1,1));   ASSERT_EQ(dest_cores.at(3), CoreCoord(5,1));
            ASSERT_EQ(dest_cores.at(4), CoreCoord(1,2));   ASSERT_EQ(dest_cores.at(5), CoreCoord(5,2));
            ASSERT_EQ(dest_cores.at(6), CoreCoord(1,3));   ASSERT_EQ(dest_cores.at(7), CoreCoord(5,3));

        }
        {
            auto const& dest_cores = ttnn::utils::OutputTensorShardAddrGenArgGenerator::compute_worker_coord_worker_dest_cores (
                shard_type, global_shard_dest_cores, global_shard_dest_cores.size(), global_shard_dest_cores.size() * ring_size, num_workers, 3, is_shard_orientation_row_major);
            ASSERT_EQ(dest_cores.size(), 8);
            ASSERT_EQ(dest_cores.at(0), CoreCoord(1,0));   ASSERT_EQ(dest_cores.at(1), CoreCoord(5,0));
            ASSERT_EQ(dest_cores.at(2), CoreCoord(1,1));   ASSERT_EQ(dest_cores.at(3), CoreCoord(5,1));
            ASSERT_EQ(dest_cores.at(4), CoreCoord(1,2));   ASSERT_EQ(dest_cores.at(5), CoreCoord(5,2));
            ASSERT_EQ(dest_cores.at(6), CoreCoord(1,3));   ASSERT_EQ(dest_cores.at(7), CoreCoord(5,3));

        }
        {
            auto const& dest_cores = ttnn::utils::OutputTensorShardAddrGenArgGenerator::compute_worker_coord_worker_dest_cores (
                shard_type, global_shard_dest_cores, global_shard_dest_cores.size(), global_shard_dest_cores.size() * ring_size, num_workers, 4, is_shard_orientation_row_major);
            ASSERT_EQ(dest_cores.size(), 8);
            ASSERT_EQ(dest_cores.at(0), CoreCoord(2,0));   ASSERT_EQ(dest_cores.at(1), CoreCoord(6,0));
            ASSERT_EQ(dest_cores.at(2), CoreCoord(2,1));   ASSERT_EQ(dest_cores.at(3), CoreCoord(6,1));
            ASSERT_EQ(dest_cores.at(4), CoreCoord(2,2));   ASSERT_EQ(dest_cores.at(5), CoreCoord(6,2));
            ASSERT_EQ(dest_cores.at(6), CoreCoord(2,3));   ASSERT_EQ(dest_cores.at(7), CoreCoord(6,3));

        }
        {
            auto const& dest_cores = ttnn::utils::OutputTensorShardAddrGenArgGenerator::compute_worker_coord_worker_dest_cores (
                shard_type, global_shard_dest_cores, global_shard_dest_cores.size(), global_shard_dest_cores.size() * ring_size, num_workers, 5, is_shard_orientation_row_major);
            ASSERT_EQ(dest_cores.size(), 8);
            ASSERT_EQ(dest_cores.at(0), CoreCoord(2,0));   ASSERT_EQ(dest_cores.at(1), CoreCoord(6,0));
            ASSERT_EQ(dest_cores.at(2), CoreCoord(2,1));   ASSERT_EQ(dest_cores.at(3), CoreCoord(6,1));
            ASSERT_EQ(dest_cores.at(4), CoreCoord(2,2));   ASSERT_EQ(dest_cores.at(5), CoreCoord(6,2));
            ASSERT_EQ(dest_cores.at(6), CoreCoord(2,3));   ASSERT_EQ(dest_cores.at(7), CoreCoord(6,3));

        }
        {
            auto const& dest_cores = ttnn::utils::OutputTensorShardAddrGenArgGenerator::compute_worker_coord_worker_dest_cores (
                shard_type, global_shard_dest_cores, global_shard_dest_cores.size(), global_shard_dest_cores.size() * ring_size, num_workers, 6, is_shard_orientation_row_major);
            ASSERT_EQ(dest_cores.size(), 8);
            ASSERT_EQ(dest_cores.at(0), CoreCoord(3,0));   ASSERT_EQ(dest_cores.at(1), CoreCoord(7,0));
            ASSERT_EQ(dest_cores.at(2), CoreCoord(3,1));   ASSERT_EQ(dest_cores.at(3), CoreCoord(7,1));
            ASSERT_EQ(dest_cores.at(4), CoreCoord(3,2));   ASSERT_EQ(dest_cores.at(5), CoreCoord(7,2));
            ASSERT_EQ(dest_cores.at(6), CoreCoord(3,3));   ASSERT_EQ(dest_cores.at(7), CoreCoord(7,3));

        }
        {
            auto const& dest_cores = ttnn::utils::OutputTensorShardAddrGenArgGenerator::compute_worker_coord_worker_dest_cores (
                shard_type, global_shard_dest_cores, global_shard_dest_cores.size(), global_shard_dest_cores.size() * ring_size, num_workers, 7, is_shard_orientation_row_major);
            ASSERT_EQ(dest_cores.size(), 8);
            ASSERT_EQ(dest_cores.at(0), CoreCoord(3,0));   ASSERT_EQ(dest_cores.at(1), CoreCoord(7,0));
            ASSERT_EQ(dest_cores.at(2), CoreCoord(3,1));   ASSERT_EQ(dest_cores.at(3), CoreCoord(7,1));
            ASSERT_EQ(dest_cores.at(4), CoreCoord(3,2));   ASSERT_EQ(dest_cores.at(5), CoreCoord(7,2));
            ASSERT_EQ(dest_cores.at(6), CoreCoord(3,3));   ASSERT_EQ(dest_cores.at(7), CoreCoord(7,3));

        }
    }


    // TODO: Add a 64 core, 8 ring size test
}


TEST(AllGatherUtils, OutputTensorShardAddrGenArgGenerator_GetIntraCoreStrideInShards) {

    {
        uint32_t input_shard_grid_size = 2;
        uint32_t num_workers = 2;
        uint32_t ring_size = 8;
        auto stride = ttnn::utils::OutputTensorShardAddrGenArgGenerator::get_intra_core_stride_in_shards(input_shard_grid_size,num_workers,ring_size);
        ASSERT_EQ(stride, 2);
    }
    {
        uint32_t input_shard_grid_size = 4;
        uint32_t num_workers = 2;
        uint32_t ring_size = 8;
        auto stride = ttnn::utils::OutputTensorShardAddrGenArgGenerator::get_intra_core_stride_in_shards(input_shard_grid_size,num_workers,ring_size);
        ASSERT_EQ(stride, 3);
    }
    {
        uint32_t input_shard_grid_size = 16;
        uint32_t num_workers = 4;
        uint32_t ring_size = 8;
        auto stride = ttnn::utils::OutputTensorShardAddrGenArgGenerator::get_intra_core_stride_in_shards(input_shard_grid_size,num_workers,ring_size);
        // Since we should be striding past the end of the core for this case, we don't care
        // so either of these values would be valid
        // the first would be the hypothetical stride if ring_size was bigger
        // stride = 1 would be equivalent to no special extra stride
        ASSERT_TRUE(stride == 5);
    }
    {
        uint32_t input_shard_grid_size = 56;
        uint32_t num_workers = 1;
        uint32_t ring_size = 8;
        auto stride = ttnn::utils::OutputTensorShardAddrGenArgGenerator::get_intra_core_stride_in_shards(input_shard_grid_size,num_workers,ring_size);
        ASSERT_EQ(stride, 1);
    }

}

TEST(AllGatherUtils, OutputTensorShardAddrGenArgGenerator_GetContiguousChunkCount) {

    {
        uint32_t input_shard_grid_size = 1;
        uint32_t num_workers = 1;
        uint32_t ring_size = 8;
        auto num_contiguous_shards = ttnn::utils::OutputTensorShardAddrGenArgGenerator::get_contiguous_chunks_before_stride(input_shard_grid_size, num_workers, ring_size);
        TT_ASSERT(num_contiguous_shards, 1);
    }
    {
        uint32_t input_shard_grid_size = 2;
        uint32_t num_workers = 2;
        uint32_t ring_size = 8;
        auto num_contiguous_shards = ttnn::utils::OutputTensorShardAddrGenArgGenerator::get_contiguous_chunks_before_stride(input_shard_grid_size, num_workers, ring_size);
        TT_ASSERT(num_contiguous_shards, 1);
    }
    {
        uint32_t input_shard_grid_size = 4;
        uint32_t num_workers = 2;
        uint32_t ring_size = 8;
        auto num_contiguous_shards = ttnn::utils::OutputTensorShardAddrGenArgGenerator::get_contiguous_chunks_before_stride(input_shard_grid_size, num_workers, ring_size);
        TT_ASSERT(num_contiguous_shards, 2);
    }
    {
        uint32_t input_shard_grid_size = 16;
        uint32_t num_workers = 4;
        uint32_t ring_size = 8;
        auto num_contiguous_shards = ttnn::utils::OutputTensorShardAddrGenArgGenerator::get_contiguous_chunks_before_stride(input_shard_grid_size, num_workers, ring_size);
        TT_ASSERT(num_contiguous_shards, 4);
    }
    {
        uint32_t input_shard_grid_size = 56;
        uint32_t num_workers = 1;
        uint32_t ring_size = 8;
        auto num_contiguous_shards = ttnn::utils::OutputTensorShardAddrGenArgGenerator::get_contiguous_chunks_before_stride(input_shard_grid_size, num_workers, ring_size);
        TT_ASSERT(num_contiguous_shards, 1);
    }
    {
        uint32_t input_shard_grid_size = 32;
        uint32_t num_workers = 8;
        uint32_t ring_size = 8;
        auto num_contiguous_shards = ttnn::utils::OutputTensorShardAddrGenArgGenerator::get_contiguous_chunks_before_stride(input_shard_grid_size, num_workers, ring_size);
        TT_ASSERT(num_contiguous_shards, 4);
    }
}

TEST(AllGatherUtils, OutputTensorShardAddrGenArgGenerator_GetContiguousChunksBeforeStrideAndContiguousChunksBeforeStride) {
    {
        uint32_t input_shard_grid_size = 1;
        uint32_t num_workers = 1;
        uint32_t ring_size = 8;
        auto num_contiguous_shards = ttnn::utils::OutputTensorShardAddrGenArgGenerator::get_contiguous_chunks_before_stride(input_shard_grid_size, num_workers, ring_size);
        auto intra_core_stride_in_chunks = ttnn::utils::OutputTensorShardAddrGenArgGenerator::get_intra_core_stride_in_shards(input_shard_grid_size, num_workers, ring_size);
        ASSERT_EQ(num_contiguous_shards, intra_core_stride_in_chunks);
    }
    {
        uint32_t input_shard_grid_size = 2;
        uint32_t num_workers = 1;
        uint32_t ring_size = 8;
        auto num_contiguous_shards = ttnn::utils::OutputTensorShardAddrGenArgGenerator::get_contiguous_chunks_before_stride(input_shard_grid_size, num_workers, ring_size);
        auto intra_core_stride_in_chunks = ttnn::utils::OutputTensorShardAddrGenArgGenerator::get_intra_core_stride_in_shards(input_shard_grid_size, num_workers, ring_size);
        ASSERT_EQ(num_contiguous_shards, 1);
        ASSERT_EQ(intra_core_stride_in_chunks, 1);
    }
    {
        uint32_t input_shard_grid_size = 16;
        uint32_t num_workers = 4;
        uint32_t ring_size = 8;
        auto num_contiguous_shards = ttnn::utils::OutputTensorShardAddrGenArgGenerator::get_contiguous_chunks_before_stride(input_shard_grid_size, num_workers, ring_size);
        auto intra_core_stride_in_chunks = ttnn::utils::OutputTensorShardAddrGenArgGenerator::get_intra_core_stride_in_shards(input_shard_grid_size, num_workers, ring_size);
        ASSERT_TRUE(num_contiguous_shards == 4 && intra_core_stride_in_chunks == 5);
    }
    {
        uint32_t input_shard_grid_size = 32;
        uint32_t num_workers = 8;
        uint32_t ring_size = 8;
        auto num_contiguous_shards = ttnn::utils::OutputTensorShardAddrGenArgGenerator::get_contiguous_chunks_before_stride(input_shard_grid_size, num_workers, ring_size);
        auto intra_core_stride_in_chunks = ttnn::utils::OutputTensorShardAddrGenArgGenerator::get_intra_core_stride_in_shards(input_shard_grid_size, num_workers, ring_size);
        ASSERT_TRUE(num_contiguous_shards == 4 && intra_core_stride_in_chunks == 5);
    }
}

//////////////////////////////////////////////////////
/// ttnn::utils::InputTensorShardAddrGenArgGenerator TESTS
//////////////////////////////////////////////////////

TEST(AllGatherUtils, InputTensorShardAddrGenArgGenerator_CtorGenerateDestCoresWidthSharding_2Workers_WorkerIdx0_2Cores)
    {
        CoreRangeSet const& all_shard_cores = CoreRangeSet({CoreRange{CoreCoord(0,0)}, CoreCoord(1,0)});
        uint32_t num_workers = 2;
        uint32_t worker_index = 0;
        auto const& dest_cores = ttnn::utils::InputTensorShardAddrGenArgGenerator::ctor_generate_dest_cores(
            CoreRangeSet(all_shard_cores), worker_index, num_workers);
        ASSERT_EQ(dest_cores.size(), 1);
        ASSERT_EQ(dest_cores.at(0), CoreCoord(0,0));
    }
TEST(AllGatherUtils, InputTensorShardAddrGenArgGenerator_CtorGenerateDestCoresWidthSharding_2Workers_WorkerIdx1_2Cores)
    {
        CoreRangeSet const& all_shard_cores = CoreRangeSet({CoreRange{CoreCoord(0,0), CoreCoord(1,0)}});
        uint32_t num_workers = 2;
        uint32_t worker_index = 1;
        auto const& dest_cores = ttnn::utils::InputTensorShardAddrGenArgGenerator::ctor_generate_dest_cores(
            all_shard_cores, worker_index, num_workers);
        ASSERT_EQ(dest_cores.size(), 1);
        ASSERT_EQ(dest_cores.at(0), CoreCoord(1,0));
    }

TEST(AllGatherUtils, InputTensorShardAddrGenArgGenerator_CtorGenerateDestCoresWidthSharding_2Workers_WorkerIdx0_4Cores)
    {
        CoreRangeSet const& all_shard_cores = CoreRangeSet({CoreRange{CoreCoord(0,0), CoreCoord(3,0)}});
        uint32_t num_workers = 2;
        uint32_t worker_index = 0;
        auto const& dest_cores = ttnn::utils::InputTensorShardAddrGenArgGenerator::ctor_generate_dest_cores(
            all_shard_cores, worker_index, num_workers);
        ASSERT_EQ(dest_cores.size(), 2);
        ASSERT_EQ(dest_cores.at(0), CoreCoord(0,0));
        ASSERT_EQ(dest_cores.at(1), CoreCoord(1,0));
    }
TEST(AllGatherUtils, InputTensorShardAddrGenArgGenerator_CtorGenerateDestCoresWidthSharding_2Workers_WorkerIdx1_4Cores)
    {
        CoreRangeSet const& all_shard_cores = CoreRangeSet({CoreRange{CoreCoord(0,0), CoreCoord(3,0)}});
        uint32_t num_workers = 2;
        uint32_t worker_index = 1;
        auto const& dest_cores = ttnn::utils::InputTensorShardAddrGenArgGenerator::ctor_generate_dest_cores(
            all_shard_cores, worker_index, num_workers);
        ASSERT_EQ(dest_cores.size(), 2);
        ASSERT_EQ(dest_cores.at(0), CoreCoord(2,0));
        ASSERT_EQ(dest_cores.at(1), CoreCoord(3,0));
    }
TEST(AllGatherUtils, InputTensorShardAddrGenArgGenerator_CtorGenerateDestCoresWidthSharding_4Workers_WorkerIdx0_16Cores)
    {
        CoreRangeSet const& all_shard_cores = CoreRangeSet({CoreRange{CoreCoord(0,0), CoreCoord(7,1)}});
        uint32_t num_workers = 4;
        uint32_t worker_index = 0;
        auto const& dest_cores = ttnn::utils::InputTensorShardAddrGenArgGenerator::ctor_generate_dest_cores(
            all_shard_cores, worker_index, num_workers);
        ASSERT_EQ(dest_cores.size(), 4);
        ASSERT_EQ(dest_cores.at(0), CoreCoord(0,0));
        ASSERT_EQ(dest_cores.at(1), CoreCoord(1,0));
        ASSERT_EQ(dest_cores.at(2), CoreCoord(2,0));
        ASSERT_EQ(dest_cores.at(3), CoreCoord(3,0));
    }
TEST(AllGatherUtils, InputTensorShardAddrGenArgGenerator_CtorGenerateDestCoresWidthSharding_4Workers_WorkerIdx1_16Cores)
    {
        CoreRangeSet const& all_shard_cores = CoreRangeSet({CoreRange{CoreCoord(0,0), CoreCoord(7,1)}});
        uint32_t num_workers = 4;
        uint32_t worker_index = 1;
        auto const& dest_cores = ttnn::utils::InputTensorShardAddrGenArgGenerator::ctor_generate_dest_cores(
            all_shard_cores, worker_index, num_workers);
        ASSERT_EQ(dest_cores.size(), 4);
        ASSERT_EQ(dest_cores.at(0), CoreCoord(4,0));
        ASSERT_EQ(dest_cores.at(1), CoreCoord(5,0));
        ASSERT_EQ(dest_cores.at(2), CoreCoord(6,0));
        ASSERT_EQ(dest_cores.at(3), CoreCoord(7,0));
    }
TEST(AllGatherUtils, InputTensorShardAddrGenArgGenerator_CtorGenerateDestCoresWidthSharding_4Workers_WorkerIdx2_16Cores)
    {
        CoreRangeSet const& all_shard_cores = CoreRangeSet({CoreRange{CoreCoord(0,0), CoreCoord(7,1)}});
        uint32_t num_workers = 4;
        uint32_t worker_index = 2;
        auto const& dest_cores = ttnn::utils::InputTensorShardAddrGenArgGenerator::ctor_generate_dest_cores(
            all_shard_cores, worker_index, num_workers);
        ASSERT_EQ(dest_cores.size(), 4);
        ASSERT_EQ(dest_cores.at(0), CoreCoord(0,1));
        ASSERT_EQ(dest_cores.at(1), CoreCoord(1,1));
        ASSERT_EQ(dest_cores.at(2), CoreCoord(2,1));
        ASSERT_EQ(dest_cores.at(3), CoreCoord(3,1));
    }
TEST(AllGatherUtils, InputTensorShardAddrGenArgGenerator_CtorGenerateDestCoresWidthSharding_4Workers_WorkerIdx3_16Cores)
    {
        CoreRangeSet const& all_shard_cores = CoreRangeSet({CoreRange{CoreCoord(0,0), CoreCoord(7,1)}});
        uint32_t num_workers = 4;
        uint32_t worker_index = 3;
        auto const& dest_cores = ttnn::utils::InputTensorShardAddrGenArgGenerator::ctor_generate_dest_cores(
            all_shard_cores, worker_index, num_workers);
        ASSERT_EQ(dest_cores.size(), 4);
        ASSERT_EQ(dest_cores.at(0), CoreCoord(4,1));
        ASSERT_EQ(dest_cores.at(1), CoreCoord(5,1));
        ASSERT_EQ(dest_cores.at(2), CoreCoord(6,1));
        ASSERT_EQ(dest_cores.at(3), CoreCoord(7,1));
    }
TEST(AllGatherUtils, InputTensorShardAddrGenArgGenerator_CtorGenerateDestCoresWidthSharding_8Workers_WorkerIdx0_32Cores)
    {
        CoreRangeSet const& all_shard_cores = CoreRangeSet({CoreRange{CoreCoord(0,0), CoreCoord(7,3)}});
        uint32_t num_workers = 8;
        uint32_t worker_index = 0;
        auto const& dest_cores = ttnn::utils::InputTensorShardAddrGenArgGenerator::ctor_generate_dest_cores(
            all_shard_cores, worker_index, num_workers);
        ASSERT_EQ(dest_cores.size(), 4);
        ASSERT_EQ(dest_cores.at(0), CoreCoord(0,0));
        ASSERT_EQ(dest_cores.at(1), CoreCoord(1,0));
        ASSERT_EQ(dest_cores.at(2), CoreCoord(2,0));
        ASSERT_EQ(dest_cores.at(3), CoreCoord(3,0));
    }
TEST(AllGatherUtils, InputTensorShardAddrGenArgGenerator_CtorGenerateDestCoresWidthSharding_8Workers_WorkerIdx1_32Cores)
    {
        CoreRangeSet const& all_shard_cores = CoreRangeSet({CoreRange{CoreCoord(0,0), CoreCoord(7,3)}});
        uint32_t num_workers = 8;
        uint32_t worker_index = 1;
        auto const& dest_cores = ttnn::utils::InputTensorShardAddrGenArgGenerator::ctor_generate_dest_cores(
            all_shard_cores, worker_index, num_workers);
        ASSERT_EQ(dest_cores.size(), 4);
        ASSERT_EQ(dest_cores.at(0), CoreCoord(4,0));
        ASSERT_EQ(dest_cores.at(1), CoreCoord(5,0));
        ASSERT_EQ(dest_cores.at(2), CoreCoord(6,0));
        ASSERT_EQ(dest_cores.at(3), CoreCoord(7,0));
    }
TEST(AllGatherUtils, InputTensorShardAddrGenArgGenerator_CtorGenerateDestCoresWidthSharding_8Workers_WorkerIdx2_32Cores)
    {
        CoreRangeSet const& all_shard_cores = CoreRangeSet({CoreRange{CoreCoord(0,0), CoreCoord(7,3)}});
        uint32_t num_workers = 8;
        uint32_t worker_index = 2;
        auto const& dest_cores = ttnn::utils::InputTensorShardAddrGenArgGenerator::ctor_generate_dest_cores(
            all_shard_cores, worker_index, num_workers);
        ASSERT_EQ(dest_cores.size(), 4);
        ASSERT_EQ(dest_cores.at(0), CoreCoord(0,1));
        ASSERT_EQ(dest_cores.at(1), CoreCoord(1,1));
        ASSERT_EQ(dest_cores.at(2), CoreCoord(2,1));
        ASSERT_EQ(dest_cores.at(3), CoreCoord(3,1));
    }
TEST(AllGatherUtils, InputTensorShardAddrGenArgGenerator_CtorGenerateDestCoresWidthSharding_8Workers_WorkerIdx3_32Cores)
    {
        CoreRangeSet const& all_shard_cores = CoreRangeSet({CoreRange{CoreCoord(0,0), CoreCoord(7,3)}});
        uint32_t num_workers = 8;
        uint32_t worker_index = 3;
        std::cout << "sup" << std::endl;
        auto const& dest_cores = ttnn::utils::InputTensorShardAddrGenArgGenerator::ctor_generate_dest_cores(
            all_shard_cores, worker_index, num_workers);
        std::cout << "hey" << std::endl;
        ASSERT_EQ(dest_cores.size(), 4);
        ASSERT_EQ(dest_cores.at(0), CoreCoord(4,1));
        ASSERT_EQ(dest_cores.at(1), CoreCoord(5,1));
        ASSERT_EQ(dest_cores.at(2), CoreCoord(6,1));
        ASSERT_EQ(dest_cores.at(3), CoreCoord(7,1));
    }



TEST(AllGatherUtilsDevice, AddrGenAdvanceWidthSharded_RingSize8RingIndex0_NumWorkers2WorkerIndex0ShardGridSize2_ClockWise) {
    { // ring_size = 8, 2 workers, worker 0, shard grid size = 2
        bool is_clockwise = true;
        uint16_t curr_core_chunk_index = 0;
        uint16_t curr_worker_index = 0;
        uint16_t contiguous_chunk_count = 1;
        uint16_t current_core_chunks_visited = 0;
        const uint16_t total_chunks_per_core = 8; // shared between all workers
        const uint16_t num_dest_cores = 2;
        const uint16_t intra_core_stride_in_shards = 2; // skip 1
        const uint16_t contiguous_chunks_before_stride = 1;

        ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
            curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
        ASSERT_EQ(curr_core_chunk_index, 6);
        ASSERT_EQ(curr_worker_index, 1);
        ASSERT_EQ(contiguous_chunk_count, 1);

        ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
            curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
        ASSERT_EQ(curr_core_chunk_index, 4);
        ASSERT_EQ(curr_worker_index, 1);
        ASSERT_EQ(contiguous_chunk_count, 1);

        ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
            curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
        ASSERT_EQ(curr_core_chunk_index, 2);
        ASSERT_EQ(curr_worker_index, 1);
        ASSERT_EQ(contiguous_chunk_count, 1);

        ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
            curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
        ASSERT_EQ(curr_core_chunk_index, 0);
        ASSERT_EQ(curr_worker_index, 1);
        ASSERT_EQ(contiguous_chunk_count, 1);

        // Should have moved to the next core
        ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
            curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
        ASSERT_EQ(curr_core_chunk_index, 6);
        ASSERT_EQ(curr_worker_index, 0);
        ASSERT_EQ(contiguous_chunk_count, 1);

        // Should have moved to the next core
        ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
            curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
        ASSERT_EQ(curr_core_chunk_index, 4);
        ASSERT_EQ(curr_worker_index, 0);
        ASSERT_EQ(contiguous_chunk_count, 1);
    }
}

TEST(AllGatherUtilsDevice, AddrGenAdvanceWidthSharded_RingSize8RingIndex0_NumWorkers2WorkerIndex0ShardGridSize2_CounterClockWise) {
    { // ring_size = 8, 2 workers, worker 0, shard grid size = 2
        bool is_clockwise = false;
        uint16_t curr_core_chunk_index = 0;
        uint16_t curr_worker_index = 0;
        uint16_t contiguous_chunk_count = 1;
        uint16_t current_core_chunks_visited = 0;
        const uint16_t total_chunks_per_core = 8; // shared between all workers
        const uint16_t num_dest_cores = 2;
        const uint16_t intra_core_stride_in_shards = 2; // skip 1
        const uint16_t contiguous_chunks_before_stride = 1;

        ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
            curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
        ASSERT_EQ(curr_core_chunk_index, 2);
        ASSERT_EQ(curr_worker_index, 0);
        ASSERT_EQ(contiguous_chunk_count, 1);

        ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
            curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
        ASSERT_EQ(curr_core_chunk_index, 4);
        ASSERT_EQ(curr_worker_index, 0);
        ASSERT_EQ(contiguous_chunk_count, 1);

        ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
            curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
        ASSERT_EQ(curr_core_chunk_index, 6);
        ASSERT_EQ(curr_worker_index, 0);
        ASSERT_EQ(contiguous_chunk_count, 1);

        ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
            curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
        ASSERT_EQ(curr_core_chunk_index, 0);
        ASSERT_EQ(curr_worker_index, 1);
        ASSERT_EQ(contiguous_chunk_count, 1);

        // Should have moved to the next core
        ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
            curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
        ASSERT_EQ(curr_core_chunk_index, 2);
        ASSERT_EQ(curr_worker_index, 1);
        ASSERT_EQ(contiguous_chunk_count, 1);

        // Should have moved to the next core
        ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
            curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
        ASSERT_EQ(curr_core_chunk_index, 4);
        ASSERT_EQ(curr_worker_index, 1);
        ASSERT_EQ(contiguous_chunk_count, 1);
    }
}

TEST(AllGatherUtilsDevice, AddrGenAdvanceWidthSharded_RingSize8RingIndex0_NumWorkers2WorkerIndex1ShardGridSize2_CounterClockWise) {
     // ring_size = 8, 2 workers, worker 1, shard grid size = 2
        bool is_clockwise = false;
        uint16_t curr_core_chunk_index = 1;
        uint16_t curr_worker_index = 0;
        uint16_t contiguous_chunk_count = 1;
        uint16_t current_core_chunks_visited = 0;
        const uint16_t total_chunks_per_core = 8; // shared between all workers
        const uint16_t num_dest_cores = 2;
        const uint16_t intra_core_stride_in_shards = 2; // skip 1
        const uint16_t contiguous_chunks_before_stride = 1;

        ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
            curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
        ASSERT_EQ(curr_core_chunk_index, 3);
        ASSERT_EQ(curr_worker_index, 0);
        ASSERT_EQ(contiguous_chunk_count, 1);

        ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
            curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
        ASSERT_EQ(curr_core_chunk_index, 5);
        ASSERT_EQ(curr_worker_index, 0);
        ASSERT_EQ(contiguous_chunk_count, 1);

        ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
            curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
        ASSERT_EQ(curr_core_chunk_index, 7);
        ASSERT_EQ(curr_worker_index, 0);
        ASSERT_EQ(contiguous_chunk_count, 1);

        ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
            curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
        ASSERT_EQ(curr_core_chunk_index, 1);
        ASSERT_EQ(curr_worker_index, 1);
        ASSERT_EQ(contiguous_chunk_count, 1);

        // Should have moved to the next core
        ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
            curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
        ASSERT_EQ(curr_core_chunk_index, 3);
        ASSERT_EQ(curr_worker_index, 1);
        ASSERT_EQ(contiguous_chunk_count, 1);

        // Should have moved to the next core
        ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
            curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
        ASSERT_EQ(curr_core_chunk_index, 5);
        ASSERT_EQ(curr_worker_index, 1);
        ASSERT_EQ(contiguous_chunk_count, 1);
}


TEST(AllGatherUtilsDevice, AddrGenAdvanceWidthSharded_RingSize8RingIndex1_NumWorkers2WorkerIndex0ShardGridSize2_CounterClockWise) {
        // ring_size = 8, 2 workers, worker 0, shard grid size = 2
        bool is_clockwise = false;
        uint16_t curr_core_chunk_index = 0;
        uint16_t curr_worker_index = 1;
        uint16_t contiguous_chunk_count = 1;
        const uint16_t total_chunks_per_core = 8; // shared between all workers
        const uint16_t num_dest_cores = 2;
        const uint16_t intra_core_stride_in_shards = 2; // skip 1
        const uint16_t contiguous_chunks_before_stride = 1;

        ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
            curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
        ASSERT_EQ(curr_core_chunk_index, 2);
        ASSERT_EQ(curr_worker_index, 1);
        ASSERT_EQ(contiguous_chunk_count, 1);
        ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
            curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
        ASSERT_EQ(curr_core_chunk_index, 4);
        ASSERT_EQ(curr_worker_index, 1);
        ASSERT_EQ(contiguous_chunk_count, 1);

        ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
            curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
        ASSERT_EQ(curr_core_chunk_index, 6);
        ASSERT_EQ(curr_worker_index, 1);
        ASSERT_EQ(contiguous_chunk_count, 1);

        ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
            curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
        ASSERT_EQ(curr_core_chunk_index, 0);
        ASSERT_EQ(curr_worker_index, 0);
        ASSERT_EQ(contiguous_chunk_count, 1);

        // Should have moved to the next core
        ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
            curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
        ASSERT_EQ(curr_core_chunk_index, 2);
        ASSERT_EQ(curr_worker_index, 0);
        ASSERT_EQ(contiguous_chunk_count, 1);

        // Should have moved to the next core
        ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
            curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
        ASSERT_EQ(curr_core_chunk_index, 4);
        ASSERT_EQ(curr_worker_index, 0);
        ASSERT_EQ(contiguous_chunk_count, 1);
}

TEST(AllGatherUtilsDevice, AddrGenAdvanceWidthSharded_RingSize8RingIndex1_NumWorkers2WorkerIndex1ShardGridSize2_CounterClockWise) {
    // ring_size = 8, 2 workers, worker 1, shard grid size = 2
    bool is_clockwise = false;
    uint16_t curr_core_chunk_index = 1;
    uint16_t curr_worker_index = 1;
    uint16_t contiguous_chunk_count = 1;
    uint16_t current_core_chunks_visited = 0;
    const uint16_t total_chunks_per_core = 8; // shared between all workers
    const uint16_t num_dest_cores = 2;
    const uint16_t intra_core_stride_in_shards = 2; // skip 1
    const uint16_t contiguous_chunks_before_stride = 1;

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 3);
    ASSERT_EQ(curr_worker_index, 1);
    ASSERT_EQ(contiguous_chunk_count, 1);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 5);
    ASSERT_EQ(curr_worker_index, 1);
    ASSERT_EQ(contiguous_chunk_count, 1);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 7);
    ASSERT_EQ(curr_worker_index, 1);
    ASSERT_EQ(contiguous_chunk_count, 1);
    // ASSERT_EQ(current_core_chunks_visited, 3);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 1);
    ASSERT_EQ(curr_worker_index, 0);
    ASSERT_EQ(contiguous_chunk_count, 1);

    // Should have moved to the next core
    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 3);
    ASSERT_EQ(curr_worker_index, 0);
    ASSERT_EQ(contiguous_chunk_count, 1);

    // Should have moved to the next core
    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 5);
    ASSERT_EQ(curr_worker_index, 0);
    ASSERT_EQ(contiguous_chunk_count, 1);

}

TEST(AllGatherUtilsDevice, AddrGenAdvanceWidthSharded_RingSize8RingIndex0_NumWorkers4WorkerIndex0ShardGridSize16_CounterClockWise) {
    bool is_clockwise = false;
    uint16_t curr_core_chunk_index = 0;
    uint16_t curr_worker_index = 0;
    uint16_t contiguous_chunk_count = 1;
    uint16_t current_core_chunks_visited = 0;
    const uint16_t total_chunks_per_core = 8; // shared between all workers
    const uint16_t num_dest_cores = 16;
    const uint16_t intra_core_stride_in_shards = 5; // skip 4
    const uint16_t contiguous_chunks_before_stride = 4;

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 1);
    ASSERT_EQ(curr_worker_index, 0);
    ASSERT_EQ(contiguous_chunk_count, 2);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 2);
    ASSERT_EQ(curr_worker_index, 0);
    ASSERT_EQ(contiguous_chunk_count, 3);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 3);
    ASSERT_EQ(curr_worker_index, 0);
    ASSERT_EQ(contiguous_chunk_count, 4);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 0);
    ASSERT_EQ(curr_worker_index, 1);
    ASSERT_EQ(contiguous_chunk_count, 1);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 1);
    ASSERT_EQ(curr_worker_index, 1);
    ASSERT_EQ(contiguous_chunk_count, 2);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 2);
    ASSERT_EQ(curr_worker_index, 1);
    ASSERT_EQ(contiguous_chunk_count, 3);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 3);
    ASSERT_EQ(curr_worker_index, 1);
    ASSERT_EQ(contiguous_chunk_count, 4);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 0);
    ASSERT_EQ(curr_worker_index, 2);
    ASSERT_EQ(contiguous_chunk_count, 1);
}


TEST(AllGatherUtilsDevice, AddrGenAdvanceWidthSharded_RingSize8RingIndex1_NumWorkers4WorkerIndex3ShardGridSize16_CounterClockWise) {
    bool is_clockwise = false;
    uint16_t curr_core_chunk_index = 4;
    uint16_t curr_worker_index = 0;
    uint16_t contiguous_chunk_count = 1;
    uint16_t current_core_chunks_visited = 0;
    const uint16_t total_chunks_per_core = 8; // shared between all workers
    const uint16_t num_dest_cores = 16;
    const uint16_t intra_core_stride_in_shards = 5; // skip 1
    const uint16_t contiguous_chunks_before_stride = 4;

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 5);
    ASSERT_EQ(curr_worker_index, 0);
    ASSERT_EQ(contiguous_chunk_count, 2);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 6);
    ASSERT_EQ(curr_worker_index, 0);
    ASSERT_EQ(contiguous_chunk_count, 3);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 7);
    ASSERT_EQ(curr_worker_index, 0);
    ASSERT_EQ(contiguous_chunk_count, 4);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 4);
    ASSERT_EQ(curr_worker_index, 1);
    ASSERT_EQ(contiguous_chunk_count, 1);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 5);
    ASSERT_EQ(curr_worker_index, 1);
    ASSERT_EQ(contiguous_chunk_count, 2);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 6);
    ASSERT_EQ(curr_worker_index, 1);
    ASSERT_EQ(contiguous_chunk_count, 3);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 7);
    ASSERT_EQ(curr_worker_index, 1);
    ASSERT_EQ(contiguous_chunk_count, 4);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 4);
    ASSERT_EQ(curr_worker_index, 2);
    ASSERT_EQ(contiguous_chunk_count, 1);
}

TEST(AllGatherUtilsDevice, AddrGenAdvanceWidthSharded_RingSize8RingIndex0_NumWorkers8WorkerIndex0ShardGridSize32_CounterClockWise) {
    bool is_clockwise = false;
    uint16_t curr_core_chunk_index = 0;
    uint16_t curr_worker_index = 0;
    uint16_t contiguous_chunk_count = 1;
    uint16_t current_core_chunks_visited = 0;
    const uint16_t total_chunks_per_core = 8; // shared between all workers
    const uint16_t num_dest_cores = 8;
    const uint16_t intra_core_stride_in_shards = 5; // skip 4
    const uint16_t contiguous_chunks_before_stride = 4;

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 1);
    ASSERT_EQ(curr_worker_index, 0);
    ASSERT_EQ(contiguous_chunk_count, 2);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 2);
    ASSERT_EQ(curr_worker_index, 0);
    ASSERT_EQ(contiguous_chunk_count, 3);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 3);
    ASSERT_EQ(curr_worker_index, 0);
    ASSERT_EQ(contiguous_chunk_count, 4);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 0);
    ASSERT_EQ(curr_worker_index, 1);
    ASSERT_EQ(contiguous_chunk_count, 1);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 1);
    ASSERT_EQ(curr_worker_index, 1);
    ASSERT_EQ(contiguous_chunk_count, 2);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 2);
    ASSERT_EQ(curr_worker_index, 1);
    ASSERT_EQ(contiguous_chunk_count, 3);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 3);
    ASSERT_EQ(curr_worker_index, 1);
    ASSERT_EQ(contiguous_chunk_count, 4);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 0);
    ASSERT_EQ(curr_worker_index, 2);
    ASSERT_EQ(contiguous_chunk_count, 1);

    // check for wraparound
    for (uint32_t i = 0; i < num_dest_cores; i++) {
        for (uint32_t c = 0; c < contiguous_chunks_before_stride; c++) {
            ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
                curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
        }
    }
    ASSERT_EQ(curr_core_chunk_index, 0);
    ASSERT_EQ(curr_worker_index, 2);
    ASSERT_EQ(contiguous_chunk_count, 1);
}


TEST(AllGatherUtilsDevice, AddrGenAdvanceWidthSharded_RingSize8RingIndex0_NumWorkers8WorkerIndex1ShardGridSize32_CounterClockWise) {
    bool is_clockwise = false;
    uint16_t curr_core_chunk_index = 4;
    uint16_t curr_worker_index = 0;
    uint16_t contiguous_chunk_count = 1;
    uint16_t current_core_chunks_visited = 0;
    const uint16_t total_chunks_per_core = 8; // shared between all workers
    const uint16_t num_dest_cores = 8;
    const uint16_t intra_core_stride_in_shards = 5; // skip 4
    const uint16_t contiguous_chunks_before_stride = 4;

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index, contiguous_chunk_count, total_chunks_per_core, num_dest_cores, intra_core_stride_in_shards, contiguous_chunks_before_stride, is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 5);
    ASSERT_EQ(curr_worker_index, 0);
    ASSERT_EQ(contiguous_chunk_count, 2);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 6);
    ASSERT_EQ(curr_worker_index, 0);
    ASSERT_EQ(contiguous_chunk_count, 3);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 7);
    ASSERT_EQ(curr_worker_index, 0);
    ASSERT_EQ(contiguous_chunk_count, 4);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 4);
    ASSERT_EQ(curr_worker_index, 1);
    ASSERT_EQ(contiguous_chunk_count, 1);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 5);
    ASSERT_EQ(curr_worker_index, 1);
    ASSERT_EQ(contiguous_chunk_count, 2);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 6);
    ASSERT_EQ(curr_worker_index, 1);
    ASSERT_EQ(contiguous_chunk_count, 3);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 7);
    ASSERT_EQ(curr_worker_index, 1);
    ASSERT_EQ(contiguous_chunk_count, 4);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 4);
    ASSERT_EQ(curr_worker_index, 2);
    ASSERT_EQ(contiguous_chunk_count, 1);

    // check for wraparound
    for (uint32_t i = 0; i < num_dest_cores; i++) {
        for (uint32_t c = 0; c < contiguous_chunks_before_stride; c++) {
            ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
                curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
        }
    }
    ASSERT_EQ(curr_core_chunk_index, 4);
    ASSERT_EQ(curr_worker_index, 2);
    ASSERT_EQ(contiguous_chunk_count, 1);
}

TEST(AllGatherUtilsDevice, AddrGenAdvanceWidthSharded_RingSize8RingIndex0_NumWorkers8WorkerIndex2ShardGridSize32_CounterClockWise) {
    bool is_clockwise = false;
    uint16_t curr_core_chunk_index = 0;
    uint16_t curr_worker_index = 0;
    uint16_t contiguous_chunk_count = 1;
    uint16_t current_core_chunks_visited = 0;
    const uint16_t total_chunks_per_core = 8; // shared between all workers
    const uint16_t num_dest_cores = 8;
    const uint16_t intra_core_stride_in_shards = 5; // skip 4
    const uint16_t contiguous_chunks_before_stride = 4;

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index, curr_worker_index, contiguous_chunk_count, total_chunks_per_core, num_dest_cores, intra_core_stride_in_shards, contiguous_chunks_before_stride, is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 1);
    ASSERT_EQ(curr_worker_index, 0);
    ASSERT_EQ(contiguous_chunk_count, 2);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 2);
    ASSERT_EQ(curr_worker_index, 0);
    ASSERT_EQ(contiguous_chunk_count, 3);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 3);
    ASSERT_EQ(curr_worker_index, 0);
    ASSERT_EQ(contiguous_chunk_count, 4);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 0);
    ASSERT_EQ(curr_worker_index, 1);
    ASSERT_EQ(contiguous_chunk_count, 1);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 1);
    ASSERT_EQ(curr_worker_index, 1);
    ASSERT_EQ(contiguous_chunk_count, 2);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 2);
    ASSERT_EQ(curr_worker_index, 1);
    ASSERT_EQ(contiguous_chunk_count, 3);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 3);
    ASSERT_EQ(curr_worker_index, 1);
    ASSERT_EQ(contiguous_chunk_count, 4);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 0);
    ASSERT_EQ(curr_worker_index, 2);
    ASSERT_EQ(contiguous_chunk_count, 1);

    // check for wraparound
    for (uint32_t i = 0; i < num_dest_cores; i++) {
        for (uint32_t c = 0; c < contiguous_chunks_before_stride; c++) {
            ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
                curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
        }
    }
    ASSERT_EQ(curr_core_chunk_index, 0);
    ASSERT_EQ(curr_worker_index, 2);
    ASSERT_EQ(contiguous_chunk_count, 1);
}


TEST(AllGatherUtilsDevice, AddrGenAdvanceWidthSharded_RingSize8RingIndex0_NumWorkers8WorkerIndex3ShardGridSize32_CounterClockWise) {
    bool is_clockwise = false;
    uint16_t curr_core_chunk_index = 4;
    uint16_t curr_worker_index = 0;
    uint16_t contiguous_chunk_count = 1;
    uint16_t current_core_chunks_visited = 0;
    const uint16_t total_chunks_per_core = 8; // shared between all workers
    const uint16_t num_dest_cores = 8;
    const uint16_t intra_core_stride_in_shards = 5; // skip 4
    const uint16_t contiguous_chunks_before_stride = 4;

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 5);
    ASSERT_EQ(curr_worker_index, 0);
    ASSERT_EQ(contiguous_chunk_count, 2);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 6);
    ASSERT_EQ(curr_worker_index, 0);
    ASSERT_EQ(contiguous_chunk_count, 3);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 7);
    ASSERT_EQ(curr_worker_index, 0);
    ASSERT_EQ(contiguous_chunk_count, 4);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 4);
    ASSERT_EQ(curr_worker_index, 1);
    ASSERT_EQ(contiguous_chunk_count, 1);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 5);
    ASSERT_EQ(curr_worker_index, 1);
    ASSERT_EQ(contiguous_chunk_count, 2);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 6);
    ASSERT_EQ(curr_worker_index, 1);
    ASSERT_EQ(contiguous_chunk_count, 3);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 7);
    ASSERT_EQ(curr_worker_index, 1);
    ASSERT_EQ(contiguous_chunk_count, 4);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 4);
    ASSERT_EQ(curr_worker_index, 2);
    ASSERT_EQ(contiguous_chunk_count, 1);

    // check for wraparound
    for (uint32_t i = 0; i < num_dest_cores; i++) {
        for (uint32_t c = 0; c < contiguous_chunks_before_stride; c++) {
            ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
                curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
        }
    }
    ASSERT_EQ(curr_core_chunk_index, 4);
    ASSERT_EQ(curr_worker_index, 2);
    ASSERT_EQ(contiguous_chunk_count, 1);
}


TEST(AllGatherUtilsDevice, AddrGenAdvanceWidthSharded_RingSize8RingIndex1_NumWorkers8WorkerIndex2ShardGridSize32_CounterClockWise) {
    bool is_clockwise = false;
    uint16_t curr_core_chunk_index = 0;
    uint16_t curr_worker_index = 1;
    uint16_t contiguous_chunk_count = 1;
    uint16_t current_core_chunks_visited = 0;
    const uint16_t total_chunks_per_core = 8; // shared between all workers
    const uint16_t num_dest_cores = 8;
    const uint16_t intra_core_stride_in_shards = 5; // skip 4
    const uint16_t contiguous_chunks_before_stride = 4;

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 1);
    ASSERT_EQ(curr_worker_index, 1);
    ASSERT_EQ(contiguous_chunk_count, 2);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 2);
    ASSERT_EQ(curr_worker_index, 1);
    ASSERT_EQ(contiguous_chunk_count, 3);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 3);
    ASSERT_EQ(curr_worker_index, 1);
    ASSERT_EQ(contiguous_chunk_count, 4);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 0);
    ASSERT_EQ(curr_worker_index, 2);
    ASSERT_EQ(contiguous_chunk_count, 1);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 1);
    ASSERT_EQ(curr_worker_index, 2);
    ASSERT_EQ(contiguous_chunk_count, 2);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 2);
    ASSERT_EQ(curr_worker_index, 2);
    ASSERT_EQ(contiguous_chunk_count, 3);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 3);
    ASSERT_EQ(curr_worker_index, 2);
    ASSERT_EQ(contiguous_chunk_count, 4);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 0);
    ASSERT_EQ(curr_worker_index, 3);
    ASSERT_EQ(contiguous_chunk_count, 1);

    // Check for wraparound
    for (uint32_t i = 0; i < num_dest_cores; i++) {
        for (uint32_t c = 0; c < contiguous_chunks_before_stride; c++) {
            ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
                curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
        }
    }
    ASSERT_EQ(curr_core_chunk_index, 0);
    ASSERT_EQ(curr_worker_index, 3);
    ASSERT_EQ(contiguous_chunk_count, 1);
}

TEST(AllGatherUtilsDevice, AddrGenAdvanceWidthSharded_RingSize8RingIndex0_NumWorkers8WorkerIndex0ShardGridSize32_ClockWise) {
    bool is_clockwise = true;
    uint16_t curr_core_chunk_index = 0;
    uint16_t curr_worker_index = 0;
    uint16_t contiguous_chunk_count = 1;
    uint16_t current_core_chunks_visited = 0;
    const uint16_t total_chunks_per_core = 8; // shared between all workers
    const uint16_t num_dest_cores = 8;
    const uint16_t intra_core_stride_in_shards = 5; // skip 4
    const uint16_t contiguous_chunks_before_stride = 4;

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 1);
    ASSERT_EQ(curr_worker_index, 0);
    ASSERT_EQ(contiguous_chunk_count, 2);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 2);
    ASSERT_EQ(curr_worker_index, 0);
    ASSERT_EQ(contiguous_chunk_count, 3);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 3);
    ASSERT_EQ(curr_worker_index, 0);
    ASSERT_EQ(contiguous_chunk_count, 4);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 0);
    ASSERT_EQ(curr_worker_index, 7);
    ASSERT_EQ(contiguous_chunk_count, 1);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 1);
    ASSERT_EQ(curr_worker_index, 7);
    ASSERT_EQ(contiguous_chunk_count, 2);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 2);
    ASSERT_EQ(curr_worker_index, 7);
    ASSERT_EQ(contiguous_chunk_count, 3);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 3);
    ASSERT_EQ(curr_worker_index, 7);
    ASSERT_EQ(contiguous_chunk_count, 4);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 0);
    ASSERT_EQ(curr_worker_index, 6);
    ASSERT_EQ(contiguous_chunk_count, 1);

    // check for wraparound
    for (uint32_t i = 0; i < num_dest_cores; i++) {
        for (uint32_t c = 0; c < contiguous_chunks_before_stride; c++) {
            ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
                curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
        }
    }
    ASSERT_EQ(curr_core_chunk_index, 0);
    ASSERT_EQ(curr_worker_index, 6);
    ASSERT_EQ(contiguous_chunk_count, 1);
}

TEST(AllGatherUtilsDevice, AddrGenAdvanceWidthSharded_RingSize8RingIndex0_NumWorkers8WorkerIndex1ShardGridSize32_ClockWise) {
    bool is_clockwise = true;
    uint16_t curr_core_chunk_index = 4;
    uint16_t curr_worker_index = 0;
    uint16_t contiguous_chunk_count = 1;
    uint16_t current_core_chunks_visited = 0;
    const uint16_t total_chunks_per_core = 8; // shared between all workers
    const uint16_t num_dest_cores = 8;
    const uint16_t intra_core_stride_in_shards = 5; // skip 4
    const uint16_t contiguous_chunks_before_stride = 4;

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 5);
    ASSERT_EQ(curr_worker_index, 0);
    ASSERT_EQ(contiguous_chunk_count, 2);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 6);
    ASSERT_EQ(curr_worker_index, 0);
    ASSERT_EQ(contiguous_chunk_count, 3);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 7);
    ASSERT_EQ(curr_worker_index, 0);
    ASSERT_EQ(contiguous_chunk_count, 4);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 4);
    ASSERT_EQ(curr_worker_index, 7);
    ASSERT_EQ(contiguous_chunk_count, 1);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 5);
    ASSERT_EQ(curr_worker_index, 7);
    ASSERT_EQ(contiguous_chunk_count, 2);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 6);
    ASSERT_EQ(curr_worker_index, 7);
    ASSERT_EQ(contiguous_chunk_count, 3);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 7);
    ASSERT_EQ(curr_worker_index, 7);
    ASSERT_EQ(contiguous_chunk_count, 4);

    ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
        curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
    ASSERT_EQ(curr_core_chunk_index, 4);
    ASSERT_EQ(curr_worker_index, 6);
    ASSERT_EQ(contiguous_chunk_count, 1);

    // check for wraparound
    for (uint32_t i = 0; i < num_dest_cores; i++) {
        for (uint32_t c = 0; c < contiguous_chunks_before_stride; c++) {
            ttnn::utils::ccl::all_gather::addr_gen_advance_width_sharded(
                curr_core_chunk_index,curr_worker_index,contiguous_chunk_count,total_chunks_per_core,num_dest_cores,intra_core_stride_in_shards,contiguous_chunks_before_stride,is_clockwise);
        }
    }
    ASSERT_EQ(curr_core_chunk_index, 4);
    ASSERT_EQ(curr_worker_index, 6);
    ASSERT_EQ(contiguous_chunk_count, 1);
}
