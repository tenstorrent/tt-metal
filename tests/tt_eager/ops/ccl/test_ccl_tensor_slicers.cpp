// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"

static constexpr std::array<noc_grid_index_t, 8> worker_to_routing_x_wormhole = {
    1,2,3,4,6,7,8,9
};

static constexpr std::array<noc_grid_index_t, 10> worker_to_routing_y_wormhole = {
    1,2,3,4,5,7,8,9,10,11
};

void run_width_sharded_tensor_slice_indexer_get_page_location_test(
    std::size_t pages_per_shard,

    std::size_t shard_grid_height,
    std::size_t shard_grid_width,

    std::size_t worker_shard_cores_start_y,
    std::size_t worker_shard_cores_start_x,

    bool is_shard_grid_transposed) {

    const std::size_t global_num_pages = pages_per_shard * shard_grid_width * shard_grid_height;

    auto addrgen = WidthShardedAddressGenerator<UnharvestedWormholeWorkerToNocLookup>(
        UnharvestedWormholeWorkerToNocLookup(),
        device_shard_spec_t{
            shard_grid_height,
            shard_grid_width,
            worker_shard_cores_start_y,
            worker_shard_cores_start_x,
            pages_per_shard,
            is_shard_grid_transposed},
        1024,
        0x0);

    std::size_t page_id = 0;

    if (!is_shard_grid_transposed) {
        for (std::size_t y_logical = worker_shard_cores_start_y; y_logical < worker_shard_cores_start_y + shard_grid_height; y_logical++) {
            for (std::size_t x_logical = worker_shard_cores_start_x; x_logical < worker_shard_cores_start_x + shard_grid_width; x_logical++) {
                for (std::size_t p = 0; p < pages_per_shard; p++) {
                    auto const& result = addrgen.get_page_location(page_id);
                    ASSERT_EQ(result.core_location.noc_x, worker_to_routing_x_wormhole.at(x_logical));
                    ASSERT_EQ(result.core_location.noc_y, worker_to_routing_y_wormhole.at(y_logical));
                    ASSERT_EQ(result.page_offset, p);
                    page_id++;
                }
            }
        }
    } else {
        for (std::size_t x_logical = worker_shard_cores_start_x; x_logical < worker_shard_cores_start_x + shard_grid_width; x_logical++) {
            for (std::size_t y_logical = worker_shard_cores_start_y; y_logical < worker_shard_cores_start_y + shard_grid_height; y_logical++) {
                for (std::size_t p = 0; p < pages_per_shard; p++) {
                    auto const& result = addrgen.get_page_location(page_id);
                    ASSERT_EQ(result.core_location.noc_x, worker_to_routing_x_wormhole.at(x_logical));
                    ASSERT_EQ(result.core_location.noc_y, worker_to_routing_y_wormhole.at(y_logical));
                    ASSERT_EQ(result.page_offset, p);
                    page_id++;
                }
            }
        }

    }
}



TEST(CclWidthShardedTensorSliceIndexer_Wormhole, 1_PagePerShard_1x8_ShardGrid_LogicalCores_y0_x0_to_y0_x8_no_transpose_nocxy_from_page_id) {
    static constexpr std::size_t pages_per_shard = 1;

    static constexpr std::size_t shard_grid_height = 1;
    static constexpr std::size_t shard_grid_width = 8;

    static constexpr std::size_t worker_shard_cores_start_y = 0;
    static constexpr std::size_t worker_shard_cores_start_x = 0;

    bool is_shard_grid_transposed = false;

    run_width_sharded_tensor_slice_indexer_get_page_location_test(
        pages_per_shard,

        shard_grid_height,
        shard_grid_width,

        worker_shard_cores_start_y,
        worker_shard_cores_start_x,

        is_shard_grid_transposed);
}

TEST(CclWidthShardedTensorSliceIndexer_Wormhole, SweepWormhole) {
    std::size_t max_worker_rows = 10;
    std::size_t max_worker_cols = 8;

    for (auto pages_per_shard : {1,2,3,4,5,7,8}) {
    for (auto shard_grid_offset_logical_y : {0,1,2,3,4,5,6,7,8,9}) {
    for (auto shard_grid_offset_logical_x : {0,1,2,3,4,5,6,7}) {
    for (std::size_t shard_grid_height = 1; shard_grid_height < (max_worker_rows - shard_grid_offset_logical_y); shard_grid_height++) {
    for (std::size_t shard_grid_width = 1; shard_grid_width < (max_worker_cols - shard_grid_offset_logical_x); shard_grid_width++) {
    for (bool transpose_shard_grid : {false, true}) {
        run_width_sharded_tensor_slice_indexer_get_page_location_test(
            pages_per_shard,
            shard_grid_height,
            shard_grid_width,
            shard_grid_offset_logical_y,
            shard_grid_offset_logical_x,
            transpose_shard_grid);
    }}}}}}
}

// class CclWidthShardedTensorSliceIndexer_Wormhole : public ::testing::TestWithParam<std::tuple<int, int, int, int, int, bool>> {
// };

// TEST_P(CclWidthShardedTensorSliceIndexer_Wormhole, SweepWormhole) {
//     auto params = GetParam();
//     int pages_per_shard = std::get<0>(params);
//     int shard_grid_height = std::get<1>(params);
//     int shard_grid_width = std::get<2>(params);
//     int shard_grid_offset_logical_y = std::get<3>(params);
//     int shard_grid_offset_logical_x = std::get<4>(params);
//     bool transpose_shard_grid = std::get<5>(params);

//     if (shard_grid_offset_logical_y + shard_grid_height >= 10) {
//         GTEST_SKIP();
//     }
//     if (shard_grid_offset_logical_x + shard_grid_width >= 8) {
//         GTEST_SKIP();
//     }

//     run_width_sharded_tensor_slice_indexer_get_page_location_test(
//         pages_per_shard,
//         shard_grid_height,
//         shard_grid_width,
//         shard_grid_offset_logical_y,
//         shard_grid_offset_logical_x,
//         transpose_shard_grid);
// }

// struct TestParamName {
//     template <class ParamType>
//     std::string operator()(const ::testing::TestParamInfo<ParamType>& info) const {
//         auto params = info.param;
//         std::ostringstream name;
//         name << "pages_per_shard_" << std::get<0>(params)
//              << "_shard_grid_height_" << std::get<1>(params)
//              << "_shard_grid_width_" << std::get<2>(params)
//              << "_shard_grid_offset_logical_y_" << std::get<3>(params)
//              << "_shard_grid_offset_logical_x_" << std::get<4>(params)
//              << "_transpose_shard_grid_" << (std::get<5>(params) ? "true" : "false");
//         return name.str();
//     }
// };

// INSTANTIATE_TEST_SUITE_P(SweepWormholeTest, CclWidthShardedTensorSliceIndexer_Wormhole,
//     ::testing::Combine(
//         ::testing::Values(1, 2, 3, 4, 5),  // pages_per_shard
//         ::testing::Range(0, 10),  // shard_grid_offset_logical_y
//         ::testing::Range(0, 8),  // shard_grid_offset_logical_x
//         ::testing::Range(1, 10),  // shard_grid_height
//         ::testing::Range(1, 8),  // shard_grid_width
//         ::testing::Bool()  // transpose_shard_grid
//     ),
//     TestParamName()
// );
