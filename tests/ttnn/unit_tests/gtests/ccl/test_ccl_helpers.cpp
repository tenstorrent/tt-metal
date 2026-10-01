// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "tests/tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "ttnn/config.hpp"
#include "ttnn/device_operation_detail.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/common/host/ccl_topology_utils.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/shared_with_host/snake_ring.hpp"
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_rank_mapping.hpp"
#include <umd/device/types/xy_pair.hpp>
#include <umd/device/types/arch.hpp>
#include "common/tt_backend_api_types.hpp"

namespace {

void check_snake_ring_bijection(uint32_t rows, uint32_t cols, ttnn::ccl::snake_ring::Orientation orientation) {
    const uint32_t ring_size = rows * cols;
    std::vector<bool> seen_tensor_ranks(ring_size, false);
    for (uint32_t transport_rank = 0; transport_rank < ring_size; ++transport_rank) {
        const uint32_t row = ttnn::ccl::snake_ring::coordinate_row(transport_rank, rows, cols, orientation);
        const uint32_t col = ttnn::ccl::snake_ring::coordinate_col(transport_rank, rows, cols, orientation);
        ASSERT_LT(row, rows);
        ASSERT_LT(col, cols);
        EXPECT_EQ(ttnn::ccl::snake_ring::index_from_coordinate(row, col, rows, cols, orientation), transport_rank);

        const uint32_t tensor_rank = ttnn::ccl::snake_ring::row_major_index(transport_rank, rows, cols, orientation);
        EXPECT_EQ(tensor_rank, row * cols + col);
        ASSERT_LT(tensor_rank, ring_size);
        EXPECT_FALSE(seen_tensor_ranks[tensor_rank]);
        seen_tensor_ranks[tensor_rank] = true;

        const uint32_t lane_count = orientation == ttnn::ccl::snake_ring::Orientation::Row ? rows : cols;
        if (transport_rank + 1 < ring_size || lane_count % 2 == 0) {
            const uint32_t next_rank = (transport_rank + 1) % ring_size;
            const uint32_t next_row = ttnn::ccl::snake_ring::coordinate_row(next_rank, rows, cols, orientation);
            const uint32_t next_col = ttnn::ccl::snake_ring::coordinate_col(next_rank, rows, cols, orientation);
            const uint32_t raw_row_delta = row > next_row ? row - next_row : next_row - row;
            const uint32_t raw_col_delta = col > next_col ? col - next_col : next_col - col;
            const uint32_t cyclic_row_delta = std::min(raw_row_delta, rows - raw_row_delta);
            const uint32_t cyclic_col_delta = std::min(raw_col_delta, cols - raw_col_delta);
            EXPECT_EQ(cyclic_row_delta + cyclic_col_delta, 1u);
        }
    }
    EXPECT_TRUE(std::all_of(seen_tensor_ranks.begin(), seen_tensor_ranks.end(), [](bool seen) { return seen; }));
}

// Open path: same order and same neighbor steps, but the last rank need not reach the first.
void check_snake_ring_open_path(uint32_t rows, uint32_t cols, ttnn::ccl::snake_ring::Orientation orientation) {
    const uint32_t ring_size = rows * cols;
    std::vector<bool> seen_tensor_ranks(ring_size, false);
    for (uint32_t transport_rank = 0; transport_rank < ring_size; ++transport_rank) {
        const uint32_t row = ttnn::ccl::snake_ring::coordinate_row(transport_rank, rows, cols, orientation);
        const uint32_t col = ttnn::ccl::snake_ring::coordinate_col(transport_rank, rows, cols, orientation);
        ASSERT_LT(row, rows);
        ASSERT_LT(col, cols);
        EXPECT_EQ(ttnn::ccl::snake_ring::index_from_coordinate(row, col, rows, cols, orientation), transport_rank);

        const uint32_t tensor_rank = ttnn::ccl::snake_ring::row_major_index(transport_rank, rows, cols, orientation);
        EXPECT_EQ(tensor_rank, row * cols + col);
        ASSERT_LT(tensor_rank, ring_size);
        EXPECT_FALSE(seen_tensor_ranks[tensor_rank]);
        seen_tensor_ranks[tensor_rank] = true;

        if (transport_rank + 1 < ring_size) {
            const uint32_t next_rank = transport_rank + 1;
            const uint32_t next_row = ttnn::ccl::snake_ring::coordinate_row(next_rank, rows, cols, orientation);
            const uint32_t next_col = ttnn::ccl::snake_ring::coordinate_col(next_rank, rows, cols, orientation);
            const uint32_t row_delta = row > next_row ? row - next_row : next_row - row;
            const uint32_t col_delta = col > next_col ? col - next_col : next_col - col;
            EXPECT_EQ(row_delta + col_delta, 1u)
                << "orientation " << static_cast<uint32_t>(orientation) << " on " << rows << "x" << cols << " step "
                << transport_rank << " -> " << next_rank << " is not a nearest neighbor";
        }
    }
    EXPECT_TRUE(std::all_of(seen_tensor_ranks.begin(), seen_tensor_ranks.end(), [](bool seen) { return seen; }));
}

}  // namespace

TEST(CclHelpers, SnakeRingMappingsAreBijectionsWithRowMajorTensorRanks) {
    constexpr std::array<std::array<uint32_t, 2>, 4> mesh_shapes{{{2, 2}, {2, 4}, {8, 4}, {3, 2}}};
    for (const auto& shape : mesh_shapes) {
        check_snake_ring_bijection(shape[0], shape[1], ttnn::ccl::snake_ring::Orientation::Row);
        check_snake_ring_bijection(shape[0], shape[1], ttnn::ccl::snake_ring::Orientation::Column);
    }
}

TEST(CclHelpers, BoustrophedonOpenPathCoversEveryMeshIncludingThoseWithNoCycle) {
    // None of these shapes is guaranteed a cycle, but every one of them has a path.
    // 8x4 included because that is what a plain (non-torus) 2D fabric falls back to.
    constexpr std::array<std::array<uint32_t, 2>, 7> mesh_shapes{
        {{1, 8}, {8, 1}, {1, 2}, {3, 3}, {5, 5}, {3, 5}, {8, 4}}};
    for (const auto& shape : mesh_shapes) {
        check_snake_ring_open_path(shape[0], shape[1], ttnn::ccl::snake_ring::Orientation::Row);
        check_snake_ring_open_path(shape[0], shape[1], ttnn::ccl::snake_ring::Orientation::Column);
    }
}

TEST(CclHelpers, OpenPathOnAOneWideMeshIsExactlyTheAxisLine) {
    // A 1xN full-mesh route must match the axis order device for device, or the gather is reordered.
    for (uint32_t transport_rank = 0; transport_rank < 8; ++transport_rank) {
        EXPECT_EQ(
            ttnn::ccl::snake_ring::coordinate_row(transport_rank, 1, 8, ttnn::ccl::snake_ring::Orientation::Row), 0u);
        EXPECT_EQ(
            ttnn::ccl::snake_ring::coordinate_col(transport_rank, 1, 8, ttnn::ccl::snake_ring::Orientation::Row),
            transport_rank);
        EXPECT_EQ(
            ttnn::ccl::snake_ring::row_major_index(transport_rank, 1, 8, ttnn::ccl::snake_ring::Orientation::Row),
            transport_rank);
        EXPECT_EQ(
            ttnn::ccl::snake_ring::row_major_index(transport_rank, 8, 1, ttnn::ccl::snake_ring::Orientation::Row),
            transport_rank);
    }
}

TEST(CclHelpers, RingAttentionRankMappingKeepsAxisIdentityAndPlacesFullMeshInRowMajorOrder) {
    for (uint32_t transport_rank = 0; transport_rank < 6; ++transport_rank) {
        EXPECT_EQ(
            (ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<false>(
                transport_rank, 0, 0, ttnn::ccl::snake_ring::Orientation::Column)),
            transport_rank);
    }

    constexpr std::array<uint32_t, 4> row_snake_2x2{0, 1, 3, 2};
    for (uint32_t transport_rank = 0; transport_rank < row_snake_2x2.size(); ++transport_rank) {
        EXPECT_EQ(
            (ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<true>(
                transport_rank, 2, 2, ttnn::ccl::snake_ring::Orientation::Row)),
            row_snake_2x2[transport_rank]);
    }

    constexpr std::array<uint32_t, 6> column_snake_3x2{0, 2, 4, 5, 3, 1};
    for (uint32_t transport_rank = 0; transport_rank < column_snake_3x2.size(); ++transport_rank) {
        EXPECT_EQ(
            (ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<true>(
                transport_rank, 3, 2, ttnn::ccl::snake_ring::Orientation::Column)),
            column_snake_3x2[transport_rank]);
    }
}

TEST(CclHelpers, CreateEriscDatamoverBuilder_Chan4_PageSize2048_RRBufferSharingMode) {
    std::size_t num_channels = 4;
    uint32_t page_size = 2048;
    ttnn::ccl::EriscDataMoverBufferSharingMode buffer_sharing_mode =
        ttnn::ccl::EriscDataMoverBufferSharingMode::ROUND_ROBIN;
    ttnn::ccl::EriscDataMoverTerminationMode termination_mode =
        ttnn::ccl::EriscDataMoverTerminationMode::MESSAGE_COUNT_REACHED;

    std::size_t num_buffers_per_channel = 1;
    auto edm_builder = create_erisc_datamover_builder(
        num_channels, page_size, num_buffers_per_channel, buffer_sharing_mode, termination_mode);
    std::vector<uint32_t> worker_semaphore_ids = {0, 1, 2, 3};
    std::vector<uint32_t> message_counts = {256, 512, 24, 1};
    const std::vector<std::vector<ttnn::ccl::WorkerXY>>& worker_coords = {
        {ttnn::ccl::WorkerXY{1, 1}, ttnn::ccl::WorkerXY{2, 1}},
        {ttnn::ccl::WorkerXY{3, 1}},
        {ttnn::ccl::WorkerXY{4, 1}, ttnn::ccl::WorkerXY{5, 1}, ttnn::ccl::WorkerXY{6, 1}},
        {ttnn::ccl::WorkerXY{1, 2}},
    };
    std::vector<bool> is_sender_channel{true, false, true, false};

    std::vector<ttnn::ccl::EriscDatamoverBuilder::ChannelBufferInterface> channel_buffer_interfaces;
    channel_buffer_interfaces.reserve(num_channels);
    for (std::size_t i = 0; i < num_channels; i++) {
        const ttnn::ccl::EriscDatamoverBuilder::ChannelBufferInterface& channel_buffer_interface =
            (is_sender_channel[i])
                ? edm_builder.add_sender_channel(worker_semaphore_ids[i], message_counts[i], worker_coords[i])
                : edm_builder.add_receiver_channel(worker_semaphore_ids[i], message_counts[i], worker_coords[i]);
        channel_buffer_interfaces.push_back(channel_buffer_interface);
        ASSERT_TRUE(channel_buffer_interface.eth_buffer_l1_address > 0);
        ASSERT_TRUE(channel_buffer_interface.eth_semaphore_l1_address > 0);
    }

    const auto& active_channels = edm_builder.get_active_channels();
    ASSERT_EQ(active_channels.size(), num_channels);
    for (std::size_t i = 0; i < active_channels.size(); ++i) {
        ASSERT_EQ(active_channels[i].channel, i);
        ASSERT_EQ(active_channels[i].is_sender, is_sender_channel.at(i));
        ASSERT_EQ(active_channels[i].worker_coords, worker_coords.at(i));
        ASSERT_TRUE(active_channels[i].worker_semaphore_id == worker_semaphore_ids.at(i));
        ASSERT_TRUE(active_channels[i].num_eth_messages_to_forward == message_counts.at(i));
    }
}

TEST(CclHelpers, EriscDatamoverConfig_GetEdmHandshakeAddress_GT_0) {
    ttnn::ccl::EriscDatamoverConfig config;
    for (std::size_t i = 0; i < 8; i++) {
        ASSERT_TRUE(config.get_edm_handshake_address() > 0);
    }
}
TEST(CclHelpers, EriscDatamoverConfig_GetSemaphoresBaseAddress_GT_0) {
    ttnn::ccl::EriscDatamoverConfig config;
    for (std::size_t i = 0; i < 8; i++) {
        ASSERT_TRUE(
            config.get_semaphores_base_address(i) >=
            (config.get_edm_handshake_address() + config.handshake_location_size +
             config.edm_receiver_first_level_ack_source_word_size));
    }
}

TEST(CclHelpers, EriscDatamoverConfig_GetBuffersBaseAddress_GT_0) {
    ttnn::ccl::EriscDatamoverConfig config;
    for (std::size_t i = 0; i < 8; i++) {
        ASSERT_TRUE(
            config.get_buffers_base_address(i) >= (config.get_edm_handshake_address() + config.handshake_location_size +
                                                   config.edm_receiver_first_level_ack_source_word_size));
    }
}

/////////////////////////////////////////
// TEST AdvanceSliceRowMajor
/////////////////////////////////////////
//                                               x_y             x_y             x_y
TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_0_0__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_1) {
    const auto expected = ttnn::ccl::coord_t(1, 0);
    const auto& result = ttnn::ccl::advance_slice_row_major({0, 0}, {1, 1}, {2, 2}, 1);
    ASSERT_EQ(result.x, expected.x);
    ASSERT_EQ(result.y, expected.y);
}
TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_1_0__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_1) {
    const auto expected = ttnn::ccl::coord_t(0, 1);
    const auto& result = ttnn::ccl::advance_slice_row_major({1, 0}, {1, 1}, {2, 2}, 1);
    ASSERT_EQ(result.x, expected.x);
    ASSERT_EQ(result.y, expected.y);
}
TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_0_1__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_1) {
    const auto expected = ttnn::ccl::coord_t(1, 1);
    const auto& result = ttnn::ccl::advance_slice_row_major({0, 1}, {1, 1}, {2, 2}, 1);
    ASSERT_EQ(result.x, expected.x);
    ASSERT_EQ(result.y, expected.y);
}
TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_0_0__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_2) {
    const auto expected = ttnn::ccl::coord_t(0, 1);
    const auto& result = ttnn::ccl::advance_slice_row_major({0, 0}, {1, 1}, {2, 2}, 2);
    ASSERT_EQ(result.x, expected.x);
    ASSERT_EQ(result.y, expected.y);
}
TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_1_0__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_2) {
    const auto expected = ttnn::ccl::coord_t(1, 1);
    const auto& result = ttnn::ccl::advance_slice_row_major({1, 0}, {1, 1}, {2, 2}, 2);
    ASSERT_EQ(result.x, expected.x);
    ASSERT_EQ(result.y, expected.y);
}

// Test cases pulled from LLama 70B prefill configurations
// chip 0 worker 0 link 0 reader unidirectional
TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_0_0__InnerShape_24_1__OuterShape_32_4__NumWorkers_4) {
    const auto worker_slice_offset = ttnn::ccl::coord_t(0, 0);
    const auto worker_slice_shape = ttnn::ccl::coord_t(24, 1);
    const auto tensor_slice_shape = ttnn::ccl::coord_t(32, 4);
    const uint32_t num_workers = 4;

    const auto expected = ttnn::ccl::coord_t(0, 2);
    const auto& result_offset =
        ttnn::ccl::advance_slice_row_major(worker_slice_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_EQ(result_offset.x, expected.x);
    ASSERT_EQ(result_offset.y, expected.y);

    const auto& result_offset2 =
        ttnn::ccl::advance_slice_row_major(result_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_TRUE(result_offset2.x >= tensor_slice_shape.x || result_offset2.y >= tensor_slice_shape.y);
}

TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_24_0__InnerShape_24_1__OuterShape_32_4__NumWorkers_4) {
    const auto worker_slice_offset = ttnn::ccl::coord_t(24, 0);
    const auto worker_slice_shape = ttnn::ccl::coord_t(24, 1);
    const auto tensor_slice_shape = ttnn::ccl::coord_t(32, 4);
    const uint32_t num_workers = 4;

    const auto expected = ttnn::ccl::coord_t(24, 2);
    const auto& result_offset =
        ttnn::ccl::advance_slice_row_major(worker_slice_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_EQ(result_offset.x, expected.x);
    ASSERT_EQ(result_offset.y, expected.y);

    const auto& result_offset2 =
        ttnn::ccl::advance_slice_row_major(result_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_TRUE(result_offset2.x >= tensor_slice_shape.x || result_offset2.y >= tensor_slice_shape.y);
}

TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_0_1__InnerShape_24_1__OuterShape_32_4__NumWorkers_4) {
    const auto worker_slice_offset = ttnn::ccl::coord_t(0, 1);
    const auto worker_slice_shape = ttnn::ccl::coord_t(24, 1);
    const auto tensor_slice_shape = ttnn::ccl::coord_t(32, 4);
    const uint32_t num_workers = 4;

    const auto expected = ttnn::ccl::coord_t(0, 3);
    const auto& result_offset =
        ttnn::ccl::advance_slice_row_major(worker_slice_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_EQ(result_offset.x, expected.x);
    ASSERT_EQ(result_offset.y, expected.y);

    const auto& result_offset2 =
        ttnn::ccl::advance_slice_row_major(result_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_TRUE(result_offset2.x >= tensor_slice_shape.x || result_offset2.y >= tensor_slice_shape.y);
}

TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_24_1__InnerShape_24_1__OuterShape_32_4__NumWorkers_4) {
    const auto worker_slice_offset = ttnn::ccl::coord_t(24, 1);
    const auto worker_slice_shape = ttnn::ccl::coord_t(24, 1);
    const auto tensor_slice_shape = ttnn::ccl::coord_t(32, 4);
    const uint32_t num_workers = 4;

    const auto expected = ttnn::ccl::coord_t(24, 3);
    const auto& result_offset =
        ttnn::ccl::advance_slice_row_major(worker_slice_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_EQ(result_offset.x, expected.x);
    ASSERT_EQ(result_offset.y, expected.y);

    const auto& result_offset2 =
        ttnn::ccl::advance_slice_row_major(result_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_TRUE(result_offset2.x >= tensor_slice_shape.x || result_offset2.y >= tensor_slice_shape.y);
}

// Test that we successfully go out of bounds on the last iteration
TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_0_1__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_2) {
    const auto& result = ttnn::ccl::advance_slice_row_major({0, 1}, {1, 1}, {2, 2}, 2);
    ASSERT_TRUE(result.x >= 2 || result.y >= 2);
}
TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_1_1__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_2) {
    const auto& result = ttnn::ccl::advance_slice_row_major({1, 1}, {1, 1}, {2, 2}, 2);
    ASSERT_TRUE(result.x >= 2 || result.y >= 2);
}

TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_0_0__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_3) {
    const auto expected = ttnn::ccl::coord_t(1, 1);
    const auto& result = ttnn::ccl::advance_slice_row_major({0, 0}, {1, 1}, {2, 2}, 3);
    ASSERT_EQ(result.x, expected.x);
    ASSERT_EQ(result.y, expected.y);
}
TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_1_1__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_3) {
    const auto outer_shape = ttnn::ccl::coord_t(2, 2);
    const auto inner_offset = ttnn::ccl::coord_t(1, 1);
    const auto inner_shape = ttnn::ccl::coord_t(1, 1);
    const uint32_t num_parallel_workers = 3;
    const auto& result =
        ttnn::ccl::advance_slice_row_major(inner_offset, inner_shape, outer_shape, num_parallel_workers);
    ASSERT_TRUE(result.x >= outer_shape.x || result.y >= outer_shape.y);
}
TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_24_0__InnerShape_24_0__OuterShape_32_4__NumActiveSlices_4) {
    const auto expected = ttnn::ccl::coord_t(24, 2);
    const auto outer_shape = ttnn::ccl::coord_t(32, 4);
    const auto inner_offset = ttnn::ccl::coord_t(24, 0);
    const auto inner_shape = ttnn::ccl::coord_t(24, 1);
    const uint32_t num_parallel_workers = 4;
    const auto& result =
        ttnn::ccl::advance_slice_row_major(inner_offset, inner_shape, outer_shape, num_parallel_workers);
    ASSERT_EQ(result.x, expected.x);
    ASSERT_EQ(result.y, expected.y);
}

/////////////////////////////////////////
// TEST AdvanceWrappedSliceRowMajor
/////////////////////////////////////////
//                                               x_y             x_y             x_y
TEST(CclHelper_AdvanceWrappedSliceRowMajor, InnerOffset_0_0__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_1) {
    const auto expected = ttnn::ccl::coord_t(1, 0);
    const auto& result = ttnn::ccl::advance_wrapped_slice_row_major({0, 0}, {1, 1}, {2, 2}, 1);
    ASSERT_EQ(result.x, expected.x);
    ASSERT_EQ(result.y, expected.y);
}
TEST(CclHelper_AdvanceWrappedSliceRowMajor, InnerOffset_1_0__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_1) {
    const auto expected = ttnn::ccl::coord_t(0, 1);
    const auto& result = ttnn::ccl::advance_wrapped_slice_row_major({1, 0}, {1, 1}, {2, 2}, 1);
    ASSERT_EQ(result.x, expected.x);
    ASSERT_EQ(result.y, expected.y);
}
TEST(CclHelper_AdvanceWrappedSliceRowMajor, InnerOffset_0_1__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_1) {
    const auto expected = ttnn::ccl::coord_t(1, 1);
    const auto& result = ttnn::ccl::advance_wrapped_slice_row_major({0, 1}, {1, 1}, {2, 2}, 1);
    ASSERT_EQ(result.x, expected.x);
    ASSERT_EQ(result.y, expected.y);
}
TEST(CclHelper_AdvanceWrappedSliceRowMajor, InnerOffset_0_0__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_2) {
    const auto expected = ttnn::ccl::coord_t(0, 1);
    const auto& result = ttnn::ccl::advance_wrapped_slice_row_major({0, 0}, {1, 1}, {2, 2}, 2);
    ASSERT_EQ(result.x, expected.x);
    ASSERT_EQ(result.y, expected.y);
}
TEST(CclHelper_AdvanceWrappedSliceRowMajor, InnerOffset_1_0__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_2) {
    const auto expected = ttnn::ccl::coord_t(1, 1);
    const auto& result = ttnn::ccl::advance_wrapped_slice_row_major({1, 0}, {1, 1}, {2, 2}, 2);
    ASSERT_EQ(result.x, expected.x);
    ASSERT_EQ(result.y, expected.y);
}

// Test cases pulled from LLama 70B prefill configurations
// chip 0 worker 0 link 0 reader unidirectional
TEST(CclHelper_AdvanceWrappedSliceRowMajor, InnerOffset_0_0__InnerShape_24_1__OuterShape_32_4__NumWorkers_4) {
    const auto worker_slice_offset = ttnn::ccl::coord_t(0, 0);
    const auto worker_slice_shape = ttnn::ccl::coord_t(24, 1);
    const auto tensor_slice_shape = ttnn::ccl::coord_t(32, 4);
    const uint32_t num_workers = 4;

    const auto expected = ttnn::ccl::coord_t(0, 3);  // Updated
    const auto& result_offset = ttnn::ccl::advance_wrapped_slice_row_major(
        worker_slice_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_EQ(result_offset.x, expected.x);
    ASSERT_EQ(result_offset.y, expected.y);

    const auto& result_offset2 =
        ttnn::ccl::advance_wrapped_slice_row_major(result_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_TRUE(result_offset2.x >= tensor_slice_shape.x || result_offset2.y >= tensor_slice_shape.y);
}

TEST(CclHelper_AdvanceWrappedSliceRowMajor, InnerOffset_24_0__InnerShape_24_1__OuterShape_32_4__NumWorkers_4) {
    const auto worker_slice_offset = ttnn::ccl::coord_t(24, 0);
    const auto worker_slice_shape = ttnn::ccl::coord_t(24, 1);
    const auto tensor_slice_shape = ttnn::ccl::coord_t(32, 4);
    const uint32_t num_workers = 4;

    const auto expected = ttnn::ccl::coord_t(24, 3);  // Updated
    const auto& result_offset = ttnn::ccl::advance_wrapped_slice_row_major(
        worker_slice_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_EQ(result_offset.x, expected.x);
    ASSERT_EQ(result_offset.y, expected.y);

    const auto& result_offset2 =
        ttnn::ccl::advance_wrapped_slice_row_major(result_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_TRUE(result_offset2.x >= tensor_slice_shape.x || result_offset2.y >= tensor_slice_shape.y);
}

TEST(CclHelper_AdvanceWrappedSliceRowMajor, InnerOffset_0_0__InnerShape_44_1__OuterShape_32_4__NumWorkers_2) {  // New
    const auto worker_slice_offset = ttnn::ccl::coord_t(0, 0);
    const auto worker_slice_shape = ttnn::ccl::coord_t(44, 1);
    const auto tensor_slice_shape = ttnn::ccl::coord_t(32, 4);
    const uint32_t num_workers = 2;

    const auto expected = ttnn::ccl::coord_t(24, 2);
    const auto& result_offset = ttnn::ccl::advance_wrapped_slice_row_major(
        worker_slice_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_EQ(result_offset.x, expected.x);
    ASSERT_EQ(result_offset.y, expected.y);

    const auto& result_offset2 =
        ttnn::ccl::advance_wrapped_slice_row_major(result_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_TRUE(result_offset2.x >= tensor_slice_shape.x || result_offset2.y >= tensor_slice_shape.y);
}

// Test that we successfully go out of bounds on the last iteration
TEST(CclHelper_AdvanceWrappedSliceRowMajor, InnerOffset_0_1__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_2) {
    const auto& result = ttnn::ccl::advance_wrapped_slice_row_major({0, 1}, {1, 1}, {2, 2}, 2);
    ASSERT_TRUE(result.x >= 2 || result.y >= 2);
}
TEST(CclHelper_AdvanceWrappedSliceRowMajor, InnerOffset_1_1__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_2) {
    const auto& result = ttnn::ccl::advance_wrapped_slice_row_major({1, 1}, {1, 1}, {2, 2}, 2);
    ASSERT_TRUE(result.x >= 2 || result.y >= 2);
}

TEST(CclHelper_AdvanceWrappedSliceRowMajor, InnerOffset_0_0__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_3) {
    const auto expected = ttnn::ccl::coord_t(1, 1);
    const auto& result = ttnn::ccl::advance_wrapped_slice_row_major({0, 0}, {1, 1}, {2, 2}, 3);
    ASSERT_EQ(result.x, expected.x);
    ASSERT_EQ(result.y, expected.y);
}
TEST(CclHelper_AdvanceWrappedSliceRowMajor, InnerOffset_1_1__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_3) {
    const auto outer_shape = ttnn::ccl::coord_t(2, 2);
    const auto inner_offset = ttnn::ccl::coord_t(1, 1);
    const auto inner_shape = ttnn::ccl::coord_t(1, 1);
    const uint32_t num_parallel_workers = 3;
    const auto& result =
        ttnn::ccl::advance_wrapped_slice_row_major(inner_offset, inner_shape, outer_shape, num_parallel_workers);
    ASSERT_TRUE(result.x >= outer_shape.x || result.y >= outer_shape.y);
}
TEST(CclHelper_AdvanceWrappedSliceRowMajor, InnerOffset_16_1__InnerShape_24_1__OuterShape_32_4__NumWorkers_4) {
    const auto worker_slice_offset = ttnn::ccl::coord_t(16, 1);  // Updated
    const auto worker_slice_shape = ttnn::ccl::coord_t(24, 1);
    const auto tensor_slice_shape = ttnn::ccl::coord_t(32, 4);
    const uint32_t num_workers = 4;

    const auto& result_offset = ttnn::ccl::advance_wrapped_slice_row_major(
        worker_slice_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_TRUE(result_offset.x >= tensor_slice_shape.x || result_offset.y >= tensor_slice_shape.y);
}

TEST(CclHelper_AdvanceWrappedSliceRowMajor, InnerOffset_8_2__InnerShape_24_1__OuterShape_32_4__NumWorkers_4) {
    const auto worker_slice_offset = ttnn::ccl::coord_t(8, 2);  // Updated
    const auto worker_slice_shape = ttnn::ccl::coord_t(24, 1);
    const auto tensor_slice_shape = ttnn::ccl::coord_t(32, 4);
    const uint32_t num_workers = 4;

    const auto& result_offset = ttnn::ccl::advance_wrapped_slice_row_major(
        worker_slice_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_TRUE(result_offset.x >= tensor_slice_shape.x || result_offset.y >= tensor_slice_shape.y);
}

/////////////////////////////////////////
// Test RingReduceScatterTensorSlicer
/////////////////////////////////////////
TEST(Ccl_RingReduceScatterTensorSlicer, ComputeWorkerSliceOffsets_AllWorkersSameRow) {
    auto worker_slice_shapes = std::vector<tt_xy_pair>(4, {2, 2});
    tt_xy_pair tensor_slice_shape = {8, 4};
    const auto& worker_slice_offsets =
        ttnn::ccl::RingReduceScatterTensorSlicer::compute_worker_slice_offsets(worker_slice_shapes, tensor_slice_shape);
    ASSERT_EQ(worker_slice_offsets.at(0), tt_xy_pair(0, 0));
    ASSERT_EQ(worker_slice_offsets.at(1), tt_xy_pair(2, 0));
    ASSERT_EQ(worker_slice_offsets.at(2), tt_xy_pair(4, 0));
    ASSERT_EQ(worker_slice_offsets.at(3), tt_xy_pair(6, 0));
}
TEST(Ccl_RingReduceScatterTensorSlicer, ComputeWorkerSliceOffsets_1WorkerWrapToNextRowAligned) {
    auto worker_slice_shapes = std::vector<tt_xy_pair>(4, {2, 2});
    tt_xy_pair tensor_slice_shape = {6, 4};
    const auto& worker_slice_offsets =
        ttnn::ccl::RingReduceScatterTensorSlicer::compute_worker_slice_offsets(worker_slice_shapes, tensor_slice_shape);
    ASSERT_EQ(worker_slice_offsets.at(0), tt_xy_pair(0, 0));
    ASSERT_EQ(worker_slice_offsets.at(1), tt_xy_pair(2, 0));
    ASSERT_EQ(worker_slice_offsets.at(2), tt_xy_pair(4, 0));
    ASSERT_EQ(worker_slice_offsets.at(3), tt_xy_pair(0, 2));
}
TEST(Ccl_RingReduceScatterTensorSlicer, ComputeWorkerSliceOffsets_1WorkerWrapToNextRowMisaligned) {
    auto worker_slice_shapes = std::vector<tt_xy_pair>(4, {2, 2});
    tt_xy_pair tensor_slice_shape = {5, 4};
    const auto& worker_slice_offsets =
        ttnn::ccl::RingReduceScatterTensorSlicer::compute_worker_slice_offsets(worker_slice_shapes, tensor_slice_shape);
    ASSERT_EQ(worker_slice_offsets.at(0), tt_xy_pair(0, 0));
    ASSERT_EQ(worker_slice_offsets.at(1), tt_xy_pair(2, 0));
    ASSERT_EQ(worker_slice_offsets.at(2), tt_xy_pair(4, 0));
    ASSERT_EQ(worker_slice_offsets.at(3), tt_xy_pair(0, 2));
}

TEST(Ccl_RingReduceScatterTensorSlicer, ComputeWorkerSliceOffsets_MultipleWorkersWrapToNextRowAligned) {
    auto worker_slice_shapes = std::vector<tt_xy_pair>(8, {2, 2});
    tt_xy_pair tensor_slice_shape = {10, 4};
    const auto& worker_slice_offsets =
        ttnn::ccl::RingReduceScatterTensorSlicer::compute_worker_slice_offsets(worker_slice_shapes, tensor_slice_shape);
    ASSERT_EQ(worker_slice_offsets.at(0), tt_xy_pair(0, 0));
    ASSERT_EQ(worker_slice_offsets.at(1), tt_xy_pair(2, 0));
    ASSERT_EQ(worker_slice_offsets.at(2), tt_xy_pair(4, 0));
    ASSERT_EQ(worker_slice_offsets.at(3), tt_xy_pair(6, 0));
    ASSERT_EQ(worker_slice_offsets.at(4), tt_xy_pair(8, 0));
    ASSERT_EQ(worker_slice_offsets.at(5), tt_xy_pair(0, 2));
    ASSERT_EQ(worker_slice_offsets.at(6), tt_xy_pair(2, 2));
    ASSERT_EQ(worker_slice_offsets.at(7), tt_xy_pair(4, 2));
}

TEST(Ccl_RingReduceScatterTensorSlicer, ComputeWorkerSliceOffsets_MultipleWorkersWrapToNextRowMisaligned) {
    auto worker_slice_shapes = std::vector<tt_xy_pair>(8, {2, 2});
    tt_xy_pair tensor_slice_shape = {9, 4};
    const auto& worker_slice_offsets =
        ttnn::ccl::RingReduceScatterTensorSlicer::compute_worker_slice_offsets(worker_slice_shapes, tensor_slice_shape);
    ASSERT_EQ(worker_slice_offsets.at(0), tt_xy_pair(0, 0));
    ASSERT_EQ(worker_slice_offsets.at(1), tt_xy_pair(2, 0));
    ASSERT_EQ(worker_slice_offsets.at(2), tt_xy_pair(4, 0));
    ASSERT_EQ(worker_slice_offsets.at(3), tt_xy_pair(6, 0));
    ASSERT_EQ(worker_slice_offsets.at(4), tt_xy_pair(8, 0));
    ASSERT_EQ(worker_slice_offsets.at(5), tt_xy_pair(0, 2));
    ASSERT_EQ(worker_slice_offsets.at(6), tt_xy_pair(2, 2));
    ASSERT_EQ(worker_slice_offsets.at(7), tt_xy_pair(4, 2));
}

TEST(Ccl_RingReduceScatterTensorSlicer, ComputeWorkerSliceOffsets_NMinus1WorkersWrapToNextRowAligned) {
    auto worker_slice_shapes = std::vector<tt_xy_pair>(3, {4, 4});
    tt_xy_pair tensor_slice_shape = {4, 12};
    const auto& worker_slice_offsets =
        ttnn::ccl::RingReduceScatterTensorSlicer::compute_worker_slice_offsets(worker_slice_shapes, tensor_slice_shape);
    ASSERT_EQ(worker_slice_offsets.at(0), tt_xy_pair(0, 0));
    ASSERT_EQ(worker_slice_offsets.at(1), tt_xy_pair(0, 4));
    ASSERT_EQ(worker_slice_offsets.at(2), tt_xy_pair(0, 8));
}

TEST(Ccl_RingReduceScatterTensorSlicer, ComputeWorkerSliceOffsets_NMinus1WorkersWrapToNextRowMisaligned) {
    auto worker_slice_shapes = std::vector<tt_xy_pair>(3, {4, 3});
    tt_xy_pair tensor_slice_shape = {3, 12};
    const auto& worker_slice_offsets =
        ttnn::ccl::RingReduceScatterTensorSlicer::compute_worker_slice_offsets(worker_slice_shapes, tensor_slice_shape);
    ASSERT_EQ(worker_slice_offsets.at(0), tt_xy_pair(0, 0));
    ASSERT_EQ(worker_slice_offsets.at(1), tt_xy_pair(0, 3));
    ASSERT_EQ(worker_slice_offsets.at(2), tt_xy_pair(0, 6));
}

/////////////////////////////////////////
// Test RingReduceScatterWrappedTensorSlicer
/////////////////////////////////////////
TEST(Ccl_RingReduceScatterWrappedTensorSlicer, ComputeWorkerSliceWrappedOffsets_AllWorkersSameRow) {
    auto worker_slice_shapes = std::vector<tt_xy_pair>(4, {2, 1});
    tt_xy_pair tensor_slice_shape = {8, 1};
    const auto& worker_slice_offsets = ttnn::ccl::RingReduceScatterWrappedTensorSlicer::compute_worker_slice_offsets(
        worker_slice_shapes, tensor_slice_shape);
    ASSERT_EQ(worker_slice_offsets.at(0), tt_xy_pair(0, 0));
    ASSERT_EQ(worker_slice_offsets.at(1), tt_xy_pair(2, 0));
    ASSERT_EQ(worker_slice_offsets.at(2), tt_xy_pair(4, 0));
    ASSERT_EQ(worker_slice_offsets.at(3), tt_xy_pair(6, 0));
}
TEST(Ccl_RingReduceScatterWrappedTensorSlicer, ComputeWorkerSliceWrappedOffsets_1WorkerWrapToNextRowAligned) {
    auto worker_slice_shapes = std::vector<tt_xy_pair>(4, {2, 1});
    tt_xy_pair tensor_slice_shape = {6, 2};
    const auto& worker_slice_offsets = ttnn::ccl::RingReduceScatterWrappedTensorSlicer::compute_worker_slice_offsets(
        worker_slice_shapes, tensor_slice_shape);
    ASSERT_EQ(worker_slice_offsets.at(0), tt_xy_pair(0, 0));
    ASSERT_EQ(worker_slice_offsets.at(1), tt_xy_pair(2, 0));
    ASSERT_EQ(worker_slice_offsets.at(2), tt_xy_pair(4, 0));
    ASSERT_EQ(worker_slice_offsets.at(3), tt_xy_pair(0, 1));
}
TEST(Ccl_RingReduceScatterWrappedTensorSlicer, ComputeWorkerSliceWrappedOffsets_1WorkerWrapToNextRowMisaligned) {
    auto worker_slice_shapes = std::vector<tt_xy_pair>(4, {2, 1});
    tt_xy_pair tensor_slice_shape = {5, 2};
    const auto& worker_slice_offsets = ttnn::ccl::RingReduceScatterWrappedTensorSlicer::compute_worker_slice_offsets(
        worker_slice_shapes, tensor_slice_shape);
    ASSERT_EQ(worker_slice_offsets.at(0), tt_xy_pair(0, 0));
    ASSERT_EQ(worker_slice_offsets.at(1), tt_xy_pair(2, 0));
    ASSERT_EQ(worker_slice_offsets.at(2), tt_xy_pair(4, 0));
    ASSERT_EQ(worker_slice_offsets.at(3), tt_xy_pair(1, 1));
}
TEST(Ccl_RingReduceScatterWrappedTensorSlicer, ComputeWorkerSliceWrappedOffsets_MultipleWorkersWrapToNextRowAligned) {
    auto worker_slice_shapes = std::vector<tt_xy_pair>(8, {2, 1});
    tt_xy_pair tensor_slice_shape = {10, 2};
    const auto& worker_slice_offsets = ttnn::ccl::RingReduceScatterWrappedTensorSlicer::compute_worker_slice_offsets(
        worker_slice_shapes, tensor_slice_shape);
    ASSERT_EQ(worker_slice_offsets.at(0), tt_xy_pair(0, 0));
    ASSERT_EQ(worker_slice_offsets.at(1), tt_xy_pair(2, 0));
    ASSERT_EQ(worker_slice_offsets.at(2), tt_xy_pair(4, 0));
    ASSERT_EQ(worker_slice_offsets.at(3), tt_xy_pair(6, 0));
    ASSERT_EQ(worker_slice_offsets.at(4), tt_xy_pair(8, 0));
    ASSERT_EQ(worker_slice_offsets.at(5), tt_xy_pair(0, 1));
    ASSERT_EQ(worker_slice_offsets.at(6), tt_xy_pair(2, 1));
    ASSERT_EQ(worker_slice_offsets.at(7), tt_xy_pair(4, 1));
}

TEST(
    Ccl_RingReduceScatterWrappedTensorSlicer, ComputeWorkerSliceWrappedOffsets_MultipleWorkersWrapToNextRowMisaligned) {
    auto worker_slice_shapes = std::vector<tt_xy_pair>(8, {2, 1});
    tt_xy_pair tensor_slice_shape = {9, 2};
    const auto& worker_slice_offsets = ttnn::ccl::RingReduceScatterWrappedTensorSlicer::compute_worker_slice_offsets(
        worker_slice_shapes, tensor_slice_shape);
    ASSERT_EQ(worker_slice_offsets.at(0), tt_xy_pair(0, 0));
    ASSERT_EQ(worker_slice_offsets.at(1), tt_xy_pair(2, 0));
    ASSERT_EQ(worker_slice_offsets.at(2), tt_xy_pair(4, 0));
    ASSERT_EQ(worker_slice_offsets.at(3), tt_xy_pair(6, 0));
    ASSERT_EQ(worker_slice_offsets.at(4), tt_xy_pair(8, 0));
    ASSERT_EQ(worker_slice_offsets.at(5), tt_xy_pair(1, 1));
    ASSERT_EQ(worker_slice_offsets.at(6), tt_xy_pair(3, 1));
    ASSERT_EQ(worker_slice_offsets.at(7), tt_xy_pair(5, 1));
}

TEST(Ccl_RingReduceScatterWrappedTensorSlicer, ComputeWorkerSliceWrappedOffsets_NMinus1WorkersWrapToNextRowAligned) {
    auto worker_slice_shapes = std::vector<tt_xy_pair>(3, {16, 1});
    tt_xy_pair tensor_slice_shape = {4, 12};
    const auto& worker_slice_offsets = ttnn::ccl::RingReduceScatterWrappedTensorSlicer::compute_worker_slice_offsets(
        worker_slice_shapes, tensor_slice_shape);
    ASSERT_EQ(worker_slice_offsets.at(0), tt_xy_pair(0, 0));
    ASSERT_EQ(worker_slice_offsets.at(1), tt_xy_pair(0, 4));
    ASSERT_EQ(worker_slice_offsets.at(2), tt_xy_pair(0, 8));
}

TEST(Ccl_RingReduceScatterWrappedTensorSlicer, ComputeWorkerSliceWrappedOffsets_NMinus1WorkersWrapToNextRowMisaligned) {
    auto worker_slice_shapes = std::vector<tt_xy_pair>(3, {11, 1});
    tt_xy_pair tensor_slice_shape = {3, 12};
    const auto& worker_slice_offsets = ttnn::ccl::RingReduceScatterWrappedTensorSlicer::compute_worker_slice_offsets(
        worker_slice_shapes, tensor_slice_shape);
    ASSERT_EQ(worker_slice_offsets.at(0), tt_xy_pair(0, 0));
    ASSERT_EQ(worker_slice_offsets.at(1), tt_xy_pair(2, 3));
    ASSERT_EQ(worker_slice_offsets.at(2), tt_xy_pair(1, 7));
}

TEST(
    Ccl_InterleavedTensorWorkerSlice_ComputeNumWorkerSliceIterations,
    InnerOffset_0_0__InnerShape_24_1__OuterShape_32_4__NumActiveSlices_4) {
    auto worker_slice = ttnn::ccl::InterleavedTensorWorkerSlice(
        tt_xy_pair(99999, 99999),  // tensor shape shouldn't affect the result
        tt_xy_pair(32, 4),
        tt_xy_pair(24, 1),
        tt_xy_pair(0, 0));
    uint32_t num_workers = 4;
    auto num_iterations = worker_slice.compute_num_worker_slice_iterations(num_workers);
    auto expected = 2;
    ASSERT_EQ(num_iterations, expected);
}

TEST(
    Ccl_InterleavedTensorWorkerSlice_ComputeNumWorkerSliceIterations,
    InnerOffset_24_0__InnerShape_24_1__OuterShape_32_4__NumActiveSlices_4) {
    auto worker_slice = ttnn::ccl::InterleavedTensorWorkerSlice(
        tt_xy_pair(99999, 99999),  // tensor shape shouldn't affect the result
        tt_xy_pair(32, 4),
        tt_xy_pair(24, 1),
        tt_xy_pair(24, 0));
    uint32_t num_workers = 4;
    auto num_iterations = worker_slice.compute_num_worker_slice_iterations(num_workers);
    auto expected = 2;
    ASSERT_EQ(num_iterations, expected);
}

TEST(
    Ccl_InterleavedTensorWorkerSlice_ComputeNumWorkerSliceIterations,
    InnerOffset_0_1__InnerShape_24_1__OuterShape_32_4__NumActiveSlices_4) {
    auto worker_slice = ttnn::ccl::InterleavedTensorWorkerSlice(
        tt_xy_pair(99999, 99999),  // tensor shape shouldn't affect the result
        tt_xy_pair(32, 4),
        tt_xy_pair(24, 1),
        tt_xy_pair(0, 1));
    uint32_t num_workers = 4;
    auto num_iterations = worker_slice.compute_num_worker_slice_iterations(num_workers);
    auto expected = 2;
    ASSERT_EQ(num_iterations, expected);
}

TEST(
    Ccl_InterleavedTensorWorkerSlice_ComputeNumWorkerSliceIterations,
    InnerOffset_24_1__InnerShape_24_1__OuterShape_32_4__NumActiveSlices_4) {
    auto worker_slice = ttnn::ccl::InterleavedTensorWorkerSlice(
        tt_xy_pair(99999, 99999),  // tensor shape shouldn't affect the result
        tt_xy_pair(32, 4),
        tt_xy_pair(24, 1),
        tt_xy_pair(24, 0));
    uint32_t num_workers = 4;
    auto num_iterations = worker_slice.compute_num_worker_slice_iterations(num_workers);
    auto expected = 2;
    ASSERT_EQ(num_iterations, expected);
}
// Wrapped version
TEST(
    Ccl_InterleavedTensorWorkerSlice_ComputeNumWrappedWorkerSliceIterations,
    InnerOffset_0_0__InnerShape_24_1__OuterShape_32_4__NumActiveSlices_4) {
    auto worker_slice = ttnn::ccl::InterleavedTensorWorkerSlice(
        tt_xy_pair(99999, 99999),  // tensor shape shouldn't affect the result
        tt_xy_pair(32, 4),
        tt_xy_pair(24, 1),
        tt_xy_pair(0, 0),
        true);
    uint32_t num_workers = 4;
    auto num_iterations = worker_slice.compute_num_worker_slice_iterations(num_workers);
    auto expected = 2;
    ASSERT_EQ(num_iterations, expected);
}

TEST(
    Ccl_InterleavedTensorWorkerSlice_ComputeNumWrappedWorkerSliceIterations,
    InnerOffset_24_0__InnerShape_24_1__OuterShape_32_4__NumActiveSlices_4) {
    auto worker_slice = ttnn::ccl::InterleavedTensorWorkerSlice(
        tt_xy_pair(99999, 99999),  // tensor shape shouldn't affect the result
        tt_xy_pair(32, 4),
        tt_xy_pair(24, 1),
        tt_xy_pair(24, 0),
        true);
    uint32_t num_workers = 4;
    auto num_iterations = worker_slice.compute_num_worker_slice_iterations(num_workers);
    auto expected = 2;
    ASSERT_EQ(num_iterations, expected);
}

TEST(
    Ccl_InterleavedTensorWorkerSlice_ComputeNumWrappedWorkerSliceIterations,
    InnerOffset_16_1__InnerShape_24_1__OuterShape_32_4__NumActiveSlices_4) {
    auto worker_slice = ttnn::ccl::InterleavedTensorWorkerSlice(
        tt_xy_pair(99999, 99999),  // tensor shape shouldn't affect the result
        tt_xy_pair(32, 4),
        tt_xy_pair(24, 1),
        tt_xy_pair(16, 1),
        true);  // Updated
    uint32_t num_workers = 4;
    auto num_iterations = worker_slice.compute_num_worker_slice_iterations(num_workers);
    auto expected = 1;  // Updated
    ASSERT_EQ(num_iterations, expected);
}

TEST(
    Ccl_InterleavedTensorWorkerSlice_ComputeNumWrappedWorkerSliceIterations,
    InnerOffset_8_2__InnerShape_24_1__OuterShape_32_4__NumActiveSlices_4) {
    auto worker_slice = ttnn::ccl::InterleavedTensorWorkerSlice(
        tt_xy_pair(99999, 99999),  // tensor shape shouldn't affect the result
        tt_xy_pair(32, 4),
        tt_xy_pair(24, 1),
        tt_xy_pair(8, 2),
        true);
    uint32_t num_workers = 4;
    auto num_iterations = worker_slice.compute_num_worker_slice_iterations(num_workers);
    auto expected = 1;
    ASSERT_EQ(num_iterations, expected);
}

// ---------------------------------------------------------------------------------------------------------------------
// ccl_topology_utils: output TensorTopology labels of the collective family. Pure functions on (TensorTopology,
// cluster_axis, MeshShape, rank); no device. Meshes: (1,2), (1,8), (2,4), (8,4), (2,2,2).
// ---------------------------------------------------------------------------------------------------------------------

namespace {

namespace topo = ttnn::operations::ccl::common;
using tt::tt_metal::TensorTopology;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshCoordinateRange;
using tt::tt_metal::distributed::MeshShape;
using TopoPlacement = tt::tt_metal::distributed::MeshMapperConfig::Placement;
using TopoReplicate = tt::tt_metal::distributed::MeshMapperConfig::Replicate;
using TopoShard = tt::tt_metal::distributed::MeshMapperConfig::Shard;

constexpr uint32_t kRank = 4;

std::vector<MeshCoordinate> row_major_coords(const MeshShape& mesh) {
    std::vector<MeshCoordinate> coords;
    for (const auto& coord : MeshCoordinateRange(mesh)) {
        coords.push_back(coord);
    }
    return coords;
}

// One placement per mesh axis, as ShardTensor2dMesh / create_mesh_mapper produce.
TensorTopology nd_label(
    const MeshShape& mesh, const std::vector<TopoPlacement>& placements, std::vector<MeshCoordinate> coords = {}) {
    if (coords.empty()) {
        coords = row_major_coords(mesh);
    }
    return TensorTopology(
        mesh, ttsl::SmallVector<TopoPlacement>(placements.begin(), placements.end()), std::move(coords));
}

// {N},[placement] over the mesh in row-major order, as ShardTensorToMesh / ReplicateTensorToMesh produce.
TensorTopology collapsed_label(
    const MeshShape& mesh, const TopoPlacement& placement, std::vector<MeshCoordinate> coords = {}) {
    if (coords.empty()) {
        coords = row_major_coords(mesh);
    }
    return TensorTopology(MeshShape(static_cast<uint32_t>(mesh.mesh_size())), {placement}, std::move(coords));
}

// Sets ttnn::CONFIG.strict_ccl_topology for the scope and restores it after.
class StrictCclTopologyScope {
public:
    explicit StrictCclTopologyScope(bool strict) : previous_(ttnn::CONFIG.get<"strict_ccl_topology">()) {
        ttnn::CONFIG.set<"strict_ccl_topology">(strict);
    }
    ~StrictCclTopologyScope() { ttnn::CONFIG.set<"strict_ccl_topology">(previous_); }

private:
    bool previous_;
};

template <typename F>
std::string message_of(F&& f) {
    try {
        f();
    } catch (const std::exception& e) {
        return e.what();
    }
    return {};
}

const std::vector<MeshShape>& all_meshes() {
    static const std::vector<MeshShape> meshes{
        MeshShape(1, 2), MeshShape(1, 8), MeshShape(2, 4), MeshShape(8, 4), MeshShape(2, 2, 2)};
    return meshes;
}

}  // namespace

TEST(CclTopologyUtils, UncollapseExpandsCollapsedLabelsPerMeshAxis) {
    for (const auto& mesh : all_meshes()) {
        const auto replicated = topo::uncollapse_placements(collapsed_label(mesh, TopoReplicate{}), mesh);
        ASSERT_TRUE(replicated.has_value()) << mesh;
        ASSERT_EQ(replicated->size(), mesh.dims()) << mesh;
        for (const auto& placement : *replicated) {
            EXPECT_TRUE(std::holds_alternative<TopoReplicate>(placement)) << mesh;
        }

        // Shard{d} on every axis of size > 1 (row-major hierarchical sharding), Replicate on a size-1 axis.
        const auto sharded = topo::uncollapse_placements(collapsed_label(mesh, TopoShard{3}), mesh);
        ASSERT_TRUE(sharded.has_value()) << mesh;
        ASSERT_EQ(sharded->size(), mesh.dims()) << mesh;
        for (size_t axis = 0; axis < mesh.dims(); ++axis) {
            const TopoPlacement expected =
                mesh[static_cast<int32_t>(axis)] > 1 ? TopoPlacement{TopoShard{3}} : TopoPlacement{TopoReplicate{}};
            EXPECT_EQ((*sharded)[axis], expected) << mesh << " axis " << axis;
        }
    }

    // An N-D label comes back verbatim, Shard dims spelled as given (negative or stale).
    const MeshShape mesh(2, 4);
    const std::vector<TopoPlacement> placements{TopoShard{-2}, TopoShard{7}};
    const auto verbatim = topo::uncollapse_placements(nd_label(mesh, placements), mesh);
    ASSERT_TRUE(verbatim.has_value());
    EXPECT_EQ(std::vector<TopoPlacement>(verbatim->begin(), verbatim->end()), placements);
}

TEST(CclTopologyUtils, UncollapseRefusesLabelsThatDoNotCoverTheMeshRowMajor) {
    const MeshShape mesh(2, 4);
    std::string reason;

    // ShardTensorToMesh with fewer chunks than devices: {4},[Shard{0}] over the first four coordinates.
    auto coords = row_major_coords(mesh);
    coords.erase(coords.begin() + 4, coords.end());  // MeshCoordinate has no default ctor: no resize()
    const TensorTopology fewer_shards(MeshShape(4), {TopoShard{0}}, coords);
    EXPECT_FALSE(topo::uncollapse_placements(fewer_shards, mesh, &reason).has_value());
    EXPECT_NE(reason.find("row-major"), std::string::npos) << reason;

    // Eight coordinates that are not the row-major enumeration of the mesh (a column-major walk).
    std::vector<MeshCoordinate> column_major;
    for (uint32_t c = 0; c < 4; ++c) {
        for (uint32_t r = 0; r < 2; ++r) {
            column_major.emplace_back(r, c);
        }
    }
    const TensorTopology permuted(MeshShape(8), {TopoShard{0}}, column_major);
    EXPECT_FALSE(topo::uncollapse_placements(permuted, mesh, &reason).has_value());
    EXPECT_NE(reason.find("row-major"), std::string::npos) << reason;

    // A 1-D distribution shape with two placements is neither form.
    const TensorTopology mismatched(MeshShape(8), {TopoShard{0}, TopoShard{1}}, row_major_coords(mesh));
    EXPECT_FALSE(topo::uncollapse_placements(mismatched, mesh, &reason).has_value());
    EXPECT_NE(reason.find("neither"), std::string::npos) << reason;
}

TEST(CclTopologyUtils, AllGatherReplicatesTheClusterAxisOfAnNDLabelAndKeepsTheRest) {
    const MeshShape mesh(2, 4);
    const auto in = nd_label(mesh, {TopoShard{2}, TopoShard{3}});

    EXPECT_EQ(
        topo::all_gather_output_topology(in, 1, mesh, kRank, /*gathered_dim=*/3),
        nd_label(mesh, {TopoShard{2}, TopoReplicate{}}));
    EXPECT_EQ(
        topo::all_gather_output_topology(in, 0, mesh, kRank, /*gathered_dim=*/2),
        nd_label(mesh, {TopoReplicate{}, TopoShard{3}}));

    // Rule (e): another axis sharding the gathered dim keeps its Shard -- its pieces are still distinct after the
    // gather. The old prim all_gather rule replicated every axis whose Shard dim equalled the gather dim.
    EXPECT_EQ(
        topo::all_gather_output_topology(nd_label(mesh, {TopoShard{3}, TopoReplicate{}}), 1, mesh, kRank, 3),
        nd_label(mesh, {TopoShard{3}, TopoReplicate{}}));

    // A size-1 axis keeps whatever it held; the gathered axis of a 1x8 N-D label becomes Replicate.
    const MeshShape line(1, 8);
    EXPECT_EQ(
        topo::all_gather_output_topology(nd_label(line, {TopoReplicate{}, TopoShard{3}}), 1, line, kRank, 3),
        nd_label(line, {TopoReplicate{}, TopoReplicate{}}));

    // An out-of-range cluster_axis is left to the op's validation: nullopt, no throw even under strict mode.
    StrictCclTopologyScope strict(true);
    EXPECT_FALSE(topo::all_gather_output_topology(in, 2, mesh, kRank, 3).has_value());
}

TEST(CclTopologyUtils, AllGatherWholeMeshReplicatesEverythingAndKeepsTheDistributionShape) {
    for (const auto& mesh : all_meshes()) {
        // Collapsed input: the ring order is the label's coordinate order, so shape and coords are kept.
        const auto collapsed = collapsed_label(mesh, TopoShard{-2});
        EXPECT_EQ(
            topo::all_gather_output_topology(collapsed, std::nullopt, mesh, kRank, /*gathered_dim=*/3),
            collapsed_label(mesh, TopoReplicate{}))
            << mesh;

        std::vector<TopoPlacement> nd(mesh.dims(), TopoShard{3});
        std::vector<TopoPlacement> all_replicate(mesh.dims(), TopoReplicate{});
        EXPECT_EQ(
            topo::all_gather_output_topology(nd_label(mesh, nd), std::nullopt, mesh, kRank, 3),
            nd_label(mesh, all_replicate))
            << mesh;
    }
}

TEST(CclTopologyUtils, AllGatherOfACollapsedShardAlongTheInnermostAxis) {
    // 2-D meshes: the inner axis (1) gathers the fine pieces back into the row's chunk -> [Shard{d}, Replicate].
    for (const auto& mesh : {MeshShape(2, 4), MeshShape(8, 4)}) {
        const auto in = collapsed_label(mesh, TopoShard{3});
        const auto out = topo::all_gather_output_topology(in, 1, mesh, kRank, 3);
        ASSERT_TRUE(out.has_value()) << mesh;
        EXPECT_EQ(*out, nd_label(mesh, {TopoShard{3}, TopoReplicate{}})) << mesh;
        EXPECT_EQ(out->mesh_coords(), in.mesh_coords()) << mesh;
    }

    // A line (1xN): the collapsed axis is the gathered axis, so the collapsed spelling is kept.
    for (const auto& mesh : {MeshShape(1, 2), MeshShape(1, 8)}) {
        EXPECT_EQ(
            topo::all_gather_output_topology(collapsed_label(mesh, TopoShard{3}), 1, mesh, kRank, 3),
            collapsed_label(mesh, TopoReplicate{}))
            << mesh;
    }

    // Three non-trivial axes: gathering along the innermost leaves Shard{3} on two axes next to a Replicate axis,
    // which no label expresses (rule (c)): warn-only gives nullopt, strict throws.
    const MeshShape cube(2, 2, 2);
    {
        StrictCclTopologyScope warn_only(false);
        EXPECT_FALSE(
            topo::all_gather_output_topology(collapsed_label(cube, TopoShard{3}), 2, cube, kRank, 3).has_value());
    }
    StrictCclTopologyScope strict(true);
    const auto message =
        message_of([&] { topo::all_gather_output_topology(collapsed_label(cube, TopoShard{3}), 2, cube, kRank, 3); });
    EXPECT_NE(message.find("express"), std::string::npos) << message;
}

TEST(CclTopologyUtils, AllGatherOfACollapsedShardAlongAnOuterAxisIsRefusedUnlessAnotherDimIsGathered) {
    const MeshShape mesh(2, 4);
    const auto in = collapsed_label(mesh, TopoShard{3});

    // Rule (d): gathering dim 3 along axis 0 interleaves the pieces axis 1 keeps apart.
    {
        StrictCclTopologyScope strict(false);
        EXPECT_FALSE(topo::all_gather_output_topology(in, 0, mesh, kRank, 3).has_value());
    }
    {
        StrictCclTopologyScope strict(true);
        const auto message = message_of([&] { topo::all_gather_output_topology(in, 0, mesh, kRank, 3); });
        EXPECT_NE(message.find("interleave"), std::string::npos) << message;

        // Gathering a different dim leaves the dim-3 pieces where they are: honest along either axis.
        EXPECT_EQ(
            topo::all_gather_output_topology(in, 0, mesh, kRank, /*gathered_dim=*/2),
            nd_label(mesh, {TopoReplicate{}, TopoShard{3}}));

        // all_reduce / all_broadcast concatenate nothing, so the outer axis is fine for them.
        EXPECT_EQ(
            topo::all_gather_output_topology(in, 0, mesh, kRank, 3, /*require_contiguous_gather=*/false),
            nd_label(mesh, {TopoReplicate{}, TopoShard{3}}));
        EXPECT_EQ(
            topo::all_reduce_output_topology(in, 0, mesh, kRank), nd_label(mesh, {TopoReplicate{}, TopoShard{3}}));
        EXPECT_EQ(
            topo::all_broadcast_output_topology(in, 1, mesh, kRank), nd_label(mesh, {TopoShard{3}, TopoReplicate{}}));
    }
}

TEST(CclTopologyUtils, AllGatherComparesShardDimsNormalisedAndIgnoresOutOfRangeOnes) {
    const MeshShape mesh(2, 4);
    StrictCclTopologyScope strict(true);

    // -1 and 3 are the same axis of a rank-4 tensor: the interleave check fires.
    const auto message =
        message_of([&] { topo::all_gather_output_topology(collapsed_label(mesh, TopoShard{-1}), 0, mesh, kRank, 3); });
    EXPECT_NE(message.find("interleave"), std::string::npos) << message;

    // Existing placements are kept as spelled.
    EXPECT_EQ(
        topo::all_gather_output_topology(collapsed_label(mesh, TopoShard{-1}), 1, mesh, kRank, -1),
        nd_label(mesh, {TopoShard{-1}, TopoReplicate{}}));

    // A stale dim left by a rank-changing op (#52331) matches nothing and is never an error.
    EXPECT_EQ(
        topo::all_gather_output_topology(nd_label(mesh, {TopoShard{7}, TopoShard{3}}), 1, mesh, kRank, 3),
        nd_label(mesh, {TopoShard{7}, TopoReplicate{}}));
    EXPECT_EQ(
        topo::reduce_scatter_output_topology(nd_label(mesh, {TopoShard{7}, TopoReplicate{}}), 1, mesh, kRank, 3),
        nd_label(mesh, {TopoShard{7}, TopoShard{3}}));
}

TEST(CclTopologyUtils, ReduceScatterShardsTheClusterAxisWithTheNormalisedDim) {
    const MeshShape mesh(2, 4);
    StrictCclTopologyScope strict(true);

    // Rule (f): only normalised dims are written.
    EXPECT_EQ(
        topo::reduce_scatter_output_topology(nd_label(mesh, {TopoReplicate{}, TopoReplicate{}}), 1, mesh, kRank, -1),
        nd_label(mesh, {TopoReplicate{}, TopoShard{3}}));
    // Another dim sharded elsewhere is kept.
    EXPECT_EQ(
        topo::reduce_scatter_output_topology(nd_label(mesh, {TopoShard{2}, TopoReplicate{}}), 1, mesh, kRank, 3),
        nd_label(mesh, {TopoShard{2}, TopoShard{3}}));
    // A different Shard on the scattered axis itself is overwritten (reduce_scatter_minimal_async precedent).
    EXPECT_EQ(
        topo::reduce_scatter_output_topology(nd_label(mesh, {TopoReplicate{}, TopoShard{2}}), 1, mesh, kRank, 3),
        nd_label(mesh, {TopoReplicate{}, TopoShard{3}}));

    // Collapsed Replicate over a 2-D mesh: only an N-D label can say "Shard here, Replicate there"; the coords carry
    // over.
    const auto replicated = collapsed_label(mesh, TopoReplicate{});
    EXPECT_EQ(
        topo::reduce_scatter_output_topology(replicated, 0, mesh, kRank, 3),
        nd_label(mesh, {TopoShard{3}, TopoReplicate{}}));
    EXPECT_EQ(
        topo::reduce_scatter_output_topology(replicated, 1, mesh, kRank, 3),
        nd_label(mesh, {TopoReplicate{}, TopoShard{3}}));

    // Collapsed Shard{2} scattered on dim 3: both axes are honest.
    EXPECT_EQ(
        topo::reduce_scatter_output_topology(collapsed_label(mesh, TopoShard{2}), 1, mesh, kRank, 3),
        nd_label(mesh, {TopoShard{2}, TopoShard{3}}));
    EXPECT_EQ(
        topo::reduce_scatter_output_topology(collapsed_label(mesh, TopoShard{2}), 0, mesh, kRank, 3),
        nd_label(mesh, {TopoShard{3}, TopoShard{2}}));

    // A line keeps the collapsed spelling on its one axis, for Replicate and Shard inputs alike.
    for (const auto& line : {MeshShape(1, 2), MeshShape(1, 8)}) {
        EXPECT_EQ(
            topo::reduce_scatter_output_topology(collapsed_label(line, TopoReplicate{}), 1, line, kRank, -1),
            collapsed_label(line, TopoShard{3}))
            << line;
        EXPECT_EQ(
            topo::reduce_scatter_output_topology(collapsed_label(line, TopoShard{3}), 1, line, kRank, 3),
            collapsed_label(line, TopoShard{3}))
            << line;
    }

    // Out-of-range cluster_axis: nullopt, no throw (validation's job); out-of-range dim: a refusal.
    EXPECT_FALSE(topo::reduce_scatter_output_topology(replicated, 2, mesh, kRank, 3).has_value());
    const auto message = message_of([&] { topo::reduce_scatter_output_topology(replicated, 1, mesh, kRank, 4); });
    EXPECT_NE(message.find("out of range"), std::string::npos) << message;
}

TEST(CclTopologyUtils, ReduceScatterOfTheSameDimCollapsesOnlyAlongAnInnerAxis) {
    // Plan 1a.2(c) as amended: outer axis holds the coarse chunks, the scattered (inner) axis splits them into fine
    // pieces -> row-major hierarchical -> the collapsed label carrying the INPUT's coordinates.
    const MeshShape mesh(2, 4);
    auto distinctive_coords = row_major_coords(mesh);
    std::reverse(distinctive_coords.begin(), distinctive_coords.end());
    const auto outer_sharded = nd_label(mesh, {TopoShard{3}, TopoReplicate{}}, distinctive_coords);
    {
        StrictCclTopologyScope strict(true);
        const auto out = topo::reduce_scatter_output_topology(outer_sharded, 1, mesh, kRank, 3);
        ASSERT_TRUE(out.has_value());
        EXPECT_EQ(*out, TensorTopology(MeshShape(8), {TopoShard{3}}, distinctive_coords));
        EXPECT_EQ(
            topo::reduce_scatter_output_topology(
                nd_label(MeshShape(8, 4), {TopoShard{3}, TopoReplicate{}}), 1, MeshShape(8, 4), kRank, 3),
            collapsed_label(MeshShape(8, 4), TopoShard{3}));

        // The collapsed Shard{d} input itself: scattering d along the inner axis reproduces its own label.
        EXPECT_EQ(
            topo::reduce_scatter_output_topology(collapsed_label(mesh, TopoShard{3}), 1, mesh, kRank, 3),
            collapsed_label(mesh, TopoShard{3}));
        const MeshShape cube(2, 2, 2);
        EXPECT_EQ(
            topo::reduce_scatter_output_topology(collapsed_label(cube, TopoShard{3}), 2, cube, kRank, 3),
            collapsed_label(cube, TopoShard{3}));

        // A size-1 axis that shards the dim holds its whole extent: Replicate, and no collapse is needed.
        const MeshShape line(1, 8);
        EXPECT_EQ(
            topo::reduce_scatter_output_topology(nd_label(line, {TopoShard{3}, TopoReplicate{}}), 1, line, kRank, 3),
            nd_label(line, {TopoReplicate{}, TopoShard{3}}));
    }

    // The inner axis already shards d and the OUTER axis is scattered: column-major (device (r, c) holds piece
    // c * R + r), which no label describes. This was the data-lossy [Shard{d}, Replicate] the old clear-same-dim rule
    // emitted. Warn-only: nullopt (union default); strict: TT_FATAL.
    const auto inner_sharded = nd_label(mesh, {TopoReplicate{}, TopoShard{3}});
    {
        StrictCclTopologyScope strict(false);
        EXPECT_FALSE(topo::reduce_scatter_output_topology(inner_sharded, 0, mesh, kRank, 3).has_value());
        EXPECT_FALSE(
            topo::reduce_scatter_output_topology(collapsed_label(mesh, TopoShard{3}), 0, mesh, kRank, 3).has_value());
    }
    StrictCclTopologyScope strict(true);
    const auto message = message_of([&] { topo::reduce_scatter_output_topology(inner_sharded, 0, mesh, kRank, 3); });
    EXPECT_NE(message.find("express"), std::string::npos) << message;
    EXPECT_NE(
        message_of([&] {
            topo::reduce_scatter_output_topology(collapsed_label(mesh, TopoShard{3}), 0, mesh, kRank, 3);
        }).find("express"),
        std::string::npos);
    EXPECT_NE(
        message_of([&] {
            topo::reduce_scatter_output_topology(
                collapsed_label(MeshShape(2, 2, 2), TopoShard{3}), 1, MeshShape(2, 2, 2), kRank, 3);
        }).find("express"),
        std::string::npos);

    // The scattered axis held a different Shard while another axis shards d: two dims on one axis at once.
    EXPECT_NE(
        message_of([&] {
            topo::reduce_scatter_output_topology(nd_label(mesh, {TopoShard{3}, TopoShard{2}}), 1, mesh, kRank, 3);
        }).find("not expressible"),
        std::string::npos);
    // Any other Shard on any axis blocks the collapse.
    EXPECT_NE(
        message_of([&] {
            const MeshShape cube(2, 2, 2);
            topo::reduce_scatter_output_topology(
                nd_label(cube, {TopoShard{3}, TopoShard{2}, TopoReplicate{}}), 2, cube, kRank, 3);
        }).find("express"),
        std::string::npos);
}

TEST(CclTopologyUtils, ReduceScatterWholeMeshIsTheCollapsedLabelOverTheInputCoordinates) {
    StrictCclTopologyScope strict(true);
    for (const auto& mesh : all_meshes()) {
        // Collapsed input: piece i lands on ring rank i, the label's own order.
        EXPECT_EQ(
            topo::reduce_scatter_output_topology(collapsed_label(mesh, TopoReplicate{}), std::nullopt, mesh, kRank, 3),
            collapsed_label(mesh, TopoShard{3}))
            << mesh;
        EXPECT_EQ(
            topo::reduce_scatter_output_topology(collapsed_label(mesh, TopoShard{3}), std::nullopt, mesh, kRank, -1),
            collapsed_label(mesh, TopoShard{3}))
            << mesh;

        // N-D input, Replicate or Shard{d} everywhere: the whole-mesh ring walks the coordinates in order, so the
        // result is the collapsed label over the input's coordinates -- on a line too (mesh_partition precedent).
        std::vector<TopoPlacement> all_replicate(mesh.dims(), TopoReplicate{});
        auto reversed = row_major_coords(mesh);
        std::reverse(reversed.begin(), reversed.end());
        const auto out =
            topo::reduce_scatter_output_topology(nd_label(mesh, all_replicate, reversed), std::nullopt, mesh, kRank, 3);
        ASSERT_TRUE(out.has_value()) << mesh;
        EXPECT_EQ(*out, TensorTopology(MeshShape(static_cast<uint32_t>(mesh.mesh_size())), {TopoShard{3}}, reversed))
            << mesh;
    }

    // Another dim still sharded on a non-trivial axis sits next to the new piece: not expressible.
    const MeshShape mesh(2, 4);
    const auto message = message_of([&] {
        topo::reduce_scatter_output_topology(
            nd_label(mesh, {TopoShard{2}, TopoReplicate{}}), std::nullopt, mesh, kRank, 3);
    });
    EXPECT_NE(message.find("not expressible"), std::string::npos) << message;
    // ... but a Shard of the scattered dim composes, and a Shard on a size-1 axis is the whole extent.
    EXPECT_EQ(
        topo::reduce_scatter_output_topology(
            nd_label(mesh, {TopoShard{3}, TopoReplicate{}}), std::nullopt, mesh, kRank, 3),
        collapsed_label(mesh, TopoShard{3}));
    const MeshShape line(1, 8);
    EXPECT_EQ(
        topo::reduce_scatter_output_topology(
            nd_label(line, {TopoShard{2}, TopoReplicate{}}), std::nullopt, line, kRank, 3),
        collapsed_label(line, TopoShard{3}));
}

TEST(CclTopologyUtils, MeshPartitionAndAllToAllShareTheReduceScatterLabel) {
    const MeshShape mesh(2, 4);
    const auto in = nd_label(mesh, {TopoShard{2}, TopoReplicate{}});
    const auto expected = topo::reduce_scatter_output_topology(in, 1, mesh, kRank, 3);
    EXPECT_EQ(topo::mesh_partition_output_topology(in, 1, mesh, kRank, 3), expected);
    EXPECT_EQ(topo::all_to_all_output_topology(in, 1, mesh, kRank, 3), expected);
}

TEST(CclTopologyUtils, StrictModeThrowsWhereWarnOnlyReturnsNullopt) {
    const MeshShape mesh(2, 4);
    auto coords = row_major_coords(mesh);
    coords.erase(coords.begin() + 4, coords.end());  // MeshCoordinate has no default ctor: no resize()
    const TensorTopology fewer_shards(MeshShape(4), {TopoShard{3}}, coords);

    {
        StrictCclTopologyScope strict(false);
        EXPECT_FALSE(topo::all_gather_output_topology(fewer_shards, 1, mesh, kRank, 3).has_value());
        EXPECT_FALSE(topo::reduce_scatter_output_topology(fewer_shards, 1, mesh, kRank, 3).has_value());
        EXPECT_FALSE(topo::all_reduce_output_topology(fewer_shards, 1, mesh, kRank).has_value());
    }
    StrictCclTopologyScope strict(true);
    EXPECT_THROW(topo::all_gather_output_topology(fewer_shards, 1, mesh, kRank, 3), std::runtime_error);
    EXPECT_THROW(topo::reduce_scatter_output_topology(fewer_shards, 1, mesh, kRank, 3), std::runtime_error);
    const auto message = message_of([&] { topo::all_reduce_output_topology(fewer_shards, 1, mesh, kRank); });
    EXPECT_NE(message.find("row-major"), std::string::npos) << message;
}

// ---------------------------------------------------------------------------------------------------------------------
// The topology-level overload of the framework's union default (compute_output_placements_and_shape). The fused
// collective + matmul ops label their matmul output with the union of the collective's result label and the weight
// (and bias) labels through this overload, so its rule has to be the one launch() applies to plain ops: first Shard
// on a mesh axis wins, a tensor dim already claimed by another axis reads Replicate, lower-rank sharded labels are
// dropped, the distribution shape is the per-axis maximum.
// ---------------------------------------------------------------------------------------------------------------------

namespace {

using UnionPlacements = std::vector<TopoPlacement>;

std::pair<UnionPlacements, MeshShape> union_of(const std::vector<TensorTopology>& labels) {
    std::vector<std::reference_wrapper<const TensorTopology>> refs(labels.begin(), labels.end());
    auto [placements, shape] = ttnn::device_operation::detail::compute_output_placements_and_shape(refs);
    return {UnionPlacements(placements.begin(), placements.end()), shape};
}

}  // namespace

TEST(TopologyUnion, FirstShardOnAnAxisWinsAndAClaimedDimReadsReplicate) {
    const MeshShape mesh(2, 4);

    // A reduce_scatter result [Shard{2}, Shard{3}] with a weight [Replicate, Shard{3}]: the weight's Shard{3} is the
    // dim axis 1 already holds, so nothing changes.
    {
        const auto [placements, shape] =
            union_of({nd_label(mesh, {TopoShard{2}, TopoShard{3}}), nd_label(mesh, {TopoReplicate{}, TopoShard{3}})});
        EXPECT_EQ(shape, mesh);
        EXPECT_EQ(placements, (UnionPlacements{TopoShard{2}, TopoShard{3}}));
    }
    // Two labels sharding different dims on the same axis: the earliest-seen Shard is kept; an axis only the second
    // label shards takes that label's Shard.
    {
        const auto [placements, shape] =
            union_of({nd_label(mesh, {TopoShard{2}, TopoReplicate{}}), nd_label(mesh, {TopoShard{3}, TopoShard{1}})});
        EXPECT_EQ(shape, mesh);
        EXPECT_EQ(placements, (UnionPlacements{TopoShard{2}, TopoShard{1}}));
    }
    // A dim already claimed on one axis is not sharded again on another: the second label's Shard{3} on axis 0 is
    // ignored because axis 1 already shards dim 3, so axis 0 stays Replicate.
    {
        const auto [placements, shape] = union_of(
            {nd_label(mesh, {TopoReplicate{}, TopoShard{3}}), nd_label(mesh, {TopoShard{3}, TopoReplicate{}})});
        EXPECT_EQ(shape, mesh);
        EXPECT_EQ(placements, (UnionPlacements{TopoReplicate{}, TopoShard{3}}));
    }
}

TEST(TopologyUnion, LowerRankShardedLabelsAreDroppedAndStridesAreTheMaximum) {
    const MeshShape mesh(2, 4);

    // An N-D all_gather result with a collapsed 1-D weight label: the collapsed label has the lower distribution
    // rank, so it does not contribute (the union default drops it the same way).
    {
        const auto [placements, shape] =
            union_of({nd_label(mesh, {TopoShard{2}, TopoReplicate{}}), collapsed_label(mesh, TopoShard{3})});
        EXPECT_EQ(shape, mesh);
        EXPECT_EQ(placements, (UnionPlacements{TopoShard{2}, TopoReplicate{}}));
    }
    // A fully-replicated label never decides the rank while something is sharded, whichever order they come in.
    {
        const auto [placements, shape] =
            union_of({collapsed_label(mesh, TopoReplicate{}), nd_label(mesh, {TopoShard{3}, TopoReplicate{}})});
        EXPECT_EQ(shape, mesh);
        EXPECT_EQ(placements, (UnionPlacements{TopoShard{3}, TopoReplicate{}}));
    }
    // Nothing sharded: the first label's rank, all Replicate.
    {
        const auto [placements, shape] =
            union_of({collapsed_label(mesh, TopoReplicate{}), nd_label(mesh, {TopoReplicate{}, TopoReplicate{}})});
        EXPECT_EQ(shape, MeshShape(8));
        EXPECT_EQ(placements, (UnionPlacements{TopoReplicate{}}));
    }
    // Same rank, different distribution shapes: the result's shape is the per-axis maximum.
    {
        const auto [placements, shape] = union_of(
            {nd_label(MeshShape(1, 8), {TopoReplicate{}, TopoShard{3}}),
             nd_label(MeshShape(2, 4), {TopoShard{2}, TopoReplicate{}})});
        EXPECT_EQ(shape, MeshShape(2, 8));
        EXPECT_EQ(placements, (UnionPlacements{TopoShard{2}, TopoShard{3}}));
    }
}
