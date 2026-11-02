// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
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
#include "ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/all_gather_concat_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/all_gather_minimal_matmul_async_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/minimal_matmul_strided_reduce_scatter_async/device/minimal_matmul_strided_reduce_scatter_async_op.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/experimental/ccl/strided_all_gather_minimal_matmul_async/device/strided_all_gather_minimal_matmul_async_op.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include <tt-umd/types/xy_pair.hpp>
#include <tt-umd/types/arch.hpp>
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
        std::forward<F>(f)();
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

// mesh_partition of tensor dim 3, as the op's hook calls the helper (the MeshPartition cases below).
std::optional<TensorTopology> partition_dim3(
    const TensorTopology& input, const MeshShape& mesh, std::optional<uint32_t> cluster_axis) {
    return topo::mesh_partition_output_topology(input, cluster_axis, mesh, kRank, /*out_dim=*/3);
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

TEST(CclTopologyUtils, UncollapseRefusesAnNDLabelWhoseAxesDoNotAlignWithTheMesh) {
    const MeshShape mesh(2, 4);
    std::string reason;

    // A (4, 2) mapper on a (2, 4) mesh does not fit, so the mapper distributes in row-major order over the mesh
    // (distribution_mode.cpp): the label's axis 0 (four chunks) is not the mesh's axis 0 (two rows). Same for (1, 8).
    const TensorTopology remapped(MeshShape(4, 2), {TopoShard{2}, TopoReplicate{}}, row_major_coords(mesh));
    EXPECT_FALSE(topo::uncollapse_placements(remapped, mesh, &reason).has_value());
    EXPECT_NE(reason.find("align"), std::string::npos) << reason;
    const TensorTopology flattened(MeshShape(1, 8), {TopoReplicate{}, TopoShard{3}}, row_major_coords(mesh));
    EXPECT_FALSE(topo::uncollapse_placements(flattened, mesh, &reason).has_value());
    EXPECT_NE(reason.find("align"), std::string::npos) << reason;

    // Mesh-shaped, but the coordinates are not the row-major walk of the mesh.
    auto reversed = row_major_coords(mesh);
    std::reverse(reversed.begin(), reversed.end());
    EXPECT_FALSE(topo::uncollapse_placements(nd_label(mesh, {TopoShard{2}, TopoReplicate{}}, reversed), mesh, &reason)
                     .has_value());
    EXPECT_NE(reason.find("align"), std::string::npos) << reason;

    // A block of the mesh whose axes are the mesh's (an N-D mapper with an offset) is accepted verbatim.
    const std::vector<MeshCoordinate> second_row{
        MeshCoordinate(1, 0), MeshCoordinate(1, 1), MeshCoordinate(1, 2), MeshCoordinate(1, 3)};
    const std::vector<TopoPlacement> placements{TopoReplicate{}, TopoShard{3}};
    const auto block = topo::uncollapse_placements(nd_label(MeshShape(1, 4), placements, second_row), mesh, &reason);
    ASSERT_TRUE(block.has_value()) << reason;
    EXPECT_EQ(std::vector<TopoPlacement>(block->begin(), block->end()), placements);
    const std::vector<MeshCoordinate> right_half{
        MeshCoordinate(0, 2), MeshCoordinate(0, 3), MeshCoordinate(1, 2), MeshCoordinate(1, 3)};
    EXPECT_TRUE(
        topo::uncollapse_placements(nd_label(MeshShape(2, 2), {TopoShard{3}, TopoReplicate{}}, right_half), mesh)
            .has_value());

    // The refusal reaches the ops like any other: nullopt in warn-only mode, TT_FATAL in strict mode. Before this
    // check the (4, 2) label came back [Replicate, Replicate] from a gather along mesh axis 0, which combines chunks
    // {0, 2} in columns 0-1 and {1, 3} in columns 2-3: an over-claimed Replicate.
    {
        StrictCclTopologyScope warn_only(false);
        EXPECT_FALSE(topo::all_gather_output_topology(remapped, 0, mesh, kRank, 2).has_value());
    }
    StrictCclTopologyScope strict(true);
    EXPECT_NE(
        message_of([&] { topo::all_gather_output_topology(remapped, 0, mesh, kRank, 2); }).find("align"),
        std::string::npos);
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

    // Only the gathered axis is replicated: another axis sharding the gathered dim keeps its Shard, since its pieces
    // are still distinct after the gather. The old prim all_gather rule replicated every axis whose Shard dim equalled
    // the gather dim.
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
    // which no label expresses (finalise collapses a repeated dim only when every non-trivial axis shards it):
    // warn-only gives nullopt, strict throws.
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

    // The interleave refusal (require_contiguous_gather): gathering dim 3 along axis 0 interleaves the pieces axis 1
    // keeps apart.
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

TEST(CclTopologyUtils, AllGatherOfAnNDLabelThatShardsTheGatheredDimTwiceFollowsTheCollapsedRules) {
    // No mapper builds [Shard{d}, Shard{d}] (chunk_ndim requires unique dims), but update_tensor_topology accepts it.
    // The helper reads it the way `finalise` does -- row-major hierarchical sharding, like a collapsed Shard{d} -- so
    // gathering d along the inner axis is honest and along the outer axis interleaves.
    const MeshShape mesh(2, 4);
    const auto twice = nd_label(mesh, {TopoShard{3}, TopoShard{3}});
    StrictCclTopologyScope strict(true);
    EXPECT_EQ(
        topo::all_gather_output_topology(twice, 1, mesh, kRank, 3), nd_label(mesh, {TopoShard{3}, TopoReplicate{}}));
    EXPECT_NE(
        message_of([&] { topo::all_gather_output_topology(twice, 0, mesh, kRank, 3); }).find("interleave"),
        std::string::npos);
    // Gathering another dim moves no dim-3 piece, and all_reduce concatenates nothing: honest along the outer axis.
    EXPECT_EQ(
        topo::all_gather_output_topology(twice, 0, mesh, kRank, 2), nd_label(mesh, {TopoReplicate{}, TopoShard{3}}));
    EXPECT_EQ(topo::all_reduce_output_topology(twice, 0, mesh, kRank), nd_label(mesh, {TopoReplicate{}, TopoShard{3}}));
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

    // Only normalised dims are written into a placement.
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
    // The outer axis holds the coarse chunks and the scattered (inner) axis splits each into fine pieces, so device
    // (r, c) holds piece r * C + c: row-major hierarchical sharding, i.e. the collapsed label over the input's
    // coordinates.
    const MeshShape mesh(2, 4);
    const auto outer_sharded = nd_label(mesh, {TopoShard{3}, TopoReplicate{}});
    {
        StrictCclTopologyScope strict(true);
        const auto out = topo::reduce_scatter_output_topology(outer_sharded, 1, mesh, kRank, 3);
        ASSERT_TRUE(out.has_value());
        EXPECT_EQ(*out, collapsed_label(mesh, TopoShard{3}));
        EXPECT_EQ(out->mesh_coords(), outer_sharded.mesh_coords());
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
    // emitted. Warn-only: nullopt (the op keeps the input's label); strict: TT_FATAL.
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

    // The scattered axis held a different Shard while the outer axis shards d. The reduction sums the row, which
    // consumes the dim-2 pieces (the same overwrite as [Replicate, Shard{2}] above), so the rows' pieces compose: the
    // collapsed label. A partition keeps each device's dim-2 piece next to its new dim-3 piece: two dims on one axis.
    const auto held_other = nd_label(mesh, {TopoShard{3}, TopoShard{2}});
    EXPECT_EQ(topo::reduce_scatter_output_topology(held_other, 1, mesh, kRank, 3), collapsed_label(mesh, TopoShard{3}));
    EXPECT_NE(
        message_of([&] {
            topo::mesh_partition_output_topology(held_other, 1, mesh, kRank, 3);
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

        // N-D input, Replicate everywhere: the whole-mesh ring walks the coordinates in order, so the result is the
        // collapsed label over the input's coordinates -- on a line too (mesh_partition precedent).
        std::vector<TopoPlacement> all_replicate(mesh.dims(), TopoReplicate{});
        EXPECT_EQ(
            topo::reduce_scatter_output_topology(nd_label(mesh, all_replicate), std::nullopt, mesh, kRank, 3),
            collapsed_label(mesh, TopoShard{3}))
            << mesh;
    }

    // A tensor on one row of a 2x4 mesh (an N-D mapper with an offset): a whole-mesh scatter spans every device of
    // the mesh, four of which the label does not describe, so it is refused (a tensor that really lives on four
    // devices is one on a 1x4 sub-mesh device, whose own shape is what its hooks pass).
    const MeshShape mesh(2, 4);
    const std::vector<MeshCoordinate> second_row{
        MeshCoordinate(1, 0), MeshCoordinate(1, 1), MeshCoordinate(1, 2), MeshCoordinate(1, 3)};
    EXPECT_NE(
        message_of([&] {
            topo::reduce_scatter_output_topology(
                nd_label(MeshShape(1, 4), {TopoReplicate{}, TopoReplicate{}}, second_row),
                std::nullopt,
                mesh,
                kRank,
                3);
        }).find("covers"),
        std::string::npos);

    // Another dim sharded on a non-trivial axis: the whole-mesh reduction sums every device's tensor into the piece
    // each receives, so the dim-2 pieces are consumed and the collapsed label is honest (the old hook emitted
    // [Shard{3}, Shard{3}] here, which concat_ndim cannot compose). A partition keeps each device's dim-2 piece next
    // to its new dim-3 piece, which no label states: still refused for mesh_partition / all_to_all.
    const auto other_dim = nd_label(mesh, {TopoShard{2}, TopoReplicate{}});
    EXPECT_EQ(
        topo::reduce_scatter_output_topology(other_dim, std::nullopt, mesh, kRank, 3),
        collapsed_label(mesh, TopoShard{3}));
    const auto message =
        message_of([&] { topo::mesh_partition_output_topology(other_dim, std::nullopt, mesh, kRank, 3); });
    EXPECT_NE(message.find("not expressible"), std::string::npos) << message;
    // A stale Shard dim left by a rank-changing op (#52331), positive or negative, matches nothing: the reduction
    // consumes it like any other placement, while the whole-mesh partition reads a Shard that is not of the scattered
    // dim on a non-trivial axis as a Shard of another dim and refuses.
    for (const auto& stale : {TopoShard{7}, TopoShard{-5}}) {
        EXPECT_EQ(
            topo::reduce_scatter_output_topology(
                nd_label(mesh, {stale, TopoReplicate{}}), std::nullopt, mesh, kRank, 3),
            collapsed_label(mesh, TopoShard{3}))
            << stale.dim;
        EXPECT_NE(
            message_of([&] {
                topo::mesh_partition_output_topology(
                    nd_label(mesh, {stale, TopoReplicate{}}), std::nullopt, mesh, kRank, 3);
            }).find("not expressible"),
            std::string::npos)
            << stale.dim;
    }
    // A Shard of the scattered dim composes for both, and a Shard on a size-1 axis is the whole extent.
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

TEST(CclTopologyUtils, MeshPartitionAndAllToAllShareTheReduceScatterLabelExceptWhereNothingIsSummed) {
    const MeshShape mesh(2, 4);
    const auto in = nd_label(mesh, {TopoShard{2}, TopoReplicate{}});
    const auto expected = topo::reduce_scatter_output_topology(in, 1, mesh, kRank, 3);
    ASSERT_TRUE(expected.has_value());
    EXPECT_EQ(topo::mesh_partition_output_topology(in, 1, mesh, kRank, 3), expected);
    EXPECT_EQ(topo::all_to_all_output_topology(in, 1, mesh, kRank, 3), expected);
    EXPECT_EQ(
        topo::all_to_all_output_topology(collapsed_label(mesh, TopoShard{3}), 1, mesh, kRank, 3),
        topo::reduce_scatter_output_topology(collapsed_label(mesh, TopoShard{3}), 1, mesh, kRank, 3));

    // Where the reduction consumes a Shard of another dim, the partition keeps it next to the new piece: refused.
    StrictCclTopologyScope warn_only(false);
    EXPECT_TRUE(topo::reduce_scatter_output_topology(in, std::nullopt, mesh, kRank, 3).has_value());
    EXPECT_FALSE(topo::mesh_partition_output_topology(in, std::nullopt, mesh, kRank, 3).has_value());
    EXPECT_FALSE(topo::all_to_all_output_topology(in, std::nullopt, mesh, kRank, 3).has_value());
    const auto held_other = nd_label(mesh, {TopoShard{3}, TopoShard{2}});
    EXPECT_TRUE(topo::reduce_scatter_output_topology(held_other, 1, mesh, kRank, 3).has_value());
    EXPECT_FALSE(topo::mesh_partition_output_topology(held_other, 1, mesh, kRank, 3).has_value());
    EXPECT_FALSE(topo::all_to_all_output_topology(held_other, 1, mesh, kRank, 3).has_value());
    // ... also when no other axis shards the dim: reduce_scatter overwrites the scattered axis's Shard{2} (the sum
    // consumed those pieces), a partition would leave device c with chunk c of dim 2 and piece c of dim 3.
    const auto held_other_only = nd_label(mesh, {TopoReplicate{}, TopoShard{2}});
    EXPECT_EQ(
        topo::reduce_scatter_output_topology(held_other_only, 1, mesh, kRank, 3),
        nd_label(mesh, {TopoReplicate{}, TopoShard{3}}));
    EXPECT_FALSE(topo::mesh_partition_output_topology(held_other_only, 1, mesh, kRank, 3).has_value());
    EXPECT_FALSE(topo::all_to_all_output_topology(held_other_only, 1, mesh, kRank, 3).has_value());
}

TEST(CclTopologyUtils, WholeMeshPartitionOfACollapsedShardOfAnotherDimIsRefused) {
    // The collapsed form of the other-dim refusal above: {N}, [Shard{2}] partitioned on dim 3 leaves device k with
    // chunk k of dim 2 next to piece k of dim 3, which no label states, so the partition kinds refuse before the
    // collapsed early return; the reduction consumes the chunks and is labelled. Same-dim and Replicate collapsed
    // inputs compose for every kind.
    for (const auto& mesh : {MeshShape(1, 8), MeshShape(2, 4)}) {
        const auto other_dim = collapsed_label(mesh, TopoShard{2});
        EXPECT_EQ(
            topo::reduce_scatter_output_topology(other_dim, std::nullopt, mesh, kRank, 3),
            collapsed_label(mesh, TopoShard{3}))
            << mesh;
        {
            StrictCclTopologyScope warn_only(false);
            EXPECT_FALSE(topo::mesh_partition_output_topology(other_dim, std::nullopt, mesh, kRank, 3).has_value())
                << mesh;
            EXPECT_FALSE(topo::all_to_all_output_topology(other_dim, std::nullopt, mesh, kRank, 3).has_value()) << mesh;
        }
        StrictCclTopologyScope strict(true);
        EXPECT_NE(
            message_of([&] {
                topo::mesh_partition_output_topology(other_dim, std::nullopt, mesh, kRank, 3);
            }).find("not expressible"),
            std::string::npos)
            << mesh;
        EXPECT_EQ(
            topo::mesh_partition_output_topology(collapsed_label(mesh, TopoShard{3}), std::nullopt, mesh, kRank, 3),
            collapsed_label(mesh, TopoShard{3}))
            << mesh;
        EXPECT_EQ(
            topo::all_to_all_output_topology(collapsed_label(mesh, TopoReplicate{}), std::nullopt, mesh, kRank, 3),
            collapsed_label(mesh, TopoShard{3}))
            << mesh;
    }
}

TEST(CclTopologyUtils, WholeMeshScatterOfARowMajorReshapeIsExact) {
    // A 2-D mapper whose mesh_shape_override does not fit the mesh ({1,8} or {4,2} over a 2x4) lays the tensor out
    // row-major over every device: its axes are not the mesh's, so a cluster-axis op cannot address them and is
    // refused, but a whole-mesh scatter addresses no axis -- ring rank i gets piece i -- and the collapsed label over
    // the same coordinates is exact. A partition
    // still reads the other-dim restriction off the label's own axes.
    const MeshShape mesh(2, 4);
    const auto all = row_major_coords(mesh);
    StrictCclTopologyScope warn_only(false);
    for (const auto& reshape : {MeshShape(1, 8), MeshShape(4, 2)}) {
        const auto same_dim = nd_label(reshape, {TopoReplicate{}, TopoShard{3}}, all);
        EXPECT_EQ(
            topo::reduce_scatter_output_topology(same_dim, std::nullopt, mesh, kRank, 3),
            collapsed_label(mesh, TopoShard{3}))
            << reshape;
        EXPECT_EQ(
            topo::mesh_partition_output_topology(same_dim, std::nullopt, mesh, kRank, 3),
            collapsed_label(mesh, TopoShard{3}))
            << reshape;
        EXPECT_FALSE(topo::reduce_scatter_output_topology(same_dim, 1, mesh, kRank, 3).has_value()) << reshape;
        EXPECT_FALSE(topo::mesh_partition_output_topology(same_dim, 1, mesh, kRank, 3).has_value()) << reshape;

        const auto other_dim = nd_label(reshape, {TopoReplicate{}, TopoShard{2}}, all);
        EXPECT_EQ(
            topo::reduce_scatter_output_topology(other_dim, std::nullopt, mesh, kRank, 3),
            collapsed_label(mesh, TopoShard{3}))
            << reshape;
        EXPECT_FALSE(topo::mesh_partition_output_topology(other_dim, std::nullopt, mesh, kRank, 3).has_value())
            << reshape;
    }
}

TEST(CclTopologyUtils, OpsRefuseALabelThatDoesNotCoverTheDevicesTheyReach) {
    // Coverage: a (1, 4) row block on a 2x4 is addressed along axis 1,
    // which it spans, but not along axis 0 (extent 1 of 2: the groups pair each of its devices with one the label
    // does not describe) nor over the whole mesh (eight devices, four described). Both the host form of the label
    // (the block's own coordinates) and the device form (the storage's, every device) are judged by the shape.
    const MeshShape mesh(2, 4);
    const auto all = row_major_coords(mesh);
    std::vector<MeshCoordinate> row0(all.begin(), all.begin() + 4);
    StrictCclTopologyScope warn_only(false);
    for (const auto& coords : {row0, all}) {
        const auto row = nd_label(MeshShape(1, 4), {TopoReplicate{}, TopoReplicate{}}, coords);
        // The spanned axis: labelled within the block, coordinates kept.
        EXPECT_EQ(
            topo::reduce_scatter_output_topology(row, 1, mesh, kRank, 3),
            nd_label(MeshShape(1, 4), {TopoReplicate{}, TopoShard{3}}, coords));
        EXPECT_EQ(
            topo::mesh_partition_output_topology(row, 1, mesh, kRank, 3),
            nd_label(MeshShape(1, 4), {TopoReplicate{}, TopoShard{3}}, coords));
        EXPECT_EQ(
            topo::all_gather_output_topology(row, 1, mesh, kRank, 3),
            nd_label(MeshShape(1, 4), {TopoReplicate{}, TopoReplicate{}}, coords));
        // The axis the block does not span, and the whole mesh: refused for every family.
        for (const auto& axis : {std::optional<uint32_t>{0}, std::optional<uint32_t>{}}) {
            EXPECT_FALSE(topo::reduce_scatter_output_topology(row, axis, mesh, kRank, 3).has_value());
            EXPECT_FALSE(topo::mesh_partition_output_topology(row, axis, mesh, kRank, 3).has_value());
            EXPECT_FALSE(topo::all_to_all_output_topology(row, axis, mesh, kRank, 3).has_value());
            EXPECT_FALSE(topo::all_gather_output_topology(row, axis, mesh, kRank, 3).has_value());
            EXPECT_FALSE(topo::all_broadcast_output_topology(row, axis, mesh, kRank).has_value());
            EXPECT_FALSE(topo::all_reduce_output_topology(row, axis, mesh, kRank).has_value());
        }
    }
    // A 2x2 block at an offset spans axis 0 but not axis 1.
    const std::vector<MeshCoordinate> block{
        MeshCoordinate(0, 2), MeshCoordinate(0, 3), MeshCoordinate(1, 2), MeshCoordinate(1, 3)};
    const auto square = nd_label(MeshShape(2, 2), {TopoReplicate{}, TopoShard{2}}, block);
    EXPECT_EQ(
        topo::mesh_partition_output_topology(square, 0, mesh, kRank, 3),
        nd_label(MeshShape(2, 2), {TopoShard{3}, TopoShard{2}}, block));
    EXPECT_FALSE(topo::mesh_partition_output_topology(square, 1, mesh, kRank, 3).has_value());
    StrictCclTopologyScope strict(true);
    EXPECT_NE(
        message_of([&] { topo::all_gather_output_topology(square, 1, mesh, kRank, 3); }).find("covers"),
        std::string::npos);
}

// mesh_partition's own cases: the scatter rule's partition kind (a Shard of another dim is kept, not consumed), as the
// op's hook calls it for tensor dim 3; warn-only, so a refusal is a nullopt. The device twins are the mesh_partition
// section of tests/nightly/t3000/ccl/test_ccl_topology_labels.py.

// An N-D label takes Shard{dim} on the partitioned axis and keeps the input's placement elsewhere -- unless the
// partitioned axis itself held a Shard of another dim: each device would then keep its chunk of that dim next to its
// new piece of dim, two dims on one axis, which no label states, so that case is refused.
TEST(CclTopologyUtils, MeshPartitionNdLabelTakesShardDimOnThePartitionedAxisUnlessItHeldAnotherShard) {
    StrictCclTopologyScope warn_only(false);
    const MeshShape mesh(2, 4);
    const auto input = nd_label(mesh, {TopoShard{2}, TopoReplicate{}});
    EXPECT_EQ(partition_dim3(input, mesh, 1), nd_label(mesh, {TopoShard{2}, TopoShard{3}}));
    EXPECT_FALSE(partition_dim3(input, mesh, 0).has_value());  // axis 0 holds Shard{2}
}

// One chunk along a size-1 axis is the whole extent.
TEST(CclTopologyUtils, MeshPartitionSameDimOnASizeOneAxisBecomesReplicate) {
    StrictCclTopologyScope warn_only(false);
    const MeshShape mesh(1, 8);
    EXPECT_EQ(
        partition_dim3(nd_label(mesh, {TopoShard{3}, TopoReplicate{}}), mesh, 1),
        nd_label(mesh, {TopoReplicate{}, TopoShard{3}}));
}

// An outer axis already holding Shard{dim} composes with the partitioned axis into row-major hierarchical sharding,
// which the collapsed label states exactly when the partitioned axis held Replicate; when it already held Shard{dim}
// (a label the mappers refuse to build) the collapsed label describes the output bytes, as for a whole-mesh partition
// of a sharded input.
TEST(CclTopologyUtils, MeshPartitionSameDimOnAnOuterAxisCollapsesRowMajor) {
    StrictCclTopologyScope warn_only(false);
    const MeshShape mesh(2, 4);
    for (const TopoPlacement& partitioned : {TopoPlacement{TopoReplicate{}}, TopoPlacement{TopoShard{3}}}) {
        const auto input = nd_label(mesh, {TopoShard{3}, partitioned});
        EXPECT_EQ(partition_dim3(input, mesh, 1), collapsed_label(mesh, TopoShard{3})) << input;
    }
    // Shard dims in the label are compared normalised: -1 is dim 3 at rank 4.
    const auto negative = partition_dim3(nd_label(mesh, {TopoShard{-1}, TopoReplicate{}}), mesh, 1);
    ASSERT_TRUE(negative.has_value());
    EXPECT_EQ(negative->distribution_shape(), MeshShape(8));
    // An out-of-range Shard dim (left behind by a rank-changing op) does not count as sharding dim.
    EXPECT_EQ(
        partition_dim3(nd_label(mesh, {TopoShard{7}, TopoReplicate{}}), mesh, 1),
        nd_label(mesh, {TopoShard{7}, TopoShard{3}}));
}

// Everything else that would shard dim on two axes has no label.
TEST(CclTopologyUtils, MeshPartitionSameDimOnAnInnerAxisOrAnotherShardOnThePartitionedAxisIsRefused) {
    StrictCclTopologyScope warn_only(false);
    const MeshShape mesh(2, 4);
    // Columns hold fine slices of dim; partitioning across rows cuts them again: column-major, not expressible.
    EXPECT_FALSE(partition_dim3(nd_label(mesh, {TopoReplicate{}, TopoShard{3}}), mesh, 0).has_value());
    // The outer axis shards dim but the partitioned axis held a different Shard.
    EXPECT_FALSE(partition_dim3(nd_label(mesh, {TopoShard{3}, TopoShard{2}}), mesh, 1).has_value());
    // A third non-trivial axis would make the collapsed label over-claim N distinct slices.
    const MeshShape mesh3(2, 2, 2);
    EXPECT_FALSE(
        partition_dim3(nd_label(mesh3, {TopoShard{3}, TopoReplicate{}, TopoReplicate{}}), mesh3, 1).has_value());
}

// A whole-mesh partition of a replicated input is {N},[Shard{dim}] in row-major device order, whether the input
// label is N-D or collapsed.
TEST(CclTopologyUtils, MeshPartitionWholeMeshCollapsesAReplicatedInput) {
    StrictCclTopologyScope warn_only(false);
    for (const MeshShape& mesh : {MeshShape(1, 8), MeshShape(2, 4), MeshShape(8, 4)}) {
        const auto expected = collapsed_label(mesh, TopoShard{3});
        EXPECT_EQ(partition_dim3(nd_label(mesh, {TopoReplicate{}, TopoReplicate{}}), mesh, std::nullopt), expected)
            << mesh;
        EXPECT_EQ(partition_dim3(collapsed_label(mesh, TopoReplicate{}), mesh, std::nullopt), expected) << mesh;
    }
}

// A whole-mesh partition of an input already sharded: Shard{dim} on a non-trivial axis is overwritten (the label then
// describes the output bytes: device k holds chunk k of its own input shard, not a re-slicing of the input's global
// view); a Shard of another dim on a non-trivial axis has no label, N-D or collapsed; a Shard on a size-1 axis is the
// whole extent and does not block.
TEST(CclTopologyUtils, MeshPartitionWholeMeshOverwritesShardDimButNotAnotherDim) {
    StrictCclTopologyScope warn_only(false);
    const MeshShape mesh(2, 4);
    EXPECT_EQ(
        partition_dim3(nd_label(mesh, {TopoShard{3}, TopoReplicate{}}), mesh, std::nullopt),
        collapsed_label(mesh, TopoShard{3}));
    EXPECT_FALSE(partition_dim3(nd_label(mesh, {TopoShard{2}, TopoReplicate{}}), mesh, std::nullopt).has_value());
    EXPECT_FALSE(partition_dim3(collapsed_label(mesh, TopoShard{2}), mesh, std::nullopt).has_value());
    const MeshShape line(1, 8);
    EXPECT_EQ(
        partition_dim3(nd_label(line, {TopoShard{2}, TopoReplicate{}}), line, std::nullopt),
        collapsed_label(line, TopoShard{3}));
}

// On a line the collapsed label's only axis is the partitioned axis; whatever it held is overwritten.
TEST(CclTopologyUtils, MeshPartitionCollapsedLabelOnALineTakesShardDimWhateverItHeld) {
    StrictCclTopologyScope warn_only(false);
    const MeshShape mesh(1, 8);
    for (const TopoPlacement& held :
         {TopoPlacement{TopoReplicate{}}, TopoPlacement{TopoShard{1}}, TopoPlacement{TopoShard{3}}}) {
        const auto input = collapsed_label(mesh, held);
        EXPECT_EQ(partition_dim3(input, mesh, 1), collapsed_label(mesh, TopoShard{3})) << input;
    }
}

// A collapsed Replicate over a multi-axis mesh is spelled out per axis so that one axis can be Shard{dim}.
TEST(CclTopologyUtils, MeshPartitionCollapsedReplicateOnAMultiAxisMeshUncollapses) {
    StrictCclTopologyScope warn_only(false);
    const MeshShape mesh(2, 4);
    const auto input = collapsed_label(mesh, TopoReplicate{});
    EXPECT_EQ(partition_dim3(input, mesh, 0), nd_label(mesh, {TopoShard{3}, TopoReplicate{}}));
    EXPECT_EQ(partition_dim3(input, mesh, 1), nd_label(mesh, {TopoReplicate{}, TopoShard{3}}));
}

// A collapsed Shard over a multi-axis mesh is read per axis as row-major hierarchical sharding of its dim. A Shard
// of another dim is refused along either axis (the partitioned axis holds a piece of two dims). A Shard of dim itself
// collapses again along the inner axis (each device keeps a sub-piece of its piece: N distinct, ordered slices, the
// same spelling as the input) and is refused along the outer axis (column-major).
TEST(CclTopologyUtils, MeshPartitionCollapsedShardOnAMultiAxisMeshIsReadPerAxis) {
    StrictCclTopologyScope warn_only(false);
    const MeshShape mesh(2, 4);
    for (uint32_t axis : {0U, 1U}) {
        EXPECT_FALSE(partition_dim3(collapsed_label(mesh, TopoShard{2}), mesh, axis).has_value()) << "axis " << axis;
    }
    EXPECT_EQ(partition_dim3(collapsed_label(mesh, TopoShard{3}), mesh, 1), collapsed_label(mesh, TopoShard{3}));
    EXPECT_FALSE(partition_dim3(collapsed_label(mesh, TopoShard{3}), mesh, 0).has_value());
}

// A collapsed label has no axes, so one over fewer devices than the mesh says nothing about where its devices sit;
// the fewer-shards label whose N equals the column count merely looks like a line's label.
TEST(CclTopologyUtils, MeshPartitionCollapsedLabelOverFewerDevicesThanTheMeshIsRefusedForEveryClusterAxis) {
    StrictCclTopologyScope warn_only(false);
    const MeshShape mesh(2, 4);
    auto coords = row_major_coords(mesh);
    coords.erase(coords.begin() + 4, coords.end());
    const TensorTopology fewer_shards(MeshShape(4), {TopoShard{2}}, coords);
    for (const auto& axis : {std::optional<uint32_t>{0}, std::optional<uint32_t>{1}, std::optional<uint32_t>{}}) {
        EXPECT_FALSE(partition_dim3(fewer_shards, mesh, axis).has_value())
            << "cluster_axis " << (axis.has_value() ? static_cast<int>(*axis) : -1);
    }
}

// N-D sub-mesh labels (a mesh_shape_override that fits per axis; the coordinates are the block's own): the label is
// kept, and the partition labelled within the block, only when the cluster axis spans the mesh along that axis, so
// that every partition group lies inside the block. Otherwise a group reaches a device the label does not describe.
TEST(CclTopologyUtils, MeshPartitionSubMeshLabelIsPartitionedWithinItsBlockOnlyAlongAFullAxis) {
    StrictCclTopologyScope warn_only(false);
    const MeshShape mesh(2, 4);
    auto row0 = row_major_coords(mesh);
    row0.erase(row0.begin() + 4, row0.end());
    const auto row_block = nd_label(MeshShape(1, 4), {TopoReplicate{}, TopoReplicate{}}, row0);
    EXPECT_EQ(  // 4 == 4
        partition_dim3(row_block, mesh, 1),
        nd_label(MeshShape(1, 4), {TopoReplicate{}, TopoShard{3}}, row0));
    EXPECT_FALSE(partition_dim3(row_block, mesh, 0).has_value());             // 1 < 2
    EXPECT_FALSE(partition_dim3(row_block, mesh, std::nullopt).has_value());  // 4 < 8

    // A {2,2} block at an offset (mesh_offset_override): axis 0 spans the mesh, axis 1 does not.
    const std::vector<MeshCoordinate> block{
        MeshCoordinate(0, 2), MeshCoordinate(0, 3), MeshCoordinate(1, 2), MeshCoordinate(1, 3)};
    const auto square = nd_label(MeshShape(2, 2), {TopoReplicate{}, TopoShard{2}}, block);
    EXPECT_EQ(partition_dim3(square, mesh, 0), nd_label(MeshShape(2, 2), {TopoShard{3}, TopoShard{2}}, block));
    EXPECT_FALSE(partition_dim3(square, mesh, 1).has_value());
}

// Row-major reshapes of the whole mesh ({1,8} or {4,2} over 2x4; a mesh_shape_override that does not fit per axis):
// the coordinates enumerate the mesh row-major, so a whole-mesh partition has its exact collapsed label, but the
// label's axes are not the mesh axes the op partitions along.
TEST(CclTopologyUtils, MeshPartitionRowMajorReshapeOfTheMeshPartitionsAsAWholeOnly) {
    StrictCclTopologyScope warn_only(false);
    const MeshShape mesh(2, 4);
    const auto all = row_major_coords(mesh);
    const auto reshaped = nd_label(MeshShape(1, 8), {TopoReplicate{}, TopoReplicate{}}, all);
    EXPECT_EQ(partition_dim3(reshaped, mesh, std::nullopt), collapsed_label(mesh, TopoShard{3}));
    for (uint32_t axis : {0U, 1U}) {
        EXPECT_FALSE(partition_dim3(reshaped, mesh, axis).has_value()) << "axis " << axis;
    }
    const auto transposed = nd_label(MeshShape(4, 2), {TopoReplicate{}, TopoReplicate{}}, all);
    EXPECT_FALSE(partition_dim3(transposed, mesh, 1).has_value());
}

// The coordinates must agree with the partition, which the factory computes from the device's own coordinate, not
// from the label. The mappers always write agreeing coordinates (SUBMESH: an axis-aligned block; ROW_MAJOR: the
// mesh's row-major enumeration); a label put together by hand with update_tensor_topology need not. Addressing an
// axis requires the whole label to be a row-major block of the mesh; a whole-mesh partition takes the label's
// coordinates as its order (the Tensor overload respells that label over the storage's row-major coordinates, the
// order the factory partitions in).
TEST(CclTopologyUtils, MeshPartitionLabelWhoseCoordinatesDisagreeWithThePartitionIsRefused) {
    StrictCclTopologyScope warn_only(false);
    const MeshShape mesh(2, 4);
    // A block shape over a row of devices: every device has global row 0 and takes chunk 0; a per-axis label would
    // have said tensor row 1 holds chunk 1.
    const std::vector<MeshCoordinate> row{
        MeshCoordinate(0, 0), MeshCoordinate(0, 1), MeshCoordinate(0, 2), MeshCoordinate(0, 3)};
    EXPECT_FALSE(
        partition_dim3(nd_label(MeshShape(2, 2), {TopoReplicate{}, TopoReplicate{}}, row), mesh, 0).has_value());

    // A full-mesh N-D label with two coordinates swapped across rows is not a row-major block: refused along either
    // axis; over the whole mesh the pure rule keeps the label's order (see above).
    auto swapped = row_major_coords(mesh);
    std::swap(swapped[1], swapped[5]);  // (0,1) <-> (1,1)
    const auto permuted_nd = nd_label(mesh, {TopoReplicate{}, TopoReplicate{}}, swapped);
    EXPECT_FALSE(partition_dim3(permuted_nd, mesh, 0).has_value());
    EXPECT_FALSE(partition_dim3(permuted_nd, mesh, 1).has_value());
    EXPECT_EQ(partition_dim3(permuted_nd, mesh, std::nullopt), collapsed_label(mesh, TopoShard{3}, swapped));

    // Collapsed labels: a line with reversed coordinates, the 2x4 with the swap above.
    const MeshShape line(1, 8);
    auto reversed = row_major_coords(line);
    std::reverse(reversed.begin(), reversed.end());
    EXPECT_FALSE(partition_dim3(collapsed_label(line, TopoReplicate{}, reversed), line, 1).has_value());
    const auto permuted_collapsed = collapsed_label(mesh, TopoReplicate{}, swapped);
    EXPECT_FALSE(partition_dim3(permuted_collapsed, mesh, 0).has_value());
    EXPECT_FALSE(partition_dim3(permuted_collapsed, mesh, 1).has_value());
}

// Inputs validation rejects right after the hook, or that carry no placements, leave the input's label in place (the
// hook runs before validation and must not crash first).
TEST(CclTopologyUtils, MeshPartitionOutOfRangeClusterAxisOrEmptyLabelLeavesTheInputLabel) {
    StrictCclTopologyScope warn_only(false);
    const MeshShape mesh(2, 4);
    EXPECT_FALSE(partition_dim3(nd_label(mesh, {TopoReplicate{}, TopoReplicate{}}), mesh, 2).has_value());
    const TensorTopology no_placements(mesh, {}, row_major_coords(mesh));
    EXPECT_FALSE(partition_dim3(no_placements, mesh, 1).has_value());
    EXPECT_FALSE(partition_dim3(no_placements, mesh, std::nullopt).has_value());
}

TEST(CclTopologyUtils, SilenceIntermediateRefusalsReturnsNulloptQuietly) {
    // all_reduce_async labels its result from its own input and runs reduce_scatter / all_gather on intermediates
    // nobody reads; inside the scope their refusals are nullopt without a log or a TT_FATAL, honest labels are
    // unaffected, scopes nest, and strict mode is back the moment the scope ends.
    const MeshShape mesh(2, 4);
    const auto in = collapsed_label(mesh, TopoShard{3});
    StrictCclTopologyScope strict(true);
    {
        const topo::SilenceIntermediateRefusals silence_refusals;
        EXPECT_FALSE(topo::all_gather_output_topology(in, 0, mesh, kRank, 3).has_value());
        EXPECT_FALSE(topo::reduce_scatter_output_topology(in, 0, mesh, kRank, 3).has_value());
        EXPECT_EQ(
            topo::all_gather_output_topology(in, 1, mesh, kRank, 3), nd_label(mesh, {TopoShard{3}, TopoReplicate{}}));
        {
            const topo::SilenceIntermediateRefusals nested;
            EXPECT_FALSE(topo::all_gather_output_topology(in, 0, mesh, kRank, 3).has_value());
        }
        EXPECT_FALSE(topo::all_gather_output_topology(in, 0, mesh, kRank, 3).has_value());
    }
    EXPECT_THROW(topo::all_gather_output_topology(in, 0, mesh, kRank, 3), std::runtime_error);
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

UnionPlacements placements_of(const TensorTopology& label) {
    return UnionPlacements(label.placements().begin(), label.placements().end());
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

// union_topology: the fused hooks' union as a TensorTopology, with one rule beyond the union's -- a sharded first
// operand (the collective's label) fixes the result's rank and sharded operands of another rank are left out.
TEST(TopologyUnion, UnionTopologyKeepsTheRankOfAShardedFirstOperand) {
    const MeshShape mesh(2, 4);
    const auto coords = row_major_coords(mesh);
    const auto union_topology = [&](const std::vector<TensorTopology>& labels) {
        std::vector<std::reference_wrapper<const TensorTopology>> refs(labels.begin(), labels.end());
        return ttnn::operations::ccl::common::union_topology(refs, coords);
    };

    // A whole-mesh reduce_scatter result ({N}, [Shard{3}]: every device holds a distinct piece) with two N-D
    // elementwise (addcmul) operands sharded on dim 3 along axis 1. The union alone keeps only the highest-rank
    // sharded labels, so it would drop the reduce_scatter label and read [Replicate, Shard{3}]: Replicate on an axis
    // whose devices hold different pieces. The first operand's rank wins and the result is the reduce_scatter label
    // over the given coordinates.
    {
        const auto result = union_topology(
            {collapsed_label(mesh, TopoShard{3}),
             nd_label(mesh, {TopoReplicate{}, TopoShard{3}}),
             nd_label(mesh, {TopoReplicate{}, TopoShard{3}})});
        EXPECT_EQ(result.distribution_shape(), MeshShape(8));
        EXPECT_EQ(placements_of(result), (UnionPlacements{TopoShard{3}}));
        EXPECT_EQ(result.mesh_coords(), coords);
    }
    // An uncollapsed [Shard{3}, Shard{3}] first operand unions to [Shard{3}, Replicate]: a dim claimed on one axis
    // reads Replicate on the next, so only the collapsed form carries "a distinct piece on every device".
    {
        const auto [placements, shape] =
            union_of({nd_label(mesh, {TopoShard{3}, TopoShard{3}}), nd_label(mesh, {TopoReplicate{}, TopoShard{3}})});
        EXPECT_EQ(shape, mesh);
        EXPECT_EQ(placements, (UnionPlacements{TopoShard{3}, TopoReplicate{}}));
    }
    // An N-D first operand: a collapsed sharded weight is left out (the union drops it too) while a same-rank bias
    // still contributes its Shard.
    {
        const auto result = union_topology(
            {nd_label(mesh, {TopoShard{2}, TopoReplicate{}}),
             collapsed_label(mesh, TopoShard{3}),
             nd_label(mesh, {TopoReplicate{}, TopoShard{3}})});
        EXPECT_EQ(result.distribution_shape(), mesh);
        EXPECT_EQ(placements_of(result), (UnionPlacements{TopoShard{2}, TopoShard{3}}));
    }
    // A fully replicated first operand (a whole-mesh all_gather result) decides nothing: the sharded operands' rank
    // and placements, as the union has them (the lower-rank collapsed bias is dropped).
    {
        const auto result = union_topology(
            {collapsed_label(mesh, TopoReplicate{}),
             nd_label(mesh, {TopoReplicate{}, TopoShard{3}}),
             collapsed_label(mesh, TopoShard{3})});
        EXPECT_EQ(result.distribution_shape(), mesh);
        EXPECT_EQ(placements_of(result), (UnionPlacements{TopoReplicate{}, TopoShard{3}}));
    }
    // Nothing sharded anywhere: the first operand's rank, all Replicate, as the union has it.
    {
        const auto result = union_topology(
            {collapsed_label(mesh, TopoReplicate{}), nd_label(mesh, {TopoReplicate{}, TopoReplicate{}})});
        EXPECT_EQ(result.distribution_shape(), MeshShape(8));
        EXPECT_EQ(placements_of(result), (UnionPlacements{TopoReplicate{}}));
    }
}

TEST(CclTopologyUtils, RefuseIsHandledLikeTheHelpersOwnRefusals) {
    // A hook-level refusal (a label the op cannot renumber into its output layout, a ring order no label carries)
    // goes through the same handling as the helpers' own: nullopt in warn-only mode, TT_FATAL under strict mode,
    // silent inside a SilenceIntermediateRefusals scope -- so strict mode covers every "no honest label", not only the
    // helpers'.
    {
        StrictCclTopologyScope warn_only(false);
        EXPECT_FALSE(topo::refuse("hook: no honest label", topo::RefusalFamily::Gather).has_value());
        EXPECT_FALSE(topo::refuse("hook: no honest label", topo::RefusalFamily::Scatter).has_value());
    }
    StrictCclTopologyScope strict(true);
    {
        const topo::SilenceIntermediateRefusals silence_refusals;
        EXPECT_FALSE(topo::refuse("hook: intermediate", topo::RefusalFamily::Scatter).has_value());
    }
    const auto message = message_of([&] { topo::refuse("hook: no honest label", topo::RefusalFamily::Gather); });
    EXPECT_NE(message.find("hook: no honest label"), std::string::npos) << message;
    EXPECT_THROW(topo::refuse("hook: no honest label", topo::RefusalFamily::Scatter), std::runtime_error);
}

TEST(CclTopologyUtils, UncollapseJudgesASubRangeLabelOnADeviceTensorByItsShape) {
    // A device tensor's label lists its storage's coordinates: every device of the mesh, however small the block its
    // input occupied (an op output's buffer spans the mesh and update_output_tensor_topologies keeps each output's
    // own coordinates). from_torch(ShardTensor2dMesh(mesh_shape=(1, C))) on a 2x4 therefore reaches the ops as a
    // (1, 4) distribution over eight coordinates, which no (1, 4)-shaped block matches. With more coordinates than
    // the distribution has devices the shape decides alone: a shape that fits the mesh per axis was mapped as a
    // block (its axes are the mesh's), one that does not fit was laid out row-major over the mesh.
    const MeshShape mesh(2, 4);
    const auto all = row_major_coords(mesh);
    std::string reason;

    const std::vector<TopoPlacement> row_placements{TopoReplicate{}, TopoShard{3}};
    const auto row = topo::uncollapse_placements(nd_label(MeshShape(1, 4), row_placements, all), mesh, &reason);
    ASSERT_TRUE(row.has_value()) << reason;
    EXPECT_EQ(std::vector<TopoPlacement>(row->begin(), row->end()), row_placements);
    const std::vector<TopoPlacement> column_placements{TopoShard{3}, TopoShard{2}};
    const auto column = topo::uncollapse_placements(nd_label(MeshShape(2, 1), column_placements, all), mesh, &reason);
    ASSERT_TRUE(column.has_value()) << reason;
    EXPECT_EQ(std::vector<TopoPlacement>(column->begin(), column->end()), column_placements);

    // (4, 1) does not fit a 2x4 mesh (four chunks along an axis of two rows): the mapper laid it out row-major over
    // row 0, so its axis 0 is the mesh's axis 1 and no per-axis edit describes it -- refused as before, whatever the
    // coordinates say.
    EXPECT_FALSE(
        topo::uncollapse_placements(nd_label(MeshShape(4, 1), {TopoShard{3}, TopoReplicate{}}, all), mesh, &reason)
            .has_value());
    EXPECT_NE(reason.find("align"), std::string::npos) << reason;

    // With exactly the distribution's coordinates (a host tensor straight from the mapper) the block check still
    // applies: a (1, 4) row over four coordinates that are not a row is refused.
    const std::vector<MeshCoordinate> not_a_row{
        MeshCoordinate(0, 0), MeshCoordinate(1, 0), MeshCoordinate(0, 1), MeshCoordinate(1, 1)};
    EXPECT_FALSE(
        topo::uncollapse_placements(nd_label(MeshShape(1, 4), row_placements, not_a_row), mesh, &reason).has_value());
    EXPECT_NE(reason.find("align"), std::string::npos) << reason;

    // The ops' per-axis edits then apply: a broadcast along axis 1 of the (1, 4) row replicates it, keeping the
    // distribution shape and the (storage) coordinates.
    const auto broadcast =
        topo::all_broadcast_output_topology(nd_label(MeshShape(1, 4), row_placements, all), 1, mesh, kRank);
    ASSERT_TRUE(broadcast.has_value());
    EXPECT_EQ(*broadcast, nd_label(MeshShape(1, 4), {TopoReplicate{}, TopoReplicate{}}, all));
}

// ---------------------------------------------------------------------------------------------------------------------
// The fused all-gather + matmul ops label every output they create: their compute_output_topologies hook returns one
// label per output compute_output_specs gives, for each attribute that changes the output count (FSDP, chunks, and
// explicit chunk widths, which take precedence over chunks). The hooks count from the attributes, so these tests pin
// that count against the specs on device tensors carrying real mesh labels.
// ---------------------------------------------------------------------------------------------------------------------

namespace {

using FusedAllGatherMatmulHookTest = tt::tt_metal::GenericMeshDeviceFixture;

// A bf16 tile-layout tensor of `shape` placed on `mesh` by `mapper`.
ttnn::Tensor mesh_tensor(
    tt::tt_metal::distributed::MeshDevice& mesh,
    const ttnn::Shape& shape,
    const ttnn::distributed::TensorToMesh& mapper) {
    const tt::tt_metal::TensorSpec spec(
        shape,
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            tt::tt_metal::MemoryConfig{}));
    return ttnn::distributed::distribute_tensor(
        ttnn::Tensor::from_vector(std::vector<float>(shape.volume(), 1.0F), spec), mapper, mesh);
}

}  // namespace

TEST_F(FusedAllGatherMatmulHookTest, AllGatherMinimalMatmulAsyncLabelsEveryOutput) {
    using ttnn::experimental::prim::AllGatherMinimalMatmulAsyncInputs;
    using ttnn::experimental::prim::AllGatherMinimalMatmulAsyncOp;
    using ttnn::experimental::prim::AllGatherMinimalMatmulAsyncParams;
    auto& mesh = *mesh_device_;
    const MeshShape& shape = mesh.shape();
    if (mesh.num_devices() < 2 || shape.dims() != 2) {
        GTEST_SKIP() << "needs a 2-D mesh with more than one device";
    }
    // The activation gathers its last dim along the axis with more devices, the weight its K dim along the other.
    const uint32_t cluster_axis = shape[1] > 1 ? 1 : 0;
    const uint32_t fsdp_axis = 1 - cluster_axis;
    const auto num_devices = static_cast<uint32_t>(mesh.num_devices());
    const ttnn::Tensor input = mesh_tensor(
        mesh, ttnn::Shape{1, 1, 32, 32 * num_devices}, *ttnn::distributed::shard_tensor_to_mesh_mapper(mesh, 3));
    const ttnn::Tensor weight =
        mesh_tensor(mesh, ttnn::Shape{1, 1, 32, 96}, *ttnn::distributed::replicate_tensor_to_mesh_mapper(mesh));
    const std::optional<tt::tt_metal::GlobalSemaphore> no_barrier;

    struct Case {
        int32_t chunks;
        std::vector<uint32_t> chunk_sizes;
        bool fsdp;
    };
    for (const auto& c : std::vector<Case>{
             {1, {}, false},
             {3, {}, false},
             {2, {64, 32}, false},
             {1, {32, 32, 32}, false},
             {1, {}, true},
             {3, {}, true},
             {2, {64, 32}, true}}) {
        const AllGatherMinimalMatmulAsyncParams params(
            /*config=*/std::nullopt,
            /*fused_activation=*/std::nullopt,
            /*output_mem_config=*/std::nullopt,
            /*output_dtype=*/std::nullopt,
            ttnn::DeviceComputeKernelConfig{},
            /*num_links=*/1,
            /*ring_size=*/shape[cluster_axis],
            ttnn::ccl::Topology::Linear,
            /*semaphore=*/{},
            cluster_axis,
            no_barrier,
            /*using_persistent_buffers=*/false,
            /*force_transpose=*/false,
            /*num_workers_per_link=*/1,
            /*num_buffers_per_channel=*/1,
            /*fused_ternary_scalar=*/std::nullopt,
            c.chunks,
            /*dim=*/-1,
            c.fsdp ? std::optional<uint32_t>(fsdp_axis) : std::nullopt,
            /*fsdp_ring_size=*/c.fsdp ? shape[fsdp_axis] : 1,
            /*fsdp_semaphore=*/{},
            /*using_persistent_weight_buffer=*/false,
            ttnn::ccl::Topology::Linear,
            /*fuse_swiglu=*/false,
            c.chunk_sizes);
        const AllGatherMinimalMatmulAsyncInputs args{.input_tensor = input, .weight_tensor = weight};
        EXPECT_EQ(
            AllGatherMinimalMatmulAsyncOp::compute_output_topologies(params, args).size(),
            AllGatherMinimalMatmulAsyncOp::compute_output_specs(params, args).size())
            << "chunks " << c.chunks << ", " << c.chunk_sizes.size() << " explicit chunk widths, fsdp " << c.fsdp;
    }
}

TEST_F(FusedAllGatherMatmulHookTest, StridedAllGatherMinimalMatmulAsyncLabelsEveryOutput) {
    using ttnn::experimental::prim::MinimalMatmulParams;
    using ttnn::experimental::prim::StridedAllGatherAsync;
    using ttnn::experimental::prim::StridedAllGatherAsyncParams;
    using ttnn::experimental::prim::StridedAllGatherMinimalMatmulAsync;
    using ttnn::experimental::prim::StridedAllGatherMinimalMatmulAsyncInputs;
    using ttnn::experimental::prim::StridedAllGatherMinimalMatmulAsyncParams;
    auto& mesh = *mesh_device_;
    const MeshShape& shape = mesh.shape();
    if (mesh.num_devices() < 2 || shape.dims() != 2) {
        GTEST_SKIP() << "needs a 2-D mesh with more than one device";
    }
    const uint32_t cluster_axis = shape[1] > 1 ? 1 : 0;
    const auto num_devices = static_cast<uint32_t>(mesh.num_devices());
    const ttnn::Tensor input = mesh_tensor(
        mesh, ttnn::Shape{1, 1, 32, 32 * num_devices}, *ttnn::distributed::shard_tensor_to_mesh_mapper(mesh, 3));
    const ttnn::Tensor weight =
        mesh_tensor(mesh, ttnn::Shape{1, 1, 32, 96}, *ttnn::distributed::replicate_tensor_to_mesh_mapper(mesh));

    for (const int32_t chunks : {1, 2, 3}) {
        const StridedAllGatherMinimalMatmulAsyncParams params{
            StridedAllGatherAsyncParams{
                /*dim=*/3,
                /*num_links=*/1,
                /*ring_size=*/shape[cluster_axis],
                tt::tt_metal::MemoryConfig{},
                ttnn::ccl::Topology::Linear,
                /*semaphore=*/{},
                cluster_axis,
                /*num_workers_per_link=*/std::nullopt,
                /*num_buffers_per_channel=*/std::nullopt,
                /*mm_cores_y=*/std::nullopt,
                /*mm_block_ht=*/std::nullopt,
                /*mm_block_wt=*/std::nullopt},
            MinimalMatmulParams{.chunks = chunks},
            /*all_gather_core_grid_offset=*/tt::tt_metal::CoreCoord{0, 0},
            /*read_local_slice_from_input=*/false,
            StridedAllGatherAsync{}};
        const StridedAllGatherMinimalMatmulAsyncInputs args{input, weight, /*persistent_output_buffer=*/std::nullopt};
        EXPECT_EQ(
            StridedAllGatherMinimalMatmulAsync::compute_output_topologies(params, args).size(),
            StridedAllGatherMinimalMatmulAsync::compute_output_specs(params, args).size())
            << "chunks " << chunks;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Two fused ops whose operands carry labels of different distribution ranks, on a 2x4 mesh. Each test builds the op's
// attributes and inputs, places device tensors with real mesh mappers, and calls the op's compute_output_topologies:
// the hook itself is pinned, with no fabric and no kernel involved.
// ---------------------------------------------------------------------------------------------------------------------

namespace {

using FusedOpHookLabel2x4Test = tt::tt_metal::MeshDevice2x4Fixture;

}  // namespace

// A whole-mesh reduce_scatter of a replicated matmul labels its output {8},[Shard(3)] over the input's storage
// coordinates. The fused addcmul operands carry the N-D [Replicate, Shard(3)] label of a 2-D mapper: their distribution
// rank differs from the scatter label's, so union_topology leaves them out and the scatter label stands.
TEST_F(FusedOpHookLabel2x4Test, MinimalMatmulStridedReduceScatterAsyncKeepsTheScatterLabelOverAddcmulOperands) {
    using ttnn::experimental::prim::MinimalMatmulParams;
    using ttnn::experimental::prim::MinimalMatmulStridedReduceScatterAsync;
    using ttnn::experimental::prim::MinimalMatmulStridedReduceScatterAsyncInputs;
    using ttnn::experimental::prim::MinimalMatmulStridedReduceScatterAsyncParams;
    auto& mesh = *mesh_device_;
    const MeshShape& shape = mesh.shape();
    const auto replicate = ttnn::distributed::replicate_tensor_to_mesh_mapper(mesh);
    const auto addcmul_mapper = ttnn::distributed::create_mesh_mapper(
        mesh, ttnn::distributed::MeshMapperConfig{.placements = {TopoReplicate{}, TopoShard{3}}});
    const ttnn::Tensor input = mesh_tensor(mesh, ttnn::Shape{1, 1, 32, 64}, *replicate);
    const ttnn::Tensor weight = mesh_tensor(mesh, ttnn::Shape{1, 1, 64, 256}, *replicate);
    const ttnn::Tensor addcmul_a = mesh_tensor(mesh, ttnn::Shape{1, 1, 32, 256 * shape[1]}, *addcmul_mapper);
    const ttnn::Tensor addcmul_b = mesh_tensor(mesh, ttnn::Shape{1, 1, 32, 256 * shape[1]}, *addcmul_mapper);
    ASSERT_EQ(addcmul_a.tensor_topology(), nd_label(shape, {TopoReplicate{}, TopoShard{3}}));

    const MinimalMatmulStridedReduceScatterAsyncParams params{
        .matmul_struct = MinimalMatmulParams{},
        .fused_ternary_scalar = 1.0F,
        .dim = 3,
        .num_links = 1,
        .ring_size = static_cast<uint32_t>(mesh.num_devices()),
        .rs_output_mem_config = tt::tt_metal::MemoryConfig{},
        .rs_intermediate_mem_config = std::nullopt,
        .topology = ttnn::ccl::Topology::Ring,
        .semaphore = {},
        .barrier_semaphore = std::nullopt,
        .using_persistent_buffers = false,
        .sub_device_id = std::nullopt,
        .cluster_axis = std::nullopt,
        .num_workers_per_link = std::nullopt,
        .num_buffers_per_channel = std::nullopt,
        .chunk_width_in_mm_blocks = std::nullopt,
        .mm_window_blocks = std::nullopt,
        .reduce_scatter_core_grid_offset = tt::tt_metal::CoreCoord{0, 0}};
    const MinimalMatmulStridedReduceScatterAsyncInputs args{
        .input_tensor = input,
        .weight_tensor = weight,
        .optional_rs_intermediate_tensor = std::nullopt,
        .optional_rs_output_tensor = std::nullopt,
        .addcmul_input_tensor1 = addcmul_a,
        .addcmul_input_tensor2 = addcmul_b};

    const auto labels = MinimalMatmulStridedReduceScatterAsync::compute_output_topologies(params, args);
    ASSERT_EQ(labels.size(), 3u);
    const TensorTopology matmul = collapsed_label(shape, TopoReplicate{});
    EXPECT_EQ(labels[0], matmul);                                // the matmul output
    EXPECT_EQ(labels[1], matmul);                                // the reduce_scatter intermediate
    EXPECT_EQ(labels[2], collapsed_label(shape, TopoShard{3}));  // the reduce_scatter output
}

// all_gather_concat gathers the users dim along mesh axis 1 and folds the heads into the last dim of its
// [seq, 1, users, num_heads * head_dim] output: an N-D [Shard(2), Shard(1)] input (heads on axis 0, users on axis 1)
// reads [Shard(3), Replicate] after it.
TEST_F(FusedOpHookLabel2x4Test, AllGatherConcatRenumbersTheHeadsShardOfAnNdLabel) {
    using ttnn::experimental::prim::AllGatherConcatDeviceOperation;
    using ttnn::experimental::prim::AllGatherConcatInputs;
    using ttnn::experimental::prim::AllGatherConcatParams;
    auto& mesh = *mesh_device_;
    const MeshShape& shape = mesh.shape();
    const auto mapper = ttnn::distributed::create_mesh_mapper(
        mesh, ttnn::distributed::MeshMapperConfig{.placements = {TopoShard{2}, TopoShard{1}}});
    // [seq, users, heads, head_dim], with 8 users and 32 heads on each device.
    const ttnn::Tensor input = mesh_tensor(mesh, ttnn::Shape{1, 8 * shape[1], 32 * shape[0], 128}, *mapper);
    ASSERT_EQ(input.tensor_topology(), nd_label(shape, {TopoShard{2}, TopoShard{1}}));

    const tt::tt_metal::CoreRangeSet semaphore_cores{
        tt::tt_metal::CoreRange{tt::tt_metal::CoreCoord{0, 0}, tt::tt_metal::CoreCoord{0, 0}}};
    const AllGatherConcatParams params(
        /*dim=*/1,
        /*num_links=*/1,
        /*ring_size=*/shape[1],
        tt::tt_metal::MemoryConfig{},
        ttnn::ccl::Topology::Linear,
        ttnn::global_semaphore::create_global_semaphore(&mesh, semaphore_cores, 0),
        /*sub_device_id=*/std::nullopt,
        /*num_heads=*/32,
        /*use_noc1_only=*/false,
        /*cluster_axis=*/1);
    const AllGatherConcatInputs args{.input_tensor = input, .buffer_tensor = input};

    const auto labels = AllGatherConcatDeviceOperation::compute_output_topologies(params, args);
    ASSERT_EQ(labels.size(), 1u);
    EXPECT_EQ(labels[0], nd_label(shape, {TopoShard{3}, TopoReplicate{}}));
}
