// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/tt_xy_pair.h"
#include "gtest/gtest.h"
#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/ccl_host_datastructures.hpp"

TEST(CclHelpers, CreateEriscDatamoverBuilder_Chan4_PageSize2048_RRBufferSharingMode) {
    std::size_t num_channels = 4;
    uint32_t page_size = 2048;
    ttnn::ccl::EriscDataMoverBufferSharingMode buffer_sharing_mode = ttnn::ccl::EriscDataMoverBufferSharingMode::ROUND_ROBIN;
    ttnn::ccl::EriscDataMoverTerminationMode termination_mode = ttnn::ccl::EriscDataMoverTerminationMode::MESSAGE_COUNT_REACHED;

    std::size_t num_buffers_per_channel = 1;
    auto edm_builder = create_erisc_datamover_builder(num_channels, page_size, num_buffers_per_channel, buffer_sharing_mode, termination_mode);
    std::vector<uint32_t> worker_semaphore_ids = {0, 1, 2, 3};
    std::vector<uint32_t> message_counts = {256, 512, 24, 1};
    std::vector<std::vector<ttnn::ccl::WorkerXY>> const& worker_coords = {
        {ttnn::ccl::WorkerXY{1, 1}, ttnn::ccl::WorkerXY{2, 1}},
        {ttnn::ccl::WorkerXY{3, 1}},
        {ttnn::ccl::WorkerXY{4, 1}, ttnn::ccl::WorkerXY{5, 1}, ttnn::ccl::WorkerXY{6, 1}},
        {ttnn::ccl::WorkerXY{1, 2}},
    };
    std::vector<bool> is_sender_channel{true, false, true, false};

    std::vector<ttnn::ccl::EriscDatamoverBuilder::ChannelBufferInterface> channel_buffer_interfaces;
    channel_buffer_interfaces.reserve(num_channels);
    for (std::size_t i = 0; i < num_channels; i++) {
        ttnn::ccl::EriscDatamoverBuilder::ChannelBufferInterface const& channel_buffer_interface =
            (is_sender_channel[i])
                ? edm_builder.add_sender_channel(worker_semaphore_ids[i], message_counts[i], worker_coords[i])
                : edm_builder.add_receiver_channel(worker_semaphore_ids[i], message_counts[i], worker_coords[i]);
        channel_buffer_interfaces.push_back(channel_buffer_interface);
        ASSERT_TRUE(channel_buffer_interface.eth_buffer_l1_address > 0);
        ASSERT_TRUE(channel_buffer_interface.eth_semaphore_l1_address > 0);
    }

    auto const& active_channels = edm_builder.get_active_channels();
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
    for (std::size_t i = 0; i < 8; i++) {
        ASSERT_TRUE(ttnn::ccl::EriscDatamoverConfig::get_edm_handshake_address() > 0);
    }
}
TEST(CclHelpers, EriscDatamoverConfig_GetSemaphoresBaseAddress_GT_0) {
    for (std::size_t i = 0; i < 8; i++) {
        ASSERT_TRUE(
            ttnn::ccl::EriscDatamoverConfig::get_semaphores_base_address(i) >=
            (ttnn::ccl::EriscDatamoverConfig::get_edm_handshake_address() +
             ttnn::ccl::EriscDatamoverConfig::handshake_location_size +
             ttnn::ccl::EriscDatamoverConfig::edm_receiver_first_level_ack_source_word_size));
    }
}

TEST(CclHelpers, EriscDatamoverConfig_GetBuffersBaseAddress_GT_0) {
    for (std::size_t i = 0; i < 8; i++) {
        ASSERT_TRUE(
            ttnn::ccl::EriscDatamoverConfig::get_buffers_base_address(i) >=
            (ttnn::ccl::EriscDatamoverConfig::get_edm_handshake_address() +
             ttnn::ccl::EriscDatamoverConfig::handshake_location_size +
             ttnn::ccl::EriscDatamoverConfig::edm_receiver_first_level_ack_source_word_size));
    }
}

TEST(CclHelpers, EriscDatamoverConfig_ComputeBufferSize_GT_0) {
    for (std::size_t i = 0; i < 8; i++) {
        ASSERT_TRUE(
            ttnn::ccl::EriscDatamoverConfig::get_buffers_base_address(i) >=
            (ttnn::ccl::EriscDatamoverConfig::get_edm_handshake_address() +
             ttnn::ccl::EriscDatamoverConfig::handshake_location_size +
             ttnn::ccl::EriscDatamoverConfig::edm_receiver_first_level_ack_source_word_size));
    }
}

/////////////////////////////////////////
// TEST AdvanceSliceRowMajor
/////////////////////////////////////////
//                                               x_y             x_y             x_y
TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_0_0__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_1) {
    const auto expected = ttnn::ccl::coord_t(1, 0);
    auto const& result = ttnn::ccl::advance_slice_row_major({0, 0}, {1, 1}, {2, 2}, 1);
    ASSERT_EQ(result.x, expected.x);
    ASSERT_EQ(result.y, expected.y);
}
TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_1_0__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_1) {
    const auto expected = ttnn::ccl::coord_t(0, 1);
    auto const& result = ttnn::ccl::advance_slice_row_major({1, 0}, {1, 1}, {2, 2}, 1);
    ASSERT_EQ(result.x, expected.x);
    ASSERT_EQ(result.y, expected.y);
}
TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_0_1__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_1) {
    const auto expected = ttnn::ccl::coord_t(1, 1);
    auto const& result = ttnn::ccl::advance_slice_row_major({0, 1}, {1, 1}, {2, 2}, 1);
    ASSERT_EQ(result.x, expected.x);
    ASSERT_EQ(result.y, expected.y);
}
TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_0_0__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_2) {
    const auto expected = ttnn::ccl::coord_t(0, 1);
    auto const& result = ttnn::ccl::advance_slice_row_major({0, 0}, {1, 1}, {2, 2}, 2);
    ASSERT_EQ(result.x, expected.x);
    ASSERT_EQ(result.y, expected.y);
}
TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_1_0__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_2) {
    const auto expected = ttnn::ccl::coord_t(1, 1);
    auto const& result = ttnn::ccl::advance_slice_row_major({1, 0}, {1, 1}, {2, 2}, 2);
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
    auto const& result_offset = ttnn::ccl::advance_slice_row_major(worker_slice_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_EQ(result_offset.x, expected.x);
    ASSERT_EQ(result_offset.y, expected.y);


    auto const& result_offset2 = ttnn::ccl::advance_slice_row_major(result_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_TRUE(result_offset2.x >= tensor_slice_shape.x || result_offset2.y >= tensor_slice_shape.y);
}

TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_24_0__InnerShape_24_1__OuterShape_32_4__NumWorkers_4) {
    const auto worker_slice_offset = ttnn::ccl::coord_t(24, 0);
    const auto worker_slice_shape = ttnn::ccl::coord_t(24, 1);
    const auto tensor_slice_shape = ttnn::ccl::coord_t(32, 4);
    const uint32_t num_workers = 4;

    const auto expected = ttnn::ccl::coord_t(24, 2);
    auto const& result_offset = ttnn::ccl::advance_slice_row_major(worker_slice_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_EQ(result_offset.x, expected.x);
    ASSERT_EQ(result_offset.y, expected.y);

    auto const& result_offset2 = ttnn::ccl::advance_slice_row_major(result_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_TRUE(result_offset2.x >= tensor_slice_shape.x || result_offset2.y >= tensor_slice_shape.y);
}

TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_0_1__InnerShape_24_1__OuterShape_32_4__NumWorkers_4) {
    const auto worker_slice_offset = ttnn::ccl::coord_t(0, 1);
    const auto worker_slice_shape = ttnn::ccl::coord_t(24, 1);
    const auto tensor_slice_shape = ttnn::ccl::coord_t(32, 4);
    const uint32_t num_workers = 4;

    const auto expected = ttnn::ccl::coord_t(0, 3);
    auto const& result_offset = ttnn::ccl::advance_slice_row_major(worker_slice_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_EQ(result_offset.x, expected.x);
    ASSERT_EQ(result_offset.y, expected.y);

    auto const& result_offset2 = ttnn::ccl::advance_slice_row_major(result_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_TRUE(result_offset2.x >= tensor_slice_shape.x || result_offset2.y >= tensor_slice_shape.y);
}


TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_24_1__InnerShape_24_1__OuterShape_32_4__NumWorkers_4) {
    const auto worker_slice_offset = ttnn::ccl::coord_t(24, 1);
    const auto worker_slice_shape = ttnn::ccl::coord_t(24, 1);
    const auto tensor_slice_shape = ttnn::ccl::coord_t(32, 4);
    const uint32_t num_workers = 4;

    const auto expected = ttnn::ccl::coord_t(24, 3);
    auto const& result_offset = ttnn::ccl::advance_slice_row_major(worker_slice_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_EQ(result_offset.x, expected.x);
    ASSERT_EQ(result_offset.y, expected.y);

    auto const& result_offset2 = ttnn::ccl::advance_slice_row_major(result_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_TRUE(result_offset2.x >= tensor_slice_shape.x || result_offset2.y >= tensor_slice_shape.y);
}


// Test that we successfully go out of bounds on the last iteration
TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_0_1__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_2) {
    auto const& result = ttnn::ccl::advance_slice_row_major({0, 1}, {1, 1}, {2, 2}, 2);
    ASSERT_TRUE(result.x >= 2 || result.y >= 2);
}
TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_1_1__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_2) {
    auto const& result = ttnn::ccl::advance_slice_row_major({1, 1}, {1, 1}, {2, 2}, 2);
    ASSERT_TRUE(result.x >= 2 || result.y >= 2);
}

TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_0_0__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_3) {
    const auto expected = ttnn::ccl::coord_t(1, 1);
    auto const& result = ttnn::ccl::advance_slice_row_major({0, 0}, {1, 1}, {2, 2}, 3);
    ASSERT_EQ(result.x, expected.x);
    ASSERT_EQ(result.y, expected.y);
}
TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_1_1__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_3) {
    const auto expected = ttnn::ccl::coord_t(1, 1);
    const auto outer_shape = ttnn::ccl::coord_t(2, 2);
    const auto inner_offset = ttnn::ccl::coord_t(1, 1);
    const auto inner_shape = ttnn::ccl::coord_t(1, 1);
    const uint32_t num_parallel_workers = 3;
    auto const& result =
        ttnn::ccl::advance_slice_row_major(inner_offset, inner_shape, outer_shape, num_parallel_workers);
    ASSERT_TRUE(result.x >= outer_shape.x || result.y >= outer_shape.y);
}
TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_24_0__InnerShape_24_0__OuterShape_32_4__NumActiveSlices_4) {
    const auto expected = ttnn::ccl::coord_t(24, 2);
    const auto outer_shape = ttnn::ccl::coord_t(32, 4);
    const auto inner_offset = ttnn::ccl::coord_t(24, 0);
    const auto inner_shape = ttnn::ccl::coord_t(24, 1);
    const uint32_t num_parallel_workers = 4;
    auto const& result =
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
    auto const& result = ttnn::ccl::advance_wrapped_slice_row_major({0, 0}, {1, 1}, {2, 2}, 1);
    ASSERT_EQ(result.x, expected.x);
    ASSERT_EQ(result.y, expected.y);
}
TEST(CclHelper_AdvanceWrappedSliceRowMajor, InnerOffset_1_0__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_1) {
    const auto expected = ttnn::ccl::coord_t(0, 1);
    auto const& result = ttnn::ccl::advance_wrapped_slice_row_major({1, 0}, {1, 1}, {2, 2}, 1);
    ASSERT_EQ(result.x, expected.x);
    ASSERT_EQ(result.y, expected.y);
}
TEST(CclHelper_AdvanceWrappedSliceRowMajor, InnerOffset_0_1__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_1) {
    const auto expected = ttnn::ccl::coord_t(1, 1);
    auto const& result = ttnn::ccl::advance_wrapped_slice_row_major({0, 1}, {1, 1}, {2, 2}, 1);
    ASSERT_EQ(result.x, expected.x);
    ASSERT_EQ(result.y, expected.y);
}
TEST(CclHelper_AdvanceWrappedSliceRowMajor, InnerOffset_0_0__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_2) {
    const auto expected = ttnn::ccl::coord_t(0, 1);
    auto const& result = ttnn::ccl::advance_wrapped_slice_row_major({0, 0}, {1, 1}, {2, 2}, 2);
    ASSERT_EQ(result.x, expected.x);
    ASSERT_EQ(result.y, expected.y);
}
TEST(CclHelper_AdvanceWrappedSliceRowMajor, InnerOffset_1_0__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_2) {
    const auto expected = ttnn::ccl::coord_t(1, 1);
    auto const& result = ttnn::ccl::advance_wrapped_slice_row_major({1, 0}, {1, 1}, {2, 2}, 2);
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

    const auto expected = ttnn::ccl::coord_t(0, 3); // Updated
    auto const& result_offset = ttnn::ccl::advance_wrapped_slice_row_major(worker_slice_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_EQ(result_offset.x, expected.x);
    ASSERT_EQ(result_offset.y, expected.y);


    auto const& result_offset2 = ttnn::ccl::advance_wrapped_slice_row_major(result_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_TRUE(result_offset2.x >= tensor_slice_shape.x || result_offset2.y >= tensor_slice_shape.y);
}

TEST(CclHelper_AdvanceWrappedSliceRowMajor, InnerOffset_24_0__InnerShape_24_1__OuterShape_32_4__NumWorkers_4) {
    const auto worker_slice_offset = ttnn::ccl::coord_t(24, 0);
    const auto worker_slice_shape = ttnn::ccl::coord_t(24, 1);
    const auto tensor_slice_shape = ttnn::ccl::coord_t(32, 4);
    const uint32_t num_workers = 4;

    const auto expected = ttnn::ccl::coord_t(24, 3); // Updated
    auto const& result_offset = ttnn::ccl::advance_wrapped_slice_row_major(worker_slice_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_EQ(result_offset.x, expected.x);
    ASSERT_EQ(result_offset.y, expected.y);

    auto const& result_offset2 = ttnn::ccl::advance_wrapped_slice_row_major(result_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_TRUE(result_offset2.x >= tensor_slice_shape.x || result_offset2.y >= tensor_slice_shape.y);
}

TEST(CclHelper_AdvanceWrappedSliceRowMajor, InnerOffset_0_0__InnerShape_44_1__OuterShape_32_4__NumWorkers_2) { // New
    const auto worker_slice_offset = ttnn::ccl::coord_t(0, 0);
    const auto worker_slice_shape = ttnn::ccl::coord_t(44, 1);
    const auto tensor_slice_shape = ttnn::ccl::coord_t(32, 4);
    const uint32_t num_workers = 2;

    const auto expected = ttnn::ccl::coord_t(24, 2);
    auto const& result_offset = ttnn::ccl::advance_wrapped_slice_row_major(worker_slice_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_EQ(result_offset.x, expected.x);
    ASSERT_EQ(result_offset.y, expected.y);

    auto const& result_offset2 = ttnn::ccl::advance_wrapped_slice_row_major(result_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_TRUE(result_offset2.x >= tensor_slice_shape.x || result_offset2.y >= tensor_slice_shape.y);
}


// Test that we successfully go out of bounds on the last iteration
TEST(CclHelper_AdvanceWrappedSliceRowMajor, InnerOffset_0_1__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_2) {
    auto const& result = ttnn::ccl::advance_wrapped_slice_row_major({0, 1}, {1, 1}, {2, 2}, 2);
    ASSERT_TRUE(result.x >= 2 || result.y >= 2);
}
TEST(CclHelper_AdvanceWrappedSliceRowMajor, InnerOffset_1_1__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_2) {
    auto const& result = ttnn::ccl::advance_wrapped_slice_row_major({1, 1}, {1, 1}, {2, 2}, 2);
    ASSERT_TRUE(result.x >= 2 || result.y >= 2);
}

TEST(CclHelper_AdvanceWrappedSliceRowMajor, InnerOffset_0_0__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_3) {
    const auto expected = ttnn::ccl::coord_t(1, 1);
    auto const& result = ttnn::ccl::advance_wrapped_slice_row_major({0, 0}, {1, 1}, {2, 2}, 3);
    ASSERT_EQ(result.x, expected.x);
    ASSERT_EQ(result.y, expected.y);
}
TEST(CclHelper_AdvanceWrappedSliceRowMajor, InnerOffset_1_1__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_3) {
    const auto expected = ttnn::ccl::coord_t(1, 1);
    const auto outer_shape = ttnn::ccl::coord_t(2, 2);
    const auto inner_offset = ttnn::ccl::coord_t(1, 1);
    const auto inner_shape = ttnn::ccl::coord_t(1, 1);
    const uint32_t num_parallel_workers = 3;
    auto const& result =
        ttnn::ccl::advance_wrapped_slice_row_major(inner_offset, inner_shape, outer_shape, num_parallel_workers);
    ASSERT_TRUE(result.x >= outer_shape.x || result.y >= outer_shape.y);
}
TEST(CclHelper_AdvanceWrappedSliceRowMajor, InnerOffset_16_1__InnerShape_24_1__OuterShape_32_4__NumWorkers_4) {
    const auto worker_slice_offset = ttnn::ccl::coord_t(16, 1); // Updated
    const auto worker_slice_shape = ttnn::ccl::coord_t(24, 1);
    const auto tensor_slice_shape = ttnn::ccl::coord_t(32, 4);
    const uint32_t num_workers = 4;

    auto const& result_offset = ttnn::ccl::advance_wrapped_slice_row_major(worker_slice_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_TRUE(result_offset.x >= tensor_slice_shape.x || result_offset.y >= tensor_slice_shape.y);
}


TEST(CclHelper_AdvanceWrappedSliceRowMajor, InnerOffset_8_2__InnerShape_24_1__OuterShape_32_4__NumWorkers_4) {
    const auto worker_slice_offset = ttnn::ccl::coord_t(8, 2); // Updated
    const auto worker_slice_shape = ttnn::ccl::coord_t(24, 1);
    const auto tensor_slice_shape = ttnn::ccl::coord_t(32, 4);
    const uint32_t num_workers = 4;

    auto const& result_offset = ttnn::ccl::advance_wrapped_slice_row_major(worker_slice_offset, worker_slice_shape, tensor_slice_shape, num_workers);
    ASSERT_TRUE(result_offset.x >= tensor_slice_shape.x || result_offset.y >= tensor_slice_shape.y);
}

/////////////////////////////////////////
// Test RingReduceScatterTensorSlicer
/////////////////////////////////////////
TEST(Ccl_RingReduceScatterTensorSlicer, ComputeWorkerSliceOffsets_AllWorkersSameRow) {
    auto worker_slice_shapes = std::vector<tt_xy_pair>(4, {2, 2});
    tt_xy_pair tensor_slice_shape = {8, 4};
    auto const& worker_slice_offsets = ttnn::ccl::RingReduceScatterTensorSlicer::compute_worker_slice_offsets(
        worker_slice_shapes, tensor_slice_shape);
    ASSERT_EQ(worker_slice_offsets.at(0), tt_xy_pair(0, 0));
    ASSERT_EQ(worker_slice_offsets.at(1), tt_xy_pair(2, 0));
    ASSERT_EQ(worker_slice_offsets.at(2), tt_xy_pair(4, 0));
    ASSERT_EQ(worker_slice_offsets.at(3), tt_xy_pair(6, 0));
}
TEST(Ccl_RingReduceScatterTensorSlicer, ComputeWorkerSliceOffsets_1WorkerWrapToNextRowAligned) {
    auto worker_slice_shapes = std::vector<tt_xy_pair>(4, {2, 2});
    tt_xy_pair tensor_slice_shape = {6, 4};
    auto const& worker_slice_offsets = ttnn::ccl::RingReduceScatterTensorSlicer::compute_worker_slice_offsets(
        worker_slice_shapes, tensor_slice_shape);
    ASSERT_EQ(worker_slice_offsets.at(0), tt_xy_pair(0, 0));
    ASSERT_EQ(worker_slice_offsets.at(1), tt_xy_pair(2, 0));
    ASSERT_EQ(worker_slice_offsets.at(2), tt_xy_pair(4, 0));
    ASSERT_EQ(worker_slice_offsets.at(3), tt_xy_pair(0, 2));
}
TEST(Ccl_RingReduceScatterTensorSlicer, ComputeWorkerSliceOffsets_1WorkerWrapToNextRowMisaligned) {

    auto worker_slice_shapes = std::vector<tt_xy_pair>(4, {2, 2});
    tt_xy_pair tensor_slice_shape = {5, 4};
    auto const& worker_slice_offsets = ttnn::ccl::RingReduceScatterTensorSlicer::compute_worker_slice_offsets(
        worker_slice_shapes, tensor_slice_shape);
    ASSERT_EQ(worker_slice_offsets.at(0), tt_xy_pair(0, 0));
    ASSERT_EQ(worker_slice_offsets.at(1), tt_xy_pair(2, 0));
    ASSERT_EQ(worker_slice_offsets.at(2), tt_xy_pair(4, 0));
    ASSERT_EQ(worker_slice_offsets.at(3), tt_xy_pair(0, 2));

}

TEST(Ccl_RingReduceScatterTensorSlicer, ComputeWorkerSliceOffsets_MultipleWorkersWrapToNextRowAligned) {
    auto worker_slice_shapes = std::vector<tt_xy_pair>(8, {2, 2});
    tt_xy_pair tensor_slice_shape = {10, 4};
    auto const& worker_slice_offsets = ttnn::ccl::RingReduceScatterTensorSlicer::compute_worker_slice_offsets(
        worker_slice_shapes, tensor_slice_shape);
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
    auto const& worker_slice_offsets = ttnn::ccl::RingReduceScatterTensorSlicer::compute_worker_slice_offsets(
        worker_slice_shapes, tensor_slice_shape);
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
    auto const& worker_slice_offsets = ttnn::ccl::RingReduceScatterTensorSlicer::compute_worker_slice_offsets(
        worker_slice_shapes, tensor_slice_shape);
    ASSERT_EQ(worker_slice_offsets.at(0), tt_xy_pair(0, 0));
    ASSERT_EQ(worker_slice_offsets.at(1), tt_xy_pair(0, 4));
    ASSERT_EQ(worker_slice_offsets.at(2), tt_xy_pair(0, 8));
}

TEST(Ccl_RingReduceScatterTensorSlicer, ComputeWorkerSliceOffsets_NMinus1WorkersWrapToNextRowMisaligned) {
    auto worker_slice_shapes = std::vector<tt_xy_pair>(3, {4, 3});
    tt_xy_pair tensor_slice_shape = {3, 12};
    auto const& worker_slice_offsets = ttnn::ccl::RingReduceScatterTensorSlicer::compute_worker_slice_offsets(
        worker_slice_shapes, tensor_slice_shape);
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
    auto const& worker_slice_offsets = ttnn::ccl::RingReduceScatterWrappedTensorSlicer::compute_worker_slice_offsets(worker_slice_shapes, tensor_slice_shape);
    ASSERT_EQ(worker_slice_offsets.at(0), tt_xy_pair(0, 0));
    ASSERT_EQ(worker_slice_offsets.at(1), tt_xy_pair(2, 0));
    ASSERT_EQ(worker_slice_offsets.at(2), tt_xy_pair(4, 0));
    ASSERT_EQ(worker_slice_offsets.at(3), tt_xy_pair(6, 0));
}
TEST(Ccl_RingReduceScatterWrappedTensorSlicer, ComputeWorkerSliceWrappedOffsets_1WorkerWrapToNextRowAligned) {
    auto worker_slice_shapes = std::vector<tt_xy_pair>(4, {2, 1});
    tt_xy_pair tensor_slice_shape = {6, 2};
    auto const& worker_slice_offsets = ttnn::ccl::RingReduceScatterWrappedTensorSlicer::compute_worker_slice_offsets(worker_slice_shapes, tensor_slice_shape);
    ASSERT_EQ(worker_slice_offsets.at(0), tt_xy_pair(0, 0));
    ASSERT_EQ(worker_slice_offsets.at(1), tt_xy_pair(2, 0));
    ASSERT_EQ(worker_slice_offsets.at(2), tt_xy_pair(4, 0));
    ASSERT_EQ(worker_slice_offsets.at(3), tt_xy_pair(0, 1));
}
TEST(Ccl_RingReduceScatterWrappedTensorSlicer, ComputeWorkerSliceWrappedOffsets_1WorkerWrapToNextRowMisaligned) {
    auto worker_slice_shapes = std::vector<tt_xy_pair>(4, {2, 1});
    tt_xy_pair tensor_slice_shape = {5, 2};
    auto const& worker_slice_offsets = ttnn::ccl::RingReduceScatterWrappedTensorSlicer::compute_worker_slice_offsets(worker_slice_shapes, tensor_slice_shape);
    ASSERT_EQ(worker_slice_offsets.at(0), tt_xy_pair(0, 0));
    ASSERT_EQ(worker_slice_offsets.at(1), tt_xy_pair(2, 0));
    ASSERT_EQ(worker_slice_offsets.at(2), tt_xy_pair(4, 0));
    ASSERT_EQ(worker_slice_offsets.at(3), tt_xy_pair(1, 1));
}
TEST(Ccl_RingReduceScatterWrappedTensorSlicer, ComputeWorkerSliceWrappedOffsets_MultipleWorkersWrapToNextRowAligned) {
    auto worker_slice_shapes = std::vector<tt_xy_pair>(8, {2, 1});
    tt_xy_pair tensor_slice_shape = {10, 2};
    auto const& worker_slice_offsets = ttnn::ccl::RingReduceScatterWrappedTensorSlicer::compute_worker_slice_offsets(worker_slice_shapes, tensor_slice_shape);
    ASSERT_EQ(worker_slice_offsets.at(0), tt_xy_pair(0, 0));
    ASSERT_EQ(worker_slice_offsets.at(1), tt_xy_pair(2, 0));
    ASSERT_EQ(worker_slice_offsets.at(2), tt_xy_pair(4, 0));
    ASSERT_EQ(worker_slice_offsets.at(3), tt_xy_pair(6, 0));
    ASSERT_EQ(worker_slice_offsets.at(4), tt_xy_pair(8, 0));
    ASSERT_EQ(worker_slice_offsets.at(5), tt_xy_pair(0, 1));
    ASSERT_EQ(worker_slice_offsets.at(6), tt_xy_pair(2, 1));
    ASSERT_EQ(worker_slice_offsets.at(7), tt_xy_pair(4, 1));
}

TEST(Ccl_RingReduceScatterWrappedTensorSlicer, ComputeWorkerSliceWrappedOffsets_MultipleWorkersWrapToNextRowMisaligned) {
    auto worker_slice_shapes = std::vector<tt_xy_pair>(8, {2, 1});
    tt_xy_pair tensor_slice_shape = {9, 2};
    auto const& worker_slice_offsets = ttnn::ccl::RingReduceScatterWrappedTensorSlicer::compute_worker_slice_offsets(worker_slice_shapes, tensor_slice_shape);
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
    auto const& worker_slice_offsets = ttnn::ccl::RingReduceScatterWrappedTensorSlicer::compute_worker_slice_offsets(worker_slice_shapes, tensor_slice_shape);
    ASSERT_EQ(worker_slice_offsets.at(0), tt_xy_pair(0, 0));
    ASSERT_EQ(worker_slice_offsets.at(1), tt_xy_pair(0, 4));
    ASSERT_EQ(worker_slice_offsets.at(2), tt_xy_pair(0, 8));
}

TEST(Ccl_RingReduceScatterWrappedTensorSlicer, ComputeWorkerSliceWrappedOffsets_NMinus1WorkersWrapToNextRowMisaligned) {
    auto worker_slice_shapes = std::vector<tt_xy_pair>(3, {11, 1});
    tt_xy_pair tensor_slice_shape = {3, 12};
    auto const& worker_slice_offsets = ttnn::ccl::RingReduceScatterWrappedTensorSlicer::compute_worker_slice_offsets(worker_slice_shapes, tensor_slice_shape);
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
        true); // Updated
    uint32_t num_workers = 4;
    auto num_iterations = worker_slice.compute_num_worker_slice_iterations(num_workers);
    auto expected = 1; // Updated
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
