// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include "tt_metal/impl/dispatch/vector_aligned.hpp"
#include <utility>
#include <vector>

#include <tt_stl/span.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/impl/dispatch/device_command_calculator.hpp"
#include "tt_metal/impl/dispatch/dispatch_settings.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_commands.hpp"

namespace tt::tt_metal {

// These are host-only command-layout checks, but they link into unit_tests_dispatch which runs against
// hardware, so resolving the global MetalContext here is an acceptable single documented bridge.
class DeviceCommandTest : public ::testing::Test {
protected:
    MetalContext& ctx_ = MetalContext::instance();
};

TEST_F(DeviceCommandTest, CPU_AddDispatchWait) {
    DeviceCommandCalculator calculator(ctx_);
    calculator.add_dispatch_wait();

    HostMemDeviceCommand command(ctx_, calculator.write_offset_bytes());
    command.add_dispatch_wait(0, 0, 0, 0, 0);
    EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
}

TEST_F(DeviceCommandTest, CPU_AddDispatchWaitWithPrefetchStall) {
    DeviceCommandCalculator calculator(ctx_);
    calculator.add_dispatch_wait_with_prefetch_stall();

    HostMemDeviceCommand command(ctx_, calculator.write_offset_bytes());
    command.add_dispatch_wait_with_prefetch_stall(0, 0, 0, 0, 0);
    EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
}

TEST_F(DeviceCommandTest, CPU_AddPrefetchRelayLinear) {
    DeviceCommandCalculator calculator(ctx_);
    calculator.add_prefetch_relay_linear();

    HostMemDeviceCommand command(ctx_, calculator.write_offset_bytes());
    command.add_prefetch_relay_linear(0, 0, 0);
    EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
}

TEST_F(DeviceCommandTest, CPU_AddData) {
    DeviceCommandCalculator calculator(ctx_);
    calculator.add_data(32);

    HostMemDeviceCommand command(ctx_, calculator.write_offset_bytes());
    uint32_t data[1] = {};
    command.add_data(data, 4, 32);
    EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
}

TEST_F(DeviceCommandTest, CPU_AddDispatchWriteLinear) {
    {
        DeviceCommandCalculator calculator(ctx_);
        calculator.add_dispatch_write_linear<false, false>(5);

        HostMemDeviceCommand command(ctx_, calculator.write_offset_bytes());
        command.add_dispatch_write_linear<false, false>(0, 0, 0, 5);
        EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
    }
    {
        DeviceCommandCalculator calculator(ctx_);
        calculator.add_dispatch_write_linear<true, true>(5);

        HostMemDeviceCommand command(ctx_, calculator.write_offset_bytes());
        uint32_t data[2] = {};
        command.add_dispatch_write_linear<true, true>(0, 0, 0, 5, data);
        EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
    }
    {
        DeviceCommandCalculator calculator(ctx_);
        calculator.add_dispatch_write_linear<true, false>(5);

        HostMemDeviceCommand command(ctx_, calculator.write_offset_bytes());
        command.add_dispatch_write_linear<true, false>(0, 0, 0, 5);
        EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
    }
    {
        DeviceCommandCalculator calculator(ctx_);
        calculator.add_dispatch_write_linear<false, true>(5);

        HostMemDeviceCommand command(ctx_, calculator.write_offset_bytes());
        uint32_t data[2] = {};
        command.add_dispatch_write_linear<false, true>(0, 0, 0, 5, data);
        EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
    }
}

TEST_F(DeviceCommandTest, CPU_AddDispatchGoSignalMcast) {
    DeviceCommandCalculator calculator(ctx_);
    calculator.add_dispatch_go_signal_mcast();

    HostMemDeviceCommand command(ctx_, calculator.write_offset_bytes());
    command.add_dispatch_go_signal_mcast(0, 0, 0, 0, DispatcherSelect::DISPATCH_MASTER);
    EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
}

TEST_F(DeviceCommandTest, CPU_AddNotifyDispatchSGoSignalCmd) {
    DeviceCommandCalculator calculator(ctx_);
    calculator.add_notify_dispatch_s_go_signal_cmd();

    HostMemDeviceCommand command(ctx_, calculator.write_offset_bytes());
    command.add_notify_dispatch_s_go_signal_cmd(0, 0);
    EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
}

TEST_F(DeviceCommandTest, CPU_AddDispatchSetNumWorkerSems) {
    DeviceCommandCalculator calculator(ctx_);
    calculator.add_dispatch_set_num_worker_sems();

    HostMemDeviceCommand command(ctx_, calculator.write_offset_bytes());
    command.add_dispatch_set_num_worker_sems(0, DispatcherSelect::DISPATCH_MASTER);
    EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
}

TEST_F(DeviceCommandTest, CPU_AddDispatchSetSubDeviceWorkerCounts) {
    constexpr uint32_t num_sub_devices = 3;
    std::array<uint32_t, num_sub_devices> workers_per_sub_device = {4, 7, 2};

    DeviceCommandCalculator calculator(ctx_);
    calculator.add_dispatch_set_sub_device_worker_counts(num_sub_devices);

    HostMemDeviceCommand command(ctx_, calculator.write_offset_bytes());
    command.add_dispatch_set_sub_device_worker_counts(
        ttsl::Span<const uint32_t>(workers_per_sub_device.data(), workers_per_sub_device.size()),
        DispatcherSelect::DISPATCH_MASTER);
    EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
}

TEST_F(DeviceCommandTest, CPU_AddDispatchSetWriteOffsets) {
    DeviceCommandCalculator calculator(ctx_);
    calculator.add_dispatch_set_write_offsets(CQ_DISPATCH_MAX_WRITE_OFFSETS);

    HostMemDeviceCommand command(ctx_, calculator.write_offset_bytes());
    std::vector<uint32_t> offsets(CQ_DISPATCH_MAX_WRITE_OFFSETS, 0);
    command.add_dispatch_set_write_offsets(offsets);
    EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
}

TEST_F(DeviceCommandTest, CPU_AddDispatchTerminate) {
    DeviceCommandCalculator calculator(ctx_);
    calculator.add_dispatch_terminate();

    HostMemDeviceCommand command(ctx_, calculator.write_offset_bytes());
    command.add_dispatch_terminate();
    EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
}

TEST_F(DeviceCommandTest, CPU_AddDispatchWritePaged) {
    {
        DeviceCommandCalculator calculator(ctx_);
        calculator.add_dispatch_write_paged<false>(1, 5);
        // Do PCIE alignment for out-of-line data
        calculator.add_alignment();
        HostMemDeviceCommand command(ctx_, calculator.write_offset_bytes());
        command.add_dispatch_write_paged<false>(false, 0, 0, 0, 1, 5);
        command.align_write_offset();
        EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
    }
    {
        DeviceCommandCalculator calculator(ctx_);
        calculator.add_dispatch_write_paged<true>(1, 5);

        HostMemDeviceCommand command(ctx_, calculator.write_offset_bytes());
        uint32_t data[2] = {};
        command.add_dispatch_write_paged<true>(false, 0, 0, 0, 1, 5, data);
        EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
    }
}

TEST_F(DeviceCommandTest, CPU_AddPrefetchRelayPaged) {
    DeviceCommandCalculator calculator(ctx_);
    calculator.add_prefetch_relay_paged();

    HostMemDeviceCommand command(ctx_, calculator.write_offset_bytes());
    command.add_prefetch_relay_paged(0, 0, 0, 0, 0, 0);
    EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
}

TEST_F(DeviceCommandTest, CPU_AddPrefetchRelayPagedPacked) {
    DeviceCommandCalculator calculator(ctx_);
    calculator.add_prefetch_relay_paged_packed(1);

    HostMemDeviceCommand command(ctx_, calculator.write_offset_bytes());
    std::vector<CQPrefetchRelayPagedPackedSubCmd> sub_cmds(1);
    command.add_prefetch_relay_paged_packed(0, sub_cmds, 1);
    EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
}

template <typename T>
class WritePackedCommandTest : public ::testing::Test {
protected:
    MetalContext& ctx_ = MetalContext::instance();
};

using TestTypes = testing::Types<CQDispatchWritePackedMulticastSubCmd, CQDispatchWritePackedUnicastSubCmd>;
TYPED_TEST_SUITE(WritePackedCommandTest, TestTypes);

TYPED_TEST(WritePackedCommandTest, CPU_AddDispatchWritePacked) {
    {
        DeviceCommandCalculator calculator(this->ctx_);
        calculator.add_dispatch_write_packed<TypeParam>(2, 5, 100, /*no_stride*/ false);

        HostMemDeviceCommand command(this->ctx_, calculator.write_offset_bytes());
        std::vector<TypeParam> sub_cmds(2);
        uint32_t data[1] = {};
        std::vector<std::pair<const void*, uint32_t>> data_collection{{data, 4}, {data, 4}};
        command.add_dispatch_write_packed<TypeParam>(0, 2, 0, 5, 0, sub_cmds, data_collection, 100, 0, false);
        EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
    }
    {
        DeviceCommandCalculator calculator(this->ctx_);
        calculator.add_dispatch_write_packed<TypeParam>(2, 5, 100, /*no_stride*/ true);

        HostMemDeviceCommand command(this->ctx_, calculator.write_offset_bytes());
        std::vector<TypeParam> sub_cmds(2);
        uint32_t data[1] = {};
        std::vector<std::pair<const void*, uint32_t>> data_collection{{data, 4}};
        command.add_dispatch_write_packed<TypeParam>(0, 2, 0, 5, 0, sub_cmds, data_collection, 100, 0, true);
        EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
    }
}

namespace {

constexpr uint32_t kDispatchPageSize = 1u << DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE;

// Builds a packed write whose single data copy is shared by all sub cmds.
template <typename PackedSubCmd>
void build_no_stride_write_packed(
    MetalContext& ctx, uint32_t num_sub_cmds, uint32_t packed_data_sizeB, uint32_t packed_write_max_unicast_sub_cmds) {
    DeviceCommandCalculator calculator(ctx);
    calculator.add_dispatch_write_packed<PackedSubCmd>(
        num_sub_cmds, packed_data_sizeB, packed_write_max_unicast_sub_cmds, /*no_stride*/ true);

    HostMemDeviceCommand command(ctx, calculator.write_offset_bytes());
    std::vector<PackedSubCmd> sub_cmds(num_sub_cmds);
    std::vector<uint8_t> data(packed_data_sizeB);
    std::vector<std::pair<const void*, uint32_t>> data_collection{{data.data(), packed_data_sizeB}};
    command.add_dispatch_write_packed<PackedSubCmd>(
        0,
        num_sub_cmds,
        0,
        packed_data_sizeB,
        0,
        sub_cmds,
        data_collection,
        packed_write_max_unicast_sub_cmds,
        0,
        /*no_stride*/ true);
    EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
}

}  // namespace

TYPED_TEST(WritePackedCommandTest, CPU_GetMaxWritePackedSubCmdsNoStrideFitsDispatchPage) {
    const uint32_t l1_alignment = this->ctx_.hal().get_alignment(HalMemType::L1);
    // Large enough that neither the prefetch command size nor the sub cmd cap is the binding limit.
    constexpr uint32_t max_prefetch_command_size = 1u << 20;
    constexpr uint32_t packed_write_max_unicast_sub_cmds = 4096;
    const uint32_t sub_cmd_cap = std::is_same_v<TypeParam, CQDispatchWritePackedUnicastSubCmd>
                                     ? packed_write_max_unicast_sub_cmds
                                     : get_packed_write_max_multicast_sub_cmds(packed_write_max_unicast_sub_cmds);

    DeviceCommandCalculator calculator(this->ctx_);
    for (uint32_t data_words = 1; data_words <= kDispatchPageSize / sizeof(uint32_t); data_words++) {
        const uint32_t data_sizeB = data_words * sizeof(uint32_t);
        const uint32_t max_sub_cmds = calculator.get_max_write_packed_sub_cmds<TypeParam>(
            data_words, max_prefetch_command_size, packed_write_max_unicast_sub_cmds, /*no_stride*/ true);
        ASSERT_LT(max_sub_cmds, sub_cmd_cap) << "data_words=" << data_words;
        if (max_sub_cmds != 0) {
            EXPECT_LE(
                get_write_packed_first_data_end<TypeParam>(max_sub_cmds, data_sizeB, l1_alignment), kDispatchPageSize)
                << "data_words=" << data_words;
        }
        EXPECT_GT(
            get_write_packed_first_data_end<TypeParam>(max_sub_cmds + 1, data_sizeB, l1_alignment), kDispatchPageSize)
            << "data_words=" << data_words;
    }

    // A payload that fills the page leaves no room for the header.
    EXPECT_EQ(
        calculator.get_max_write_packed_sub_cmds<TypeParam>(
            kDispatchPageSize / sizeof(uint32_t),
            max_prefetch_command_size,
            packed_write_max_unicast_sub_cmds,
            /*no_stride*/ true),
        0u);
    // Small payloads are still limited by the sub cmd cap.
    constexpr uint32_t small_unicast_cap = 64;
    const uint32_t small_cap = std::is_same_v<TypeParam, CQDispatchWritePackedUnicastSubCmd>
                                   ? small_unicast_cap
                                   : get_packed_write_max_multicast_sub_cmds(small_unicast_cap);
    EXPECT_EQ(
        calculator.get_max_write_packed_sub_cmds<TypeParam>(
            4, max_prefetch_command_size, small_unicast_cap, /*no_stride*/ true),
        small_cap);
}

TYPED_TEST(WritePackedCommandTest, CPU_AddDispatchWritePackedNoStrideRejectsPageOverflow) {
    constexpr uint32_t max_prefetch_command_size = 1u << 20;
    constexpr uint32_t packed_write_max_unicast_sub_cmds = 4096;
    constexpr uint32_t data_sizeB = 2048;
    DeviceCommandCalculator calculator(this->ctx_);
    const uint32_t max_sub_cmds = calculator.get_max_write_packed_sub_cmds<TypeParam>(
        data_sizeB / sizeof(uint32_t),
        max_prefetch_command_size,
        packed_write_max_unicast_sub_cmds,
        /*no_stride*/ true);
    ASSERT_GT(max_sub_cmds, 0u);

    EXPECT_NO_THROW(build_no_stride_write_packed<TypeParam>(
        this->ctx_, max_sub_cmds, data_sizeB, packed_write_max_unicast_sub_cmds));
    EXPECT_ANY_THROW(build_no_stride_write_packed<TypeParam>(
        this->ctx_, max_sub_cmds + 1, data_sizeB, packed_write_max_unicast_sub_cmds));
}

TYPED_TEST(WritePackedCommandTest, CPU_AddDispatchWritePackedRejectsDataLargerThanPage) {
    constexpr uint32_t data_sizeB = kDispatchPageSize + sizeof(uint32_t);
    constexpr uint32_t num_sub_cmds = 1;
    constexpr uint32_t packed_write_max_unicast_sub_cmds = 64;
    DeviceCommandCalculator calculator(this->ctx_);
    calculator.add_dispatch_write_packed<TypeParam>(
        num_sub_cmds, data_sizeB, packed_write_max_unicast_sub_cmds, /*no_stride*/ false);

    HostMemDeviceCommand command(this->ctx_, calculator.write_offset_bytes());
    std::vector<TypeParam> sub_cmds(num_sub_cmds);
    std::vector<uint8_t> data(data_sizeB);
    std::vector<std::pair<const void*, uint32_t>> data_collection{{data.data(), data_sizeB}};
    EXPECT_ANY_THROW(command.add_dispatch_write_packed<TypeParam>(
        0,
        num_sub_cmds,
        0,
        data_sizeB,
        0,
        sub_cmds,
        data_collection,
        packed_write_max_unicast_sub_cmds,
        0,
        /*no_stride*/ false));
}

TEST_F(DeviceCommandTest, CPU_AddDispatchWritePackedLarge) {
    {
        DeviceCommandCalculator calculator(ctx_);
        calculator.add_dispatch_write_packed_large(1);

        HostMemDeviceCommand command(ctx_, calculator.write_offset_bytes());
        std::vector<CQDispatchWritePackedLargeSubCmd> sub_cmds(1);
        command.add_dispatch_write_packed_large(CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_TYPE_UNKNOWN, 0, 1, sub_cmds);
        EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
    }
    {
        DeviceCommandCalculator calculator(ctx_);
        calculator.add_dispatch_write_packed_large(1, 4);

        HostMemDeviceCommand command(ctx_, calculator.write_offset_bytes());
        std::vector<CQDispatchWritePackedLargeSubCmd> sub_cmds(1);

        uint8_t data[4] = {};
        std::vector<ttsl::Span<const uint8_t>> data_collection{{data, 4}};
        command.add_dispatch_write_packed_large(
            CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_TYPE_UNKNOWN, 0, 1, sub_cmds, data_collection, nullptr);
        EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
    }
}

TYPED_TEST(WritePackedCommandTest, CPU_RandomAddDispatchWritePacked) {
    srand(0);
    for (size_t i = 0; i < 100; i++) {
        DeviceCommandCalculator calculator(this->ctx_);
        uint32_t random_start = (rand() % 4) % 32;
        calculator.add_data(random_start);
        uint32_t num_sub_cmds = (rand() % 100) + 1;
        uint32_t sub_cmd_sizeB = (rand() % 2000) + 1;
        uint32_t max_prefetch_command_size = 16384;
        uint32_t packed_write_max_unicast_sub_cmds = 64;

        std::vector<std::pair<uint32_t, uint32_t>> packed_cmd_payloads;
        calculator.insert_write_packed_payloads<TypeParam>(
            num_sub_cmds,
            sub_cmd_sizeB,
            max_prefetch_command_size,
            packed_write_max_unicast_sub_cmds,
            packed_cmd_payloads);

        uint32_t data[2001] = {};
        std::vector<std::pair<const void*, uint32_t>> data_collection;
        data_collection.reserve(num_sub_cmds);
        for (size_t j = 0; j < num_sub_cmds; j++) {
            data_collection.push_back({data, sub_cmd_sizeB});
        }

        HostMemDeviceCommand command(this->ctx_, calculator.write_offset_bytes());
        command.add_data(data, 0, random_start);
        uint32_t curr_sub_cmd_idx = 0;
        for (const auto& [sub_cmd_ct, payload_size] : packed_cmd_payloads) {
            std::vector<TypeParam> sub_cmds(sub_cmd_ct);
            command.add_dispatch_write_packed<TypeParam>(
                0,
                sub_cmd_ct,
                0,
                sub_cmd_sizeB,
                payload_size,
                sub_cmds,
                data_collection,
                packed_write_max_unicast_sub_cmds,
                0,
                false,
                curr_sub_cmd_idx);
            curr_sub_cmd_idx += sub_cmd_ct;
        }
        EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
    }
}

}  // namespace tt::tt_metal
