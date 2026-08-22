// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include "tt_metal/impl/dispatch/vector_aligned.hpp"
#include <utility>
#include <vector>

#include <tt_stl/span.hpp>
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/impl/dispatch/device_command_calculator.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_commands.hpp"
#include "tt_metal/impl/program/program_command_sequence.hpp"

namespace tt::tt_metal {

TEST(DeviceCommandTest, ProgramConfigBatchingSplitsAtPrefetchEntryLimit) {
    constexpr uint32_t max_prefetch_command_size = 131072;
    constexpr uint32_t pcie_alignment = 16;
    constexpr uint32_t l1_alignment = 16;
    constexpr std::array<uint32_t, 4> command_sizes = {70016, 60000, 60000, 29056};

    EXPECT_TRUE(dispatch_write_packed_large_requires_new_command(
        /*current_subcommand_count=*/2,
        /*current_data_size_bytes=*/120000,
        /*next_data_size_bytes=*/99072,
        pcie_alignment,
        l1_alignment,
        max_prefetch_command_size));
    EXPECT_FALSE(dispatch_write_packed_large_requires_new_command(
        /*current_subcommand_count=*/1,
        /*current_data_size_bytes=*/60000,
        /*next_data_size_bytes=*/60000,
        pcie_alignment,
        l1_alignment,
        max_prefetch_command_size));
    EXPECT_TRUE(dispatch_write_packed_large_requires_new_command(
        /*current_subcommand_count=*/CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_MAX_SUB_CMDS,
        /*current_data_size_bytes=*/16,
        /*next_data_size_bytes=*/16,
        pcie_alignment,
        l1_alignment,
        max_prefetch_command_size));
    EXPECT_LE(
        dispatch_write_packed_large_size_bytes(1, 99072, pcie_alignment, l1_alignment), max_prefetch_command_size);

    EXPECT_EQ(std::accumulate(command_sizes.begin(), command_sizes.end(), 0U), 219072);
    EXPECT_TRUE(std::ranges::all_of(command_sizes, [](uint32_t size) { return size <= max_prefetch_command_size; }));
}

TEST(DeviceCommandTest, ProgramConfigCommandsRetainPrefetchEntryBoundaries) {
    constexpr size_t command_count = 4;
    ProgramCommandSequence program_commands;
    for (size_t command_index = 0; command_index < command_count; ++command_index) {
        program_commands.program_config_buffer_command_sequences.emplace_back();
    }

    size_t visited_command_count = 0;
    program_commands.visit_program_config_buffer_commands(
        [&visited_command_count](const HostMemDeviceCommand&) { ++visited_command_count; });
    EXPECT_EQ(visited_command_count, command_count);
}

TEST(DeviceCommandTest, AddDispatchWait) {
    DeviceCommandCalculator calculator;
    calculator.add_dispatch_wait();

    HostMemDeviceCommand command(calculator.write_offset_bytes());
    command.add_dispatch_wait(0, 0, 0, 0, 0);
    EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
}

TEST(DeviceCommandTest, AddDispatchWaitWithPrefetchStall) {
    DeviceCommandCalculator calculator;
    calculator.add_dispatch_wait_with_prefetch_stall();

    HostMemDeviceCommand command(calculator.write_offset_bytes());
    command.add_dispatch_wait_with_prefetch_stall(0, 0, 0, 0, 0);
    EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
}

TEST(DeviceCommandTest, AddPrefetchRelayLinear) {
    DeviceCommandCalculator calculator;
    calculator.add_prefetch_relay_linear();

    HostMemDeviceCommand command(calculator.write_offset_bytes());
    command.add_prefetch_relay_linear(0, 0, 0);
    EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
}

TEST(DeviceCommandTest, AddData) {
    DeviceCommandCalculator calculator;
    calculator.add_data(32);

    HostMemDeviceCommand command(calculator.write_offset_bytes());
    uint32_t data[1] = {};
    command.add_data(data, 4, 32);
    EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
}

TEST(DeviceCommandTest, AddDispatchWriteLinear) {
    {
        DeviceCommandCalculator calculator;
        calculator.add_dispatch_write_linear<false, false>(5);

        HostMemDeviceCommand command(calculator.write_offset_bytes());
        command.add_dispatch_write_linear<false, false>(0, 0, 0, 5);
        EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
    }
    {
        DeviceCommandCalculator calculator;
        calculator.add_dispatch_write_linear<true, true>(5);

        HostMemDeviceCommand command(calculator.write_offset_bytes());
        uint32_t data[2] = {};
        command.add_dispatch_write_linear<true, true>(0, 0, 0, 5, data);
        EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
    }
    {
        DeviceCommandCalculator calculator;
        calculator.add_dispatch_write_linear<true, false>(5);

        HostMemDeviceCommand command(calculator.write_offset_bytes());
        command.add_dispatch_write_linear<true, false>(0, 0, 0, 5);
        EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
    }
    {
        DeviceCommandCalculator calculator;
        calculator.add_dispatch_write_linear<false, true>(5);

        HostMemDeviceCommand command(calculator.write_offset_bytes());
        uint32_t data[2] = {};
        command.add_dispatch_write_linear<false, true>(0, 0, 0, 5, data);
        EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
    }
}

TEST(DeviceCommandTest, AddDispatchGoSignalMcast) {
    DeviceCommandCalculator calculator;
    calculator.add_dispatch_go_signal_mcast();

    HostMemDeviceCommand command(calculator.write_offset_bytes());
    command.add_dispatch_go_signal_mcast(0, 0, 0, 0, 0, 0, DispatcherSelect::DISPATCH_MASTER);
    EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
}

TEST(DeviceCommandTest, AddNotifyDispatchSGoSignalCmd) {
    DeviceCommandCalculator calculator;
    calculator.add_notify_dispatch_s_go_signal_cmd();

    HostMemDeviceCommand command(calculator.write_offset_bytes());
    command.add_notify_dispatch_s_go_signal_cmd(0, 0);
    EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
}

TEST(DeviceCommandTest, AddDispatchSetNumWorkerSems) {
    DeviceCommandCalculator calculator;
    calculator.add_dispatch_set_num_worker_sems();

    HostMemDeviceCommand command(calculator.write_offset_bytes());
    command.add_dispatch_set_num_worker_sems(0, DispatcherSelect::DISPATCH_MASTER);
    EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
}

TEST(DeviceCommandTest, AddDispatchSetSubDeviceWorkerCounts) {
    constexpr uint32_t num_sub_devices = 3;
    std::array<uint32_t, num_sub_devices> workers_per_sub_device = {4, 7, 2};

    DeviceCommandCalculator calculator;
    calculator.add_dispatch_set_sub_device_worker_counts(num_sub_devices);

    HostMemDeviceCommand command(calculator.write_offset_bytes());
    command.add_dispatch_set_sub_device_worker_counts(
        ttsl::Span<const uint32_t>(workers_per_sub_device.data(), workers_per_sub_device.size()),
        DispatcherSelect::DISPATCH_MASTER);
    EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
}

TEST(DeviceCommandTest, AddDispatchSetGoSignalNocData) {
    DeviceCommandCalculator calculator;
    calculator.add_dispatch_set_go_signal_noc_data(5);

    HostMemDeviceCommand command(calculator.write_offset_bytes());
    vector_aligned<uint32_t> data(5);
    command.add_dispatch_set_go_signal_noc_data(data, DispatcherSelect::DISPATCH_MASTER);
    EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
}

TEST(DeviceCommandTest, AddDispatchSetWriteOffsets) {
    DeviceCommandCalculator calculator;
    calculator.add_dispatch_set_write_offsets(CQ_DISPATCH_MAX_WRITE_OFFSETS);

    HostMemDeviceCommand command(calculator.write_offset_bytes());
    std::vector<uint32_t> offsets(CQ_DISPATCH_MAX_WRITE_OFFSETS, 0);
    command.add_dispatch_set_write_offsets(offsets);
    EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
}

TEST(DeviceCommandTest, AddDispatchTerminate) {
    DeviceCommandCalculator calculator;
    calculator.add_dispatch_terminate();

    HostMemDeviceCommand command(calculator.write_offset_bytes());
    command.add_dispatch_terminate();
    EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
}

TEST(DeviceCommandTest, AddDispatchWritePaged) {
    {
        DeviceCommandCalculator calculator;
        calculator.add_dispatch_write_paged<false>(1, 5);
        // Do PCIE alignment for out-of-line data
        calculator.add_alignment();
        HostMemDeviceCommand command(calculator.write_offset_bytes());
        command.add_dispatch_write_paged<false>(false, 0, 0, 0, 1, 5);
        command.align_write_offset();
        EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
    }
    {
        DeviceCommandCalculator calculator;
        calculator.add_dispatch_write_paged<true>(1, 5);

        HostMemDeviceCommand command(calculator.write_offset_bytes());
        uint32_t data[2] = {};
        command.add_dispatch_write_paged<true>(false, 0, 0, 0, 1, 5, data);
        EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
    }
}

TEST(DeviceCommandTest, AddPrefetchRelayPaged) {
    DeviceCommandCalculator calculator;
    calculator.add_prefetch_relay_paged();

    HostMemDeviceCommand command(calculator.write_offset_bytes());
    command.add_prefetch_relay_paged(0, 0, 0, 0, 0, 0);
    EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
}

TEST(DeviceCommandTest, AddPrefetchRelayPagedPacked) {
    DeviceCommandCalculator calculator;
    calculator.add_prefetch_relay_paged_packed(1);

    HostMemDeviceCommand command(calculator.write_offset_bytes());
    std::vector<CQPrefetchRelayPagedPackedSubCmd> sub_cmds(1);
    command.add_prefetch_relay_paged_packed(0, sub_cmds, 1);
    EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
}

template <typename T>
class WritePackedCommandTest : public ::testing::Test {};

using TestTypes = testing::Types<CQDispatchWritePackedMulticastSubCmd, CQDispatchWritePackedUnicastSubCmd>;
TYPED_TEST_SUITE(WritePackedCommandTest, TestTypes);

TYPED_TEST(WritePackedCommandTest, AddDispatchWritePacked) {
    {
        DeviceCommandCalculator calculator;
        calculator.add_dispatch_write_packed<TypeParam>(2, 5, 100, /*no_stride*/ false);

        HostMemDeviceCommand command(calculator.write_offset_bytes());
        std::vector<TypeParam> sub_cmds(2);
        uint32_t data[1] = {};
        std::vector<std::pair<const void*, uint32_t>> data_collection{{data, 4}, {data, 4}};
        command.add_dispatch_write_packed<TypeParam>(0, 2, 0, 5, 0, sub_cmds, data_collection, 100, 0, false);
        EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
    }
    {
        DeviceCommandCalculator calculator;
        calculator.add_dispatch_write_packed<TypeParam>(2, 5, 100, /*no_stride*/ true);

        HostMemDeviceCommand command(calculator.write_offset_bytes());
        std::vector<TypeParam> sub_cmds(2);
        uint32_t data[1] = {};
        std::vector<std::pair<const void*, uint32_t>> data_collection{{data, 4}};
        command.add_dispatch_write_packed<TypeParam>(0, 2, 0, 5, 0, sub_cmds, data_collection, 100, 0, true);
        EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
    }
}

TEST(DeviceCommandTest, AddDispatchWritePackedLarge) {
    {
        DeviceCommandCalculator calculator;
        calculator.add_dispatch_write_packed_large(1);

        HostMemDeviceCommand command(calculator.write_offset_bytes());
        std::vector<CQDispatchWritePackedLargeSubCmd> sub_cmds(1);
        command.add_dispatch_write_packed_large(CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_TYPE_UNKNOWN, 0, 1, sub_cmds);
        EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
    }
    {
        DeviceCommandCalculator calculator;
        calculator.add_dispatch_write_packed_large(1, 4);

        HostMemDeviceCommand command(calculator.write_offset_bytes());
        std::vector<CQDispatchWritePackedLargeSubCmd> sub_cmds(1);

        uint8_t data[4] = {};
        std::vector<ttsl::Span<const uint8_t>> data_collection{{data, 4}};
        command.add_dispatch_write_packed_large(
            CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_TYPE_UNKNOWN, 0, 1, sub_cmds, data_collection, nullptr);
        EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
    }
}

TYPED_TEST(WritePackedCommandTest, RandomAddDispatchWritePacked) {
    srand(0);
    for (size_t i = 0; i < 100; i++) {
        DeviceCommandCalculator calculator;
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

        HostMemDeviceCommand command(calculator.write_offset_bytes());
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
