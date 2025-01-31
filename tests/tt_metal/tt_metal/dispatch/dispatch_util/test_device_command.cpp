// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/device_command.hpp>
#include "cq_commands.hpp"
#include "memcpy.hpp"
#include <gtest/gtest.h>
#include "tt_metal/impl/dispatch/device_command_calculator.hpp"

TEST(DeviceCommandTest, AddDispatchWait) {
    DeviceCommandCalculator calculator;
    calculator.add_dispatch_wait();

    HostMemDeviceCommand command(calculator.write_offset_bytes());
    command.add_dispatch_wait(0, 0, 0);
    EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
}

TEST(DeviceCommandTest, AddDispatchWaitWithPrefetchStall) {
    DeviceCommandCalculator calculator;
    calculator.add_dispatch_wait_with_prefetch_stall();

    HostMemDeviceCommand command(calculator.write_offset_bytes());
    command.add_dispatch_wait_with_prefetch_stall(0, 0, 0);
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

TEST(DeviceCommandTest, AddDispatchSetGoSignalNocData) {
    DeviceCommandCalculator calculator;
    calculator.add_dispatch_set_go_signal_noc_data(5);

    HostMemDeviceCommand command(calculator.write_offset_bytes());
    vector_memcpy_aligned<uint32_t> data(5);
    command.add_dispatch_set_go_signal_noc_data(data, DispatcherSelect::DISPATCH_MASTER);
    EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
}

TEST(DeviceCommandTest, AddDispatchSetWriteOffsets) {
    DeviceCommandCalculator calculator;
    calculator.add_dispatch_set_write_offsets();

    HostMemDeviceCommand command(calculator.write_offset_bytes());
    command.add_dispatch_set_write_offsets(0, 0, 0);
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

        HostMemDeviceCommand command(calculator.write_offset_bytes());
        command.add_dispatch_write_paged<false>(0, 0, 0, 0, 1, 5);
        EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
    }
    {
        DeviceCommandCalculator calculator;
        calculator.add_dispatch_write_paged<true>(1, 5);

        HostMemDeviceCommand command(calculator.write_offset_bytes());
        uint32_t data[2] = {};
        command.add_dispatch_write_paged<true>(0, 0, 0, 0, 1, 5, data);
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

TEST(DeviceCommandTest, AddDispatchWritePacked) {
    {
        DeviceCommandCalculator calculator;
        calculator.add_dispatch_write_packed<CQDispatchWritePackedMulticastSubCmd>(2, 5, 100, /*no_stride*/ false);

        HostMemDeviceCommand command(calculator.write_offset_bytes());
        std::vector<CQDispatchWritePackedMulticastSubCmd> sub_cmds(2);
        uint32_t data[1] = {};
        std::vector<std::pair<const void*, uint32_t>> data_collection{{data, 4}, {data, 4}};
        command.add_dispatch_write_packed<CQDispatchWritePackedMulticastSubCmd>(
            2, 0, 5, 0, sub_cmds, data_collection, 100, 0, false);
        EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
    }
    {
        DeviceCommandCalculator calculator;
        calculator.add_dispatch_write_packed<CQDispatchWritePackedMulticastSubCmd>(2, 5, 100, /*no_stride*/ true);

        HostMemDeviceCommand command(calculator.write_offset_bytes());
        std::vector<CQDispatchWritePackedMulticastSubCmd> sub_cmds(2);
        uint32_t data[1] = {};
        std::vector<std::pair<const void*, uint32_t>> data_collection{{data, 4}};
        command.add_dispatch_write_packed<CQDispatchWritePackedMulticastSubCmd>(
            2, 0, 5, 0, sub_cmds, data_collection, 100, 0, true);
        EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
    }
}

TEST(DeviceCommandTest, AddDispatchWritePackedLarge) {
    {
        DeviceCommandCalculator calculator;
        calculator.add_dispatch_write_packed_large(1);

        HostMemDeviceCommand command(calculator.write_offset_bytes());
        std::vector<CQDispatchWritePackedLargeSubCmd> sub_cmds(1);
        command.add_dispatch_write_packed_large(0, 1, sub_cmds);
        EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
    }
    {
        DeviceCommandCalculator calculator;
        calculator.add_dispatch_write_packed_large(1, 4);

        HostMemDeviceCommand command(calculator.write_offset_bytes());
        std::vector<CQDispatchWritePackedLargeSubCmd> sub_cmds(1);

        uint8_t data[4] = {};
        std::vector<tt::stl::Span<const uint8_t>> data_collection{{data, 4}};
        command.add_dispatch_write_packed_large(0, 1, sub_cmds, data_collection, nullptr);
        EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
    }
}
