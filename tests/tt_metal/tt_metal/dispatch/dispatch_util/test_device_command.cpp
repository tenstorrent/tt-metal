// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include "tt_metal/impl/dispatch/vector_aligned.hpp"
#include <utility>
#include <vector>

#include <tt_stl/span.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/impl/dispatch/device_command_calculator.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_commands.hpp"
#include "tt_metal/impl/program/program_config_command_generator.hpp"
#include "tt_metal/impl/program/program_command_sequence.hpp"
#include "tt_metal/impl/program/program_command_sequence_writer.hpp"

namespace tt::tt_metal {

// These are host-only command-layout checks, but they link into unit_tests_dispatch which runs against
// hardware, so resolving the global MetalContext here is an acceptable single documented bridge.
class DeviceCommandTest : public ::testing::Test {
protected:
    MetalContext& ctx_ = MetalContext::instance();
};

class RecordingCommandQueueManager {
public:
    void* issue_queue_reserve(uint32_t size_bytes, uint32_t) {
        issue_queue_reserve_sizes.push_back(size_bytes);
        return nullptr;
    }

    uint32_t get_issue_queue_write_ptr(uint32_t) const { return issue_queue_write_ptr; }

    void cq_write(const void*, uint32_t size_bytes, uint32_t write_ptr) {
        cq_write_sizes.push_back(size_bytes);
        cq_write_ptrs.push_back(write_ptr);
    }

    void issue_queue_push_back(uint32_t size_bytes, uint32_t) {
        issue_queue_push_sizes.push_back(size_bytes);
    }

    void fetch_queue_reserve_back(uint32_t) { ++fetch_queue_reserve_count; }

    void fetch_queue_write(uint32_t size_bytes, uint32_t, bool = false) {
        fetch_queue_write_sizes.push_back(size_bytes);
    }

    static constexpr uint32_t issue_queue_write_ptr = 0x1000;
    std::vector<uint32_t> issue_queue_reserve_sizes;
    std::vector<uint32_t> cq_write_sizes;
    std::vector<uint32_t> cq_write_ptrs;
    std::vector<uint32_t> issue_queue_push_sizes;
    size_t fetch_queue_reserve_count = 0;
    std::vector<uint32_t> fetch_queue_write_sizes;
};

TEST_F(DeviceCommandTest, CPU_ProgramConfigBatchingSplitsAtPrefetchEntryLimit) {
    constexpr uint32_t max_prefetch_command_size = 131072;
    const uint32_t pcie_alignment = ctx_.hal().get_alignment(HalMemType::HOST);
    const uint32_t l1_alignment = ctx_.hal().get_alignment(HalMemType::L1);
    constexpr uint32_t transfer_size = 60000;
    constexpr uint32_t transfer_address_stride = 70000;
    constexpr uint32_t transfer_count = 4;

    std::array<std::vector<uint8_t>, transfer_count> transfer_payloads;
    program_dispatch::BatchedTransfers batched_transfers;
    auto& transfers_for_destination = batched_transfers[{/*noc_xy_addr=*/0x1234, /*num_mcast_dests=*/1}];
    for (uint32_t transfer_index = 0; transfer_index < transfer_count; ++transfer_index) {
        transfer_payloads[transfer_index].resize(transfer_size);
        const uint32_t start_address = transfer_index * transfer_address_stride;
        transfers_for_destination.emplace(
            start_address,
            std::vector<program_dispatch::Transfer>{program_dispatch::Transfer{
                .start = start_address,
                .data = ttsl::Span<const uint8_t>(
                    transfer_payloads[transfer_index].data(), transfer_payloads[transfer_index].size())}});
    }

    DeviceCommandCalculator calculator(pcie_alignment, l1_alignment);
    program_dispatch::BatchedTransferGenerator generator(program_dispatch::ProgramConfigCommandOptions{
        .pcie_alignment = pcie_alignment,
        .l1_alignment = l1_alignment,
        .max_prefetch_command_size = max_prefetch_command_size,
        .watcher_assert_enabled = false});
    generator.construct_commands(batched_transfers, calculator);
    ASSERT_EQ(generator.command_count(), 2);

    ProgramCommandSequence program_commands(ctx_);
    generator.assemble_commands(program_commands, program_commands.program_config_buffer_command_sequences);
    ASSERT_EQ(program_commands.program_config_buffer_command_sequences.size(), generator.command_count());

    std::vector<uint32_t> command_sizes;
    for (const HostMemDeviceCommand& command : program_commands.program_config_buffer_command_sequences) {
        EXPECT_EQ(command.size_bytes(), command.write_offset_bytes());
        EXPECT_LE(command.size_bytes(), max_prefetch_command_size);
        command_sizes.push_back(command.size_bytes());
    }
    EXPECT_EQ(program_commands.get_program_config_buffer_size(), calculator.write_offset_bytes());

    constexpr uint32_t command_queue_id = 7;
    constexpr bool stall_first = false;
    constexpr bool stall_before_program = false;
    constexpr bool send_binary = false;
    const program_dispatch::queue_write_detail::ProgramCommandWritePlan write_plan =
        program_dispatch::queue_write_detail::make_program_command_write_plan(
            program_commands,
            stall_first,
            stall_before_program,
            send_binary,
            max_prefetch_command_size);
    ASSERT_FALSE(write_plan.one_shot);

    RecordingCommandQueueManager manager;
    program_dispatch::queue_write_detail::write_program_command_sequence_to_queue(
        program_commands,
        manager,
        command_queue_id,
        stall_first,
        stall_before_program,
        send_binary,
        write_plan);

    EXPECT_EQ(manager.issue_queue_reserve_sizes, command_sizes);
    EXPECT_EQ(manager.cq_write_sizes, command_sizes);
    EXPECT_EQ(
        manager.cq_write_ptrs,
        std::vector<uint32_t>(command_sizes.size(), RecordingCommandQueueManager::issue_queue_write_ptr));
    EXPECT_EQ(manager.issue_queue_push_sizes, command_sizes);
    EXPECT_EQ(manager.fetch_queue_reserve_count, command_sizes.size());
    EXPECT_EQ(manager.fetch_queue_write_sizes, command_sizes);
}

TEST_F(DeviceCommandTest, CPU_OneShotPlanCountsBothStallPlacements) {
    constexpr uint32_t stall_command_size = 16;
    constexpr uint32_t max_prefetch_command_size = 131072;
    constexpr uint32_t command_queue_id = 7;
    constexpr bool stall_first = true;
    constexpr bool stall_before_program = true;
    constexpr bool send_binary = false;

    ProgramCommandSequence program_commands(ctx_);
    program_commands.stall_command_sequences[program_commands.current_stall_seq_idx] =
        HostMemDeviceCommand(ctx_, stall_command_size);
    const program_dispatch::queue_write_detail::ProgramCommandWritePlan write_plan =
        program_dispatch::queue_write_detail::make_program_command_write_plan(
            program_commands,
            stall_first,
            stall_before_program,
            send_binary,
            max_prefetch_command_size);
    ASSERT_TRUE(write_plan.one_shot);
    ASSERT_EQ(write_plan.fetch_size, 2 * stall_command_size);

    RecordingCommandQueueManager manager;
    program_dispatch::queue_write_detail::write_program_command_sequence_to_queue(
        program_commands,
        manager,
        command_queue_id,
        stall_first,
        stall_before_program,
        send_binary,
        write_plan);

    EXPECT_EQ(manager.issue_queue_reserve_sizes, std::vector<uint32_t>{write_plan.fetch_size});
    EXPECT_EQ(manager.cq_write_sizes, (std::vector<uint32_t>{stall_command_size, stall_command_size}));
    EXPECT_EQ(
        manager.cq_write_ptrs,
        (std::vector<uint32_t>{
            RecordingCommandQueueManager::issue_queue_write_ptr,
            RecordingCommandQueueManager::issue_queue_write_ptr + stall_command_size}));
    EXPECT_EQ(manager.issue_queue_push_sizes, std::vector<uint32_t>{write_plan.fetch_size});
    EXPECT_EQ(manager.fetch_queue_reserve_count, 1);
    EXPECT_EQ(manager.fetch_queue_write_sizes, std::vector<uint32_t>{write_plan.fetch_size});
}

TEST_F(DeviceCommandTest, CPU_ProgramConfigBatchingRejectsOversizedTransfer) {
    constexpr uint32_t max_prefetch_command_size = 50000;
    constexpr uint32_t transfer_size = 60000;
    const uint32_t pcie_alignment = ctx_.hal().get_alignment(HalMemType::HOST);
    const uint32_t l1_alignment = ctx_.hal().get_alignment(HalMemType::L1);

    std::vector<uint8_t> transfer_payload(transfer_size);
    program_dispatch::BatchedTransfers batched_transfers;
    batched_transfers[{/*noc_xy_addr=*/0x1234, /*num_mcast_dests=*/1}].emplace(
        0,
        std::vector<program_dispatch::Transfer>{program_dispatch::Transfer{
            .start = 0, .data = ttsl::Span<const uint8_t>(transfer_payload.data(), transfer_payload.size())}});

    DeviceCommandCalculator calculator(pcie_alignment, l1_alignment);
    program_dispatch::BatchedTransferGenerator generator(program_dispatch::ProgramConfigCommandOptions{
        .pcie_alignment = pcie_alignment,
        .l1_alignment = l1_alignment,
        .max_prefetch_command_size = max_prefetch_command_size,
        .watcher_assert_enabled = false});
    try {
        generator.construct_commands(batched_transfers, calculator);
        FAIL() << "Expected an oversized program-configuration transfer to be rejected";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string_view(error.what()).find("single program-configuration transfer"), std::string_view::npos);
    }
}

TEST_F(DeviceCommandTest, CPU_CommandContextSurvivesMove) {
    constexpr uint32_t transfer_size = 17;
    const uint32_t l1_alignment = ctx_.hal().get_alignment(HalMemType::L1);

    DeviceCommandCalculator calculator(ctx_);
    calculator.add_dispatch_write_packed_large(/*num_sub_cmds=*/1, transfer_size);

    HostMemDeviceCommand original_command(ctx_, calculator.write_offset_bytes());
    HostMemDeviceCommand moved_command(std::move(original_command));
    std::vector<CQDispatchWritePackedLargeSubCmd> subcommands(1);
    std::array<uint8_t, transfer_size> payload{};
    const std::vector<ttsl::Span<const uint8_t>> data_collection{
        ttsl::Span<const uint8_t>(payload.data(), payload.size())};
    std::vector<uint8_t*> data_collection_locations;
    moved_command.add_dispatch_write_packed_large(
        CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_TYPE_CBS_SEMS_CRTAS,
        l1_alignment,
        static_cast<uint16_t>(subcommands.size()),
        subcommands,
        data_collection,
        &data_collection_locations);

    EXPECT_EQ(moved_command.size_bytes(), moved_command.write_offset_bytes());
}

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
    command.add_dispatch_go_signal_mcast(0, 0, 0, 0, 0, 0, DispatcherSelect::DISPATCH_MASTER);
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

TEST_F(DeviceCommandTest, CPU_AddDispatchSetGoSignalNocData) {
    DeviceCommandCalculator calculator(ctx_);
    calculator.add_dispatch_set_go_signal_noc_data(5);

    HostMemDeviceCommand command(ctx_, calculator.write_offset_bytes());
    vector_aligned<uint32_t> data(5);
    command.add_dispatch_set_go_signal_noc_data(data, DispatcherSelect::DISPATCH_MASTER);
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
