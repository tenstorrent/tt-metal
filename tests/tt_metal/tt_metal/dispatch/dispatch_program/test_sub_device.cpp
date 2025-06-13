// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stddef.h>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/event.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/sub_device.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include "command_queue_fixture.hpp"
#include <tt-metalium/data_types.hpp>
#include "dispatch_test_utils.hpp"
#include "gtest/gtest.h"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/kernel_types.hpp>
#include "llrt.hpp"
#include "multi_command_queue_fixture.hpp"
#include <tt-metalium/program.hpp>
#include <tt-metalium/runtime_args_data.hpp>
#include <tt_stl/span.hpp>
#include <tt_stl/strong_type.hpp>
#include "sub_device_test_utils.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/stimulus.hpp"
#include "umd/device/types/xy_pair.h"

namespace tt::tt_metal {

constexpr uint32_t k_local_l1_size = 3200;
const std::string k_coordinates_kernel_path = "tests/tt_metal/tt_metal/test_kernels/misc/read_my_coordinates.cpp";

TEST_F(CommandQueueSingleCardFixture, TensixTestSubDeviceCBAllocation) {
    auto* device = devices_[0];
    CoreRangeSet sharded_cores_1 = CoreRange({0, 0}, {2, 2});
    SubDevice sub_device_1(std::array{sharded_cores_1});
    auto sub_device_manager_1 = device->create_sub_device_manager({sub_device_1}, k_local_l1_size);
    DeviceAddr l1_unreserved_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    DeviceAddr l1_max_size = device->l1_size_per_core();
    DeviceAddr l1_total_size = l1_max_size - l1_unreserved_base;
    device->load_sub_device_manager(sub_device_manager_1);
    uint32_t global_buffer_size = l1_total_size - k_local_l1_size * 2;
    ShardSpecBuffer global_shard_spec_buffer =
        ShardSpecBuffer(sharded_cores_1, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {sharded_cores_1.num_cores(), 1});
    ShardedBufferConfig global_shard_config = {
        device,
        sharded_cores_1.num_cores() * global_buffer_size,
        global_buffer_size,
        BufferType::L1,
        TensorMemoryLayout::HEIGHT_SHARDED,
        global_shard_spec_buffer};

    auto global_buffer = CreateBuffer(global_shard_config);
    Program program = CreateProgram();

    uint32_t cb_size = k_local_l1_size;
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(k_local_l1_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, hal::get_l1_alignment());
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, sharded_cores_1, cb_src0_config);

    program.allocate_circular_buffers(device);
    detail::ValidateCircularBufferRegion(program, device);
    UpdateCircularBufferTotalSize(program, cb_src0, k_local_l1_size * 3);
    program.allocate_circular_buffers(device);
    EXPECT_THROW(detail::ValidateCircularBufferRegion(program, device), std::exception);
    global_buffer.reset();
    detail::ValidateCircularBufferRegion(program, device);
    ShardSpecBuffer local_shard_spec_buffer =
        ShardSpecBuffer(sharded_cores_1, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {sharded_cores_1.num_cores(), 1});

    uint32_t local_buffer_size = k_local_l1_size / 2;
    ShardedBufferConfig local_shard_config = {
        device,
        sharded_cores_1.num_cores() * local_buffer_size,
        local_buffer_size,
        BufferType::L1,
        TensorMemoryLayout::HEIGHT_SHARDED,
        local_shard_spec_buffer};

    auto local_buffer = CreateBuffer(local_shard_config, SubDeviceId{0});
    EXPECT_THROW(detail::ValidateCircularBufferRegion(program, device), std::exception);
    UpdateCircularBufferTotalSize(program, cb_src0, k_local_l1_size / 4);
    program.allocate_circular_buffers(device);
    detail::ValidateCircularBufferRegion(program, device);
}

void test_sub_device_synchronization(IDevice* device) {
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})})});
    CoreRangeSet sharded_cores_1 = CoreRange({0, 0}, {2, 2});

    auto sharded_cores_1_vec = corerange_to_cores(sharded_cores_1, std::nullopt, true);

    ShardSpecBuffer shard_spec_buffer_1 =
        ShardSpecBuffer(sharded_cores_1, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {sharded_cores_1.num_cores(), 1});
    uint32_t page_size_1 = 32;
    ShardedBufferConfig shard_config_1 = {
        nullptr,
        sharded_cores_1.num_cores() * page_size_1,
        page_size_1,
        BufferType::L1,
        TensorMemoryLayout::HEIGHT_SHARDED,
        shard_spec_buffer_1};
    auto input_1 =
        tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, shard_config_1.size / sizeof(uint32_t));

    std::array sub_device_ids_to_block = {SubDeviceId{0}};

    auto sub_device_manager = device->create_sub_device_manager({sub_device_1, sub_device_2}, k_local_l1_size);

    shard_config_1.device = device;

    std::vector<CoreCoord> physical_cores_1;
    physical_cores_1.reserve(sharded_cores_1_vec.size());
    for (const auto& core : sharded_cores_1_vec) {
        physical_cores_1.push_back(device->worker_core_from_logical_core(core));
    }

    device->load_sub_device_manager(sub_device_manager);

    auto [program, syncer_core, global_semaphore] = create_single_sync_program(device, sub_device_2);
    EnqueueProgram(device->command_queue(), program, false);
    device->set_sub_device_stall_group(sub_device_ids_to_block);

    auto buffer_1 = CreateBuffer(shard_config_1, sub_device_ids_to_block[0]);

    // Test blocking synchronize doesn't stall
    Synchronize(device, 0);

    // Test blocking write buffer doesn't stall
    EnqueueWriteBuffer(device->command_queue(), buffer_1, input_1, true);

    // Test record event won't cause a stall
    auto event = std::make_shared<Event>();
    EnqueueRecordEvent(device->command_queue(), event);
    Synchronize(device, 0);

    // Test blocking read buffer doesn't stall
    std::vector<uint32_t> output_1;
    EnqueueReadBuffer(device->command_queue(), buffer_1, output_1, true);
    EXPECT_EQ(input_1, output_1);
    auto input_1_it = input_1.begin();
    for (const auto& physical_core : physical_cores_1) {
        auto readback = tt::llrt::read_hex_vec_from_core(device->id(), physical_core, buffer_1->address(), page_size_1);
        EXPECT_TRUE(std::equal(input_1_it, input_1_it + page_size_1 / sizeof(uint32_t), readback.begin()));
        input_1_it += page_size_1 / sizeof(uint32_t);
    }
    auto sem_addr = global_semaphore.address();
    auto physical_syncer_core = device->worker_core_from_logical_core(syncer_core);
    tt::llrt::write_hex_vec_to_core(device->id(), physical_syncer_core, std::vector<uint32_t>{1}, sem_addr);

    // Full synchronization
    device->reset_sub_device_stall_group();
    Synchronize(device);
}

TEST_F(CommandQueueSingleCardFixture, TensixTestSubDeviceSynchronization) {
    auto* device = devices_[0];
    test_sub_device_synchronization(device);
}

TEST_F(MultiCommandQueueSingleDeviceFixture, TensixTestMultiCQSubDeviceSynchronization) {
    test_sub_device_synchronization(device_);
}

TEST_F(CommandQueueSingleCardFixture, TensixTestSubDeviceBasicPrograms) {
    constexpr uint32_t k_num_iters = 5;
    auto* device = devices_[0];
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})})});
    auto sub_device_manager = device->create_sub_device_manager({sub_device_1, sub_device_2}, k_local_l1_size);
    device->load_sub_device_manager(sub_device_manager);

    auto [waiter_program, syncer_program, incrementer_program, global_sem] =
        create_basic_sync_program(device, sub_device_1, sub_device_2);

    for (uint32_t i = 0; i < k_num_iters; i++) {
        EnqueueProgram(device->command_queue(), waiter_program, false);
        device->set_sub_device_stall_group({SubDeviceId{0}});
        // Test blocking on one sub-device
        EnqueueProgram(device->command_queue(), syncer_program, true);
        EnqueueProgram(device->command_queue(), incrementer_program, false);
        device->reset_sub_device_stall_group();
    }
    Synchronize(device);
    detail::DumpDeviceProfileResults(device);
}

TEST_F(CommandQueueSingleCardFixture, TensixTestSubDeviceBasicProgramsReuse) {
    constexpr uint32_t k_num_iters = 5;
    auto* device = devices_[0];
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})})});
    // sub-device 3 and 4 are supersets of sub-device 1 and 2 respectively
    SubDevice sub_device_3(std::array{CoreRangeSet(std::vector{CoreRange({0, 0}, {2, 2}), CoreRange({5, 5}, {5, 5})})});
    SubDevice sub_device_4(std::array{
        CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4}), CoreRange({6, 6}, {6, 6})})});
    auto sub_device_manager_1 = device->create_sub_device_manager({sub_device_1, sub_device_2}, k_local_l1_size);
    auto sub_device_manager_2 = device->create_sub_device_manager({sub_device_4, sub_device_3}, k_local_l1_size);
    device->load_sub_device_manager(sub_device_manager_1);

    auto [waiter_program, syncer_program, incrementer_program, global_sem] =
        create_basic_sync_program(device, sub_device_1, sub_device_2);

    // Run programs on sub-device manager 1
    for (uint32_t i = 0; i < k_num_iters; i++) {
        EnqueueProgram(device->command_queue(), waiter_program, false);
        device->set_sub_device_stall_group({SubDeviceId{0}});
        // Test blocking on one sub-device
        EnqueueProgram(device->command_queue(), syncer_program, true);
        EnqueueProgram(device->command_queue(), incrementer_program, false);
        device->reset_sub_device_stall_group();
    }
    Synchronize(device);

    // Rerun programs on sub-device manager 2
    device->load_sub_device_manager(sub_device_manager_2);
    for (uint32_t i = 0; i < k_num_iters; i++) {
        EnqueueProgram(device->command_queue(), waiter_program, false);
        device->set_sub_device_stall_group({SubDeviceId{1}});
        // Test blocking on one sub-device
        EnqueueProgram(device->command_queue(), syncer_program, true);
        EnqueueProgram(device->command_queue(), incrementer_program, false);
        device->reset_sub_device_stall_group();
    }
    Synchronize(device);
    detail::DumpDeviceProfileResults(device);
}

TEST_F(CommandQueueSingleCardFixture, TensixActiveEthTestSubDeviceBasicEthPrograms) {
    auto* device = devices_[0];
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    uint32_t num_iters = 5;
    if (!does_device_have_active_eth_cores(device)) {
        GTEST_SKIP() << "Skipping test because device " << device->id() << " does not have any active ethernet cores";
    }
    auto eth_core = *device->get_active_ethernet_cores(true).begin();
    SubDevice sub_device_2(std::array{
        CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})}),
        CoreRangeSet(CoreRange(eth_core, eth_core))});
    auto sub_device_manager = device->create_sub_device_manager({sub_device_1, sub_device_2}, k_local_l1_size);
    device->load_sub_device_manager(sub_device_manager);

    auto [waiter_program, syncer_program, incrementer_program, global_sem] =
        create_basic_eth_sync_program(device, sub_device_1, sub_device_2);

    for (uint32_t i = 0; i < num_iters; i++) {
        EnqueueProgram(device->command_queue(), waiter_program, false);
        device->set_sub_device_stall_group({SubDeviceId{0}});
        // Test blocking on one sub-device
        EnqueueProgram(device->command_queue(), syncer_program, true);
        EnqueueProgram(device->command_queue(), incrementer_program, false);
        device->reset_sub_device_stall_group();
    }
    Synchronize(device);
    detail::DumpDeviceProfileResults(device);
}

// Ensure each core in the sub device aware of their own logical coordinate. Same binary used in multiple sub devices.
TEST_F(CommandQueueSingleCardProgramFixture, TensixTestSubDeviceMyLogicalCoordinates) {
    auto* device = devices_[0];
    uint32_t local_l1_size = 3200;
    // Make 2 sub devices.
    // origin means top left.
    // for sub_device_1 = 0,0. so relative coordinates are the same as logical.
    // for sub_device_2 = 3,3. so relative coordinates are offset by x=3,y=3 from logical.
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {6, 6})})});

    auto sub_device_manager = device->create_sub_device_manager({sub_device_1, sub_device_2}, k_local_l1_size);
    device->load_sub_device_manager(sub_device_manager);

    const auto sub_device_1_cores = device->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0});
    const auto sub_device_2_cores = device->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{1});

    uint32_t cb_addr = device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);
    std::vector<uint32_t> compile_args{cb_addr};

    // Start kernels on each sub device and verify their coordinates
    Program program_1 = tt::tt_metal::CreateProgram();
    tt::tt_metal::CreateKernel(
        program_1,
        k_coordinates_kernel_path,
        sub_device_1_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = compile_args});

    Program program_2 = tt::tt_metal::CreateProgram();
    tt::tt_metal::CreateKernel(
        program_2,
        k_coordinates_kernel_path,
        sub_device_2_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = compile_args});

    EnqueueProgram(device->command_queue(), program_1, false);
    EnqueueProgram(device->command_queue(), program_2, false);
    Finish(device->command_queue());
    device->reset_sub_device_stall_group();
    Synchronize(device);  // Ensure this CQ is cleared. Each CQ can only work on 1 sub device

    // Check coordinates
    tt::tt_metal::verify_kernel_coordinates(
        tt::BRISC, sub_device_1_cores, device, tt::tt_metal::SubDeviceId{0}, cb_addr);
    tt::tt_metal::verify_kernel_coordinates(
        tt::NCRISC, sub_device_2_cores, device, tt::tt_metal::SubDeviceId{1}, cb_addr);
}

TEST_F(CommandQueueSingleCardProgramFixture, TensixActiveEthTestSubDeviceMyLogicalCoordinates) {
    auto* device = devices_[0];
    CoreRangeSet sub_device_1_worker_cores{CoreRange({0, 0}, {2, 2})};
    SubDevice sub_device_1(std::array{sub_device_1_worker_cores});
    uint32_t num_iters = 5;
    if (!does_device_have_active_eth_cores(device)) {
        GTEST_SKIP() << "Skipping test because device " << device->id() << " does not have any active ethernet cores";
    }
    auto eth_core = *device->get_active_ethernet_cores(true).begin();
    CoreRangeSet sub_device_2_worker_cores{std::vector{CoreRange{{3, 3}, {3, 3}}, CoreRange{{4, 4}, {4, 4}}}};
    CoreRangeSet sub_device_2_eth_cores{CoreRange(eth_core, eth_core)};

    SubDevice sub_device_2(std::array{sub_device_2_worker_cores, sub_device_2_eth_cores});
    auto sub_device_manager = device->create_sub_device_manager({sub_device_1, sub_device_2}, k_local_l1_size);
    device->load_sub_device_manager(sub_device_manager);

    uint32_t cb_addr_eth = hal::get_erisc_l1_unreserved_base();
    uint32_t cb_addr_worker = device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);

    // Start kernels on each sub device and verify their coordinates
    Program program_1 = tt::tt_metal::CreateProgram();
    tt::tt_metal::CreateKernel(
        program_1,
        k_coordinates_kernel_path,
        sub_device_1_worker_cores,
        DataMovementConfig{
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                cb_addr_worker,
            }});

    Program program_2 = tt::tt_metal::CreateProgram();
    tt::tt_metal::CreateKernel(
        program_2,
        k_coordinates_kernel_path,
        sub_device_2_worker_cores,
        DataMovementConfig{
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                cb_addr_worker,
            }});

    tt::tt_metal::CreateKernel(
        program_2,
        k_coordinates_kernel_path,
        sub_device_2_eth_cores,
        EthernetConfig{
            .noc = NOC::RISCV_0_default,
            .processor = DataMovementProcessor::RISCV_0,
            .compile_args = {
                cb_addr_eth,
            }});

    EnqueueProgram(device->command_queue(), program_1, false);
    device->set_sub_device_stall_group({SubDeviceId{0}});
    EnqueueProgram(device->command_queue(), program_2, false);
    Finish(device->command_queue());
    device->reset_sub_device_stall_group();
    Synchronize(device);

    tt::tt_metal::verify_kernel_coordinates(
        tt::RISCV::BRISC, sub_device_1_worker_cores, device, tt::tt_metal::SubDeviceId{0}, cb_addr_worker);
    tt::tt_metal::verify_kernel_coordinates(
        tt::RISCV::NCRISC, sub_device_2_worker_cores, device, tt::tt_metal::SubDeviceId{1}, cb_addr_worker);
    tt::tt_metal::verify_kernel_coordinates(
        tt::RISCV::ERISC0, sub_device_2_eth_cores, device, tt::tt_metal::SubDeviceId{1}, cb_addr_eth);
}

// Ensure the relative coordinate for the worker is updated correctly when it is used for multiple sub device
// configurations
TEST_F(CommandQueueSingleCardProgramFixture, TensixTestSubDeviceMyLogicalCoordinatesSubDeviceSwitch) {
    auto* device = devices_[0];
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})})});
    SubDevice sub_device_3(std::array{CoreRangeSet(CoreRange({4, 4}, {6, 6}))});
    SubDevice sub_device_4(std::array{CoreRangeSet(std::vector{CoreRange({2, 2}, {2, 2}), CoreRange({3, 3}, {3, 3})})});
    std::vector<SubDeviceManagerId> sub_device_managers{
        device->create_sub_device_manager({sub_device_1, sub_device_2}, k_local_l1_size),
        device->create_sub_device_manager({sub_device_3, sub_device_4}, k_local_l1_size),
    };

    uint32_t cb_addr = device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);
    std::vector<uint32_t> compile_args{cb_addr};

    for (int i = 0; i < sub_device_managers.size(); ++i) {
        device->load_sub_device_manager(sub_device_managers[i]);
        const auto sub_device_cores = device->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{i});

        Program program = tt::tt_metal::CreateProgram();
        tt::tt_metal::CreateKernel(
            program,
            k_coordinates_kernel_path,
            sub_device_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = compile_args});

        EnqueueProgram(device->command_queue(), program, false);
        Finish(device->command_queue());
        device->reset_sub_device_stall_group();
        Synchronize(device);  // Ensure this CQ is cleared. Each CQ can only work on 1 sub device

        // Check coordinates
        tt::tt_metal::verify_kernel_coordinates(
            tt::BRISC, sub_device_cores, device, tt::tt_metal::SubDeviceId{i}, cb_addr);
    }
}

// Test that RTAs will be correctly updated when using the same program on multiple subdevice managers.
TEST_F(CommandQueueSingleCardFixture, TensixTestSubDeviceProgramReuseRtas) {
    constexpr uint32_t k_num_iters = 5;
    auto* device = devices_[0];
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(CoreRange({3, 3}, {3, 3}))});
    // Sub device IDs are swapped between the two sub device managers.
    auto sub_device_manager_1 = device->create_sub_device_manager({sub_device_1, sub_device_2}, k_local_l1_size);
    auto sub_device_manager_2 = device->create_sub_device_manager({sub_device_2, sub_device_1}, k_local_l1_size);

    uint32_t l1_unreserved_base = device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);

    tt_metal::Program program = tt_metal::CreateProgram();
    CoreCoord core = {3, 3};
    std::array<uint32_t, 1> unique_runtime_args = {101};
    std::array<uint32_t, 1> common_runtime_args = {201};

    tt_metal::KernelHandle add_two_ints_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/add_common_and_unique_rta.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = {l1_unreserved_base}});

    tt_metal::SetCommonRuntimeArgs(program, add_two_ints_kernel, common_runtime_args);

    for (size_t i = 0; i < k_num_iters; i++) {
        for (auto& sub_device_manager : {sub_device_manager_1, sub_device_manager_2}) {
            device->load_sub_device_manager(sub_device_manager);
            unique_runtime_args[0] += 1;
            common_runtime_args[0] += 2;
            tt_metal::SetRuntimeArgs(program, add_two_ints_kernel, core, unique_runtime_args);
            tt_metal::GetCommonRuntimeArgs(program, add_two_ints_kernel)[0] = common_runtime_args[0];

            // Enqueue twice to ensure waits are correct.
            EnqueueProgram(device->command_queue(), program, false);
            EnqueueProgram(device->command_queue(), program, false);
            Synchronize(device);
            std::vector<uint32_t> kernel_result;
            tt_metal::detail::ReadFromDeviceL1(device, core, l1_unreserved_base, sizeof(int), kernel_result);
            EXPECT_EQ(kernel_result[0], unique_runtime_args[0] + common_runtime_args[0]);
        }
    }
}

TEST_F(MultiCommandQueueSingleDeviceFixture, TensixTestSubDeviceCQOwnership) {
    constexpr uint32_t k_num_iters = 5;
    auto* device = device_;
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(CoreRange({3, 3}, {3, 3}))});
    // Sub device IDs are swapped between the two sub device managers.
    auto sub_device_manager = device->create_sub_device_manager({sub_device_1, sub_device_2}, k_local_l1_size);
    device->load_sub_device_manager(sub_device_manager);

    tt_metal::Program program_1 = tt_metal::CreateProgram();
    tt_metal::Program program_2 = tt_metal::CreateProgram();
    // On sub device 1.
    tt_metal::CreateKernel(
        program_1,
        "tt_metal/kernels/dataflow/blank.cpp",
        CoreRangeSet(CoreRange({0, 0}, {2, 2})),
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});
    // On sub device 2.
    tt_metal::CreateKernel(
        program_2,
        "tt_metal/kernels/dataflow/blank.cpp",
        CoreRangeSet(CoreRange({3, 3}, {3, 3})),
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});
    EnqueueProgram(device->command_queue(0), program_1, false);
    auto early_event = std::make_shared<Event>();
    EnqueueRecordEvent(device->command_queue(1), early_event);
    EnqueueProgram(device->command_queue(1), program_2, false);

    // CQ 0 owns sub device 1, CQ 1 owns sub device 2.
    EXPECT_THROW(EnqueueProgram(device->command_queue(1), program_1, false), std::exception);

    // Finish allows transfering ownership of sub device 1.
    Finish(device->command_queue(0));
    EnqueueProgram(device->command_queue(1), program_1, false);

    // CQ 1 owns sub devices 1 and 2.
    EXPECT_THROW(EnqueueProgram(device->command_queue(0), program_2, false), std::exception);

    // Waiting on an event before the last program was queued does not allow transferring ownership of sub device 2.
    EnqueueWaitForEvent(device->command_queue(0), early_event);
    EXPECT_THROW(EnqueueProgram(device->command_queue(0), program_2, false), std::exception);

    // Later event allows transferring ownership of sub device 2 to CQ 0
    auto event1 = std::make_shared<Event>();
    auto event2 = std::make_shared<Event>();
    EnqueueRecordEvent(device->command_queue(1), event1);
    EnqueueRecordEvent(device->command_queue(1), event2);
    EnqueueWaitForEvent(device->command_queue(0), event2);
    EnqueueProgram(device->command_queue(0), program_2, false);

    Synchronize(device);
    // Synchronize allows transferring ownership of either subdevice.
    EnqueueProgram(device->command_queue(0), program_1, false);
    EnqueueProgram(device->command_queue(1), program_2, false);
    Synchronize(device);
}

}  // namespace tt::tt_metal
