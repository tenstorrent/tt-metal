// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>
#include <umd/device/tt_core_coordinates.h>
#include <chrono>
#include <thread>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include "get_platform_architecture.hpp"
#include "hal.hpp"
#include "impl/dispatch/command_queue_common.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "device_fixture.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt.hpp"
#include "rtoptions.hpp"
#include "tt_cluster.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include <tt-metalium/utils.hpp>
#include <fmt/ranges.h>
#include <umd/device/types/arch.h>

namespace tt::tt_metal {

using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::basic::device {

/// @brief Ping number of bytes for specified grid_size
/// @param device
/// @param byte_size - size in bytes
/// @param l1_byte_address - l1 address to target for all cores
/// @param grid_size - grid size. will ping all cores from {0,0} to grid_size (non-inclusive)
/// @return
bool l1_ping(
    tt_metal::IDevice* device, const size_t& byte_size, const size_t& l1_byte_address, const CoreCoord& grid_size) {
    bool pass = true;
    auto inputs = generate_uniform_random_vector<uint32_t>(0, UINT32_MAX, byte_size / sizeof(uint32_t));
    for (int y = 0; y < grid_size.y; y++) {
        for (int x = 0; x < grid_size.x; x++) {
            CoreCoord dest_core(static_cast<size_t>(x), static_cast<size_t>(y));
            tt_metal::detail::WriteToDeviceL1(device, dest_core, l1_byte_address, inputs);
        }
    }

    for (int y = 0; y < grid_size.y; y++) {
        for (int x = 0; x < grid_size.x; x++) {
            CoreCoord dest_core(static_cast<size_t>(x), static_cast<size_t>(y));
            std::vector<uint32_t> dest_core_data;
            tt_metal::detail::ReadFromDeviceL1(device, dest_core, l1_byte_address, byte_size, dest_core_data);
            pass &= (dest_core_data == inputs);
            if (not pass) {
                log_error(tt::LogTest, "Mismatch at Core: ={}", dest_core.str());
            }
        }
    }
    return pass;
}

/// @brief Ping number of bytes for specified channels
/// @param device
/// @param byte_size - size in bytes
/// @param l1_byte_address - l1 address to target for all cores
/// @param num_channels - num_channels. will ping all channels from {0} to num_channels (non-inclusive)
/// @return
bool dram_ping(
    tt_metal::IDevice* device,
    const size_t& byte_size,
    const size_t& dram_byte_address,
    const unsigned int& num_channels) {
    bool pass = true;
    auto inputs = generate_uniform_random_vector<uint32_t>(0, UINT32_MAX, byte_size / sizeof(uint32_t));
    for (unsigned int channel = 0; channel < num_channels; channel++) {
        tt_metal::detail::WriteToDeviceDRAMChannel(device, channel, dram_byte_address, inputs);
    }

    for (unsigned int channel = 0; channel < num_channels; channel++) {
        std::vector<uint32_t> dest_channel_data;
        tt_metal::detail::ReadFromDeviceDRAMChannel(device, channel, dram_byte_address, byte_size, dest_channel_data);
        pass &= (dest_channel_data == inputs);
        if (not pass) {
            std::cout << "Mismatch at Channel: " << channel << std::endl;
        }
    }
    return pass;
}
}  // namespace unit_tests::basic::device

TEST_F(DeviceFixture, PingAllLegalDramChannels) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        {
            size_t start_byte_address = devices_.at(id)->allocator()->get_base_allocator_addr(HalMemType::DRAM);
            ASSERT_TRUE(unit_tests::basic::device::dram_ping(
                devices_.at(id), 4, start_byte_address, devices_.at(id)->num_dram_channels()));
            ASSERT_TRUE(unit_tests::basic::device::dram_ping(
                devices_.at(id), 12, start_byte_address, devices_.at(id)->num_dram_channels()));
            ASSERT_TRUE(unit_tests::basic::device::dram_ping(
                devices_.at(id), 16, start_byte_address, devices_.at(id)->num_dram_channels()));
            ASSERT_TRUE(unit_tests::basic::device::dram_ping(
                devices_.at(id), 1024, start_byte_address, devices_.at(id)->num_dram_channels()));
            ASSERT_TRUE(unit_tests::basic::device::dram_ping(
                devices_.at(id), 2 * 1024, start_byte_address, devices_.at(id)->num_dram_channels()));
            ASSERT_TRUE(unit_tests::basic::device::dram_ping(
                devices_.at(id), 32 * 1024, start_byte_address, devices_.at(id)->num_dram_channels()));
        }
        {
            size_t start_byte_address = devices_.at(id)->dram_size_per_channel() - 32 * 1024;
            ASSERT_TRUE(unit_tests::basic::device::dram_ping(
                devices_.at(id), 4, start_byte_address, devices_.at(id)->num_dram_channels()));
            ASSERT_TRUE(unit_tests::basic::device::dram_ping(
                devices_.at(id), 12, start_byte_address, devices_.at(id)->num_dram_channels()));
            ASSERT_TRUE(unit_tests::basic::device::dram_ping(
                devices_.at(id), 16, start_byte_address, devices_.at(id)->num_dram_channels()));
            ASSERT_TRUE(unit_tests::basic::device::dram_ping(
                devices_.at(id), 1024, start_byte_address, devices_.at(id)->num_dram_channels()));
            ASSERT_TRUE(unit_tests::basic::device::dram_ping(
                devices_.at(id), 2 * 1024, start_byte_address, devices_.at(id)->num_dram_channels()));
            ASSERT_TRUE(unit_tests::basic::device::dram_ping(
                devices_.at(id), 32 * 1024, start_byte_address, devices_.at(id)->num_dram_channels()));
        }
    }
}
TEST_F(DeviceFixture, PingIllegalDramChannels) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        auto num_channels = devices_.at(id)->num_dram_channels() + 1;
        size_t start_byte_address = devices_.at(id)->allocator()->get_base_allocator_addr(HalMemType::DRAM);
        ;
        ASSERT_ANY_THROW(unit_tests::basic::device::dram_ping(devices_.at(id), 4, start_byte_address, num_channels));
    }
}

TEST_F(DeviceFixture, TensixPingAllLegalL1Cores) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        {
            size_t start_byte_address = devices_.at(id)->allocator()->get_base_allocator_addr(HalMemType::L1);
            ASSERT_TRUE(unit_tests::basic::device::l1_ping(
                devices_.at(id), 4, start_byte_address, devices_.at(id)->logical_grid_size()));
            ASSERT_TRUE(unit_tests::basic::device::l1_ping(
                devices_.at(id), 12, start_byte_address, devices_.at(id)->logical_grid_size()));
            ASSERT_TRUE(unit_tests::basic::device::l1_ping(
                devices_.at(id), 16, start_byte_address, devices_.at(id)->logical_grid_size()));
            ASSERT_TRUE(unit_tests::basic::device::l1_ping(
                devices_.at(id), 1024, start_byte_address, devices_.at(id)->logical_grid_size()));
            ASSERT_TRUE(unit_tests::basic::device::l1_ping(
                devices_.at(id), 2 * 1024, start_byte_address, devices_.at(id)->logical_grid_size()));
            ASSERT_TRUE(unit_tests::basic::device::l1_ping(
                devices_.at(id), 32 * 1024, start_byte_address, devices_.at(id)->logical_grid_size()));
        }
        {
            size_t start_byte_address = devices_.at(id)->l1_size_per_core() - 32 * 1024;
            ASSERT_TRUE(unit_tests::basic::device::l1_ping(
                devices_.at(id), 4, start_byte_address, devices_.at(id)->logical_grid_size()));
            ASSERT_TRUE(unit_tests::basic::device::l1_ping(
                devices_.at(id), 12, start_byte_address, devices_.at(id)->logical_grid_size()));
            ASSERT_TRUE(unit_tests::basic::device::l1_ping(
                devices_.at(id), 16, start_byte_address, devices_.at(id)->logical_grid_size()));
            ASSERT_TRUE(unit_tests::basic::device::l1_ping(
                devices_.at(id), 1024, start_byte_address, devices_.at(id)->logical_grid_size()));
            ASSERT_TRUE(unit_tests::basic::device::l1_ping(
                devices_.at(id), 2 * 1024, start_byte_address, devices_.at(id)->logical_grid_size()));
            ASSERT_TRUE(unit_tests::basic::device::l1_ping(
                devices_.at(id), 32 * 1024, start_byte_address, devices_.at(id)->logical_grid_size()));
        }
    }
}

TEST_F(DeviceFixture, TensixPingIllegalL1Cores) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        auto grid_size = devices_.at(id)->logical_grid_size();
        grid_size.x++;
        grid_size.y++;
        size_t start_byte_address = devices_.at(id)->allocator()->get_base_allocator_addr(HalMemType::L1);
        ASSERT_ANY_THROW(unit_tests::basic::device::l1_ping(devices_.at(id), 4, start_byte_address, grid_size));
    }
}

// Harvesting tests

// Test methodology:
// 1. Host write single uint32_t value to each L1 bank
// 2. Launch a kernel to read and increment the value in each bank
// 3. Host validates that the value from step 1 has been incremented
// Purpose of this test is to ensure that L1 reader/writer APIs do not target harvested cores
TEST_F(DeviceFixture, TensixValidateKernelDoesNotTargetHarvestedCores) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        uint32_t num_l1_banks = devices_.at(id)->allocator()->get_num_banks(BufferType::L1);
        std::vector<uint32_t> host_input(1);
        std::map<uint32_t, uint32_t> bank_id_to_value;
        uint32_t l1_address = this->devices_.at(id)->l1_size_per_core() - 2048;
        for (uint32_t bank_id = 0; bank_id < num_l1_banks; bank_id++) {
            host_input[0] = bank_id + 1;
            bank_id_to_value[bank_id] = host_input.at(0);
            CoreCoord logical_core = this->devices_.at(id)->allocator()->get_logical_core_from_bank_id(bank_id);
            uint32_t write_address =
                l1_address + this->devices_.at(id)->allocator()->get_bank_offset(BufferType::L1, bank_id);
            tt_metal::detail::WriteToDeviceL1(this->devices_.at(id), logical_core, write_address, host_input);
        }

        tt_metal::Program program = tt_metal::CreateProgram();
        std::string kernel_name = "tests/tt_metal/tt_metal/test_kernels/misc/ping_legal_l1s.cpp";
        CoreCoord logical_target_core(0, 0);
        uint32_t intermediate_l1_addr = devices_.at(id)->allocator()->get_base_allocator_addr(HalMemType::L1);
        uint32_t size_bytes = host_input.size() * sizeof(uint32_t);
        tt_metal::KernelHandle kernel_id = tt_metal::CreateKernel(
            program,
            kernel_name,
            logical_target_core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::NOC_0,
                .compile_args = {l1_address, intermediate_l1_addr, size_bytes}});

        tt_metal::detail::LaunchProgram(this->devices_.at(id), program);

        std::vector<uint32_t> output;
        for (uint32_t bank_id = 0; bank_id < num_l1_banks; bank_id++) {
            CoreCoord logical_core = this->devices_.at(id)->allocator()->get_logical_core_from_bank_id(bank_id);
            uint32_t read_address =
                l1_address + this->devices_.at(id)->allocator()->get_bank_offset(BufferType::L1, bank_id);
            tt_metal::detail::ReadFromDeviceL1(this->devices_.at(id), logical_core, read_address, size_bytes, output);
            ASSERT_EQ(output.size(), host_input.size());
            uint32_t expected_value =
                bank_id_to_value.at(bank_id) + 1;  // ping_legal_l1s kernel increments each value it reads
            ASSERT_TRUE(output.at(0) == expected_value) << "Logical core " + logical_core.str() + " should have " +
                                                               std::to_string(expected_value) + " but got " +
                                                               std::to_string(output.at(0));
        }
    }
}

// For a given collection of MMIO device and remote devices, ensure that channels are unique
TEST_F(DeviceFixture, TestDeviceToHostMemChannelAssignment) {
    std::unordered_map<chip_id_t, std::set<chip_id_t>> mmio_device_to_device_group;
    for (unsigned int dev_id = 0; dev_id < num_devices_; dev_id++) {
        chip_id_t assoc_mmio_dev_id =
            tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(dev_id);
        std::set<chip_id_t>& device_ids = mmio_device_to_device_group[assoc_mmio_dev_id];
        device_ids.insert(dev_id);
    }

    for (const auto& [mmio_dev_id, device_group] : mmio_device_to_device_group) {
        EXPECT_EQ(
            tt::tt_metal::MetalContext::instance().get_cluster().get_num_host_channels(mmio_dev_id),
            device_group.size());
        std::unordered_set<uint16_t> channels;
        for (const chip_id_t& device_id : device_group) {
            channels.insert(
                tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(device_id));
        }
        EXPECT_EQ(channels.size(), device_group.size());
    }
}

// Test to ensure writing from 16B aligned L1 address to 16B aligned PCIe address works
TEST_F(DeviceFixture, TensixTestL1ToPCIeAt16BAlignedAddress) {
    tt_metal::Program program = tt_metal::CreateProgram();
    IDevice* device = this->devices_.at(0);
    EXPECT_TRUE(device->is_mmio_capable());
    CoreCoord logical_core(0, 0);

    uint32_t base_l1_src_address = device->allocator()->get_base_allocator_addr(HalMemType::L1) +
                                   MetalContext::instance().hal().get_alignment(HalMemType::L1);
    // This is a slow dispatch test dispatch core type is needed to query DispatchMemMap
    uint32_t base_pcie_dst_address =
        MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED) +
        MetalContext::instance().hal().get_alignment(HalMemType::L1);

    uint32_t size_bytes = 2048 * 128;
    std::vector<uint32_t> src = generate_uniform_random_vector<uint32_t>(0, UINT32_MAX, size_bytes / sizeof(uint32_t));
    EXPECT_EQ(MetalContext::instance().hal().get_alignment(HalMemType::L1), 16);
    uint32_t num_16b_writes = size_bytes / MetalContext::instance().hal().get_alignment(HalMemType::L1);

    tt_metal::detail::WriteToDeviceL1(device, logical_core, base_l1_src_address, src);

    auto pcie_writer = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/pcie_write_16b.cpp",
        logical_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_0_default,
            .compile_args = {base_l1_src_address, base_pcie_dst_address, num_16b_writes}});

    tt_metal::detail::LaunchProgram(device, program);

    std::vector<uint32_t> result(size_bytes / sizeof(uint32_t));
    chip_id_t mmio_device_id =
        tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device->id());
    uint16_t channel =
        tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(device->id());
    tt::tt_metal::MetalContext::instance().get_cluster().read_sysmem(
        result.data(), size_bytes, base_pcie_dst_address, mmio_device_id, channel);

    EXPECT_EQ(src, result);
}

// One risc caches L1 address then polls for expected value that other risc on same core writes after some delay
// Expected test scenarios:
// 1. `invalidate_cache` is true: pass
// 2. `invalidate_cache` is false: hang because periodic HW cache flush is default disabled
// 3. `invalidate_cache` is false and env var `TT_METAL_ENABLE_HW_CACHE_INVALIDATION` is set: pass
TEST_F(BlackholeSingleCardFixture, TensixL1DataCache) {
    CoreCoord core{0, 0};

    uint32_t l1_unreserved_base = device_->allocator()->get_base_allocator_addr(HalMemType::L1);
    std::vector<uint32_t> random_vec(1, 0xDEADBEEF);
    tt_metal::detail::WriteToDeviceL1(device_, core, l1_unreserved_base, random_vec);

    uint32_t value_to_write = 39;
    bool invalidate_cache =
        true;  // To make sure this test passes on CI set this to true but can be modified for local debug
    tt_metal::Program program = tt_metal::CreateProgram();

    uint32_t sem0_id = tt_metal::CreateSemaphore(program, core, 0);

    tt_metal::KernelHandle kernel0 = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/poll_l1.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::NOC_0});

    tt_metal::SetRuntimeArgs(
        program, kernel0, core, {l1_unreserved_base, value_to_write, sem0_id, (uint32_t)invalidate_cache});

    tt_metal::KernelHandle kernel1 = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/write_to_break_poll.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::NOC_1});

    tt_metal::SetRuntimeArgs(program, kernel1, core, {l1_unreserved_base, value_to_write, sem0_id});

    tt_metal::detail::LaunchProgram(device_, program);

    tt_metal::detail::ReadFromDeviceL1(device_, core, l1_unreserved_base, sizeof(uint32_t), random_vec);
    EXPECT_EQ(random_vec[0], value_to_write);
}

// TEST(Debugging, OpenAndClose) {
//     for (int i = 0; i < 100000; ++i) {
//         [[maybe_unused]] auto device = CreateDevice(0);
//         device->close();
//     }
// }

void wait_for_heartbeat_custom(chip_id_t device_id, const CoreCoord& virtual_core, uint32_t heartbeat_addr) {
    uint32_t heartbeat_val = llrt::read_hex_vec_from_core(device_id, virtual_core, heartbeat_addr, sizeof(uint32_t))[0];
    uint32_t previous_heartbeat_val = heartbeat_val;
    const auto start = std::chrono::high_resolution_clock::now();
    constexpr auto k_sleep_time = std::chrono::nanoseconds{50};

    while (heartbeat_val == previous_heartbeat_val) {
        std::this_thread::sleep_for(k_sleep_time);
        tt_driver_atomics::lfence();
        previous_heartbeat_val = heartbeat_val;
        heartbeat_val = llrt::read_hex_vec_from_core(device_id, virtual_core, heartbeat_addr, sizeof(uint32_t))[0];
        const auto now = std::chrono::high_resolution_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        if (elapsed > 10000) {
            std::cout << "timed out waiting" << std::endl;
            std::abort();
        }
    }

    std::cout << "wait_for_heartbeat done" << std::endl;
}

void output_debug_info(
    tt::tt_metal::IDevice* device,
    uint32_t debug_dump_addr,
    uint32_t buffer_base_addr,
    uint32_t failure_index,
    const CoreCoord& virtual_core,
    const CoreCoord& other_debug_core) {
    // print buffer data N-16 from failure index
    failure_index = std::max(failure_index, static_cast<uint32_t>(16));
    uint32_t buffer_start = buffer_base_addr + ((failure_index - 16) * sizeof(uint32_t));
    const auto buffer_n16 =
        llrt::read_hex_vec_from_core(device->id(), virtual_core, buffer_start, sizeof(uint32_t) * 32);
    const auto debug_contents =
        llrt::read_hex_vec_from_core(device->id(), virtual_core, debug_dump_addr, sizeof(uint32_t) * 32);
    std::cout << fmt::format(
                     "Debug Info\nBuffer[{} : {}] = {:#010x}\nDebug Info: {:#010x}",
                     failure_index,
                     failure_index + 32,
                     fmt::join(buffer_n16, ", "),
                     fmt::join(debug_contents, ", "))
              << "\n";
    // the core we copied to logical tensix 0,0. can we trust host reads from eth 0,11?
    const auto other_core_l1 =
        llrt::read_hex_vec_from_core(device->id(), other_debug_core, 0x20000, sizeof(uint32_t) * 8);
    std::cout << fmt::format("Worker 0,0 (virt 1,2) Debug Info: {:#010x}", fmt::join(other_core_l1, ", ")) << "\n";
}

void do_debug_test_2(
    tt::tt_metal::IDevice* device,
    uint32_t debug_dump_addr,
    uint32_t buffer_base,
    uint32_t arg_base,
    uint32_t num_writes,
    const CoreCoord& virtual_core,
    const CoreCoord& other_debug_core_virt,
    bool is_eth,
    uint32_t tensix_heartbeat_addr = 0) {
    // Copy debug to other core
    // Write dummy messages to process

    uint32_t response_buffer_addr = buffer_base + 2;
    uint32_t stop_buffer_addr = buffer_base + 4;

    llrt::write_hex_vec_to_core(
            device->id(), virtual_core, std::vector<uint32_t>{0, 0}, buffer_base);

    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(device->id());
    
    uint32_t i = 0;
        
    while(true) {
        // uint32_t msg_i = i & 0xffff;

        uint16_t msg = (((uint8_t)i) << 8) | (uint8_t)i;
        llrt::write_hex_vec_to_core(
            device->id(), virtual_core, std::vector<uint16_t>{msg}, buffer_base);

        // tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(device->id());

        std::cout << "Sent message to kernel: " << std::hex << (uint32_t)msg << std::dec << std::endl;

        // Wait for kernel to process message
        uint8_t stop_signal =
            llrt::read_hex_vec_from_core(device->id(), virtual_core, stop_buffer_addr, sizeof(uint32_t))[0] & 0xff;

        std::cout << "stop signal " << std::hex << (uint32_t)stop_signal << std::dec << std::endl;

        if (stop_signal == 1) {
            // uint16_t last_val_kernel =
            // llrt::read_hex_vec_from_core(device->id(), virtual_core, response_buffer_addr, sizeof(uint16_t))[0];

            std::cout << "Host received stop signal from kernel" << std::endl;
            while(true) {}
            // std::cout << "Last value written by kernel: 0x" << std::hex << (uint32_t)last_val_kernel << std::dec << std::endl;
        }

        i++;
    }
}

void do_debug_test(
    tt::tt_metal::IDevice* device,
    uint32_t debug_dump_addr,
    uint32_t buffer_base,
    uint32_t arg_base,
    uint32_t num_writes,
    const CoreCoord& virtual_core,
    const CoreCoord& other_debug_core_virt,
    bool is_eth,
    uint32_t tensix_heartbeat_addr = 0) {
    // Copy debug to other core
    // Write dummy messages to process
    uint16_t i = 1;
    while(true) {

        if (i == 0) {
            i++;
            continue;
        }

        std::cout << "waiting for heartbeat" << std::endl;
        if (is_eth) {
            llrt::internal_::wait_for_heartbeat(device->id(), virtual_core);
        } else {
            wait_for_heartbeat_custom(device->id(), virtual_core, tensix_heartbeat_addr);
        }
        std::cout << "done heartbeat" << std::endl;
        uint32_t msg_i = i << 16 | i;
        // llrt::write_hex_vec_to_core(device->id(), virtual_core, std::vector<uint32_t>{msg_i}, arg_base);
        // std::this_thread::sleep_for(std::chrono::nanoseconds(50));
        // tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(device->id());
        uint32_t dest_buffer_addr = buffer_base;
        llrt::write_hex_vec_to_core(
            device->id(), virtual_core, std::vector<uint32_t>{msg_i}, dest_buffer_addr);
        tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(device->id());

        std::cout << "message " << std::dec << i << " sent" << " " << std::hex << (msg_i) << " to "
                  << dest_buffer_addr << std::endl;

        uint32_t mailbox_val =
            llrt::read_hex_vec_from_core(device->id(), virtual_core, dest_buffer_addr, sizeof(uint32_t))[0];

        std::cout << "mailbox_val 0x" << std::hex << mailbox_val << std::dec << std::endl;

        uint32_t stop_signal =
            llrt::read_hex_vec_from_core(device->id(), virtual_core, buffer_base + 8, sizeof(uint32_t))[0] & 0xff;

        std::cout << "stop signal 0x" << std::hex << stop_signal << std::dec << std::endl;

        if (stop_signal == 1) {
            std::cout << "Host received stop signal from kernel" << std::endl;
            uint32_t last_val =
                llrt::read_hex_vec_from_core(device->id(), virtual_core, buffer_base + 4, sizeof(uint32_t))[0];
            std::cout << "Last value written by kernel: 0x" << std::hex << last_val << std::dec << std::endl;
        }

        if (stop_signal == 2) {
            std::cout << "Host received DOUBLE WRITE signal" << std::endl;
            uint32_t last_val =
                llrt::read_hex_vec_from_core(device->id(), virtual_core, buffer_base + 4, sizeof(uint32_t))[0];
            std::cout << "Last value written by kernel: 0x" << std::hex << last_val << std::dec << std::endl;
        }

        // Wait for kernel to process message
        // uint32_t mailbox_val =
        //     llrt::read_hex_vec_from_core(device->id(), virtual_core, dest_buffer_addr, sizeof(uint32_t))[0];
        // uint32_t msg_status = mailbox_val & 0xffff0000;
        // {
        //     const auto start = std::chrono::high_resolution_clock::now();
        //     constexpr auto k_sleep_time = std::chrono::nanoseconds{10};
        //     while (msg_status != 0xd0e50000) {
        //         std::this_thread::sleep_for(std::chrono::nanoseconds(k_sleep_time));
        //         mailbox_val =
        //             llrt::read_hex_vec_from_core(device->id(), virtual_core, dest_buffer_addr, sizeof(uint32_t))[0];
        //         msg_status = mailbox_val & 0xffff0000;

        //         const auto now = std::chrono::high_resolution_clock::now();
        //         const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        //         if (elapsed > 10000) {
        //             std::cout << "timed out waiting" << std::endl;
        //             output_debug_info(device, debug_dump_addr, buffer_base, i, virtual_core, other_debug_core_virt);
        //             std::abort();
        //         }
        //     }
        // }

        // We didnt write what we wanted?
        // if ((mailbox_val & 0xffff) != msg_i) {
        //     TT_THROW("Read msg i {} but expected {}", mailbox_val, msg_i);
        // }

        i++;
    }

    std::cout << "Check final value\n";
    // Output the buffer values to see if we receive the final value on the core
    auto buffer_val = llrt::read_hex_vec_from_core(
        device->id(), virtual_core, buffer_base + ((num_writes - 1) * sizeof(uint32_t)), sizeof(uint32_t))[0];
    EXPECT_EQ(buffer_val, 0xd0e50000 | ((num_writes - 1) & 0xffff));

    std::cout << "Kernel done" << std::endl;
    llrt::write_hex_vec_to_core(device->id(), virtual_core, std::vector<uint32_t>{0xdead0000}, buffer_base);
}

TEST(Debugging, Test_Eth) {
    if (!std::getenv("TT_METAL_SLOW_DISPATCH_MODE")) {
        GTEST_SKIP();
    }
    auto device = CreateDevice(0);

    for (int i = 0; i < 1; ++i) {
        constexpr uint32_t num_writes = 60000;
        // NOTE: kernel ring buffer size = 16,384
        constexpr uint32_t total_size = num_writes * sizeof(uint32_t);
        uint32_t l1_base =
            MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
        uint32_t arg_base = l1_base;
        uint32_t buffer_base = l1_base + 16;

        std::cout << "l1_base " << std::hex << l1_base << std::endl;
        std::cout << "buffer_base " << std::hex << buffer_base << std::endl;
        const auto& other_debug_core_virt = device->virtual_core_from_logical_core(CoreCoord{1, 1}, CoreType::WORKER);

        std::vector<uint32_t> ct_args = {
            num_writes, buffer_base, arg_base, 0x7CC70, 0x36b0, other_debug_core_virt.x, other_debug_core_virt.y};
        std::vector<uint32_t> zero_buffer(num_writes, 0xdeadbeef);

        const auto& core = CoreCoord{0, 0};
        // Zero buffer
        const auto& virtual_core = device->virtual_core_from_logical_core(core, CoreType::ETH);
        llrt::write_hex_vec_to_core(device->id(), virtual_core, zero_buffer, buffer_base);
        llrt::write_hex_vec_to_core(device->id(), virtual_core, std::vector<uint32_t>{0}, arg_base);
        tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(device->id());
        std::cout << "virtual_core " << virtual_core.str() << std::endl;

        tt_metal::Program program = tt_metal::CreateProgram();
        tt_metal::KernelHandle kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/device/test_kernel_2.cpp",
            core,
            tt_metal::EthernetConfig{.eth_mode = IDLE, .processor = tt_metal::DataMovementProcessor::RISCV_0, .compile_args = ct_args});

        // tt_metal::KernelHandle reader_kernel = tt_metal::CreateKernel(
        //     program,
        //     "tests/tt_metal/tt_metal/device/reader_kernel.cpp",
        //     core,
        //     tt_metal::EthernetConfig{.eth_mode = IDLE, .processor = tt_metal::DataMovementProcessor::RISCV_1, .compile_args = ct_args});

        tt_metal::detail::LaunchProgram(device, program, false);

        do_debug_test(device, 0x36b0, buffer_base, arg_base, num_writes, virtual_core, other_debug_core_virt, true);

        detail::WaitProgramDone(device, program, false);
    }
}

TEST(Debugging, Test_Tensix) {
    if (!std::getenv("TT_METAL_SLOW_DISPATCH_MODE")) {
        GTEST_SKIP();
    }
    auto device = CreateDevice(0);
    for (int i = 0; i < 1; ++i) {
        constexpr uint32_t num_writes = 60000;
        constexpr uint32_t total_size = num_writes * sizeof(uint32_t);
        uint32_t l1_base = MetalContext::instance().hal().get_dev_addr(
            HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
        uint32_t heartbeat_base = l1_base;
        uint32_t debug_dump_base = heartbeat_base + 16;
        uint32_t arg_base = debug_dump_base + 64;
        uint32_t buffer_base = arg_base + 16;

        std::cout << "l1_base " << std::hex << l1_base << std::endl;
        std::cout << "buffer_base " << std::hex << buffer_base << std::endl;
        const auto& other_debug_core_virt = device->virtual_core_from_logical_core(CoreCoord{1, 1}, CoreType::WORKER);

        std::vector<uint32_t> ct_args = {
            num_writes,
            buffer_base,
            arg_base,
            heartbeat_base,
            debug_dump_base,
            other_debug_core_virt.x,
            other_debug_core_virt.y};
        std::vector<uint32_t> zero_buffer(num_writes, 0);

        const auto& core = CoreCoord{0, 0};
        // Zero buffer
        const auto& virtual_core = device->virtual_core_from_logical_core(core, CoreType::WORKER);
        llrt::write_hex_vec_to_core(device->id(), virtual_core, zero_buffer, buffer_base);
        llrt::write_hex_vec_to_core(device->id(), virtual_core, std::vector<uint32_t>{0}, arg_base);
        // tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(device->id());
        std::this_thread::sleep_for(std::chrono::seconds(2));

        tt_metal::Program program = tt_metal::CreateProgram();
        tt_metal::KernelHandle kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/device/test_kernel_2.cpp",
            core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .compile_args = ct_args});
        tt_metal::detail::LaunchProgram(device, program, false);

        do_debug_test(
            device,
            debug_dump_base,
            buffer_base,
            arg_base,
            num_writes,
            virtual_core,
            other_debug_core_virt,
            false,
            heartbeat_base);

        detail::WaitProgramDone(device, program, false);
    }
}

TEST(Debugging, TestBasicWritesEth) {
    auto rt_options = llrt::RunTimeOptions();
    auto hal = Hal(tt::ARCH::BLACKHOLE, false);
    auto cluster = std::make_unique<tt::Cluster>(rt_options, hal);

    CoreCoord virtual_eth_core = {20, 25};
    std::vector<uint32_t> data = {0xabcd1234};
    std::vector<uint32_t> read;
    for (int i = 0; i < 800000; ++i) {
        cluster->write_core(data.data(), data.size() * sizeof(uint32_t), tt_cxy_pair{0, virtual_eth_core.x, virtual_eth_core.y}, 0x20000);
        for (int j = 0; j < 25; ++j) {
            cluster->read_core(read, data.size() * sizeof(uint32_t), tt_cxy_pair{0, virtual_eth_core.x, virtual_eth_core.y}, 0x20000);
        }
    }

    std::cout << "done" << std::endl;
}

TEST(Debugging, TestBasicWritesTensix) {
    auto rt_options = llrt::RunTimeOptions();
    auto hal = Hal(tt::ARCH::BLACKHOLE, false);
    auto cluster = std::make_unique<tt::Cluster>(rt_options, hal);

    CoreCoord virtual_eth_core = {1, 2};
    std::vector<uint32_t> data = {0xabcd1234};
    std::vector<uint32_t> read;
    for (int i = 0; i < 800000; ++i) {
        cluster->write_core(
            data.data(),
            data.size() * sizeof(uint32_t),
            tt_cxy_pair{0, virtual_eth_core.x, virtual_eth_core.y},
            0x20000);
        for (int j = 0; j < 25; ++j) {
            cluster->read_core(
                read, data.size() * sizeof(uint32_t), tt_cxy_pair{0, virtual_eth_core.x, virtual_eth_core.y}, 0x20000);
        }
    }

    std::cout << "done" << std::endl;
}

}  // namespace tt::tt_metal
