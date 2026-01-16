// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <umd/device/types/core_coordinates.hpp>
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
#include "impl/dispatch/command_queue_common.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "device_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_cluster.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/vector_aligned.hpp>
#include "math.hpp"
#include <impl/dispatch/dispatch_mem_map.hpp>
#include <distributed/mesh_device_impl.hpp>

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
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const size_t& byte_size,
    const size_t& l1_byte_address,
    const CoreCoord& grid_size) {
    auto* device = mesh_device->get_devices()[0];
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
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const size_t& byte_size,
    const size_t& dram_byte_address,
    const unsigned int& num_channels) {
    auto* device = mesh_device->get_devices()[0];
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

TEST_F(MeshDeviceFixture, PingAllLegalDramChannels) {
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
            size_t start_byte_address = devices_.at(id)->dram_size_per_channel() - (32 * 1024);
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
TEST_F(MeshDeviceFixture, PingIllegalDramChannels) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        auto num_channels = devices_.at(id)->num_dram_channels() + 1;
        size_t start_byte_address = devices_.at(id)->allocator()->get_base_allocator_addr(HalMemType::DRAM);
        ;
        ASSERT_ANY_THROW(unit_tests::basic::device::dram_ping(devices_.at(id), 4, start_byte_address, num_channels));
    }
}

TEST_F(MeshDeviceFixture, TensixPingAllLegalL1Cores) {
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
            size_t start_byte_address = devices_.at(id)->l1_size_per_core() - (32 * 1024);
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

TEST_F(MeshDeviceFixture, TensixPingIllegalL1Cores) {
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
TEST_F(MeshDeviceFixture, TensixValidateKernelDoesNotTargetHarvestedCores) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        auto mesh_device = this->devices_.at(id);
        auto* device = mesh_device->get_devices()[0];
        uint32_t num_l1_banks = mesh_device->allocator()->get_num_banks(BufferType::L1);
        std::vector<uint32_t> host_input(1);
        std::map<uint32_t, uint32_t> bank_id_to_value;
        uint32_t l1_address = mesh_device->l1_size_per_core() - 2048;
        for (uint32_t bank_id = 0; bank_id < num_l1_banks; bank_id++) {
            host_input[0] = bank_id + 1;
            bank_id_to_value[bank_id] = host_input.at(0);
            CoreCoord logical_core = mesh_device->allocator()->get_logical_core_from_bank_id(bank_id);
            uint32_t write_address = l1_address + mesh_device->allocator()->get_bank_offset(BufferType::L1, bank_id);
            tt_metal::detail::WriteToDeviceL1(device, logical_core, write_address, host_input);
        }

        auto& cq = mesh_device->mesh_command_queue();
        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        tt_metal::Program program = tt_metal::CreateProgram();

        std::string kernel_name = "tests/tt_metal/tt_metal/test_kernels/misc/ping_legal_l1s.cpp";
        CoreCoord logical_target_core(0, 0);
        uint32_t intermediate_l1_addr = devices_.at(id)->allocator()->get_base_allocator_addr(HalMemType::L1);
        uint32_t size_bytes = host_input.size() * sizeof(uint32_t);
        tt_metal::CreateKernel(
            program,
            kernel_name,
            logical_target_core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::NOC_0,
                .compile_args = {l1_address, intermediate_l1_addr, size_bytes}});

        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);

        std::vector<uint32_t> output;
        for (uint32_t bank_id = 0; bank_id < num_l1_banks; bank_id++) {
            CoreCoord logical_core = mesh_device->allocator()->get_logical_core_from_bank_id(bank_id);
            uint32_t read_address = l1_address + mesh_device->allocator()->get_bank_offset(BufferType::L1, bank_id);
            tt_metal::detail::ReadFromDeviceL1(device, logical_core, read_address, size_bytes, output);
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
TEST_F(MeshDeviceFixture, TestDeviceToHostMemChannelAssignment) {
    std::unordered_map<ChipId, std::set<ChipId>> mmio_device_to_device_group;
    for (unsigned int dev_id = 0; dev_id < num_devices_; dev_id++) {
        ChipId assoc_mmio_dev_id =
            tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(dev_id);
        std::set<ChipId>& device_ids = mmio_device_to_device_group[assoc_mmio_dev_id];
        device_ids.insert(dev_id);
    }

    for (const auto& [mmio_dev_id, device_group] : mmio_device_to_device_group) {
        EXPECT_EQ(
            tt::tt_metal::MetalContext::instance().get_cluster().get_num_host_channels(mmio_dev_id),
            device_group.size());
        std::unordered_set<uint16_t> channels;
        for (const ChipId& device_id : device_group) {
            channels.insert(
                tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(device_id));
        }
        EXPECT_EQ(channels.size(), device_group.size());
    }
}

// Test to ensure writing from 16B aligned L1 address to 16B aligned PCIe address works
TEST_F(MeshDeviceFixture, TensixTestL1ToPCIeAt16BAlignedAddress) {
    auto mesh_device = this->devices_.at(0);
    auto* device = mesh_device->get_devices()[0];
    auto& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    tt_metal::Program program = tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    EXPECT_TRUE(device->is_mmio_capable());
    CoreCoord logical_core(0, 0);

    uint32_t base_l1_src_address = mesh_device->allocator()->get_base_allocator_addr(HalMemType::L1) +
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

    CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/pcie_write_16b.cpp",
        logical_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_0_default,
            .compile_args = {base_l1_src_address, base_pcie_dst_address, num_16b_writes}});

    distributed::EnqueueMeshWorkload(cq, workload, false);

    std::vector<uint32_t> result(size_bytes / sizeof(uint32_t));
    ChipId mmio_device_id =
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
    const auto& mesh_device = devices_.at(0);
    auto* const device = mesh_device->get_devices()[0];

    uint32_t l1_unreserved_base = mesh_device->allocator()->get_base_allocator_addr(HalMemType::L1);
    std::vector<uint32_t> random_vec(1, 0xDEADBEEF);
    tt_metal::detail::WriteToDeviceL1(device, core, l1_unreserved_base, random_vec);

    uint32_t value_to_write = 39;
    bool invalidate_cache =
        true;  // To make sure this test passes on CI set this to true but can be modified for local debug
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    tt_metal::Program program = tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    uint32_t sem0_id = tt_metal::CreateSemaphore(program_, core, 0);

    tt_metal::KernelHandle kernel0 = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/poll_l1.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::NOC_0});

    tt_metal::SetRuntimeArgs(
        program_, kernel0, core, {l1_unreserved_base, value_to_write, sem0_id, (uint32_t)invalidate_cache});

    tt_metal::KernelHandle kernel1 = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/write_to_break_poll.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::NOC_1});

    tt_metal::SetRuntimeArgs(program_, kernel1, core, {l1_unreserved_base, value_to_write, sem0_id});

    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);

    tt_metal::detail::ReadFromDeviceL1(device, core, l1_unreserved_base, sizeof(uint32_t), random_vec);
    EXPECT_EQ(random_vec[0], value_to_write);
}

// Test to verify that logical to virtual coordinate mapping is correct.
// Reads mapping from L1 memory and compares it to the host mapping.
TEST_F(MeshDeviceFixture, VerifyLogicalToVirtualMap) {
    std::map<CoreCoord, CoreCoord> logical_to_virtual_map;

    auto mesh_device = this->devices_.at(0);
    auto* device = mesh_device->get_devices()[0];
    auto& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    tt_metal::Program program = tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    auto logical_grid_size = device->logical_grid_size();
    for (int r = 0; r < logical_grid_size.y; r++) {
        for (int c = 0; c < logical_grid_size.x; c++) {
            CoreCoord logical_coord(c, r);
            auto virtual_coord = device->virtual_core_from_logical_core(logical_coord, CoreType::WORKER);
            logical_to_virtual_map[logical_coord] = virtual_coord;
        }
    }

    CoreRange logical_core_range(CoreCoord(0, 0), CoreCoord(logical_grid_size.x - 1, logical_grid_size.y - 1));
    uint32_t l1_unreserved_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    std::array<std::vector<uint32_t>, 2> host_buffers;
    uint32_t read_size = sizeof(uint32_t) * (logical_grid_size.x + logical_grid_size.y);

    auto kernel0_l1_address = l1_unreserved_base;
    tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/read_logical_to_virtual_table.cpp",
        logical_core_range,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::NOC_0,
            .compile_args = {kernel0_l1_address, logical_grid_size.x, logical_grid_size.y}});

    auto kernel1_l1_address = l1_unreserved_base + (tt::round_up(logical_grid_size.x, 4) * sizeof(uint32_t)) +
                              (tt::round_up(logical_grid_size.y, 4) * sizeof(uint32_t));
    tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/read_logical_to_virtual_table.cpp",
        logical_core_range,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::NOC_1,
            .compile_args = {kernel1_l1_address, logical_grid_size.x, logical_grid_size.y}});

    distributed::EnqueueMeshWorkload(cq, workload, false);

    for (size_t x = 0; x < logical_grid_size.x; x++) {
        for (size_t y = 0; y < logical_grid_size.y; y++) {
            CoreCoord logical_core(x, y);
            tt_metal::detail::ReadFromDeviceL1(device, logical_core, kernel0_l1_address, read_size, host_buffers[0]);
            tt_metal::detail::ReadFromDeviceL1(device, logical_core, kernel1_l1_address, read_size, host_buffers[1]);

            for (size_t index = 0; index < host_buffers.size(); index++) {
                std::vector<uint32_t> logical_col_to_virtual_col(logical_grid_size.x);
                std::vector<uint32_t> logical_row_to_virtual_row(logical_grid_size.y);
                size_t host_buffer_offset = 0;

                for (size_t i = 0; i < logical_grid_size.x; i++) {
                    logical_col_to_virtual_col[i] = host_buffers[index][host_buffer_offset++];
                }
                for (size_t i = 0; i < logical_grid_size.y; i++) {
                    logical_row_to_virtual_row[i] = host_buffers[index][host_buffer_offset++];
                }

                for (auto test_coords : logical_to_virtual_map) {
                    auto virtual_coord = tt_xy_pair{
                        logical_col_to_virtual_col[test_coords.first.x],
                        logical_row_to_virtual_row[test_coords.first.y]};
                    EXPECT_EQ(test_coords.second, virtual_coord)
                        << "At logical core X,Y " << x << "," << y << " on RISCV_" << index
                        << " logical coord: " << test_coords.first.str()
                        << " does not match expected virtual coord: " << test_coords.second.str()
                        << " instead found virtual coord: " << virtual_coord.str();
                }
            }
        }
    }
}

// Test to ensure writing from 16B aligned L1 address to 16B aligned pinned memory works using MeshDevice
TEST_F(MeshDeviceFixture, MeshL1ToPinnedMemoryAt16BAlignedAddress) {
    using tt::tt_metal::distributed::EnqueueMeshWorkload;
    using tt::tt_metal::distributed::MeshCoordinate;
    using tt::tt_metal::distributed::MeshCoordinateRange;
    using tt::tt_metal::distributed::MeshCoordinateRangeSet;
    using tt::tt_metal::distributed::MeshWorkload;
    using tt::tt_metal::experimental::PinnedMemory;

    auto mesh_device = devices_.at(0);

    // Skip if mapping to NOC isn't supported on this system
    if (!experimental::GetMemoryPinningParameters(*mesh_device).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
    }

    // Use first device from the mesh for this test
    MeshCoordinate target_coord(0, 0);
    IDevice* device = mesh_device->impl().get_device(target_coord);
    EXPECT_TRUE(device->is_mmio_capable());

    CoreCoord logical_core(0, 0);

    uint32_t base_l1_src_address = device->allocator()->get_base_allocator_addr(HalMemType::L1) +
                                   MetalContext::instance().hal().get_alignment(HalMemType::L1);

    uint32_t size_bytes = 2048 * 128;
    std::vector<uint32_t> src =
        tt::test_utils::generate_uniform_random_vector<uint32_t>(0, UINT32_MAX, size_bytes / sizeof(uint32_t));
    EXPECT_EQ(MetalContext::instance().hal().get_alignment(HalMemType::L1), 16);
    uint32_t num_16b_writes = size_bytes / MetalContext::instance().hal().get_write_alignment(HalMemType::HOST);

    // Allocate and pin host memory
    auto aligned_buf = std::make_shared<tt::tt_metal::vector_aligned<uint32_t>>(size_bytes / sizeof(uint32_t), 0);
    tt::tt_metal::HostBuffer host_buffer_view(
        tt::stl::Span<uint32_t>(aligned_buf->data(), aligned_buf->size()), tt::tt_metal::MemoryPin(aligned_buf));
    auto coordinate_range_set = MeshCoordinateRangeSet(MeshCoordinateRange(target_coord, target_coord));
    auto pinned_memory = experimental::PinnedMemory::Create(
        *mesh_device,
        coordinate_range_set,
        host_buffer_view,
        true  // map_to_noc
    );
    ASSERT_EQ(
        reinterpret_cast<uintptr_t>(aligned_buf->data()) %
            MetalContext::instance().hal().get_write_alignment(HalMemType::HOST),
        0);

    // Write source data to L1
    tt_metal::detail::WriteToDeviceL1(device, logical_core, base_l1_src_address, src);

    // Create program and kernel for mesh workload
    tt_metal::Program program = tt_metal::CreateProgram();

    auto noc_addr = pinned_memory->get_noc_addr(device->id());
    ASSERT_TRUE(noc_addr.has_value());
    ASSERT_EQ(noc_addr.value().device_id, device->id());

    uint32_t dst_lo = static_cast<uint32_t>(noc_addr.value().addr & 0xFFFFFFFFull);
    uint32_t dst_hi = static_cast<uint32_t>(noc_addr.value().addr >> 32);
    uint32_t pcie_xy_enc = noc_addr.value().pcie_xy_enc;

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/pcie_write_16b_wwrite.cpp",
        logical_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_0_default,
            .compile_args = {base_l1_src_address, dst_lo, dst_hi, num_16b_writes, pcie_xy_enc}});

    // Create mesh workload and add program
    MeshWorkload mesh_workload;
    MeshCoordinateRange device_range(target_coord, target_coord);
    mesh_workload.add_program(device_range, std::move(program));

    // Launch workload using mesh command queue
    auto& mesh_cq = mesh_device->mesh_command_queue();
    EnqueueMeshWorkload(mesh_cq, mesh_workload, true);  // blocking = true

    // Verify the data was written correctly to pinned memory
    // Compare with a std::vector copy to avoid allocator type mismatch in EXPECT_EQ
    std::vector<uint32_t> aligned_copy(aligned_buf->begin(), aligned_buf->end());
    EXPECT_EQ(src, aligned_copy);
}

}  // namespace tt::tt_metal
