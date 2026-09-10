// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Regression test for NOC_TARG_ADDR_MID / NOC_RET_ADDR_MID cleanup after a PCIe-routed transaction.
// noc_async_read_pcie()/noc_async_write_pcie() set that register to route through the PCIe core, then
// clear it back to 0 afterward, since plain noc_async_read()/noc_async_write() no longer touch it and
// assume it's already 0. This test checks both that the register is actually cleared, and that the next
// ordinary on-chip transaction on the same command buffer lands at its intended address rather than being
// misrouted by a leftover value. Blackhole-specific; skips elsewhere.

#include "multi_device_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "dm_common.hpp"
#include <distributed/mesh_device_impl.hpp>

namespace tt::tt_metal {

using namespace std;

namespace unit_tests::dm::pcie_mid_cleanup {

void fill_pattern(std::vector<uint32_t>& data, uint32_t tag) {
    for (uint32_t i = 0; i < data.size(); ++i) {
        data[i] = (tag << 16) | (i & 0xFFFFu);
    }
}

}  // namespace unit_tests::dm::pcie_mid_cleanup

TEST_F(GenericMeshDeviceFixture, PCIeMidCleanup) {
    auto mesh_device = get_mesh_device();
    IDevice* device = mesh_device->impl().get_device(0);

    if (device->arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "NOC_TARG_ADDR_MID/NOC_RET_ADDR_MID PCIe routing is Blackhole-specific";
    }

    const CoreCoord logical_core = {0, 0};
    constexpr uint32_t transfer_size = 64;
    constexpr uint32_t num_words = transfer_size / sizeof(uint32_t);

    auto l1_info = unit_tests::dm::get_l1_address_and_size(mesh_device, logical_core);
    ASSERT_GE(l1_info.size, 7 * transfer_size) << "Insufficient L1 for test buffers";
    uint32_t l1_base = static_cast<uint32_t>(l1_info.base_address);

    const uint32_t l1_pcie_read_scratch_addr = l1_base + 0 * transfer_size;
    const uint32_t l1_onchip_read_src_addr = l1_base + 1 * transfer_size;
    const uint32_t l1_onchip_read_dst_addr = l1_base + 2 * transfer_size;
    const uint32_t l1_write_src_addr = l1_base + 3 * transfer_size;
    const uint32_t l1_onchip_write_src_addr = l1_base + 4 * transfer_size;
    const uint32_t l1_onchip_write_dst_addr = l1_base + 5 * transfer_size;
    const uint32_t l1_mid_result_addr = l1_base + 6 * transfer_size;

    // PCIe source/destination offsets, well clear of dispatch's own PCIe usage window.
    uint64_t dev_pcie_base = MetalContext::instance().get_cluster().get_pcie_base_addr_from_device(device->id());
    const uint32_t pcie_read_offset = static_cast<uint32_t>(dev_pcie_base + 1024 * 1024 * 50);
    const uint32_t pcie_write_offset = static_cast<uint32_t>(dev_pcie_base + 1024 * 1024 * 51);

    // Seed on-chip source patterns and poison the on-chip destinations so we can tell whether the
    // kernel's post-PCIe on-chip transaction actually landed correctly, or was misrouted.
    std::vector<uint32_t> onchip_read_pattern(num_words);
    unit_tests::dm::pcie_mid_cleanup::fill_pattern(onchip_read_pattern, 0xAAAA);
    std::vector<uint32_t> write_src_pattern(num_words);
    unit_tests::dm::pcie_mid_cleanup::fill_pattern(write_src_pattern, 0xBBBB);
    std::vector<uint32_t> onchip_write_pattern(num_words);
    unit_tests::dm::pcie_mid_cleanup::fill_pattern(onchip_write_pattern, 0xCCCC);
    std::vector<uint32_t> poison(num_words, 0xDEADBEEFu);

    ASSERT_TRUE(detail::WriteToDeviceL1(device, logical_core, l1_onchip_read_src_addr, onchip_read_pattern));
    ASSERT_TRUE(detail::WriteToDeviceL1(device, logical_core, l1_onchip_read_dst_addr, poison));
    ASSERT_TRUE(detail::WriteToDeviceL1(device, logical_core, l1_write_src_addr, write_src_pattern));
    ASSERT_TRUE(detail::WriteToDeviceL1(device, logical_core, l1_onchip_write_src_addr, onchip_write_pattern));
    ASSERT_TRUE(detail::WriteToDeviceL1(device, logical_core, l1_onchip_write_dst_addr, poison));

    CoreCoord self_phys = mesh_device->worker_core_from_logical_core(logical_core);
    uint32_t packed_self_coords = (self_phys.x << 16) | (self_phys.y & 0xFFFF);

    Program program = CreateProgram();
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/pcie_mid_cleanup/kernels/pcie_mid_cleanup.cpp",
        logical_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                pcie_read_offset,
                pcie_write_offset,
                l1_pcie_read_scratch_addr,
                l1_onchip_read_src_addr,
                l1_onchip_read_dst_addr,
                l1_write_src_addr,
                l1_onchip_write_src_addr,
                l1_onchip_write_dst_addr,
                l1_mid_result_addr,
                transfer_size,
                packed_self_coords}});

    program.set_runtime_id(unit_tests::dm::runtime_host_id++);
    auto mesh_workload = distributed::MeshWorkload();
    auto target_devices = distributed::MeshCoordinateRange(distributed::MeshCoordinate({0, 0}));
    mesh_workload.add_program(target_devices, std::move(program));

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    distributed::Finish(cq);

    // The core invariant: MID must be exactly 0 after each PCIe call clears it.
    std::vector<uint32_t> mid_result(2);
    ASSERT_TRUE(detail::ReadFromDeviceL1(
        device,
        logical_core,
        l1_mid_result_addr,
        std::span<uint8_t>(reinterpret_cast<uint8_t*>(mid_result.data()), mid_result.size() * sizeof(uint32_t))));
    EXPECT_EQ(mid_result[0], 0u) << "NOC_TARG_ADDR_MID not cleared after noc_async_read_pcie";
    EXPECT_EQ(mid_result[1], 0u) << "NOC_RET_ADDR_MID not cleared after noc_async_write_pcie";

    // The behavioral consequence: the on-chip read/write issued right after each PCIe call, on the same
    // command buffer, must have landed correctly -- not been misrouted by a leftover MID value.
    std::vector<uint32_t> onchip_read_result(num_words);
    ASSERT_TRUE(detail::ReadFromDeviceL1(
        device,
        logical_core,
        l1_onchip_read_dst_addr,
        std::span<uint8_t>(
            reinterpret_cast<uint8_t*>(onchip_read_result.data()), onchip_read_result.size() * sizeof(uint32_t))));
    EXPECT_EQ(onchip_read_result, onchip_read_pattern)
        << "On-chip read after noc_async_read_pcie landed incorrectly (read_cmd_buf MID not cleared in time)";

    std::vector<uint32_t> onchip_write_result(num_words);
    ASSERT_TRUE(detail::ReadFromDeviceL1(
        device,
        logical_core,
        l1_onchip_write_dst_addr,
        std::span<uint8_t>(
            reinterpret_cast<uint8_t*>(onchip_write_result.data()), onchip_write_result.size() * sizeof(uint32_t))));
    EXPECT_EQ(onchip_write_result, onchip_write_pattern)
        << "On-chip write after noc_async_write_pcie landed incorrectly (write_cmd_buf MID not cleared in time)";
}

}  // namespace tt::tt_metal
