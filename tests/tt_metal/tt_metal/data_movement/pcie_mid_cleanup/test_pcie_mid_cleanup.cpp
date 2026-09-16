// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Regression test for NOC_TARG_ADDR_MID / NOC_RET_ADDR_MID cleanup after a PCIe-routed transaction. The
// register is sticky per command buffer and the plain noc_async_read/write no longer clear it, so a
// leftover value silently misroutes the next on-chip transaction to host memory. The test also confirms
// the PCIe transfers themselves reached host memory, so a routing bit that is never set cannot pass.
// Blackhole only.

#include <cstring>

#include "device_fixture.hpp"
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

// Offsets within the channel-0 host hugepage. Kept clear of the dispatch CQ headroom.
constexpr uint32_t kHostReadOffset = 1024 * 1024 * 50;
constexpr uint32_t kHostWriteOffset = 1024 * 1024 * 51;
constexpr uint32_t kHostBatchWriteOffset = 1024 * 1024 * 52;

void fill_pattern(std::vector<uint32_t>& data, uint32_t tag) {
    for (uint32_t i = 0; i < data.size(); ++i) {
        data[i] = (tag << 16) | (i & 0xFFFFu);
    }
}

}  // namespace unit_tests::dm::pcie_mid_cleanup

TEST_F(UnitMeshFastDispatchFixture, PCIeMidCleanup) {
    namespace test_consts = unit_tests::dm::pcie_mid_cleanup;

    auto mesh_device = get_mesh_device();
    IDevice* device = mesh_device->impl().get_device(0);

    if (device->arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "NOC_TARG_ADDR_MID/NOC_RET_ADDR_MID PCIe routing is Blackhole-specific";
    }

    const CoreCoord logical_core = {0, 0};
    constexpr uint32_t transfer_size = 64;
    constexpr uint32_t num_words = transfer_size / sizeof(uint32_t);

    auto l1_info = unit_tests::dm::get_l1_address_and_size(mesh_device, logical_core);
    ASSERT_GE(l1_info.size, 10 * transfer_size) << "Insufficient L1 for test buffers";
    uint32_t l1_base = static_cast<uint32_t>(l1_info.base_address);

    const uint32_t l1_pcie_read_scratch_addr = l1_base + 0 * transfer_size;
    const uint32_t l1_pcie_batch_read_scratch_addr = l1_base + 1 * transfer_size;
    const uint32_t l1_onchip_read_src_addr = l1_base + 2 * transfer_size;
    const uint32_t l1_onchip_read_dst_addr = l1_base + 3 * transfer_size;
    const uint32_t l1_batch_onchip_read_dst_addr = l1_base + 4 * transfer_size;
    const uint32_t l1_write_src_addr = l1_base + 5 * transfer_size;
    const uint32_t l1_onchip_write_src_addr = l1_base + 6 * transfer_size;
    const uint32_t l1_onchip_write_dst_addr = l1_base + 7 * transfer_size;
    const uint32_t l1_batch_onchip_write_dst_addr = l1_base + 8 * transfer_size;
    const uint32_t l1_mid_result_addr = l1_base + 9 * transfer_size;

    auto& cluster = MetalContext::instance().get_cluster();
    const ChipId mmio_device_id = cluster.get_associated_mmio_device(device->id());
    const uint16_t channel = cluster.get_assigned_channel_for_device(device->id());

    void* host_hugepage_base = cluster.host_dma_address(0, mmio_device_id, channel);
    ASSERT_NE(host_hugepage_base, nullptr) << "Host hugepage is not mapped for this device";
    const uint32_t channel_size = cluster.get_host_channel_size(mmio_device_id, channel);
    ASSERT_GE(channel_size, test_consts::kHostBatchWriteOffset + transfer_size) << "Host channel too small";

    // The kernel ORs these offsets into NOC_XY_PCIE_ENCODING, which supplies the routing bit itself, so
    // the offsets passed down must be hugepage-relative. That is only the same as truncating
    // pcie_base + offset to 32 bits because the base has no low bits; assert that rather than assume it.
    const uint64_t dev_pcie_base = cluster.get_pcie_base_addr_from_device(device->id());
    ASSERT_EQ(dev_pcie_base & 0xFFFFFFFFull, 0u)
        << "PCIe base has low bits set, so hugepage offsets no longer map directly to device PCIe offsets";

    std::vector<uint32_t> onchip_read_pattern(num_words);
    test_consts::fill_pattern(onchip_read_pattern, 0xAAAA);
    std::vector<uint32_t> write_src_pattern(num_words);
    test_consts::fill_pattern(write_src_pattern, 0xBBBB);
    std::vector<uint32_t> onchip_write_pattern(num_words);
    test_consts::fill_pattern(onchip_write_pattern, 0xCCCC);
    std::vector<uint32_t> host_read_pattern(num_words);
    test_consts::fill_pattern(host_read_pattern, 0xEEEE);
    std::vector<uint32_t> poison(num_words, 0xDEADBEEFu);

    // Seed the hugepage region the kernel reads, and poison the two it writes, so an absent or misrouted
    // PCIe transfer cannot look like a pass.
    auto* hugepage = reinterpret_cast<uint8_t*>(host_hugepage_base);
    std::memcpy(hugepage + test_consts::kHostReadOffset, host_read_pattern.data(), transfer_size);
    std::memcpy(hugepage + test_consts::kHostWriteOffset, poison.data(), transfer_size);
    std::memcpy(hugepage + test_consts::kHostBatchWriteOffset, poison.data(), transfer_size);
    tt_driver_atomics::sfence();

    // Seed on-chip source patterns and poison every destination, including the two L1 landing zones for
    // the PCIe reads.
    ASSERT_TRUE(detail::WriteToDeviceL1(device, logical_core, l1_pcie_read_scratch_addr, poison));
    ASSERT_TRUE(detail::WriteToDeviceL1(device, logical_core, l1_pcie_batch_read_scratch_addr, poison));
    ASSERT_TRUE(detail::WriteToDeviceL1(device, logical_core, l1_onchip_read_src_addr, onchip_read_pattern));
    ASSERT_TRUE(detail::WriteToDeviceL1(device, logical_core, l1_onchip_read_dst_addr, poison));
    ASSERT_TRUE(detail::WriteToDeviceL1(device, logical_core, l1_batch_onchip_read_dst_addr, poison));
    ASSERT_TRUE(detail::WriteToDeviceL1(device, logical_core, l1_write_src_addr, write_src_pattern));
    ASSERT_TRUE(detail::WriteToDeviceL1(device, logical_core, l1_onchip_write_src_addr, onchip_write_pattern));
    ASSERT_TRUE(detail::WriteToDeviceL1(device, logical_core, l1_onchip_write_dst_addr, poison));
    ASSERT_TRUE(detail::WriteToDeviceL1(device, logical_core, l1_batch_onchip_write_dst_addr, poison));
    ASSERT_TRUE(detail::WriteToDeviceL1(device, logical_core, l1_mid_result_addr, poison));

    MetalContext::instance().get_cluster().l1_barrier(device->id());

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
                test_consts::kHostReadOffset,
                test_consts::kHostWriteOffset,
                test_consts::kHostBatchWriteOffset,
                l1_pcie_read_scratch_addr,
                l1_pcie_batch_read_scratch_addr,
                l1_onchip_read_src_addr,
                l1_onchip_read_dst_addr,
                l1_batch_onchip_read_dst_addr,
                l1_write_src_addr,
                l1_onchip_write_src_addr,
                l1_onchip_write_dst_addr,
                l1_batch_onchip_write_dst_addr,
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

    auto read_l1 = [&](uint32_t addr, std::vector<uint32_t>& out) {
        out.resize(num_words);
        return detail::ReadFromDeviceL1(
            device,
            logical_core,
            addr,
            std::span<uint8_t>(reinterpret_cast<uint8_t*>(out.data()), out.size() * sizeof(uint32_t)));
    };

    std::vector<uint32_t> mid_result(5);
    ASSERT_TRUE(detail::ReadFromDeviceL1(
        device,
        logical_core,
        l1_mid_result_addr,
        std::span<uint8_t>(reinterpret_cast<uint8_t*>(mid_result.data()), mid_result.size() * sizeof(uint32_t))));
    // Check liveness first. Without this, a kernel that never ran reports as four MID failures and sends
    // you looking at the MID logic instead of at why the kernel did nothing.
    ASSERT_EQ(mid_result[4], 0x5A5A5A5Au)
        << "Kernel body never executed, so the results below mean nothing. Check that the kernel compiled "
           "for this arch and that no kernel-disabling debug env var is set.";

    // Routing actually worked: the PCIe reads landed host data in L1, and the PCIe writes landed L1 data
    // in the hugepage. Without these, a build that never sets MID would still pass every check below.
    std::vector<uint32_t> pcie_read_result;
    ASSERT_TRUE(read_l1(l1_pcie_read_scratch_addr, pcie_read_result));
    EXPECT_EQ(pcie_read_result, host_read_pattern) << "noc_async_read_pcie did not read host memory";

    std::vector<uint32_t> pcie_batch_read_result;
    ASSERT_TRUE(read_l1(l1_pcie_batch_read_scratch_addr, pcie_batch_read_result));
    EXPECT_EQ(pcie_batch_read_result, host_read_pattern)
        << "noc_async_read_set_pcie_state batch did not read host memory";

    std::vector<uint32_t> host_write_result(num_words);
    std::memcpy(host_write_result.data(), hugepage + test_consts::kHostWriteOffset, transfer_size);
    EXPECT_EQ(host_write_result, write_src_pattern) << "noc_async_write_pcie did not write host memory";

    std::vector<uint32_t> host_batch_write_result(num_words);
    std::memcpy(host_batch_write_result.data(), hugepage + test_consts::kHostBatchWriteOffset, transfer_size);
    EXPECT_EQ(host_batch_write_result, write_src_pattern)
        << "noc_async_write_set_pcie_state batch did not write host memory";

    // MID must be exactly 0 once each PCIe path is done with it. Slots 0 and 1 cover the single
    // transaction wrappers, which clear it themselves. Slots 2 and 3 cover the batch path.
    EXPECT_EQ(mid_result[0], 0u) << "NOC_TARG_ADDR_MID not cleared by noc_async_read_pcie";
    EXPECT_EQ(mid_result[1], 0u) << "NOC_RET_ADDR_MID not cleared by noc_async_write_pcie";
    EXPECT_EQ(mid_result[2], 0u) << "NOC_TARG_ADDR_MID not cleared by noc_async_read_clear_pcie_state";
    EXPECT_EQ(mid_result[3], 0u) << "NOC_RET_ADDR_MID not cleared by noc_async_write_clear_pcie_state";

    // The behavioral consequence: the on-chip transfer issued after each PCIe path, on the same command
    // buffer, must have landed correctly rather than being misrouted by a stale MID.
    std::vector<uint32_t> result;
    ASSERT_TRUE(read_l1(l1_onchip_read_dst_addr, result));
    EXPECT_EQ(result, onchip_read_pattern) << "On-chip read after noc_async_read_pcie was misrouted";

    ASSERT_TRUE(read_l1(l1_batch_onchip_read_dst_addr, result));
    EXPECT_EQ(result, onchip_read_pattern) << "On-chip read after the PCIe read batch was misrouted";

    ASSERT_TRUE(read_l1(l1_onchip_write_dst_addr, result));
    EXPECT_EQ(result, onchip_write_pattern) << "On-chip write after noc_async_write_pcie was misrouted";

    ASSERT_TRUE(read_l1(l1_batch_onchip_write_dst_addr, result));
    EXPECT_EQ(result, onchip_write_pattern) << "On-chip write after the PCIe write batch was misrouted";
}

}  // namespace tt::tt_metal
