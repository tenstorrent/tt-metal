// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Exercises device reads/writes to host memory through the PCIe NOC path (host hugepage /
// sysmem), using the same noc_async_read / noc_async_write pattern as cq_prefetch.cpp and
// test_kernels/.../pcie_write_16b.cpp. Intended as a fast alternative to SD prefetch
// TestTerminate when validating Quasar VCS simulator host DMA.

#include "device_fixture.hpp"

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "dispatch/memcpy.hpp"
#include <impl/debug/dprint_server.hpp>
#include <impl/debug/watcher_server.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt.hpp"
#include "llrt/tt_cluster.hpp"

#include <chrono>
#include <thread>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace tt::tt_metal {

namespace {

// Offsets within channel-0 host hugepage / sysmem. Keep clear of dispatch CQ headroom.
constexpr uint32_t kHostSrcOffset = 0x10000;
constexpr uint32_t kHostDstOffset = 0x11000;
constexpr uint32_t kTransferSizeBytes = 128;

void fill_pattern(std::vector<uint32_t>& data) {
    for (uint32_t i = 0; i < data.size(); ++i) {
        data[i] = 0xA5A50000u | (i & 0xFFFFu);
    }
}

// MeshDevice close destroys MetalContext, which stops watcher/dprint server threads.
// On Quasar sim, watcher dumps can take tens of seconds and contend with DPRINT reads.
// Wait for watcher to finish a couple of dump cycles first (MeshWatcherFixture pattern),
// then drain DPRINT before teardown.
void sync_debug_servers_before_teardown() {
    auto& ctx = MetalContext::instance();
    if (ctx.rtoptions().get_watcher_enabled() && ctx.watcher_server()) {
        const int dumps_at_end = ctx.watcher_server()->dump_count();
        while (ctx.watcher_server()->dump_count() < dumps_at_end + 2) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    if (ctx.rtoptions().get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint) && ctx.dprint_server()) {
        ctx.dprint_server()->await();
    }
}

}  // namespace

TEST_F(QuasarMeshDeviceSingleCardFixture, HostHugepagePcieLoopback) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];
    TT_FATAL(device->is_mmio_capable(), "Host hugepage test requires an MMIO-capable device");

    auto& cluster = MetalContext::instance().get_cluster();
    const ChipId mmio_device_id = cluster.get_associated_mmio_device(device->id());
    const uint16_t channel = cluster.get_assigned_channel_for_device(device->id());

    void* host_hugepage_base = cluster.host_dma_address(0, mmio_device_id, channel);
    ASSERT_NE(host_hugepage_base, nullptr) << "Host hugepage is not mapped for this device";

    const uint32_t channel_size = cluster.get_host_channel_size(mmio_device_id, channel);
    ASSERT_GE(channel_size, kHostDstOffset + kTransferSizeBytes) << "Host channel too small for test buffers";

    // Device-side PCIe byte offsets mirror host hugepage offsets (see SimulationSysmemManager mapping).
    const uint64_t pcie_base = cluster.get_pcie_base_addr_from_device(device->id());
    const uint32_t host_src_pcie_addr = static_cast<uint32_t>(pcie_base + kHostSrcOffset);
    const uint32_t host_dst_pcie_addr = static_cast<uint32_t>(pcie_base + kHostDstOffset);

    const experimental::NodeCoord node{0, 0};
    const uint32_t l1_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    const uint32_t l1_staging_addr = l1_base + l1_alignment * 8;

    // Step 1: Host writes known pattern to src region in hugepage.
    std::vector<uint32_t> expected(kTransferSizeBytes / sizeof(uint32_t));
    fill_pattern(expected);
    auto* host_src_ptr = reinterpret_cast<uint8_t*>(host_hugepage_base) + kHostSrcOffset;
    std::memcpy(host_src_ptr, expected.data(), kTransferSizeBytes);

    // Poison dst so we can tell the kernel actually wrote it.
    std::vector<uint32_t> poison(expected.size(), 0xDEADBEEFu);
    auto* host_dst_ptr = reinterpret_cast<uint8_t*>(host_hugepage_base) + kHostDstOffset;
    std::memcpy(host_dst_ptr, poison.data(), kTransferSizeBytes);
    tt_driver_atomics::sfence();

    // Step 2-4: Kernel reads host src -> L1, then L1 -> host dst (via compile-time args).
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    const experimental::KernelSpecName DM_KERNEL{"device_pcie_loopback"};

    experimental::KernelSpec dm_kernel_spec{
        .unique_id = DM_KERNEL,
        .source = OVERRIDE_KERNEL_PREFIX
        "tests/tt_metal/tt_metal/data_movement/device_pcie_loopback/kernels/device_pcie_loopback.cpp",
        .num_threads = 1,
        .compile_time_args =
            {{"host_src_pcie_addr", host_src_pcie_addr},
             {"host_dst_pcie_addr", host_dst_pcie_addr},
             {"l1_staging_addr", l1_staging_addr},
             {"transfer_size_bytes", kTransferSizeBytes}},
        .hw_config = experimental::DataMovementGen2Config{},
    };

    experimental::WorkUnitSpec main_wu{
        .name = "main",
        .kernels = {DM_KERNEL},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "device_pcie_loopback",
        .kernels = {dm_kernel_spec},
        .work_units = {main_wu},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    // Step 5: Host verifies dst hugepage region matches src.
    std::vector<uint32_t> host_dst_readback(expected.size());
    std::memcpy(host_dst_readback.data(), host_dst_ptr, kTransferSizeBytes);
    EXPECT_EQ(host_dst_readback, expected) << "Host dst hugepage mismatch after device PCIe write";

    // Step 6: Host verifies L1 staging matches src.
    std::vector<uint32_t> l1_readback;
    detail::ReadFromDeviceL1(device, node, l1_staging_addr, kTransferSizeBytes, l1_readback);
    EXPECT_EQ(l1_readback, expected) << "L1 staging mismatch after device PCIe read";

    sync_debug_servers_before_teardown();
}

}  // namespace tt::tt_metal
