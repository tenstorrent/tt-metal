// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <thread>

#include <fmt/ranges.h>
#include <gtest/gtest.h>
#include <umd/device/types/arch.h>
#include <umd/device/types/cluster_descriptor_types.h>
#include <umd/device/types/xy_pair.h>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/control_plane.hpp>

#include "allocator.hpp"
#include "context/metal_context.hpp"
#include "core_coord.hpp"
#include "data_types.hpp"
#include "fabric_edm_packet_header.hpp"
#include "lite_fabric.hpp"
#include "hal_types.hpp"
#include "kernel.hpp"
#include "kernel_types.hpp"
#include "lite_fabric_constants.hpp"
#include "rtoptions.hpp"
#include "llrt/hal.hpp"
#include "tt_cluster.hpp"
#include "tt_memory.h"
#include "build.hpp"
#include "lite_fabric_host.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal.hpp"

#define CHECK_TEST_REQS()                                                                       \
    if (tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()) != tt::ARCH::BLACKHOLE) { \
        GTEST_SKIP() << "Blackhole only";                                                       \
    }                                                                                           \
    if (tt::tt_metal::GetNumAvailableDevices() != 2) {                                          \
        GTEST_SKIP() << "2 Devices are required";                                               \
    }                                                                                           \
    if (!std::getenv("TT_METAL_SLOW_DISPATCH_MODE")) {                                          \
        GTEST_SKIP() << "TT_METAL_SLOW_DISPATCH_MODE required";                                 \
    }                                                                                           \
    if (!std::getenv("TT_METAL_CLEAR_L1")) {                                                    \
        GTEST_SKIP() << "TT_METAL_CLEAR_L1 required";                                           \
    }

TEST(Tunneling, LiteFabricInitWithMetal) {
    CHECK_TEST_REQS();

    auto devices = tt::tt_metal::detail::CreateDevices({0, 1});
    auto desc = lite_fabric::GetSystemDescriptor2Devices(devices, 0, 1);

    auto lite_fabric = lite_fabric::LaunchLiteFabricWithMetal(devices, desc);

    lite_fabric::TerminateLiteFabric(tt::tt_metal::MetalContext::instance().get_cluster(), desc);

    tt::tt_metal::detail::WaitProgramDone(devices[0], *lite_fabric);

    tt::tt_metal::detail::CloseDevices(devices);
}

TEST(Tunneling, LiteFabricWrites) {
    CHECK_TEST_REQS();

    auto devices = tt::tt_metal::detail::CreateDevices({0, 1});
    auto desc = lite_fabric::GetSystemDescriptor2Devices(devices, 0, 1);

    auto lite_fabric = lite_fabric::LaunchLiteFabricWithMetal(devices, desc);
    const auto& tunnel = desc.tunnels_from_mmio[0];
    log_info(tt::LogTest, "Tunnel: {} -> {}", tunnel.mmio_cxy_virtual().str(), tunnel.connected_cxy_virtual().str());

    uint32_t sender_channel_base = lite_fabric::LiteFabricMemoryMap::get_send_channel_addr();
    lite_fabric::HostToLiteFabricInterface<SENDER_NUM_BUFFERS_ARRAY[0], CHANNEL_BUFFER_SIZE> host_interface;
    host_interface.host_interface_on_device_addr = lite_fabric::LiteFabricMemoryMap::get_host_interface_addr();

    // This will wrap the sender channel multiple times and write to all worker cores
    uint32_t payload_size_bytes = 128 * 1024;
    uint32_t max_payload_size = host_interface.get_max_payload_data_size_bytes();
    uint32_t num_pages = payload_size_bytes / max_payload_size;

    uint32_t l1_base = devices[1]->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    log_info(tt::LogMetal, "Device 1 Grid {}", devices[1]->compute_with_storage_grid_size().str());

    std::unordered_map<CoreCoord, std::vector<uint32_t>> test_data_per_worker;
    for (int worker_x = 0; worker_x < devices[1]->compute_with_storage_grid_size().x; ++worker_x) {
        for (int worker_y = 0; worker_y < devices[1]->compute_with_storage_grid_size().y; ++worker_y) {
            CoreCoord logical_worker{worker_x, worker_y};
            CoreCoord virtual_worker = devices[1]->virtual_core_from_logical_core(logical_worker, CoreType::WORKER);
            const uint64_t dest_noc_upper =
                (uint64_t(virtual_worker.y) << (36 + 6)) | (uint64_t(virtual_worker.x) << 36);
            uint64_t dest_noc_addr = dest_noc_upper | (uint64_t)l1_base;

            test_data_per_worker[logical_worker] = create_random_vector_of_bfloat16(
                payload_size_bytes, 100, std::chrono::system_clock::now().time_since_epoch().count(), 1.0f);

            for (int page = 0; page < num_pages; ++page) {
                uint64_t page_dest = dest_noc_addr + (page * max_payload_size);

                tt::tt_fabric::LowLatencyPacketHeader header;
                header.to_chip_unicast(1);
                header.to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{page_dest}, max_payload_size);

                uint32_t header_address = host_interface.get_next_send_buffer_slot_address(sender_channel_base);
                uint32_t data_address = header_address + sizeof(tt::tt_fabric::LowLatencyPacketHeader);

                host_interface.wait_for_empty_write_slot(tunnel.mmio_cxy_virtual());
                host_interface.send_payload_without_header_non_blocking_from_address(
                    (void*)(reinterpret_cast<uintptr_t>(test_data_per_worker[logical_worker].data()) +
                            (page * max_payload_size)),
                    max_payload_size,
                    tunnel.mmio_cxy_virtual(),
                    sender_channel_base);
                host_interface.send_payload_flush_non_blocking_from_address(
                    header, tunnel.mmio_cxy_virtual(), sender_channel_base);

                log_info(
                    tt::LogMetal,
                    "Device {} Worker Core {} Page {} Written to sender channel. Header {:#x} Data: {:#x} Dest: {:#x}",
                    1,
                    logical_worker,
                    page,
                    header_address,
                    data_address,
                    page_dest);
            }
        }
    }

    // Checking written data using PCIe. Sleep because barrier is not implemented yet
    host_interface.barrier(tunnel.mmio_cxy_virtual());
    std::this_thread::sleep_for(std::chrono::seconds(5));

    for (int worker_x = 0; worker_x < devices[1]->compute_with_storage_grid_size().x; ++worker_x) {
        for (int worker_y = 0; worker_y < devices[1]->compute_with_storage_grid_size().y; ++worker_y) {
            CoreCoord logical_worker{worker_x, worker_y};
            CoreCoord virtual_worker = devices[1]->virtual_core_from_logical_core(logical_worker, CoreType::WORKER);
            std::vector<uint32_t> read_data(payload_size_bytes / sizeof(uint32_t));

            tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                read_data.data(), payload_size_bytes, tt_cxy_pair{1, virtual_worker}, l1_base);
            ASSERT_EQ(read_data, test_data_per_worker[logical_worker])
                << fmt::format("Data mismatch for worker {}", logical_worker.str());
        }
    }

    lite_fabric::TerminateLiteFabric(tt::tt_metal::MetalContext::instance().get_cluster(), desc);
    tt::tt_metal::detail::WaitProgramDone(devices[0], *lite_fabric);

    tt::tt_metal::detail::CloseDevices(devices);
}
