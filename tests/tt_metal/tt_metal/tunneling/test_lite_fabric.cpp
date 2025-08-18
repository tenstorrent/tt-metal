// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>

#include <fmt/ranges.h>
#include <gtest/gtest.h>
#include <umd/device/types/arch.h>
#include <umd/device/types/cluster_descriptor_types.h>
#include <umd/device/types/xy_pair.h>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/control_plane.hpp>

#include "context/metal_context.hpp"
#include "core_coord.hpp"
#include "lite_fabric.hpp"
#include "rtoptions.hpp"
#include "llrt/hal.hpp"
#include "tt_cluster.hpp"
#include "lite_fabric_host_util.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal.hpp"
#include "build.hpp"

#define CHECK_TEST_REQS()                                                                       \
    if (tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()) != tt::ARCH::BLACKHOLE) { \
        GTEST_SKIP() << "Blackhole only";                                                       \
    }                                                                                           \
    if (tt::tt_metal::GetNumAvailableDevices() != 2) {                                          \
        GTEST_SKIP() << "2 Devices are required";                                               \
    }

TEST(Tunneling, DISABLED_LiteFabricInitWithMetal) {
    CHECK_TEST_REQS();

    auto devices = tt::tt_metal::detail::CreateDevices({0, 1});
    auto desc = lite_fabric::GetSystemDescriptor2Devices(0, 1);

    auto lite_fabric = lite_fabric::LaunchLiteFabricWithMetal(devices, desc);
    lite_fabric::TerminateLiteFabricWithMetal(tt::tt_metal::MetalContext::instance().get_cluster(), desc);
    tt::tt_metal::detail::WaitProgramDone(devices[0], *lite_fabric);
    tt::tt_metal::detail::CloseDevices(devices);
}

TEST(Tunneling, LiteFabricBuildOnly) {
    auto home_directory = std::filesystem::path(std::getenv("TT_METAL_HOME"));
    auto output_directory = home_directory / "lite_fabric";
    auto rtoptions = tt::llrt::RunTimeOptions();
    auto hal = tt::tt_metal::Hal(tt::ARCH::BLACKHOLE, false);
    auto cluster = std::make_shared<tt::Cluster>(rtoptions, hal);

    if (lite_fabric::CompileLiteFabric(*cluster.get(), home_directory, output_directory)) {
        throw std::runtime_error("Failed to compile lite fabric");
    }
    if (lite_fabric::LinkLiteFabric(home_directory, output_directory, output_directory / "lite_fabric.elf")) {
        throw std::runtime_error("Failed to link lite fabric");
    }
}

TEST(Tunneling, LiteFabricInit) {
    auto rtoptions = tt::llrt::RunTimeOptions();
    auto hal = tt::tt_metal::Hal(tt::ARCH::BLACKHOLE, false);
    auto cluster = std::make_shared<tt::Cluster>(rtoptions, hal);

    auto desc = lite_fabric::GetSystemDescriptor2Devices(0, 1);

    auto home_directory = std::filesystem::path(std::getenv("TT_METAL_HOME"));
    auto output_directory = home_directory / "lite_fabric";
    lite_fabric::LaunchLiteFabric(*cluster.get(), hal, desc);
    lite_fabric::TerminateLiteFabricWithMetal(*cluster.get(), desc);
    lite_fabric::SetResetState(*cluster.get(), desc, true);
}

TEST(Tunneling, LiteFabricWriteAllCores) {
    CHECK_TEST_REQS();

    auto rtoptions = tt::llrt::RunTimeOptions();
    auto hal = tt::tt_metal::Hal(tt::ARCH::BLACKHOLE, false);
    auto cluster = std::make_shared<tt::Cluster>(rtoptions, hal);

    auto desc = lite_fabric::GetSystemDescriptor2Devices(0, 1);

    auto grid_size = cluster->get_soc_desc(1).get_grid_size(CoreType::TENSIX);

    lite_fabric::LaunchLiteFabric(*cluster.get(), hal, desc);
    const auto& tunnel = desc.tunnels_from_mmio[0];
    log_info(tt::LogTest, "Tunnel: {} -> {}", tunnel.mmio_cxy_virtual().str(), tunnel.connected_cxy_virtual().str());

    uint32_t sender_channel_base = lite_fabric::LiteFabricMemoryMap::get_send_channel_addr();
    auto host_interface = lite_fabric::LiteFabricMemoryMap::make_host_interface();

    // This will wrap the sender channel multiple times and write to all worker cores
    uint32_t payload_size_bytes = 4096;  // (128 * 1024) + 512;
    uint32_t l1_base = 0x10000;
    log_info(tt::LogMetal, "Device 1 Grid {}", grid_size.str());

    std::unordered_map<CoreCoord, std::vector<uint32_t>> test_data_per_worker;
    for (int worker_x = 0; worker_x < grid_size.x; ++worker_x) {
        for (int worker_y = 0; worker_y < grid_size.y; ++worker_y) {
            CoreCoord logical_worker{worker_x, worker_y};
            log_info(tt::LogMetal, "Writing to worker {}", logical_worker.str());
            CoreCoord virtual_worker =
                cluster->get_virtual_coordinate_from_logical_coordinates(1, logical_worker, CoreType::WORKER);
            const uint64_t dest_noc_upper =
                (uint64_t(virtual_worker.y) << (36 + 6)) | (uint64_t(virtual_worker.x) << 36);
            uint64_t dest_noc_addr = dest_noc_upper | (uint64_t)l1_base;

            test_data_per_worker[logical_worker] = create_random_vector_of_bfloat16(
                payload_size_bytes, 100, std::chrono::system_clock::now().time_since_epoch().count(), 1.0f);

            host_interface.write_any_len(
                test_data_per_worker[logical_worker].data(),
                payload_size_bytes,
                tunnel.mmio_cxy_virtual(),
                dest_noc_addr);
        }
    }

    host_interface.barrier(tunnel.mmio_cxy_virtual());

    for (int worker_x = 0; worker_x < grid_size.x; ++worker_x) {
        for (int worker_y = 0; worker_y < grid_size.y; ++worker_y) {
            CoreCoord logical_worker{worker_x, worker_y};
            CoreCoord virtual_worker =
                cluster->get_virtual_coordinate_from_logical_coordinates(1, logical_worker, CoreType::WORKER);
            std::vector<uint32_t> read_data(payload_size_bytes / sizeof(uint32_t));

            tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                read_data.data(), payload_size_bytes, tt_cxy_pair{1, virtual_worker}, l1_base);
            ASSERT_EQ(read_data, test_data_per_worker[logical_worker])
                << fmt::format("Data mismatch for worker {}", logical_worker.str());
        }
    }

    lite_fabric::TerminateLiteFabric(*cluster.get(), desc);
}

TEST(Tunneling, LiteFabricReads) {
    CHECK_TEST_REQS();

    auto rtoptions = tt::llrt::RunTimeOptions();
    auto hal = tt::tt_metal::Hal(tt::ARCH::BLACKHOLE, false);
    auto cluster = std::make_shared<tt::Cluster>(rtoptions, hal);

    auto desc = lite_fabric::GetSystemDescriptor2Devices(0, 1);

    lite_fabric::LaunchLiteFabric(*cluster.get(), hal, desc);
    const auto& tunnel = desc.tunnels_from_mmio[0];
    log_info(tt::LogTest, "Tunnel: {} -> {}", tunnel.mmio_cxy_virtual().str(), tunnel.connected_cxy_virtual().str());

    auto host_interface = lite_fabric::LiteFabricMemoryMap::make_host_interface();

    uint32_t payload_size_bytes = 4 * 1024;
    uint32_t max_payload_size = host_interface.get_max_payload_data_size_bytes();
    uint32_t num_pages = payload_size_bytes / max_payload_size;

    uint32_t l1_base = 0x10000;
    log_info(tt::LogMetal, "Device 1 Grid {}", cluster->get_soc_desc(1).get_grid_size(CoreType::TENSIX).str());

    std::unordered_map<CoreCoord, std::vector<uint32_t>> test_data_per_worker;
    CoreCoord logical_worker{0, 0};
    CoreCoord virtual_worker =
        cluster->get_virtual_coordinate_from_logical_coordinates(1, logical_worker, CoreType::WORKER);
    const uint64_t dest_noc_upper = (uint64_t(virtual_worker.y) << (36 + 6)) | (uint64_t(virtual_worker.x) << 36);
    uint64_t dest_noc_addr = dest_noc_upper | (uint64_t)l1_base;

    auto allOnes = create_random_vector_of_bfloat16(
        payload_size_bytes, 100, std::chrono::system_clock::now().time_since_epoch().count(), 1.0f);
    auto allTwos = create_random_vector_of_bfloat16(
        payload_size_bytes, 100, std::chrono::system_clock::now().time_since_epoch().count(), 2.0f);

    uint64_t onesNocAddr = dest_noc_addr;
    uint64_t twosNocAddr = dest_noc_addr + payload_size_bytes;

    host_interface.write_any_len(allOnes.data(), payload_size_bytes, tunnel.mmio_cxy_virtual(), onesNocAddr);
    host_interface.write_any_len(allTwos.data(), payload_size_bytes, tunnel.mmio_cxy_virtual(), twosNocAddr);

    // Try reading back the data we just wrote
    host_interface.barrier(tunnel.mmio_cxy_virtual());

    log_info(tt::LogMetal, "Reading back data from Device 1 worker core {}", logical_worker.str());
    {
        // Read out
        std::vector<uint32_t> read_data(payload_size_bytes / sizeof(uint32_t));
        host_interface.read_any_len(read_data.data(), payload_size_bytes, tunnel.mmio_cxy_virtual(), onesNocAddr);
        log_info(
            tt::LogMetal,
            "Read out data from {} {:#x} {} elements",
            tunnel.mmio_cxy_virtual().str(),
            onesNocAddr,
            read_data.size());

        ASSERT_EQ(read_data, allOnes);
    }

    log_info(tt::LogMetal, "Reading back data from Device 1 worker core {}", logical_worker.str());
    {
        // Read out
        std::vector<uint32_t> read_data(payload_size_bytes / sizeof(uint32_t));
        host_interface.read_any_len(read_data.data(), payload_size_bytes, tunnel.mmio_cxy_virtual(), twosNocAddr);
        log_info(
            tt::LogMetal,
            "Read out data from {} {:#x} {} elements",
            tunnel.mmio_cxy_virtual().str(),
            twosNocAddr,
            read_data.size());

        ASSERT_EQ(read_data, allTwos);
    }

    lite_fabric::TerminateLiteFabric(*cluster.get(), desc);
}

TEST(Tunneling, LiteFabricBarrier) {
    CHECK_TEST_REQS();

    auto rtoptions = tt::llrt::RunTimeOptions();
    auto hal = tt::tt_metal::Hal(tt::ARCH::BLACKHOLE, false);
    auto cluster = std::make_shared<tt::Cluster>(rtoptions, hal);

    auto desc = lite_fabric::GetSystemDescriptor2Devices(0, 1);

    lite_fabric::LaunchLiteFabric(*cluster.get(), hal, desc);
    const auto& tunnel = desc.tunnels_from_mmio[0];

    auto host_interface = lite_fabric::LiteFabricMemoryMap::make_host_interface();

    host_interface.barrier(tunnel.mmio_cxy_virtual());
    lite_fabric::TerminateLiteFabric(*cluster.get(), desc);
}

TEST(Tunneling, LiteFabricSmallWrites) {
    CHECK_TEST_REQS();

    auto rtoptions = tt::llrt::RunTimeOptions();
    auto hal = tt::tt_metal::Hal(tt::ARCH::BLACKHOLE, false);
    auto cluster = std::make_shared<tt::Cluster>(rtoptions, hal);

    auto desc = lite_fabric::GetSystemDescriptor2Devices(0, 1);

    lite_fabric::LaunchLiteFabric(*cluster.get(), hal, desc);
    const auto& tunnel = desc.tunnels_from_mmio[0];

    auto host_interface = lite_fabric::LiteFabricMemoryMap::make_host_interface();

    CoreCoord logical_worker{0, 0};
    CoreCoord virtual_worker =
        cluster->get_virtual_coordinate_from_logical_coordinates(1, logical_worker, CoreType::WORKER);
    uint32_t l1_base = 0x10000;
    uint64_t dest_noc_upper = (uint64_t(virtual_worker.y) << (36 + 6)) | (uint64_t(virtual_worker.x) << 36);
    uint64_t dest_noc_addr = dest_noc_upper | (uint64_t)l1_base;

    // due to unaligned, the read will be only 1B of the original data. mask off the irrelevant bits
    // ensure they did not overwrite each other
    constexpr uint32_t num_writes = 64;
    std::vector<uint32_t> write_data = create_random_vector_of_bfloat16(
        num_writes * sizeof(uint32_t), 100, std::chrono::system_clock::now().time_since_epoch().count(), 1.0f);
    for (int i = 1; i < num_writes; ++i) {
        host_interface.write_any_len(&write_data[i], 1, tunnel.mmio_cxy_virtual(), dest_noc_addr + i);
        write_data[i] = write_data[i] & 0xff;
    }
    host_interface.barrier(tunnel.mmio_cxy_virtual());

    for (int i = 1; i < num_writes; ++i) {
        std::vector<uint32_t> read_data(1);
        tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            read_data.data(), 1, tt_cxy_pair{1, virtual_worker}, l1_base + i);
        log_info(
            tt::LogMetal, "Read data {} from {:#x} {:#x} expecting {:#x}", i, l1_base + i, read_data[0], write_data[i]);
        ASSERT_EQ(read_data[0], write_data[i]);
    }

    lite_fabric::TerminateLiteFabric(*cluster.get(), desc);
}

TEST(Tunneling, LiteFabricUnalignedWrites) {
    CHECK_TEST_REQS();

    auto rtoptions = tt::llrt::RunTimeOptions();
    auto hal = tt::tt_metal::Hal(tt::ARCH::BLACKHOLE, false);
    auto cluster = std::make_shared<tt::Cluster>(rtoptions, hal);

    auto desc = lite_fabric::GetSystemDescriptor2Devices(0, 1);

    lite_fabric::LaunchLiteFabric(*cluster.get(), hal, desc);
    const auto& tunnel = desc.tunnels_from_mmio[0];

    auto host_interface = lite_fabric::LiteFabricMemoryMap::make_host_interface();

    CoreCoord logical_worker{0, 0};
    CoreCoord virtual_worker =
        cluster->get_virtual_coordinate_from_logical_coordinates(1, logical_worker, CoreType::WORKER);
    uint32_t l1_base = 0x10000;

    // Make destination not aligned by various amounts
    for (int aligned_offset = 0; aligned_offset < 16; ++aligned_offset) {
        uint32_t addr = l1_base + aligned_offset;
        uint64_t dest_noc_upper = (uint64_t(virtual_worker.y) << (36 + 6)) | (uint64_t(virtual_worker.x) << 36);
        uint64_t dest_noc_addr = dest_noc_upper | (uint64_t)addr;
        log_info(tt::LogMetal, "Testing unaligned write to {:#x}", addr);
        std::vector<uint32_t> test_data = create_random_vector_of_bfloat16(
            4096, 100, std::chrono::system_clock::now().time_since_epoch().count(), 1.0f);

        host_interface.write_any_len(
            test_data.data(), test_data.size() * sizeof(uint32_t), tunnel.mmio_cxy_virtual(), dest_noc_addr);
        host_interface.barrier(tunnel.mmio_cxy_virtual());

        std::vector<uint32_t> read_data(test_data.size());
        tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            read_data.data(), test_data.size() * sizeof(uint32_t), tt_cxy_pair{1, virtual_worker}, addr);
        ASSERT_EQ(read_data, test_data);
    }

    lite_fabric::TerminateLiteFabric(*cluster.get(), desc);
}

TEST(Tunneling, LiteFabricUnalignedReads) {
    CHECK_TEST_REQS();

    auto rtoptions = tt::llrt::RunTimeOptions();
    auto hal = tt::tt_metal::Hal(tt::ARCH::BLACKHOLE, false);
    auto cluster = std::make_shared<tt::Cluster>(rtoptions, hal);

    auto desc = lite_fabric::GetSystemDescriptor2Devices(0, 1);

    lite_fabric::LaunchLiteFabric(*cluster.get(), hal, desc);
    const auto& tunnel = desc.tunnels_from_mmio[0];

    auto host_interface = lite_fabric::LiteFabricMemoryMap::make_host_interface();

    CoreCoord logical_worker{0, 0};
    CoreCoord virtual_worker =
        cluster->get_virtual_coordinate_from_logical_coordinates(1, logical_worker, CoreType::WORKER);
    uint32_t l1_base = 0x10000;
    uint64_t dest_noc_upper = (uint64_t(virtual_worker.y) << (36 + 6)) | (uint64_t(virtual_worker.x) << 36);
    uint64_t dest_noc_addr = dest_noc_upper | (uint64_t)l1_base;

    // Using a known good API for this
    std::vector<uint32_t> write_data =
        create_random_vector_of_bfloat16(4096, 100, std::chrono::system_clock::now().time_since_epoch().count(), 1.0f);
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        write_data.data(), write_data.size() * sizeof(uint32_t), tt_cxy_pair{1, virtual_worker}, l1_base);
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(1);

    // Read a 256-byte slice (64 elements) from an unaligned address (l1_base + 7)
    constexpr size_t kNumBytesToRead = 64 * sizeof(uint32_t);
    const size_t unaligned_offset_bytes = 7;

    std::vector<uint8_t> read_bytes(kNumBytesToRead);
    host_interface.read(
        read_bytes.data(), read_bytes.size(), tunnel.mmio_cxy_virtual(), dest_noc_addr + unaligned_offset_bytes);

    // Build expected bytes by taking a byte view of the original write buffer and slicing by the same offset
    const uint8_t* write_bytes = reinterpret_cast<const uint8_t*>(write_data.data());
    std::vector<uint8_t> expected_bytes(read_bytes.size());
    std::memcpy(expected_bytes.data(), write_bytes + unaligned_offset_bytes, read_bytes.size());

    ASSERT_EQ(read_bytes, expected_bytes);

    lite_fabric::TerminateLiteFabric(*cluster.get(), desc);
}

TEST(Tunneling, LiteFabricP300) {
    CHECK_TEST_REQS();
    auto rtoptions = tt::llrt::RunTimeOptions();
    auto hal = tt::tt_metal::Hal(tt::ARCH::BLACKHOLE, false);
    auto cluster = std::make_shared<tt::Cluster>(rtoptions, hal);

    if (cluster->get_board_type(0) != BoardType::P300)  {
        GTEST_SKIP() << "P300 board type is required";
    }

    auto desc = lite_fabric::GetSystemDescriptor2Devices(0, 1);
    for (const auto& tunnel : desc.tunnels_from_mmio) {
        log_info(
            tt::LogTest,
            "Tunnel from device {} core {} (virtual={}) to device {} core {} (virtual={})",
            tunnel.mmio_id,
            tunnel.mmio_core_logical.str(),
            tunnel.mmio_cxy_virtual().str(),
            tunnel.connected_id,
            tunnel.connected_core_logical.str(),
            tunnel.connected_cxy_virtual().str());
    }
    for (const auto& [device_id, channel_mask] : desc.enabled_eth_channels) {
        log_info(tt::LogTest, "Device {} enabled eth channel mask {:0b}", device_id, channel_mask);
    }

    lite_fabric::LaunchLiteFabric(*cluster.get(), hal, desc);
    lite_fabric::TerminateLiteFabric(*cluster.get(), desc);
}
