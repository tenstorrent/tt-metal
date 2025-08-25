// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>

#include <fmt/ranges.h>
#include <gtest/gtest.h>
#include <umd/device/types/arch.h>
#include <umd/device/types/cluster_descriptor_types.h>
#include <umd/device/types/xy_pair.h>

#include <enchantum/enchantum.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/fabric.hpp>

#include "allocator.hpp"
#include "context/metal_context.hpp"
#include "core_coord.hpp"
#include "distributed.hpp"
#include "fabric_types.hpp"
#include "kernel_types.hpp"
#include "mesh_workload.hpp"
#include "rtoptions.hpp"
#include "llrt/hal.hpp"
#include "tt_cluster.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal.hpp"

#include "tt_metal/lite_fabric/hw/inc/host_interface.hpp"
#include "tt_metal/lite_fabric/host_util.hpp"
#include "tt_metal/lite_fabric/build.hpp"

#define CHECK_TEST_REQS()                                                                       \
    if (tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()) != tt::ARCH::BLACKHOLE) { \
        GTEST_SKIP() << "Blackhole only";                                                       \
    }                                                                                           \
    if (tt::tt_metal::GetNumAvailableDevices() < 2) {                                           \
        GTEST_SKIP() << "At least 2 Devices are required";                                      \
    }                                                                                           \
    if (tt::tt_metal::GetClusterType() != tt::tt_metal::ClusterType::P150 &&                    \
        tt::tt_metal::GetClusterType() != tt::tt_metal::ClusterType::P300) {                    \
        GTEST_SKIP() << "P150/P300 only";                                                       \
    }

namespace {

// Performs write and read-back verification test across all worker cores
// Returns the test data map for further verification if needed
template <typename HOST_INTERFACE>
std::unordered_map<CoreCoord, std::vector<uint32_t>> perform_write_read_test(
    const lite_fabric::SystemDescriptor& desc,
    HOST_INTERFACE& host_interface,
    uint32_t payload_size_bytes,
    uint32_t l1_base,
    uint32_t device_id) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& tunnel = desc.tunnels_from_mmio[0];
    auto grid_size = cluster.get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);

    log_info(tt::LogMetal, "Device {} Grid {}", device_id, grid_size.str());

    std::unordered_map<CoreCoord, std::vector<uint32_t>> test_data_per_worker;

    // Write test data to all worker cores
    for (int worker_x = 0; worker_x < grid_size.x; ++worker_x) {
        for (int worker_y = 0; worker_y < grid_size.y; ++worker_y) {
            CoreCoord logical_worker{worker_x, worker_y};
            log_debug(tt::LogMetal, "Writing to worker {}", logical_worker.str());
            CoreCoord virtual_worker =
                cluster.get_virtual_coordinate_from_logical_coordinates(device_id, logical_worker, CoreType::WORKER);
            const uint64_t dest_noc_upper =
                (uint64_t(virtual_worker.y) << (36 + 6)) | (uint64_t(virtual_worker.x) << 36);
            uint64_t dest_noc_addr = dest_noc_upper | (uint64_t)l1_base;

            test_data_per_worker[logical_worker] = create_random_vector_of_bfloat16(
                payload_size_bytes, 100, std::chrono::system_clock::now().time_since_epoch().count(), 1.0f);

            host_interface.write_any_len(
                test_data_per_worker[logical_worker].data(), payload_size_bytes, dest_noc_addr);
        }
    }

    host_interface.barrier();

    // Read back and verify data
    for (int worker_x = 0; worker_x < grid_size.x; ++worker_x) {
        for (int worker_y = 0; worker_y < grid_size.y; ++worker_y) {
            CoreCoord logical_worker{worker_x, worker_y};
            CoreCoord virtual_worker =
                cluster.get_virtual_coordinate_from_logical_coordinates(device_id, logical_worker, CoreType::WORKER);
            std::vector<uint32_t> read_data(payload_size_bytes / sizeof(uint32_t));

            cluster.read_core(read_data.data(), payload_size_bytes, tt_cxy_pair{device_id, virtual_worker}, l1_base);
            EXPECT_EQ(read_data, test_data_per_worker[logical_worker])
                << fmt::format("Data mismatch for worker {}", logical_worker.str());
        }
    }

    return test_data_per_worker;
}

// Performs write and read-back verification test on a single worker core with two different data sets
// Tests the read functionality by writing known data and reading it back
template <typename HOST_INTERFACE>
void perform_read_test(
    const lite_fabric::SystemDescriptor& desc,
    HOST_INTERFACE& host_interface,
    uint32_t payload_size_bytes,
    uint32_t l1_base,
    uint32_t device_id,
    CoreCoord target_worker = CoreCoord{0, 0}) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& tunnel = desc.tunnels_from_mmio[0];

    log_info(
        tt::LogMetal,
        "Device {} Grid {}",
        device_id,
        cluster.get_soc_desc(device_id).get_grid_size(CoreType::TENSIX).str());

    CoreCoord virtual_worker =
        cluster.get_virtual_coordinate_from_logical_coordinates(device_id, target_worker, CoreType::WORKER);
    const uint64_t dest_noc_upper = (uint64_t(virtual_worker.y) << (36 + 6)) | (uint64_t(virtual_worker.x) << 36);
    uint64_t dest_noc_addr = dest_noc_upper | (uint64_t)l1_base;

    // Create two different test data sets
    auto allOnes = create_random_vector_of_bfloat16(
        payload_size_bytes, 100, std::chrono::system_clock::now().time_since_epoch().count(), 1.0f);
    auto allTwos = create_random_vector_of_bfloat16(
        payload_size_bytes, 100, std::chrono::system_clock::now().time_since_epoch().count(), 2.0f);

    uint64_t onesNocAddr = dest_noc_addr;
    uint64_t twosNocAddr = dest_noc_addr + payload_size_bytes;

    // Write both data sets to different addresses
    host_interface.write_any_len(allOnes.data(), payload_size_bytes, onesNocAddr);
    host_interface.write_any_len(allTwos.data(), payload_size_bytes, twosNocAddr);

    // Barrier to ensure writes complete
    host_interface.barrier();

    log_info(tt::LogMetal, "Reading back data from Device {} worker core {}", device_id, target_worker.str());

    // Read back first data set and verify
    {
        std::vector<uint32_t> read_data(payload_size_bytes / sizeof(uint32_t));
        host_interface.read_any_len(read_data.data(), payload_size_bytes, onesNocAddr);
        log_info(
            tt::LogMetal,
            "Read out data from {} {:#x} {} elements",
            tunnel.mmio_cxy_virtual().str(),
            onesNocAddr,
            read_data.size());

        EXPECT_EQ(read_data, allOnes) << "First data set read verification failed";
    }

    // Read back second data set and verify
    {
        std::vector<uint32_t> read_data(payload_size_bytes / sizeof(uint32_t));
        host_interface.read_any_len(read_data.data(), payload_size_bytes, twosNocAddr);
        log_info(
            tt::LogMetal,
            "Read out data from {} {:#x} {} elements",
            tunnel.mmio_cxy_virtual().str(),
            twosNocAddr,
            read_data.size());

        EXPECT_EQ(read_data, allTwos) << "Second data set read verification failed";
    }
}

// Performs write and read-back verification test with unaligned addresses
// Tests writing to addresses with various alignment offsets to verify unaligned access handling
template <typename HOST_INTERFACE>
void perform_unaligned_write_test(
    const lite_fabric::SystemDescriptor& desc,
    HOST_INTERFACE& host_interface,
    uint32_t payload_size_bytes,
    uint32_t l1_base,
    uint32_t device_id,
    CoreCoord target_worker = CoreCoord{0, 0},
    int max_alignment_offset = 16) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& tunnel = desc.tunnels_from_mmio[0];

    CoreCoord virtual_worker =
        cluster.get_virtual_coordinate_from_logical_coordinates(device_id, target_worker, CoreType::WORKER);

    std::vector<uint32_t> test_data = create_random_vector_of_bfloat16(
        payload_size_bytes, 100, std::chrono::system_clock::now().time_since_epoch().count(), 1.0f);

    // Test various alignment offsets
    for (int aligned_offset = 0; aligned_offset < max_alignment_offset; ++aligned_offset) {
        uint32_t addr = l1_base + aligned_offset;
        uint64_t dest_noc_upper = (uint64_t(virtual_worker.y) << (36 + 6)) | (uint64_t(virtual_worker.x) << 36);
        uint64_t dest_noc_addr = dest_noc_upper | (uint64_t)addr;

        log_info(tt::LogMetal, "Testing unaligned write to {:#x} (offset {})", addr, aligned_offset);

        // Write data to unaligned address
        host_interface.write_any_len(test_data.data(), test_data.size() * sizeof(uint32_t), dest_noc_addr);
        host_interface.barrier();

        // Read back and verify
        std::vector<uint32_t> read_data(test_data.size());
        cluster.read_core(
            read_data.data(), test_data.size() * sizeof(uint32_t), tt_cxy_pair{device_id, virtual_worker}, addr);

        EXPECT_EQ(read_data, test_data) << fmt::format(
            "Unaligned write/read failed at offset {} (addr {:#x})", aligned_offset, addr);
    }
}

// Performs unaligned read test by writing data with standard API and reading with unaligned offset
// Tests reading from unaligned addresses using byte-level comparison
template <typename HOST_INTERFACE>
void perform_unaligned_read_test(
    const lite_fabric::SystemDescriptor& desc,
    HOST_INTERFACE& host_interface,
    uint32_t write_data_size_bytes,
    uint32_t l1_base,
    uint32_t device_id,
    CoreCoord target_worker = CoreCoord{0, 0},
    size_t read_size_bytes = 64 * sizeof(uint32_t),
    size_t unaligned_offset_bytes = 7) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& tunnel = desc.tunnels_from_mmio[0];

    CoreCoord virtual_worker =
        cluster.get_virtual_coordinate_from_logical_coordinates(device_id, target_worker, CoreType::WORKER);
    uint64_t dest_noc_upper = (uint64_t(virtual_worker.y) << (36 + 6)) | (uint64_t(virtual_worker.x) << 36);
    uint64_t dest_noc_addr = dest_noc_upper | (uint64_t)l1_base;

    // Write data using known good API (standard cluster write_core)
    std::vector<uint32_t> write_data = create_random_vector_of_bfloat16(
        write_data_size_bytes, 100, std::chrono::system_clock::now().time_since_epoch().count(), 1.0f);

    cluster.write_core(
        write_data.data(), write_data.size() * sizeof(uint32_t), tt_cxy_pair{device_id, virtual_worker}, l1_base);
    cluster.l1_barrier(device_id);

    log_info(
        tt::LogMetal,
        "Testing unaligned read from offset {} bytes (addr {:#x})",
        unaligned_offset_bytes,
        l1_base + unaligned_offset_bytes);

    // Read a slice from an unaligned address using host interface
    std::vector<uint8_t> read_bytes(read_size_bytes);
    host_interface.read(read_bytes.data(), read_bytes.size(), dest_noc_addr + unaligned_offset_bytes);

    // Build expected bytes by taking a byte view of the original write buffer and slicing by the same offset
    const uint8_t* write_bytes = reinterpret_cast<const uint8_t*>(write_data.data());
    std::vector<uint8_t> expected_bytes(read_bytes.size());
    std::memcpy(expected_bytes.data(), write_bytes + unaligned_offset_bytes, read_bytes.size());

    EXPECT_EQ(read_bytes, expected_bytes) << fmt::format(
        "Unaligned read failed at offset {} bytes (addr {:#x})",
        unaligned_offset_bytes,
        l1_base + unaligned_offset_bytes);
}

// Performs small write test by writing 1 byte at a time to consecutive addresses
// Tests small write operations and byte-level read-back verification with masking
template <typename HOST_INTERFACE>
void perform_small_write_test(
    const lite_fabric::SystemDescriptor& desc,
    HOST_INTERFACE& host_interface,
    uint32_t l1_base,
    uint32_t device_id,
    CoreCoord target_worker = CoreCoord{0, 0},
    uint32_t num_writes = 64,
    uint32_t start_index = 1) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& tunnel = desc.tunnels_from_mmio[0];

    CoreCoord virtual_worker =
        cluster.get_virtual_coordinate_from_logical_coordinates(device_id, target_worker, CoreType::WORKER);
    uint64_t dest_noc_upper = (uint64_t(virtual_worker.y) << (36 + 6)) | (uint64_t(virtual_worker.x) << 36);
    uint64_t dest_noc_addr = dest_noc_upper | (uint64_t)l1_base;

    log_info(tt::LogMetal, "Testing {} small (1-byte) writes starting from index {}", num_writes, start_index);

    // Create test data vector
    std::vector<uint32_t> write_data = create_random_vector_of_bfloat16(
        num_writes * sizeof(uint32_t), 100, std::chrono::system_clock::now().time_since_epoch().count(), 1.0f);

    // Write 1 byte at a time to consecutive addresses
    for (int i = start_index; i < num_writes; ++i) {
        host_interface.write_any_len(&write_data[i], 1, dest_noc_addr + i);
        // Due to unaligned 1-byte write, mask off the irrelevant bits for comparison
        write_data[i] = write_data[i] & 0xff;
    }
    host_interface.barrier();

    // Read back and verify each 1-byte write
    for (int i = start_index; i < num_writes; ++i) {
        std::vector<uint32_t> read_data(1);
        cluster.read_core(read_data.data(), 1, tt_cxy_pair{device_id, virtual_worker}, l1_base + i);

        log_debug(
            tt::LogMetal,
            "Read data {} from {:#x} got {:#x} expecting {:#x}",
            i,
            l1_base + i,
            read_data[0],
            write_data[i]);

        EXPECT_EQ(read_data[0], write_data[i]) << fmt::format(
            "Small write verification failed at index {} (addr {:#x}): got {:#x}, expected {:#x}",
            i,
            l1_base + i,
            read_data[0],
            write_data[i]);
    }

    log_info(tt::LogMetal, "Successfully verified {} small writes", num_writes - start_index);
}

void perform_basic_combo_test(
    const lite_fabric::SystemDescriptor& desc,
    lite_fabric::HostToFabricLiteInterface<lite_fabric::SENDER_NUM_BUFFERS_ARRAY[0], lite_fabric::CHANNEL_BUFFER_SIZE>&
        host_interface) {
    const auto& tunnel = desc.tunnels_from_mmio[0];
    log_info(tt::LogTest, "Tunnel: {} -> {}", tunnel.mmio_cxy_virtual().str(), tunnel.connected_cxy_virtual().str());

    // This will wrap the sender channel multiple times and write to all worker cores
    uint32_t payload_size_bytes = 4096;
    uint32_t l1_base = 0x10000;
    uint32_t num_writes = 64;
    uint32_t start_index = 1;
    CoreCoord target_worker{0, 0};
    size_t read_size_bytes = 64 * sizeof(uint32_t);  // 256-byte slice
    size_t unaligned_offset_bytes = 7;
    int max_alignment_offset = 16;

    int remote_device_id = desc.tunnels_from_mmio[0].connected_id;

    perform_write_read_test(desc, host_interface, payload_size_bytes, l1_base, remote_device_id);

    perform_unaligned_read_test(
        desc,
        host_interface,
        payload_size_bytes,
        l1_base,
        remote_device_id,
        target_worker,
        read_size_bytes,
        unaligned_offset_bytes);

    host_interface.barrier();

    perform_read_test(desc, host_interface, payload_size_bytes, l1_base, remote_device_id, target_worker);

    perform_small_write_test(desc, host_interface, l1_base, remote_device_id, target_worker, num_writes, start_index);

    perform_unaligned_write_test(
        desc, host_interface, payload_size_bytes, l1_base, remote_device_id, target_worker, max_alignment_offset);
}

}  // anonymous namespace

struct FabricLiteTestConfig {
    bool standalone{false};
    tt::tt_fabric::FabricConfig fabric_config{tt::tt_fabric::FabricConfig::DISABLED};
};

// Lite Fabric Test Fixture
class FabricLite : public testing::TestWithParam<FabricLiteTestConfig> {
protected:
    inline static lite_fabric::SystemDescriptor desc;
    inline static lite_fabric::
        HostToFabricLiteInterface<lite_fabric::SENDER_NUM_BUFFERS_ARRAY[0], lite_fabric::CHANNEL_BUFFER_SIZE>
            host_interface;

    // Instance variables instead of static ones for parameter-dependent resources
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device_;
    bool fabric_configured_{false};

    static void SetUpTestSuite() {
        auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        auto& hal = tt::tt_metal::MetalContext::instance().hal();
        desc = lite_fabric::GetSystemDescriptor2Devices(cluster, 0, 1);

        lite_fabric::LaunchLiteFabric(cluster, hal, desc);

        host_interface =
            lite_fabric::FabricLiteMemoryMap::make_host_interface(desc.tunnels_from_mmio[0].mmio_cxy_virtual());
    }

    static void TearDownTestSuite() {
        auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        auto& hal = tt::tt_metal::MetalContext::instance().hal();

        lite_fabric::TerminateLiteFabric(cluster, desc);
    }

    void SetUp() override {
        CHECK_TEST_REQS();

        if (desc.tunnels_from_mmio.empty()) {
            GTEST_SKIP() << "No tunnels found";
        }

        // Configure fabric if needed
        if (GetParam().fabric_config != tt::tt_fabric::FabricConfig::DISABLED) {
            tt::tt_fabric::SetFabricConfig(GetParam().fabric_config);
            fabric_configured_ = true;
        }

        // Create mesh device if not standalone
        if (!GetParam().standalone) {
            if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE")) {
                GTEST_SKIP() << "Fast dispatch is required for this test (remove TT_METAL_SLOW_DISPATCH_MODE)";
            }

            auto number_of_devices = tt::tt_metal::GetNumAvailableDevices();
            mesh_device_ = tt::tt_metal::distributed::MeshDevice::create(tt::tt_metal::distributed::MeshDeviceConfig(
                tt::tt_metal::distributed::MeshShape{number_of_devices, 1}));
        }
    }

    void TearDown() override {
        // Clean up mesh device
        if (mesh_device_) {
            mesh_device_.reset();
        }

        // Reset fabric configuration
        if (fabric_configured_) {
            tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
            fabric_configured_ = false;
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    FabricLiteFixture,
    FabricLite,
    ::testing::Values(
        // Standalone tests (no mesh device, no fabric)
        FabricLiteTestConfig{.standalone = true},
        // Standard tests with mesh device but no fabric
        FabricLiteTestConfig{.standalone = false}
        // Test with 1D fabric active (full fabric)
        // FabricLiteTestConfig{.standalone = false, .fabric_config = tt::tt_fabric::FabricConfig::FABRIC_1D}
        // Test with 2D fabric active (full fabric)
        // FabricLiteTestConfig{.standalone = false, .fabric_config = tt::tt_fabric::FabricConfig::FABRIC_2D}
        ),
    [](const testing::TestParamInfo<FabricLiteTestConfig>& info) {
        std::string name;
        if (info.param.standalone) {
            name = "Standalone";
        } else if (info.param.fabric_config == tt::tt_fabric::FabricConfig::DISABLED) {
            name = "MeshDevice";
        } else {
            name = "MeshDevice_";
            name += enchantum::to_string(info.param.fabric_config);
        }
        return name;
    });

TEST(FabricLiteBuild, BuildOnly) {
    auto home_directory = std::filesystem::path(std::getenv("TT_METAL_HOME"));
    auto output_directory = home_directory / "lite_fabric";
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    if (lite_fabric::CompileFabricLite(cluster, home_directory, output_directory)) {
        throw std::runtime_error("Failed to compile");
    }
    if (lite_fabric::LinkFabricLite(home_directory, output_directory, output_directory / "lite_fabric.elf")) {
        throw std::runtime_error("Failed to link");
    }
}

TEST_P(FabricLite, Init) { EXPECT_GT(desc.tunnels_from_mmio.size(), 0) << "No tunnels found"; }

TEST_P(FabricLite, Writes) {
    const auto& tunnel = desc.tunnels_from_mmio[0];
    log_info(tt::LogTest, "Tunnel: {} -> {}", tunnel.mmio_cxy_virtual().str(), tunnel.connected_cxy_virtual().str());

    // This will wrap the sender channel multiple times and write to all worker cores
    uint32_t payload_size_bytes = 4096;  // (128 * 1024) + 512;
    uint32_t l1_base = 0x10000;
    auto remote_device_id = desc.tunnels_from_mmio[0].connected_id;

    // Use the extracted function to perform the write/read test
    perform_write_read_test(desc, host_interface, payload_size_bytes, l1_base, remote_device_id);
}

TEST_P(FabricLite, Reads) {
    const auto& tunnel = desc.tunnels_from_mmio[0];
    log_info(tt::LogTest, "Tunnel: {} -> {}", tunnel.mmio_cxy_virtual().str(), tunnel.connected_cxy_virtual().str());

    uint32_t payload_size_bytes = 4 * 1024;
    uint32_t l1_base = 0x10000;
    CoreCoord target_worker{0, 0};
    auto remote_device_id = desc.tunnels_from_mmio[0].connected_id;

    // Use the extracted function to perform the read test
    perform_read_test(desc, host_interface, payload_size_bytes, l1_base, remote_device_id, target_worker);
}

TEST_P(FabricLite, Barrier) {
    const auto& tunnel = desc.tunnels_from_mmio[0];

    host_interface.barrier();
}

TEST_P(FabricLite, WritesSmall) {
    uint32_t l1_base = 0x10000;
    CoreCoord target_worker{0, 0};
    uint32_t num_writes = 64;
    uint32_t start_index = 1;  // Skip index 0 to avoid overwriting each other
    auto remote_device_id = desc.tunnels_from_mmio[0].connected_id;

    perform_small_write_test(desc, host_interface, l1_base, remote_device_id, target_worker, num_writes, start_index);
}

TEST_P(FabricLite, WritesUnaligned) {
    uint32_t payload_size_bytes = 512;
    uint32_t l1_base = 0x10000;
    CoreCoord target_worker{0, 0};
    int max_alignment_offset = 16;

    perform_unaligned_write_test(
        desc, host_interface, payload_size_bytes, l1_base, 1, target_worker, max_alignment_offset);
}

TEST_P(FabricLite, ReadsUnaligned) {
    uint32_t write_data_size_bytes = 4096;
    uint32_t l1_base = 0x10000;
    CoreCoord target_worker{0, 0};
    size_t read_size_bytes = 64 * sizeof(uint32_t);  // 256-byte slice
    size_t unaligned_offset_bytes = 7;
    auto remote_device_id = desc.tunnels_from_mmio[0].connected_id;

    perform_unaligned_read_test(
        desc,
        host_interface,
        write_data_size_bytes,
        l1_base,
        remote_device_id,
        target_worker,
        read_size_bytes,
        unaligned_offset_bytes);
}

TEST_P(FabricLite, FunctionPointerTable) {
    const auto& tunnel = desc.tunnels_from_mmio[0];

    auto service_func_offset = lite_fabric::FabricLiteMemoryMap::get_service_channel_func_addr();
    // This value can be read from the MMIO device. It's the same across all devices
    uint32_t service_func = 0;
    tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        (void*)&service_func, sizeof(uint32_t), tunnel.mmio_cxy_virtual(), service_func_offset);
    // Service function is expected to be somewhere in L1 but Local
    ASSERT_TRUE(service_func > 0x60000 && service_func < 0x70000) << "Expected service function to be in L1 range";
    // First instruction should be a stack allocation
}

TEST_P(FabricLite, BasicFunctions) {
    if (GetParam().standalone) {
        GTEST_SKIP() << "BasicFunctions test requires mesh device (not standalone)";
    }
    perform_basic_combo_test(desc, host_interface);
}

TEST_P(FabricLite, ActiveEthKernelDevice0) {
    if (!mesh_device_) {
        GTEST_SKIP() << "Mesh device required for this test";
    }
    if (GetParam().fabric_config != tt::tt_fabric::FabricConfig::DISABLED) {
        GTEST_SKIP() << "ActiveEthKernelDevice0 test requires fabric to be disabled";
    }
    // Launch an active eth kernel on device 0 which calls the servicing routine and ensure we can still send/recv
    // through lite fabric
    const std::string kernel_path = "tests/tt_metal/tt_metal/tunneling/test_kernels/service_channels.cpp";
    auto program = tt::tt_metal::CreateProgram();

    uint32_t run_signal_addr = mesh_device_->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    uint32_t run_signal = 1;

    const auto mmio_eth_core = desc.tunnels_from_mmio[0].mmio_core_logical;
    const auto mmio_eth_core_virtual = desc.tunnels_from_mmio[0].mmio_cxy_virtual();

    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        &run_signal, sizeof(uint32_t), mmio_eth_core_virtual, run_signal_addr);

    tt::tt_metal::CreateKernel(
        program,
        kernel_path,
        mmio_eth_core,
        tt::tt_metal::EthernetConfig{
            .eth_mode = tt::tt_metal::SENDER,
            .processor = static_cast<tt::tt_metal::DataMovementProcessor>(0),
            .compile_args = {run_signal_addr},
        });

    tt::tt_metal::distributed::MeshWorkload workload;
    auto zero_coord = tt::tt_metal::distributed::MeshCoordinate::zero_coordinate(mesh_device_->shape().dims());
    auto device_range = tt::tt_metal::distributed::MeshCoordinateRange(zero_coord, zero_coord);
    tt::tt_metal::distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);

    log_info(
        tt::LogTest,
        "========== Enqueue Active Eth kernel on device 0 {} (virtual={}) ==========",
        mmio_eth_core.str(),
        mmio_eth_core_virtual.str());

    tt::tt_metal::distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload, false);

    // Do lite fabric actions while the kernel is running
    // This will hang if the kernel doesn't call service_lite_fabric_channels()
    log_info(tt::LogTest, "========== Kernel running. Performing basic combo test while kernel is running ==========");
    perform_basic_combo_test(desc, host_interface);

    run_signal = 0;
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        &run_signal, sizeof(uint32_t), mmio_eth_core_virtual, run_signal_addr);
    Finish(mesh_device_->mesh_command_queue());

    // Try again now we are in metal firmware
    log_info(tt::LogTest, "========== Kernel done. Performing basic combo test while in metal firmware ==========");
    perform_basic_combo_test(desc, host_interface);
}
