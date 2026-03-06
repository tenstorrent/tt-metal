// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>
#include <numeric>
#include <cstring>

// Prefer to use API rather than internals in this
// test as we are testing end to end functionality
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/context/metalium_env.hpp>
#include <tt-metalium/experimental/hal.hpp>
#include <tt-metalium/experimental/tt_metal.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>

#include <umd/device/types/arch.hpp>
#include "device/mock_device_util.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {

namespace {

constexpr int kExitBadContextId = 1;
constexpr int kExitBadNumDevices = 2;
constexpr int kExitBadMeshSize = 3;
constexpr int kExitWorkFailed = 5;
constexpr int kExitBufferVerificationFailed = 10;

// Helper function to perform buffer operations and kernel launch on a MeshDevice.
// Uses a sharded MeshBuffer so this works on both single- and multi-device meshes.
void PerformDeviceWork(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t data_pattern,
    const std::string& process_name,
    const std::string& kernel_identifier) {
    constexpr uint32_t kElementsPerShard = 1024;
    constexpr uint32_t kShardSize = kElementsPerShard * sizeof(uint32_t);

    const size_t num_rows = mesh_device->num_rows();
    const size_t num_cols = mesh_device->num_cols();
    const size_t num_devices = mesh_device->num_devices();

    distributed::ShardedBufferConfig buffer_config{
        .global_size = kShardSize * num_devices,
        .global_buffer_shape = {kElementsPerShard * num_rows, num_cols},
        .shard_shape = {kElementsPerShard, 1},
        .shard_orientation = ShardOrientation::ROW_MAJOR,
    };
    distributed::DeviceLocalBufferConfig local_config{
        .page_size = kShardSize,
        .buffer_type = BufferType::DRAM,
    };
    auto mesh_buffer = distributed::MeshBuffer::create(buffer_config, local_config, mesh_device.get());

    std::vector<uint32_t> write_data(kElementsPerShard * num_devices);
    std::iota(write_data.begin(), write_data.end(), data_pattern);

    auto& mesh_cq = mesh_device->mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(mesh_cq, mesh_buffer, write_data, false);

    std::vector<uint32_t> read_data;
    distributed::EnqueueReadMeshBuffer(mesh_cq, read_data, mesh_buffer, true);

    if (read_data != write_data) {
        throw std::runtime_error("Buffer read/write verification failed");
    }

    auto program = CreateProgram();
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    auto core_grid = mesh_device->compute_with_storage_grid_size();
    auto core_range = CoreRange({0, 0}, {core_grid.x - 1, core_grid.y - 1});

    std::string kernel_src = "void kernel_main() {\n    // " + kernel_identifier + "\n}";

    CreateKernelFromString(
        program,
        kernel_src,
        core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    distributed::MeshWorkload workload;
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_cq, workload, true);

    log_info(tt::LogTest, "{}: Successfully completed buffer ops and kernel launch", process_name);
}

[[noreturn]] void RunChildWithVisibleDevices(
    const std::string& visible_devices,
    size_t expected_num_chips,
    uint32_t data_pattern,
    const std::string& child_name) {
    setenv("TT_VISIBLE_DEVICES", visible_devices.c_str(), 1);

    auto child_env = std::make_shared<MetaliumEnv>();
    int child_context_id = tt::tt_metal::experimental::CreateContext(child_env);
    if (child_context_id != SILICON_CONTEXT_ID) {
        _exit(kExitBadContextId);
    }

    auto child_num_devices = tt::tt_metal::experimental::GetNumAvailableDevices(child_env);
    log_info(tt::LogTest, "{}: TT_VISIBLE_DEVICES={}, num_devices={}", child_name, visible_devices, child_num_devices);
    if (child_num_devices != expected_num_chips) {
        _exit(kExitBadNumDevices);
    }

    auto child_mesh_shape = MetalContext::instance(child_context_id).get_system_mesh().shape();
    auto child_mesh_device =
        distributed::MeshDevice::create(child_context_id, distributed::MeshDeviceConfig(child_mesh_shape));
    if (child_mesh_device->num_devices() != expected_num_chips) {
        _exit(kExitBadMeshSize);
    }
    log_info(tt::LogTest, "{}: opened MeshDevice with {} device(s)", child_name, child_mesh_device->num_devices());

    try {
        PerformDeviceWork(child_mesh_device, data_pattern, child_name, child_name + " kernel");
    } catch (const std::exception& e) {
        log_error(tt::LogTest, "{}: Work failed: {}", child_name, e.what());
        if (std::string(e.what()).find("Buffer read/write verification failed") != std::string::npos) {
            _exit(kExitBufferVerificationFailed);
        }
        _exit(kExitWorkFailed);
    }

    _exit(0);
}

}  // namespace

TEST(MetalContextIntegrationTest, HelloWorld) {
    auto mesh_shape = tt_metal::distributed::SystemMesh::instance().shape();
    auto mesh_device_config = distributed::MeshDeviceConfig(mesh_shape);
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create(mesh_device_config);
    EXPECT_EQ(mesh_device->num_devices(), mesh_shape.mesh_size());
    // Required for this unit test because legacy behaviour of MetalContext is not to close the cluster until atexit
    // Close it right now so remaining tests can proceed
    mesh_device->close();
    tt::tt_metal::experimental::DestroyAllContexts();

    // It was found that during ~MeshDevice, some calls to MetalContext::instance() were made which caused
    // MetalContext to implicitly reinitialize thus undoing the effects of DestroyAllContexts().
    // Subsequent tests will hang if that happens.
}

TEST(MetalContextIntegrationTest, HelloWorldExplicit) {
    auto env = std::make_shared<MetaliumEnv>();
    int context_id = tt::tt_metal::experimental::CreateContext(env);

    auto mesh_shape = tt_metal::distributed::SystemMesh::instance(context_id).shape();
    auto mesh_device_config = distributed::MeshDeviceConfig(mesh_shape);
    auto mesh_device = distributed::MeshDevice::create(context_id, mesh_device_config);

    mesh_device->close();
    tt::tt_metal::experimental::DestroyAllContexts();
}

TEST(MetalContextIntegrationTest, HelloWorldQueryThenCreate) {
    auto env = std::make_shared<MetaliumEnv>();
    int context_id = tt::tt_metal::experimental::CreateContext(env);

    size_t l1_size = tt::tt_metal::experimental::hal::get_l1_size(*env);
    size_t trace_region_size = l1_size * 0.3;
    size_t l1_small_region_size = l1_size * 0.05;

    auto mesh_shape = tt_metal::distributed::SystemMesh::instance(context_id).shape();
    auto mesh_device_config = distributed::MeshDeviceConfig(mesh_shape);
    auto mesh_device =
        distributed::MeshDevice::create(context_id, mesh_device_config, trace_region_size, l1_small_region_size);

    mesh_device->close();
    tt::tt_metal::experimental::DestroyAllContexts();
}

TEST(MetalContextIntegrationTest, HalFunctions) {
    // Create a MetaliumEnv and query hardware state
    MetaliumEnv env;
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_arch(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_arch_name(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_l1_size(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_dram_alignment(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_l1_alignment(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_pcie_alignment(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_erisc_l1_unreserved_base(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_erisc_l1_unreserved_size(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_max_worker_l1_unreserved_size(env));
}

TEST(MetalContextIntegrationTest, HalFunctionsWithMock) {
    auto env_settings = MetaliumEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2));
    MetaliumEnv env(env_settings);
    EXPECT_EQ(tt::tt_metal::experimental::hal::get_arch(env), tt::ARCH::BLACKHOLE);
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_l1_size(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_dram_alignment(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_l1_alignment(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_pcie_alignment(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_erisc_l1_unreserved_base(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_erisc_l1_unreserved_size(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_max_worker_l1_unreserved_size(env));
}

TEST(MetalContextIntegrationTest, MockDevice) {
    {
        auto mock_env_bh_1 = std::make_shared<MetaliumEnv>(
            MetaliumEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2)));
        auto mock_context_id = tt::tt_metal::experimental::CreateContext(mock_env_bh_1);
        EXPECT_NE(mock_context_id, SILICON_CONTEXT_ID);

        auto mesh_config_mock = distributed::MeshDeviceConfig(distributed::MeshShape(2));
        auto mock_device = distributed::MeshDevice::create(mock_context_id, mesh_config_mock);

        auto env = std::make_shared<MetaliumEnv>();
        auto context_id = tt::tt_metal::experimental::CreateContext(env);
        auto mesh_shape = tt::tt_metal::MetalContext::instance(context_id).get_system_mesh().shape();
        auto mesh_config_silicon = distributed::MeshDeviceConfig(mesh_shape);
        auto silicon_device = distributed::MeshDevice::create(context_id, mesh_config_silicon);
    }

    tt::tt_metal::experimental::DestroyAllContexts();
}

TEST(MetalContextIntegrationTest, CoexistingSiliconAndMockDevice) {
    {
        // Create mock mesh device with 1 blackhole chip
        auto mock_env_bh_1 = std::make_shared<MetaliumEnv>(
            MetaliumEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 1)));
        auto mock_context_id_bh_1 = tt::tt_metal::MetalContext::create_instance(mock_env_bh_1);
        log_info(tt::LogTest, "MetaliumEnv (mock) created with context id {}", mock_context_id_bh_1);
        EXPECT_NE(mock_context_id_bh_1, SILICON_CONTEXT_ID);

        auto mock_mesh_shape_bh_1 =
            tt::tt_metal::MetalContext::instance(mock_context_id_bh_1).get_system_mesh().shape();
        auto mock_mesh_device_config_bh_1 = distributed::MeshDeviceConfig(mock_mesh_shape_bh_1);
        std::shared_ptr<distributed::MeshDevice> mock_mesh_device_bh_1 =
            distributed::MeshDevice::create(mock_context_id_bh_1, mock_mesh_device_config_bh_1);
        log_info(tt::LogTest, "Created mock mesh device with shape {}", mock_mesh_device_bh_1->shape().dims());

        // Create mock mesh device with 2 blackhole chips
        auto mock_env_bh_2 = std::make_shared<MetaliumEnv>(
            MetaliumEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2)));
        auto mock_context_id_bh_2 = tt::tt_metal::MetalContext::create_instance(mock_env_bh_2);
        log_info(tt::LogTest, "MetaliumEnv (mock) created with context id {}", mock_context_id_bh_2);
        EXPECT_NE(mock_context_id_bh_2, SILICON_CONTEXT_ID);

        auto mock_mesh_shape_bh_2 =
            tt::tt_metal::MetalContext::instance(mock_context_id_bh_2).get_system_mesh().shape();
        auto mock_mesh_device_config_bh_2 = distributed::MeshDeviceConfig(mock_mesh_shape_bh_2);
        std::shared_ptr<distributed::MeshDevice> mock_mesh_device_bh_2 =
            distributed::MeshDevice::create(mock_context_id_bh_2, mock_mesh_device_config_bh_2);
        log_info(tt::LogTest, "Created mock mesh device with shape {}", mock_mesh_device_bh_2->shape().dims());

        // Create silicon mesh
        auto silicon_env = std::make_shared<MetaliumEnv>();
        auto silicon_context_id = tt::tt_metal::MetalContext::create_instance(silicon_env);
        log_info(tt::LogTest, "MetaliumEnv (silicon) created with context id {}", silicon_context_id);
        EXPECT_EQ(silicon_context_id, SILICON_CONTEXT_ID);

        auto mesh_shape = tt::tt_metal::MetalContext::instance(silicon_context_id).get_system_mesh().shape();
        auto mesh_device_config = distributed::MeshDeviceConfig(mesh_shape);
        std::shared_ptr<distributed::MeshDevice> mesh_device =
            distributed::MeshDevice::create(silicon_context_id, mesh_device_config);
        log_info(tt::LogTest, "Created silicon mesh device with shape {}", mesh_device->shape().dims());

        EXPECT_NE(mock_context_id_bh_1, mock_context_id_bh_2);
        ASSERT_EQ(mock_mesh_device_bh_1->get_devices().size(), 1);
        ASSERT_EQ(mock_mesh_device_bh_2->get_devices().size(), 2);
    }

    tt::tt_metal::experimental::DestroyAllContexts();
}

// Same test as above but reverse the order to ensure no hangs due to unexpected internal objects created for the
// incorrect context id
TEST(MetalContextIntegrationTest, CoexistingMockAndSiliconDevice) {
    {
        // Create silicon mesh
        auto silicon_env = std::make_shared<MetaliumEnv>();
        auto silicon_context_id = tt::tt_metal::MetalContext::create_instance(silicon_env);
        log_info(tt::LogTest, "MetaliumEnv (silicon) created with context id {}", silicon_context_id);
        EXPECT_EQ(silicon_context_id, SILICON_CONTEXT_ID);

        auto mesh_shape = tt::tt_metal::MetalContext::instance(silicon_context_id).get_system_mesh().shape();
        auto mesh_device_config = distributed::MeshDeviceConfig(mesh_shape);
        std::shared_ptr<distributed::MeshDevice> mesh_device =
            distributed::MeshDevice::create(silicon_context_id, mesh_device_config);
        log_info(tt::LogTest, "Created silicon mesh device with shape {}", mesh_device->shape().dims());

        // Create mock mesh device with 1 blackhole chip
        auto mock_env_bh_1 = std::make_shared<MetaliumEnv>(
            MetaliumEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 1)));
        auto mock_context_id_bh_1 = tt::tt_metal::MetalContext::create_instance(mock_env_bh_1);
        log_info(tt::LogTest, "MetaliumEnv (mock) created with context id {}", mock_context_id_bh_1);
        EXPECT_NE(mock_context_id_bh_1, SILICON_CONTEXT_ID);

        auto mock_mesh_shape_bh_1 =
            tt::tt_metal::MetalContext::instance(mock_context_id_bh_1).get_system_mesh().shape();
        auto mock_mesh_device_config_bh_1 = distributed::MeshDeviceConfig(mock_mesh_shape_bh_1);
        std::shared_ptr<distributed::MeshDevice> mock_mesh_device_bh_1 =
            distributed::MeshDevice::create(mock_context_id_bh_1, mock_mesh_device_config_bh_1);
        log_info(tt::LogTest, "Created mock mesh device with shape {}", mock_mesh_device_bh_1->shape().dims());

        // Create mock mesh device with 2 blackhole chips
        auto mock_env_bh_2 = std::make_shared<MetaliumEnv>(
            MetaliumEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2)));
        auto mock_context_id_bh_2 = tt::tt_metal::MetalContext::create_instance(mock_env_bh_2);
        log_info(tt::LogTest, "MetaliumEnv (mock) created with context id {}", mock_context_id_bh_2);
        EXPECT_NE(mock_context_id_bh_2, SILICON_CONTEXT_ID);

        auto mock_mesh_shape_bh_2 =
            tt::tt_metal::MetalContext::instance(mock_context_id_bh_2).get_system_mesh().shape();
        auto mock_mesh_device_config_bh_2 = distributed::MeshDeviceConfig(mock_mesh_shape_bh_2);
        std::shared_ptr<distributed::MeshDevice> mock_mesh_device_bh_2 =
            distributed::MeshDevice::create(mock_context_id_bh_2, mock_mesh_device_config_bh_2);
        log_info(tt::LogTest, "Created mock mesh device with shape {}", mock_mesh_device_bh_2->shape().dims());

        EXPECT_NE(mock_context_id_bh_1, mock_context_id_bh_2);
        ASSERT_EQ(mock_mesh_device_bh_1->get_devices().size(), 1);
        ASSERT_EQ(mock_mesh_device_bh_2->get_devices().size(), 2);
    }

    tt::tt_metal::experimental::DestroyAllContexts();
}

TEST(MetalContextIntegrationTest, ForkMockAndRealDevice) {
    // Query hardware state before forking
    {
        auto env = std::make_shared<MetaliumEnv>();
        auto silicon_context_id = tt::tt_metal::experimental::CreateContext(env);
        EXPECT_EQ(silicon_context_id, SILICON_CONTEXT_ID);

        auto arch = tt::tt_metal::experimental::hal::get_arch(*env);
        auto l1_size = tt::tt_metal::experimental::hal::get_l1_size(*env);
        auto num_devices = tt::tt_metal::experimental::GetNumAvailableDevices(env);
        log_info(tt::LogTest, "Pre-fork: arch={}, L1 size={}, num_devices={}", arch, l1_size, num_devices);
        EXPECT_GT(l1_size, 0u);
        EXPECT_GT(num_devices, 0u);
    }

    // Tear down all state so we can safely fork
    tt::tt_metal::experimental::DestroyAllContexts();

    int pipe_fd[2];
    ASSERT_EQ(pipe(pipe_fd), 0) << "pipe() failed";

    pid_t pid = fork();
    if (pid == -1) {
        close(pipe_fd[0]);
        close(pipe_fd[1]);
        FAIL() << "Failed to fork";
    }

    if (pid == 0) {
        close(pipe_fd[1]);

        auto mock_env = std::make_shared<MetaliumEnv>(
            MetaliumEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2).value()));
        int mock_context_id = MetalContext::create_instance(mock_env);

        if (mock_context_id < 1 || !MetalContext::instance(mock_context_id).rtoptions().get_mock_enabled()) {
            _exit(1);
        }

        if (tt::tt_metal::experimental::hal::get_arch(*mock_env) != tt::ARCH::BLACKHOLE) {
            _exit(2);
        }

        auto mock_mesh_shape = MetalContext::instance(mock_context_id).get_system_mesh().shape();
        auto mock_mesh_device =
            distributed::MeshDevice::create(mock_context_id, distributed::MeshDeviceConfig(mock_mesh_shape));
        if (mock_mesh_device->get_devices().size() != 2) {
            _exit(3);
        }

        char byte = 0;
        if (read(pipe_fd[0], &byte, 1) != 1) {
            _exit(4);
        }
        close(pipe_fd[0]);

        _exit(0);
    }

    // Parent: real device work
    close(pipe_fd[0]);

    auto silicon_env = std::make_shared<MetaliumEnv>();
    auto silicon_context_id = tt::tt_metal::experimental::CreateContext(silicon_env);
    ASSERT_EQ(silicon_context_id, SILICON_CONTEXT_ID);

    auto real_arch = tt::tt_metal::experimental::hal::get_arch(*silicon_env);
    auto real_l1_size = tt::tt_metal::experimental::hal::get_l1_size(*silicon_env);
    log_info(tt::LogTest, "Parent (real): arch={}, L1 size={}", real_arch, real_l1_size);
    EXPECT_GT(real_l1_size, 0u);

    auto mesh_shape = MetalContext::instance(silicon_context_id).get_system_mesh().shape();
    auto silicon_mesh_device =
        distributed::MeshDevice::create(silicon_context_id, distributed::MeshDeviceConfig(mesh_shape));
    EXPECT_GT(silicon_mesh_device->num_devices(), 0u);
    log_info(tt::LogTest, "Parent: created silicon mesh device with shape {}", silicon_mesh_device->shape().dims());

    // Signal child to exit
    char byte = 1;
    ASSERT_EQ(write(pipe_fd[1], &byte, 1), 1) << "write(pipe) failed";
    close(pipe_fd[1]);

    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);

    silicon_mesh_device.reset();
    tt::tt_metal::experimental::DestroyAllContexts();
}

TEST(MetalContextIntegrationTest, ForkWithDisjointDevices) {
    size_t num_mmio_devices = 0;
    size_t num_available_devices = 0;
    {
        auto env = std::make_shared<MetaliumEnv>();
        auto context_id = tt::tt_metal::experimental::CreateContext(env);
        EXPECT_EQ(context_id, SILICON_CONTEXT_ID);
        num_mmio_devices = tt::tt_metal::experimental::GetNumPCIeDevices(env);
        num_available_devices = tt::tt_metal::experimental::GetNumAvailableDevices(env);
        log_info(tt::LogTest, "System has {} PCIe devices, {} total chips", num_mmio_devices, num_available_devices);
    }
    tt::tt_metal::experimental::DestroyAllContexts();

    if (num_mmio_devices < 2) {
        GTEST_SKIP() << "ForkWithDisjointDevices requires at least 2 MMIO devices, found " << num_mmio_devices;
    }

    // Split PCIe devices into two halves
    size_t half = num_mmio_devices / 2;
    std::string first_half_devices;
    std::string second_half_devices;
    for (size_t i = 0; i < num_mmio_devices; i++) {
        auto& target = (i < half) ? first_half_devices : second_half_devices;
        if (!target.empty()) {
            target += ",";
        }
        target += std::to_string(i);
    }
    size_t chips_per_pcie_device = num_available_devices / num_mmio_devices;
    size_t expected_chips_first_half = half * chips_per_pcie_device;
    size_t expected_chips_second_half = (num_mmio_devices - half) * chips_per_pcie_device;
    log_info(
        tt::LogTest,
        "Splitting: child 1 gets PCIe devices [{}] ({} chips), child 2 gets [{}] ({} chips)",
        first_half_devices,
        expected_chips_first_half,
        second_half_devices,
        expected_chips_second_half);

    pid_t pid1 = fork();
    if (pid1 == -1) {
        FAIL() << "Failed to fork first child";
    }
    if (pid1 == 0) {
        RunChildWithVisibleDevices(first_half_devices, expected_chips_first_half, 0x12340000, "Child 1");
    }

    pid_t pid2 = fork();
    if (pid2 == -1) {
        FAIL() << "Failed to fork second child";
    }
    if (pid2 == 0) {
        RunChildWithVisibleDevices(second_half_devices, expected_chips_second_half, 0x56780000, "Child 2");
    }

    auto wait_for_child = [](pid_t pid, const std::string& name) {
        int status = 0;
        ASSERT_EQ(waitpid(pid, &status, 0), pid);
        ASSERT_TRUE(WIFEXITED(status));
        EXPECT_EQ(WEXITSTATUS(status), 0)
            << name << " exited with code " << WEXITSTATUS(status) << " (" << kExitBadContextId << "=context_id"
            << ", " << kExitBadNumDevices << "=num_devices"
            << ", " << kExitBadMeshSize << "=mesh_size"
            << ", " << kExitWorkFailed << "=work_failed"
            << ", " << kExitBufferVerificationFailed << "=buffer_verification)";
    };

    wait_for_child(pid1, "Child 1");
    wait_for_child(pid2, "Child 2");
}

TEST(MetalContextIntegrationTest, OpenTwoMeshDevicesWithSystemMesh) {}

}  // namespace tt::tt_metal
