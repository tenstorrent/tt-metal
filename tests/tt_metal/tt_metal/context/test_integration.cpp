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
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>

#include <umd/device/types/arch.hpp>

// Internal access
#include "impl/context/context_types.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "context/metal_env_accessor.hpp"
#include "device/mock_device_util.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {

namespace {

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

    MetalEnv child_env;

    auto child_num_devices = child_env.get_num_pcie_devices();
    log_info(tt::LogTest, "{}: TT_VISIBLE_DEVICES={}, num_devices={}", child_name, visible_devices, child_num_devices);
    if (child_num_devices != expected_num_chips) {
        _exit(kExitBadNumDevices);
    }

    auto child_mesh_shape = child_env.get_system_mesh().shape();
    auto child_mesh_device = child_env.create_mesh_device(distributed::MeshDeviceConfig(child_mesh_shape));
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

TEST(MetalContextIntegrationTest, Legacy) {
    auto mesh_shape = tt_metal::distributed::SystemMesh::instance().shape();
    auto mesh_device_config = distributed::MeshDeviceConfig(mesh_shape);
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create(mesh_device_config);
    EXPECT_EQ(mesh_device->num_devices(), mesh_shape.mesh_size());
    // Required for this unit test because legacy behaviour of MetalContext is not to close the cluster until atexit
    // Close it right now so remaining tests can proceed
    mesh_device->close();

    // It was found that during ~MeshDevice, some calls to MetalContext::instance() were made which caused
    // MetalContext to implicitly reinitialize thus undoing the effects of DestroyAllContexts().
    // Subsequent tests will hang if that happens.
    tt::tt_metal::detail::ReleaseOwnership();
}

TEST(MetalContextIntegrationTest, HelloWorld) {
    MetalEnv env;

    auto mesh_shape = env.get_system_mesh().shape();
    auto mesh_device_config = distributed::MeshDeviceConfig(mesh_shape);
    auto mesh_device = env.create_mesh_device(mesh_device_config);

    mesh_device->close();
}

TEST(MetalContextIntegrationTest, HelloWorldQueryThenCreate) {
    ContextId context_id;
    {
        MetalEnv env;

        size_t l1_size = env.get_l1_size();
        size_t trace_region_size = l1_size * 0.3;
        size_t l1_small_region_size = l1_size * 0.05;

        auto mesh_shape = env.get_system_mesh().shape();
        auto mesh_device_config = distributed::MeshDeviceConfig(mesh_shape);
        auto mesh_device = env.create_mesh_device(mesh_device_config, trace_region_size, l1_small_region_size);
        context_id = mesh_device->impl().get_context_id();
    }

    // We only support 1 MetalEnv <-> 1 MetalContext instance for the physical cluster right now
    ASSERT_EQ(context_id, DEFAULT_CONTEXT_ID);

    // Assert that the MetalContext instance was cleaned up after MeshDevice close
    ASSERT_FALSE(MetalContext::instance_exists(context_id));

    // Check if can create another env
    // If this hangs, it means there is a dangling cluster open somewhere
    {
        MetalEnv env;
        ASSERT_NO_THROW(env.get_num_pcie_devices());
    }
}

TEST(MetalContextIntegrationTest, MockDeviceOnly) {
    ContextId context_id;
    {
        MetalEnv mock_env_bh_1{MetalEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 1))};

        auto mesh_config_mock = distributed::MeshDeviceConfig(distributed::MeshShape(1));
        auto mock_device = mock_env_bh_1.create_mesh_device(mesh_config_mock);
        context_id = mock_device->impl().get_context_id();

        // Test buffer allocation and deallocation
        constexpr size_t page_size = 4096;
        constexpr size_t buffer_size = page_size * 12;
        distributed::DeviceLocalBufferConfig local_config{.page_size = buffer_size, .buffer_type = BufferType::L1};
        distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};
        auto buffer = distributed::MeshBuffer::create(buffer_config, local_config, mock_device.get());
        ASSERT_GT(buffer->address(), 0);
        ASSERT_TRUE(buffer->is_allocated());
        buffer->deallocate();
        ASSERT_FALSE(buffer->is_allocated());

        // Test command queue operations
        auto& cq = mock_device->mesh_command_queue();
        constexpr size_t num_elements = 16;
        std::vector<uint32_t> write_data(num_elements);
        std::iota(write_data.begin(), write_data.end(), 0xDEADBEEF);

        distributed::EnqueueWriteMeshBuffer(cq, buffer, write_data, true);

        std::vector<uint32_t> read_data;
        distributed::EnqueueReadMeshBuffer(cq, read_data, buffer, true);

        // TODO: Uncomment this once CreateProgram and CreateKernel stop implicitly creating the physical metal context
        // https://github.com/tenstorrent/tt-metal/issues/39849
        // auto program = CreateProgram();
        // distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mock_device->shape());
        // auto core_grid = mock_device->compute_with_storage_grid_size();
        // auto core_range = CoreRange({0, 0}, {core_grid.x - 1, core_grid.y - 1});

        // CreateKernelFromString(
        //     program,
        //     "void kernel_main() {}",
        //     core_range,
        //     DataMovementConfig{});

        // distributed::MeshWorkload workload;
        // workload.add_program(device_range, std::move(program));
        // distributed::EnqueueMeshWorkload(cq, workload, true);
    }

    // Assert that we didn't implicitly create the physical metal context
    ASSERT_FALSE(MetalContext::instance_exists(DEFAULT_CONTEXT_ID));

    // Assert that the MetalContext instance was cleaned up after MeshDevice close
    ASSERT_FALSE(MetalContext::instance_exists(context_id));
}

TEST(MetalContextIntegrationTest, MockDeviceCommandQueueOnly) {
    // This will try to run some actions on the mock device's command queue and then check
    // it didn't implicitly create the physical metal context
    {
        MetalEnv mock_env_bh_1{MetalEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 1))};

        auto mesh_config_mock = distributed::MeshDeviceConfig(distributed::MeshShape(1));
        auto mock_device = mock_env_bh_1.create_mesh_device(mesh_config_mock);
    }

    // Assert that we didn't implicitly create the physical metal context
    ASSERT_FALSE(MetalContext::instance_exists(DEFAULT_CONTEXT_ID));
}

TEST(MetalContextIntegrationTest, CoexistingSiliconAndMockDevice) {
    // Create mock mesh device with 1 blackhole chip
    MetalEnv mock_env_bh_1{MetalEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 1))};
    auto mock_mesh_shape_bh_1 = mock_env_bh_1.get_system_mesh().shape();
    auto mock_mesh_device_config_bh_1 = distributed::MeshDeviceConfig(mock_mesh_shape_bh_1);
    std::shared_ptr<distributed::MeshDevice> mock_mesh_device_bh_1 =
        mock_env_bh_1.create_mesh_device(mock_mesh_device_config_bh_1);
    log_info(tt::LogTest, "Created mock mesh device with shape {}", mock_mesh_device_bh_1->shape().dims());

    // Create mock mesh device with 2 blackhole chips
    MetalEnv mock_env_bh_2{MetalEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2))};
    auto mock_mesh_shape_bh_2 = mock_env_bh_2.get_system_mesh().shape();
    auto mock_mesh_device_config_bh_2 = distributed::MeshDeviceConfig(mock_mesh_shape_bh_2);
    auto mock_mesh_device_bh_2 = mock_env_bh_2.create_mesh_device(mock_mesh_device_config_bh_2);
    log_info(tt::LogTest, "Created mock mesh device with shape {}", mock_mesh_device_bh_2->shape().dims());

    // Create silicon mesh
    MetalEnv env;
    auto mesh_shape = env.get_system_mesh().shape();
    auto mesh_device_config = distributed::MeshDeviceConfig(mesh_shape);
    std::shared_ptr<distributed::MeshDevice> mesh_device = env.create_mesh_device(mesh_device_config);
    log_info(tt::LogTest, "Created silicon mesh device with shape {}", mesh_device->shape().dims());

    ASSERT_EQ(mock_mesh_device_bh_1->get_devices().size(), 1);
    ASSERT_EQ(mock_mesh_device_bh_2->get_devices().size(), 2);
}

// Same test as above but reverse the order to ensure no hangs due to unexpected internal objects created for the
// incorrect context id
TEST(MetalContextIntegrationTest, CoexistingMockAndSiliconDevice) {
    // Create silicon mesh
    MetalEnv silicon_env;

    auto mesh_shape = silicon_env.get_system_mesh().shape();
    auto mesh_device_config = distributed::MeshDeviceConfig(mesh_shape);
    std::shared_ptr<distributed::MeshDevice> mesh_device = silicon_env.create_mesh_device(mesh_device_config);
    log_info(tt::LogTest, "Created silicon mesh device with shape {}", mesh_device->shape().dims());

    // Create mock mesh device with 1 blackhole chip
    MetalEnv mock_env_bh_1{MetalEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 1))};
    auto mock_mesh_shape_bh_1 = mock_env_bh_1.get_system_mesh().shape();
    auto mock_mesh_device_config_bh_1 = distributed::MeshDeviceConfig(mock_mesh_shape_bh_1);
    auto mock_mesh_device_bh_1 = mock_env_bh_1.create_mesh_device(mock_mesh_device_config_bh_1);
    log_info(tt::LogTest, "Created mock mesh device with shape {}", mock_mesh_device_bh_1->shape().dims());

    // Create mock mesh device with 2 blackhole chips
    MetalEnv mock_env_bh_2{MetalEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2))};

    auto mock_mesh_shape_bh_2 = mock_env_bh_2.get_system_mesh().shape();
    auto mock_mesh_device_config_bh_2 = distributed::MeshDeviceConfig(mock_mesh_shape_bh_2);
    auto mock_mesh_device_bh_2 = mock_env_bh_2.create_mesh_device(mock_mesh_device_config_bh_2);
    log_info(tt::LogTest, "Created mock mesh device with shape {}", mock_mesh_device_bh_2->shape().dims());

    ASSERT_EQ(mock_mesh_device_bh_1->get_devices().size(), 1);
    ASSERT_EQ(mock_mesh_device_bh_2->get_devices().size(), 2);
}

TEST(MetalContextIntegrationTest, ForkMockAndRealDevice) {
    // Query hardware state before forking
    {
        MetalEnv env;

        auto arch = env.get_arch();
        auto l1_size = env.get_l1_size();
        auto num_devices = env.get_num_available_devices();
        log_info(tt::LogTest, "Pre-fork: arch={}, L1 size={}, num_devices={}", arch, l1_size, num_devices);
        EXPECT_GT(l1_size, 0u);
        EXPECT_GT(num_devices, 0u);
    }

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

        MetalEnv mock_env{MetalEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2).value())};

        if (!MetalEnvAccessor(mock_env).impl().get_rtoptions().get_mock_enabled()) {
            _exit(1);
        }

        if (mock_env.get_arch() != tt::ARCH::BLACKHOLE) {
            _exit(2);
        }

        auto mock_mesh_shape = mock_env.get_system_mesh().shape();
        auto mock_mesh_device = mock_env.create_mesh_device(distributed::MeshDeviceConfig(mock_mesh_shape));
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

    MetalEnv env;

    auto real_arch = env.get_arch();
    auto real_l1_size = env.get_l1_size();
    log_info(tt::LogTest, "Parent (real): arch={}, L1 size={}", real_arch, real_l1_size);
    EXPECT_GT(real_l1_size, 0u);

    auto mesh_shape = env.get_system_mesh().shape();
    auto silicon_mesh_device = env.create_mesh_device(distributed::MeshDeviceConfig(mesh_shape));
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
}

TEST(MetalContextIntegrationTest, ForkWithDisjointDevices) {
    size_t num_mmio_devices = 0;
    size_t num_available_devices = 0;
    {
        MetalEnv env;
        num_mmio_devices = env.get_num_pcie_devices();
        num_available_devices = env.get_num_available_devices();
        log_info(tt::LogTest, "System has {} PCIe devices, {} total chips", num_mmio_devices, num_available_devices);
    }

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
            << name << " exited with code " << WEXITSTATUS(status) << "(" << kExitBadNumDevices << "=num_devices"
            << ", " << kExitBadMeshSize << "=mesh_size"
            << ", " << kExitWorkFailed << "=work_failed"
            << ", " << kExitBufferVerificationFailed << "=buffer_verification)";
    };

    wait_for_child(pid1, "Child 1");
    wait_for_child(pid2, "Child 2");
}

}  // namespace tt::tt_metal
