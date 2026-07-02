// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
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
#include "dispatch/system_memory_manager.hpp"
#include "impl/context/context_types.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "context/metal_env_accessor.hpp"
#include "device/mock_device_util.hpp"
#include "impl/context/metal_context.hpp"

#include <ttnn/graph/graph_query_op_constraints.hpp>
#include <ttnn/operations/ccl/all_gather/all_gather.hpp>
#include <ttnn/tensor/layout/page_config.hpp>
#include <ttnn/tensor/layout/tensor_layout.hpp>
#include <ttnn/tensor/tensor_ops.hpp>
#include <ttnn/tensor/tensor_spec.hpp>
#include <ttnn/tensor/types.hpp>
#include <ttnn/types.hpp>

#include <limits>
#include <optional>

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
    // A no-op kernel only validates the create / JIT / dispatch pipeline,
    // so a single worker core is sufficient. (0, 0) is always a logical
    // compute core regardless of dispatch-core configuration.
    auto core_range = CoreRange({0, 0}, {0, 0});

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
    // The system mesh shape may be smaller than expected_num_chips when chips are not fabric-connected
    // (e.g. standalone PCIe cards without direct chip-to-chip links). Verify at least one device opened.
    if (child_mesh_device->num_devices() == 0) {
        _exit(kExitBadMeshSize);
    }
    log_info(
        tt::LogTest,
        "{}: opened MeshDevice with {} device(s) (expected up to {})",
        child_name,
        child_mesh_device->num_devices(),
        expected_num_chips);

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

ttnn::TensorSpec MakeAllGatherInputSpec() {
    return ttnn::TensorSpec(
        ttnn::Shape(tt::tt_metal::Array4D{4, 2, 5 * 32, 7 * 32}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));
}

ttnn::graph::ConstraintQueryResponse RunAllGatherConstraintQuery(distributed::MeshDevice* device) {
    auto sharded_topology = TensorTopology::create_sharded_tensor_topology(device->shape(), /*shard_dim=*/0);
    ttnn::graph::DistributedTensorSpec dist_input{MakeAllGatherInputSpec(), sharded_topology};

    return ttnn::graph::query_op_constraints(
        [](auto&&... args) { return ttnn::all_gather(std::forward<decltype(args)>(args)...); },
        device,
        dist_input,
        /*dim=*/3,
        /*cluster_axis=*/std::optional<uint32_t>(1),
        /*subdevice_id=*/std::optional<SubDeviceId>{},
        /*memory_config=*/std::optional<MemoryConfig>{},
        /*optional_output_tensor=*/std::optional<::ttnn::Tensor>{},
        /*num_links=*/std::optional<uint32_t>(1),
        /*topology=*/std::optional<tt_fabric::Topology>(tt_fabric::Topology::Linear));
}

struct LegacyMockFabricCleanup {
    bool active = false;
    ~LegacyMockFabricCleanup() {
        if (active) {
            tt_fabric::SetFabricConfig(tt_fabric::FabricConfig::DISABLED);
            experimental::disable_mock_mode();
        }
    }
};

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

        // Test command queue operations. Source vector is sized to fill the entire buffer so
        // EnqueueWriteMeshBuffer's precondition (src bytes >= mesh buffer bytes, added in #43429)
        // is satisfied; the prior 16-element vector was a pre-#43429 leftover.
        auto& cq = mock_device->mesh_command_queue();
        constexpr size_t num_elements = buffer_size / sizeof(uint32_t);
        std::vector<uint32_t> write_data(num_elements);
        std::iota(write_data.begin(), write_data.end(), 0xDEADBEEFu);

        distributed::EnqueueWriteMeshBuffer(cq, buffer, write_data, true);

        std::vector<uint32_t> read_data;
        distributed::EnqueueReadMeshBuffer(cq, read_data, buffer, true);

        auto program = CreateProgram();
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mock_device->shape());
        // Single worker core; see comment in PerformDeviceWork.
        auto core_range = CoreRange({0, 0}, {0, 0});

        CreateKernelFromString(program, "void kernel_main() {}", core_range, DataMovementConfig{});

        distributed::MeshWorkload workload;
        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, true);
    }

    // Assert that we didn't implicitly create the physical metal context
    ASSERT_FALSE(MetalContext::instance_exists(DEFAULT_CONTEXT_ID));

    // Assert that the MetalContext instance was cleaned up after MeshDevice close
    ASSERT_FALSE(MetalContext::instance_exists(context_id));
}

TEST(MetalContextIntegrationTest, CoexistingSiliconAndMockDevice) {
    // BuildEnvManager is a process-global singleton whose kernel/firmware build-state index tables
    // are sized once from the HAL of whichever context first triggers add_build_env(). When the
    // mock arch differs from the silicon arch (e.g. mock-BH on a WH runner), the second context
    // ends up indexing a wrong-sized table and JIT'd kernels reference incorrect dispatch
    // addresses, causing silicon dispatch to hang. Forge's real flow is same-arch coexistence
    // (mock used as a compile-time stand-in for the live silicon), so probe the silicon arch and
    // create the mock to match. Cross-arch coexistence is a separate `BuildEnvManager`-singleton
    // limitation tracked outside #38445.
    tt::ARCH silicon_arch;
    {
        MetalEnv probe;
        silicon_arch = probe.get_arch();
    }

    // Create mock mesh device with 1 chip on the silicon arch
    MetalEnv mock_env_1{MetalEnvDescriptor(experimental::get_mock_cluster_desc_name(silicon_arch, 1))};
    auto mock_mesh_shape_1 = mock_env_1.get_system_mesh().shape();
    auto mock_mesh_device_config_1 = distributed::MeshDeviceConfig(mock_mesh_shape_1);
    std::shared_ptr<distributed::MeshDevice> mock_mesh_device_1 =
        mock_env_1.create_mesh_device(mock_mesh_device_config_1);
    log_info(tt::LogTest, "Created mock mesh device with shape {}", mock_mesh_device_1->shape().dims());

    // Create mock mesh device with 2 chips on the silicon arch
    MetalEnv mock_env_2{MetalEnvDescriptor(experimental::get_mock_cluster_desc_name(silicon_arch, 2))};
    auto mock_mesh_shape_2 = mock_env_2.get_system_mesh().shape();
    auto mock_mesh_device_config_2 = distributed::MeshDeviceConfig(mock_mesh_shape_2);
    auto mock_mesh_device_2 = mock_env_2.create_mesh_device(mock_mesh_device_config_2);
    log_info(tt::LogTest, "Created mock mesh device with shape {}", mock_mesh_device_2->shape().dims());

    // Create silicon mesh
    MetalEnv env;
    auto mesh_shape = env.get_system_mesh().shape();
    auto mesh_device_config = distributed::MeshDeviceConfig(mesh_shape);
    std::shared_ptr<distributed::MeshDevice> mesh_device = env.create_mesh_device(mesh_device_config);
    log_info(tt::LogTest, "Created silicon mesh device with shape {}", mesh_device->shape().dims());

    ASSERT_EQ(mock_mesh_device_1->get_devices().size(), 1);
    ASSERT_EQ(mock_mesh_device_2->get_devices().size(), 2);

    // Run a no-op program on both the mock and silicon devices side-by-side. This exercises the
    // full create-program / create-kernel / JIT-compile / enqueue-workload pipeline through both
    // contexts in the same process to validate mock+silicon coexistence (issue #38445).
    auto run_noop_program = [](distributed::MeshDevice& target, const std::string& label) {
        auto program = CreateProgram();
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(target.shape());
        // Single worker core; see comment in PerformDeviceWork.
        auto core_range = CoreRange({0, 0}, {0, 0});

        CreateKernelFromString(program, "void kernel_main() {}", core_range, DataMovementConfig{});

        distributed::MeshWorkload workload;
        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(target.mesh_command_queue(), workload, true);
        log_info(tt::LogTest, "Successfully enqueued no-op program on {}", label);
    };

    run_noop_program(*mock_mesh_device_1, "mock_mesh_device_1");
    run_noop_program(*mesh_device, "silicon_mesh_device");
}

// Same test as above but reverse the order to ensure no hangs due to unexpected internal objects created for the
// incorrect context id. See `CoexistingSiliconAndMockDevice` for why mock arch is matched to silicon arch.
TEST(MetalContextIntegrationTest, CoexistingMockAndSiliconDevice) {
    // Create silicon mesh
    MetalEnv silicon_env;
    const tt::ARCH silicon_arch = silicon_env.get_arch();

    auto mesh_shape = silicon_env.get_system_mesh().shape();
    auto mesh_device_config = distributed::MeshDeviceConfig(mesh_shape);
    std::shared_ptr<distributed::MeshDevice> mesh_device = silicon_env.create_mesh_device(mesh_device_config);
    log_info(tt::LogTest, "Created silicon mesh device with shape {}", mesh_device->shape().dims());

    // Create mock mesh device with 1 chip on the silicon arch
    MetalEnv mock_env_1{MetalEnvDescriptor(experimental::get_mock_cluster_desc_name(silicon_arch, 1))};
    auto mock_mesh_shape_1 = mock_env_1.get_system_mesh().shape();
    auto mock_mesh_device_config_1 = distributed::MeshDeviceConfig(mock_mesh_shape_1);
    auto mock_mesh_device_1 = mock_env_1.create_mesh_device(mock_mesh_device_config_1);
    log_info(tt::LogTest, "Created mock mesh device with shape {}", mock_mesh_device_1->shape().dims());

    // Create mock mesh device with 2 chips on the silicon arch
    MetalEnv mock_env_2{MetalEnvDescriptor(experimental::get_mock_cluster_desc_name(silicon_arch, 2))};

    auto mock_mesh_shape_2 = mock_env_2.get_system_mesh().shape();
    auto mock_mesh_device_config_2 = distributed::MeshDeviceConfig(mock_mesh_shape_2);
    auto mock_mesh_device_2 = mock_env_2.create_mesh_device(mock_mesh_device_config_2);
    log_info(tt::LogTest, "Created mock mesh device with shape {}", mock_mesh_device_2->shape().dims());

    ASSERT_EQ(mock_mesh_device_1->get_devices().size(), 1);
    ASSERT_EQ(mock_mesh_device_2->get_devices().size(), 2);

    // Run a no-op program on both the silicon and mock devices in this reversed-creation-order
    // case to confirm the JIT/compile/enqueue pipeline does not depend on which context was
    // opened first.
    auto run_noop_program = [](distributed::MeshDevice& target, const std::string& label) {
        auto program = CreateProgram();
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(target.shape());
        // Single worker core; see comment in PerformDeviceWork.
        auto core_range = CoreRange({0, 0}, {0, 0});

        CreateKernelFromString(program, "void kernel_main() {}", core_range, DataMovementConfig{});

        distributed::MeshWorkload workload;
        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(target.mesh_command_queue(), workload, true);
        log_info(tt::LogTest, "Successfully enqueued no-op program on {}", label);
    };

    run_noop_program(*mesh_device, "silicon_mesh_device");
    run_noop_program(*mock_mesh_device_1, "mock_mesh_device_1");
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

TEST(MetalContextIntegrationTest, MeshDevicePropagatesContextId) {
    MetalEnvDescriptor desc(experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 8));
    MetalEnv env(desc);
    auto mesh_shape = env.get_system_mesh().shape();
    auto mesh_device_config = distributed::MeshDeviceConfig(mesh_shape);
    auto mesh_device = env.create_mesh_device(mesh_device_config);

    ContextId context_id = mesh_device->impl().get_context_id();

    EXPECT_NE(mesh_device->impl().get_context_id(), DEFAULT_CONTEXT_ID);

    auto submesh_0 = mesh_device->create_submesh(distributed::MeshShape(1, 1));
    EXPECT_EQ(submesh_0->impl().get_context_id(), context_id);

    auto submesh_1 = mesh_device->create_submesh(distributed::MeshShape(1, 2));
    EXPECT_EQ(submesh_1->impl().get_context_id(), context_id);

    SystemMemoryManager& sysmem_manager = mesh_device->impl().get_devices()[0]->sysmem_manager();
    EXPECT_EQ(sysmem_manager.get_context_id(), context_id);
}

TEST(MetalEnvMockCCL, FabricInDescriptor_CreatesMeshAndQuerySucceeds) {
    auto mock_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 2);
    ASSERT_TRUE(mock_path.has_value());

    FabricConfigDescriptor fabric_desc{};
    fabric_desc.fabric_config = tt_fabric::FabricConfig::FABRIC_1D;
    fabric_desc.reliability_mode = tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE;
    fabric_desc.num_routing_planes = std::numeric_limits<uint8_t>::max();
    MetalEnv env{MetalEnvDescriptor{mock_path, fabric_desc}};

    auto device = env.create_mesh_device(distributed::MeshDeviceConfig{distributed::MeshShape{1u, 2u}});
    ASSERT_NE(device, nullptr);
    auto response = RunAllGatherConstraintQuery(device.get());
    EXPECT_EQ(response.status, ttnn::graph::ExecutionStatus::Success)
        << "query failed: " << response.error_message.value_or("(no message)");
}

TEST(MetalEnvMockCCL, FabricAfterDeviceCreation_QuerySucceeds) {
    auto mock_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 2);
    ASSERT_TRUE(mock_path.has_value());

    MetalEnv env{MetalEnvDescriptor{mock_path}};
    auto device = env.create_mesh_device(distributed::MeshDeviceConfig{distributed::MeshShape{1u, 2u}});
    ASSERT_NE(device, nullptr);

    ASSERT_TRUE(MetalEnvAccessor{env}.impl().set_fabric_config(
        tt_fabric::FabricConfig::FABRIC_1D, tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE));

    auto response = RunAllGatherConstraintQuery(device.get());
    EXPECT_EQ(response.status, ttnn::graph::ExecutionStatus::Success)
        << "query failed: " << response.error_message.value_or("(no message)");
}

TEST(MetalEnvMockCCL, LegacyConfigureMockMode_QueryPasses) {
    LegacyMockFabricCleanup cleanup;
    experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, /*num_chips=*/2);
    cleanup.active = true;

    // Fabric must be configured before opening devices; set_fabric_config rejects
    // non-DISABLED changes while devices are still open.
    tt_fabric::SetFabricConfig(
        tt_fabric::FabricConfig::FABRIC_1D, tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
    MetalContext::instance().initialize_fabric_config();

    ttnn::graph::ConstraintQueryResponse response;
    {
        auto device = distributed::MeshDevice::create(distributed::MeshDeviceConfig{distributed::MeshShape{1u, 2u}});
        ASSERT_NE(device, nullptr);
        response = RunAllGatherConstraintQuery(device.get());
    }

    EXPECT_EQ(response.status, ttnn::graph::ExecutionStatus::Success)
        << "query failed: " << response.error_message.value_or("(no message)");
}

}  // namespace tt::tt_metal
