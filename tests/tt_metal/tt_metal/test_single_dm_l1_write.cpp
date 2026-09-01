// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"
#include "context/metal_context.hpp"

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <numeric>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

// This test requires simulator environment
TEST_F(QuasarMeshDeviceSingleCardFixture, SingleDmL1Write) {
    // Skip if simulator is not available
    char* env_var = std::getenv("TT_METAL_SIMULATOR");
    if (env_var == nullptr) {
        GTEST_SKIP() << "This test can only be run using a simulator. Set TT_METAL_SIMULATOR environment variable.";
    }

    auto mesh_device = devices_[0];

    // Single-core L1 MeshBuffer on node {0,0}: the kernel writes `value` to buf->address() and we
    // read it back through the mesh command queue.
    const CoreRangeSet shard_grid(CoreRange({0, 0}, {0, 0}));
    const ShardSpecBuffer shard_spec(
        shard_grid,
        /*shard_shape=*/{1, 1},
        ShardOrientation::ROW_MAJOR,
        /*page_shape=*/{1, 1},
        /*tensor2d_shape_in_pages=*/{1, 1});
    distributed::DeviceLocalBufferConfig local_cfg{
        .page_size = sizeof(uint32_t),
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED),
    };
    distributed::ReplicatedBufferConfig global_cfg{.size = sizeof(uint32_t)};
    auto buf = distributed::MeshBuffer::create(global_cfg, local_cfg, mesh_device.get());
    const uint32_t address = buf->address();
    const uint32_t value = 0x12345678;
    env_var = std::getenv("TT_METAL_DPRINT_CORES");
    if (env_var == nullptr) {
        std::cerr << "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to 0,0 to see the output of "
                     "the Data Movement kernels."
                  << std::endl;
        std::cerr << "WARNING: For example, export TT_METAL_DPRINT_CORES=0,0" << std::endl;
    }

    // We are going to use the first device (0) and the first core (0, 0) on the device.
    const experimental::NodeCoord node{0, 0};
    // Command queue lets us submit work (execute programs and read/write buffers) to the device.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    // Prepare a workload and a device coordinate range that spans the mesh.
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    const experimental::KernelSpecName DM_KERNEL{"dm_kernel"};

    experimental::KernelSpec dm_kernel_spec{
        .unique_id = DM_KERNEL,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/simple_l1_write.cpp",
        .num_threads = 2,
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"address"},
                .common_runtime_arg_names = {"value"},
            },
        .hw_config = experimental::DataMovementHardwareConfig{},
    };

    experimental::WorkUnitSpec main_wu{
        .name = "main",
        .kernels = {DM_KERNEL},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "single_dm_l1_write",
        .kernels = {dm_kernel_spec},
        .work_units = {main_wu},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = DM_KERNEL,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(node, {{"address", address}}),
        .common_runtime_arg_values = {{"value", value}},
    }};
    experimental::SetProgramRunArgs(program, params);
    std::cout << "Hello, Core {0, 0} on Device 0, Please start execution. I will standby for your communication."
              << std::endl;

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);
    std::vector<uint32_t> outputs;
    distributed::EnqueueReadMeshBuffer(cq, outputs, buf, /*blocking=*/true);

    ASSERT_EQ(outputs[0], value) << "Got the value " << std::hex << outputs[0] << " instead of " << value;
}

// First check for the full-grid tests: confirm the grid is 8x4 (32 nodes), then host-write and
// read back L1 on every node from origin {0,0}. Proves the grid size and that every node's L1 is
// reachable, printing which node failed if any.
TEST_F(QuasarMeshDeviceSingleCardFixture, GridProbeStep0) {
    if (std::getenv("TT_METAL_SIMULATOR") == nullptr) {
        GTEST_SKIP() << "This test can only be run using a simulator.";
    }
    auto mesh_device = devices_[0];

    const auto grid = mesh_device->compute_with_storage_grid_size();
    std::cout << "[STEP0] compute_with_storage_grid_size = " << grid.x << " x " << grid.y
              << "  (nodes=" << (grid.x * grid.y) << ")" << std::endl;

    // Skip on the smaller 1x3/2x3 configs; this suite targets the 8x4 Quasar grid.
    if (grid.x != 8u || grid.y != 4u) {
        GTEST_SKIP() << "grid-test suite targets the 8x4 Quasar sim config (got " << grid.x << "x" << grid.y << ")";
    }

    const uint32_t num_nodes = grid.x * grid.y;
    const CoreRangeSet shard_grid(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    const ShardSpecBuffer shard_spec(
        shard_grid,
        /*shard_shape=*/{1, 1},
        ShardOrientation::ROW_MAJOR,
        /*page_shape=*/{1, 1},
        /*tensor2d_shape_in_pages=*/{num_nodes, 1});
    distributed::DeviceLocalBufferConfig local_cfg{
        .page_size = sizeof(uint32_t),
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED),
    };
    distributed::ReplicatedBufferConfig global_cfg{.size = num_nodes * sizeof(uint32_t)};
    auto buf = distributed::MeshBuffer::create(global_cfg, local_cfg, mesh_device.get());

    std::vector<uint32_t> src(num_nodes);
    std::iota(src.begin(), src.end(), 0u);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(cq, buf, src);
    std::vector<uint32_t> dst;
    distributed::EnqueueReadMeshBuffer(cq, dst, buf, /*blocking=*/true);

    uint32_t ok = 0, fail = 0;
    for (uint32_t i = 0; i < num_nodes; ++i) {
        if (i < dst.size() && dst[i] == src[i]) {
            ++ok;
        } else {
            ++fail;
            std::cout << "[STEP0] MISMATCH node index " << i << " got " << (i < dst.size() ? dst[i] : 0u)
                      << " expected " << src[i] << std::endl;
        }
    }
    std::cout << "[STEP0] per-node L1 MeshBuffer write/read: ok=" << ok << " fail=" << fail << " total=" << num_nodes
              << std::endl;
    EXPECT_EQ(fail, 0u);
    EXPECT_EQ(grid.x, 8u) << "expected 8-wide grid";
    EXPECT_EQ(grid.y, 4u) << "expected 4-tall grid";
}

// Full-grid DM->L1 smoke test. Run one kernel on all 32 nodes at once (target_nodes spans
// {0,0}..{7,3}), giving each node a runtime arg that writes a value unique to that node. Read
// every node back and print a PASS/FAIL grid map. Checks that the kernel and per-node runtime
// args fan out correctly across the whole grid.
TEST_F(QuasarMeshDeviceSingleCardFixture, FullGridDmL1Write_L1a) {
    if (std::getenv("TT_METAL_SIMULATOR") == nullptr) {
        GTEST_SKIP() << "This test can only be run using a simulator.";
    }
    IDevice* dev = devices_[0]->get_devices()[0];
    auto mesh_device = devices_[0];
    const auto grid = mesh_device->compute_with_storage_grid_size();
    // Skip on the smaller 1x3/2x3 configs; this suite targets the 8x4 Quasar grid.
    if (grid.x != 8u || grid.y != 4u) {
        GTEST_SKIP() << "full-grid test targets the 8x4 Quasar sim config (got " << grid.x << "x" << grid.y << ")";
    }

    const uint32_t address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    auto sig_of = [&](uint32_t x, uint32_t y) -> uint32_t { return static_cast<uint32_t>(x + y * grid.x); };

    // Seed every node with a sentinel so a node the kernel never reached shows up as unwritten.
    for (uint32_t y = 0; y < grid.y; ++y) {
        for (uint32_t x = 0; x < grid.x; ++x) {
            std::vector<uint32_t> z{0xffffffffu};
            tt_metal::detail::WriteToDeviceL1(dev, experimental::NodeCoord{x, y}, address, z);
        }
    }

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());

    const experimental::KernelSpecName DM_KERNEL{"dm_kernel"};
    experimental::KernelSpec dm_kernel_spec{
        .unique_id = DM_KERNEL,
        .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/simple_l1_write.cpp",
        .num_threads = 2,
        .runtime_arg_schema = {.runtime_arg_names = {"address", "value"}},
        .hw_config = experimental::DataMovementHardwareConfig{},
    };

    // Fan the SAME kernel to ALL 32 nodes.
    const experimental::NodeRange all_nodes(
        experimental::NodeCoord{0, 0}, experimental::NodeCoord{grid.x - 1, grid.y - 1});
    experimental::WorkUnitSpec main_wu{.name = "main", .kernels = {DM_KERNEL}, .target_nodes = all_nodes};
    experimental::ProgramSpec spec{.name = "full_grid_l1a", .kernels = {dm_kernel_spec}, .work_units = {main_wu}};
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    // Give each node its own value through its per-node runtime args.
    experimental::ProgramRunArgs params;
    experimental::ProgramRunArgs::KernelRunArgs kra{.kernel = DM_KERNEL};
    for (uint32_t y = 0; y < grid.y; ++y) {
        for (uint32_t x = 0; x < grid.x; ++x) {
            experimental::AddRuntimeArgsForNode(
                kra.runtime_arg_values, experimental::NodeCoord{x, y}, {{"address", address}, {"value", sig_of(x, y)}});
        }
    }
    params.kernel_run_args = {kra};
    experimental::SetProgramRunArgs(program, params);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    // Verify per node + print a PASS/FAIL grid map (top row = highest y).
    uint32_t ok = 0, fail = 0;
    std::string map_str;
    for (int y = static_cast<int>(grid.y) - 1; y >= 0; --y) {
        std::string row;
        for (uint32_t x = 0; x < grid.x; ++x) {
            std::vector<uint32_t> r(1, 0xdeadbeefu);
            tt_metal::detail::ReadFromDeviceL1(
                dev, experimental::NodeCoord{x, static_cast<uint32_t>(y)}, address, sizeof(uint32_t), r);
            const uint32_t sig = sig_of(x, static_cast<uint32_t>(y));
            if (r[0] == sig) {
                row += ". ";
                ++ok;
            } else {
                row += "X ";
                ++fail;
                std::cout << "[L1a] FAIL node(" << x << "," << y << ") got 0x" << std::hex << r[0] << " expected 0x"
                          << sig << std::dec << std::endl;
            }
        }
        map_str += "[L1a] y=" + std::to_string(y) + "  " + row + "\n";
    }
    std::cout << "[L1a] " << grid.x << "x" << grid.y << " kernel-fan map (. ok / X fail):\n" << map_str;
    std::cout << "[L1a] ok=" << ok << " fail=" << fail << " total=" << (grid.x * grid.y) << std::endl;
    EXPECT_EQ(fail, 0u);
}

// Full-grid compute smoke test. Run the known-good risc_math compute kernel on all 32 nodes;
// every node must produce the same fixed 16-value output. Exercises the compute pipeline
// (unpack/math/pack) across the whole grid.
TEST_F(QuasarMeshDeviceSingleCardFixture, FullGridCompute_L1c) {
    if (std::getenv("TT_METAL_SIMULATOR") == nullptr) {
        GTEST_SKIP() << "This test can only be run using a simulator.";
    }
    IDevice* dev = devices_[0]->get_devices()[0];
    auto mesh_device = devices_[0];
    const auto grid = mesh_device->compute_with_storage_grid_size();
    // Skip on the smaller 1x3/2x3 configs; this suite targets the 8x4 Quasar grid.
    if (grid.x != 8u || grid.y != 4u) {
        GTEST_SKIP() << "full-grid test targets the 8x4 Quasar sim config (got " << grid.x << "x" << grid.y << ")";
    }

    const uint32_t l1_address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const std::vector<uint32_t> expected = {4, 6, 5, 9, 8, 10, 9, 13, 12, 14, 13, 17, 16, 18, 17, 21};

    // Pre-fill every node's 16-word output slot with a sentinel that never appears in `expected`,
    // so a node whose compute never ran (or whose output landed on the wrong node) reads back the
    // sentinel and fails instead of silently passing.
    for (uint32_t y = 0; y < grid.y; ++y) {
        for (uint32_t x = 0; x < grid.x; ++x) {
            std::vector<uint32_t> z(16, 0xC0FFEEu);
            tt_metal::detail::WriteToDeviceL1(dev, experimental::NodeCoord{x, y}, l1_address, z);
        }
    }

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());

    const experimental::KernelSpecName COMPUTE_KERNEL{"risc_math"};
    experimental::KernelSpec compute_kernel_spec{
        .unique_id = COMPUTE_KERNEL,
        .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/compute/risc_math.cpp",
        .num_threads = 4,
        .runtime_arg_schema = {.runtime_arg_names = {"l1_address"}},
        .hw_config = experimental::ComputeHardwareConfig{},
    };

    const experimental::NodeRange all_nodes(
        experimental::NodeCoord{0, 0}, experimental::NodeCoord{grid.x - 1, grid.y - 1});
    experimental::WorkUnitSpec main_wu{.name = "main", .kernels = {COMPUTE_KERNEL}, .target_nodes = all_nodes};
    experimental::ProgramSpec spec{
        .name = "full_grid_compute", .kernels = {compute_kernel_spec}, .work_units = {main_wu}};
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    experimental::ProgramRunArgs params;
    experimental::ProgramRunArgs::KernelRunArgs kra{.kernel = COMPUTE_KERNEL};
    for (uint32_t y = 0; y < grid.y; ++y) {
        for (uint32_t x = 0; x < grid.x; ++x) {
            experimental::AddRuntimeArgsForNode(
                kra.runtime_arg_values, experimental::NodeCoord{x, y}, {{"l1_address", l1_address}});
        }
    }
    params.kernel_run_args = {kra};
    experimental::SetProgramRunArgs(program, params);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    uint32_t ok = 0, fail = 0;
    std::string map_str;
    for (int y = static_cast<int>(grid.y) - 1; y >= 0; --y) {
        std::string row;
        for (uint32_t x = 0; x < grid.x; ++x) {
            std::vector<uint32_t> r(16, 0xdeadbeefu);
            tt_metal::detail::ReadFromDeviceL1(
                dev, experimental::NodeCoord{x, static_cast<uint32_t>(y)}, l1_address, 16 * sizeof(uint32_t), r);
            if (r == expected) {
                row += ". ";
                ++ok;
            } else {
                row += "X ";
                ++fail;
                std::cout << "[L1c] FAIL node(" << x << "," << y << ") first=" << r[0] << " (expected " << expected[0]
                          << ")" << std::endl;
            }
        }
        map_str += "[L1c] y=" + std::to_string(y) + "  " + row + "\n";
    }
    std::cout << "[L1c] " << grid.x << "x" << grid.y << " compute map (. ok / X fail):\n" << map_str;
    std::cout << "[L1c] ok=" << ok << " fail=" << fail << " total=" << (grid.x * grid.y) << std::endl;
    EXPECT_EQ(fail, 0u);
}

// Grid NoC multicast fan-out. One source node (logical {0,0}) multicasts a value to the same L1
// address on every node in the full-grid rectangle, exercising the NoC multicast path. Host
// pre-seeds a sentinel on every node and reads all 32 back, printing a PASS/FAIL grid map: a
// dropped row or column shows up as a node that never updated.
TEST_F(QuasarMeshDeviceSingleCardFixture, GridMulticastFanOut) {
    if (std::getenv("TT_METAL_SIMULATOR") == nullptr) {
        GTEST_SKIP() << "This test can only be run using a simulator.";
    }
    IDevice* dev = devices_[0]->get_devices()[0];
    auto mesh_device = devices_[0];
    const auto grid = mesh_device->compute_with_storage_grid_size();
    if (grid.x != 8u || grid.y != 4u) {
        GTEST_SKIP() << "full-grid test targets the 8x4 Quasar sim config (got " << grid.x << "x" << grid.y << ")";
    }

    const uint32_t address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t value = 0x5eeded42u;
    const uint32_t sentinel = 0xdeadbeefu;

    // Seed every node with a sentinel so a node the multicast never reached shows as unwritten.
    for (uint32_t y = 0; y < grid.y; ++y) {
        for (uint32_t x = 0; x < grid.x; ++x) {
            std::vector<uint32_t> z{sentinel};
            tt_metal::detail::WriteToDeviceL1(dev, experimental::NodeCoord{x, y}, address, z);
        }
    }

    // Source at logical {0,0}; multicast rectangle = physical coords spanning the whole 8x4 grid.
    const experimental::NodeCoord src_node{0, 0};
    const CoreCoord p_lo = mesh_device->worker_core_from_logical_core(experimental::NodeCoord{0, 0});
    const CoreCoord p_hi = mesh_device->worker_core_from_logical_core(experimental::NodeCoord{grid.x - 1, grid.y - 1});
    const uint32_t num_dests = grid.x * grid.y - 1;  // rectangle minus the (auto-excluded) source

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());

    const experimental::KernelSpecName MCAST{"mcast_writer"};
    experimental::KernelSpec mcast_spec{
        .unique_id = MCAST,
        .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/grid_multicast_writer.cpp",
        .num_threads = 1,
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"value", "result_addr", "mcast_x_start", "mcast_y_start", "mcast_x_end", "mcast_y_end", "num_dests"}},
        .hw_config = experimental::DataMovementHardwareConfig{},
    };
    experimental::WorkUnitSpec wu{.name = "main", .kernels = {MCAST}, .target_nodes = src_node};
    experimental::ProgramSpec spec{.name = "grid_mcast_fanout", .kernels = {mcast_spec}, .work_units = {wu}};
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = MCAST,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
            src_node,
            {{"value", value},
             {"result_addr", address},
             {"mcast_x_start", static_cast<uint32_t>(p_lo.x)},
             {"mcast_y_start", static_cast<uint32_t>(p_lo.y)},
             {"mcast_x_end", static_cast<uint32_t>(p_hi.x)},
             {"mcast_y_end", static_cast<uint32_t>(p_hi.y)},
             {"num_dests", num_dests}})}};
    experimental::SetProgramRunArgs(program, params);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    // Verify per node + print a PASS/FAIL grid map (top row = highest y).
    uint32_t ok = 0, fail = 0;
    std::string map_str;
    for (int y = static_cast<int>(grid.y) - 1; y >= 0; --y) {
        std::string row;
        for (uint32_t x = 0; x < grid.x; ++x) {
            std::vector<uint32_t> r(1, 0u);
            tt_metal::detail::ReadFromDeviceL1(
                dev, experimental::NodeCoord{x, static_cast<uint32_t>(y)}, address, sizeof(uint32_t), r);
            if (r[0] == value) {
                row += ". ";
                ++ok;
            } else {
                row += "X ";
                ++fail;
                std::cout << "[MCAST] FAIL node(" << x << "," << y << ") got 0x" << std::hex << r[0] << " expected 0x"
                          << value << std::dec << std::endl;
            }
        }
        map_str += "[MCAST] y=" + std::to_string(y) + "  " + row + "\n";
    }
    std::cout << "[MCAST] " << grid.x << "x" << grid.y << " multicast map (. ok / X fail):\n" << map_str;
    std::cout << "[MCAST] ok=" << ok << " fail=" << fail << " total=" << (grid.x * grid.y) << std::endl;
    EXPECT_EQ(fail, 0u);
}
