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
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include <map>
#include <numeric>
#include <thread>
#include <chrono>

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
        .hw_config = experimental::DataMovementGen2Config{},
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
    Program program = experimental::MakeProgramFromSpec(this->device(), spec);

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
            slow_dispatch::WriteToL1(this->device(), experimental::NodeCoord{x, y}, address, z);
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
        .hw_config = experimental::DataMovementGen2Config{},
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
            slow_dispatch::ReadFromL1(
                this->device(), experimental::NodeCoord{x, static_cast<uint32_t>(y)}, address, sizeof(uint32_t), r);
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
            slow_dispatch::WriteToL1(this->device(), experimental::NodeCoord{x, y}, l1_address, z);
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
        .hw_config = experimental::ComputeGen2Config{},
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
            slow_dispatch::ReadFromL1(
                this->device(),
                experimental::NodeCoord{x, static_cast<uint32_t>(y)},
                l1_address,
                16 * sizeof(uint32_t),
                r);
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
            slow_dispatch::WriteToL1(this->device(), experimental::NodeCoord{x, y}, address, z);
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
        .hw_config = experimental::DataMovementGen2Config{},
    };
    experimental::WorkUnitSpec wu{.name = "main", .kernels = {MCAST}, .target_nodes = src_node};
    experimental::ProgramSpec spec{.name = "grid_mcast_fanout", .kernels = {mcast_spec}, .work_units = {wu}};
    Program program = experimental::MakeProgramFromSpec(this->device(), spec);

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
            slow_dispatch::ReadFromL1(
                this->device(), experimental::NodeCoord{x, static_cast<uint32_t>(y)}, address, sizeof(uint32_t), r);
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


// qsr.s1 bring-up probe: does a DEVICE-initiated NoC write reach the tile the host intended?
// Every test that hangs on qsr.s1 (DmLoopback, BmmMultinode, PackReluZero, the SD prefetcher battery)
// performs device-initiated inter-tile NoC writes; every test that passes stays on-tile or goes via the
// host. One DM kernel on logical {0,0} unicasts a marker to another worker's L1 and to a DRAM tile; the
// host seeds sentinels on every worker and reads everything back, including L1[0] (the firmware launch
// word), and prints where the writes landed.
TEST_F(QuasarMeshDeviceSingleCardFixture, Qsr1NocUnicastProbe) {
    if (std::getenv("TT_METAL_SIMULATOR") == nullptr) {
        GTEST_SKIP() << "This test can only be run using a simulator.";
    }
    auto mesh_device = devices_[0];
    const auto grid = mesh_device->compute_with_storage_grid_size();
    if (grid.x * grid.y < 2) {
        GTEST_SKIP() << "needs at least two worker nodes (got " << grid.x << "x" << grid.y << ")";
    }
    const uint32_t result_addr = MetalContext::instance().hal().get_dev_addr(
                                     HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED) +
                                 0x1000;
    const uint32_t value = 0x51DE0001u;
    const uint32_t sentinel = 0xCAFEF00Du;
    // First 23 MiB of DRAM are reserved by metal; probe just above the allocator base.
    const uint32_t dram_addr =
        static_cast<uint32_t>(mesh_device->get_devices()[0]->allocator()->get_base_allocator_addr(HalMemType::DRAM)) + 0x1000;

    const experimental::NodeCoord src{0, 0};
    const experimental::NodeCoord dst = (grid.x > 1) ? experimental::NodeCoord{1, 0} : experimental::NodeCoord{0, 1};
    const CoreCoord dst_phys = mesh_device->worker_core_from_logical_core(dst);
    const auto& soc = MetalContext::instance().get_cluster().get_soc_desc(mesh_device->get_devices()[0]->id());
    const CoreCoord dram_core = soc.get_preferred_worker_core_for_dram_view(0, NOC::NOC_0);
    std::cout << "[NOCPROBE] src logical " << src.x << "," << src.y << "  dst logical " << dst.x << "," << dst.y
              << " -> phys " << dst_phys.x << "-" << dst_phys.y << "  dram ch0 core " << dram_core.x << "-"
              << dram_core.y << "  result_addr 0x" << std::hex << result_addr << " dram_addr 0x" << dram_addr
              << std::dec << std::endl;

    std::map<std::pair<uint32_t, uint32_t>, uint32_t> l1zero_before;
    for (uint32_t y = 0; y < grid.y; ++y) {
        for (uint32_t x = 0; x < grid.x; ++x) {
            std::vector<uint32_t> s{sentinel};
            slow_dispatch::WriteToL1(this->device(), experimental::NodeCoord{x, y}, result_addr, s);
            std::vector<uint32_t> z(1, 0);
            slow_dispatch::ReadFromL1(this->device(), experimental::NodeCoord{x, y}, 0, sizeof(uint32_t), z);
            l1zero_before[{x, y}] = z[0];
        }
    }
    {
        std::vector<uint32_t> s{sentinel};
        tt::tt_metal::detail::WriteToDeviceDRAMChannel(mesh_device->get_devices()[0], 0, dram_addr, s);
    }

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    const experimental::KernelSpecName K{"noc_probe"};
    experimental::KernelSpec spec_k{
        .unique_id = K,
        .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/qsr1_noc_unicast_probe.cpp",
        .num_threads = 1,
        .runtime_arg_schema =
            {.runtime_arg_names = {"value", "result_addr", "dst_x", "dst_y", "dram_x", "dram_y", "dram_addr"}},
        .hw_config = experimental::DataMovementGen2Config{},
    };
    experimental::WorkUnitSpec wu{.name = "main", .kernels = {K}, .target_nodes = src};
    experimental::ProgramSpec spec{.name = "qsr1_noc_probe", .kernels = {spec_k}, .work_units = {wu}};
    Program program = experimental::MakeProgramFromSpec(this->device(), spec);
    experimental::ProgramRunArgs params;
    params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = K,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
            src,
            {{"value", value},
             {"result_addr", result_addr},
             {"dst_x", static_cast<uint32_t>(dst_phys.x)},
             {"dst_y", static_cast<uint32_t>(dst_phys.y)},
             {"dram_x", static_cast<uint32_t>(dram_core.x)},
             {"dram_y", static_cast<uint32_t>(dram_core.y)},
             {"dram_addr", dram_addr}})}};
    experimental::SetProgramRunArgs(program, params);
    workload.add_program(device_range, std::move(program));
    // Non-blocking launch: a barrier-less kernel finishes on its own, and if the launch itself never
    // completes we still want to read the tiles. Give the emulator time, then inspect.
    distributed::EnqueueMeshWorkload(cq, workload, false);
    std::this_thread::sleep_for(std::chrono::seconds(60));
    {
        std::vector<uint32_t> f(1, 0);
        slow_dispatch::ReadFromL1(this->device(), src, result_addr + 4, sizeof(uint32_t), f);
        std::cout << "[NOCPROBE] kernel done-flag on SRC = 0x" << std::hex << f[0] << std::dec
                  << (f[0] == 0xD0DE0001u ? "  (kernel ran to completion)" : "  (kernel did NOT reach the end)") << std::endl;
        std::vector<uint32_t> g(4, 0);
        slow_dispatch::ReadFromL1(this->device(), src, result_addr + 8, 4 * sizeof(uint32_t), g);
        std::cout << "[NOCPROBE] firmware self-view on SRC: my_x=" << (g[0] & 0xff) << " my_y=" << ((g[0] >> 8) & 0xff)
                  << " noc_index=" << ((g[0] >> 16) & 0xff) << "  raw NOC_NODE_ID=0x" << std::hex << g[1]
                  << "  programmed DEST_COORD=0x" << g[2] << " (x=" << std::dec << (g[2] & 0x3f) << ",y=" << ((g[2] >> 6) & 0x3f)
                  << ")  DEST_ADDR=0x" << std::hex << g[3] << std::dec << "   [host intended dst phys " << dst_phys.x << "-"
                  << dst_phys.y << "]" << std::endl;
    }

    bool dst_ok = false, others_clean = true, l1zero_ok = true;
    for (uint32_t y = 0; y < grid.y; ++y) {
        for (uint32_t x = 0; x < grid.x; ++x) {
            std::vector<uint32_t> r(1, 0), z(1, 0);
            slow_dispatch::ReadFromL1(this->device(), experimental::NodeCoord{x, y}, result_addr, sizeof(uint32_t), r);
            slow_dispatch::ReadFromL1(this->device(), experimental::NodeCoord{x, y}, 0, sizeof(uint32_t), z);
            const CoreCoord phys = mesh_device->worker_core_from_logical_core(experimental::NodeCoord{x, y});
            const bool is_src = (x == src.x && y == src.y), is_dst = (x == dst.x && y == dst.y);
            std::cout << "[NOCPROBE] node " << x << "," << y << " (phys " << phys.x << "-" << phys.y << ")"
                      << (is_src ? " SRC" : is_dst ? " DST" : "    ") << "  result=0x" << std::hex << r[0]
                      << "  L1[0] before=0x" << l1zero_before[{x, y}] << " after=0x" << z[0] << std::dec << std::endl;
            if (is_dst) {
                dst_ok = (r[0] == value);
            } else if (!is_src && r[0] != sentinel) {
                others_clean = false;
            }
            if (z[0] != l1zero_before[{x, y}]) {
                l1zero_ok = false;
            }
        }
    }
    std::vector<uint32_t> d(1, 0);
    tt::tt_metal::detail::ReadFromDeviceDRAMChannel(mesh_device->get_devices()[0], 0, dram_addr, sizeof(uint32_t), d);
    std::cout << "[NOCPROBE] dram ch0 @0x" << std::hex << dram_addr << " = 0x" << d[0] << std::dec
              << (d[0] == value ? "  (device DRAM write LANDED)" : "  (device DRAM write DID NOT land)") << std::endl;
    std::cout << "[NOCPROBE] VERDICT: L1 unicast " << (dst_ok ? "landed on intended tile" : "MISSED intended tile")
              << "; other tiles " << (others_clean ? "untouched" : "CORRUPTED") << "; L1[0] "
              << (l1zero_ok ? "intact" : "CLOBBERED") << "; DRAM " << (d[0] == value ? "ok" : "MISSED") << std::endl;
    EXPECT_TRUE(dst_ok);
    EXPECT_TRUE(others_clean);
    EXPECT_TRUE(l1zero_ok);
    EXPECT_EQ(d[0], value);
}
