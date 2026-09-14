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
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

// qsr.s1 ATT boot image + map (host-safe constexpr data; also compiled host-side by unit_tests_noc).
#include "internal/tt-2xx/quasar/noc/att/temporary_programming/grendel_qsr1_att_data.h"

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

// =====================================================================================================
// qsr.s1 ATT NoC probe (host side). Kernel: test_kernels/dataflow/qsr1_att_noc_probe.cpp.
//
// One emulator run must localize any remaining NoC failure on the qsr.s1 t6x4 model under the ATT
// transport, so this probe is deliberately verbose: every host read is wrapped so a rejected access is
// reported (never fatal), every result is printed as a `[ATTPROBE]` line, and the final per-step verdict
// table is printed BEFORE any EXPECT fires. If the kernel hangs, the process terminates with
// std::_Exit(1) before fixture teardown (teardown would wait forever on the hung kernel).
//
// Environment knobs:
//   TT_METAL_QSR1_PROBE_TIMEOUT_S      poll timeout for the kernel (default 240)
//   TT_METAL_QSR1_PROBE_POLL_MS        poll interval (default 2000)
//   TT_METAL_QSR1_PROBE_STEP_MASK      bit N set -> run kernel step N (default 0x1FBFE = steps 1..9 and the
//                                      read-size bracket 11..16; steps 0, 10 and 17 are markers and always run)
//   TT_METAL_QSR1_PROBE_MCAST_RECT_COUNT=1  pass num_dests = ATT rectangle count (width*height, what
//                                      noc_nonblocking_api_v3.h ASSERTs) instead of the live-tile count
//   TT_METAL_QSR1_PROBE_BURST_REGS=1   read ATT register ranges as one multi-word read instead of 4 B each
// =====================================================================================================
namespace qsr1_att_probe {

// ---- result block layout (mirrors the kernel) ----
constexpr uint32_t BLOCK_BYTES = 0x400;
constexpr uint32_t OFF_HDR = 0x000;
constexpr uint32_t OFF_STEP = 0x080;
constexpr uint32_t OFF_DST_S3 = 0x240;
constexpr uint32_t OFF_DST_S5 = 0x280;
constexpr uint32_t OFF_LOCAL_S1 = 0x2C0;
constexpr uint32_t OFF_REMOTE_BASE = 0x300;  // remote landing slots: [0x300, 0x400) on every worker
constexpr uint32_t OFF_REMOTE_S2 = 0x300;
constexpr uint32_t OFF_REMOTE_S3_SRC = 0x340;
constexpr uint32_t OFF_REMOTE_S7 = 0x380;
constexpr uint32_t OFF_REMOTE_S8 = 0x3C0;
constexpr uint32_t HDR_MAGIC = 0x00;
constexpr uint32_t HDR_SELF = 0x04;
constexpr uint32_t HDR_NODE_ID = 0x08;
constexpr uint32_t HDR_SEM_ADDR = 0x0C;
constexpr uint32_t HDR_UNICAST_ADDR = 0x10;
constexpr uint32_t HDR_DRAM_GEN_ADDR = 0x18;
constexpr uint32_t HDR_DRAM_BANK_ADDR = 0x20;
constexpr uint32_t HDR_SEM_NOC_ADDR = 0x28;
constexpr uint32_t HDR_MCAST_COL = 0x30;
constexpr uint32_t HDR_MCAST_GRID = 0x38;
constexpr uint32_t HDR_BANK_XY = 0x40;
constexpr uint32_t HDR_BANK_OFF = 0x44;
constexpr uint32_t HDR_S3_VALUE = 0x48;
constexpr uint32_t HDR_S5_VALUE = 0x4C;
constexpr uint32_t HDR_STEP_MASK = 0x50;
constexpr uint32_t HDR_NUM_DRAM_BANKS = 0x54;
constexpr uint32_t HDR_S3_SRC_ADDR = 0x58;
constexpr uint32_t MAGIC = 0xA77B10C0u;
constexpr uint32_t STEP_DONE_TAG = 0xD0DE0000u;
constexpr uint32_t STEP_SKIP_TAG = 0x5C1B0000u;
constexpr uint32_t ORIGINAL_END_STEP = 10;  // unconditional marker: steps 0..9 walked
constexpr uint32_t FINAL_STEP = 17;
constexpr uint32_t NUM_STEPS = FINAL_STEP + 1;
constexpr uint32_t XFER_WORDS = 4;  // 16 B per NoC transfer
constexpr uint32_t L1_SENTINEL = 0xCAFEF00Du;
constexpr uint32_t DRAM_SENTINEL = 0xD4A0F00Du;
constexpr uint32_t DRAM_PROBE_WORDS = 32;  // 128 B seeded/read per DRAM channel
constexpr uint32_t DEFAULT_STEP_MASK = 0x1FBFEu;  // steps 1..9 + read-size bracket 11..16

// ---- read-size bracket (kernel steps 11..16; mirrors the kernel) ----
constexpr uint32_t RD_FIRST_STEP = 11;
constexpr uint32_t RD_STEPS = 6;
constexpr uint32_t RD_L1_STEP = 16;  // 8192 B from the dst worker's L1 instead of DRAM
constexpr uint32_t RD_SIZES[RD_STEPS] = {4096, 8128, 8192, 8256, 16384, 8192};
constexpr uint32_t OFF_RD_REC = 0x400;  // per-step 64 B records on the src worker, [0x400, 0x580)
constexpr uint32_t RD_REC_STRIDE = 0x40;
constexpr uint32_t RD_REC_BYTES = RD_STEPS * RD_REC_STRIDE;
constexpr uint32_t OFF_RD_LANDING = 0x10000;  // per-step 16 KiB landing regions on the src worker, [0x10000, 0x28000)
constexpr uint32_t RD_LANDING_STRIDE = 0x4000;
constexpr uint32_t OFF_RD_L1_SRC = 0x30000;  // 8 KiB host-seeded pattern on every worker (s16 source)
constexpr uint32_t RD_L1_SRC_BYTES = 0x2000;
constexpr uint32_t DRAM_RD_SEED_BYTES = 0x8000;  // 32 KiB of dram_rd_word() at dram_rd_addr on every channel
constexpr uint32_t REC_SIZE = 0x00;
constexpr uint32_t REC_FIRST = 0x04;
constexpr uint32_t REC_LAST = 0x08;
constexpr uint32_t REC_SRC = 0x0C;  // 64-bit (lo, hi)
constexpr uint32_t REC_DST = 0x14;
constexpr uint32_t REC_PHASE = 0x18;
constexpr uint32_t PHASE_PENDING = 0;
constexpr uint32_t PHASE_SKIPPED = 1;
constexpr uint32_t PHASE_ISSUING = 2;
constexpr uint32_t PHASE_ISSUED = 3;
constexpr uint32_t PHASE_DONE = 4;

inline uint32_t payload_word(uint32_t step, uint32_t i) { return 0x51DE0000u | (step << 8) | i; }
inline uint32_t s3_source_word(uint32_t lx, uint32_t ly, uint32_t i) {
    return 0x5EED0000u | (lx << 8) | (ly << 4) | i;
}
// Bracket seed patterns: word index in the low 16 bits so first/last words identify their offset.
inline uint32_t dram_rd_word(uint32_t i) { return 0xDA7A0000u | (i & 0xFFFFu); }
inline uint32_t l1_rd_word(uint32_t lx, uint32_t ly, uint32_t i) {
    return 0xB1000000u | ((lx & 0xFu) << 20) | ((ly & 0xFu) << 16) | (i & 0xFFFFu);
}
inline const char* rd_phase_str(uint32_t p) {
    switch (p) {
        case PHASE_PENDING: return "PENDING(never reached)";
        case PHASE_SKIPPED: return "SKIPPED";
        case PHASE_ISSUING: return "ISSUING(inside noc_async_read)";
        case PHASE_ISSUED: return "ISSUED(inside read barrier)";
        case PHASE_DONE: return "DONE";
        default: return "UNKNOWN";
    }
}

// ---- ATT tile-NIU register map (tile-local byte offsets; host reads route via the per-tile config window)
constexpr uint32_t ATT_ENABLE_TABLES = 0x02010000u;
constexpr uint32_t ATT_LAST_TARG_LO = 0x02010008u;  // last pre-translation target address lo/hi
constexpr uint32_t ATT_LAST_TARG_HI = 0x0201000Cu;
constexpr uint32_t ATT_LAST_RET_LO = 0x02010010u;  // last pre-translation return address lo/hi
constexpr uint32_t ATT_LAST_RET_HI = 0x02010014u;
constexpr uint32_t ATT_LAST_XLAT_BASE = 0x02010018u;  // 4 translated words 0x18..0x24
constexpr uint32_t ATT_LAST_COORDS = 0x02010028u;     // {dest_y[23:18],dest_x[17:12],src_y[11:6],src_x[5:0]}
constexpr uint32_t ATT_DEBUG_MISC = 0x0201002Cu;      // bit 11 no_match, bit 10 more_than_one_match (sticky)
constexpr uint32_t ATT_MASK_SLOT_BASE = 0x02010030u;
constexpr uint32_t ATT_MASK_SLOT_STRIDE = 0x18u;
constexpr uint32_t ATT_ENDPOINT_BASE = 0x02012000u;
constexpr uint32_t ATT_PROBED_SLOTS[] = {2, 4, 5, 13, 14};
constexpr uint32_t ATT_LIVE_OFFSET = 2;  // firmware-visible live node id = descriptor coordinate + (2, 2)

inline std::string hexstr(uint64_t v, int width = 8) {
    std::ostringstream os;
    os << "0x" << std::hex << std::setw(width) << std::setfill('0') << v;
    return os.str();
}

inline uint32_t env_u32(const char* name, uint32_t dflt) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') {
        return dflt;
    }
    return static_cast<uint32_t>(std::strtoul(v, nullptr, 0));
}

inline bool env_flag(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && *v != '\0' && *v != '0';
}

struct WordRead {
    uint32_t addr = 0;
    std::optional<uint32_t> value;
    std::string error;
};

// Single 4-byte host read at a tile-local offset (the proven path for ATT register offsets >= 16 MiB).
inline WordRead read_word(distributed::MeshDevice& dev, const experimental::NodeCoord& node, uint32_t addr) {
    WordRead r;
    r.addr = addr;
    try {
        std::vector<uint32_t> v(1, 0);
        slow_dispatch::ReadFromL1(dev, node, addr, sizeof(uint32_t), v);
        r.value = v.at(0);
    } catch (const std::exception& e) {
        r.error = e.what();
    } catch (...) {
        r.error = "unknown exception";
    }
    return r;
}

inline std::vector<WordRead> read_words(
    distributed::MeshDevice& dev, const experimental::NodeCoord& node, uint32_t addr, uint32_t n) {
    std::vector<WordRead> out;
    out.reserve(n);
    if (env_flag("TT_METAL_QSR1_PROBE_BURST_REGS")) {
        std::vector<uint32_t> v(n, 0);
        std::string err;
        try {
            slow_dispatch::ReadFromL1(dev, node, addr, n * sizeof(uint32_t), v);
        } catch (const std::exception& e) {
            err = e.what();
        } catch (...) {
            err = "unknown exception";
        }
        for (uint32_t i = 0; i < n; ++i) {
            WordRead r;
            r.addr = addr + 4 * i;
            if (err.empty()) {
                r.value = v.at(i);
            } else {
                r.error = err;
            }
            out.push_back(r);
        }
        return out;
    }
    for (uint32_t i = 0; i < n; ++i) {
        out.push_back(read_word(dev, node, addr + 4 * i));
    }
    return out;
}

// Multi-word read of an ordinary L1 data region (result blocks), wrapped.
inline std::optional<std::vector<uint32_t>> read_block(
    distributed::MeshDevice& dev, const experimental::NodeCoord& node, uint32_t addr, uint32_t nwords, std::string& err) {
    try {
        std::vector<uint32_t> v(nwords, 0);
        slow_dispatch::ReadFromL1(dev, node, addr, nwords * sizeof(uint32_t), v);
        return v;
    } catch (const std::exception& e) {
        err = e.what();
    } catch (...) {
        err = "unknown exception";
    }
    return std::nullopt;
}

// Expected register image: base tile image, then the delivery overrides (later write wins), exactly as
// the boot replay applies them. Endpoint row 256 is boot-patched per tile and handled by the caller.
inline const std::map<uint32_t, uint32_t>& expected_image() {
    static const std::map<uint32_t, uint32_t> image = [] {
        std::map<uint32_t, uint32_t> m;
        for (const auto& w : grendel_qsr1_att_program::GRENDEL_QSR1_TILE_ATT_WRITES) {
            m[w.address] = w.data;
        }
        for (const auto& w : grendel_qsr1_att_program::GRENDEL_QSR1_DELIVERY_OVERRIDES) {
            m[w.address] = w.data;
        }
        return m;
    }();
    return image;
}

struct CtlFields {
    uint32_t mask, ep_idx, ep_size, tbl_off, translate;
};
inline CtlFields decode_ctl(uint32_t ctl) {
    return {ctl & 0x3fu, (ctl >> 6) & 0x3fu, (ctl >> 12) & 0x3fu, (ctl >> 18) & 0x3ffu, (ctl >> 28) & 0x1u};
}
inline std::string ctl_str(const CtlFields& f) {
    std::ostringstream os;
    os << "{mask=" << f.mask << " ep_idx=" << f.ep_idx << " ep_size=" << f.ep_size << " tbl_off=" << f.tbl_off
       << " translate=" << f.translate << "}";
    return os.str();
}

// Endpoint word (y << 6) | x in the live frame, plus the descriptor coordinate it names.
inline std::string ep_word_str(uint32_t w) {
    const uint32_t x = w & 0x3fu, y = (w >> 6) & 0x3fu;
    std::ostringstream os;
    os << "live(" << x << "," << y << ")";
    if (x >= ATT_LIVE_OFFSET && y >= ATT_LIVE_OFFSET) {
        os << "=desc " << (x - ATT_LIVE_OFFSET) << "-" << (y - ATT_LIVE_OFFSET);
    }
    return os.str();
}

inline uint32_t live_id_of(const CoreCoord& virt) {
    return ((static_cast<uint32_t>(virt.y) + ATT_LIVE_OFFSET) << 6) | (static_cast<uint32_t>(virt.x) + ATT_LIVE_OFFSET);
}

// Which ATT window (if any) a 64-bit NoC operand falls in, and what it selects. `size` is the transfer the
// operand will carry (default: the 16 B probe transfer) for the in-window check.
inline std::string decode_operand(uint64_t a, uint32_t size = XFER_WORDS * sizeof(uint32_t)) {
    namespace cfg = grendel_qsr1_att_config;
    struct Entry {
        const char* name;
        const noc_att::Window* w;
    };
    const Entry entries[] = {
        {"LOOPBACK_SCRATCH(slot13)", &cfg::LOOPBACK_SCRATCH_WINDOW},
        {"WORKER(slot4)", &cfg::WORKER_WINDOW},
        {"DRAM(slot5)", &cfg::DRAM_WINDOW},
        {"TILE(slot14)", &cfg::TILE_WINDOW},
    };
    std::ostringstream os;
    bool any = false;
    for (const auto& e : entries) {
        if (!e.w->matches(a)) {
            continue;
        }
        any = true;
        const uint32_t sel = e.w->selector(a);
        const uint32_t row = e.w->endpoint_index(a);
        const uint64_t local = e.w->local_address(a);
        os << e.name << " selector " << sel << " -> endpoint row " << row << ", local " << hexstr(local, 9);
        if (e.w == &cfg::WORKER_WINDOW && sel < std::size(cfg::ATT_WORKER_ENDPOINT_WORDS)) {
            os << " = worker " << ep_word_str(cfg::ATT_WORKER_ENDPOINT_WORDS[sel]);
        } else if (e.w == &cfg::TILE_WINDOW && sel < std::size(cfg::ATT_FULL_TILE_ENDPOINT_WORDS)) {
            os << " = tile " << ep_word_str(cfg::ATT_FULL_TILE_ENDPOINT_WORDS[sel]);
        } else if (e.w == &cfg::DRAM_WINDOW) {
            os << " = DRAM channel " << sel;
        }
        if (!e.w->transfer_supported(local, size)) {
            os << " [TRANSFER OF " << size << " B OUTSIDE WINDOW]";
        }
        os << "; ";
    }
    if (!any) {
        os << "NO ATT WINDOW MATCHES (raw/XY-style operand " << hexstr(a, 16) << ")";
    }
    return os.str();
}

// The packed software multicast descriptor get_noc_multicast_addr produces under ATT, and what the V3
// issue path resolves it to (rectangle_count is what noc_nonblocking_api_v3.h ASSERTs num_dests equals).
inline std::string decode_mcast_descriptor(uint64_t d) {
    constexpr uint32_t LB = noc_att::DESCRIPTOR_LOCAL_BITS;
    constexpr uint32_t NB = noc_att::DESCRIPTOR_NODE_BITS;
    const uint32_t end_x = (d >> LB) & 0x3fu;
    const uint32_t end_y = (d >> (LB + NB)) & 0x3fu;
    const uint32_t start_x = (d >> (LB + 2 * NB)) & 0x3fu;
    const uint32_t start_y = (d >> (LB + 3 * NB)) & 0x3fu;
    const uint64_t local = d & ((uint64_t{1} << LB) - 1);
    const noc_att::NocMulticastAddress m =
        noc_att::resolve_worker_multicast<grendel_qsr1_att_config::MAP>(d, XFER_WORDS * sizeof(uint32_t));
    std::ostringstream os;
    os << "rect (" << start_x << "," << start_y << ")-(" << end_x << "," << end_y << ") local " << hexstr(local, 9)
       << " -> V3 rectangle_count=" << m.rectangle_count << " extent_xy=" << hexstr(m.extent_xy) << " start "
       << (m.rectangle_count ? decode_operand(m.start_address) : std::string("INVALID RECTANGLE (would __builtin_trap)"));
    return os.str();
}

struct DumpStats {
    uint32_t checks = 0, matches = 0, mismatches = 0, rejected = 0, unchecked = 0;
};

inline void report(
    DumpStats& st,
    const std::string& tag,
    const std::string& name,
    const WordRead& r,
    std::optional<uint32_t> expected,
    const std::string& extra = "") {
    std::ostringstream os;
    os << "[ATTPROBE] " << tag << " " << name << " @" << hexstr(r.addr) << " = ";
    if (!r.value.has_value()) {
        ++st.rejected;
        os << "REJECTED(" << r.error << ")";
    } else {
        os << hexstr(*r.value);
        if (expected.has_value()) {
            ++st.checks;
            os << " expected " << hexstr(*expected);
            if (*r.value == *expected) {
                ++st.matches;
                os << " MATCH";
            } else {
                ++st.mismatches;
                os << " MISMATCH";
            }
        } else {
            ++st.unchecked;
            os << " (no expectation)";
        }
    }
    if (!extra.empty()) {
        os << "  " << extra;
    }
    std::cout << os.str() << std::endl;
}

inline std::string debug_misc_str(uint32_t v) {
    std::ostringstream os;
    os << "no_match(bit11)=" << ((v >> 11) & 1u) << " more_than_one_match(bit10)=" << ((v >> 10) & 1u);
    return os.str();
}

inline std::string last_coords_str(uint32_t v) {
    std::ostringstream os;
    os << "src(" << (v & 0x3fu) << "," << ((v >> 6) & 0x3fu) << ") dest(" << ((v >> 12) & 0x3fu) << ","
       << ((v >> 18) & 0x3fu) << ") [live frame]";
    return os.str();
}

// STAGE 0 / post-run register dump of one live tile. full=true reads the whole probe set (enable, debug,
// last-address regs, mask slots 2/4/5/13/14, endpoint rows 96..159, 256, 312..315); full=false reads
// only DEBUG_MISC + the last-address debug registers.
inline DumpStats dump_att_registers(
    distributed::MeshDevice& dev,
    const experimental::NodeCoord& node,
    const CoreCoord& virt,
    const std::string& phase,
    bool full) {
    namespace cfg = grendel_qsr1_att_config;
    DumpStats st;
    std::ostringstream tg;
    tg << phase << " tile " << virt.x << "-" << virt.y << " (logical " << node.x << "," << node.y << ")";
    const std::string tag = tg.str();
    const auto& image = expected_image();
    auto expect_of = [&](uint32_t addr) -> std::optional<uint32_t> {
        auto it = image.find(addr);
        return it == image.end() ? std::nullopt : std::optional<uint32_t>(it->second);
    };

    if (full) {
        report(st, tag, "ENABLE_TABLES", read_word(dev, node, ATT_ENABLE_TABLES), expect_of(ATT_ENABLE_TABLES));
    }
    {
        const WordRead r = read_word(dev, node, ATT_DEBUG_MISC);
        report(st, tag, "DEBUG_MISC", r, std::nullopt, r.value ? debug_misc_str(*r.value) : "");
    }
    {
        // Last-address debug registers, contiguous 0x08..0x28: read as one 9-word range, labelled by address.
        const std::pair<uint32_t, const char*> last_regs[] = {
            {ATT_LAST_TARG_LO, "LAST_TARG_LO(pre)"},   {ATT_LAST_TARG_HI, "LAST_TARG_HI(pre)"},
            {ATT_LAST_RET_LO, "LAST_RET_LO(pre)"},     {ATT_LAST_RET_HI, "LAST_RET_HI(pre)"},
            {ATT_LAST_XLAT_BASE + 0x0, "LAST_XLAT_0"}, {ATT_LAST_XLAT_BASE + 0x4, "LAST_XLAT_1"},
            {ATT_LAST_XLAT_BASE + 0x8, "LAST_XLAT_2"}, {ATT_LAST_XLAT_BASE + 0xC, "LAST_XLAT_3"},
            {ATT_LAST_COORDS, "LAST_COORDS"},
        };
        const auto regs = read_words(dev, node, ATT_LAST_TARG_LO, std::size(last_regs));
        for (uint32_t i = 0; i < regs.size(); ++i) {
            std::string extra;
            const char* name = "LAST_?";
            for (const auto& [addr, nm] : last_regs) {
                if (addr == regs[i].addr) {
                    name = nm;
                }
            }
            if (regs[i].addr == ATT_LAST_COORDS && regs[i].value) {
                extra = last_coords_str(*regs[i].value);
            }
            report(st, tag, name, regs[i], std::nullopt, extra);
        }
        if (regs[0].value && regs[1].value) {
            const uint64_t pre = (uint64_t{*regs[1].value} << 32) | *regs[0].value;
            std::cout << "[ATTPROBE] " << tag << " last pre-translation target " << hexstr(pre, 16) << " -> "
                      << decode_operand(pre) << std::endl;
        }
    }
    if (!full) {
        return st;
    }

    struct SlotWindow {
        uint32_t slot;
        const noc_att::Window* w;
        const char* name;
    };
    const SlotWindow slot_windows[] = {
        {4, &cfg::WORKER_WINDOW, "WORKER_WINDOW"},
        {5, &cfg::DRAM_WINDOW, "DRAM_WINDOW"},
        {13, &cfg::LOOPBACK_SCRATCH_WINDOW, "LOOPBACK_SCRATCH_WINDOW"},
        {14, &cfg::TILE_WINDOW, "TILE_WINDOW"},
    };
    for (uint32_t slot : ATT_PROBED_SLOTS) {
        const uint32_t base = ATT_MASK_SLOT_BASE + ATT_MASK_SLOT_STRIDE * slot;
        const std::string sn = "SLOT" + std::to_string(slot);
        const WordRead ctl = read_word(dev, node, base + 0x00);
        const WordRead ep_lo = read_word(dev, node, base + 0x08);
        const WordRead ep_hi = read_word(dev, node, base + 0x0C);
        const WordRead bar_lo = read_word(dev, node, base + 0x10);
        const WordRead bar_hi = read_word(dev, node, base + 0x14);
        report(st, tag, sn + ".CTL", ctl, expect_of(base + 0x00), ctl.value ? ctl_str(decode_ctl(*ctl.value)) : "");
        report(st, tag, sn + ".EP_LO", ep_lo, expect_of(base + 0x08));
        report(st, tag, sn + ".EP_HI", ep_hi, expect_of(base + 0x0C));
        report(st, tag, sn + ".BAR_LO", bar_lo, expect_of(base + 0x10));
        report(st, tag, sn + ".BAR_HI", bar_hi, expect_of(base + 0x14));
        for (const auto& sw : slot_windows) {
            if (sw.slot != slot || !ctl.value) {
                continue;
            }
            const CtlFields f = decode_ctl(*ctl.value);
            const bool fields_ok = f.mask == sw.w->mask_bits && f.ep_idx == sw.w->endpoint_shift &&
                                   f.ep_size == sw.w->endpoint_size && f.tbl_off == sw.w->endpoint_table_offset &&
                                   f.translate == (sw.w->translate_address ? 1u : 0u);
            ++st.checks;
            if (fields_ok) {
                ++st.matches;
            } else {
                ++st.mismatches;
            }
            std::cout << "[ATTPROBE] " << tag << " " << sn << ".CTL fields " << ctl_str(f) << " vs map " << sw.name
                      << " {mask=" << unsigned(sw.w->mask_bits) << " ep_idx=" << unsigned(sw.w->endpoint_shift)
                      << " ep_size=" << unsigned(sw.w->endpoint_size) << " tbl_off=" << sw.w->endpoint_table_offset
                      << " translate=" << (sw.w->translate_address ? 1 : 0) << "} " << (fields_ok ? "MATCH" : "MISMATCH")
                      << std::endl;
            if (ep_lo.value && ep_hi.value) {
                const uint64_t ep = (uint64_t{*ep_hi.value} << 32) | *ep_lo.value;
                const bool cmp_ok = ep == sw.w->compare;
                ++st.checks;
                if (cmp_ok) {
                    ++st.matches;
                } else {
                    ++st.mismatches;
                }
                std::cout << "[ATTPROBE] " << tag << " " << sn << ".EP(compare) " << hexstr(ep, 16) << " vs map "
                          << hexstr(sw.w->compare, 16) << " " << (cmp_ok ? "MATCH" : "MISMATCH") << std::endl;
            }
        }
    }

    auto dump_rows = [&](uint32_t first, uint32_t count, const char* label) {
        const auto words = read_words(dev, node, ATT_ENDPOINT_BASE + 4 * first, count);
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t row = first + i;
            const uint32_t addr = ATT_ENDPOINT_BASE + 4 * row;
            std::optional<uint32_t> exp = expect_of(addr);
            std::string extra;
            if (addr == grendel_qsr1_att_program::GRENDEL_ATT_LOOPBACK_EP256_ADDR) {
                exp = live_id_of(virt);
                extra = "(boot-patched self endpoint)";
            } else if (row >= 128 && row < 128 + std::size(cfg::ATT_WORKER_ENDPOINT_WORDS)) {
                extra = "worker selector " + std::to_string(row - 128) + " " +
                        ep_word_str(cfg::ATT_WORKER_ENDPOINT_WORDS[row - 128]);
            } else if (row >= 256 && row < 256 + std::size(cfg::ATT_FULL_TILE_ENDPOINT_WORDS)) {
                extra = "full-tile selector " + std::to_string(row - 256) + " " +
                        ep_word_str(cfg::ATT_FULL_TILE_ENDPOINT_WORDS[row - 256]);
            } else if (row >= 96 && row < 100) {
                extra = "DRAM channel " + std::to_string(row - 96) + " ingress";
            }
            if (words[i].value) {
                extra += (extra.empty() ? "" : " ") + std::string("read=") + ep_word_str(*words[i].value);
            }
            report(st, tag, std::string(label) + ".ROW" + std::to_string(row), words[i], exp, extra);
        }
    };
    dump_rows(96, 32, "EP");   // DRAM window rows 96..99 + 100..127
    dump_rows(128, 32, "EP");  // worker window rows 128..159
    dump_rows(256, 1, "EP");   // loopback ep256 (boot-patched self)
    dump_rows(312, 4, "EP");   // full-tile rows 312..315

    // The two headline checks the orchestrator asked for, as one-line summaries.
    {
        const uint32_t exp_rows[4] = {0x246u, 0x24au, 0x089u, 0x085u};
        const auto rows = read_words(dev, node, ATT_ENDPOINT_BASE + 4 * 96, 4);
        bool ok = true;
        std::ostringstream os;
        for (uint32_t i = 0; i < 4; ++i) {
            ok = ok && rows[i].value && *rows[i].value == exp_rows[i];
            os << (rows[i].value ? hexstr(*rows[i].value, 3) : std::string("REJECTED")) << (i < 3 ? "," : "");
        }
        std::cout << "[ATTPROBE] " << tag << " SUMMARY rows 96..99 = {" << os.str()
                  << "} expected {0x246,0x24a,0x089,0x085} " << (ok ? "MATCH" : "MISMATCH") << std::endl;
        const WordRead ctl5 = read_word(dev, node, ATT_MASK_SLOT_BASE + ATT_MASK_SLOT_STRIDE * 5);
        const bool ctl5_ok = ctl5.value && decode_ctl(*ctl5.value).ep_idx == 33 &&
                             decode_ctl(*ctl5.value).ep_size == 5 && decode_ctl(*ctl5.value).tbl_off == 96;
        std::cout << "[ATTPROBE] " << tag << " SUMMARY slot 5 ctl "
                  << (ctl5.value ? ctl_str(decode_ctl(*ctl5.value)) : std::string("REJECTED"))
                  << " expected {ep_idx=33 ep_size=5 tbl_off=96} " << (ctl5_ok ? "MATCH" : "MISMATCH") << std::endl;
    }
    std::cout << "[ATTPROBE] " << tag << " DUMP STATS checks=" << st.checks << " matches=" << st.matches
              << " mismatches=" << st.mismatches << " rejected=" << st.rejected << " unchecked=" << st.unchecked
              << std::endl;
    return st;
}

struct Verdict {
    uint32_t step;
    std::string marker;     // DONE / SKIP / MISSING
    std::string operation;
    std::string result;     // PASS / FAIL / N/A
    std::string detail;
    bool counts_as_failure;
};

}  // namespace qsr1_att_probe

// STAGE 0 only: dump the ATT tables of the two column tiles (logical {0,0} = desc 2-2, {1,0} = desc 2-5)
// and compare them to the boot image metal was built against. No kernel, so a boot-table drift check
// costs one short emulator session.
TEST_F(QuasarMeshDeviceSingleCardFixture, Qsr1AttRegisterDump) {
    using namespace qsr1_att_probe;
    if (std::getenv("TT_METAL_SIMULATOR") == nullptr) {
        GTEST_SKIP() << "This test can only be run using a simulator.";
    }
    auto mesh_device = devices_[0];
    const auto grid = mesh_device->compute_with_storage_grid_size();
    std::cout << "[ATTPROBE] compute grid " << grid.x << "x" << grid.y << std::endl;
    std::vector<experimental::NodeCoord> tiles{experimental::NodeCoord{0, 0}};
    if (grid.x > 1) {
        tiles.push_back(experimental::NodeCoord{1, 0});
    } else if (grid.y > 1) {
        tiles.push_back(experimental::NodeCoord{0, 1});
    }
    DumpStats total;
    for (const auto& node : tiles) {
        const CoreCoord virt = mesh_device->worker_core_from_logical_core(node);
        const DumpStats st = dump_att_registers(this->device(), node, virt, "STAGE0", /*full=*/true);
        total.checks += st.checks;
        total.matches += st.matches;
        total.mismatches += st.mismatches;
        total.rejected += st.rejected;
        total.unchecked += st.unchecked;
    }
    std::cout << "[ATTPROBE] REGISTER DUMP TOTAL checks=" << total.checks << " matches=" << total.matches
              << " mismatches=" << total.mismatches << " rejected=" << total.rejected << std::endl;
    EXPECT_EQ(total.mismatches, 0u) << "ATT boot tables drifted from grendel_qsr1_att_data.h";
    EXPECT_EQ(total.rejected, 0u) << "host could not read some ATT registers";
}

// Full probe: STAGE 0 register dump, then one DM kernel on logical {0,0} walking the NoC operation
// classes step by step (see the kernel header for the step table), host polling the per-step markers,
// then a complete read-back of every worker's landing slots, DRAM, the remote semaphore, the operands
// the kernel resolved, and the ATT debug registers on every live tile, ending in a per-step VERDICT table.
TEST_F(QuasarMeshDeviceSingleCardFixture, Qsr1AttNocProbe) {
    using namespace qsr1_att_probe;
    using experimental::NodeCoord;
    if (std::getenv("TT_METAL_SIMULATOR") == nullptr) {
        GTEST_SKIP() << "This test can only be run using a simulator.";
    }
    auto mesh_device = devices_[0];
    distributed::MeshDevice& dev = this->device();
    IDevice* dev0 = mesh_device->get_devices()[0];
    const auto grid = mesh_device->compute_with_storage_grid_size();
    if (grid.x * grid.y < 2) {
        GTEST_SKIP() << "needs at least two worker nodes (got " << grid.x << "x" << grid.y << ")";
    }

    // ---- topology --------------------------------------------------------------------------------
    std::vector<NodeCoord> nodes;
    std::map<std::pair<uint32_t, uint32_t>, CoreCoord> virt_of;
    for (uint32_t y = 0; y < grid.y; ++y) {
        for (uint32_t x = 0; x < grid.x; ++x) {
            const NodeCoord n{x, y};
            nodes.push_back(n);
            virt_of[{x, y}] = mesh_device->worker_core_from_logical_core(n);
        }
    }
    const NodeCoord src{0, 0};
    const NodeCoord dst = (grid.x > 1) ? NodeCoord{1, 0} : NodeCoord{0, 1};
    const CoreCoord src_v = virt_of[{0, 0}];
    const CoreCoord dst_v = virt_of[{static_cast<uint32_t>(dst.x), static_cast<uint32_t>(dst.y)}];
    std::cout << "[ATTPROBE] compute grid " << grid.x << "x" << grid.y << "  src logical " << src.x << "," << src.y
              << " -> virtual " << src_v.x << "-" << src_v.y << "  dst logical " << dst.x << "," << dst.y
              << " -> virtual " << dst_v.x << "-" << dst_v.y << std::endl;
    for (const auto& n : nodes) {
        const CoreCoord v = virt_of[{static_cast<uint32_t>(n.x), static_cast<uint32_t>(n.y)}];
        std::cout << "[ATTPROBE] worker logical " << n.x << "," << n.y << " -> virtual " << v.x << "-" << v.y
                  << "  live id " << hexstr(live_id_of(v), 3) << std::endl;
    }
    const int num_dram_ch = dev0->num_dram_channels();
    const auto& soc = MetalContext::instance().get_cluster().get_soc_desc(dev0->id());
    for (int ch = 0; ch < num_dram_ch; ++ch) {
        const CoreCoord c = soc.get_preferred_worker_core_for_dram_view(ch, NOC::NOC_0);
        std::cout << "[ATTPROBE] DRAM channel " << ch << " preferred core " << c.x << "-" << c.y << std::endl;
    }

    const uint32_t result_addr = MetalContext::instance().hal().get_dev_addr(
                                     HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED) +
                                 0x4000;
    const uint32_t dram_base = static_cast<uint32_t>(dev0->allocator()->get_base_allocator_addr(HalMemType::DRAM));
    const uint32_t dram_addr = (dram_base + 0x4000u + 0xFFFu) & ~0xFFFu;  // s4/s5, 4 KiB aligned
    const uint32_t dram_addr_alt = dram_addr + 64;                        // s9
    const uint32_t dram_rd_addr = dram_addr + 0x1000u;  // s11..s15 source: 32 KiB seeded per channel, 4 KiB aligned
    const uint32_t bank_id = 0;
    const uint32_t step_mask = env_u32("TT_METAL_QSR1_PROBE_STEP_MASK", DEFAULT_STEP_MASK);
    const uint32_t timeout_s = env_u32("TT_METAL_QSR1_PROBE_TIMEOUT_S", 240);
    const uint32_t poll_ms = env_u32("TT_METAL_QSR1_PROBE_POLL_MS", 2000);
    const bool mcast_rect_count = env_flag("TT_METAL_QSR1_PROBE_MCAST_RECT_COUNT");
    std::cout << "[ATTPROBE] result_addr " << hexstr(result_addr) << "  dram_addr " << hexstr(dram_addr) << " (alt "
              << hexstr(dram_addr_alt) << ", allocator base " << hexstr(dram_base) << ")  bank_id " << bank_id
              << "  step_mask " << hexstr(step_mask, 5) << "  timeout " << timeout_s << " s  poll " << poll_ms << " ms"
              << std::endl;
    std::cout << "[ATTPROBE] read-size bracket: dram_rd_addr " << hexstr(dram_rd_addr) << " (seed "
              << DRAM_RD_SEED_BYTES << " B per channel)  records " << hexstr(result_addr + OFF_RD_REC) << "  landing ["
              << hexstr(result_addr + OFF_RD_LANDING) << ", "
              << hexstr(result_addr + OFF_RD_LANDING + RD_STEPS * RD_LANDING_STRIDE) << ")  l1 src "
              << hexstr(result_addr + OFF_RD_L1_SRC) << " (" << RD_L1_SRC_BYTES << " B on every worker)" << std::endl;

    // Multicast rectangles in virtual coordinates: bounding box of the member workers.
    struct Rect {
        uint32_t x0, y0, x1, y1;
    };
    auto bbox = [&](const std::vector<NodeCoord>& members) {
        Rect r{UINT32_MAX, UINT32_MAX, 0, 0};
        for (const auto& m : members) {
            const CoreCoord v = virt_of[{static_cast<uint32_t>(m.x), static_cast<uint32_t>(m.y)}];
            r.x0 = std::min<uint32_t>(r.x0, v.x);
            r.y0 = std::min<uint32_t>(r.y0, v.y);
            r.x1 = std::max<uint32_t>(r.x1, v.x);
            r.y1 = std::max<uint32_t>(r.y1, v.y);
        }
        return r;
    };
    auto live_in = [&](const Rect& r, bool exclude_src) {
        uint32_t n = 0;
        for (const auto& m : nodes) {
            const CoreCoord v = virt_of[{static_cast<uint32_t>(m.x), static_cast<uint32_t>(m.y)}];
            const bool inside = v.x >= r.x0 && v.x <= r.x1 && v.y >= r.y0 && v.y <= r.y1;
            const bool is_src = (m.x == src.x && m.y == src.y);
            if (inside && !(exclude_src && is_src)) {
                ++n;
            }
        }
        return n;
    };
    const Rect col = bbox({src, dst});
    const Rect full = bbox(nodes);
    const uint32_t col_rect_count = (col.x1 - col.x0 + 1) * (col.y1 - col.y0 + 1);
    const uint32_t full_rect_count = (full.x1 - full.x0 + 1) * (full.y1 - full.y0 + 1);
    const uint32_t col_live = live_in(col, /*exclude_src=*/true);
    const uint32_t full_live = live_in(full, /*exclude_src=*/true);
    const uint32_t col_num_dests = mcast_rect_count ? col_rect_count : col_live;
    const uint32_t grid_num_dests = mcast_rect_count ? full_rect_count : full_live;
    std::cout << "[ATTPROBE] s7 column rect (" << col.x0 << "," << col.y0 << ")-(" << col.x1 << "," << col.y1
              << ") rectangle tiles " << col_rect_count << " live-excluding-self " << col_live << " -> num_dests "
              << col_num_dests << std::endl;
    std::cout << "[ATTPROBE] s8 grid rect (" << full.x0 << "," << full.y0 << ")-(" << full.x1 << "," << full.y1
              << ") rectangle tiles " << full_rect_count << " live-excluding-self " << full_live << " -> num_dests "
              << grid_num_dests << std::endl;
    std::cout << "[ATTPROBE] NOTE noc_nonblocking_api_v3.h ASSERTs num_dests == rectangle tiles (" << col_rect_count
              << "/" << full_rect_count << "); with a watcher build the live-count default trips that assert. "
                 "Set TT_METAL_QSR1_PROBE_MCAST_RECT_COUNT=1 to pass the rectangle count instead."
              << std::endl;

    // ---- STAGE 0: ATT registers on the two column tiles, before any kernel --------------------------
    DumpStats stage0;
    for (const NodeCoord& n : {src, dst}) {
        const DumpStats st = dump_att_registers(
            dev, n, virt_of[{static_cast<uint32_t>(n.x), static_cast<uint32_t>(n.y)}], "STAGE0", /*full=*/true);
        stage0.checks += st.checks;
        stage0.matches += st.matches;
        stage0.mismatches += st.mismatches;
        stage0.rejected += st.rejected;
    }
    std::cout << "[ATTPROBE] STAGE0 TOTAL checks=" << stage0.checks << " matches=" << stage0.matches
              << " mismatches=" << stage0.mismatches << " rejected=" << stage0.rejected << std::endl;

    // ---- seed sentinels ----------------------------------------------------------------------------
    // Each worker: [0x000, 0x300) zeroed (header/markers/local slots), [0x300, 0x400) landing slots =
    // L1_SENTINEL, except the s3 source slot which carries a per-tile word so the read tells us which
    // tile it really came from. Each DRAM channel: 128 B of DRAM_SENTINEL at dram_addr.
    for (const auto& n : nodes) {
        const uint32_t lx = static_cast<uint32_t>(n.x), ly = static_cast<uint32_t>(n.y);
        std::vector<uint32_t> zero(OFF_REMOTE_BASE / sizeof(uint32_t), 0u);
        std::vector<uint32_t> landing((BLOCK_BYTES - OFF_REMOTE_BASE) / sizeof(uint32_t), L1_SENTINEL);
        for (uint32_t i = 0; i < XFER_WORDS; ++i) {
            landing[(OFF_REMOTE_S3_SRC - OFF_REMOTE_BASE) / sizeof(uint32_t) + i] = s3_source_word(lx, ly, i);
        }
        try {
            slow_dispatch::WriteToL1(dev, n, result_addr, zero);
            slow_dispatch::WriteToL1(dev, n, result_addr + OFF_REMOTE_BASE, landing);
        } catch (const std::exception& e) {
            std::cout << "[ATTPROBE] seed of worker " << lx << "," << ly << " REJECTED: " << e.what() << std::endl;
        }
    }
    for (int ch = 0; ch < num_dram_ch; ++ch) {
        std::vector<uint32_t> s(DRAM_PROBE_WORDS, DRAM_SENTINEL);
        try {
            tt::tt_metal::detail::WriteToDeviceDRAMChannel(dev0, ch, dram_addr, s);
        } catch (const std::exception& e) {
            std::cout << "[ATTPROBE] seed of DRAM channel " << ch << " REJECTED: " << e.what() << std::endl;
        }
    }
    // Read-size bracket seeds: 32 KiB of dram_rd_word() at dram_rd_addr on every channel (s11..s15 source, whichever
    // channel bank 0 resolves to), 8 KiB of a per-tile l1_rd_word() at OFF_RD_L1_SRC on every worker (s16 source),
    // and on the src worker the record block + the six landing regions zeroed so delivered words can be counted.
    {
        std::vector<uint32_t> s(DRAM_RD_SEED_BYTES / sizeof(uint32_t));
        for (uint32_t i = 0; i < s.size(); ++i) {
            s[i] = dram_rd_word(i);
        }
        for (int ch = 0; ch < num_dram_ch; ++ch) {
            try {
                tt::tt_metal::detail::WriteToDeviceDRAMChannel(dev0, ch, dram_rd_addr, s);
            } catch (const std::exception& e) {
                std::cout << "[ATTPROBE] bracket seed of DRAM channel " << ch << " REJECTED: " << e.what() << std::endl;
            }
        }
    }
    for (const auto& n : nodes) {
        const uint32_t lx = static_cast<uint32_t>(n.x), ly = static_cast<uint32_t>(n.y);
        std::vector<uint32_t> l1src(RD_L1_SRC_BYTES / sizeof(uint32_t));
        for (uint32_t i = 0; i < l1src.size(); ++i) {
            l1src[i] = l1_rd_word(lx, ly, i);
        }
        try {
            slow_dispatch::WriteToL1(dev, n, result_addr + OFF_RD_L1_SRC, l1src);
        } catch (const std::exception& e) {
            std::cout << "[ATTPROBE] bracket L1 seed of worker " << lx << "," << ly << " REJECTED: " << e.what()
                      << std::endl;
        }
    }
    {
        std::vector<uint32_t> zero_rec(RD_REC_BYTES / sizeof(uint32_t), 0u);
        std::vector<uint32_t> zero_landing(RD_STEPS * RD_LANDING_STRIDE / sizeof(uint32_t), 0u);
        try {
            slow_dispatch::WriteToL1(dev, src, result_addr + OFF_RD_REC, zero_rec);
            slow_dispatch::WriteToL1(dev, src, result_addr + OFF_RD_LANDING, zero_landing);
        } catch (const std::exception& e) {
            std::cout << "[ATTPROBE] bracket record/landing zeroing on src REJECTED: " << e.what() << std::endl;
        }
    }

    // ---- program -----------------------------------------------------------------------------------
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    const experimental::KernelSpecName K{"att_noc_probe"};
    const experimental::SemaphoreSpecName SEM{"probe_sem"};
    experimental::SemaphoreSpec sem_spec{
        .unique_id = SEM,
        .target_nodes = experimental::NodeRange(NodeCoord{0, 0}, NodeCoord{grid.x - 1, grid.y - 1}),
    };
    experimental::KernelSpec kspec{
        .unique_id = K,
        .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/qsr1_att_noc_probe.cpp",
        .num_threads = 1,
        .semaphore_bindings = {{.semaphore_spec_name = SEM, .accessor_name = "probe_sem"}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"result_addr", "dst_x",   "dst_y",   "dram_bank_id",  "dram_addr", "dram_addr_alt",
                  "col_x0",      "col_y0",  "col_x1",  "col_y1",        "col_num_dests",
                  "grid_x0",     "grid_y0", "grid_x1", "grid_y1",       "grid_num_dests", "step_mask",
                  "dram_rd_addr"}},
        .hw_config = experimental::DataMovementGen2Config{},
    };
    experimental::WorkUnitSpec wu{.name = "main", .kernels = {K}, .target_nodes = src};
    experimental::ProgramSpec spec{
        .name = "qsr1_att_noc_probe", .kernels = {kspec}, .semaphores = {sem_spec}, .work_units = {wu}};
    Program program = experimental::MakeProgramFromSpec(this->device(), spec);
    experimental::ProgramRunArgs params;
    params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = K,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
            src,
            {{"result_addr", result_addr},
             {"dst_x", static_cast<uint32_t>(dst_v.x)},
             {"dst_y", static_cast<uint32_t>(dst_v.y)},
             {"dram_bank_id", bank_id},
             {"dram_addr", dram_addr},
             {"dram_addr_alt", dram_addr_alt},
             {"col_x0", col.x0},
             {"col_y0", col.y0},
             {"col_x1", col.x1},
             {"col_y1", col.y1},
             {"col_num_dests", col_num_dests},
             {"grid_x0", full.x0},
             {"grid_y0", full.y0},
             {"grid_x1", full.x1},
             {"grid_y1", full.y1},
             {"grid_num_dests", grid_num_dests},
             {"step_mask", step_mask},
             {"dram_rd_addr", dram_rd_addr}})}};
    experimental::SetProgramRunArgs(program, params);
    workload.add_program(device_range, std::move(program));

    std::cout << "[ATTPROBE] launching kernel on logical " << src.x << "," << src.y << " (non-blocking)" << std::endl;
    const auto t0 = std::chrono::steady_clock::now();
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    auto elapsed_s = [&]() {
        return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - t0).count();
    };
    std::cout << "[ATTPROBE] launch returned after " << std::fixed << std::setprecision(1) << elapsed_s() << " s"
              << std::endl;

    // ---- poll the step markers -----------------------------------------------------------------------
    std::array<std::string, NUM_STEPS> marker_state;
    marker_state.fill("MISSING");
    std::array<uint32_t, NUM_STEPS> marker_raw{};
    int last_completed = -1;
    bool final_seen = false;
    uint32_t poll_errors = 0;
    while (true) {
        std::string err;
        const auto m = read_block(dev, src, result_addr + OFF_STEP, NUM_STEPS, err);
        if (!m) {
            if (++poll_errors <= 3) {
                std::cout << "[ATTPROBE] marker poll REJECTED: " << err << std::endl;
            }
        } else {
            for (uint32_t s = 0; s < NUM_STEPS; ++s) {
                marker_raw[s] = (*m)[s];
                if (marker_state[s] != "MISSING") {
                    continue;
                }
                if ((*m)[s] == (STEP_DONE_TAG | s)) {
                    marker_state[s] = "DONE";
                    last_completed = std::max<int>(last_completed, static_cast<int>(s));
                    std::cout << "[ATTPROBE] step " << s << " done (+" << std::fixed << std::setprecision(1)
                              << elapsed_s() << " s)" << std::endl;
                } else if ((*m)[s] == (STEP_SKIP_TAG | s)) {
                    marker_state[s] = "SKIP";
                    last_completed = std::max<int>(last_completed, static_cast<int>(s));
                    std::cout << "[ATTPROBE] step " << s << " skipped by step_mask (+" << std::fixed
                              << std::setprecision(1) << elapsed_s() << " s)" << std::endl;
                }
            }
            if (marker_state[FINAL_STEP] == "DONE") {
                final_seen = true;
                break;
            }
        }
        if (elapsed_s() > static_cast<double>(timeout_s)) {
            std::cout << "[ATTPROBE] poll timeout after " << timeout_s << " s; last completed step "
                      << last_completed << std::endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
    }

    // ---- read back everything (also on the hang path: host reads keep working) -----------------------
    std::string err;
    const auto src_block = read_block(dev, src, result_addr, OFF_REMOTE_BASE / sizeof(uint32_t), err);
    if (!src_block) {
        std::cout << "[ATTPROBE] src result block read REJECTED: " << err << std::endl;
    }
    auto hdr32 = [&](uint32_t off) -> uint32_t { return src_block ? (*src_block)[(OFF_HDR + off) / 4] : 0u; };
    auto hdr64 = [&](uint32_t off) -> uint64_t { return (uint64_t{hdr32(off + 4)} << 32) | hdr32(off); };
    auto local_word = [&](uint32_t off, uint32_t i) -> uint32_t { return src_block ? (*src_block)[off / 4 + i] : 0u; };
    const bool header_ok = src_block && hdr32(HDR_MAGIC) == MAGIC;
    if (!header_ok) {
        std::cout << "[ATTPROBE] kernel header NOT written (magic " << hexstr(hdr32(HDR_MAGIC))
                  << "): the kernel never started; this is a launch/firmware problem, not a NoC step" << std::endl;
    } else {
        const uint32_t self = hdr32(HDR_SELF);
        std::cout << "[ATTPROBE] kernel self-view: my_x=" << (self & 0xff) << " my_y=" << ((self >> 8) & 0xff)
                  << " noc_index=" << ((self >> 16) & 0xff) << "  raw NOC_NODE_ID=" << hexstr(hdr32(HDR_NODE_ID))
                  << "  [host: virtual " << src_v.x << "-" << src_v.y << ", live id " << hexstr(live_id_of(src_v), 3)
                  << "]" << std::endl;
        std::cout << "[ATTPROBE] kernel view: NUM_DRAM_BANKS=" << hdr32(HDR_NUM_DRAM_BANKS)
                  << " dram_bank_to_noc_xy[noc][" << bank_id << "]=" << hexstr(hdr32(HDR_BANK_XY)) << " (x="
                  << (hdr32(HDR_BANK_XY) & 0x3f) << ",y=" << ((hdr32(HDR_BANK_XY) >> 6) & 0x3f)
                  << ")  bank_to_dram_offset[" << bank_id << "]=" << hexstr(hdr32(HDR_BANK_OFF)) << "  step_mask "
                  << hexstr(hdr32(HDR_STEP_MASK), 5) << "  sem L1 offset " << hexstr(hdr32(HDR_SEM_ADDR)) << std::endl;
        std::cout << "[ATTPROBE] kernel local landing slots: s3 read landing[0]=" << hexstr(local_word(OFF_DST_S3, 0))
                  << " s5 read landing[0]=" << hexstr(local_word(OFF_DST_S5, 0)) << " (kernel-recorded s3="
                  << hexstr(hdr32(HDR_S3_VALUE)) << " s5=" << hexstr(hdr32(HDR_S5_VALUE)) << ")" << std::endl;
        std::cout << "[ATTPROBE] operand s2 unicast  " << hexstr(hdr64(HDR_UNICAST_ADDR), 16) << " -> "
                  << decode_operand(hdr64(HDR_UNICAST_ADDR)) << std::endl;
        std::cout << "[ATTPROBE] operand s3 read src " << hexstr(hdr64(HDR_S3_SRC_ADDR), 16) << " -> "
                  << decode_operand(hdr64(HDR_S3_SRC_ADDR)) << std::endl;
        std::cout << "[ATTPROBE] operand s4/s5 DRAM addrgen " << hexstr(hdr64(HDR_DRAM_GEN_ADDR), 16) << " -> "
                  << decode_operand(hdr64(HDR_DRAM_GEN_ADDR)) << std::endl;
        std::cout << "[ATTPROBE] operand s9 DRAM bank-id  " << hexstr(hdr64(HDR_DRAM_BANK_ADDR), 16) << " -> "
                  << decode_operand(hdr64(HDR_DRAM_BANK_ADDR)) << std::endl;
        std::cout << "[ATTPROBE] operand s6 semaphore " << hexstr(hdr64(HDR_SEM_NOC_ADDR), 16) << " -> "
                  << decode_operand(hdr64(HDR_SEM_NOC_ADDR)) << std::endl;
        std::cout << "[ATTPROBE] operand s7 mcast column descriptor " << hexstr(hdr64(HDR_MCAST_COL), 16) << " -> "
                  << decode_mcast_descriptor(hdr64(HDR_MCAST_COL)) << std::endl;
        std::cout << "[ATTPROBE] operand s8 mcast grid descriptor   " << hexstr(hdr64(HDR_MCAST_GRID), 16) << " -> "
                  << decode_mcast_descriptor(hdr64(HDR_MCAST_GRID)) << std::endl;
    }

    // Landing slots on every worker.
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> landing;
    for (const auto& n : nodes) {
        const uint32_t lx = static_cast<uint32_t>(n.x), ly = static_cast<uint32_t>(n.y);
        std::string e2;
        const auto b = read_block(dev, n, result_addr + OFF_REMOTE_BASE, (BLOCK_BYTES - OFF_REMOTE_BASE) / 4, e2);
        if (!b) {
            std::cout << "[ATTPROBE] landing slots of worker " << lx << "," << ly << " REJECTED: " << e2 << std::endl;
            continue;
        }
        landing[{lx, ly}] = *b;
        auto w = [&](uint32_t off) { return (*b)[(off - OFF_REMOTE_BASE) / 4]; };
        const CoreCoord v = virt_of[{lx, ly}];
        std::cout << "[ATTPROBE] worker " << lx << "," << ly << " (virtual " << v.x << "-" << v.y << ")"
                  << ((lx == src.x && ly == src.y) ? " SRC" : (lx == dst.x && ly == dst.y) ? " DST" : "    ")
                  << "  s2-slot=" << hexstr(w(OFF_REMOTE_S2)) << "  s3-src-slot=" << hexstr(w(OFF_REMOTE_S3_SRC))
                  << "  s7-slot=" << hexstr(w(OFF_REMOTE_S7)) << "  s8-slot=" << hexstr(w(OFF_REMOTE_S8)) << std::endl;
    }
    auto slot_is = [&](const NodeCoord& n, uint32_t off, uint32_t step) -> bool {
        auto it = landing.find({static_cast<uint32_t>(n.x), static_cast<uint32_t>(n.y)});
        if (it == landing.end()) {
            return false;
        }
        for (uint32_t i = 0; i < XFER_WORDS; ++i) {
            if (it->second[(off - OFF_REMOTE_BASE) / 4 + i] != payload_word(step, i)) {
                return false;
            }
        }
        return true;
    };
    auto slot_untouched = [&](const NodeCoord& n, uint32_t off) -> bool {
        auto it = landing.find({static_cast<uint32_t>(n.x), static_cast<uint32_t>(n.y)});
        return it != landing.end() && it->second[(off - OFF_REMOTE_BASE) / 4] == L1_SENTINEL;
    };
    auto list_hits = [&](uint32_t off, uint32_t step) {
        std::string s;
        for (const auto& n : nodes) {
            if (slot_is(n, off, step)) {
                s += "(" + std::to_string(n.x) + "," + std::to_string(n.y) + ")";
            }
        }
        return s.empty() ? std::string("none") : s;
    };

    // DRAM, every channel.
    std::map<int, std::vector<uint32_t>> dram;
    for (int ch = 0; ch < num_dram_ch; ++ch) {
        try {
            std::vector<uint32_t> d;
            tt::tt_metal::detail::ReadFromDeviceDRAMChannel(dev0, ch, dram_addr, DRAM_PROBE_WORDS * sizeof(uint32_t), d);
            dram[ch] = d;
            std::cout << "[ATTPROBE] DRAM channel " << ch << " @" << hexstr(dram_addr) << ": [0]=" << hexstr(d[0])
                      << " [1]=" << hexstr(d[1]) << " ... [16]=" << hexstr(d[16]) << " [17]=" << hexstr(d[17]) << std::endl;
        } catch (const std::exception& e) {
            std::cout << "[ATTPROBE] DRAM channel " << ch << " read REJECTED: " << e.what() << std::endl;
        }
    }
    auto dram_has = [&](uint32_t byte_off, uint32_t step) {
        std::string s;
        for (const auto& [ch, d] : dram) {
            bool ok = true;
            for (uint32_t i = 0; i < XFER_WORDS; ++i) {
                ok = ok && d[byte_off / 4 + i] == payload_word(step, i);
            }
            if (ok) {
                s += "ch" + std::to_string(ch) + " ";
            }
        }
        return s;
    };

    // Semaphore on every worker (address recorded by the kernel; identical offset on all program cores).
    const uint32_t sem_addr = header_ok ? hdr32(HDR_SEM_ADDR) : 0u;
    std::map<std::pair<uint32_t, uint32_t>, WordRead> sem_vals;
    if (header_ok && sem_addr != 0 && sem_addr < (4u << 20)) {
        for (const auto& n : nodes) {
            const uint32_t lx = static_cast<uint32_t>(n.x), ly = static_cast<uint32_t>(n.y);
            sem_vals[{lx, ly}] = read_word(dev, n, sem_addr);
            const auto& r = sem_vals[{lx, ly}];
            std::cout << "[ATTPROBE] semaphore @" << hexstr(sem_addr) << " on worker " << lx << "," << ly << " = "
                      << (r.value ? hexstr(*r.value) : "REJECTED(" + r.error + ")") << std::endl;
        }
    } else {
        std::cout << "[ATTPROBE] semaphore address unknown (" << hexstr(sem_addr) << "); skipping semaphore read-back"
                  << std::endl;
    }

    // Read-size bracket (s11..s16): kernel records + landing regions on the src worker. Host reads keep working
    // on the hang path, so a hung step still reports its phase and how many words actually landed.
    struct RdResult {
        uint32_t step = 0, size = 0, words = 0;
        uint32_t rec_size = 0;  // size as the kernel recorded it (must equal `size`)
        uint32_t phase = 0, first = 0, last = 0, dst = 0;
        uint64_t src = 0;
        uint32_t exp_first = 0, exp_last = 0;
        uint32_t landed_match = 0;   // landing words equal to the seed at their index
        uint32_t landed_prefix = 0;  // leading run of matching words (bytes delivered in order)
        bool have_rec = false, have_landing = false;
    };
    std::array<RdResult, RD_STEPS> rd{};
    {
        std::string e3;
        const auto recs = read_block(dev, src, result_addr + OFF_RD_REC, RD_REC_BYTES / sizeof(uint32_t), e3);
        if (!recs) {
            std::cout << "[ATTPROBE] read-size bracket records REJECTED: " << e3 << std::endl;
        }
        for (uint32_t k = 0; k < RD_STEPS; ++k) {
            RdResult& r = rd[k];
            r.step = RD_FIRST_STEP + k;
            r.size = RD_SIZES[k];
            r.words = r.size / sizeof(uint32_t);
            const bool from_l1 = r.step == RD_L1_STEP;
            auto expect = [&](uint32_t i) {
                return from_l1 ? l1_rd_word(static_cast<uint32_t>(dst.x), static_cast<uint32_t>(dst.y), i)
                               : dram_rd_word(i);
            };
            r.exp_first = expect(0);
            r.exp_last = expect(r.words - 1);
            if (recs) {
                const uint32_t w = (RD_REC_STRIDE / sizeof(uint32_t)) * k;
                r.have_rec = true;
                r.rec_size = (*recs)[w + REC_SIZE / 4];
                r.phase = (*recs)[w + REC_PHASE / 4];
                r.first = (*recs)[w + REC_FIRST / 4];
                r.last = (*recs)[w + REC_LAST / 4];
                r.dst = (*recs)[w + REC_DST / 4];
                r.src = (uint64_t{(*recs)[w + REC_SRC / 4 + 1]} << 32) | (*recs)[w + REC_SRC / 4];
            }
            std::string e4;
            const auto land = read_block(dev, src, result_addr + OFF_RD_LANDING + RD_LANDING_STRIDE * k, r.words, e4);
            if (land) {
                r.have_landing = true;
                bool prefix = true;
                for (uint32_t i = 0; i < r.words; ++i) {
                    const bool ok = (*land)[i] == expect(i);
                    r.landed_match += ok ? 1u : 0u;
                    prefix = prefix && ok;
                    r.landed_prefix += prefix ? 1u : 0u;
                }
            } else {
                std::cout << "[ATTPROBE] s" << r.step << " landing region read REJECTED: " << e4 << std::endl;
            }
            std::cout << "[ATTPROBE] s" << r.step << " read " << std::setw(5) << r.size << " B <- "
                      << (from_l1 ? "dst worker L1" : "DRAM         ") << ": marker " << marker_state[r.step]
                      << "  phase " << rd_phase_str(r.phase) << "  kernel first " << hexstr(r.first) << " (exp "
                      << hexstr(r.exp_first) << ") last " << hexstr(r.last) << " (exp " << hexstr(r.exp_last)
                      << ")  landing " << r.landed_match << "/" << r.words << " words match seed, in-order prefix "
                      << r.landed_prefix * 4 << " B  src " << hexstr(r.src, 16) << " -> " << decode_operand(r.src, r.size)
                      << (r.have_rec && r.rec_size != r.size
                              ? "  KERNEL/HOST SIZE MISMATCH (kernel recorded " + std::to_string(r.rec_size) + " B)"
                              : "")
                      << std::endl;
        }
        // One-line bracket summary for the DRAM sizes (ascending), then the L1 control read.
        uint32_t largest_done = 0;
        std::string first_not_done;
        for (uint32_t k = 0; k < RD_STEPS; ++k) {
            const RdResult& r = rd[k];
            if (r.step == RD_L1_STEP) {
                continue;
            }
            if (marker_state[r.step] == "DONE") {
                largest_done = std::max(largest_done, r.size);
            } else if (marker_state[r.step] == "MISSING" && first_not_done.empty()) {
                first_not_done = std::to_string(r.size) + " B (phase " + rd_phase_str(r.phase) + ", in-order prefix " +
                                 std::to_string(r.landed_prefix * 4) + " B landed)";
            }
        }
        const RdResult& l1 = rd[RD_L1_STEP - RD_FIRST_STEP];
        std::cout << "[ATTPROBE] READ-SIZE BRACKET: largest completed DRAM read " << largest_done
                  << " B; first uncompleted DRAM read " << (first_not_done.empty() ? std::string("none") : first_not_done)
                  << "; L1 8192 B from dst worker " << marker_state[RD_L1_STEP] << " (phase " << rd_phase_str(l1.phase)
                  << ", in-order prefix " << l1.landed_prefix * 4 << " B landed)" << std::endl;
    }

    // ATT debug registers on every live tile: which tile latched no_match / what was the last target?
    for (const auto& n : nodes) {
        dump_att_registers(dev, n, virt_of[{static_cast<uint32_t>(n.x), static_cast<uint32_t>(n.y)}], "POSTRUN", false);
    }

    // ---- verdicts ------------------------------------------------------------------------------------
    std::vector<Verdict> verdicts;
    auto add = [&](uint32_t step, const std::string& op, bool applicable, bool pass, const std::string& detail) {
        const std::string mk = marker_state[step];
        std::string res = !applicable ? "N/A" : (pass ? "PASS" : "FAIL");
        if (mk == "SKIP") {
            res = "N/A";
        }
        verdicts.push_back(Verdict{step, mk, op, res, detail, applicable && mk != "SKIP" && !pass});
    };
    add(0, "header (self id + resolved operands)", true, header_ok, header_ok ? "written" : "missing");
    {
        bool ok = true;
        for (uint32_t i = 0; i < XFER_WORDS; ++i) {
            ok = ok && local_word(OFF_LOCAL_S1, i) == payload_word(1, i);
        }
        add(1, "local L1 write", true, ok, "src slot[0]=" + hexstr(local_word(OFF_LOCAL_S1, 0)));
    }
    {
        const bool landed = slot_is(dst, OFF_REMOTE_S2, 2);
        add(2, "unicast write 16 B -> dst worker + write barrier", true, landed,
            std::string(landed ? "LANDED on dst" : "MISSED dst") + "; tiles holding s2 payload: " +
                list_hits(OFF_REMOTE_S2, 2));
    }
    {
        const uint32_t v = header_ok ? hdr32(HDR_S3_VALUE) : 0u;
        const bool ok = v == s3_source_word(dst.x, dst.y, 0);
        std::string d = "kernel read " + hexstr(v) + " expected " + hexstr(s3_source_word(dst.x, dst.y, 0));
        if (!ok && (v & 0xFFFF0000u) == 0x5EED0000u) {
            d += " -> came from logical (" + std::to_string((v >> 8) & 0xf) + "," + std::to_string((v >> 4) & 0xf) + ")";
        } else if (!ok && v == L1_SENTINEL) {
            d += " -> read hit a plain landing slot (wrong offset/tile)";
        }
        add(3, "unicast read 16 B <- dst worker + read barrier", true, ok, d);
    }
    {
        const std::string hits = dram_has(0, 4);
        const bool ok = !hits.empty();
        add(4, "DRAM write 16 B via InterleavedAddrGen<true> + barrier", true, ok,
            "channels holding s4 payload: " + (hits.empty() ? std::string("none") : hits) +
                (dram.count(0) ? " ch0[0]=" + hexstr(dram[0][0]) : std::string()));
    }
    {
        const uint32_t v = header_ok ? hdr32(HDR_S5_VALUE) : 0u;
        const bool ok = v == payload_word(4, 0);
        std::string d = "kernel read back " + hexstr(v);
        if (v == DRAM_SENTINEL) {
            d += " = host sentinel -> DRAM read path works, s4 write missed";
        } else if (v == 0 && marker_state[5] == "DONE") {
            d += " = 0 -> read returned nothing usable";
        }
        add(5, "DRAM read 16 B back via addrgen + barrier", true, ok, d);
    }
    {
        auto it = sem_vals.find({static_cast<uint32_t>(dst.x), static_cast<uint32_t>(dst.y)});
        const bool have = it != sem_vals.end() && it->second.value.has_value();
        const bool ok = have && *it->second.value == 1u;
        std::string d = have ? "dst semaphore = " + hexstr(*it->second.value) : "dst semaphore unreadable";
        for (const auto& [k, r] : sem_vals) {
            if (r.value && *r.value != 0 && !(k.first == dst.x && k.second == dst.y)) {
                d += " STRAY on (" + std::to_string(k.first) + "," + std::to_string(k.second) + ")=" + hexstr(*r.value);
            }
        }
        add(6, "noc_semaphore_inc(+1) -> dst worker + atomic barrier", true, ok, d);
    }
    {
        const bool landed = slot_is(dst, OFF_REMOTE_S7, 7);
        const bool self_clean = slot_untouched(src, OFF_REMOTE_S7);
        add(7, "multicast 16 B column rect (num_dests=" + std::to_string(col_num_dests) + ") + barrier", true,
            landed && self_clean,
            "tiles holding s7 payload: " + list_hits(OFF_REMOTE_S7, 7) +
                (self_clean ? "; self untouched" : "; SELF WRITTEN (loopback semantics?)"));
    }
    {
        bool all = true;
        for (const auto& n : nodes) {
            if (n.x == src.x && n.y == src.y) {
                continue;
            }
            all = all && slot_is(n, OFF_REMOTE_S8, 8);
        }
        const bool self_clean = slot_untouched(src, OFF_REMOTE_S8);
        add(8, "multicast 16 B grid rect (num_dests=" + std::to_string(grid_num_dests) + ") + barrier", true,
            all && self_clean,
            "tiles holding s8 payload: " + list_hits(OFF_REMOTE_S8, 8) +
                (self_clean ? "; self untouched" : "; SELF WRITTEN (loopback semantics?)"));
    }
    {
        const std::string hits = dram_has(dram_addr_alt - dram_addr, 9);
        const bool ok = !hits.empty();
        add(9, "DRAM write 16 B via get_noc_addr_from_bank_id<true> + barrier [EXTRA]", true, ok,
            "channels holding s9 payload at +64: " + (hits.empty() ? std::string("none") : hits));
    }
    add(ORIGINAL_END_STEP, "checkpoint marker: steps 0..9 walked", true, marker_state[ORIGINAL_END_STEP] == "DONE",
        marker_state[ORIGINAL_END_STEP] == "DONE" ? "original probe end reached" : "never reached");
    for (uint32_t k = 0; k < RD_STEPS; ++k) {
        const RdResult& r = rd[k];
        const bool done = marker_state[r.step] == "DONE";
        const bool first_ok = r.first == r.exp_first;
        const bool last_ok = r.last == r.exp_last;
        const std::string op = (r.step == RD_L1_STEP ? "L1 read " : "DRAM read ") + std::to_string(r.size) +
                               (r.step == RD_L1_STEP ? " B <- dst worker + read barrier [bracket]"
                                                     : " B via addrgen + read barrier [bracket]");
        std::string d = std::to_string(r.size) + " B: " + (done ? "DONE" : marker_state[r.step]) + "; first word " +
                        (first_ok ? "MATCH" : "MISMATCH(" + hexstr(r.first) + ")") + " last word " +
                        (last_ok ? "MATCH" : "MISMATCH(" + hexstr(r.last) + ")") + "; landing " +
                        std::to_string(r.landed_match) + "/" + std::to_string(r.words) + " words, in-order prefix " +
                        std::to_string(r.landed_prefix * 4) + " B";
        if (!done && marker_state[r.step] == "MISSING") {
            if (r.phase == PHASE_ISSUED) {
                d += " -> read issued, barrier never returned";
            } else if (r.phase == PHASE_ISSUING) {
                d += " -> stuck inside noc_async_read (command issue)";
            } else if (r.phase == PHASE_PENDING) {
                d += " -> never reached (earlier step hung)";
            }
        }
        add(r.step, op, true, done && first_ok && last_ok, d);
    }
    add(FINAL_STEP, "final marker", true, final_seen, final_seen ? "kernel ran to completion" : "never reached");

    std::cout << "[ATTPROBE] ================================ VERDICT TABLE ================================"
              << std::endl;
    std::cout << "[ATTPROBE] step | marker  | result | operation                                                    | detail"
              << std::endl;
    for (const auto& v : verdicts) {
        std::cout << "[ATTPROBE] s" << std::left << std::setw(3) << v.step << "| " << std::setw(8) << v.marker << "| "
                  << std::setw(7) << v.result << "| " << std::setw(61) << v.operation << "| " << v.detail << std::endl;
    }
    std::cout << "[ATTPROBE] STAGE0 mismatches=" << stage0.mismatches << " rejected=" << stage0.rejected << std::endl;

    if (!final_seen) {
        std::cout << "[ATTPROBE] last completed step " << last_completed << std::endl;
        std::cout << "[ATTPROBE] HANG after step " << last_completed << std::endl;
        std::cout.flush();
        std::fflush(stdout);
        std::fflush(stderr);
        // Fixture teardown would wait forever on the hung kernel; the emulator server exits ~90 s after
        // the client disappears.
        std::_Exit(1);
    }

    EXPECT_EQ(stage0.mismatches, 0u) << "STAGE0 ATT tables drifted from the boot image metal was built against";
    for (const auto& v : verdicts) {
        EXPECT_FALSE(v.counts_as_failure) << "step " << v.step << " (" << v.operation << "): " << v.detail;
    }
}
