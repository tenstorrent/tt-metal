// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

// Device coverage for worker-rectangle multicast: one sender issues an
// ordinary multicast to a peer-only rectangle and a loopback multicast to the
// rectangle spanning itself and the peer. Every destination region is poisoned
// first, and the sender's own copy is read back too, so a dropped destination
// or a wrong destination count cannot pass.
// Needs two worker nodes in a row (the 2x3 simulator/emulator grids).
TEST_F(QuasarMeshDeviceSingleCardFixture, DmWorkerMulticast) {
    if (std::getenv("TT_METAL_SIMULATOR") == nullptr) {
        GTEST_SKIP() << "This test can only be run using a simulator. Set TT_METAL_SIMULATOR environment variable.";
    }
    const auto grid = this->device().compute_with_storage_grid_size();
    if (grid.x < 2) {
        GTEST_SKIP() << "needs >= 2 worker nodes in a row for a multicast rectangle (got " << grid.x << "x" << grid.y
                     << ")";
    }

    const CoreCoord sender_logical{0, 0};
    const CoreCoord peer_logical{1, 0};
    const experimental::NodeCoord sender_node{0, 0};
    const CoreCoord sender_noc = this->device().worker_core_from_logical_core(sender_logical);
    const CoreCoord peer_noc = this->device().worker_core_from_logical_core(peer_logical);

    constexpr uint32_t size_bytes = 256;
    constexpr uint32_t num_words = size_bytes / sizeof(uint32_t);
    const uint32_t l1_base = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t src_addr = l1_base;
    const uint32_t dst_addr_unicast_rect = l1_base + 1024;
    const uint32_t dst_addr_loopback_rect = l1_base + 2048;

    std::vector<uint32_t> inputs(num_words);
    for (uint32_t i = 0; i < num_words; i++) {
        inputs[i] = 0x3C000000u | i;
    }
    std::vector<uint32_t> poison(num_words, 0xDEADBEEF);
    slow_dispatch::WriteToL1(this->device(), sender_logical, src_addr, inputs);
    for (const CoreCoord& core : {sender_logical, peer_logical}) {
        slow_dispatch::WriteToL1(this->device(), core, dst_addr_unicast_rect, poison);
        slow_dispatch::WriteToL1(this->device(), core, dst_addr_loopback_rect, poison);
    }
    MetalContext::instance().get_cluster().l1_barrier(this->device().get_device_ids()[0]);

    distributed::MeshCommandQueue& cq = this->device().mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(this->device().shape());

    const experimental::KernelSpecName SENDER{"worker_multicast_pair"};
    experimental::ProgramSpec spec{
        .name = "dm_worker_multicast",
        .kernels = {experimental::KernelSpec{
            .unique_id = SENDER,
            .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/worker_multicast_pair.cpp",
            .num_threads = 1,
            .runtime_arg_schema =
                {
                    .runtime_arg_names =
                        {"src_addr",
                         "dst_addr_unicast_rect",
                         "dst_addr_loopback_rect",
                         "size_bytes",
                         "self_x",
                         "self_y",
                         "peer_x",
                         "peer_y"},
                },
            .hw_config = experimental::DataMovementGen2Config{},
        }},
        .work_units = {experimental::WorkUnitSpec{
            .name = "main",
            .kernels = {SENDER},
            .target_nodes = sender_node,
        }},
    };
    Program program = experimental::MakeProgramFromSpec(this->device(), spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = SENDER,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
            sender_node,
            {{"src_addr", src_addr},
             {"dst_addr_unicast_rect", dst_addr_unicast_rect},
             {"dst_addr_loopback_rect", dst_addr_loopback_rect},
             {"size_bytes", size_bytes},
             {"self_x", static_cast<uint32_t>(sender_noc.x)},
             {"self_y", static_cast<uint32_t>(sender_noc.y)},
             {"peer_x", static_cast<uint32_t>(peer_noc.x)},
             {"peer_y", static_cast<uint32_t>(peer_noc.y)}})});
    experimental::SetProgramRunArgs(program, params);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    auto check = [&](const CoreCoord& core, uint32_t addr, const std::vector<uint32_t>& expected, const char* what) {
        std::vector<uint32_t> out;
        slow_dispatch::ReadFromL1(this->device(), core, addr, size_bytes, out);
        ASSERT_EQ(out.size(), expected.size());
        for (uint32_t i = 0; i < expected.size(); i++) {
            ASSERT_EQ(out[i], expected[i]) << what << " on logical core " << core.str() << " word " << i << ": got 0x"
                                           << std::hex << out[i] << " expected 0x" << expected[i];
        }
    };
    // Peer-only rectangle: the peer receives, the sender's copy stays poisoned.
    check(peer_logical, dst_addr_unicast_rect, inputs, "peer-only multicast");
    check(sender_logical, dst_addr_unicast_rect, poison, "peer-only multicast must not touch the sender");
    // Loopback rectangle: both receive.
    check(peer_logical, dst_addr_loopback_rect, inputs, "loopback multicast (peer)");
    check(sender_logical, dst_addr_loopback_rect, inputs, "loopback multicast (sender)");
}
