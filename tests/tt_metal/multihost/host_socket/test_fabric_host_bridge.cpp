// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// E2H2H2E: a fabric unicast whose only path between two meshes is a host-bridged cable.
//
// Two ranks, one 1x2 mesh each, the cables between the meshes built host-bridged
// (TT_METAL_FABRIC_HOST_BRIDGE=intermesh). A worker on the lower mesh sends packets through the bridged
// router with the stock 1D tx kernel; the stock rx kernel on the far mesh checks them. Nothing in the
// kernels or the fabric knows the cable is virtual.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/fabric_host_bridge.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "hostdevcommon/fabric_common.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "host_socket_test_utils.hpp"

namespace tt::tt_metal::distributed {
namespace {

using tt::tt_fabric::FabricNodeId;

constexpr const char* kTxKernel = "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_tx.cpp";
constexpr const char* kRxKernel = "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_rx.cpp";

// Same carve-up of worker L1 as the fabric unit tests' generate_worker_mem_map().
constexpr uint32_t kPacketHeaderReservedBytes = 45056;
constexpr uint32_t kDataSpaceReservedBytes = 851968;
constexpr uint32_t kTestResultsBytes = 128;

struct LinkEnd {
    MeshCoordinate device{0, 0};
    FabricNodeId node{tt::tt_fabric::MeshId{0}, 0};
    uint32_t chan = 0;
};

struct Link {
    std::tuple<uint32_t, uint32_t, uint32_t> key;  // canonical on both ranks
    LinkEnd local;
    FabricNodeId peer{tt::tt_fabric::MeshId{0}, 0};
    uint32_t peer_chan = 0;
};

std::vector<Link> local_intermesh_links(const std::shared_ptr<MeshDevice>& mesh) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    std::vector<Link> links;
    for (const auto& coord : MeshCoordinateRange(mesh->shape())) {
        const auto node = control_plane.get_fabric_node_id_from_physical_chip_id(mesh->get_device(coord)->id());
        // Only channels with a router: the fabric may leave some inter-mesh cables unused.
        const auto active = control_plane.get_active_fabric_eth_channels(node);
        for (const auto chan : control_plane.get_intermesh_facing_eth_chans(node)) {
            const bool has_router =
                std::any_of(active.begin(), active.end(), [&](const auto& c) { return c.first == chan; });
            if (!has_router) {
                continue;
            }
            const auto [peer, peer_chan] = control_plane.get_connected_mesh_chip_chan_ids(node, chan);
            const auto a =
                std::make_tuple(*node.mesh_id, static_cast<uint32_t>(node.chip_id), static_cast<uint32_t>(chan));
            const auto b =
                std::make_tuple(*peer.mesh_id, static_cast<uint32_t>(peer.chip_id), static_cast<uint32_t>(peer_chan));
            links.push_back(Link{
                .key = std::min(a, b),
                .local = LinkEnd{.device = coord, .node = node, .chan = chan},
                .peer = peer,
                .peer_chan = peer_chan});
        }
    }
    std::sort(links.begin(), links.end(), [](const Link& x, const Link& y) { return x.key < y.key; });
    return links;
}

std::vector<uint32_t> read_results(IDevice* device, const CoreCoord& core, uint32_t address) {
    std::vector<uint32_t> words(kTestResultsBytes / sizeof(uint32_t), 0);
    std::vector<uint8_t> bytes(kTestResultsBytes, 0);
    tt_metal::detail::ReadFromDeviceL1(device, core, address, bytes, CoreType::WORKER);
    std::memcpy(words.data(), bytes.data(), kTestResultsBytes);
    return words;
}

}  // namespace

TEST(FabricHostBridgeTest, UnicastOverBridgedIntermeshLink) {
    const auto& context = multihost::DistributedContext::get_current_world();
    ASSERT_EQ(*context->size(), 2) << "needs exactly 2 ranks";
    const int rank = *context->rank();
    const char* spec = std::getenv("TT_METAL_FABRIC_HOST_BRIDGE");
    if (spec == nullptr || *spec == '\0') {
        GTEST_SKIP()
            << "set TT_METAL_FABRIC_HOST_BRIDGE=intermesh with a dual-mesh descriptor (run_fabric_host_bridge.sh)";
    }

    // 2D: a 1D fabric refuses more than one mesh.
    tt::tt_fabric::SetFabricConfig(
        tt::tt_fabric::FabricConfig::FABRIC_2D, tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
    auto mesh = MeshDevice::create(MeshDeviceConfig(MeshShape(1, 2)));

    auto links = local_intermesh_links(mesh);
    ASSERT_FALSE(links.empty()) << "rank " << rank << ": no inter-mesh links on this mesh";
    std::vector<FabricHostBridgeLink> bridge_links;
    for (uint32_t i = 0; i < links.size(); i++) {
        bridge_links.push_back(FabricHostBridgeLink{
            .device = links[i].local.device,
            .eth_chan = links[i].local.chan,
            .peer_rank = multihost::Rank{1 - rank},
            .link_id = i});
    }
    auto bridge = std::make_unique<FabricHostBridge>(mesh, bridge_links, /*ring_pages=*/64);
    context->barrier();

    // The lower mesh sends over link 0, the higher one receives.
    const Link& link = links[0];
    const bool is_sender = *link.local.node.mesh_id < *link.peer.mesh_id;
    IDevice* device = mesh->get_device(link.local.device);
    const CoreCoord sender_core{0, 0};
    const CoreCoord receiver_core{1, 0};
    const CoreCoord receiver_virtual = mesh->worker_core_from_logical_core(receiver_core);

    const uint32_t base = mesh->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t source_address = base + kPacketHeaderReservedBytes;
    const uint32_t results_address = source_address + kDataSpaceReservedBytes;
    const uint32_t target_address = source_address;
    const uint32_t payload_bytes = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes());
    const uint32_t num_packets = 64;
    const uint32_t seed = 0x5EED1234;  // the same on both ranks, which is why it is not the clock

    const std::vector<uint32_t> compile_args = {
        results_address,
        kTestResultsBytes,
        target_address,
        0 /*dram dst*/,
        1 /*2d*/,
        0 /*mcast*/,
        0 /*additional dir*/};
    const std::map<std::string, std::string> defines = {{"FABRIC_2D", ""}};

    // The router the sender attaches to: the forwarding channel toward the far chip, which must be one
    // of the bridged ones (every inter-mesh channel is).
    uint32_t edm_port = link.local.chan;
    if (is_sender) {
        auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
        const auto chans = control_plane.get_forwarding_eth_chans_to_chip(link.local.node, link.peer);
        ASSERT_FALSE(chans.empty()) << "no forwarding channel toward the far mesh";
        edm_port = chans.front();
        const bool bridged = std::any_of(links.begin(), links.end(), [&](const Link& l) {
            return l.local.device == link.local.device && l.local.chan == edm_port;
        });
        ASSERT_TRUE(bridged) << "forwarding channel " << edm_port << " is not host-bridged";
    }
    log_info(
        tt::LogTest,
        "rank {} {} on chip {} (mesh {} chip {}) over eth_chan {} <-> mesh {} chip {} eth_chan {}: {} packets of {} B",
        rank,
        is_sender ? "sends" : "receives",
        device->id(),
        *link.local.node.mesh_id,
        link.local.node.chip_id,
        link.local.chan,
        *link.peer.mesh_id,
        link.peer.chip_id,
        link.peer_chan,
        num_packets,
        payload_bytes);

    auto program = CreateProgram();
    if (is_sender) {
        auto kernel = CreateKernel(
            program,
            kTxKernel,
            sender_core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = compile_args,
                .defines = defines});
        std::vector<uint32_t> rt_args = {
            source_address,
            payload_bytes,
            num_packets,
            seed,
            static_cast<uint32_t>(receiver_virtual.x),
            static_cast<uint32_t>(receiver_virtual.y),
            1 /*hops*/,
            1 /*fwd range*/,
            static_cast<uint32_t>(link.peer.chip_id),
            *link.peer.mesh_id};
        const auto teardown_sem = CreateSemaphore(program, sender_core, 0);
        const auto buffer_index_sem = CreateSemaphore(program, sender_core, 0);
        tt::tt_fabric::append_worker_to_fabric_edm_sender_rt_args(
            static_cast<tt::tt_fabric::chan_id_t>(edm_port), teardown_sem, buffer_index_sem, rt_args);
        SetRuntimeArgs(program, kernel, sender_core, rt_args);
    } else {
        auto kernel = CreateKernel(
            program,
            kRxKernel,
            receiver_core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = compile_args,
                .defines = defines});
        SetRuntimeArgs(program, kernel, receiver_core, std::vector<uint32_t>{payload_bytes, num_packets, seed});
    }

    MeshWorkload workload;
    workload.add_program(MeshCoordinateRange(link.local.device, link.local.device), std::move(program));
    if (!is_sender) {
        // Receiver first, so no packet lands before it is listening.
        EnqueueMeshWorkload(mesh->mesh_command_queue(), workload, false);
    }
    context->barrier();
    if (is_sender) {
        EnqueueMeshWorkload(mesh->mesh_command_queue(), workload, false);
    }
    // A stalled bridge hangs Finish() forever, so watch it from here and dump the bridge when it stalls.
    std::atomic<bool> finished{false};
    std::thread finisher([&] {
        Finish(mesh->mesh_command_queue());
        finished.store(true, std::memory_order_release);
    });
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(host_socket_test::env_or("TT_HOST_SOCKET_BRIDGE_WAIT_S", 60));
    auto next_report = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!finished.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const auto now = std::chrono::steady_clock::now();
        if (now >= next_report || !bridge->first_error().empty()) {
            log_info(
                tt::LogTest,
                "rank {} waiting; bridge:\n{}error: '{}'",
                rank,
                bridge->describe(),
                bridge->first_error());
            next_report = now + std::chrono::seconds(5);
        }
        if (now >= deadline || !bridge->first_error().empty()) {
            log_error(tt::LogTest, "rank {}: bridge stalled\n{}", rank, bridge->describe());
            std::fflush(stdout);
            std::_Exit(3);  // Finish() cannot be interrupted
        }
    }
    finisher.join();

    const auto status = read_results(device, is_sender ? sender_core : receiver_core, results_address);
    EXPECT_EQ(status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS) << "rank " << rank;
    uint64_t bytes =
        (static_cast<uint64_t>(status[TT_FABRIC_WORD_CNT_INDEX + 1]) << 32) | status[TT_FABRIC_WORD_CNT_INDEX];

    std::vector<uint64_t> all_bytes(2, 0);
    context->all_gather(
        std::span<std::byte>(reinterpret_cast<std::byte*>(&bytes), sizeof(bytes)),
        std::span<std::byte>(reinterpret_cast<std::byte*>(all_bytes.data()), all_bytes.size() * sizeof(uint64_t)));
    EXPECT_EQ(all_bytes[0], all_bytes[1]) << "sender and receiver byte counts differ";
    EXPECT_GT(all_bytes[rank], 0u);
    log_info(tt::LogTest, "rank {}: {} bytes {}\n{}", rank, bytes, is_sender ? "sent" : "received", bridge->describe());
    EXPECT_TRUE(bridge->first_error().empty()) << bridge->first_error();

    context->barrier();
    bridge.reset();
    mesh->close();
    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
}

}  // namespace tt::tt_metal::distributed
