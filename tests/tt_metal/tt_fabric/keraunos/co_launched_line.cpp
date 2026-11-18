// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "co_launched_line.hpp"

#include <gtest/gtest.h>

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "fabric_fixture.hpp"
#include "hostdev/eth_link.h"
#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_accessor.hpp"
#include "impl/context/metal_env_impl.hpp"
#include "impl/program/program_impl.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"

namespace tt::tt_fabric::fabric_router_tests {
namespace {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::MetalContext;

constexpr uint32_t kNumPackets = 5;
constexpr auto kTimeout = std::chrono::minutes(20);
const std::string kTxKernel = "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_tx.cpp";
const std::string kRxKernel = "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_rx.cpp";

Worker worker_beside_router(FabricNodeId node, FabricNodeId toward) {
    const auto& control_plane = MetalContext::instance().get_control_plane();
    const auto chans = control_plane.get_forwarding_eth_chans_to_chip(node, toward);
    TT_FATAL(!chans.empty(), "No router on {} forwards toward {}", node, toward);
    const ChipId chip = control_plane.get_physical_chip_id_from_fabric_node_id(node);
    return {node, chip, chans.front(), BaseFabricFixture::get_worker_beside_router(chip, chans.front())};
}

// The sender's program on `line`'s first chip and the receiver's on its last: the stock 1D kernels, as the fabric
// smoke test builds them.
struct LinePrograms {
    tt::tt_metal::Program sender;
    tt::tt_metal::Program receiver;
};

LinePrograms create_line_programs(const Line& line, const WorkerMemMap& map, uint32_t seed) {
    const std::vector<uint32_t> compile_args = {
        map.test_results_address,
        map.test_results_size_bytes,
        map.target_address,
        0, /* use_dram_dst */
        0, /* is_2d_fabric */
        0, /* is_chip_multicast */
        0 /* additional_dir */};
    LinePrograms programs{
        tt::tt_metal::CreateProgram(*BaseFabricFixture::devices_map_.at(line.sender.chip)),
        tt::tt_metal::CreateProgram(*BaseFabricFixture::devices_map_.at(line.receiver.chip))};

    const FabricTestWorker& sender = line.sender.core;
    const FabricTestWorker& receiver = line.receiver.core;
    const auto sender_kernel =
        BaseFabricFixture::create_fabric_test_worker_kernel(programs.sender, kTxKernel, sender, compile_args);
    std::vector<uint32_t> sender_args = {
        map.source_l1_buffer_address,
        map.packet_payload_size_bytes,
        kNumPackets,
        seed,
        static_cast<uint32_t>(receiver.noc_core.x),
        static_cast<uint32_t>(receiver.noc_core.y),
        line.num_hops,
        1, /* fwd_range */
        line.receiver.node.chip_id,
        *line.receiver.node.mesh_id};
    const auto teardown_semaphore = tt::tt_metal::CreateSemaphore(programs.sender, sender.logical, 0, sender.core_type);
    const auto buffer_index_semaphore =
        tt::tt_metal::CreateSemaphore(programs.sender, sender.logical, 0, sender.core_type);
    append_worker_to_fabric_edm_sender_rt_args(
        line.sender.router_chan, teardown_semaphore, buffer_index_semaphore, sender_args);
    tt::tt_metal::SetRuntimeArgs(programs.sender, sender_kernel, sender.logical, sender_args);

    const auto receiver_kernel =
        BaseFabricFixture::create_fabric_test_worker_kernel(programs.receiver, kRxKernel, receiver, compile_args);
    tt::tt_metal::SetRuntimeArgs(
        programs.receiver, receiver_kernel, receiver.logical, {map.packet_payload_size_bytes, kNumPackets, seed});
    return programs;
}

// A router whose link record the extension set to the NOC kind, and the record as the extension found it.
// A fabric router as the extension prepared it: its record as found, whether its link is on the NOC, and whether it
// sends the line's traffic on that link, toward the line's last chip.
struct PreparedRouter {
    ChipId chip;
    CoreCoord virtual_core;
    EthLink as_found;
    bool noc;
    bool downstream;
};

// The router's 8-byte staging word of each Ethernet TX queue: a single-word send on a NOC link goes through them, a
// send on an Ethernet link never touches them (eth_cce_send.h send_dword).
constexpr uint32_t kNocStageWords = MEM_IERISC_FABRIC_NOC_DWORD_STAGE_SIZE / sizeof(uint64_t);
using NocStage = std::array<uint64_t, kNocStageWords>;

uint64_t eth_link_address() {
    return MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::ETH_LINK);
}

// The staging words sit directly after the link record (dev_mem_map.h); the HAL has no entry for them.
uint64_t noc_stage_address() {
    const uint64_t address = eth_link_address() + MetalContext::instance().hal().get_dev_size(
                                                      HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::ETH_LINK);
    TT_FATAL(
        address == MEM_IERISC_FABRIC_NOC_DWORD_STAGE_BASE,
        "NOC staging words at {:#x} after the HAL's link record, {:#x} in dev_mem_map.h",
        address,
        MEM_IERISC_FABRIC_NOC_DWORD_STAGE_BASE);
    return address;
}

EthLink read_eth_link(ChipId chip, const CoreCoord& virtual_core) {
    EthLink link{};
    MetalContext::instance().get_cluster().read_core(
        &link, sizeof(link), tt_cxy_pair(chip, virtual_core), eth_link_address());
    return link;
}

NocStage read_noc_stage(ChipId chip, const CoreCoord& virtual_core) {
    NocStage stage{};
    MetalContext::instance().get_cluster().read_core(
        stage.data(), sizeof(stage), tt_cxy_pair(chip, virtual_core), noc_stage_address());
    return stage;
}

// Zeroes the NOC staging words of every fabric router of `chip`, sets the NOC kind in the record of each whose link is
// in `noc_links`, and returns them all. Called from the extension under std::call_once, after device init wrote the
// records and before any device is configured.
std::vector<PreparedRouter> prepare_routers(ChipId chip, const NocLinks& noc_links) {
    const auto& control_plane = MetalContext::instance().get_control_plane();
    const auto& cluster = MetalContext::instance().get_cluster();
    const FabricNodeId node = control_plane.get_fabric_node_id_from_physical_chip_id(chip);
    std::vector<PreparedRouter> routers;
    for (const auto& [chan, direction] : control_plane.get_active_fabric_eth_channels(node)) {
        const FabricNodeId peer = control_plane.get_connected_mesh_chip_chan_ids(node, chan).first;
        const bool noc = std::any_of(noc_links.begin(), noc_links.end(), [&](const auto& link) {
            return (link.first == node.chip_id && link.second == peer.chip_id) ||
                   (link.second == node.chip_id && link.first == peer.chip_id);
        });
        const CoreCoord logical = cluster.get_soc_desc(chip).get_eth_core_for_channel(chan, tt::CoordSystem::LOGICAL);
        const CoreCoord virtual_core =
            cluster.get_virtual_coordinate_from_logical_coordinates(chip, logical, CoreType::ETH);
        PreparedRouter router{chip, virtual_core, read_eth_link(chip, virtual_core), noc, peer.chip_id > node.chip_id};
        // One write: the record, with the NOC kind on a chosen link, and the zeroed staging words directly after it.
        struct {
            EthLink record;
            NocStage stage;
        } block{router.as_found, {}};
        static_assert(sizeof(block) == sizeof(EthLink) + sizeof(NocStage));
        if (noc) {
            block.record.kind = ETH_LINK_KIND_NOC;
        }
        cluster.write_core_immediate(
            &block, sizeof(block), tt_cxy_pair(chip, virtual_core), noc_stage_address() - sizeof(EthLink));
        routers.push_back(router);
    }
    return routers;
}

// The byte count a stock 1D kernel wrote to its result block.
uint64_t bytes_of(const std::vector<uint32_t>& results) {
    return (static_cast<uint64_t>(results[TT_FABRIC_WORD_CNT_INDEX + 1]) << 32) | results[TT_FABRIC_WORD_CNT_INDEX];
}

// The worker's result block, read through the cluster.
std::vector<uint32_t> read_results(const Worker& worker, const WorkerMemMap& map) {
    const uint32_t num_words = map.test_results_size_bytes / sizeof(uint32_t);
    std::vector<uint32_t> words;
    MetalContext::instance().get_cluster().read_core(
        words,
        map.test_results_size_bytes,
        tt_cxy_pair(worker.chip, worker.core.noc_core),
        map.test_results_address - map.kernel_view_offset);
    TT_FATAL(words.size() == num_words, "Result read on chip {} returned {} words", worker.chip, words.size());
    log_info(
        tt::LogTest,
        "Chip {} worker {} status {:#x} bytes {}",
        worker.chip,
        worker.core.logical.str(),
        words[TT_FABRIC_STATUS_INDEX],
        bytes_of(words));
    return words;
}

}  // namespace

Line line_ends() {
    const auto& control_plane = MetalContext::instance().get_control_plane();
    const auto num_chips = static_cast<uint32_t>(tt::tt_metal::GetNumAvailableDevices());
    const MeshId mesh = control_plane.get_fabric_node_id_from_physical_chip_id(0).mesh_id;
    const FabricNodeId first(mesh, 0);
    const FabricNodeId last(mesh, num_chips - 1);
    return {worker_beside_router(first, last), worker_beside_router(last, first), num_chips - 1};
}

void KeraunosCoLaunchedWorkers::TearDown() {
    if (!devices_.empty()) {
        DoTearDownTestSuite();
    }
}

void KeraunosCoLaunchedWorkers::run_line_unicast(const NocLinks& noc_links) {
    if (MetalContext::instance().get_cluster().arch() != tt::ARCH::QUASAR) {
        GTEST_SKIP() << "Keraunos test lines only";
    }
    const auto num_chips = static_cast<uint32_t>(tt::tt_metal::GetNumAvailableDevices());
    for (const auto& [a, b] : noc_links) {
        if (std::max(a, b) >= num_chips) {
            GTEST_SKIP() << "Link " << a << "-" << b << " is not on this " << num_chips << "-chip line";
        }
    }
    const uint32_t seed = std::random_device{}();

    std::vector<PreparedRouter> prepared;
    std::once_flag routers_prepared;
    auto& env = tt::tt_metal::MetalEnvAccessor(MetalContext::instance().get_env()).impl();
    env.set_fabric_program_extension([&](tt::tt_metal::IDevice*, tt::tt_metal::Program&) {
        // Every call waits here, so the records and staging words of every chip are written before the first device
        // is configured. A throw lets a waiting call rerun the write, which can append duplicate entries to prepared.
        std::call_once(routers_prepared, [&] {
            if (is_mock()) {
                return;
            }
            const auto& control_plane = MetalContext::instance().get_control_plane();
            const MeshId mesh = control_plane.get_fabric_node_id_from_physical_chip_id(0).mesh_id;
            for (uint32_t i = 0; i < num_chips; ++i) {
                const auto routers = prepare_routers(
                    control_plane.get_physical_chip_id_from_fabric_node_id(FabricNodeId(mesh, i)), noc_links);
                prepared.insert(prepared.end(), routers.begin(), routers.end());
            }
        });
    });
    DoSetUpTestSuite(tt::tt_fabric::FabricConfig::FABRIC_1D);
    ASSERT_GE(devices_.size(), 2u);

    const Line line = line_ends();
    const auto& sender_device = devices_map_.at(line.sender.chip);
    const auto& receiver_device = devices_map_.at(line.receiver.chip);
    const WorkerMemMap map = generate_worker_mem_map(sender_device, tt::tt_fabric::Topology::Linear);
    log_info(
        tt::LogTest,
        "Line: {} hops, sender chip {} router chan {} worker {}, receiver chip {} router chan {} worker {} NOC ({}, "
        "{}); results at {:#x}",
        line.num_hops,
        line.sender.chip,
        line.sender.router_chan,
        line.sender.core.logical.str(),
        line.receiver.chip,
        line.receiver.router_chan,
        line.receiver.core.logical.str(),
        line.receiver.core.noc_core.x,
        line.receiver.core.noc_core.y,
        map.test_results_address);
    LinePrograms workers = create_line_programs(line, map, seed);
    if (is_mock()) {
        workers.sender.impl().compile(&tt::tt_metal::slow_dispatch::physical_device_from_unit_mesh(*sender_device));
        workers.receiver.impl().compile(&tt::tt_metal::slow_dispatch::physical_device_from_unit_mesh(*receiver_device));
        GTEST_SKIP() << "Mock: kernels compiled. MockChip reads and writes are no-ops, so nothing launches.";
    }

    // Fabric is up, so this run launched every router, and none has been told to terminate. A router's status word
    // survives from the case before on the same server, so TERMINATED here is that case's, and without this check the
    // TERMINATED after close below could be too.
    const auto routers = get_routers(/*masters_only=*/false);
    for (const auto& router : routers) {
        EXPECT_NE(read_edm_status(router), static_cast<uint32_t>(tt::tt_fabric::EDMStatus::TERMINATED))
            << "chip " << router.chip << " router " << router.logical.str() << " reads TERMINATED as fabric comes up";
    }

    // Each record the extension set was valid when it found it, so device init had written it, and still holds the NOC
    // kind after launch. Every other record holds the Ethernet kind device init writes. One router per end of
    // each chosen link: the mesh descriptors have one channel between neighbours.
    ASSERT_EQ(prepared.size(), routers.size()) << "the extension did not prepare every router";
    EXPECT_EQ(
        static_cast<size_t>(std::count_if(prepared.begin(), prepared.end(), [](const auto& r) { return r.noc; })),
        2 * noc_links.size());
    const auto prepared_router = [&](const Router& router) {
        const auto it = std::find_if(prepared.begin(), prepared.end(), [&](const PreparedRouter& r) {
            return r.chip == router.chip && r.virtual_core == router.virtual_core;
        });
        TT_FATAL(it != prepared.end(), "chip {} router {} was not prepared", router.chip, router.logical.str());
        return *it;
    };
    for (const auto& router : routers) {
        const PreparedRouter& prep = prepared_router(router);
        const EthLink now = read_eth_link(router.chip, router.virtual_core);
        const std::string where = fmt::format("chip {} router {}", router.chip, router.logical.str());
        log_info(
            tt::LogTest, "{} link record peer_tile {} valid {:#x} kind {}", where, now.peer_tile, now.valid, now.kind);
        EXPECT_EQ(now.valid, ETH_LINK_VALID) << where;
        EXPECT_EQ(prep.as_found.valid, ETH_LINK_VALID) << where;
        EXPECT_EQ(prep.as_found.kind, ETH_LINK_KIND_ETHERNET) << where;
        EXPECT_EQ(now.kind, prep.noc ? ETH_LINK_KIND_NOC : ETH_LINK_KIND_ETHERNET) << where;
    }

    // The workers run as ordinary programs on idle Ethernet cores, launched now that fabric is up.
    RunProgramNonblocking(receiver_device, std::move(workers.receiver));
    tt::tt_metal::LaunchProgram(*sender_device, std::move(workers.sender));
    WaitForSingleProgramDone(receiver_device);

    const uint64_t expected_bytes = uint64_t{map.packet_payload_size_bytes} * kNumPackets;
    for (const auto& [worker, role] : {std::pair{line.receiver, "receiver"}, std::pair{line.sender, "sender"}}) {
        const auto results = read_results(worker, map);
        EXPECT_EQ(results[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS) << role;
        EXPECT_EQ(bytes_of(results), expected_bytes) << role;
    }

    DoTearDownTestSuite();
    for (const auto& router : routers) {
        EXPECT_EQ(
            wait_for_router_terminated(router, std::chrono::duration_cast<std::chrono::seconds>(kTimeout)),
            static_cast<uint32_t>(tt::tt_fabric::EDMStatus::TERMINATED))
            << "chip " << router.chip << " router " << router.logical.str();
    }
}

}  // namespace tt::tt_fabric::fabric_router_tests
