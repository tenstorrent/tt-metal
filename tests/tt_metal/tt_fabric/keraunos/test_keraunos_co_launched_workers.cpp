// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fabric unicast on the Keraunos test lines in this directory, with the sender and receiver on hart 1 (DM1) of the end
// chips' router CCEs, co-launched with the routers through the fabric program extension: one hop on the two-chip
// descriptors, two hops through chip M on the three-chip ones. Run with TT_METAL_SLOW_DISPATCH_MODE=1. On Mock the
// test checks the launch messages and skips: UMD MockChip reads and writes are no-ops.
//
// KeraunosCoLaunchedWorkersNocLinks runs the same line with chosen links backed by the NOC: the fabric program
// extension sets the kind word of the link record on both ends of each chosen link (PLAN-0008 Phase 1b, 1c; decisions
// 62db7e799795-D2 and -D3).

#include <gtest/gtest.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "fabric_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "llrt/hal.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/fabric_init.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"

namespace tt::tt_fabric::fabric_router_tests {
namespace {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::HalL1MemAddrType;
using tt::tt_metal::HalProgrammableCoreType;
using tt::tt_metal::MetalContext;

constexpr uint32_t kNumPackets = 5;
constexpr uint32_t kResultsSizeBytes = 128;
// MEM_IERISC_UNCACHED_BASE (hw/inc/internal/tt-2xx/quasar/dev_mem_map.h): the CCE's uncached view of its SRAM. The
// workers address their buffers, their results and the router's status word through it, so stores reach SRAM, where
// the iDMA, the router and the host see them.
constexpr uint32_t kCceUncachedAlias = 0x800000;
constexpr auto kTimeout = std::chrono::minutes(20);

// The link record internal_::cce_eth::EthLink (tt-2xx/quasar/eth_cce_send.h) as the test reads it: `valid` in word 1,
// and per PLAN-0008 § Design `kind` in word 2, the first reserved word. These mirror EthLink.
constexpr uint32_t kEthLinkValidWord = 1;
constexpr uint32_t kEthLinkValid = 0x4B4E494C;  // MEM_IERISC_FABRIC_ETH_LINK_VALID
constexpr uint32_t kEthLinkKindWord = 2;
constexpr uint32_t kEthLinkKindEthernet = 0;
constexpr uint32_t kEthLinkKindNoc = 1;
using EthLinkRecord = std::array<uint32_t, 4>;

// Links of the line backed by the NOC, each by its two ends' fabric chip ids. Empty: every link is Ethernet.
using NocLinks = std::vector<std::pair<uint32_t, uint32_t>>;

const std::string kTxKernel = "tests/tt_metal/tt_fabric/keraunos/kernels/co_launched_1d_tx.cpp";
const std::string kRxKernel = "tests/tt_metal/tt_fabric/keraunos/kernels/co_launched_1d_rx.cpp";

bool is_mock() {
    return MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Mock;
}

// A worker beside a router: the router's Ethernet channel and core, on DM1.
struct Worker {
    FabricNodeId node;
    ChipId chip;
    chan_id_t router_chan;
    CoreCoord logical;
    CoreCoord virtual_core;
};

// The line's two ends: the sender on fabric chip 0, the receiver on the last fabric chip, each beside its router that
// faces the other end.
struct Line {
    Worker sender;
    Worker receiver;
    uint32_t num_hops;
};

Worker worker_beside_router(FabricNodeId node, FabricNodeId toward) {
    const auto& control_plane = MetalContext::instance().get_control_plane();
    const auto& cluster = MetalContext::instance().get_cluster();
    const auto chans = control_plane.get_forwarding_eth_chans_to_chip(node, toward);
    TT_FATAL(!chans.empty(), "No router on {} forwards toward {}", node, toward);
    const ChipId chip = control_plane.get_physical_chip_id_from_fabric_node_id(node);
    const CoreCoord logical = cluster.get_soc_desc(chip).get_eth_core_for_channel(chans.front(), tt::CoordSystem::LOGICAL);
    return {node, chip, chans.front(), logical, cluster.get_virtual_coordinate_from_logical_coordinates(chip, logical, CoreType::ETH)};
}

Line line_ends() {
    const auto& control_plane = MetalContext::instance().get_control_plane();
    const auto num_chips = static_cast<uint32_t>(tt::tt_metal::GetNumAvailableDevices());
    const MeshId mesh = control_plane.get_fabric_node_id_from_physical_chip_id(0).mesh_id;
    const FabricNodeId first(mesh, 0);
    const FabricNodeId last(mesh, num_chips - 1);
    return {worker_beside_router(first, last), worker_beside_router(last, first), num_chips - 1};
}

// The workers' buffers and results, in the kernels' view: at the top of the CCE's SRAM, above everything the router
// places. Fatal when the router's region reaches them.
WorkerMemMap co_launched_worker_mem_map() {
    const auto& hal = MetalContext::instance().hal();
    const uint32_t l1_size = hal.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::BASE);
    const auto payload = static_cast<uint32_t>(get_tt_fabric_max_payload_size_bytes());
    const uint32_t results = l1_size - kResultsSizeBytes;
    const uint32_t data = (results - kNumPackets * payload) & ~63u;
    const size_t router_end = MetalContext::instance()
                                  .get_control_plane()
                                  .get_fabric_context()
                                  .get_builder_context()
                                  .get_fabric_router_config()
                                  .allocated_l1_end_address;
    TT_FATAL(
        data >= router_end,
        "Co-launched worker buffers at {:#x} overlap the router's L1, which ends at {:#x}",
        data,
        router_end);
    return {
        .source_l1_buffer_address = data + kCceUncachedAlias,
        .packet_payload_size_bytes = payload,
        .test_results_address = results + kCceUncachedAlias,
        .target_address = data + kCceUncachedAlias,
        .notification_mailbox_address = 0,
        .test_results_size_bytes = kResultsSizeBytes,
        .kernel_view_offset = kCceUncachedAlias};
}

// Adds `kernel_file` on DM1 of `worker`'s router core, with the stock kernel's runtime args then the wrapper's
// (co_launched_worker.hpp).
void add_worker_kernel(
    tt::tt_metal::Program& program,
    const Worker& worker,
    const std::string& kernel_file,
    const WorkerMemMap& map,
    std::vector<uint32_t> runtime_args,
    uint32_t nonce) {
    const auto ready = MetalContext::instance()
                           .get_control_plane()
                           .get_fabric_context()
                           .get_builder_context()
                           .get_fabric_router_ready_address_and_signal();
    TT_FATAL(ready.has_value(), "The fabric config has no READY_FOR_TRAFFIC handshake");
    const std::vector<uint32_t> compile_args = {
        map.test_results_address,
        map.test_results_size_bytes,
        map.target_address,
        0, /* use_dram_dst */
        0, /* is_2d_fabric */
        0, /* is_chip_multicast */
        0 /* additional_dir */};
    const auto worker_args_index = static_cast<uint32_t>(runtime_args.size());
    runtime_args.insert(
        runtime_args.end(),
        {ready->first + map.kernel_view_offset, static_cast<uint32_t>(ready->second), nonce});
    const auto kernel = tt::tt_metal::CreateKernel(
        program,
        kernel_file,
        worker.logical,
        tt::tt_metal::EthernetConfig{
            .noc = tt::tt_metal::NOC::NOC_1,
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .compile_args = compile_args,
            .named_compile_args = {{"WORKER_ARGS_INDEX", worker_args_index}}});
    tt::tt_metal::SetRuntimeArgs(program, kernel, worker.logical, runtime_args);
}

// The extension: the sender on fabric chip 0's router core, the receiver on the last chip's, nothing elsewhere.
void add_line_workers(
    tt::tt_metal::IDevice* device, tt::tt_metal::Program& program, uint32_t nonce, uint32_t seed) {
    const Line line = line_ends();
    const WorkerMemMap map = co_launched_worker_mem_map();
    if (device->id() == line.sender.chip) {
        std::vector<uint32_t> args = {
            map.source_l1_buffer_address,
            map.packet_payload_size_bytes,
            kNumPackets,
            seed,
            static_cast<uint32_t>(line.receiver.virtual_core.x),
            static_cast<uint32_t>(line.receiver.virtual_core.y),
            line.num_hops,
            1, /* fwd_range */
            line.receiver.node.chip_id,
            *line.receiver.node.mesh_id};
        const auto teardown_semaphore =
            tt::tt_metal::CreateSemaphore(program, line.sender.logical, 0, CoreType::ETH);
        const auto buffer_index_semaphore =
            tt::tt_metal::CreateSemaphore(program, line.sender.logical, 0, CoreType::ETH);
        append_worker_to_fabric_edm_sender_rt_args(
            line.sender.router_chan, teardown_semaphore, buffer_index_semaphore, args);
        add_worker_kernel(program, line.sender, kTxKernel, map, std::move(args), nonce);
    } else if (device->id() == line.receiver.chip) {
        add_worker_kernel(
            program, line.receiver, kRxKernel, map, {map.packet_payload_size_bytes, kNumPackets, seed}, nonce);
    }
}

// A router whose link record the extension set to the NOC kind, and the record as the extension found it.
struct NocRouter {
    ChipId chip;
    CoreCoord virtual_core;
    EthLinkRecord as_found;
};

uint64_t eth_link_address() {
    return MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::ETH_LINK);
}

EthLinkRecord read_eth_link(ChipId chip, const CoreCoord& virtual_core) {
    std::vector<uint32_t> words;
    MetalContext::instance().get_cluster().read_core(
        words, sizeof(EthLinkRecord), tt_cxy_pair(chip, virtual_core), eth_link_address());
    TT_FATAL(words.size() == 4, "Link record read on chip {} returned {} words", chip, words.size());
    return {words[0], words[1], words[2], words[3]};
}

// Sets the NOC kind in the record of every fabric router of `chip` whose link is in `noc_links`, and returns those
// routers. Called from the extension under std::call_once, after device init wrote the records and before any device
// is configured (PLAN-0008 § Where the Phase 1 test writes `kind`, hub 62db7e799795-D8).
std::vector<NocRouter> set_noc_links(ChipId chip, const NocLinks& noc_links) {
    const auto& control_plane = MetalContext::instance().get_control_plane();
    const auto& cluster = MetalContext::instance().get_cluster();
    const FabricNodeId node = control_plane.get_fabric_node_id_from_physical_chip_id(chip);
    std::vector<NocRouter> routers;
    for (const auto& [chan, direction] : control_plane.get_active_fabric_eth_channels(node)) {
        const FabricNodeId peer = control_plane.get_connected_mesh_chip_chan_ids(node, chan).first;
        const bool noc = std::any_of(noc_links.begin(), noc_links.end(), [&](const auto& link) {
            return (link.first == node.chip_id && link.second == peer.chip_id) ||
                   (link.second == node.chip_id && link.first == peer.chip_id);
        });
        if (!noc) {
            continue;
        }
        const CoreCoord logical = cluster.get_soc_desc(chip).get_eth_core_for_channel(chan, tt::CoordSystem::LOGICAL);
        const CoreCoord virtual_core =
            cluster.get_virtual_coordinate_from_logical_coordinates(chip, logical, CoreType::ETH);
        NocRouter router{chip, virtual_core, read_eth_link(chip, virtual_core)};
        EthLinkRecord record = router.as_found;
        record[kEthLinkKindWord] = kEthLinkKindNoc;
        cluster.write_core_immediate(
            record.data(), sizeof(record), tt_cxy_pair(chip, virtual_core), eth_link_address());
        routers.push_back(router);
    }
    return routers;
}

// `num_words` words of the worker's result block from `first_word`, read through the cluster, so it works while the
// device is open or closed. Fatal on a short read.
std::vector<uint32_t> read_results(
    const Worker& worker, const WorkerMemMap& map, uint32_t first_word, uint32_t num_words) {
    std::vector<uint32_t> words;
    MetalContext::instance().get_cluster().read_core(
        words,
        num_words * sizeof(uint32_t),
        tt_cxy_pair(worker.chip, worker.virtual_core),
        map.test_results_address - map.kernel_view_offset + first_word * sizeof(uint32_t));
    TT_FATAL(words.size() == num_words, "Result read on chip {} returned {} words", worker.chip, words.size());
    return words;
}

// The byte count a stock 1D kernel wrote to its result block.
uint64_t bytes_of(const std::vector<uint32_t>& results) {
    return (static_cast<uint64_t>(results[TT_FABRIC_WORD_CNT_INDEX + 1]) << 32) | results[TT_FABRIC_WORD_CNT_INDEX];
}

// Polls the nonce word, the last of the worker's result block, once a second until it holds `nonce` or `deadline` has
// passed, then reads the whole block. The nonce is the worker's last store, so a block read after it is complete.
std::vector<uint32_t> wait_for_results(
    const Worker& worker, const WorkerMemMap& map, uint32_t nonce, std::chrono::steady_clock::time_point deadline) {
    const uint32_t nonce_word = map.test_results_size_bytes / sizeof(uint32_t) - 1;
    while (read_results(worker, map, nonce_word, 1)[0] != nonce && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    auto words = read_results(worker, map, 0, nonce_word + 1);
    log_info(
        tt::LogTest,
        "Chip {} worker {} status {:#x} bytes {} nonce {:#x} (run {:#x})",
        worker.chip,
        worker.logical.str(),
        words[TT_FABRIC_STATUS_INDEX],
        bytes_of(words),
        words.back(),
        nonce);
    return words;
}

// Every fabric router of every open device, in the form BaseFabricFixture reads EDM status from.
std::vector<BaseFabricFixture::MasterRouter> all_routers() {
    const auto& control_plane = MetalContext::instance().get_control_plane();
    const auto& cluster = MetalContext::instance().get_cluster();
    const auto ready =
        control_plane.get_fabric_context().get_builder_context().get_fabric_router_ready_address_and_signal();
    TT_FATAL(ready.has_value(), "The fabric config has no READY_FOR_TRAFFIC handshake");
    std::vector<BaseFabricFixture::MasterRouter> routers;
    for (const auto& [chip, mesh] : BaseFabricFixture::devices_map_) {
        const auto node = control_plane.get_fabric_node_id_from_physical_chip_id(chip);
        for (const auto& [chan, direction] : control_plane.get_active_fabric_eth_channels(node)) {
            const CoreCoord logical = cluster.get_soc_desc(chip).get_eth_core_for_channel(chan, tt::CoordSystem::LOGICAL);
            routers.push_back(
                {chip,
                 logical,
                 cluster.get_virtual_coordinate_from_logical_coordinates(chip, logical, CoreType::ETH),
                 ready->first});
        }
    }
    return routers;
}

// Logs every router kernel's arguments, for comparison with compat's forwarding config.
void log_router_args(ChipId chip, tt::tt_metal::Program& program) {
    const uint32_t index = MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH);
    for (const auto& [handle, kernel] : program.impl().get_kernels(index)) {
        if (kernel->name().find("fabric_erisc_router") == std::string::npos) {
            continue;
        }
        const auto named_args = kernel->named_compile_time_args();
        std::vector<std::string> named;
        for (const auto& [name, value] : std::map<std::string, uint32_t>(named_args.begin(), named_args.end())) {
            named.push_back(fmt::format("{}={}", name, value));
        }
        for (const auto& core : kernel->cores_with_runtime_args()) {
            log_info(
                tt::LogTest,
                "Chip {} router {}: named {} | ct {} | rt {}",
                chip,
                core.str(),
                fmt::join(named, ", "),
                fmt::join(kernel->compile_time_args(), ", "),
                fmt::join(kernel->runtime_args(core), ", "));
        }
    }
}

// Checks that `worker`'s router core launches its router on DM0 and the worker on DM1 in one launch message. Quasar
// Mock leaves the kernel text offsets unset (finalize_kernel_bins), so they are checked off Mock only.
void expect_co_launched(tt::tt_metal::Program& program, const Worker& worker) {
    const auto& fixture_device = BaseFabricFixture::devices_map_.at(worker.chip);
    program.impl().finalize_offsets(&tt::tt_metal::slow_dispatch::physical_device_from_unit_mesh(*fixture_device));
    const uint32_t index = MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH);
    auto* group = program.impl().kernels_on_core(worker.logical, index);
    ASSERT_NE(group, nullptr) << "chip " << worker.chip << " core " << worker.logical.str();
    auto kernel_config = group->launch_msg.view().kernel_config();
    const uint32_t enables = kernel_config.enables();
    const uint32_t dm0_offset = kernel_config.kernel_text_offset()[0];
    const uint32_t dm1_offset = kernel_config.kernel_text_offset()[1];
    log_info(
        tt::LogTest,
        "Chip {} core {}: {} kernels, enables {:#x}, kernel text offsets DM0 {:#x} DM1 {:#x}",
        worker.chip,
        worker.logical.str(),
        group->kernel_ids.size(),
        enables,
        dm0_offset,
        dm1_offset);
    EXPECT_EQ(group->kernel_ids.size(), 2u) << "chip " << worker.chip;
    EXPECT_EQ(enables & 0x3u, 0x3u) << "chip " << worker.chip;
    if (!is_mock()) {
        EXPECT_NE(dm0_offset, dm1_offset) << "chip " << worker.chip;
    }
}

class KeraunosCoLaunchedWorkers : public BaseFabricFixture {
protected:
    // Sends kNumPackets from the line's first chip to its last with the links in `noc_links` on the NOC and the rest on
    // Ethernet, then checks the link records, both workers' results and that every router terminates.
    void run_line_unicast(const NocLinks& noc_links);

    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}

    void TearDown() override {
        if (!devices_.empty()) {
            DoTearDownTestSuite();
        }
        set_fabric_program_extension({});
    }
};

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
    std::random_device random;
    const uint32_t nonce = random() | 1u;
    const uint32_t seed = random();

    std::mutex programs_mutex;
    std::map<ChipId, tt::tt_metal::Program*> programs;
    std::vector<NocRouter> noc_routers;
    std::once_flag noc_links_written;
    set_fabric_program_extension([&](tt::tt_metal::IDevice* device, tt::tt_metal::Program& program) {
        add_line_workers(device, program, nonce, seed);
        // Every call waits here, so the records of every chip are written before the first device is configured.
        std::call_once(noc_links_written, [&] {
            if (is_mock()) {
                return;
            }
            const auto& control_plane = MetalContext::instance().get_control_plane();
            const MeshId mesh = control_plane.get_fabric_node_id_from_physical_chip_id(0).mesh_id;
            for (uint32_t i = 0; i < num_chips; ++i) {
                const auto routers = set_noc_links(
                    control_plane.get_physical_chip_id_from_fabric_node_id(FabricNodeId(mesh, i)), noc_links);
                noc_routers.insert(noc_routers.end(), routers.begin(), routers.end());
            }
        });
        const std::scoped_lock lock(programs_mutex);
        programs[device->id()] = &program;
    });
    DoSetUpTestSuite(tt::tt_fabric::FabricConfig::FABRIC_1D);
    ASSERT_GE(devices_.size(), 2u);

    const Line line = line_ends();
    const WorkerMemMap map = co_launched_worker_mem_map();
    log_info(
        tt::LogTest,
        "Line: {} hops, sender chip {} router chan {} core {}, receiver chip {} router chan {} core {} NOC ({}, {}); "
        "results at {:#x}, nonce {:#x}",
        line.num_hops,
        line.sender.chip,
        line.sender.router_chan,
        line.sender.logical.str(),
        line.receiver.chip,
        line.receiver.router_chan,
        line.receiver.logical.str(),
        line.receiver.virtual_core.x,
        line.receiver.virtual_core.y,
        map.test_results_address,
        nonce);
    ASSERT_EQ(programs.size(), devices_.size()) << "the extension did not see every device";
    expect_co_launched(*programs.at(line.sender.chip), line.sender);
    expect_co_launched(*programs.at(line.receiver.chip), line.receiver);
    if (is_mock()) {
        for (const auto& [chip, program] : programs) {
            log_router_args(chip, *program);
        }
        GTEST_SKIP() << "Mock: kernels compiled and launch messages checked. MockChip reads and writes are no-ops, so "
                        "nothing launches.";
    }

    // Each record the extension set was valid when it found it, so device init had written it, and still holds the NOC
    // kind after launch. Every other record holds the Ethernet kind device init writes today. One router per end of
    // each chosen link: the mesh descriptors have one channel between neighbours.
    EXPECT_EQ(noc_routers.size(), 2 * noc_links.size());
    for (const auto& router : all_routers()) {
        const auto noc = std::find_if(noc_routers.begin(), noc_routers.end(), [&](const NocRouter& r) {
            return r.chip == router.chip && r.virtual_core == router.virtual_core;
        });
        const EthLinkRecord now = read_eth_link(router.chip, router.virtual_core);
        const std::string where = fmt::format("chip {} router {}", router.chip, router.logical.str());
        log_info(tt::LogTest, "{} link record {:#x}", where, fmt::join(now, " "));
        EXPECT_EQ(now[kEthLinkValidWord], kEthLinkValid) << where;
        if (noc == noc_routers.end()) {
            EXPECT_EQ(now[kEthLinkKindWord], kEthLinkKindEthernet) << where;
            continue;
        }
        EXPECT_EQ(noc->as_found[kEthLinkValidWord], kEthLinkValid) << where;
        EXPECT_EQ(noc->as_found[kEthLinkKindWord], kEthLinkKindEthernet) << where;
        EXPECT_EQ(now[kEthLinkKindWord], kEthLinkKindNoc) << where;
    }

    const auto deadline = std::chrono::steady_clock::now() + kTimeout;
    const uint64_t expected_bytes = uint64_t{map.packet_payload_size_bytes} * kNumPackets;
    for (const auto& [worker, role] : {std::pair{line.receiver, "receiver"}, std::pair{line.sender, "sender"}}) {
        const auto results = wait_for_results(worker, map, nonce, deadline);
        EXPECT_EQ(results.back(), nonce) << role << " did not finish this run";
        EXPECT_EQ(results[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS) << role;
        EXPECT_EQ(bytes_of(results), expected_bytes) << role;
    }

    const auto routers = all_routers();
    DoTearDownTestSuite();
    for (const auto& router : routers) {
        EXPECT_EQ(
            wait_for_router_terminated(router, std::chrono::duration_cast<std::chrono::seconds>(kTimeout)),
            static_cast<uint32_t>(tt::tt_fabric::EDMStatus::TERMINATED))
            << "chip " << router.chip << " router " << router.logical.str();
    }
}

TEST_F(KeraunosCoLaunchedWorkers, LineUnicast) { run_line_unicast({}); }

class KeraunosCoLaunchedWorkersNocLinks : public KeraunosCoLaunchedWorkers,
                                          public ::testing::WithParamInterface<NocLinks> {};

TEST_P(KeraunosCoLaunchedWorkersNocLinks, LineUnicast) { run_line_unicast(GetParam()); }

// Fabric chips 0, 1, 2 are A, M, B on the three-chip line.
INSTANTIATE_TEST_SUITE_P(
    Links,
    KeraunosCoLaunchedWorkersNocLinks,
    ::testing::Values(NocLinks{{0, 1}}, NocLinks{{0, 1}, {1, 2}}),
    [](const ::testing::TestParamInfo<NocLinks>& info) { return info.param.size() == 1 ? "AM" : "AM_MB"; });

}  // namespace
}  // namespace tt::tt_fabric::fabric_router_tests
