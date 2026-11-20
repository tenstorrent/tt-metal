// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "co_launched_line.hpp"

#include <gtest/gtest.h>

#include <fmt/format.h>

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
#include "impl/context/metal_context.hpp"
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

void KeraunosCoLaunchedWorkers::run_line_unicast() {
    if (MetalContext::instance().get_cluster().arch() != tt::ARCH::QUASAR) {
        GTEST_SKIP() << "Keraunos test lines only";
    }
    const uint32_t seed = std::random_device{}();
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
