// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// tt-coremon idle-eth aggregator, milestone M1 (PLAN_ETH_AGGREGATOR.md 4).
//
// Stands the aggregator kernel up on a REMOTE chip's idle ethernet core and
// checks that the journal it pushes lands, intact and advancing, in an idle-eth
// L1 slot on the MMIO chip -- read over plain PCIe, never over the tunnel.
//
// The transport itself is already proven: TestFabricWriteReachesRemoteL1_Control
// in test_fabric_pcie_host_target.cpp shows a fabric write from an idle eth core
// reaching ordinary L1 on the MMIO chip. What this adds is the actual payload:
// a real sweep of the Tensix util_sampler rings, a real journal, and the
// header-last ordering the host decode depends on.

#include <gtest/gtest.h>
#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "impl/kernels/kernel.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "fabric_fixture.hpp"

#include "hw/inc/util_aggregator.h"
#include "tools/ttnvtop/host/agg_core_select.hpp"

namespace tt::tt_fabric::fabric_router_tests {

// ~1 ms at 1 GHz. The Tensix rings tick at UTIL_SAMPLER_DEFAULT_PERIOD_CYCLES
// (also 1 ms), so this sweeps at roughly the production rate. Deliberately not
// a tight loop: a telemetry stream that keeps the core hot raises AICLK and
// perturbs the very utilization figures the aggregator reports.
constexpr uint32_t kSweepIntervalCycles = 1000000u;

// The host never names MEM_UTIL_SAMPLER_BASE: it is a firmware-only macro from
// dev_mem_map.h, which is not on this target's include path. We pass each core's
// NOC address with a zeroed local-address field and the kernel ORs the sampler
// base in -- see the ring_addr_table note in eth_aggregator.cpp.
// L1 layout inside the sender eth core's UNRESERVED region. All 16 B aligned:
// every one of these is either a NOC read destination or a fabric write source.
struct SenderL1 {
    uint32_t ring_table;    // num_cores * 8 B
    uint32_t last_head;     // num_cores * 4 B
    uint32_t head_scratch;  // num_cores * 16 B
    uint32_t seq_scratch;   // num_cores * 4 B
    uint32_t hdr_stage;     // 64 B
    uint32_t stage;         // stage_entries_max * 32 B
    uint32_t dbg;           // 16 B liveness markers
    uint32_t end;
};

static SenderL1 layout_sender_l1(uint32_t base, uint32_t num_cores, uint32_t stage_entries_max) {
    auto align16 = [](uint32_t v) { return (v + 15u) & ~15u; };
    SenderL1 l{};
    uint32_t p = align16(base);
    l.ring_table = p;
    p = align16(p + num_cores * 8u);
    l.last_head = p;
    p = align16(p + num_cores * 4u);
    l.head_scratch = p;
    p = align16(p + num_cores * 16u);
    l.seq_scratch = p;
    p = align16(p + num_cores * 4u);
    l.hdr_stage = p;
    p = align16(p + sizeof(util_agg_msg_t));
    l.stage = p;
    p = align16(p + stage_entries_max * sizeof(util_agg_entry_t));
    l.dbg = p;
    p = align16(p + 16u);
    l.end = p;
    return l;
}

// Build the 64-bit NOC address of `addr` on the core at TRANSLATED (tx, ty).
//
// Host-side and whole, never rebuilt in the kernel: NOC_X_PHYS_COORD() resolves
// against the kernel's own noc_index, and getting that wrong lands the access
// silently on the mirrored core.
static uint64_t noc_addr_translated(uint32_t tx, uint32_t ty, uint64_t addr) {
    constexpr uint32_t kLocalBits = 36, kNodeIdBits = 6, kCoordOff = 4;
    const uint32_t xy = (ty << ((kLocalBits % 32) + kNodeIdBits)) | (tx << (kLocalBits % 32));
    return ((uint64_t)xy << (kLocalBits - kCoordOff)) | addr;
}

TEST_F(Fabric1DFixture, TestEthAggregatorJournalLands) {
    if (!slow_dispatch_) {
        GTEST_SKIP() << "IDLE_ETH launch needs TT_METAL_SLOW_DISPATCH_MODE";
    }
    const auto& devices = this->get_devices();
    ASSERT_GE(devices.size(), 2u);
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    // Sender = a remote chip. That is the case the aggregator exists for: the
    // host cannot read it without taking NON_MMIO and risking a multi-second
    // stall under load.
    tt::tt_metal::IDevice* sender = nullptr;
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> sender_mesh;
    for (const auto& d : devices) {
        auto* dev = d->get_devices()[0];
        if (cluster.get_associated_mmio_device(dev->id()) != dev->id()) {
            sender = dev;
            sender_mesh = d;
            break;
        }
    }
    if (sender == nullptr) {
        GTEST_SKIP() << "no remote chip";
    }
    const auto mmio_id = cluster.get_associated_mmio_device(sender->id());
    tt::tt_metal::IDevice* receiver = nullptr;
    for (const auto& d : devices) {
        if (d->get_devices()[0]->id() == mmio_id) {
            receiver = d->get_devices()[0];
        }
    }
    ASSERT_NE(receiver, nullptr);

    const auto send_pick = ttnvtop::select_aggregator_eth_core(sender);
    const auto recv_pick = ttnvtop::select_aggregator_eth_core(receiver);
    if (!send_pick.ok) {
        GTEST_SKIP() << "sender: " << send_pick.reason;
    }
    if (!recv_pick.ok) {
        GTEST_SKIP() << "receiver: " << recv_pick.reason;
    }
    const tt::tt_metal::CoreCoord send_core = send_pick.core;
    const tt::tt_metal::CoreCoord recv_core = recv_pick.core;
    log_info(
        tt::LogTest,
        "aggregator on chip {} eth ch{}, landing chip {} eth ch{}",
        sender->id(),
        send_core.y,
        receiver->id(),
        recv_core.y);

    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const uint32_t eth_l1 =
        hal.get_dev_addr(tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);
    const uint32_t eth_l1_size =
        hal.get_dev_size(tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);

    // The Tensix cores to sweep, in TRANSLATED coordinates.
    //
    // Translated, not NOC0, and taken from the soc descriptor rather than a
    // hardcoded grid: WH harvests 2 Tensix ROWS on every shipped part, the mask
    // differs chip to chip within one system, and 224 of 278 shipped descriptor
    // entries carry harvest_mask 0 (80 cores, not 64). See 2.1.
    const auto& ssoc = cluster.get_soc_desc(sender->id());
    const auto tensix = ssoc.get_cores(CoreType::TENSIX, CoordSystem::TRANSLATED);
    const uint32_t num_cores = static_cast<uint32_t>(tensix.size());
    ASSERT_GT(num_cores, 0u);
    ASSERT_LE(num_cores, UTIL_AGG_MAX_CORES);
    log_info(tt::LogTest, "aggregator: sweeping {} Tensix cores on remote chip {}", num_cores, sender->id());

    const uint32_t max_payload = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes());
    const uint32_t stage_entries_max = max_payload / sizeof(util_agg_entry_t);
    ASSERT_GT(stage_entries_max, 0u);

    const SenderL1 l1 = layout_sender_l1(eth_l1, num_cores, stage_entries_max);
    ASSERT_LT(l1.end - eth_l1, eth_l1_size) << "sender scratch does not fit in idle-eth UNRESERVED";

    // The landing journal on the MMIO chip. Sized by the header, and checked
    // against what idle-eth L1 actually has rather than assumed.
    const uint32_t landing_base = eth_l1;
    ASSERT_LE(UTIL_AGG_JOURNAL_BYTES, eth_l1_size)
        << "journal (" << UTIL_AGG_JOURNAL_BYTES << " B) exceeds idle-eth UNRESERVED (" << eth_l1_size << " B)";

    // Ring-address table: one host-computed 64-bit NOC address per Tensix core.
    std::vector<uint32_t> ring_tbl;
    ring_tbl.reserve(num_cores * 2);
    for (const auto& c : tensix) {
        const uint64_t a = noc_addr_translated(c.x, c.y, /*addr=*/0);
        ring_tbl.push_back(static_cast<uint32_t>(a & 0xFFFFFFFFull));
        ring_tbl.push_back(static_cast<uint32_t>(a >> 32));
    }
    tt::tt_metal::detail::WriteToDeviceL1(sender, send_core, l1.ring_table, ring_tbl, CoreType::ETH);

    // Clear the landing header so a stale journal from a previous run cannot be
    // mistaken for a live one.
    std::vector<uint32_t> zero(sizeof(util_agg_msg_t) / 4, 0u);
    tt::tt_metal::detail::WriteToDeviceL1(receiver, recv_core, landing_base, zero, CoreType::ETH);

    const auto& rsoc = cluster.get_soc_desc(mmio_id);
    const auto rt = rsoc.translate_coord_to(
        tt::umd::CoreCoord(recv_core.x, recv_core.y, CoreType::ETH, CoordSystem::LOGICAL), CoordSystem::TRANSLATED);
    const uint64_t dest_base = noc_addr_translated(rt.x, rt.y, landing_base);
    log_info(
        tt::LogTest,
        "aggregator: landing on chip {} eth {} translated ({},{}) dest=0x{:016x}",
        mmio_id,
        recv_core.str(),
        rt.x,
        rt.y,
        dest_base);

    auto program = tt::tt_metal::CreateProgram();
    auto kernel = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/tools/ttnvtop/kernels/eth_aggregator.cpp",
        send_core,
        tt::tt_metal::EthernetConfig{
            .eth_mode = tt::tt_metal::Eth::IDLE, .processor = tt::tt_metal::DataMovementProcessor::RISCV_0});

    auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto src_node = cp.get_fabric_node_id_from_physical_chip_id(sender->id());
    const auto dst_node = cp.get_fabric_node_id_from_physical_chip_id(receiver->id());

    std::vector<uint32_t> rt_args = {
        num_cores,
        l1.ring_table,
        l1.last_head,
        l1.head_scratch,
        l1.stage,
        stage_entries_max,
        l1.hdr_stage,
        static_cast<uint32_t>(dest_base & 0xFFFFFFFFull),
        static_cast<uint32_t>(dest_base >> 32),
        UTIL_AGG_CAPACITY,
        static_cast<uint32_t>(src_node.chip_id),
        kSweepIntervalCycles,
        1u,  // unicast_hops: remote chip -> its own MMIO chip is one hop
        l1.seq_scratch,
        l1.dbg,
    };
    tt::tt_fabric::append_fabric_connection_rt_args(
        src_node, dst_node, /*link_idx=*/0, program, send_core, rt_args, CoreType::ETH);
    tt::tt_metal::SetRuntimeArgs(program, kernel, send_core, rt_args);

    this->RunProgramNonblocking(sender_mesh, program);

    // Two reads a second apart. One proves the journal arrived; the pair proves
    // it is LIVE -- a single sample cannot distinguish a running aggregator from
    // one that pushed a header once and died.
    auto read_hdr = [&]() {
        std::vector<uint32_t> h(sizeof(util_agg_msg_t) / 4, 0u);
        tt::tt_metal::detail::ReadFromDeviceL1(
            receiver, recv_core, landing_base, sizeof(util_agg_msg_t), h, CoreType::ETH);
        return h;
    };

    std::this_thread::sleep_for(std::chrono::seconds(3));
    auto h1 = read_hdr();
    std::this_thread::sleep_for(std::chrono::seconds(2));
    auto h2 = read_hdr();

    log_info(
        tt::LogTest,
        "aggregator: t1 magic=0x{:08x} head={} sweeps={} lost={} cores={}",
        h1[0],
        h1[2],
        h1[5],
        h1[6],
        h1[4]);
    log_info(
        tt::LogTest,
        "aggregator: t2 magic=0x{:08x} head={} sweeps={} lost={} cores={}",
        h2[0],
        h2[2],
        h2[5],
        h2[6],
        h2[4]);

    EXPECT_EQ(h1[0], UTIL_AGG_MAGIC) << "no journal header landed on the MMIO chip";
    EXPECT_EQ(h1[1], UTIL_AGG_VERSION);
    EXPECT_EQ(h1[4], num_cores);
    EXPECT_EQ(h1[3], UTIL_AGG_CAPACITY);

    const uint32_t sum1 = util_agg_hdr_checksum(h1[0], h1[1], h1[2], h1[3], h1[4], h1[5], h1[6], h1[7]);
    EXPECT_EQ(sum1, h1[8]) << "header checksum mismatch -- torn write, or the layout disagrees";

    EXPECT_GT(h2[5], h1[5]) << "sweep_count did not advance: aggregator is not running";
    EXPECT_GE(h2[2], h1[2]) << "head went backwards";

    // Entries must actually be arriving. A journal that only ever ships headers
    // is a working transport and a broken sweep.
    EXPECT_GT(h2[2], 0u) << "no journal entries were ever written -- sweep found no ring activity";

    if (h2[2] > 0) {
        const uint32_t n = std::min<uint32_t>(h2[2], 8u);
        std::vector<uint32_t> ent(n * sizeof(util_agg_entry_t) / 4, 0u);
        tt::tt_metal::detail::ReadFromDeviceL1(
            receiver,
            recv_core,
            landing_base + UTIL_AGG_JOURNAL_OFFSET,
            n * sizeof(util_agg_entry_t),
            ent,
            CoreType::ETH);
        for (uint32_t i = 0; i < n; i++) {
            const uint32_t* e = &ent[i * 8];
            log_info(
                tt::LogTest,
                "  entry[{}] wall=0x{:08x} kid={} fpu={} core_id={} seq={}",
                i,
                e[0],
                e[1],
                e[2],
                e[4],
                e[5]);
            EXPECT_LT(e[4], num_cores) << "entry " << i << " has an out-of-range core_id";
        }
    }
}

// Compile the aggregator kernel and stop there.
//
// TestEthAggregatorJournalLands needs a REMOTE chip, so it skips on every
// single-chip and all-MMIO system (a 4x p150a box, for instance). That would
// leave the kernel completely unbuilt on those hosts and let a syntax or type
// error reach the T3K as a runtime JIT failure. This runs anywhere there is a
// device and catches that at build time instead.
TEST_F(Fabric1DFixture, TestEthAggregatorKernelCompiles) {
    if (!slow_dispatch_) {
        GTEST_SKIP() << "IDLE_ETH needs TT_METAL_SLOW_DISPATCH_MODE";
    }
    const auto& devices = this->get_devices();
    if (devices.size() < 2u) {
        GTEST_SKIP() << "need two devices to resolve a fabric connection";
    }
    auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();

    // Find an ADJACENT pair. append_fabric_connection_rt_args rejects a
    // self-connection ("Expected different src and dst chip ids") and also any
    // pair with no route ("Could not find any forwarding direction"), and which
    // chips neighbour which varies by topology -- so probe rather than assume.
    for (size_t i = 0; i < devices.size(); i++) {
        for (size_t j = 0; j < devices.size(); j++) {
            if (i == j) {
                continue;
            }
            auto* dev = devices[i]->get_devices()[0];
            auto* peer = devices[j]->get_devices()[0];
            auto eth = dev->get_inactive_ethernet_cores();
            if (eth.empty()) {
                continue;
            }
            const tt::tt_metal::CoreCoord core = *eth.begin();

            auto program = tt::tt_metal::CreateProgram();
            auto kernel = tt::tt_metal::CreateKernel(
                program,
                "tt_metal/tools/ttnvtop/kernels/eth_aggregator.cpp",
                core,
                tt::tt_metal::EthernetConfig{
                    .eth_mode = tt::tt_metal::Eth::IDLE, .processor = tt::tt_metal::DataMovementProcessor::RISCV_0});

            std::vector<uint32_t> rt_args = {
                1u,
                0x10000u,
                0x10100u,
                0x10200u,
                0x11000u,
                8u,
                0x10800u,
                0u,
                0u,
                UTIL_AGG_CAPACITY,
                0u,
                1000u,
                1u,
                0x10300u,
                0x10400u};
            try {
                tt::tt_fabric::append_fabric_connection_rt_args(
                    cp.get_fabric_node_id_from_physical_chip_id(dev->id()),
                    cp.get_fabric_node_id_from_physical_chip_id(peer->id()),
                    /*link_idx=*/0,
                    program,
                    core,
                    rt_args,
                    CoreType::ETH);
            } catch (const std::exception&) {
                continue;  // not adjacent, or no free link -- try the next pair
            }
            tt::tt_metal::SetRuntimeArgs(program, kernel, core, rt_args);

            log_info(tt::LogTest, "compiling aggregator kernel for chip {} eth {}", dev->id(), core.str());
            tt::tt_metal::detail::CompileProgram(dev, program, /*force_slow_dispatch=*/true);
            SUCCEED();
            return;
        }
    }
    GTEST_SKIP() << "no adjacent device pair with a free idle eth core";
}

// Can the aggregator be launched with NO dispatch involvement at all?
//
// This is the production launch path, and it is the part that neither the
// transport tests nor the compile test cover. It needs no remote chip, so it
// runs on any system with an idle eth core -- including an all-MMIO Blackhole
// box where TestEthAggregatorJournalLands skips.
//
// LaunchProgram(force_slow_dispatch=true) writes binaries, runtime args and the
// launch message with plain host writes and then sets RUN_MSG_GO. The idle-erisc
// firmware is already sitting in `while (go_messages[0].signal != RUN_MSG_GO)`;
// dispatch appears in that firmware only AFTER the kernel returns, under
// `if (mode == DISPATCH_MODE_DEV)`, and our kernel never returns.
//
// wait_until_cores_done MUST be false: a kernel that never returns never writes
// RUN_MSG_DONE, so waiting for it would hang forever. That is the mechanism, not
// a bug -- see PLAN_ETH_AGGREGATOR.md 3.5.
TEST_F(Fabric1DFixture, TestEthAggregatorLaunchesWithoutDispatch) {
    const auto& devices = this->get_devices();
    ASSERT_GE(devices.size(), 1u);
    auto* dev = devices[0]->get_devices()[0];

    const auto pick = ttnvtop::select_aggregator_eth_core(dev);
    if (!pick.ok) {
        GTEST_SKIP() << pick.reason;
    }
    const tt::tt_metal::CoreCoord core = pick.core;

    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const uint32_t eth_l1 =
        hal.get_dev_addr(tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);

    // One core, so the sweep is cheap; the point here is the launch, not the data.
    const uint32_t num_cores = 1;
    const uint32_t stage_entries_max = 8;
    const SenderL1 l1 = layout_sender_l1(eth_l1, num_cores, stage_entries_max);

    // Point the single swept core at this chip's own first Tensix core, so the
    // sweep issues a real NOC read rather than reading from nowhere.
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& soc = cluster.get_soc_desc(dev->id());
    const auto tensix = soc.get_cores(CoreType::TENSIX, CoordSystem::TRANSLATED);
    ASSERT_FALSE(tensix.empty());
    const uint64_t ring0 = noc_addr_translated(tensix[0].x, tensix[0].y, /*addr=*/0);
    std::vector<uint32_t> ring_tbl = {static_cast<uint32_t>(ring0 & 0xFFFFFFFFull), static_cast<uint32_t>(ring0 >> 32)};
    tt::tt_metal::detail::WriteToDeviceL1(dev, core, l1.ring_table, ring_tbl, CoreType::ETH);

    std::vector<uint32_t> clear(4, 0u);
    tt::tt_metal::detail::WriteToDeviceL1(dev, core, l1.dbg, clear, CoreType::ETH);

    auto program = tt::tt_metal::CreateProgram();
    auto kernel = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/tools/ttnvtop/kernels/eth_aggregator.cpp",
        core,
        tt::tt_metal::EthernetConfig{
            .eth_mode = tt::tt_metal::Eth::IDLE, .processor = tt::tt_metal::DataMovementProcessor::RISCV_0});

    // Find an adjacent peer, and land the journal in ITS idle-eth L1 -- the real
    // M1 destination shape. Passing dest_base = 0 would aim every push at NOC
    // address 0 of some core we do not own, ~1000 times a second.
    auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();
    bool connected = false;
    tt::tt_metal::IDevice* landing_dev = nullptr;
    tt::tt_metal::CoreCoord landing_core;
    std::vector<uint32_t> rt_args;
    for (const auto& d : devices) {
        auto* peer = d->get_devices()[0];
        if (peer->id() == dev->id()) {
            continue;
        }
        auto peer_eth = peer->get_inactive_ethernet_cores();
        if (peer_eth.empty()) {
            continue;
        }
        const tt::tt_metal::CoreCoord pcore = *peer_eth.begin();
        const auto& psoc = cluster.get_soc_desc(peer->id());
        const auto pt = psoc.translate_coord_to(
            tt::umd::CoreCoord(pcore.x, pcore.y, CoreType::ETH, CoordSystem::LOGICAL), CoordSystem::TRANSLATED);
        const uint64_t dest = noc_addr_translated(pt.x, pt.y, eth_l1);

        std::vector<uint32_t> attempt = {
            num_cores,
            l1.ring_table,
            l1.last_head,
            l1.head_scratch,
            l1.stage,
            stage_entries_max,
            l1.hdr_stage,
            static_cast<uint32_t>(dest & 0xFFFFFFFFull),
            static_cast<uint32_t>(dest >> 32),
            UTIL_AGG_CAPACITY,
            0u,
            1000000u,
            1u,
            l1.seq_scratch,
            l1.dbg};
        try {
            tt::tt_fabric::append_fabric_connection_rt_args(
                cp.get_fabric_node_id_from_physical_chip_id(dev->id()),
                cp.get_fabric_node_id_from_physical_chip_id(peer->id()),
                /*link_idx=*/0,
                program,
                core,
                attempt,
                CoreType::ETH);
        } catch (const std::exception&) {
            continue;
        }
        rt_args = std::move(attempt);
        landing_dev = peer;
        landing_core = pcore;
        connected = true;
        log_info(
            tt::LogTest,
            "landing on chip {} eth {} translated ({},{}) dest=0x{:016x}",
            peer->id(),
            pcore.str(),
            pt.x,
            pt.y,
            dest);
        break;
    }
    if (!connected) {
        GTEST_SKIP() << "no adjacent peer with a free idle eth core";
    }

    // Clear the landing header so a stale journal cannot pass for a live one.
    std::vector<uint32_t> zero(sizeof(util_agg_msg_t) / 4, 0u);
    tt::tt_metal::detail::WriteToDeviceL1(landing_dev, landing_core, eth_l1, zero, CoreType::ETH);

    tt::tt_metal::SetRuntimeArgs(program, kernel, core, rt_args);

    // The launch itself. No command queue, no EnqueueProgram.
    tt::tt_metal::detail::LaunchProgram(dev, program, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);

    auto markers = [&]() {
        std::vector<uint32_t> m(4, 0u);
        tt::tt_metal::detail::ReadFromDeviceL1(dev, core, l1.dbg, 16, m, CoreType::ETH);
        return m;
    };

    std::this_thread::sleep_for(std::chrono::seconds(2));
    auto m1 = markers();
    std::this_thread::sleep_for(std::chrono::seconds(2));
    auto m2 = markers();

    log_info(tt::LogTest, "launch: t1 state=0x{:08x} sweeps={} head={} cores={}", m1[0], m1[1], m1[2], m1[3]);
    log_info(tt::LogTest, "launch: t2 state=0x{:08x} sweeps={} head={} cores={}", m2[0], m2[1], m2[2], m2[3]);

    ASSERT_NE(m1[0], 0u) << "kernel never started: the launch message was not picked up";
    EXPECT_EQ(m1[3], num_cores) << "kernel started but read its runtime args wrong";
    EXPECT_EQ(m1[0], 0x09E00000u) << "kernel started but the fabric connection never opened";
    EXPECT_GT(m2[1], m1[1]) << "sweep loop is not advancing";

    // The journal must actually have landed on the peer. Markers alone only
    // prove the sender believes it sent.
    std::vector<uint32_t> h(sizeof(util_agg_msg_t) / 4, 0u);
    tt::tt_metal::detail::ReadFromDeviceL1(landing_dev, landing_core, eth_l1, sizeof(util_agg_msg_t), h, CoreType::ETH);
    log_info(tt::LogTest, "landed: magic=0x{:08x} head={} sweeps={} lost={} cores={}", h[0], h[2], h[5], h[6], h[4]);
    EXPECT_EQ(h[0], UTIL_AGG_MAGIC) << "journal header never landed on the peer chip";
    EXPECT_EQ(h[4], num_cores);
    EXPECT_EQ(util_agg_hdr_checksum(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]), h[8])
        << "landed header checksum mismatch";
    EXPECT_GT(h[5], 0u) << "landed sweep_count is zero";
}

}  // namespace tt::tt_fabric::fabric_router_tests
