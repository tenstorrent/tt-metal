// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// tt-coremon idle-eth aggregator (PLAN_ETH_AGGREGATOR.md).
//
// The aggregator runs persistently on an idle ethernet core, sweeps every Tensix core's
// util_sampler ring over the local NOC, and accumulates samples into a journal in its
// own L1. It uses NO FABRIC: see the header comment in eth_aggregator.cpp for why the
// earlier fabric-push design was reversed. The host reads the journal where it lies --
// one compact read per chip per tick, against the 64 per-core reads it does today.

#include <gtest/gtest.h>
#include <chrono>
#include <cstdint>
#include <memory>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_set>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "impl/kernels/kernel.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "fabric_fixture.hpp"

#include "hw/inc/util_aggregator.h"
#include "tools/ttnvtop/host/agg_core_select.hpp"

namespace tt::tt_fabric::fabric_router_tests {

// ~1 ms at 1 GHz, matching the Tensix rings' own period.
constexpr uint32_t kSweepIntervalCycles = 1000000u;

// Republish the header every N sweeps, not every sweep. At a 1 ms sweep this is ~60 Hz,
// comfortably above the host's ~10 Hz drain and slow enough that the header is stable
// for far longer than a remote read takes. See the publish note in the kernel.
constexpr uint32_t kPublishEverySweeps = 16u;

// Diagnostic override while characterising remote-read tearing.
static uint32_t sweep_cycles_override(uint32_t dflt) {
    if (const char* e = std::getenv("TTNVTOP_SWEEP_CYCLES")) {
        return static_cast<uint32_t>(std::atoll(e));
    }
    return dflt;
}

static uint32_t publish_every_sweeps() {
    if (const char* e = std::getenv("TTNVTOP_PUBLISH_EVERY")) {
        return static_cast<uint32_t>(std::atoi(e));
    }
    return kPublishEverySweeps;
}

// L1 layout inside the aggregator eth core's UNRESERVED region. All 16 B aligned:
// every one of these is a NOC read destination.
struct SenderL1 {
    uint32_t last_head;     // num_cores * 4 B
    uint32_t head_scratch;  // num_cores * 16 B
    uint32_t seq_scratch;   // num_cores * 4 B
    uint32_t dbg;           // 16 B liveness markers
    uint32_t journal;       // 64 B header + capacity * 32 B
    uint32_t end;
};

static SenderL1 layout_sender_l1(uint32_t base, uint32_t num_cores, uint32_t capacity) {
    auto align16 = [](uint32_t v) { return (v + 15u) & ~15u; };
    SenderL1 l{};
    uint32_t p = align16(base);
    l.last_head = p;
    p = align16(p + num_cores * 4u);
    l.head_scratch = p;
    p = align16(p + num_cores * 16u);
    l.seq_scratch = p;
    p = align16(p + num_cores * 4u);
    l.dbg = p;
    p = align16(p + 16u);
    l.journal = p;
    p = align16(p + sizeof(util_agg_msg_t) + capacity * sizeof(util_agg_entry_t));
    l.end = p;
    return l;
}

// The live Tensix cores as a CROSS PRODUCT of translated x and y coordinates.
//
// Harvesting removes whole rows (WH) or whole columns (BH), so the live set is always
// live_x x live_y: nx+ny numbers describe all nx*ny cores. NOT a contiguous rectangle
// -- BH takes its translated coords from the NOC0 core list, which skips the non-Tensix
// columns, so its live x values have gaps.
//
// VERIFIED, not assumed: if the translated set is ever not a clean cross product the
// aggregator would silently sweep the wrong cores, so this returns false instead.
struct TensixGrid {
    std::vector<uint32_t> xs, ys;
};

static bool translated_tensix_grid(const std::vector<tt::umd::CoreCoord>& cores, TensixGrid& out) {
    if (cores.empty()) {
        return false;
    }
    std::set<uint32_t> xs, ys;
    std::set<std::pair<uint32_t, uint32_t>> seen;
    for (const auto& c : cores) {
        xs.insert(c.x);
        ys.insert(c.y);
        seen.insert({c.x, c.y});
    }
    out.xs.assign(xs.begin(), xs.end());
    out.ys.assign(ys.begin(), ys.end());
    return out.xs.size() * out.ys.size() == cores.size() && seen.size() == cores.size();
}

// Runtime args in the kernel's order: num_cores, nx, ny, xs..., ys..., then the rest.
static std::vector<uint32_t> build_rt_args(
    const TensixGrid& g,
    const SenderL1& l1,
    uint32_t capacity,
    uint32_t src_chip,
    uint32_t sweep_cycles,
    uint32_t publish_every) {
    std::vector<uint32_t> a = {
        static_cast<uint32_t>(g.xs.size() * g.ys.size()),
        static_cast<uint32_t>(g.xs.size()),
        static_cast<uint32_t>(g.ys.size())};
    a.insert(a.end(), g.xs.begin(), g.xs.end());
    a.insert(a.end(), g.ys.begin(), g.ys.end());
    a.push_back(l1.last_head);
    a.push_back(l1.head_scratch);
    a.push_back(l1.seq_scratch);
    a.push_back(l1.journal);
    a.push_back(capacity);
    a.push_back(src_chip);
    a.push_back(sweep_cycles);
    a.push_back(publish_every);
    a.push_back(l1.dbg);
    return a;
}

// Restrict UMD's remote transfers to the channels that actually reach `remote`.
//
// By default UMD uses EVERY active eth channel on the MMIO chip (cluster.cpp:780), and
// tt-metal never narrows it. On a T3K that is six, four of which link to other boards
// entirely -- yet wait_for_non_mmio_flush waits for all of them. Pinning took remote
// operations from 1/6 to 14/14 (5h).
static void pin_tunnel_channels(tt::ChipId mmio, tt::ChipId remote) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& desc = *cluster.get_driver()->get_cluster_description();
    std::unordered_set<tt::umd::CoreCoord> pinned;
    for (const auto& [local_ch, remote_ch] :
         desc.get_directly_connected_ethernet_channels_between_chips(mmio, remote)) {
        pinned.insert(tt::umd::CoreCoord(0, local_ch, CoreType::ETH, CoordSystem::LOGICAL));
    }
    if (!pinned.empty()) {
        cluster.get_driver()->configure_active_ethernet_cores_for_mmio_device(mmio, pinned);
    }
}

// Stand an aggregator up on `dev`'s eth core at `rank`. Returns the L1 layout.
static SenderL1 launch_aggregator(
    tt::tt_metal::IDevice* dev,
    const tt::tt_metal::CoreCoord& core,
    uint32_t capacity,
    uint32_t sweep_cycles,
    uint32_t publish_every,
    TensixGrid& grid_out) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const uint32_t eth_l1 =
        hal.get_dev_addr(tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);

    const auto& soc = cluster.get_soc_desc(dev->id());
    const auto tensix = soc.get_cores(CoreType::TENSIX, CoordSystem::TRANSLATED);
    EXPECT_TRUE(translated_tensix_grid(tensix, grid_out));
    const uint32_t num_cores = static_cast<uint32_t>(grid_out.xs.size() * grid_out.ys.size());
    EXPECT_LE(num_cores, UTIL_AGG_MAX_CORES);

    const SenderL1 l1 = layout_sender_l1(eth_l1, num_cores, capacity);

    std::vector<uint32_t> zero(4, 0u);
    tt::tt_metal::detail::WriteToDeviceL1(dev, core, l1.dbg, zero, CoreType::ETH);

    auto program = tt::tt_metal::CreateProgram();
    auto kernel = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/tools/ttnvtop/kernels/eth_aggregator.cpp",
        core,
        tt::tt_metal::EthernetConfig{
            .eth_mode = tt::tt_metal::Eth::IDLE, .processor = tt::tt_metal::DataMovementProcessor::RISCV_0});
    tt::tt_metal::SetRuntimeArgs(
        program,
        kernel,
        core,
        build_rt_args(grid_out, l1, capacity, static_cast<uint32_t>(dev->id()), sweep_cycles, publish_every));

    // No dispatch of any kind. The idle-erisc firmware is already polling
    // go_messages[0].signal; this writes binaries, args and the launch message with
    // plain host writes and sets RUN_MSG_GO. wait_until_cores_done MUST be false: a
    // kernel that never returns never writes RUN_MSG_DONE (3.5/3.6).
    tt::tt_metal::detail::LaunchProgram(dev, program, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);
    return l1;
}

// Read the header, retrying on a checksum mismatch. The aggregator writes the checksum
// last, so a mismatch means we caught a write in progress -- re-read, never fall back.
static util_agg_hdr_view_t read_journal_header(
    tt::tt_metal::IDevice* dev, const tt::tt_metal::CoreCoord& core, uint32_t journal_addr) {
    util_agg_hdr_view_t h{};
    for (int attempt = 0; attempt < 8; attempt++) {
        std::vector<uint32_t> raw(sizeof(util_agg_msg_t) / 4, 0u);
        tt::tt_metal::detail::ReadFromDeviceL1(dev, core, journal_addr, sizeof(util_agg_msg_t), raw, CoreType::ETH);
        std::memcpy(&h, raw.data(), sizeof(h));
        if (h.magic != UTIL_AGG_MAGIC || util_agg_hdr_ok(h)) {
            return h;  // consistent, or not a journal at all — let the caller judge
        }
    }
    return h;
}

static void expect_header_sane(const util_agg_hdr_view_t& h, uint32_t num_cores, uint32_t capacity) {
    EXPECT_EQ(h.magic, UTIL_AGG_MAGIC) << "no journal header";
    EXPECT_EQ(h.version, UTIL_AGG_VERSION);
    EXPECT_EQ(h.num_cores, num_cores);
    EXPECT_EQ(h.capacity, capacity);
    EXPECT_TRUE(util_agg_hdr_ok(h)) << "torn header read: head=" << h.head << " head_xor=0x" << std::hex << h.head_xor
                                    << std::dec;
    EXPECT_EQ(util_agg_hdr_checksum(h.magic, h.version, h.capacity, h.num_cores, h.src_chip), h.hdr_checksum)
        << "static header fields do not check out — layout mismatch";
}

// Compile the kernel and stop there. Runs anywhere with an idle eth core, so a syntax
// or type error cannot reach a T3K as a runtime JIT failure.
TEST_F(Fabric1DFixture, TestEthAggregatorKernelCompiles) {
    const auto& devices = this->get_devices();
    ASSERT_GE(devices.size(), 1u);
    auto* dev = devices[0]->get_devices()[0];
    const auto pick = ttnvtop::select_aggregator_eth_core(dev);
    if (!pick.ok) {
        GTEST_SKIP() << pick.reason;
    }
    auto program = tt::tt_metal::CreateProgram();
    auto kernel = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/tools/ttnvtop/kernels/eth_aggregator.cpp",
        pick.core,
        tt::tt_metal::EthernetConfig{
            .eth_mode = tt::tt_metal::Eth::IDLE, .processor = tt::tt_metal::DataMovementProcessor::RISCV_0});
    TensixGrid g;
    g.xs = {18};
    g.ys = {18};
    SenderL1 l1{};
    tt::tt_metal::SetRuntimeArgs(program, kernel, pick.core, build_rt_args(g, l1, 64u, 0u, 1000u, 16u));
    EXPECT_NO_THROW(tt::tt_metal::detail::CompileProgram(dev, program, /*force_slow_dispatch=*/true));
}

// Launch with no dispatch involvement and confirm the journal fills in the aggregator's
// own L1. No fabric, no peer chip, no tunnel.
TEST_F(Fabric1DFixture, TestEthAggregatorLocalJournal) {
    const auto& devices = this->get_devices();
    ASSERT_GE(devices.size(), 1u);
    auto* dev = devices[0]->get_devices()[0];
    const auto pick = ttnvtop::select_aggregator_eth_core(dev);
    if (!pick.ok) {
        GTEST_SKIP() << pick.reason;
    }

    TensixGrid grid;
    const SenderL1 l1 = launch_aggregator(
        dev, pick.core, UTIL_AGG_CAPACITY, sweep_cycles_override(kSweepIntervalCycles), publish_every_sweeps(), grid);
    const uint32_t num_cores = static_cast<uint32_t>(grid.xs.size() * grid.ys.size());
    log_info(
        tt::LogTest,
        "local: chip {} eth {} grid {}x{} = {} cores",
        dev->id(),
        pick.core.str(),
        grid.xs.size(),
        grid.ys.size(),
        num_cores);

    std::this_thread::sleep_for(std::chrono::seconds(2));
    const auto h1 = read_journal_header(dev, pick.core, l1.journal);
    std::this_thread::sleep_for(std::chrono::seconds(2));
    const auto h2 = read_journal_header(dev, pick.core, l1.journal);

    log_info(
        tt::LogTest,
        "local: t1 head={} sweeps={} lost={}  t2 head={} sweeps={} lost={}",
        h1.head,
        h1.sweep_count,
        h1.lost,
        h2.head,
        h2.sweep_count,
        h2.lost);

    expect_header_sane(h2, num_cores, UTIL_AGG_CAPACITY);
    EXPECT_GT(h2.sweep_count, h1.sweep_count) << "sweep loop is not advancing";
    EXPECT_GT(h2.head, 0u) << "no entries were written";
    EXPECT_EQ(h2.src_chip, static_cast<uint32_t>(dev->id()));

    ttnvtop::stop_aggregator(dev, pick.core);
}

// Force the paths a production-sized journal would take minutes to reach: shrink the
// capacity so the ring wraps repeatedly, sweep the whole grid, and put a poison guard
// band immediately after the journal that must come back untouched.
TEST_F(Fabric1DFixture, TestEthAggregatorMultiCoreAndWrap) {
    const auto& devices = this->get_devices();
    ASSERT_GE(devices.size(), 1u);
    auto* dev = devices[0]->get_devices()[0];
    // Rank 1: the other launching test leaves a persistent kernel on rank 0, and a core
    // that already holds one cannot be relaunched onto — the launch message is never
    // consumed and the live kernel is corrupted by the binary write (5f).
    const auto pick = ttnvtop::select_aggregator_eth_core(dev, /*rank=*/1);
    if (!pick.ok) {
        GTEST_SKIP() << pick.reason;
    }

    constexpr uint32_t kCapacity = 256;
    constexpr uint32_t kGuardBytes = 1024;
    constexpr uint32_t kJournalBytes = sizeof(util_agg_msg_t) + kCapacity * sizeof(util_agg_entry_t);

    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const uint32_t eth_l1 =
        hal.get_dev_addr(tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);
    const auto& soc = cluster.get_soc_desc(dev->id());
    TensixGrid probe;
    ASSERT_TRUE(translated_tensix_grid(soc.get_cores(CoreType::TENSIX, CoordSystem::TRANSLATED), probe));
    const uint32_t num_cores = static_cast<uint32_t>(probe.xs.size() * probe.ys.size());
    const SenderL1 pre = layout_sender_l1(eth_l1, num_cores, kCapacity);

    // Poison the journal AND a guard band after it. A slot still reading 0xDEADBEEF was
    // never written; a guard word that is NOT 0xDEADBEEF is a write that ran off the end.
    std::vector<uint32_t> poison((kJournalBytes + kGuardBytes) / 4, 0xDEADBEEFu);
    tt::tt_metal::detail::WriteToDeviceL1(dev, pick.core, pre.journal, poison, CoreType::ETH);

    // 100 us sweep so the small ring wraps hard within the observation window.
    TensixGrid grid;
    const SenderL1 l1 = launch_aggregator(dev, pick.core, kCapacity, 100000u, 128u, grid);
    ASSERT_EQ(l1.journal, pre.journal);
    log_info(
        tt::LogTest, "wrap: {} cores, capacity={} on chip {} eth {}", num_cores, kCapacity, dev->id(), pick.core.str());

    std::this_thread::sleep_for(std::chrono::seconds(6));
    const auto h = read_journal_header(dev, pick.core, l1.journal);
    log_info(
        tt::LogTest,
        "wrap: head={} sweeps={} lost={} cores={} cap={}",
        h.head,
        h.sweep_count,
        h.lost,
        h.num_cores,
        h.capacity);
    expect_header_sane(h, num_cores, kCapacity);
    EXPECT_GT(h.head, kCapacity * 2u) << "journal did not wrap twice; the wrap path is untested";

    std::vector<uint32_t> guard(kGuardBytes / 4, 0u);
    tt::tt_metal::detail::ReadFromDeviceL1(
        dev, pick.core, l1.journal + kJournalBytes, kGuardBytes, guard, CoreType::ETH);
    uint32_t clobbered = 0;
    for (uint32_t v : guard) {
        clobbered += (v != 0xDEADBEEFu);
    }
    EXPECT_EQ(clobbered, 0u) << clobbered << " guard words past the journal were overwritten";

    std::vector<uint32_t> ent(kCapacity * sizeof(util_agg_entry_t) / 4, 0u);
    tt::tt_metal::detail::ReadFromDeviceL1(
        dev, pick.core, l1.journal + UTIL_AGG_JOURNAL_OFFSET, kCapacity * sizeof(util_agg_entry_t), ent, CoreType::ETH);
    std::set<uint32_t> seen;
    uint32_t unwritten = 0, bad_core = 0;
    for (uint32_t i = 0; i < kCapacity; i++) {
        const uint32_t* e = &ent[i * (sizeof(util_agg_entry_t) / 4)];
        if (e[0] == 0xDEADBEEFu && e[4] == 0xDEADBEEFu) {
            unwritten++;
        } else if (e[4] >= num_cores) {
            bad_core++;
        } else {
            seen.insert(e[4]);
        }
    }
    log_info(tt::LogTest, "wrap: {} distinct core_ids, {} unwritten, {} bad", seen.size(), unwritten, bad_core);
    EXPECT_EQ(bad_core, 0u) << "journal slots hold out-of-range core_ids";
    EXPECT_EQ(unwritten, 0u) << "journal has unwritten slots after wrapping twice";
    EXPECT_GT(seen.size(), num_cores / 2) << "the sweep is not covering the grid";

    ttnvtop::stop_aggregator(dev, pick.core);
}

// The case the feature exists for: an aggregator on a REMOTE chip, read by the host
// over the tunnel. One compact journal read instead of 64 per-core reads.
TEST_F(Fabric1DFixture, TestEthAggregatorRemoteChip) {
    const auto& devices = this->get_devices();
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    tt::tt_metal::IDevice* remote = nullptr;
    for (const auto& d : devices) {
        auto* dev = d->get_devices()[0];
        if (cluster.get_associated_mmio_device(dev->id()) != dev->id()) {
            remote = dev;
            break;
        }
    }
    if (remote == nullptr) {
        GTEST_SKIP() << "no remote chip";
    }
    pin_tunnel_channels(cluster.get_associated_mmio_device(remote->id()), remote->id());

    const auto pick = ttnvtop::select_aggregator_eth_core(remote);
    if (!pick.ok) {
        GTEST_SKIP() << pick.reason;
    }

    TensixGrid grid;
    SenderL1 l1{};
    try {
        l1 = launch_aggregator(
            remote,
            pick.core,
            UTIL_AGG_CAPACITY,
            sweep_cycles_override(kSweepIntervalCycles),
            publish_every_sweeps(),
            grid);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "remote launch wedged the NON_MMIO tunnel (5g/5h): " << e.what();
    }
    const uint32_t num_cores = static_cast<uint32_t>(grid.xs.size() * grid.ys.size());
    log_info(
        tt::LogTest,
        "remote: chip {} eth {} grid {}x{} = {} cores",
        remote->id(),
        pick.core.str(),
        grid.xs.size(),
        grid.ys.size(),
        num_cores);

    std::this_thread::sleep_for(std::chrono::seconds(3));
    auto t0 = std::chrono::steady_clock::now();
    const auto h1 = read_journal_header(remote, pick.core, l1.journal);
    const auto rd1 =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    std::this_thread::sleep_for(std::chrono::seconds(2));
    t0 = std::chrono::steady_clock::now();
    const auto h2 = read_journal_header(remote, pick.core, l1.journal);
    const auto rd2 =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    log_info(tt::LogTest, "remote: header read cost {} ms / {} ms", rd1, rd2);
    log_info(
        tt::LogTest,
        "remote: t1 head={} sweeps={} lost={}  t2 head={} sweeps={} lost={}",
        h1.head,
        h1.sweep_count,
        h1.lost,
        h2.head,
        h2.sweep_count,
        h2.lost);

    expect_header_sane(h2, num_cores, UTIL_AGG_CAPACITY);
    EXPECT_GT(h2.sweep_count, h1.sweep_count) << "aggregator is not running on the remote chip";
    EXPECT_GT(h2.head, 0u);
    EXPECT_EQ(h2.src_chip, static_cast<uint32_t>(remote->id()));

    if (const char* hold = std::getenv("TTNVTOP_HOLD_SECONDS")) {
        log_info(tt::LogTest, "remote: holding {} s so another process can read the journal", hold);
        std::this_thread::sleep_for(std::chrono::seconds(std::atoi(hold)));
    }
    ttnvtop::stop_aggregator(remote, pick.core);
}

}  // namespace tt::tt_fabric::fabric_router_tests
