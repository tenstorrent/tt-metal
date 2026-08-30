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
    uint32_t last_wall;     // num_cores * 4 B
    uint32_t last_fpu;      // num_cores * 4 B
    uint32_t sample_pad;    // 16 B landing pad for one raw sample
    uint32_t dbg;           // 16 B liveness markers
    uint32_t journal;       // 64 B header + num_cores * 32 B state table
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
    l.last_wall = p;
    p = align16(p + num_cores * 4u);
    l.last_fpu = p;
    p = align16(p + num_cores * 4u);
    l.sample_pad = p;
    p = align16(p + 16u);
    l.dbg = p;
    p = align16(p + 16u);
    l.journal = p;
    p = align16(p + util_agg_bytes_for(capacity));
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

// v2: capacity == num_cores. Resolve it before laying out L1.
static uint32_t num_cores_of(tt::tt_metal::IDevice* dev) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& soc = cluster.get_soc_desc(dev->id());
    TensixGrid g;
    if (!translated_tensix_grid(soc.get_cores(CoreType::TENSIX, CoordSystem::TRANSLATED), g)) {
        return 0;
    }
    return static_cast<uint32_t>(g.xs.size() * g.ys.size());
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
    a.push_back(l1.last_wall);
    a.push_back(l1.last_fpu);
    a.push_back(l1.journal);
    a.push_back(capacity);
    a.push_back(src_chip);
    a.push_back(sweep_cycles);
    a.push_back(publish_every);
    a.push_back(l1.sample_pad);
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
        dev, pick.core, num_cores_of(dev), sweep_cycles_override(kSweepIntervalCycles), publish_every_sweeps(), grid);
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

    expect_header_sane(h2, num_cores, num_cores);
    EXPECT_GT(h2.sweep_count, h1.sweep_count) << "sweep loop is not advancing";
    EXPECT_GT(h2.head, 0u) << "no entries were written";
    EXPECT_EQ(h2.src_chip, static_cast<uint32_t>(dev->id()));

    ttnvtop::stop_aggregator(dev, pick.core);
}

// Every core in the grid must accumulate, and the accumulated numbers must be
// physically plausible. v1 tested a ring wrapping; v2 has no ring — the state table is
// fixed-size and overwritten in place, which is the point.
TEST_F(Fabric1DFixture, TestEthAggregatorStateTable) {
    const auto& devices = this->get_devices();
    ASSERT_GE(devices.size(), 1u);
    auto* dev = devices[0]->get_devices()[0];
    // Rank 1: the other launching test leaves a persistent kernel on rank 0, and a core
    // that already holds one cannot be relaunched onto (5f).
    const auto pick = ttnvtop::select_aggregator_eth_core(dev, /*rank=*/1);
    if (!pick.ok) {
        GTEST_SKIP() << pick.reason;
    }
    const uint32_t num_cores = num_cores_of(dev);
    ASSERT_GT(num_cores, 1u);

    TensixGrid grid;
    // 100 us sweep: with a fixed-size table this costs the readout nothing, which is
    // exactly what v2 buys.
    const SenderL1 l1 = launch_aggregator(dev, pick.core, num_cores, 100000u, 8u, grid);
    log_info(tt::LogTest, "state: {} cores on chip {} eth {}", num_cores, dev->id(), pick.core.str());

    auto read_states = [&]() {
        const uint32_t bytes = num_cores * sizeof(util_agg_core_state_t);
        std::vector<uint32_t> raw(bytes / 4, 0u);
        tt::tt_metal::detail::ReadFromDeviceL1(
            dev, pick.core, l1.journal + UTIL_AGG_STATES_OFFSET, bytes, raw, CoreType::ETH);
        std::vector<util_agg_core_state_t> out(num_cores);
        std::memcpy(out.data(), raw.data(), bytes);
        return out;
    };

    std::this_thread::sleep_for(std::chrono::seconds(2));
    const auto s1 = read_states();
    const auto h1 = read_journal_header(dev, pick.core, l1.journal);
    std::this_thread::sleep_for(std::chrono::seconds(3));
    const auto s2 = read_states();
    const auto h2 = read_journal_header(dev, pick.core, l1.journal);

    expect_header_sane(h2, num_cores, num_cores);
    EXPECT_GT(h2.sweep_count, h1.sweep_count) << "sweep loop is not advancing";

    // Per-core: seq must advance, and busy must never exceed wall over the interval —
    // that would mean more busy cycles than elapsed time, which is not physical.
    uint32_t advancing = 0, impossible = 0, with_samples = 0;
    double max_util = 0.0;
    for (uint32_t i = 0; i < num_cores; i++) {
        const uint32_t dseq = s2[i].seq - s1[i].seq;
        const uint32_t dwall = s2[i].wall_cycles - s1[i].wall_cycles;  // wrap-correct
        const uint32_t dbusy = s2[i].busy_cycles - s1[i].busy_cycles;
        if (dseq > 0) {
            advancing++;
        }
        if (s2[i].samples > s1[i].samples) {
            with_samples++;
        }
        if (dwall > 0) {
            const double util = static_cast<double>(dbusy) / static_cast<double>(dwall);
            max_util = std::max(max_util, util);
            if (util > 1.05) {  // 5% slack for counter/clock skew
                impossible++;
            }
        }
    }
    log_info(
        tt::LogTest,
        "state: {}/{} cores advancing, {} with new samples, max util {:.3f}, head={} lost={}",
        advancing,
        num_cores,
        with_samples,
        max_util,
        h2.head,
        h2.lost);

    EXPECT_EQ(advancing, num_cores) << "not every core's state is being updated";
    EXPECT_EQ(impossible, 0u) << impossible
                              << " cores reported busy_cycles > wall_cycles — "
                                 "the on-chip delta arithmetic is wrong";
    EXPECT_GT(h2.head, 0u) << "no samples were folded in";

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
            num_cores_of(remote),
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

    expect_header_sane(h2, num_cores, num_cores);
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
