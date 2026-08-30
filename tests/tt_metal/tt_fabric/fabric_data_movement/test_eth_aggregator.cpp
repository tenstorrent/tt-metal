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
#include <span>
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
#include "impl/program/program_impl.hpp"
#include <llrt/llrt.hpp>
#include <fstream>

#include "hw/inc/util_aggregator.h"
#include "tools/ttnvtop/host/agg_core_select.hpp"
#include "tools/ttnvtop/host/agg_layout.hpp"

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

// L1 layout, the grid, and the runtime-argument order all live in
// tools/ttnvtop/host/agg_layout.hpp — ONE definition shared with the UMD-only launcher
// that replays the emitted artifact. If the two ever disagree the aggregator reads its
// scratch from the wrong addresses and produces confident nonsense.
using SenderL1 = ttnvtop::AggL1;
using TensixGrid = ttnvtop::AggGrid;

static SenderL1 layout_sender_l1(uint32_t base, uint32_t num_cores, uint32_t /*capacity*/) {
    return ttnvtop::agg_layout(base, num_cores);
}

// Validate that the live Tensix set really is a clean cross product. If it is ever not,
// the kernel's x-list/y-list addressing would silently sweep the wrong cores.
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

static std::vector<uint32_t> build_rt_args(
    const TensixGrid& g,
    const SenderL1& l1,
    uint32_t /*capacity*/,
    uint32_t src_chip,
    uint32_t sweep_cycles,
    uint32_t publish_every) {
    // Wormhole only: BH's heartbeat address differs and its discovery skips the check.
    const bool is_wh = tt::tt_metal::MetalContext::instance().get_cluster().arch() == tt::ARCH::WORMHOLE_B0;
    return ttnvtop::agg_rt_args(
        g, l1, src_chip, sweep_cycles, publish_every, is_wh ? ttnvtop::kWormholeEthHeartbeatAddr : 0u);
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

// Emit the aggregator as a BUILD ARTIFACT a UMD-only process can launch.
//
// The aggregator must run alongside a workload, from a separate PID. Two tt-metal
// processes cannot share a device (CHIP_IN_USE), so the launcher must not link
// tt-metal. 5j proved the last step of the launch — the go word — works cross-process
// over raw UMD. What was missing is the binary and the runtime args.
//
// This snapshots the CONFIGURED L1 IMAGE rather than parsing an ELF. After
// ConfigureDeviceWithProgram has written the kernel into the kernel-config region, the
// bytes sitting there are exactly what any chip of this arch needs. Reading them back
// and replaying them elsewhere sidesteps the JIT build recipe, the linker script, and
// ELF span handling entirely — all of which are tt-metal internals we would otherwise
// have to duplicate and keep in lockstep.
//
// The artifact is produced ONCE, offline, by a process that has tt-metal. Monitoring
// then needs only UMD.
//
// `also_launch` additionally launches the program the artifact was cut from, so one run
// can produce both the artifact and the DEVICE STATE a working launch leaves behind —
// the gold side of the replay diff below.
static void emit_artifact(
    tt::tt_metal::IDevice* dev, const tt::tt_metal::CoreCoord& core, const std::string& dir, bool also_launch) {
    const uint32_t num_cores = num_cores_of(dev);
    ASSERT_GT(num_cores, 0u);
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const uint32_t eth_l1 =
        hal.get_dev_addr(tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);
    const SenderL1 l1 = layout_sender_l1(eth_l1, num_cores, num_cores);

    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& soc = cluster.get_soc_desc(dev->id());
    TensixGrid grid;
    ASSERT_TRUE(translated_tensix_grid(soc.get_cores(CoreType::TENSIX, CoordSystem::TRANSLATED), grid));

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
        build_rt_args(
            grid, l1, num_cores, static_cast<uint32_t>(dev->id()), kSweepIntervalCycles, kPublishEverySweeps));

    tt::tt_metal::detail::CompileProgram(dev, program, /*force_slow_dispatch=*/true);
    program.impl().finalize_dataflow_buffer_configs();
    if (!program.impl().is_finalized()) {
        program.impl().finalize_offsets(dev);
    }
    tt::tt_metal::detail::ConfigureDeviceWithProgram(dev, program, /*force_slow_dispatch=*/true);
    tt::tt_metal::detail::WriteRuntimeArgsToDevice(dev, program, /*force_slow_dispatch=*/true);

    const uint32_t idle_idx = hal.get_programmable_core_type_index(tt::tt_metal::HalProgrammableCoreType::IDLE_ETH);
    auto* kg = program.impl().kernels_on_core(core, idle_idx);
    ASSERT_NE(kg, nullptr);
    auto lm = kg->launch_msg.view();
    const uint32_t cfg_base = lm.kernel_config().kernel_config_base()[idle_idx];
    const uint32_t text_off = lm.kernel_config().kernel_text_offset()[0];
    const uint32_t rta_off = lm.kernel_config().rta_offset()[0].rta_offset();

    // Snapshot the kernel-config region, bounded by where the aggregator's own L1
    // begins.
    //
    // An earlier revision sized this as max(64 KB, ...) — a number invented with no
    // reference to anything. On Wormhole that spans 0x7df0..0x17e70 while the journal
    // sits at 0xe200, so replaying the image wrote 65 KB straight through the journal
    // and every scratch array. Bound it at eth_l1_unreserved: the kernel-config region
    // ends where UNRESERVED begins, by construction.
    ASSERT_GT(eth_l1, cfg_base) << "kernel config must sit below the UNRESERVED region";
    const uint32_t image_bytes = eth_l1 - cfg_base;

    // The runtime args live at cfg_base + rta_offset, and the kernel text at
    // cfg_base + kernel_text_offset. Nothing checks that the args fit in the gap, so
    // one extra argument would silently overwrite the kernel. Here: 31 words = 124 B
    // against a 128 B gap.
    const uint32_t rt_words =
        static_cast<uint32_t>(build_rt_args(grid, l1, num_cores, 0u, kSweepIntervalCycles, kPublishEverySweeps).size());
    if (text_off > rta_off) {
        ASSERT_LE(rt_words * 4u, text_off - rta_off)
            << "runtime args (" << rt_words * 4u << " B) do not fit between rta_offset and "
            << "kernel_text_offset (" << (text_off - rta_off) << " B) — they would overwrite the kernel";
    }

    std::vector<uint32_t> image(image_bytes / 4, 0u);
    tt::tt_metal::detail::ReadFromDeviceL1(dev, core, cfg_base, image_bytes, image, CoreType::ETH);

    {
        std::ofstream f(dir + "/aggregator.image", std::ios::binary);
        ASSERT_TRUE(f.good()) << "cannot write to " << dir;
        f.write(reinterpret_cast<const char*>(image.data()), image_bytes);
    }
    {
        std::ofstream f(dir + "/aggregator.desc");
        f << "arch " << static_cast<int>(cluster.arch()) << "\n";
        f << "kernel_config_base " << cfg_base << "\n";
        f << "image_bytes " << image_bytes << "\n";
        f << "rta_offset " << rta_off << "\n";
        f << "kernel_text_offset " << text_off << "\n";
        f << "launch_addr "
          << hal.get_dev_addr(tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, tt::tt_metal::HalL1MemAddrType::LAUNCH)
          << "\n";
        f << "go_msg_index_addr "
          << hal.get_dev_addr(
                 tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, tt::tt_metal::HalL1MemAddrType::GO_MSG_INDEX)
          << "\n";
        // The RESET go message. LaunchProgram sends this before the launch message
        // (send_reset_go_signal), so the replay reproduces the sequence. Note that on
        // IDLE_ETH it is a NO-OP: idle_erisc.cc's wait loop tests only
        // `go_messages[0].signal != RUN_MSG_GO` and has no RESET_READ_PTR_FROM_HOST
        // branch at all, unlike brisc.cc and active_erisc.cc. Kept because LaunchProgram
        // sends it and the point of the artifact is to be byte-identical, but it is not
        // the missing step it was taken for.
        {
            auto reset_msg = hal.get_dev_msgs_factory(tt::tt_metal::HalProgrammableCoreType::IDLE_ETH)
                                 .create<tt::tt_metal::dev_msgs::go_msg_t>();
            reset_msg.view().signal() = tt::tt_metal::dev_msgs::RUN_MSG_RESET_READ_PTR_FROM_HOST;
            const auto* rp = reinterpret_cast<const uint8_t*>(reset_msg.view().data());
            f << "reset_go_bytes " << reset_msg.view().size();
            for (size_t i = 0; i < reset_msg.view().size(); i++) {
                f << " " << static_cast<unsigned>(rp[i]);
            }
            f << "\n";
        }
        f << "go_addr "
          << hal.get_dev_addr(tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, tt::tt_metal::HalL1MemAddrType::GO_MSG)
          << "\n";
        f << "eth_l1_unreserved " << eth_l1 << "\n";

        // The launch message, captured HERE — before write_launch_msg_to_core gets to
        // mutate it. That function sets kernel_config.mode = DISPATCH_MODE_HOST and
        // LaunchProgram sets host_assigned_id, neither of which the emit path applies,
        // so these bytes are NOT what a working launch puts on the device. Both fields
        // are read by idle_erisc.cc only AFTER the kernel returns, and this kernel never
        // returns — but the replay diff is the place that settles that, not this comment.
        const auto* lp = reinterpret_cast<const uint8_t*>(lm.data());
        f << "launch_bytes " << lm.size();
        for (size_t i = 0; i < lm.size(); i++) {
            f << " " << static_cast<unsigned>(lp[i]);
        }
        f << "\n";
        const auto* gp = reinterpret_cast<const uint8_t*>(kg->go_msg.view().data());
        f << "go_bytes " << kg->go_msg.view().size();
        for (size_t i = 0; i < kg->go_msg.view().size(); i++) {
            f << " " << static_cast<unsigned>(gp[i]);
        }
        f << "\n";
    }
    log_info(
        tt::LogTest,
        "artifact: {} bytes image, cfg_base=0x{:x} text_off=0x{:x} rta_off=0x{:x} -> {}",
        image_bytes,
        cfg_base,
        text_off,
        rta_off,
        dir);

    if (also_launch) {
        // Launch the very program the artifact was cut from. ConfigureDeviceWithProgram
        // and WriteRuntimeArgsToDevice have already run, so LaunchProgram redoes them and
        // then writes the launch message and go word — the steps the replay imitates.
        tt::tt_metal::detail::LaunchProgram(
            dev, program, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);
    }
}

TEST_F(Fabric1DFixture, TestEthAggregatorEmitArtifact) {
    const char* outdir = std::getenv("TTNVTOP_EMIT_ARTIFACT");
    if (outdir == nullptr) {
        GTEST_SKIP() << "set TTNVTOP_EMIT_ARTIFACT=<dir> to emit the launch artifact";
    }
    const auto& devices = this->get_devices();
    ASSERT_GE(devices.size(), 1u);
    auto* dev = devices[0]->get_devices()[0];
    const auto pick = ttnvtop::select_aggregator_eth_core(dev);
    if (!pick.ok) {
        GTEST_SKIP() << pick.reason;
    }
    emit_artifact(dev, pick.core, std::string(outdir), /*also_launch=*/false);
}

// ---------------------------------------------------------------------------
// Artifact replay, byte for byte against a working LaunchProgram.
//
// The defect: `--launch-aggregator` reports success and never starts the kernel, while
// the same kernel via LaunchProgram on the same board starts immediately. Every
// hypothesis so far was reasoned rather than measured and two were wrong, so these
// tests capture DEVICE STATE instead of an argument. One run launches properly and
// dumps the kernel-config region plus the whole mailbox block; a second run replays the
// artifact onto the SAME core and dumps the same two regions. Anything that differs
// shows up as a byte offset.
//
//   run 1:  TTNVTOP_DUMP_DIR=<d> --gtest_filter=*EthAggregatorGoldDump*
//   run 2:  TTNVTOP_DUMP_DIR=<d> --gtest_filter=*EthAggregatorReplayDiff*
//
// Two processes, not two phases of one: stop_aggregator asserts the RISC reset and
// leaves the core with no firmware until the next device init (agg_core_select.hpp), so
// a same-process second launch could not start whatever the bytes said.

// The two regions that decide whether a kernel starts: the kernel-config region the
// firmware jumps into, and the mailbox block it reads to decide where and whether to
// jump. Everything above eth_l1_unreserved is the running kernel's own scratch and
// changes every sweep, so it is deliberately excluded.
struct DumpRegions {
    uint32_t cfg_base = 0;
    uint32_t cfg_bytes = 0;
    uint32_t mbox_base = 0;
    uint32_t mbox_bytes = 0;
};

static DumpRegions dump_regions(uint32_t cfg_base, uint32_t cfg_bytes) {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    DumpRegions r;
    r.cfg_base = cfg_base;
    r.cfg_bytes = cfg_bytes;
    r.mbox_base =
        hal.get_dev_addr(tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, tt::tt_metal::HalL1MemAddrType::MAILBOX);
    // MAILBOX up to the start of the kernel-config region: that span holds the
    // launch-message ring, the go-message array, launch_msg_rd_ptr and the go index —
    // every field idle_erisc.cc consults before jumping.
    r.mbox_bytes = cfg_base > r.mbox_base ? cfg_base - r.mbox_base : 4096u;
    return r;
}

static void write_region(
    tt::tt_metal::IDevice* dev,
    const tt::tt_metal::CoreCoord& core,
    const std::string& path,
    uint32_t addr,
    uint32_t bytes) {
    std::vector<uint32_t> buf(bytes / 4, 0u);
    tt::tt_metal::detail::ReadFromDeviceL1(dev, core, addr, bytes, buf, CoreType::ETH);
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(buf.data()), bytes);
}

static void dump_state(
    tt::tt_metal::IDevice* dev,
    const tt::tt_metal::CoreCoord& core,
    const std::string& dir,
    const std::string& tag,
    const DumpRegions& r) {
    write_region(dev, core, dir + "/" + tag + "_cfg.bin", r.cfg_base, r.cfg_bytes);
    write_region(dev, core, dir + "/" + tag + "_mbox.bin", r.mbox_base, r.mbox_bytes);
    std::ofstream f(dir + "/" + tag + "_regions.txt");
    f << "cfg_base " << r.cfg_base << "\ncfg_bytes " << r.cfg_bytes << "\nmbox_base " << r.mbox_base << "\nmbox_bytes "
      << r.mbox_bytes << "\neth_core " << core.x << " " << core.y << "\n";
    log_info(
        tt::LogTest,
        "{}: cfg 0x{:x}+{} mbox 0x{:x}+{} on eth {}",
        tag,
        r.cfg_base,
        r.cfg_bytes,
        r.mbox_base,
        r.mbox_bytes,
        core.str());
}

// Report every differing byte, grouped into runs so a one-word difference does not
// scroll past a 64 KB dump.
static uint32_t diff_region(const std::string& a_path, const std::string& b_path, uint32_t base, const char* what) {
    std::ifstream fa(a_path, std::ios::binary), fb(b_path, std::ios::binary);
    if (!fa.good() || !fb.good()) {
        log_info(tt::LogTest, "diff {}: MISSING ({} / {})", what, a_path, b_path);
        return 0xFFFFFFFFu;
    }
    const std::vector<char> a((std::istreambuf_iterator<char>(fa)), std::istreambuf_iterator<char>());
    const std::vector<char> b((std::istreambuf_iterator<char>(fb)), std::istreambuf_iterator<char>());
    if (a.size() != b.size()) {
        log_info(tt::LogTest, "diff {}: SIZE MISMATCH {} vs {}", what, a.size(), b.size());
        return 0xFFFFFFFFu;
    }
    uint32_t differing = 0;
    size_t i = 0;
    int runs_printed = 0;
    while (i < a.size()) {
        if (a[i] == b[i]) {
            i++;
            continue;
        }
        const size_t start = i;
        while (i < a.size() && a[i] != b[i]) {
            i++;
        }
        differing += static_cast<uint32_t>(i - start);
        if (runs_printed < 40) {
            uint32_t wa = 0, wb = 0;
            const size_t w = start & ~size_t(3);
            std::memcpy(&wa, a.data() + w, 4);
            std::memcpy(&wb, b.data() + w, 4);
            log_info(
                tt::LogTest,
                "  {} differs at +0x{:x} (L1 0x{:x}) len {}: gold word 0x{:08x} replay 0x{:08x}",
                what,
                start,
                base + start,
                i - start,
                wa,
                wb);
            runs_printed++;
        }
    }
    log_info(tt::LogTest, "diff {}: {} of {} bytes differ", what, differing, a.size());
    return differing;
}

// Run 1: emit the artifact, launch it the working way, confirm it is running, dump.
TEST_F(Fabric1DFixture, TestEthAggregatorGoldDump) {
    const char* dumpdir = std::getenv("TTNVTOP_DUMP_DIR");
    if (dumpdir == nullptr) {
        GTEST_SKIP() << "set TTNVTOP_DUMP_DIR=<dir>";
    }
    const auto& devices = this->get_devices();
    ASSERT_GE(devices.size(), 1u);
    auto* dev = devices[0]->get_devices()[0];
    const auto pick = ttnvtop::select_aggregator_eth_core(dev);
    if (!pick.ok) {
        GTEST_SKIP() << pick.reason;
    }
    const std::string dir(dumpdir);
    emit_artifact(dev, pick.core, dir, /*also_launch=*/true);

    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const uint32_t eth_l1 =
        hal.get_dev_addr(tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);
    const uint32_t num_cores = num_cores_of(dev);
    const SenderL1 l1 = layout_sender_l1(eth_l1, num_cores, num_cores);

    // Confirm the gold launch actually ran before its state is used as a reference.
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::vector<uint32_t> dbg(4, 0u);
    tt::tt_metal::detail::ReadFromDeviceL1(dev, pick.core, l1.dbg, 16, dbg, CoreType::ETH);
    const auto h = read_journal_header(dev, pick.core, l1.journal);
    log_info(
        tt::LogTest, "gold: marker=0x{:08x} sweeps={} head={} | hdr sweeps={}", dbg[0], dbg[1], dbg[2], h.sweep_count);
    EXPECT_GT(dbg[1], 0u) << "the gold launch itself did not start — nothing to compare against";

    // The dbg block and the journal live above eth_l1_unreserved, outside both dumped
    // regions, so a running kernel cannot make the dump differ from a stopped one.
    uint32_t cfg_base = 0, image_bytes = 0;
    {
        std::ifstream f(dir + "/aggregator.desc");
        std::string k;
        while (f >> k) {
            if (k == "kernel_config_base") {
                f >> cfg_base;
            } else if (k == "image_bytes") {
                f >> image_bytes;
            } else {
                std::string skip;
                std::getline(f, skip);
            }
        }
    }
    ASSERT_GT(cfg_base, 0u);
    dump_state(dev, pick.core, dir, "gold", dump_regions(cfg_base, image_bytes));
    ttnvtop::stop_aggregator(dev, pick.core);
}

// Run 2: replay the artifact onto the same core, dump, diff.
//
// The writes are the collector's four, in the collector's order, but issued through
// tt-metal rather than raw UMD. That is deliberate: it holds the CONTENT of the replay
// under test while removing the transport and the firmware-residency question. If the
// kernel starts here but not under the collector, the content is right and the defect is
// environmental; if it does not start here either, the diff says which bytes are wrong.
TEST_F(Fabric1DFixture, TestEthAggregatorReplayDiff) {
    const char* dumpdir = std::getenv("TTNVTOP_DUMP_DIR");
    if (dumpdir == nullptr) {
        GTEST_SKIP() << "set TTNVTOP_DUMP_DIR=<dir> (run TestEthAggregatorGoldDump there first)";
    }
    const std::string dir(dumpdir);
    const auto& devices = this->get_devices();
    ASSERT_GE(devices.size(), 1u);
    auto* dev = devices[0]->get_devices()[0];
    const auto pick = ttnvtop::select_aggregator_eth_core(dev);
    if (!pick.ok) {
        GTEST_SKIP() << pick.reason;
    }

    // Parse the artifact exactly as collector/main.cpp does.
    uint32_t cfg_base = 0, image_bytes = 0, rta_off = 0, launch_addr = 0, go_addr = 0, go_index_addr = 0, eth_l1 = 0;
    std::vector<uint8_t> launch_bytes, go_bytes, reset_go_bytes;
    {
        std::ifstream f(dir + "/aggregator.desc");
        ASSERT_TRUE(f.good()) << "no artifact in " << dir;
        std::string k;
        while (f >> k) {
            if (k == "kernel_config_base") {
                f >> cfg_base;
            } else if (k == "image_bytes") {
                f >> image_bytes;
            } else if (k == "rta_offset") {
                f >> rta_off;
            } else if (k == "launch_addr") {
                f >> launch_addr;
            } else if (k == "go_addr") {
                f >> go_addr;
            } else if (k == "go_msg_index_addr") {
                f >> go_index_addr;
            } else if (k == "eth_l1_unreserved") {
                f >> eth_l1;
            } else if (k == "launch_bytes" || k == "go_bytes" || k == "reset_go_bytes") {
                size_t n = 0;
                f >> n;
                std::vector<uint8_t> v(n);
                for (size_t i = 0; i < n; i++) {
                    unsigned b = 0;
                    f >> b;
                    v[i] = static_cast<uint8_t>(b);
                }
                if (k == "launch_bytes") {
                    launch_bytes = std::move(v);
                } else if (k == "go_bytes") {
                    go_bytes = std::move(v);
                } else {
                    reset_go_bytes = std::move(v);
                }
            } else {
                std::string skip;
                std::getline(f, skip);
            }
        }
    }
    ASSERT_GT(cfg_base, 0u);
    ASSERT_GT(image_bytes, 0u);
    ASSERT_FALSE(launch_bytes.empty());
    ASSERT_FALSE(go_bytes.empty());

    std::vector<uint32_t> image(image_bytes / 4, 0u);
    {
        std::ifstream f(dir + "/aggregator.image", std::ios::binary);
        ASSERT_TRUE(f.good());
        ASSERT_TRUE(f.read(reinterpret_cast<char*>(image.data()), image_bytes));
    }

    const uint32_t num_cores = num_cores_of(dev);
    ASSERT_GT(num_cores, 0u);
    const SenderL1 l1 = layout_sender_l1(eth_l1, num_cores, num_cores);
    TensixGrid grid;
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    ASSERT_TRUE(translated_tensix_grid(
        cluster.get_soc_desc(dev->id()).get_cores(CoreType::TENSIX, CoordSystem::TRANSLATED), grid));
    auto rt_args =
        build_rt_args(grid, l1, num_cores, static_cast<uint32_t>(dev->id()), kSweepIntervalCycles, kPublishEverySweeps);

    // Zero the markers so "is it running" cannot be answered by a stale value.
    std::vector<uint32_t> zero(4, 0u);
    tt::tt_metal::detail::WriteToDeviceL1(dev, pick.core, l1.dbg, zero, CoreType::ETH);

    // The collector's sequence, in the collector's order. The messages go out through
    // the byte-span overload so they land at exactly the size and offset the emitter
    // recorded, with no word rounding of our own.
    if (!reset_go_bytes.empty()) {
        tt::tt_metal::detail::WriteToDeviceL1(
            dev, pick.core, go_addr, std::span<const uint8_t>(reset_go_bytes), CoreType::ETH);
    }
    if (go_index_addr != 0) {
        std::vector<uint32_t> one_zero(1, 0u);
        tt::tt_metal::detail::WriteToDeviceL1(dev, pick.core, go_index_addr, one_zero, CoreType::ETH);
    }
    tt::tt_metal::detail::WriteToDeviceL1(dev, pick.core, cfg_base, image, CoreType::ETH);
    tt::tt_metal::detail::WriteToDeviceL1(dev, pick.core, cfg_base + rta_off, rt_args, CoreType::ETH);
    tt::tt_metal::detail::WriteToDeviceL1(
        dev, pick.core, launch_addr, std::span<const uint8_t>(launch_bytes), CoreType::ETH);
    tt::tt_metal::detail::WriteToDeviceL1(dev, pick.core, go_addr, std::span<const uint8_t>(go_bytes), CoreType::ETH);

    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::vector<uint32_t> dbg(4, 0u);
    tt::tt_metal::detail::ReadFromDeviceL1(dev, pick.core, l1.dbg, 16, dbg, CoreType::ETH);
    log_info(tt::LogTest, "replay: marker=0x{:08x} sweeps={} head={}", dbg[0], dbg[1], dbg[2]);

    const auto r = dump_regions(cfg_base, image_bytes);
    dump_state(dev, pick.core, dir, "replay", r);

    const uint32_t cfg_diff = diff_region(dir + "/gold_cfg.bin", dir + "/replay_cfg.bin", r.cfg_base, "kernel-config");
    const uint32_t mbox_diff = diff_region(dir + "/gold_mbox.bin", dir + "/replay_mbox.bin", r.mbox_base, "mailbox");
    log_info(
        tt::LogTest,
        "REPLAY VERDICT: running={} cfg_diff={} mbox_diff={}",
        dbg[1] > 0 ? "yes" : "NO",
        cfg_diff,
        mbox_diff);

    if (dbg[1] > 0) {
        ttnvtop::stop_aggregator(dev, pick.core);
    }
    EXPECT_GT(dbg[1], 0u) << "artifact replay did not start the kernel";
}

// Hold the devices open and do nothing else.
//
// One variable, isolated: whether a tt-metal-initialised device is present while the
// UMD-only launcher replays. Device init is what loads tt-metal's idle-erisc firmware
// onto inactive ethernet cores (risc_firmware_initializer.cpp writes their launch
// message and a RUN_MSG_INIT go word), and that firmware's wait loop is the ONLY thing
// that ever polls go_messages[0].signal for RUN_MSG_GO. Nothing else on the core reads
// it. So the replay's go word can only start a kernel if that firmware is resident.
//
// Run this in one process and `ttnvtop-collector --launch-aggregator` in another to
// settle it, then run the collector alone against the same board for the contrast.
TEST_F(Fabric1DFixture, TestEthAggregatorHoldDevice) {
    const char* secs_env = std::getenv("TTNVTOP_HOLD_SECONDS");
    if (secs_env == nullptr) {
        GTEST_SKIP() << "set TTNVTOP_HOLD_SECONDS=<n>";
    }
    const auto& devices = this->get_devices();
    ASSERT_GE(devices.size(), 1u);
    log_info(tt::LogTest, "holding {} device(s) open for {} s, doing nothing", devices.size(), secs_env);
    std::this_thread::sleep_for(std::chrono::seconds(std::atoi(secs_env)));
}

// Soak: does the aggregator survive sustained running?
//
// On the T3K the heartbeat froze at 0xabcd5904 — our signature, counter 22788 sweeps
// ~= 23 s — meaning the kernel maintained it and then STOPPED. That is the single
// blocker for launching before a workload: if the kernel is alive when the workload
// opens the device, discovery passes, and the reset that follows is survivable because
// a supervisor can relaunch. If it dies first, the workload cannot start at all.
//
// This samples the liveness markers over minutes on whatever board is present, so the
// failure can be characterised where a wedge costs nothing.
TEST_F(Fabric1DFixture, TestEthAggregatorSoak) {
    const char* secs_env = std::getenv("TTNVTOP_SOAK_SECONDS");
    if (secs_env == nullptr) {
        GTEST_SKIP() << "set TTNVTOP_SOAK_SECONDS=<n> to soak the aggregator";
    }
    const int secs = std::atoi(secs_env);
    const auto& devices = this->get_devices();
    ASSERT_GE(devices.size(), 1u);
    auto* dev = devices[0]->get_devices()[0];
    const auto pick = ttnvtop::select_aggregator_eth_core(dev);
    if (!pick.ok) {
        GTEST_SKIP() << pick.reason;
    }

    TensixGrid grid;
    const uint32_t n = num_cores_of(dev);
    const SenderL1 l1 = launch_aggregator(dev, pick.core, n, kSweepIntervalCycles, kPublishEverySweeps, grid);
    log_info(tt::LogTest, "soak: {} cores on chip {} eth {}, {} s", n, dev->id(), pick.core.str(), secs);

    uint32_t last_sweeps = 0, stalled_at = 0;
    for (int t = 0; t < secs; t += 5) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        std::vector<uint32_t> dbg(4, 0u);
        tt::tt_metal::detail::ReadFromDeviceL1(dev, pick.core, l1.dbg, 16, dbg, CoreType::ETH);
        const auto h = read_journal_header(dev, pick.core, l1.journal);
        log_info(
            tt::LogTest,
            "soak t+{}s: marker=0x{:08x} sweeps={} head={} | hdr sweeps={} head={} lost={}",
            t + 5,
            dbg[0],
            dbg[1],
            dbg[2],
            h.sweep_count,
            h.head,
            h.lost);
        if (dbg[1] == last_sweeps && stalled_at == 0 && t > 0) {
            stalled_at = t + 5;
            log_info(tt::LogTest, "soak: STALLED at t+{}s (sweeps stuck at {})", stalled_at, dbg[1]);
        }
        last_sweeps = dbg[1];
    }
    ttnvtop::stop_aggregator(dev, pick.core);
    EXPECT_EQ(stalled_at, 0u) << "aggregator stopped advancing at t+" << stalled_at << "s";
}

}  // namespace tt::tt_fabric::fabric_router_tests
