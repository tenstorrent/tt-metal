// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host side of the ethernet wall-clock sync, packaged as ONE self-contained call.
//
//     measure_link(sender_device, sender_eth_core, receiver_device, receiver_eth_core, cfg)
//
// It builds both programs, launches them, polls their status word, reads the samples straight out of eth
// L1, solves, and leaves nothing resident. It takes no init-time state and keeps none of its own, so it is
// safe to call more than once and from anywhere -- the profiler calls it at start(), and the standalone
// test calls the same function, so there is only one code path to get right.
//
// TWO THINGS TO KNOW BEFORE CALLING IT MID-RUN (today it is only called at init):
//
//   1. The eth cores belong to FABRIC once initialize_fabric_and_dispatch_fw() has run. Launching this on
//      a core a fabric router owns would evict the router. At init the cores are still free, which is why
//      that is the safe window.
//   2. detail::LaunchProgram carries a dram_barrier that MMIO-polls a core in every DRAM channel, and a
//      barrier issued while a drainer is resident with its DRAM core in stream mode is exactly the read
//      that never completes (the bring-up hang this session diagnosed). At init no drainer is resident
//      yet; mid-run they are.
//
// Both are solved the same way when we want re-syncs during a run: make the pair RESIDENT at init and
// trigger each round through an L1 mailbox, the way the drainers already work -- one launch, many
// measurements, no further LaunchProgram and no core stolen from fabric later.

#pragma once

#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"  // EthernetConfig
#include "tools/profiler/sync/eth_wallclock_sync_solve.hpp"
#include "tools/profiler/sync/eth_wallclock_sync_types.hpp"

namespace tt::tt_metal::eth_sync {

struct LinkSyncConfig {
    uint32_t n_samples = 256;
    // Spreads the samples in time. The RATE half of the solve is baseline-limited -- back-to-back trips
    // span microseconds and a few-ppm difference simply does not show up over that.
    uint32_t gap_us = 200;
    uint32_t host_timeout_ms = 60'000;
};

struct LinkSyncResult {
    EthSyncSolution solution;
    // The RAW round trips behind the solution, kept so a caller can render them. They are MEASUREMENTS,
    // not products of the fit, which is the whole reason they are worth showing: a fitted anchor cannot
    // contradict itself, but a raw t1 landing outside its own [t0,t2] contradicts the alignment outright.
    std::vector<Trip> trips;
    uint32_t sender_status = ETH_SYNC_IDLE;
    uint32_t receiver_status = ETH_SYNC_IDLE;
    size_t sender_samples = 0;
    size_t receiver_samples = 0;
};

inline const char* status_name(uint32_t s) {
    switch (s) {
        case ETH_SYNC_IDLE: return "IDLE (kernel never ran)";
        case ETH_SYNC_RUNNING: return "RUNNING (did not finish)";
        case ETH_SYNC_DONE: return "DONE";
        case ETH_SYNC_TIMEOUT_HANDSHAKE: return "TIMEOUT_HANDSHAKE (peer never joined)";
        case ETH_SYNC_TIMEOUT_TXQ: return "TIMEOUT_TXQ";
        case ETH_SYNC_TIMEOUT_WAIT: return "TIMEOUT_WAIT";
        default: return "??";
    }
}

namespace detail_host {

struct EthL1Layout {
    uint32_t handshake;
    uint32_t channel;
    uint32_t result;
};

inline EthL1Layout make_layout(uint32_t base) {
    return EthL1Layout{base, base + 64, base + 128};
}

inline std::vector<EthSyncSample> read_samples(
    tt::Cluster& cluster, int chip, const CoreCoord& core, uint32_t result_addr, uint32_t n, uint32_t& status) {
    EthSyncResult hdr{};
    cluster.read_core(&hdr, sizeof(hdr), tt_cxy_pair(chip, core), result_addr);
    status = hdr.status;
    const uint32_t got = hdr.n_samples < n ? hdr.n_samples : n;
    std::vector<EthSyncSample> out(n);
    if (got != 0) {
        cluster.read_core(
            out.data(), got * sizeof(EthSyncSample), tt_cxy_pair(chip, core), result_addr + sizeof(EthSyncResult));
    }
    out.resize(got);
    return out;
}

}  // namespace detail_host

// Measure one link. Returns the receiver-vs-sender clock relationship; solution.valid is false if the run
// could not produce enough usable round trips, in which case the status fields say why.
inline LinkSyncResult measure_link(
    IDevice* snd_dev,
    const CoreCoord& snd_eth,
    IDevice* rcv_dev,
    const CoreCoord& rcv_eth,
    const LinkSyncConfig& cfg = {}) {
    LinkSyncResult out;

    auto& cluster = MetalContext::instance().get_cluster();
    const auto& hal = MetalContext::instance().hal();
    const uint32_t base =
        static_cast<uint32_t>(hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED));
    const auto lay = detail_host::make_layout(base);

    const double ghz = cluster.get_device_aiclk(snd_dev->id()) / 1000.0;
    const uint32_t gap_cycles = static_cast<uint32_t>(cfg.gap_us * ghz * 1000.0);
    // The pacing alone needs n * gap; give the run that plus a second of slack.
    const uint64_t timeout_cycles =
        static_cast<uint64_t>(static_cast<double>(cfg.n_samples) * gap_cycles) + static_cast<uint64_t>(ghz * 1e9);

    const std::vector<uint32_t> snd_args = {
        cfg.n_samples,
        static_cast<uint32_t>(timeout_cycles & 0xFFFFFFFFull),
        static_cast<uint32_t>(timeout_cycles >> 32),
        lay.result,
        lay.channel,
        lay.handshake,
        gap_cycles,
        0u,   // resident: one-shot
        0u,   // mailbox unused
        0u,
        0u};
    std::vector<uint32_t> rcv_args = snd_args;
    rcv_args[6] = 0;  // only the sender paces; the receiver just answers

    Program p_snd = CreateProgram();
    Program p_rcv = CreateProgram();
    CreateKernel(
        p_snd,
        "tt_metal/tools/profiler/sync/eth_wallclock_sync_sender.cpp",
        snd_eth,
        EthernetConfig{.noc = NOC::RISCV_0_default, .compile_args = snd_args});
    CreateKernel(
        p_rcv,
        "tt_metal/tools/profiler/sync/eth_wallclock_sync_receiver.cpp",
        rcv_eth,
        EthernetConfig{.noc = NOC::RISCV_0_default, .compile_args = rcv_args});

    detail::CompileProgram(rcv_dev, p_rcv);
    detail::CompileProgram(snd_dev, p_snd);
    // Receiver first so it is already waiting when the sender handshakes.
    detail::LaunchProgram(rcv_dev, p_rcv, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);
    detail::LaunchProgram(snd_dev, p_snd, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);

    const CoreCoord snd_v = snd_dev->virtual_core_from_logical_core(snd_eth, CoreType::ETH);
    const CoreCoord rcv_v = rcv_dev->virtual_core_from_logical_core(rcv_eth, CoreType::ETH);

    // Poll the kernels' own status word rather than a completion API: the whole point of the bounded
    // design is that the host is never at the mercy of a kernel that will not return.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(cfg.host_timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        EthSyncResult a{}, b{};
        cluster.read_core(&a, sizeof(a), tt_cxy_pair(snd_dev->id(), snd_v), lay.result);
        cluster.read_core(&b, sizeof(b), tt_cxy_pair(rcv_dev->id(), rcv_v), lay.result);
        out.sender_status = a.status;
        out.receiver_status = b.status;
        if (a.status >= ETH_SYNC_DONE && b.status >= ETH_SYNC_DONE) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    uint32_t sa = 0, sb = 0;
    auto snd_s = detail_host::read_samples(cluster, snd_dev->id(), snd_v, lay.result, cfg.n_samples, sa);
    auto rcv_s = detail_host::read_samples(cluster, rcv_dev->id(), rcv_v, lay.result, cfg.n_samples, sb);
    out.sender_status = sa;
    out.receiver_status = sb;
    out.sender_samples = snd_s.size();
    out.receiver_samples = rcv_s.size();

    // Both kernels are deadline-bounded, so they always return -- which is what makes this wait safe.
    detail::WaitProgramDone(snd_dev, p_snd, false);
    detail::WaitProgramDone(rcv_dev, p_rcv, false);

    const size_t n = snd_s.size() < rcv_s.size() ? snd_s.size() : rcv_s.size();
    if (n >= 4) {
        out.trips = build_trips(snd_s, rcv_s, n);
        out.solution = solve(out.trips);
    }
    return out;
}

// ---- RESIDENT link: one launch, many measurement rounds -------------------------------------------
//
// The per-measurement LaunchProgram is what makes mid-capture re-syncs illegal (dram_barrier against
// resident drainers) and expensive. A resident pair is launched ONCE and then commanded through an L1
// mailbox: each GO runs one bounded measurement burst with the samples landing at the same L1 layout the
// one-shot uses, so read_samples/build_trips/solve are reused untouched.
struct ResidentLink {
    IDevice* snd_dev = nullptr;
    IDevice* rcv_dev = nullptr;
    CoreCoord snd_eth, rcv_eth, snd_v, rcv_v;
    Program p_snd, p_rcv;
    detail_host::EthL1Layout lay{};
    uint32_t mbox = 0;
    LinkSyncConfig cfg{};
    bool up = false;
};

inline ResidentLink start_resident_link(
    IDevice* snd_dev,
    const CoreCoord& snd_eth,
    IDevice* rcv_dev,
    const CoreCoord& rcv_eth,
    const LinkSyncConfig& cfg = {},
    uint64_t idle_spin_cap = 120ull * 1000 * 1000 * 1000) {
    ResidentLink L;
    L.snd_dev = snd_dev;
    L.rcv_dev = rcv_dev;
    L.snd_eth = snd_eth;
    L.rcv_eth = rcv_eth;
    L.cfg = cfg;

    auto& cluster = MetalContext::instance().get_cluster();
    const auto& hal = MetalContext::instance().hal();
    const uint32_t base =
        static_cast<uint32_t>(hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED));
    L.lay = detail_host::make_layout(base);
    L.mbox = base + 8192;  // far above result header + 256 samples; nothing else lives there

    const double ghz = cluster.get_device_aiclk(snd_dev->id()) / 1000.0;
    const uint32_t gap_cycles = static_cast<uint32_t>(cfg.gap_us * ghz * 1000.0);
    const uint64_t timeout_cycles =
        static_cast<uint64_t>(static_cast<double>(cfg.n_samples) * gap_cycles) + static_cast<uint64_t>(ghz * 1e9);

    const std::vector<uint32_t> snd_args = {
        cfg.n_samples,
        static_cast<uint32_t>(timeout_cycles & 0xFFFFFFFFull),
        static_cast<uint32_t>(timeout_cycles >> 32),
        L.lay.result,
        L.lay.channel,
        L.lay.handshake,
        gap_cycles,
        1u,  // resident
        L.mbox,
        static_cast<uint32_t>(idle_spin_cap & 0xFFFFFFFFull),
        static_cast<uint32_t>(idle_spin_cap >> 32)};
    std::vector<uint32_t> rcv_args = snd_args;
    rcv_args[6] = 0;

    CreateKernel(
        L.p_snd,
        "tt_metal/tools/profiler/sync/eth_wallclock_sync_sender.cpp",
        snd_eth,
        EthernetConfig{.noc = NOC::RISCV_0_default, .compile_args = snd_args});
    CreateKernel(
        L.p_rcv,
        "tt_metal/tools/profiler/sync/eth_wallclock_sync_receiver.cpp",
        rcv_eth,
        EthernetConfig{.noc = NOC::RISCV_0_default, .compile_args = rcv_args});
    detail::CompileProgram(rcv_dev, L.p_rcv);
    detail::CompileProgram(snd_dev, L.p_snd);
    // Receiver first, same as the one-shot.
    detail::LaunchProgram(rcv_dev, L.p_rcv, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);
    detail::LaunchProgram(snd_dev, L.p_snd, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);

    L.snd_v = snd_dev->virtual_core_from_logical_core(snd_eth, CoreType::ETH);
    L.rcv_v = rcv_dev->virtual_core_from_logical_core(rcv_eth, CoreType::ETH);
    L.up = true;
    return L;
}

inline LinkSyncResult resident_round(ResidentLink& L) {
    LinkSyncResult out;
    if (!L.up) {
        return out;
    }
    auto& cluster = MetalContext::instance().get_cluster();
    const tt_cxy_pair sc(L.snd_dev->id(), L.snd_v);
    const tt_cxy_pair rc(L.rcv_dev->id(), L.rcv_v);
    // Old DONE would satisfy the poll before the new round starts; clear both status words first.
    const uint32_t idle = ETH_SYNC_IDLE;
    cluster.write_core(&idle, sizeof(idle), sc, L.lay.result);
    cluster.write_core(&idle, sizeof(idle), rc, L.lay.result);
    const uint32_t go = 1;
    cluster.write_core(&go, sizeof(go), rc, L.mbox);  // receiver first, so it is waiting at the handshake
    cluster.write_core(&go, sizeof(go), sc, L.mbox);

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(L.cfg.host_timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        EthSyncResult a{}, b{};
        cluster.read_core(&a, sizeof(a), sc, L.lay.result);
        cluster.read_core(&b, sizeof(b), rc, L.lay.result);
        out.sender_status = a.status;
        out.receiver_status = b.status;
        if (a.status >= ETH_SYNC_DONE && b.status >= ETH_SYNC_DONE) {
            break;
        }
    }
    uint32_t sa = 0, sb = 0;
    auto snd_s = detail_host::read_samples(cluster, L.snd_dev->id(), L.snd_v, L.lay.result, L.cfg.n_samples, sa);
    auto rcv_s = detail_host::read_samples(cluster, L.rcv_dev->id(), L.rcv_v, L.lay.result, L.cfg.n_samples, sb);
    out.sender_status = sa;
    out.receiver_status = sb;
    out.sender_samples = snd_s.size();
    out.receiver_samples = rcv_s.size();
    const size_t n = snd_s.size() < rcv_s.size() ? snd_s.size() : rcv_s.size();
    if (n >= 4) {
        out.trips = build_trips(snd_s, rcv_s, n);
        out.solution = solve(out.trips);
    }
    return out;
}

inline void stop_resident_link(ResidentLink& L) {
    if (!L.up) {
        return;
    }
    auto& cluster = MetalContext::instance().get_cluster();
    const tt_cxy_pair sc(L.snd_dev->id(), L.snd_v);
    const tt_cxy_pair rc(L.rcv_dev->id(), L.rcv_v);
    const uint32_t ex = 2;
    cluster.write_core(&ex, sizeof(ex), sc, L.mbox);
    cluster.write_core(&ex, sizeof(ex), rc, L.mbox);
    // Wait for the exit markers (bounded), THEN reap -- the kernels always return, by construction.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < deadline) {
        uint32_t ma = 0, mb = 0;
        cluster.read_core(&ma, sizeof(ma), sc, L.mbox);
        cluster.read_core(&mb, sizeof(mb), rc, L.mbox);
        if (ma == 0xD00DD00Du && mb == 0xD00DD00Du) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    detail::WaitProgramDone(L.snd_dev, L.p_snd, false);
    detail::WaitProgramDone(L.rcv_dev, L.p_rcv, false);
    L.up = false;
}

}  // namespace tt::tt_metal::eth_sync
