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
        gap_cycles};
    std::vector<uint32_t> rcv_args = snd_args;
    rcv_args.back() = 0;  // only the sender paces; the receiver just answers

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
        out.solution = solve(build_trips(snd_s, rcv_s, n));
    }
    return out;
}

}  // namespace tt::tt_metal::eth_sync
