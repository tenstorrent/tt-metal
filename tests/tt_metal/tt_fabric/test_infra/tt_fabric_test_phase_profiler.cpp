// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/tt_metal/tt_fabric/test_infra/tt_fabric_test_phase_profiler.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/math.hpp>

#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/telemetry/code_profiling_types.hpp"

namespace {

// Mirrors the allocation order in the FabricEriscDatamoverConfig constructor.
uint32_t get_phase_profile_buffer_addr() {
    uint32_t addr = ::tt::tt_metal::hal::get_erisc_l1_unreserved_base();
    auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    if (rtoptions.get_enable_fabric_bw_telemetry() ||
        tt::tt_metal::MetalContext::instance().hal().get_arch() == tt::ARCH::BLACKHOLE) {
        addr += 32;
    }
    if (rtoptions.get_enable_fabric_code_profiling_rx_ch_fwd()) {
        addr += get_max_code_profiling_timer_types() * sizeof(CodeProfilingTimerResult);
    }
    return tt::align(addr, 16);
}

FabricPhaseProfile extract(const EthCoreBufferResult& result, uint32_t erisc) {
    FabricPhaseProfile profile{};
    const size_t offset = erisc * sizeof(FabricPhaseProfile);
    if (result.buffer_data.size() * sizeof(uint32_t) >= offset + sizeof(FabricPhaseProfile)) {
        std::memcpy(
            &profile, reinterpret_cast<const uint8_t*>(result.buffer_data.data()) + offset, sizeof(FabricPhaseProfile));
    }
    return profile;
}

FabricPhaseProfile diff(const FabricPhaseProfile& after, const FabricPhaseProfile& before) {
    FabricPhaseProfile d{};
    constexpr size_t words = sizeof(FabricPhaseProfile) / sizeof(uint64_t);
    std::array<uint64_t, words> a{}, b{}, r{};
    std::memcpy(a.data(), &after, sizeof(after));
    std::memcpy(b.data(), &before, sizeof(before));
    for (size_t i = 0; i < words; i++) {
        r[i] = a[i] - b[i];
    }
    std::memcpy(&d, r.data(), sizeof(d));
    return d;
}

constexpr std::array<const char*, PHASE_COUNT> phase_names = {
    "tx_send",
    "tx_txq_spin",
    "tx_completion",
    "tx_conn",
    "rx_decode_admit",
    "rx_forward",
    "rx_local",
    "rx_completion",
    "rx_ack",
    "tx_txq_prespin",
    "tx_visit",
    "rx_visit"};
constexpr std::array<const char*, EVENT_COUNT> event_names = {
    "tx_blk_credit", "tx_blk_txq", "rx_blk_admit", "rx_blk_trid"};
constexpr std::array<const char*, TXQ_REGISTER_COUNT> txq_register_names = {
    "transfers",
    "pkts_started",
    "pkts_sent",
    "words",
    "resends",
    "ctrl",
    "max_pkt_size",
    "burst_len",
    "remote_seq_timeout",
    "local_seq_update_timeout",
    "min_pkt_words",
    "accept_ahead"};

double per(uint64_t num, uint64_t den) { return den ? static_cast<double>(num) / static_cast<double>(den) : 0.0; }

bool env_set(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && std::strtoul(value, nullptr, 0) != 0;
}

// TX queue values for one test: counters as the change over the test (modulo 2^32), configuration as read after it.
std::array<std::array<uint64_t, TXQ_REGISTER_COUNT>, PHASE_PROFILE_TXQS> txq_values(
    const FabricPhaseProfile& after, const FabricPhaseProfile& before) {
    std::array<std::array<uint64_t, TXQ_REGISTER_COUNT>, PHASE_PROFILE_TXQS> v{};
    for (uint32_t q = 0; q < PHASE_PROFILE_TXQS; q++) {
        for (uint32_t r = 0; r < TXQ_REGISTER_COUNT; r++) {
            v[q][r] = r < TXQ_COUNTER_COUNT
                          ? static_cast<uint32_t>(
                                static_cast<uint32_t>(after.txq[q][r]) - static_cast<uint32_t>(before.txq[q][r]))
                          : after.txq[q][r];
        }
    }
    return v;
}

}  // namespace

PhaseProfiler::PhaseProfiler(EthCoreBufferReadback& eth_readback) : eth_readback_(eth_readback) {}

bool PhaseProfiler::is_enabled() {
    return env_set("TT_FABRIC_PHASE_PROFILE") || env_set("TT_FABRIC_TXQ_COUNTERS") ||
           env_set("TT_FABRIC_EXP_STACK_PAINT") || env_set("TT_FABRIC_DRISC_PHASE_PROFILE");
}

void PhaseProfiler::begin() {
    before_ = eth_readback_.read_buffer(get_phase_profile_buffer_addr(), PHASE_PROFILE_BUFFER_SIZE, true);
    const char* level = std::getenv("TT_FABRIC_PHASE_PROFILE");
    if (!microbench_reported_ && level != nullptr && std::strtoul(level, nullptr, 0) >= 3) {
        microbench_reported_ = true;
        report_microbench();
    }
}

void PhaseProfiler::report_microbench() {
    constexpr std::array<const char*, MB_SINK> names = {
        "cycle CSR read",
        "dependent L1 load",
        "stream register read",
        "eth txq busy read",
        "wall clock register read",
        "NoC cmd buf ready read",
        "NoC write 64B (stateless, back-to-back)",
        "NoC write 2KB (stateless, back-to-back)",
        "NoC write 64B (stateful, back-to-back)",
        "NoC inline 4B L1 write (emulated)",
        "straight-line code, 64 instr",
        "straight-line code, 128 instr",
        "straight-line code, 256 instr",
        "straight-line code, 512 instr",
        "straight-line code, 1024 instr",
        "straight-line code, 2048 instr"};
    auto results = eth_readback_.read_buffer(
        get_phase_profile_buffer_addr() + PHASE_PROFILE_BUFFER_SIZE, PHASE_PROFILE_MICROBENCH_RESULTS_SIZE, true);
    std::array<uint32_t, MB_SINK> lo{}, hi{};
    lo.fill(UINT32_MAX);
    uint32_t cores = 0;
    for (const auto& r : results) {
        if (r.buffer_data.size() <= MB_DONE || r.buffer_data[MB_DONE] != 1) {
            continue;
        }
        cores++;
        for (uint32_t i = 0; i < MB_SINK; i++) {
            lo[i] = std::min(lo[i], r.buffer_data[i]);
            hi[i] = std::max(hi[i], r.buffer_data[i]);
        }
    }
    log_info(tt::LogTest, "ERISC primitive costs over {} cores (core cycles per op, min..max):", cores);
    for (uint32_t i = 0; cores != 0 && i < MB_SINK; i++) {
        log_info(tt::LogTest, "  {:42s} {:7.2f} .. {:7.2f}", names[i], lo[i] / 100.0, hi[i] / 100.0);
    }
}

void PhaseProfiler::end_and_report(const std::string& test_name) {
    auto after = eth_readback_.read_buffer(get_phase_profile_buffer_addr(), PHASE_PROFILE_BUFFER_SIZE, true);

    std::ofstream csv;
    if (const char* csv_path = std::getenv("TT_FABRIC_PHASE_PROFILE_CSV")) {
        const bool write_header = !std::filesystem::exists(csv_path);
        csv.open(csv_path, std::ios::app);
        if (write_header) {
            csv << "test,device,eth_channel,direction,erisc,info,timer_cal,wall,cycles,pkts,iters,busy,blocked,idle,"
                   "gap";
            for (const auto* name : event_names) {
                csv << "," << name;
            }
            for (const auto* name : phase_names) {
                csv << "," << name << "_ticks," << name << "_count";
            }
            csv << ",txq_mask";
            for (uint32_t q = 0; q < PHASE_PROFILE_TXQS; q++) {
                for (const auto* name : txq_register_names) {
                    csv << ",txq" << q << "_" << name;
                }
            }
            csv << ",stack_main_sp,stack_lowest_used,stack_limit";
            csv << "\n";
        }
    }

    log_info(tt::LogTest, "Phase profile for {} (ERISC core cycles):", test_name);
    // TT_FABRIC_EXP_STACK_PAINT: per ERISC, the deepest stack seen on any router and the least free stack left.
    std::array<uint32_t, PHASE_PROFILE_MAX_ERISCS> stack_routers{}, stack_max_below_main{};
    std::array<int64_t, PHASE_PROFILE_MAX_ERISCS> stack_min_free{};
    stack_min_free.fill(INT64_MAX);
    for (const auto& end : after) {
        const EthCoreBufferResult* start = nullptr;
        for (const auto& candidate : before_) {
            if (candidate.coord == end.coord && candidate.eth_channel == end.eth_channel) {
                start = &candidate;
                break;
            }
        }
        if (start == nullptr) {
            continue;
        }
        // Blocks 0 and 1 are the router's ERISCs; PHASE_PROFILE_DRISC_BLOCK is its offload DRISC, if any.
        for (uint32_t erisc = 0; erisc < PHASE_PROFILE_BLOCKS; erisc++) {
            const auto raw = extract(end, erisc);
            const auto before = extract(*start, erisc);
            const auto d = diff(raw, before);
            const bool stack_valid = erisc < PHASE_PROFILE_MAX_ERISCS && raw.stack_painted == 1 &&
                                     raw.stack_lowest_used >= raw.stack_limit &&
                                     raw.stack_lowest_used <= raw.stack_main_sp;
            if (stack_valid) {
                stack_routers[erisc]++;
                stack_max_below_main[erisc] =
                    std::max(stack_max_below_main[erisc], raw.stack_main_sp - raw.stack_lowest_used);
                stack_min_free[erisc] =
                    std::min(stack_min_free[erisc], static_cast<int64_t>(raw.stack_lowest_used - raw.stack_limit));
            }
            const auto txq = txq_values(raw, before);
            std::string txqs;
            for (uint32_t q = 0; q < PHASE_PROFILE_TXQS; q++) {
                const auto& v = txq[q];
                if (((raw.txq_mask >> q) & 1) == 0 || v[TXQ_TRANSFERS] == 0) {
                    continue;
                }
                txqs += fmt::format(
                    " txq{}: transfers {} eth pkts {} ({:.2f}/transfer) sent {} words {} ({:.1f}/transfer) resends {}"
                    " [max pkt {} B, burst {}, accept-ahead {}, ctrl {:#x}, min pkt {} words, seq timeouts {}/{}]",
                    q,
                    v[TXQ_TRANSFERS],
                    v[TXQ_PKTS_STARTED],
                    per(v[TXQ_PKTS_STARTED], v[TXQ_TRANSFERS]),
                    v[TXQ_PKTS_SENT],
                    v[TXQ_WORDS],
                    per(v[TXQ_WORDS], v[TXQ_TRANSFERS]),
                    v[TXQ_RESENDS],
                    v[TXQ_MAX_PKT_SIZE],
                    v[TXQ_BURST_LEN],
                    v[TXQ_ACCEPT_AHEAD],
                    v[TXQ_CTRL],
                    v[TXQ_MIN_PKT_WORDS],
                    v[TXQ_REMOTE_SEQ_TIMEOUT],
                    v[TXQ_LOCAL_SEQ_UPDATE_TIMEOUT]);
            }
            if (d.pkts == 0 && txqs.empty() && !stack_valid) {
                continue;
            }
            const uint64_t active = d.busy_ticks + d.blocked_ticks + d.idle_ticks;
            std::string phases;
            for (uint32_t p = 0; p < PHASE_COUNT; p++) {
                if (d.phase_count[p] != 0) {
                    phases += fmt::format(
                        " {}={:.1f}/pkt({:.1f}x{:.2f})",
                        phase_names[p],
                        per(d.phase_ticks[p], d.pkts),
                        per(d.phase_ticks[p], d.phase_count[p]),
                        per(d.phase_count[p], d.pkts));
                }
            }
            std::string events;
            for (uint32_t e = 0; e < EVENT_COUNT; e++) {
                if (d.events[e] != 0) {
                    events += fmt::format(" {}={:.2f}/pkt", event_names[e], per(d.events[e], d.pkts));
                }
            }
            log_info(
                tt::LogTest,
                "  dev {} ch {:2d} dir {} {}: pkts {:7d} active/pkt {:6.1f} busy {:5.1f}% blocked {:5.1f}% idle "
                "{:5.1f}% busy/pkt {:6.1f} iters/pkt {:5.2f} cyc/wall {:.3f}{}{}{}",
                end.coord,
                static_cast<uint32_t>(end.eth_channel),
                static_cast<int>(end.direction),
                erisc == PHASE_PROFILE_DRISC_BLOCK ? std::string("drisc  ") : fmt::format("erisc {}", erisc),
                d.pkts,
                per(active, d.pkts),
                100.0 * per(d.busy_ticks, active),
                100.0 * per(d.blocked_ticks, active),
                100.0 * per(d.idle_ticks, active),
                per(d.busy_ticks, d.pkts),
                per(d.iters, d.pkts),
                per(d.cycle_ts, d.wall_ts),
                events,
                phases,
                txqs);
            if (csv.is_open()) {
                csv << test_name << ",\"" << fmt::format("{}", end.coord) << "\","
                    << static_cast<uint32_t>(end.eth_channel) << "," << static_cast<int>(end.direction) << "," << erisc
                    << "," << raw.info << "," << raw.timer_cal << "," << d.wall_ts << "," << d.cycle_ts << "," << d.pkts
                    << "," << d.iters << "," << d.busy_ticks << "," << d.blocked_ticks << "," << d.idle_ticks << ","
                    << d.gap_ticks;
                for (uint64_t event : d.events) {
                    csv << "," << event;
                }
                for (uint32_t p = 0; p < PHASE_COUNT; p++) {
                    csv << "," << d.phase_ticks[p] << "," << d.phase_count[p];
                }
                csv << "," << raw.txq_mask;
                for (const auto& q : txq) {
                    for (uint64_t value : q) {
                        csv << "," << value;
                    }
                }
                csv << "," << raw.stack_main_sp << "," << raw.stack_lowest_used << "," << raw.stack_limit;
                csv << "\n";
            }
        }
    }
    for (uint32_t erisc = 0; erisc < PHASE_PROFILE_MAX_ERISCS; erisc++) {
        if (stack_routers[erisc] != 0) {
            log_info(
                tt::LogTest,
                "  stack, erisc {} over {} routers: deepest {} B below kernel_main's frame, least free {} B",
                erisc,
                stack_routers[erisc],
                stack_max_below_main[erisc],
                stack_min_free[erisc]);
        }
    }
    before_.clear();
}
