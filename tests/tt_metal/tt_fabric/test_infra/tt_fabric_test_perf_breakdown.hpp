// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Opt-in cost breakdown for the test infra itself (not the fabric under test), with TT_FABRIC_TEST_PERF_BREAKDOWN=1:
//  - every per-test host phase is timed and the per-core host writes are counted,
//  - sender/receiver kernels are built with FABRIC_TEST_PERF_BREAKDOWN, which makes them
//    accumulate cycle counters into words [PERF_BREAKDOWN_BASE_WORD_INDEX, +16) of the result
//    buffer (layout must match tt_fabric_test_kernels_utils.hpp),
//  - one "PERF_BD" line per test and a run summary are logged.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <tt-logger/tt-logger.hpp>

namespace tt::tt_fabric::fabric_tests::perf_breakdown {

// ---------------------------------------------------------------------------------------------------------------------
// Knobs: every environment variable the test infra reads, each TT_FABRIC_TEST_<name>, read once per process. The
// traffic and its checks are the same with any of them on, except CORRUPT_PAYLOAD.
//
// Profiling
//   PERF_BREAKDOWN=1      Per-test host phase times and sender/receiver device cycle counters: one "PERF_BD" line per
//                         test and a run summary (read by handoff/scripts/device_time_by_*.py and log_phases.py).
// Profiling detail, each only with PERF_BREAKDOWN=1
//   PERF_HOST_DETAIL=1    Host sub-phases of create_kernels / enqueue_workload, per-device build times and workload
//                         sizes. Prepares each workload before its enqueue to time the two apart, so host times with it
//                         on are not comparable to host times with it off.
//   PERF_PER_CORE=1       One PERF_CORE line per sender core: its loop, waits and the destinations of its configs.
//   PERF_SEND_DETAIL=1    Sender per-packet step timing, in place of setup / close / mux termination.
//   PERF_HOT_DETAIL=<p>   Hot-path sender step timing on one send in every p (a power of two; 0 = off).
//   PERF_SCHED=1          Hot-path sender scheduling counters and per-config finish times.
// Other
//   ROUTER_TELEMETRY=1    Turns on the fabric routers' bandwidth counters and logs GB/s per ethernet link after every
//                         test, with validation still on (about 5% more device time).
//   CORRUPT_PAYLOAD=<m>   Verifier canary: senders corrupt one word of one packet per config (1 mid-payload, 2 the
//                         second-to-last word, 3 the third-to-last; corrupt_payload_canary in the kernels), so the run
//                         must fail with a data mismatch.
// ---------------------------------------------------------------------------------------------------------------------

inline bool read_flag_knob(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

inline uint32_t read_uint_knob(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr ? static_cast<uint32_t>(std::strtoul(v, nullptr, 10)) : 0u;
}

inline bool enabled() {
    static const bool on = read_flag_knob("TT_FABRIC_TEST_PERF_BREAKDOWN");
    return on;
}

inline bool host_detail() {
    static const bool on = enabled() && read_flag_knob("TT_FABRIC_TEST_PERF_HOST_DETAIL");
    return on;
}

inline bool perf_per_core() {
    static const bool on = enabled() && read_flag_knob("TT_FABRIC_TEST_PERF_PER_CORE");
    return on;
}

inline bool send_detail() {
    static const bool on = enabled() && read_flag_knob("TT_FABRIC_TEST_PERF_SEND_DETAIL");
    return on;
}

inline uint32_t hot_detail_period() {
    static const uint32_t period = enabled() ? read_uint_knob("TT_FABRIC_TEST_PERF_HOT_DETAIL") : 0u;
    return period;
}

inline bool perf_sched() {
    static const bool on = enabled() && read_flag_knob("TT_FABRIC_TEST_PERF_SCHED");
    return on;
}

inline bool router_telemetry() {
    static const bool on = read_flag_knob("TT_FABRIC_TEST_ROUTER_TELEMETRY");
    return on;
}

inline uint32_t corrupt_payload() {
    static const uint32_t mode = std::min<uint32_t>(read_uint_knob("TT_FABRIC_TEST_CORRUPT_PAYLOAD"), 3);
    return mode;
}

inline constexpr uint32_t PERF_BREAKDOWN_BASE_WORD_INDEX = 16;
inline constexpr uint32_t NUM_SLOTS = 8;

// Sender slots (cycles unless noted)
enum SenderSlot : uint32_t {
    S_SETUP = 0,           // kernel start -> main loop start (open connections, line sync)
    S_LOOP = 1,            // main send loop
    S_FILL = 2,            // payload pattern fill
    S_SLOT_WAIT = 3,       // waiting for a free fabric/mux write slot
    S_FLUSH = 4,           // noc_async_writes_flushed after each packet
    S_CREDIT_BLOCKED = 5,  // count: send attempts rejected for lack of receiver credits
    S_CLOSE = 6,           // close connections + end line sync
    S_MUX_TERM = 7,        // mux termination handshake
};

// Sender slots with TT_FABRIC_TEST_PERF_SEND_DETAIL=1: the per-packet send path split into steps. Slots 1-4 keep
// their meaning; setup, close and mux termination are not recorded.
enum SenderDetailSlot : uint32_t {
    SD_CREDIT_CHECK = 0,  // receiver credit check
    SD_PAYLOAD_SEND = 5,  // payload NOC write to the fabric/mux slot
    SD_HEADER_SEND = 6,   // header NOC write + slot handoff
    SD_POST = 7,          // after the flush: page pop, address/header advance, seed, credit consume
};

// Hot-state sender scheduling record (TT_FABRIC_TEST_PERF_SCHED=1), from word PERF_SCHED_BASE_WORD_INDEX of the
// result buffer: PERF_SCHED_NUM_COUNTERS u64 counters, then PERF_SCHED_MAX_CONFIGS words of per-config finish cycles
// (from loop start) and as many of per-config credit misses. Layout must match tt_fabric_test_kernels_utils.hpp.
inline constexpr uint32_t PERF_SCHED_BASE_WORD_INDEX = 512;
inline constexpr uint32_t PERF_SCHED_NUM_COUNTERS = 5;
inline constexpr uint32_t PERF_SCHED_MAX_CONFIGS = 160;

enum SenderSchedCounter : uint32_t {
    SC_PACKETS = 0,        // packets sent
    SC_IDLE_CYCLES = 1,    // loop passes that sent nothing
    SC_CREDIT_MISSES = 2,  // send attempts without receiver credits
    SC_PASSES = 3,         // passes over all configs
    SC_IDLE_PASSES = 4,    // passes that sent nothing
};

// Sampled hot-sender step record (TT_FABRIC_TEST_PERF_HOT_DETAIL=<period>), from word
// PERF_HOT_DETAIL_BASE_WORD_INDEX of the result buffer: PERF_HOT_NUM_STEPS u64, cycles except the counts. Layout must
// match tt_fabric_test_kernels_utils.hpp.
inline constexpr uint32_t PERF_HOT_DETAIL_BASE_WORD_INDEX = 864;
inline constexpr uint32_t PERF_HOT_CALIBRATION_STEPS = 64;

enum SenderHotStep : uint32_t {
    HS_SAMPLES = 0,      // count: sampled sends that sent a packet
    HS_CALIBRATION = 1,  // PERF_HOT_CALIBRATION_STEPS empty steps (the cost of one step boundary)
    HS_GAPS = 2,         // count: sampled sends followed by another send
    HS_GAP = 3,          // send loop between a sampled send's return and the next send
    HS_CREDIT = 4,       // receiver credit check
    HS_SLOT = 5,         // free-slot poll of the fabric/mux connection, including any wait
    HS_FILL = 6,         // SFPU page wait (or RISC pattern fill)
    HS_PAYLOAD = 7,      // payload NOC write
    HS_HEADER = 8,       // header NOC write + slot handoff to the fabric/mux
    HS_FLUSH = 9,        // noc_async_writes_flushed
    HS_POST = 10,        // page pop, header advance, seed, counters
    HS_CALLS = 11,       // count: all hot sends, sampled or not
    PERF_HOT_NUM_STEPS = 12,
};

// What a sender's config finish times are grouped by.
struct SchedConfigInfo {
    uint32_t hops = 0;       // fabric hops to the farthest destination
    uint32_t conn_load = 0;  // configs on the same sender sharing this config's first connection
};

// Receiver slots (cycles unless noted)
enum ReceiverSlot : uint32_t {
    R_SETUP = 0,      // kernel start -> main loop start (open credit connections)
    R_LOOP = 1,       // main validation loop
    R_VALIDATE = 2,   // payload compare
    R_ADVANCE = 3,    // advance(): bookkeeping + credit return
    R_FIRST_PKT = 4,  // loop start -> first packet seen
    R_POLL_MISS = 5,  // count: polls that found no new data
    R_CLOSE = 6,      // close credit connections (flush credits)
    R_MUX_TERM = 7,   // mux termination handshake
};

using Clock = std::chrono::steady_clock;

inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

struct Stat {
    uint64_t count = 0;
    double sum = 0;
    double max = 0;
    void add(double v) {
        count++;
        sum += v;
        max = std::max(max, v);
    }
    double avg() const { return count ? sum / count : 0; }
};

class Collector {
public:
    static Collector& get() {
        static Collector c;
        return c;
    }

    // ---- host-side ----
    // The add_* methods may be called from the per-device build threads (TestContext::compile_programs).
    void add_phase(const std::string& phase, double ms) {
        std::lock_guard<std::mutex> lock(mutex_);
        test_phases_[phase] += ms;
        run_phases_[phase].add(ms);
        if (std::find(phase_order_.begin(), phase_order_.end(), phase) == phase_order_.end()) {
            phase_order_.push_back(phase);
        }
    }
    void add_host_write(bool is_zero, double ms, uint64_t bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& s = is_zero ? zero_writes_ : arg_writes_;
        s.add(ms);
        (is_zero ? zero_bytes_ : arg_bytes_) += bytes;
    }
    void add_kernel_created() {
        std::lock_guard<std::mutex> lock(mutex_);
        kernels_created_++;
    }
    void add_device_build(double ms) {
        std::lock_guard<std::mutex> lock(mutex_);
        device_build_.add(ms);
    }
    void add_rt_args(size_t words) {
        std::lock_guard<std::mutex> lock(mutex_);
        rt_arg_words_ += words;
    }
    // One kernel created on one core: its kind (source basename) and a hash of everything that makes it a distinct
    // kernel (source, processor, NOC, compile args, defines). Distinct signatures per device = the kernel objects a
    // device would need if cores with identical kernels shared one kernel.
    void add_kernel_signature(const void* device, const std::string& src, size_t signature) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto kind = src.substr(src.find_last_of('/') + 1);
        kind = kind.substr(0, kind.find('.'));
        for (const char* prefix : {"tt_fabric_test_", "tt_fabric_"}) {
            if (kind.starts_with(prefix)) {
                kind = kind.substr(std::string(prefix).size());
                break;
            }
        }
        kernel_signatures_[kind][device].insert(signature);
        kernel_counts_[kind]++;
    }
    void add_workload(uint64_t programs, uint64_t kernels) {
        std::lock_guard<std::mutex> lock(mutex_);
        workload_programs_ += programs;
        workload_kernels_ += kernels;
    }
    void add_mux(uint32_t full_size_channels, uint32_t header_only_channels) {
        std::lock_guard<std::mutex> lock(mutex_);
        muxes_++;
        max_mux_full_ = std::max(max_mux_full_, full_size_channels);
        max_mux_header_ = std::max(max_mux_header_, header_only_channels);
        max_mux_total_ = std::max(max_mux_total_, full_size_channels + header_only_channels);
    }

    // ---- device-side ----
    void add_sender(const std::vector<uint32_t>& data, uint32_t num_configs) { add_core(sender_, data, num_configs); }
    void add_receiver(const std::vector<uint32_t>& data, uint32_t num_configs) {
        add_core(receiver_, data, num_configs);
    }
    void set_aiclk_mhz(uint32_t mhz) { aiclk_mhz_ = mhz; }

    void add_sender_hot_detail(const std::vector<uint32_t>& data, uint64_t packets) {
        if (data.size() < PERF_HOT_DETAIL_BASE_WORD_INDEX + 2 * PERF_HOT_NUM_STEPS) {
            return;
        }
        for (uint32_t i = 0; i < PERF_HOT_NUM_STEPS; i++) {
            const uint32_t idx = PERF_HOT_DETAIL_BASE_WORD_INDEX + 2 * i;
            hot_.steps[i] += static_cast<uint64_t>(data[idx]) | (static_cast<uint64_t>(data[idx + 1]) << 32);
        }
        hot_.loop += read_u64(data, S_LOOP);
        hot_.packets += packets;
        hot_.cores++;
    }

    void add_sender_sched(const std::vector<uint32_t>& data, const std::vector<SchedConfigInfo>& configs) {
        const uint32_t finish_base = PERF_SCHED_BASE_WORD_INDEX + 2 * PERF_SCHED_NUM_COUNTERS;
        if (data.size() < finish_base + 2 * PERF_SCHED_MAX_CONFIGS || configs.size() > PERF_SCHED_MAX_CONFIGS) {
            return;
        }
        std::array<uint64_t, PERF_SCHED_NUM_COUNTERS> counters{};
        for (uint32_t i = 0; i < PERF_SCHED_NUM_COUNTERS; i++) {
            const uint32_t idx = PERF_SCHED_BASE_WORD_INDEX + 2 * i;
            counters[i] = static_cast<uint64_t>(data[idx]) | (static_cast<uint64_t>(data[idx + 1]) << 32);
            sched_.counters[i].add(static_cast<double>(counters[i]));
        }
        // Waiting: blocking slot waits (slot S_SLOT_WAIT) plus the loop's idle passes / connection waits.
        const uint64_t loop = read_u64(data, S_LOOP);
        const uint64_t wait = read_u64(data, S_SLOT_WAIT) + counters[SC_IDLE_CYCLES];
        sched_.wait_cycles.add(static_cast<double>(wait));
        if (counters[SC_PACKETS] > 0) {
            sched_.busy_cycles_per_packet.add(
                static_cast<double>(loop - std::min(loop, wait)) / static_cast<double>(counters[SC_PACKETS]));
        }
        double first = std::numeric_limits<double>::max();
        double last = 0;
        for (size_t i = 0; i < configs.size(); i++) {
            const double finish = data[finish_base + i];
            sched_.finish_by_hops[configs[i].hops].add(finish);
            sched_.finish_by_load[configs[i].conn_load].add(finish);
            sched_.misses_by_load[configs[i].conn_load] += data[finish_base + PERF_SCHED_MAX_CONFIGS + i];
            first = std::min(first, finish);
            last = std::max(last, finish);
        }
        if (!configs.empty()) {
            sched_.finish_spread.add(last - first);
        }
        sched_.cores++;
    }

    void end_test(const std::string& test_name) {
        std::string host;
        double total = 0;
        for (const auto& p : phase_order_) {
            auto it = test_phases_.find(p);
            if (it == test_phases_.end()) {
                continue;
            }
            host += fmt::format(" {}={:.1f}", p, it->second);
            if (!is_nested(p)) {
                total += it->second;
            }
        }
        log_info(
            tt::LogTest,
            "PERF_BD host test={} total_ms={:.1f}{} kernels={} muxes={} mux_max_full={} mux_max_hdr={} "
            "mux_max_total={} "
            "zero_writes={}({:.1f}ms,{}B) arg_writes={}({:.1f}ms,{}B)",
            test_name,
            total,
            host,
            kernels_created_,
            muxes_,
            max_mux_full_,
            max_mux_header_,
            max_mux_total_,
            zero_writes_.count,
            zero_writes_.sum,
            zero_bytes_,
            arg_writes_.count,
            arg_writes_.sum,
            arg_bytes_);
        if (host_detail()) {
            log_info(
                tt::LogTest,
                "PERF_HOST test={} dev_build_ms={:.1f}/{:.1f}/{} programs={} kernels={} rt_arg_words={} "
                "sigs[distinct/total]={}",
                test_name,
                device_build_.sum,
                device_build_.max,
                device_build_.count,
                workload_programs_,
                workload_kernels_,
                rt_arg_words_,
                kernel_signature_summary());
        }
        device_build_ = {};
        kernel_signatures_.clear();
        kernel_counts_.clear();
        rt_arg_words_ = workload_programs_ = workload_kernels_ = 0;
        run_zero_writes_ms_ += zero_writes_.sum;
        run_arg_writes_ms_ += arg_writes_.sum;
        run_zero_writes_ += zero_writes_.count;
        run_arg_writes_ += arg_writes_.count;

        if (send_detail()) {
            log_core_stats(
                test_name,
                "snd",
                sender_,
                {"credit_chk", "loop", "fill", "slot_wait", "flush", "payload_send", "header_send", "post"});
        } else {
            log_core_stats(
                test_name,
                "snd",
                sender_,
                {"setup", "loop", "fill", "slot_wait", "flush", "#credit_blk", "close", "mux_term"});
        }
        log_core_stats(
            test_name,
            "rcv",
            receiver_,
            {"setup", "loop", "validate", "advance", "first_pkt", "#poll_miss", "close", "mux_term"});
        if (sched_.cores > 0) {
            log_sched(test_name);
        }
        if (hot_.cores > 0) {
            log_hot_detail(test_name);
        }

        test_phases_.clear();
        zero_writes_ = {};
        arg_writes_ = {};
        zero_bytes_ = arg_bytes_ = 0;
        kernels_created_ = 0;
        muxes_ = max_mux_full_ = max_mux_header_ = max_mux_total_ = 0;
        sender_ = {};
        receiver_ = {};
        sched_ = {};
        hot_ = {};
        num_tests_++;
    }

    void report_run() const {
        log_info(tt::LogTest, "PERF_BD ===== run summary over {} tests (ms) =====", num_tests_);
        double total = 0;
        for (const auto& p : phase_order_) {
            if (!is_nested(p)) {
                total += run_phases_.at(p).sum;
            }
        }
        for (const auto& p : phase_order_) {
            const auto& s = run_phases_.at(p);
            log_info(
                tt::LogTest,
                "PERF_BD summary phase={:<18} sum={:>10.1f} avg={:>8.2f} max={:>8.1f} share={:>5.1f}%",
                p,
                s.sum,
                s.avg(),
                s.max,
                total > 0 ? 100.0 * s.sum / total : 0.0);
        }
        log_info(
            tt::LogTest,
            "PERF_BD summary total={:.1f} zero_writes={} ({:.1f}ms) arg_writes={} ({:.1f}ms)",
            total,
            run_zero_writes_,
            run_zero_writes_ms_,
            run_arg_writes_,
            run_arg_writes_ms_);
    }

private:
    // Phases named "sub_*" are timed inside another phase and are left out of totals.
    static bool is_nested(const std::string& phase) { return phase.starts_with("sub_"); }

    struct CoreStats {
        std::array<Stat, NUM_SLOTS> slots{};
        Stat configs;
        uint64_t cores = 0;
    };

    static uint64_t read_u64(const std::vector<uint32_t>& data, uint32_t slot) {
        const uint32_t idx = PERF_BREAKDOWN_BASE_WORD_INDEX + 2 * slot;
        return static_cast<uint64_t>(data[idx]) | (static_cast<uint64_t>(data[idx + 1]) << 32);
    }

    static void add_core(CoreStats& s, const std::vector<uint32_t>& data, uint32_t num_configs) {
        if (data.size() < PERF_BREAKDOWN_BASE_WORD_INDEX + 2 * NUM_SLOTS) {
            return;
        }
        for (uint32_t i = 0; i < NUM_SLOTS; i++) {
            s.slots[i].add(static_cast<double>(read_u64(data, i)));
        }
        s.configs.add(num_configs);
        s.cores++;
    }

    // Values are cycles except the '#'-prefixed counters. Reported as avg/max over cores, cycles in us.
    void log_core_stats(
        const std::string& test_name,
        const char* role,
        const CoreStats& s,
        const std::array<const char*, NUM_SLOTS>& names) const {
        if (s.cores == 0) {
            return;
        }
        const double cyc_per_us = aiclk_mhz_ > 0 ? aiclk_mhz_ : 1.0;
        std::string fields;
        for (uint32_t i = 0; i < NUM_SLOTS; i++) {
            const bool is_count = names[i][0] == '#';
            const double scale = is_count ? 1.0 : 1.0 / cyc_per_us;
            fields += fmt::format(
                " {}={:.0f}/{:.0f}",
                is_count ? names[i] + 1 : names[i],
                s.slots[i].avg() * scale,
                s.slots[i].max * scale);
        }
        log_info(
            tt::LogTest,
            "PERF_BD dev test={} role={} cores={} configs/core={:.1f}/{:.0f} [avg/max, us]{}",
            test_name,
            role,
            s.cores,
            s.configs.avg(),
            s.configs.max,
            fields);
    }

    struct SchedStats {
        std::array<Stat, PERF_SCHED_NUM_COUNTERS> counters{};
        Stat wait_cycles, busy_cycles_per_packet, finish_spread;
        std::map<uint32_t, Stat> finish_by_hops, finish_by_load;
        std::map<uint32_t, uint64_t> misses_by_load;
        uint64_t cores = 0;
    };

    // Sender scheduling summary: per-core counters (avg/max), then config finish times grouped by hops and by how
    // many configs share the connection, as avg/max ms (configs) [credit misses per config].
    void log_sched(const std::string& test_name) const {
        const double cyc_per_us = aiclk_mhz_ > 0 ? aiclk_mhz_ : 1.0;
        const auto& c = sched_.counters;
        log_info(
            tt::LogTest,
            "PERF_SCHED test={} snd cores={} packets={:.0f} busy_ns/pkt={:.0f}/{:.0f} wait_us={:.0f}/{:.0f} "
            "passes={:.0f} idle_passes={:.0f} credit_miss={:.0f}/{:.0f} "
            "finish_spread_us={:.0f}/{:.0f}",
            test_name,
            sched_.cores,
            c[SC_PACKETS].avg(),
            sched_.busy_cycles_per_packet.avg() * 1000.0 / cyc_per_us,
            sched_.busy_cycles_per_packet.max * 1000.0 / cyc_per_us,
            sched_.wait_cycles.avg() / cyc_per_us,
            sched_.wait_cycles.max / cyc_per_us,
            c[SC_PASSES].avg(),
            c[SC_IDLE_PASSES].avg(),
            c[SC_CREDIT_MISSES].avg(),
            c[SC_CREDIT_MISSES].max,
            sched_.finish_spread.avg() / cyc_per_us,
            sched_.finish_spread.max / cyc_per_us);
        const double cyc_per_ms = cyc_per_us * 1000.0;
        std::string by_hops;
        for (const auto& [hops, s] : sched_.finish_by_hops) {
            by_hops += fmt::format(" {}:{:.0f}/{:.0f}({})", hops, s.avg() / cyc_per_ms, s.max / cyc_per_ms, s.count);
        }
        std::string by_load;
        for (const auto& [load, s] : sched_.finish_by_load) {
            by_load += fmt::format(
                " {}:{:.0f}/{:.0f}({})[{:.0f}]",
                load,
                s.avg() / cyc_per_ms,
                s.max / cyc_per_ms,
                s.count,
                static_cast<double>(sched_.misses_by_load.at(load)) / static_cast<double>(s.count));
        }
        log_info(tt::LogTest, "PERF_SCHED test={} finish_ms by_hops{}", test_name, by_hops);
        log_info(tt::LogTest, "PERF_SCHED test={} finish_ms by_conn_load{}", test_name, by_load);
    }

    struct HotDetailStats {
        std::array<uint64_t, PERF_HOT_NUM_STEPS> steps{};
        uint64_t loop = 0, packets = 0, cores = 0;
    };

    // Per-packet cost of the hot send path in ns, pooled over senders: the loop time per packet, then each step's
    // average over the sampled sends less the cost of one step boundary (calibration), their sum, and the timing cost
    // the sampled sends add to the loop per packet.
    void log_hot_detail(const std::string& test_name) const {
        const double ns_per_cycle = 1000.0 / (aiclk_mhz_ > 0 ? aiclk_mhz_ : 1.0);
        const auto& st = hot_.steps;
        const double cal =
            static_cast<double>(st[HS_CALIBRATION]) / static_cast<double>(PERF_HOT_CALIBRATION_STEPS * hot_.cores);
        // Steps a send does not have (fill and payload without a payload) are left at 0.
        auto per = [&](uint32_t step, uint64_t count) {
            return count && st[step] ? (static_cast<double>(st[step]) / static_cast<double>(count) - cal) * ns_per_cycle
                                     : 0.0;
        };
        const std::array<std::pair<const char*, double>, 8> parts{{
            {"gap", per(HS_GAP, st[HS_GAPS])},
            {"credit", per(HS_CREDIT, st[HS_SAMPLES])},
            {"slot", per(HS_SLOT, st[HS_SAMPLES])},
            {"fill", per(HS_FILL, st[HS_SAMPLES])},
            {"payload", per(HS_PAYLOAD, st[HS_SAMPLES])},
            {"header", per(HS_HEADER, st[HS_SAMPLES])},
            {"flush", per(HS_FLUSH, st[HS_SAMPLES])},
            {"post", per(HS_POST, st[HS_SAMPLES])},
        }};
        std::string fields;
        double sum = 0;
        uint32_t boundaries = 1;  // a sampled send's timestamps: its start, then one per step it has
        for (const auto& [name, ns] : parts) {
            fields += fmt::format(" {}={:.1f}", name, ns);
            sum += ns;
        }
        for (uint32_t step = HS_CREDIT; step <= HS_POST; step++) {
            boundaries += st[step] > 0;
        }
        const double period = hot_detail_period();
        log_info(
            tt::LogTest,
            "PERF_HOT test={} snd cores={} period={:.0f} samples={} calls/pkt={:.3f} cal={:.1f}cyc [ns/pkt] "
            "loop={:.1f}{} "
            "sum={:.1f} timing={:.1f}",
            test_name,
            hot_.cores,
            period,
            st[HS_SAMPLES],
            hot_.packets ? static_cast<double>(st[HS_CALLS]) / static_cast<double>(hot_.packets) : 0.0,
            cal,
            hot_.packets ? static_cast<double>(hot_.loop) / static_cast<double>(hot_.packets) * ns_per_cycle : 0.0,
            fields,
            sum,
            period > 0 ? boundaries * cal * ns_per_cycle / period : 0.0);
    }

    std::map<std::string, double> test_phases_;
    std::map<std::string, Stat> run_phases_;
    std::vector<std::string> phase_order_;
    Stat zero_writes_, arg_writes_;
    uint64_t zero_bytes_ = 0, arg_bytes_ = 0;
    uint64_t kernels_created_ = 0;
    std::mutex mutex_;
    Stat device_build_;
    std::map<std::string, std::map<const void*, std::set<size_t>>> kernel_signatures_;
    std::map<std::string, uint64_t> kernel_counts_;

    std::string kernel_signature_summary() const {
        std::string out;
        uint64_t distinct_all = 0, total_all = 0, max_device_distinct = 0;
        std::map<const void*, uint64_t> per_device;
        for (const auto& [kind, devices] : kernel_signatures_) {
            uint64_t distinct = 0;
            for (const auto& [device, sigs] : devices) {
                distinct += sigs.size();
                per_device[device] += sigs.size();
            }
            const uint64_t total = kernel_counts_.at(kind);
            out += fmt::format("{}{}={}/{}", out.empty() ? "" : ",", kind, distinct, total);
            distinct_all += distinct;
            total_all += total;
        }
        for (const auto& [device, n] : per_device) {
            max_device_distinct = std::max(max_device_distinct, n);
        }
        return fmt::format("{} all={}/{} max_per_device={}", out, distinct_all, total_all, max_device_distinct);
    }
    uint64_t rt_arg_words_ = 0, workload_programs_ = 0, workload_kernels_ = 0;
    uint32_t muxes_ = 0, max_mux_full_ = 0, max_mux_header_ = 0, max_mux_total_ = 0;
    CoreStats sender_, receiver_;
    SchedStats sched_;
    HotDetailStats hot_;
    uint32_t aiclk_mhz_ = 0;
    uint64_t num_tests_ = 0;
    double run_zero_writes_ms_ = 0, run_arg_writes_ms_ = 0;
    uint64_t run_zero_writes_ = 0, run_arg_writes_ = 0;
};

// Times a scope as a named phase when the breakdown is enabled.
class ScopedPhase {
public:
    explicit ScopedPhase(const char* name) : name_(name), t0_(Clock::now()) {}
    ~ScopedPhase() {
        if (enabled()) {
            Collector::get().add_phase(name_, ms_since(t0_));
        }
    }
    ScopedPhase(const ScopedPhase&) = delete;
    ScopedPhase& operator=(const ScopedPhase&) = delete;

private:
    const char* name_;
    Clock::time_point t0_;
};

// Times a scope as a nested "sub_*" phase, only with TT_FABRIC_TEST_PERF_HOST_DETAIL.
class ScopedDetail {
public:
    explicit ScopedDetail(const char* name) : name_(host_detail() ? name : nullptr) {
        if (name_ != nullptr) {
            t0_ = Clock::now();
        }
    }
    ~ScopedDetail() {
        if (name_ != nullptr) {
            Collector::get().add_phase(name_, ms_since(t0_));
        }
    }
    ScopedDetail(const ScopedDetail&) = delete;
    ScopedDetail& operator=(const ScopedDetail&) = delete;

private:
    const char* name_;
    Clock::time_point t0_;
};

}  // namespace tt::tt_fabric::fabric_tests::perf_breakdown
