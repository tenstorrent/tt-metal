// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef X280_SUITES_P2P_LATENCY_METAL_PEER_HPP
#define X280_SUITES_P2P_LATENCY_METAL_PEER_HPP

#include "harness/artifacts/artifacts.hpp"
#include "harness/measurement/measurement.hpp"
#include "harness/output/output.hpp"
#include "harness/regions/regions.hpp"
#include "harness/regions/l1_window.hpp"
#include "x280/device/x280_noc_window.hpp"
#include "p2p_latency_x280_noc_args.h"
#include "p2p_latency_frontend_cxx.hpp"
#include <gtest/gtest.h>
#include <chrono>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace x280::suites::p2p_latency::metal_peer {

namespace hreg = x280::harness::regions;
namespace hout = x280::harness::output;

// Returns a stable report label, or '?' for an unknown wire value.
inline const char* direction_name(uint32_t direction) {
    switch (direction) {
        case P2P_LATENCY_DIRECTION_A_TO_B: return "A->B";
        case P2P_LATENCY_DIRECTION_B_TO_A: return "B->A";
        default: return "?";
    }
}

// Returns a stable report label, or '?' for an unknown wire value.
inline const char* role_name(uint32_t role) {
    switch (role) {
        case P2P_LATENCY_ROLE_INITIATOR: return "initiator";
        case P2P_LATENCY_ROLE_RESPONDER: return "responder";
        default: return "?";
    }
}

// Returns a stable report label, or '?' for an unknown wire value.
inline const char* state_name(uint32_t state) {
    switch (state) {
        case P2P_LATENCY_STATE_INIT: return "INIT";
        case P2P_LATENCY_STATE_RENDEZVOUS: return "RENDEZVOUS";
        case P2P_LATENCY_STATE_WARMUP: return "WARMUP";
        case P2P_LATENCY_STATE_TIMED: return "TIMED";
        case P2P_LATENCY_STATE_DONE: return "DONE";
        case P2P_LATENCY_STATE_TIMEOUT: return "TIMEOUT";
        case P2P_LATENCY_STATE_TRAP: return "TRAP";
        case P2P_LATENCY_STATE_ARG_LAYOUT_MISMATCH: return "ARG_LAYOUT_MISMATCH";
        default: return "?";
    }
}

// Returns a stable report label, or '?' for an unknown wire value.
inline const char* tx_mode_name(uint32_t mode) {
    switch (mode) {
        case P2P_LATENCY_TRANSMIT_FIRE_AND_FORGET: return "fire_and_forget";
        case P2P_LATENCY_TRANSMIT_WAIT_FOR_DEPARTURE: return "wait_for_departure";
        case P2P_LATENCY_TRANSMIT_WAIT_FOR_COMPLETION: return "wait_for_completion";
        default: return "?";
    }
}

// The Metal peer's transmit mode, named by P2P_LATENCY_METAL_PEER_TX_MODE as its
// record label; unset or empty is wait_for_completion. Throws std::runtime_error
// for any value other than wait_for_completion or wait_for_departure.
inline p2p_latency_transmit_mode_t metal_peer_tx_mode() {
    const std::string value = x280::harness::artifacts::env_str("P2P_LATENCY_METAL_PEER_TX_MODE");
    if (value.empty() || value == tx_mode_name(P2P_LATENCY_TRANSMIT_WAIT_FOR_COMPLETION)) {
        return P2P_LATENCY_TRANSMIT_WAIT_FOR_COMPLETION;
    }
    if (value == tx_mode_name(P2P_LATENCY_TRANSMIT_WAIT_FOR_DEPARTURE)) {
        return P2P_LATENCY_TRANSMIT_WAIT_FOR_DEPARTURE;
    }
    throw std::runtime_error(
        "P2P_LATENCY_METAL_PEER_TX_MODE must be 'wait_for_completion' or 'wait_for_departure', got: " + value);
}

// The Metal peer kernel's define selecting `mode`, one metal_peer_tx_mode() returns.
inline std::pair<std::string, std::string> tx_mode_define(p2p_latency_transmit_mode_t mode) {
    return {
        "P2P_LATENCY_METAL_PEER_TX_MODE",
        mode == P2P_LATENCY_TRANSMIT_WAIT_FOR_DEPARTURE ? "P2P_LATENCY_TRANSMIT_WAIT_FOR_DEPARTURE"
                                                        : "P2P_LATENCY_TRANSMIT_WAIT_FOR_COMPLETION"};
}

struct X280Regions {
    hreg::Region args;
    hreg::Region mailbox;
    hreg::Region result;

    // Places args, inbound mailbox, and result in the tile's available LIM.
    static X280Regions compute(x280::X280Tile& tile) {
        const auto available = tile.available_lim();
        hreg::Arena arena(hreg::Region{available.begin, available.size()});
        X280Regions regions{};
        regions.args = arena.allocate(sizeof(p2p_latency_x280_noc_section_t));
        regions.mailbox = arena.allocate(P2P_LATENCY_MAILBOX_BYTES);
        regions.result = arena.allocate(sizeof(p2p_latency_result_t));
        return regions;
    }
};

struct MetalRegions {
    hreg::Region mailbox;
    hreg::Region staging;
    hreg::Region result;

    // Places cache-line-rounded local regions inside the reported L1 window.
    static MetalRegions compute(const hreg::L1Window& window) {
        hreg::Arena arena(hreg::Region{window.local.base, window.local.size});
        MetalRegions regions{};
        regions.mailbox = arena.allocate(P2P_LATENCY_MAILBOX_BYTES);
        regions.staging = arena.allocate(P2P_LATENCY_STAGING_BYTES);
        regions.result = arena.allocate(sizeof(p2p_latency_result_t));
        return regions;
    }
};

// Builds a participant's checked-layout section from an already validated placement.
inline p2p_latency_x280_noc_section_t make_x280_args(
    const p2p_latency_common_args_t& common, const p2p_latency_x280_noc_backend_args_t& backend) {
    return {
        .cookie_common = p2p_latency_cookie_for(UBENCH_SECTION_COMMON),
        .common = common,
        .cookie_backend = p2p_latency_cookie_for(UBENCH_SECTION_BACKEND),
        .backend = backend,
    };
}

// Checks both phases against the requested participant and sample contract.
inline void validate_phases(
    const p2p_latency_result_t& result,
    p2p_latency_participant_t participant,
    p2p_latency_direction_t first_direction,
    uint32_t samples) {
    ASSERT_LE(samples, P2P_LATENCY_MAX_SAMPLES) << "sample count exceeds the fixed result capacity";
    for (uint32_t phase = 0; phase < 2; ++phase) {
        const auto direction = phase == 0 ? first_direction : ::p2p_latency::opposite(first_direction);
        const auto& record = result.phases[phase];
        EXPECT_EQ(record.phase_ordinal, phase);
        EXPECT_EQ(record.direction, static_cast<uint32_t>(direction));
        EXPECT_EQ(record.role, static_cast<uint32_t>(::p2p_latency::role_for(participant, direction)));
        EXPECT_EQ(
            record.sample_kind,
            record.role == P2P_LATENCY_ROLE_INITIATOR ? P2P_LATENCY_SAMPLE_KIND_RTT : P2P_LATENCY_SAMPLE_KIND_TURN);
        ASSERT_EQ(record.samples_completed, samples);
        EXPECT_EQ(record.terminal_state, static_cast<uint32_t>(P2P_LATENCY_STATE_DONE));
        for (uint32_t i = 0; i < samples; ++i) {
            EXPECT_GT(record.raw_deltas[i], 0u);
        }
    }
}

// Host wait for one X280 session of a paired run.
constexpr std::chrono::seconds kX280WaitTimeout{60};

// Builds the warm-up calibration for a paired run from a completed pilot. One
// warm-up iteration is one exchange in each phase, so a pilot iteration's
// device time is the two phases' initiator RTTs: X280 mtime at its fixed rate,
// the peer's wall timer at peer_wall_hz. max_warmup keeps the run's tokens,
// run_nonce + 2*(W+N) - 1, inside u32; max_device_ns is kX280WaitTimeout. A
// pilot record with other than `samples` samples, or a zero peer_wall_hz, throws.
inline x280::harness::measurement::WarmupCalibrationRequest warmup_calibration_request(
    const p2p_latency_result_t& x280_pilot,
    const p2p_latency_result_t& peer_pilot,
    uint64_t peer_wall_hz,
    uint32_t default_warmup,
    uint32_t samples,
    uint32_t run_nonce) {
    namespace hmeas = x280::harness::measurement;
    if (peer_wall_hz == 0) {
        throw std::invalid_argument("pilot peer wall rate is zero");
    }
    long double device_ns = 0;
    for (uint32_t phase = 0; phase < 2; ++phase) {
        const bool x280_initiates = x280_pilot.phases[phase].role == P2P_LATENCY_ROLE_INITIATOR;
        const auto& record = x280_initiates ? x280_pilot.phases[phase] : peer_pilot.phases[phase];
        if (record.samples_completed != samples) {
            throw std::runtime_error(
                "pilot phase " + std::to_string(phase) + " completed " + std::to_string(record.samples_completed) +
                " of " + std::to_string(samples) + " samples");
        }
        const uint32_t* deltas = x280_initiates ? record.secondary_deltas : record.raw_deltas;
        uint64_t ticks = 0;
        for (uint32_t i = 0; i < samples; ++i) {
            ticks += deltas[i];
        }
        device_ns += ticks * 1.0e9L / (x280_initiates ? hmeas::kX280MtimeHz : peer_wall_hz);
    }
    const uint64_t token_warmup = (static_cast<uint64_t>(UINT32_MAX) + 1u - run_nonce) / 2u;
    return {
        .pilot_device_ns = static_cast<uint64_t>(std::llround(device_ns)),
        .pilot_iterations = samples,
        .default_warmup = default_warmup,
        .timed_iterations = samples,
        .target_device_ns = hmeas::kAiclkTargetDeviceWorkNs,
        .max_warmup = token_warmup > samples ? token_warmup - samples : 0u,
        .max_device_ns = static_cast<uint64_t>(std::chrono::nanoseconds(kX280WaitTimeout).count()),
    };
}

// Throws std::runtime_error unless the record's sample count is in
// [1, P2P_LATENCY_MAX_SAMPLES]; a corrupt count fails before any delta is read.
inline void validate_phase_record(const p2p_latency_phase_record_t& record) {
    if (record.samples_completed == 0 || record.samples_completed > P2P_LATENCY_MAX_SAMPLES) {
        throw std::runtime_error("Metal peer sample count out of range: " + std::to_string(record.samples_completed));
    }
}

/**
 * Passes when every one of the phase's samples_completed raw samples, in both
 * counters, is nonzero: a zero delta is a bad read, not a round trip. No upper
 * bound is checked. A device hang guard's expiry ends the phase TIMEOUT with
 * no sample stored, and a delta above UINT32_MAX traps, so no stored sample
 * can exceed either; the terminal-state checks catch both. The failure names
 * the counter and the first zero sample's index.
 */
inline ::testing::AssertionResult phase_samples_nonzero(const p2p_latency_phase_record_t& ph) {
    for (const auto& [name, deltas] : {std::pair{"raw", ph.raw_deltas}, std::pair{"secondary", ph.secondary_deltas}}) {
        for (uint32_t i = 0; i < ph.samples_completed; ++i) {
            if (deltas[i] == 0) {
                return ::testing::AssertionFailure()
                       << name << " sample " << i << " of " << ph.samples_completed << " is 0";
            }
        }
    }
    return ::testing::AssertionSuccess();
}

namespace detail {

inline std::vector<uint64_t> population(const uint32_t* deltas, uint32_t count) {
    return std::vector<uint64_t>(deltas, deltas + count);
}

inline const char* sample_kind_name(uint32_t kind) { return kind == P2P_LATENCY_SAMPLE_KIND_RTT ? "RTT" : "TURN"; }

}  // namespace detail

// A Metal participant's wall counter and the AICLK evidence its record carries.
struct MetalClock {
    x280::harness::measurement::Counter wall_counter;
    const x280::harness::measurement::AiclkRunEvidence& evidence;
};

/**
 * One participant's record of `phase` in the phase format every P2P member
 * writes: `identity` (the participant and its placement) first, then the
 * phase's protocol fields, the ordering of the window it writes through
 * (`window_mode`, null for a participant that opens none: x280_same_tile and
 * every Metal participant), the pair's X280 window slot as its X280 section
 * names it (`tlb_index`, null for x280_same_tile, which has none), the
 * transmit mode it ran (`tx_mode`), and its raw
 * wall and CPU sample populations. An X280 participant (backend class
 * 1) stores CPU cycles first and is on the fixed mtime clock; a Metal one
 * stores its `metal` wall counter's ticks first, with `metal`'s evidence as its
 * clock. `warmup` is the selection behind the run's warm-up count. Throws
 * std::out_of_range for a phase other than 0 or 1,
 * std::invalid_argument when `metal` is given for an X280 participant or
 * missing for a Metal one, and as validate_phase_record() does.
 */
inline hout::RecordSpec phase_record(
    const char* member,
    std::vector<hout::Field> identity,
    const p2p_latency_result_t& result,
    uint32_t phase,
    const p2p_latency_common_args_t& common,
    std::optional<uint32_t> tlb_index,
    std::optional<noc_ordering_t> window,
    p2p_latency_transmit_mode_t tx_mode,
    std::optional<MetalClock> metal,
    const x280::harness::measurement::WarmupSelection& warmup) {
    using x280::harness::measurement::Counter;
    if (phase >= 2) {
        throw std::out_of_range("Metal peer phase must be 0 or 1: " + std::to_string(phase));
    }
    const auto& record = result.phases[phase];
    validate_phase_record(record);
    const bool is_x280 = UBENCH_BACKEND_ID_CLASS(result.backend_id) == UBENCH_BACKEND_CLASS_X280;
    if (is_x280 == metal.has_value()) {
        throw std::invalid_argument(
            "backend " + std::to_string(result.backend_id) +
            ": a Metal clock is required exactly for a Metal participant");
    }
    const auto primary = detail::population(record.raw_deltas, record.samples_completed);
    const auto secondary = detail::population(record.secondary_deltas, record.samples_completed);
    auto fields = std::move(identity);
    fields.insert(
        fields.end(),
        {
            {"phase", phase},
            {"first_direction", direction_name(common.first_direction)},
            {"direction", direction_name(record.direction)},
            {"role", role_name(record.role)},
            {"sample_kind", detail::sample_kind_name(record.sample_kind)},
            {"samples", record.samples_completed},
            {"warmup", common.warmup_count},
            {"payload_size", common.payload_size},
            {"same_chip", true},
            {"window_mode", window ? hout::Value(x280::noc_ordering_name(*window)) : hout::Value()},
            {"tlb_index", tlb_index ? hout::Value(*tlb_index) : hout::Value()},
            {"tx_mode", tx_mode_name(tx_mode)},
            {"backend_id", result.backend_id},
            {"run_nonce", common.run_nonce},
            {"path_sentinel", result.path_sentinel},
            {"wall", hout::Value::population(is_x280 ? secondary : primary)},
            {"cpu_cycles", hout::Value::population(is_x280 ? primary : secondary)},
        });
    hout::RecordSpec spec{
        .suite = "p2p_latency",
        .member = member,
        .fields = std::move(fields),
        .wall_counter = is_x280 ? Counter::X280Mtime : metal->wall_counter,
        .cpu_counter = is_x280 ? Counter::X280Rdcycle : Counter::Rv32Cycle,
        .warmup = warmup,
    };
    if (is_x280) {
        spec.clock = hout::kX280MtimeClock;
    } else {
        spec.clock = metal->evidence;
    }
    return spec;
}

}  // namespace x280::suites::p2p_latency::metal_peer
#endif
