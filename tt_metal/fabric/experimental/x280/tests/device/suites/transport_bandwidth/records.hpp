// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef X280_SUITES_TRANSPORT_BW_RECORDS_HPP
#define X280_SUITES_TRANSPORT_BW_RECORDS_HPP

#include "harness/measurement/measurement.hpp"
#include "harness/output/output.hpp"

#include "kernels/transport_bandwidth/transport_bandwidth_common.h"
#include "kernels/transport_bandwidth/x280_to_tensix.h"

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace x280::suites::transport_bandwidth {

// The record label of boundary `b`. `drained` names TRANSPORT_BW_BOUNDARY_DRAINED,
// whose stop differs per member (docs/measurement.md "Quantities that share a label").
inline const char* boundary_name(uint32_t b, const char* drained) {
    switch (b) {
        case TRANSPORT_BW_BOUNDARY_DRAINED: return drained;
        case TRANSPORT_BW_BOUNDARY_INJECTION: return "INJECTION";
        default: return "?";
    }
}

// A terminal_state's name; "?" for a value outside transport_bw_state_t.
inline const char* state_name(uint32_t s) {
    switch (static_cast<transport_bw_state_t>(s)) {
        case TRANSPORT_BW_STATE_INIT: return "INIT";
        case TRANSPORT_BW_STATE_VERIFY_PATH: return "VERIFY_PATH";
        case TRANSPORT_BW_STATE_WARMUP: return "WARMUP";
        case TRANSPORT_BW_STATE_TIMED_DRAINED: return "TIMED_DRAINED";
        case TRANSPORT_BW_STATE_TIMED_INJECTION: return "TIMED_INJECTION";
        case TRANSPORT_BW_STATE_DRAIN: return "DRAIN";
        case TRANSPORT_BW_STATE_PUBLISH: return "PUBLISH";
        case TRANSPORT_BW_STATE_DONE: return "DONE";
        case TRANSPORT_BW_STATE_PATH_MISMATCH: return "PATH_MISMATCH";
        case TRANSPORT_BW_STATE_RENDEZVOUS_TIMEOUT: return "RENDEZVOUS_TIMEOUT";
        case TRANSPORT_BW_STATE_ARG_LAYOUT_MISMATCH: return "ARG_LAYOUT_MISMATCH";
        case TRANSPORT_BW_STATE_WINDOW_CLEANUP_FAILED: return "WINDOW_CLEANUP_FAILED";
    }
    return "?";
}

// One measured sweep point: `samples` transfers of `transfer_size` bytes
// after `warmup` unrecorded ones, timed to `boundary`.
struct Point {
    uint32_t transfer_size = 0;
    uint32_t boundary = 0;
    uint32_t warmup = 0;
    uint32_t samples = 0;
};

namespace detail {

inline std::vector<x280::harness::output::Field> point_fields(const Point& point, const char* drained) {
    return {
        {"transfer_size", point.transfer_size},
        {"boundary", boundary_name(point.boundary, drained)},
        {"warmup", point.warmup},
        {"samples", point.samples}};
}

inline x280::harness::output::RecordSpec::Batch batch(const transport_bw_result_t& result, const Point& point) {
    return {
        .wall_ticks = result.raw_batch_delta,
        .cpu_cycles = result.secondary_batch_delta,
        .items = point.samples,
        .logical_bytes = uint64_t{point.samples} * point.transfer_size};
}

}  // namespace detail

/**
 * The x280_to_tensix record of one point, run by `harts.size()` harts (hart h
 * at index h, at least one): the raw mtime and rdcycle batch of hart 0 over
 * its `samples` transfers of `store_width`-byte stores and their logical
 * bytes, and in `hart_brackets` every hart's raw mtime endpoints and rdcycle
 * delta; its explicit warm-up, on the fixed mtime clock. Every hart issues
 * `samples` transfers. Throws as record() does for an invalid batch, and
 * std::invalid_argument for no harts.
 */
inline x280::harness::output::RecordSpec x280_to_tensix_record(
    const std::vector<transport_bw_x280_to_tensix_result_t>& harts, const Point& point, uint32_t store_width) {
    using x280::harness::measurement::Counter;
    using x280::harness::output::Field;
    using x280::harness::output::Value;
    if (harts.empty()) {
        throw std::invalid_argument("x280_to_tensix record needs at least one hart's result");
    }
    std::vector<Value> brackets;
    for (size_t h = 0; h < harts.size(); ++h) {
        brackets.push_back(Value::object(std::vector<Field>{
            {"hart", h},
            {"wall_start", harts[h].wall_start},
            {"wall_end", harts[h].wall_end},
            {"cpu_cycles", harts[h].common.secondary_batch_delta}}));
    }
    const transport_bw_result_t& result = harts.front().common;
    auto fields = detail::point_fields(point, "RETIRED");
    fields.push_back({"store_width", store_width});
    fields.push_back({"harts", harts.size()});
    fields.push_back({"hart_brackets", Value::array(std::move(brackets))});
    return {
        .suite = "transport_bandwidth",
        .member = "x280_to_tensix",
        .fields = std::move(fields),
        .wall_counter = Counter::X280Mtime,
        .cpu_counter = Counter::X280Rdcycle,
        .batch = detail::batch(result, point),
        .warmup = x280::harness::measurement::WarmupSelection{.warmup = point.warmup},
        .clock = x280::harness::output::kX280MtimeClock,
    };
}

/**
 * The tensix_to_x280 record of one point: the raw live-timer and rv32 cycle
 * batch over `samples` transfers and their logical bytes, how the warm-up was
 * chosen and the AICLK evidence. Throws as record() does for an invalid batch.
 */
inline x280::harness::output::RecordSpec tensix_to_x280_record(
    const transport_bw_result_t& result,
    const Point& point,
    const x280::harness::measurement::WarmupSelection& selection,
    const x280::harness::measurement::AiclkRunEvidence& clock) {
    using x280::harness::measurement::Counter;
    return {
        .suite = "transport_bandwidth",
        .member = "tensix_to_x280",
        .fields = detail::point_fields(point, "DRAINED"),
        .wall_counter = Counter::TensixLiveTimer,
        .cpu_counter = Counter::Rv32Cycle,
        .batch = detail::batch(result, point),
        .warmup = selection,
        .clock = clock,
    };
}

// An idle-ERISC writer's logical Ethernet coordinate.
struct WriterAt {
    uint32_t x = 0;
    uint32_t y = 0;
};

/**
 * The erisc_to_x280 record of one point, run by `results.size()` idle ERISCs
 * (writer w at index w of `results` and `at`, at least one, the same count):
 * writer 0's raw ERISC wall-clock and cycle batch over its `samples` transfers
 * and their logical bytes, and in `writer_batches` every writer's raw batch.
 * Writers' clocks share no stated origin, so only their deltas are recorded.
 * `gate_release_host_ns` is the host time spent writing every writer's start
 * word. AICLK evidence sampled on writer 0 converts every writer's ticks.
 * Throws as record() does for an invalid batch, and std::invalid_argument for
 * no writers or mismatched lengths.
 */
inline x280::harness::output::RecordSpec erisc_to_x280_record(
    const std::vector<transport_bw_result_t>& results,
    const std::vector<WriterAt>& at,
    const Point& point,
    const x280::harness::measurement::WarmupSelection& selection,
    const x280::harness::measurement::AiclkRunEvidence& clock,
    uint64_t gate_release_host_ns) {
    using x280::harness::measurement::Counter;
    using x280::harness::output::Field;
    using x280::harness::output::Value;
    if (results.empty() || results.size() != at.size()) {
        throw std::invalid_argument("erisc_to_x280 record needs one result and one coordinate per writer");
    }
    std::vector<Value> batches;
    for (size_t w = 0; w < results.size(); ++w) {
        batches.push_back(Value::object(std::vector<Field>{
            {"writer", w},
            {"logical_x", at[w].x},
            {"logical_y", at[w].y},
            {"wall_ticks", results[w].raw_batch_delta},
            {"cpu_cycles", results[w].secondary_batch_delta}}));
    }
    auto fields = detail::point_fields(point, "DRAINED");
    fields.push_back({"writers", results.size()});
    fields.push_back({"writer_batches", Value::array(std::move(batches))});
    fields.push_back({"gate_release_host_ns", gate_release_host_ns});
    return {
        .suite = "transport_bandwidth",
        .member = "erisc_to_x280",
        .fields = std::move(fields),
        .wall_counter = Counter::EriscWallClock,
        .cpu_counter = Counter::Rv32Cycle,
        .batch = detail::batch(results.front(), point),
        .warmup = selection,
        .clock = clock,
    };
}

}  // namespace x280::suites::transport_bandwidth

#endif  // X280_SUITES_TRANSPORT_BW_RECORDS_HPP
