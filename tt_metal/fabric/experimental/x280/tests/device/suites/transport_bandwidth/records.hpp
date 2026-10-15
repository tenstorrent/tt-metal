// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef X280_SUITES_TRANSPORT_BW_RECORDS_HPP
#define X280_SUITES_TRANSPORT_BW_RECORDS_HPP

#include "harness/measurement/measurement.hpp"
#include "harness/output/output.hpp"

#include "kernels/transport_bandwidth/transport_bandwidth_common.h"

#include <cstdint>
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
 * The x280_to_tensix record of one point: the raw mtime and rdcycle batch over
 * `samples` transfers of `store_width`-byte stores and their logical bytes, its
 * explicit warm-up, on the fixed mtime clock. Throws as record() does for an
 * invalid batch.
 */
inline x280::harness::output::RecordSpec x280_to_tensix_record(
    const transport_bw_result_t& result, const Point& point, uint32_t store_width) {
    using x280::harness::measurement::Counter;
    auto fields = detail::point_fields(point, "RETIRED");
    fields.push_back({"store_width", store_width});
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

}  // namespace x280::suites::transport_bandwidth

#endif  // X280_SUITES_TRANSPORT_BW_RECORDS_HPP
