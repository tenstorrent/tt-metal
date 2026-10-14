// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef X280_SUITES_PIPELINE_STAGE_RECORDS_HPP
#define X280_SUITES_PIPELINE_STAGE_RECORDS_HPP

#include "harness/measurement/measurement.hpp"
#include "harness/output/output.hpp"
#include "tests/device/suites/pipeline_stage/suite_common.hpp"

#include "pipeline_stage_common.h"
#include "rt/x280_pinned_input.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace x280::suites::pipeline_stage::common {

// Four hops: the maximum on the declared 2x4 geometry.
constexpr uint32_t kHopCount = 4u;

/**
 * One completed launch as its record reports it. `samples` must be non-null
 * when `instrumentation` is per-sample and is ignored otherwise; `result` and
 * `samples` must outlive the record call.
 */
struct Launch {
    const pipeline_stage_result_t& result;
    uint32_t run_nonce = 0u;
    uint32_t warmup = 0u;
    uint32_t timed = 0u;
    uint32_t instrumentation = 0u;
    const pipeline_stage_samples_t* samples = nullptr;
};

namespace detail {

namespace hout = x280::harness::output;

inline std::vector<hout::Field> head_fields(const Launch& launch) {
    return {
        {"kind", "summary"},
        {"backend_id", launch.result.backend_id},
        {"run_nonce", launch.run_nonce},
        {"packet_header_bytes", launch.result.packet_header_bytes},
        {"route_buffer_capacity", FS_ROUTE_BUFFER_CAPACITY},
        {"instrumentation", instrumentation_name(launch.instrumentation)}};
}

inline void append_progress(std::vector<hout::Field>& fields, const Launch& launch) {
    fields.insert(
        fields.end(),
        {{"warmup", launch.warmup},
         {"timed_count", launch.timed},
         {"items_completed", launch.result.items_completed},
         {"stage_witness", launch.result.stage_witness},
         {"raw_batch_delta", launch.result.raw_batch_delta},
         {"secondary_batch_delta", launch.result.secondary_batch_delta},
         {"verify_sentinel", launch.result.verify_sentinel}});
}

inline void append_x280_policy(std::vector<hout::Field>& fields) {
    fields.insert(
        fields.end(),
        {{"wayenable", X280_PINNED_INPUT_WAYENABLE},
         {"pinned_way", X280_PINNED_INPUT_RESERVED_WAY},
         {"waymask_bits", X280_PINNED_INPUT_PREPARED_BITS}});
}

inline void append_forward_identity(std::vector<hout::Field>& fields) {
    fields.insert(
        fields.end(), {{"hop_count", kHopCount}, {"boundary", "decision"}, {"subject", "router forward decision"}});
}

inline hout::RecordSpec spec(
    std::vector<hout::Field> fields,
    const Launch& launch,
    const char* member,
    x280::harness::measurement::Counter wall,
    x280::harness::measurement::Counter cpu) {
    return {
        .suite = "pipeline_stage",
        .member = member,
        .fields = std::move(fields),
        .wall_counter = wall,
        .cpu_counter = cpu,
        .batch =
            hout::RecordSpec::Batch{
                .wall_ticks = launch.result.raw_batch_delta,
                .cpu_cycles = launch.result.secondary_batch_delta,
                .items = launch.timed},
    };
}

}  // namespace detail

/**
 * The X280 route_compute record: identity, the raw batch over `timed` items
 * and the rdcycle samples (empty unless instrumented), its explicit warm-up,
 * on the fixed mtime clock. Throws as validate_measurement() does.
 */
inline x280::harness::output::RecordSpec x280_route_compute_record(const Launch& launch) {
    namespace hout = x280::harness::output;
    const auto samples = validate_measurement(launch.result, launch.samples, launch.timed, launch.instrumentation);
    auto fields = detail::head_fields(launch);
    fields.insert(
        fields.end(),
        {{"fixture", "direction_table_v1"},
         {"header_count", PIPELINE_STAGE_LOOKUP_HEADER_COUNT},
         {"prepared_input_base", X280_PINNED_INPUT_BASE},
         {"prepared_input_bytes", X280_PINNED_INPUT_SPAN}});
    detail::append_x280_policy(fields);
    detail::append_progress(fields, launch);
    fields.insert(
        fields.end(),
        {{"subject", "next-hop direction-table lookup"}, {"cpu_samples", hout::Value::population(samples)}});
    auto spec = detail::spec(
        std::move(fields),
        launch,
        "route_compute",
        x280::harness::measurement::Counter::X280Mtime,
        x280::harness::measurement::Counter::X280Rdcycle);
    spec.warmup = x280::harness::measurement::WarmupSelection{.warmup = launch.warmup};
    spec.clock = hout::kX280MtimeClock;
    return spec;
}

/**
 * The X280 route_forward record: the DECIDE identity, the raw batch and the
 * rdcycle samples, its explicit warm-up, on the fixed mtime clock. Throws
 * std::invalid_argument for an instrumented run, and as validate_measurement()
 * does.
 */
inline x280::harness::output::RecordSpec x280_route_forward_record(const Launch& launch) {
    namespace hout = x280::harness::output;
    if (launch.instrumentation != PIPELINE_STAGE_INSTRUMENTATION_OFF) {
        throw std::invalid_argument("route_forward requires instrumentation to be OFF");
    }
    const auto samples = validate_measurement(launch.result, launch.samples, launch.timed, launch.instrumentation);
    auto fields = detail::head_fields(launch);
    fields.insert(
        fields.end(),
        {{"source_placement", "pinned_zero_device"},
         {"source_base", X280_PINNED_INPUT_FIRST_HEADER_BASE},
         {"source_bytes", X280_PINNED_INPUT_HEADER_BYTES}});
    detail::append_x280_policy(fields);
    detail::append_progress(fields, launch);
    detail::append_forward_identity(fields);
    fields.push_back({"cpu_samples", hout::Value::population(samples)});
    auto spec = detail::spec(
        std::move(fields),
        launch,
        "route_forward",
        x280::harness::measurement::Counter::X280Mtime,
        x280::harness::measurement::Counter::X280Rdcycle);
    spec.warmup = x280::harness::measurement::WarmupSelection{.warmup = launch.warmup};
    spec.clock = hout::kX280MtimeClock;
    return spec;
}

}  // namespace x280::suites::pipeline_stage::common

#endif  // X280_SUITES_PIPELINE_STAGE_RECORDS_HPP
