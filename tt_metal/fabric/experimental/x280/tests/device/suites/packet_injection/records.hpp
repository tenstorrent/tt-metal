// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// suites/packet_injection/records.hpp -- the packet injection records' shared
// each member's run identity and worker fields, the X280 aggregate record,
// and the RecordSpec each run and worker record is built on.

#ifndef X280_SUITES_PACKET_INJECTION_RECORDS_HPP
#define X280_SUITES_PACKET_INJECTION_RECORDS_HPP

#include "harness/measurement/measurement.hpp"
#include "harness/output/output.hpp"
#include "tests/device/suites/packet_injection/regions.hpp"

#include "kernels/packet_injection/packet_injection_abi.h"
#include "kernels/packet_injection/packet_injection_slot.hpp"
#include "x280/device/x280_state.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace x280::suites::packet_injection {

// The per-worker LIM slots as read back on a failure path. `produced` is each
// producer's own publication and the consumer slot's `consumed` / `credited` are
// X280's view of it: when a run fails with the two disagreeing, the value the
// LIM word actually holds is what says whether the publication reached LIM at
// all, so nothing here is interpreted or derived.
//
// `error` names why the slots could not be read and is empty otherwise. Only
// ok() promises the three vectors hold one entry per worker: a default
// constructed dump is not ok() either, because it was never read at all.
struct LimSlotDump {
    std::vector<uint32_t> produced;
    std::vector<uint32_t> ready;
    std::vector<pi_consumer_slot_t> slots;
    std::string error;

    bool ok() const { return error.empty() && !produced.empty(); }
};

/**
 * One run as its records report it. `successful` is the run's own outcome,
 * `report_successful` the one its run record states after the host's checks;
 * `cfg`, `layout` and `result` must outlive the record calls.
 */
struct Run {
    uint64_t run_value = 0;
    const x280::harness::regions::packet_injection::Config& cfg;
    const pi_layout_t& layout;
    uint32_t tile_count = 0;
    uint32_t worker_count = 0;
    uint64_t mtime_hz = 0;
    const pi_run_result_t& result;
    bool successful = false;
    bool report_successful = false;
};

namespace detail {

// The identity head both members' records open with: the run value, the
// configuration, the counts, the mtime rate and the run's outcome.
// The campaign validator reads run_value, the knobs, worker_count and
// successful from it.
inline std::vector<x280::harness::output::Field> run_identity(
    const Run& run, const char* family, pi_error_t expected_worker_failure) {
    const auto& cfg = run.cfg;
    const auto& r = run.result;
    return {
        {"run_value", run.run_value},
        {"mode", cfg.mode},
        {"include_creation", cfg.include_creation},
        {"family", family},
        {"tile_count", run.tile_count},
        {"worker_count", run.worker_count},
        {"S", cfg.S},
        {"B", cfg.B},
        {"C", cfg.C},
        {"K", cfg.K},
        {"inflight", cfg.inflight},
        {"W", cfg.W},
        {"N", cfg.N},
        {"mtime_hz", run.mtime_hz},
        {"successful", run.report_successful},
        {"expected_worker_failure", static_cast<uint32_t>(expected_worker_failure)},
        {"failure_code", r.failure.code},
        {"failure_phase", r.failure.phase},
        {"failure_actor", r.failure.actor},
        {"failure_worker", r.failure.worker_index},
        {"x280_state", r.state},
        {"data_quiescent", r.data_quiescent},
        {"windows_cleared", r.windows_cleared},
        {"cache_restored", r.cache_restored},
        {"measured_completed", r.measured_completed},
        {"export_valid_bytes", r.export_valid_bytes},
        {"released_group_dests", r.released_group_dests},
        {"released_endpoints", r.released_endpoints}};
}

// The prepared L3 shape every run opens with: WayEnable (Caches.md,
// CCACHE0_WAYENABLE) and the LIM ways left.
inline void append_prepared_shape(std::vector<x280::harness::output::Field>& fields) {
    fields.insert(fields.end(), {{"way_enable", X280_PINNED_INPUT_WAYENABLE}, {"lim_ways", x280::kPreparedLimWays}});
}

inline x280::harness::output::Value coordinate_list(const std::vector<std::array<uint32_t, 2>>& coords) {
    namespace hout = x280::harness::output;
    std::vector<hout::Value> list;
    for (const auto& c : coords) {
        list.push_back(hout::Value::array({c[0], c[1]}));
    }
    return hout::Value::array(std::move(list));
}

}  // namespace detail

// The tensix_to_x280 run's own identity inputs.
struct TensixRunState {
    pi_error_t expected_worker_failure = PI_ERROR_NONE;
    size_t release_guards = 0;
    size_t release_guards_overwritten = 0;
    uint32_t assumption_mask = 0;
    std::string bridge_history;
};

/**
 * The tensix_to_x280 run identity every record of the run carries: the shared
 * head, the release guards, the measurement assumptions, the arena layout,
 * the prepared shape and the ordered producer tiles.
 */
inline std::vector<x280::harness::output::Field> tensix_to_x280_identity(const Run& run, const TensixRunState& state) {
    const auto& cfg = run.cfg;
    const auto& L = run.layout;
    auto fields = detail::run_identity(
        run,
        cfg.family == x280::harness::regions::packet_injection::WorkerFamily::Paired ? "paired" : "brisc",
        state.expected_worker_failure);
    std::vector<std::array<uint32_t, 2>> tiles;
    for (const auto& tile : cfg.ordered_tiles) {
        tiles.push_back({tile.x, tile.y});
    }
    fields.insert(
        fields.end(),
        {
            {"release_guards", state.release_guards},
            {"release_guards_overwritten", state.release_guards_overwritten},
            {"mtime_provenance", "nominal_50mhz"},
            {"assumption_mask", state.assumption_mask},
            {"bridge_history", state.bridge_history},
            {"creation_boundary",
             cfg.mode == PI_MODE_THROUGHPUT ? "not_applicable"
             : cfg.include_creation         ? "before_creation"
                                            : "after_creation"},
            {"service_order", "tile_prefix_brisc_then_ncrisc"},
            {"D", L.staging_depth},
            {"dst_chip", cfg.dst_chip},
            {"dst_mesh", cfg.dst_mesh},
            {"layout_id", PI_LIM_LAYOUT},
            {"arena_bytes", L.arena_bytes},
            {"sync_arrivals_offset", L.sync_arrivals_offset},
            {"l1_tile_base", L.l1_tile_base},
            {"l1_release_offset", L.l1_release_offset},
            {"l1_worker_base_offset", L.l1_worker_base_offset},
            {"l1_worker_stride", L.l1_worker_stride},
            {"l1_tile_bytes", L.l1_tile_bytes},
            {"latency_sample_stride", L.latency_sample_stride},
            {"export_bytes", L.latency_export_bytes},
        });
    detail::append_prepared_shape(fields);
    fields.push_back({"ordered_tiles", detail::coordinate_list(tiles)});
    return fields;
}

// The erisc_to_x280_header run's own identity inputs. `writers_requested` is
// 0 for "all".
struct EriscRunState {
    uint32_t writers_requested = 0;
    uint32_t inactive_eth_available = 0;
    uint64_t erisc_timeout_budget_hz = 0;
    uint64_t resident_slots_checked = 0;
    uint64_t resident_slot_mismatches = 0;
    uint64_t guard_bytes_checked = 0;
    uint64_t guard_mismatches = 0;
    std::vector<std::array<uint32_t, 2>> writers;  // logical coordinates
};

/**
 * The erisc_to_x280_header run identity every record of the run carries: the
 * shared head, the writer request and availability, the ERISC timeout budget,
 * the arena layout, the resident and guard checks, the prepared shape and the
 * writers.
 */
inline std::vector<x280::harness::output::Field> erisc_to_x280_header_identity(
    const Run& run, const EriscRunState& state) {
    namespace hout = x280::harness::output;
    const auto& L = run.layout;
    auto fields = detail::run_identity(run, "ethernet", PI_ERROR_NONE);
    fields.insert(
        fields.end(),
        {
            {"writers_requested",
             state.writers_requested == 0u ? hout::Value("all") : hout::Value(state.writers_requested)},
            {"inactive_eth_available", state.inactive_eth_available},
            {"erisc_timeout_budget_hz", state.erisc_timeout_budget_hz},
            {"erisc_opt_level", "Os"},
            {"service_order", "sorted_writer_prefix"},
            {"D", L.staging_depth},
            {"layout_id", PI_LIM_LAYOUT},
            {"arena_bytes", L.arena_bytes},
            {"l1_tile_base", L.l1_tile_base},
            {"l1_worker_stride", L.l1_worker_stride},
            {"l1_tile_bytes", L.l1_tile_bytes},
            {"resident_slots_checked", state.resident_slots_checked},
            {"resident_slot_mismatches", state.resident_slot_mismatches},
            {"guard_bytes_checked", state.guard_bytes_checked},
            {"guard_mismatches", state.guard_mismatches},
        });
    detail::append_prepared_shape(fields);
    fields.push_back({"writers", detail::coordinate_list(state.writers)});
    return fields;
}

namespace detail {

// Worker `w`'s progress fields: the consumer slot, the LIM words (null unless
// `lim_dump.ok()`), and the producer result's counts, phase and failure code.
// `sample_count` is the worker's latency sample prefix, zero outside
// PI_MODE_LATENCY. Throws std::runtime_error if `lim_dump.ok()` but holds no
// entry for `w`.
inline void append_worker_progress(
    std::vector<x280::harness::output::Field>& fields,
    uint32_t w,
    const pi_consumer_slot_t& slot,
    const LimSlotDump& lim_dump,
    const pi_producer_result_t& result,
    uint64_t sample_count) {
    namespace hout = x280::harness::output;
    if (lim_dump.ok() && (w >= lim_dump.produced.size() || w >= lim_dump.ready.size())) {
        throw std::runtime_error(
            "LIM slot dump holds " + std::to_string(lim_dump.produced.size()) + " produced and " +
            std::to_string(lim_dump.ready.size()) + " ready words, none for worker " + std::to_string(w));
    }
    fields.insert(
        fields.end(),
        {
            {"consumer_completed", slot.phase_done},
            {"consumer_warmup_done", slot.warmup_done},
            {"consumer_consumed", slot.consumed},
            {"consumer_credited", slot.credited},
            {"lim_produced", lim_dump.ok() ? hout::Value(lim_dump.produced[w]) : hout::Value()},
            {"lim_ready", lim_dump.ok() ? hout::Value(lim_dump.ready[w]) : hout::Value()},
            {"producer_completed", result.measured_completed},
            {"producer_latency_count", result.latency_count},
            {"latency_sample_prefix", sample_count},
            {"producer_warmup_completed", result.warmup_completed},
            {"producer_reserved_progress", result.reserved_progress},
            {"producer_published_progress", result.published_progress},
            {"producer_consumed_progress", result.consumed_progress},
            {"producer_phase", result.phase},
            {"producer_failure_code", result.failure.code},
        });
}

// A throughput worker's raw CPU endpoints, its throughput interval over N
// measured headers, and its measured credit-stall episode count.
inline void append_throughput_worker(
    std::vector<x280::harness::output::Field>& fields,
    const pi_producer_result_t& result,
    uint64_t N,
    bool successful) {
    namespace hout = x280::harness::output;
    fields.insert(
        fields.end(),
        {
            {"cpu_start", result.cpu_start},
            {"cpu_end", result.cpu_end},
            {"throughput",
             hout::Value::throughput({
                 .start_ticks = result.throughput_start,
                 .end_ticks = result.throughput_end,
                 .requests = N,
                 .completed = successful,
                 .useful_bytes_per_request = sizeof(fs_hybrid_mesh_packet_header_t),
                 .slot_bytes_per_request = PI_SLOT_BYTES,
             })},
            {"credit_stall_episodes", result.stall_count},
        });
}

}  // namespace detail

// One worker as its record reports it; the references must outlive the call.
struct Worker {
    uint32_t index = 0;
    const pi_consumer_slot_t& slot;
    const LimSlotDump& lim_dump;
    const pi_producer_result_t& producer;
};

// A tensix_to_x280 worker's raw sample populations, by mode: latency samples,
// or the backpressure stall and stall-triggered RTT ticks and cycles.
struct TensixWorkerSamples {
    std::vector<uint64_t> latency;
    std::vector<uint64_t> credit_stall_cpu_cycles;
    std::vector<uint64_t> stall_triggered_rtt_cpu_cycles;
    std::vector<uint64_t> credit_stalls;
    std::vector<uint64_t> stall_triggered_rtt;
};

/**
 * A tensix_to_x280 worker record's own fields: placement (`virtual_xy` from
 * the host's selection), progress, the mode's raw samples or throughput
 * interval, and its counter writes. Throws as append_worker_progress() does.
 */
inline std::vector<x280::harness::output::Field> tensix_to_x280_worker(
    const Run& run, const Worker& worker, std::array<uint32_t, 2> virtual_xy, TensixWorkerSamples samples) {
    namespace hout = x280::harness::output;
    const auto& L = run.layout;
    const uint32_t w = worker.index;
    const uint32_t tile_index = pi_worker_tile_index(&L, w);
    const uint32_t risc = pi_worker_risc_index(&L, w);
    const auto& tile = run.cfg.ordered_tiles.at(tile_index);
    const uint64_t sample_count = samples.latency.size();
    std::vector<hout::Field> fields{
        {"worker_index", w},
        {"tile_index", tile_index},
        {"risc", risc},
        {"assigned_noc", risc},
        {"logical_x", tile.x},
        {"logical_y", tile.y},
        {"virtual_x", virtual_xy[0]},
        {"virtual_y", virtual_xy[1]},
        {"l1_base", pi_worker_l1_base(&L, w)},
    };
    detail::append_worker_progress(fields, w, worker.slot, worker.lim_dump, worker.producer, sample_count);
    if (run.cfg.mode == PI_MODE_LATENCY) {
        fields.insert(
            fields.end(),
            {
                {"cpu_cycles", hout::Value()},
                {"cpu_availability_error", "endpoints_observe_different_cpu_counters"},
                {"latency", hout::Value::population(std::move(samples.latency))},
            });
    } else if (run.cfg.mode == PI_MODE_BACKPRESSURE) {
        fields.insert(
            fields.end(),
            {
                {"credit_stall_cpu_cycles", hout::Value::population(std::move(samples.credit_stall_cpu_cycles))},
                {"stall_triggered_rtt_cpu_cycles",
                 hout::Value::population(std::move(samples.stall_triggered_rtt_cpu_cycles))},
                {"credit_stalls", hout::Value::population(std::move(samples.credit_stalls))},
                {"stall_triggered_rtt", hout::Value::population(std::move(samples.stall_triggered_rtt))},
            });
    } else if (run.cfg.mode == PI_MODE_THROUGHPUT) {
        detail::append_throughput_worker(fields, worker.producer, run.cfg.N, run.successful);
    }
    fields.push_back({"producer_counter_writes", worker.producer.measured_counter_writes});
    return fields;
}

// An ERISC writer's coordinates in each system its record names.
struct WriterPlacement {
    std::array<uint32_t, 2> logical{};
    std::array<uint32_t, 2> virt{};
    std::array<uint32_t, 2> translated{};
    std::array<uint32_t, 2> noc0{};
};

/**
 * An erisc_to_x280_header worker record's own fields: the writer's
 * coordinates, progress, its throughput interval and counter writes. Throws as
 * append_worker_progress() does.
 */
inline std::vector<x280::harness::output::Field> erisc_to_x280_header_worker(
    const Run& run, const Worker& worker, const WriterPlacement& at) {
    namespace hout = x280::harness::output;
    const uint32_t w = worker.index;
    std::vector<hout::Field> fields{
        {"worker_index", w},
        {"tile_index", w},
        {"logical_x", at.logical[0]},
        {"logical_y", at.logical[1]},
        {"virtual_x", at.virt[0]},
        {"virtual_y", at.virt[1]},
        {"translated_x", at.translated[0]},
        {"translated_y", at.translated[1]},
        {"noc0_x", at.noc0[0]},
        {"noc0_y", at.noc0[1]},
        {"l1_base", pi_worker_l1_base(&run.layout, w)},
    };
    detail::append_worker_progress(fields, w, worker.slot, worker.lim_dump, worker.producer, 0);
    detail::append_throughput_worker(fields, worker.producer, run.cfg.N, run.successful);
    fields.push_back({"producer_counter_writes", worker.producer.measured_counter_writes});
    return fields;
}

/**
 * A run or worker record of `member`: the run's `identity`, then
 * `record` ("run" or "worker") and the record's own `fields`, with the run's
 * AICLK evidence and the warm-up selection. A worker record names its
 * producer's wall counter.
 */
inline x280::harness::output::RecordSpec run_record(
    const char* member,
    std::vector<x280::harness::output::Field> identity,
    const char* record,
    std::vector<x280::harness::output::Field> fields,
    std::optional<x280::harness::measurement::Counter> wall_counter,
    const x280::harness::measurement::AiclkRunEvidence& clock,
    const x280::harness::measurement::WarmupSelection& warmup) {
    identity.push_back({"record", record});
    identity.insert(identity.end(), std::make_move_iterator(fields.begin()), std::make_move_iterator(fields.end()));
    return {
        .suite = "packet_injection",
        .member = member,
        .fields = std::move(identity),
        .wall_counter = wall_counter,
        .warmup = warmup,
        .clock = clock};
}

/**
 * X280's measured-phase record of `member`: the run's `identity`, then
 * `"record":"aggregate"`, `"start_event":"release_issued"` (the start
 * endpoint is stamped once the posted release is issued, not delivered), the
 * raw mtime and rdcycle endpoints of the P*N measured headers' interval,
 * `empty_visits` and `aggregate_error`, on the fixed 50 MHz `x280_mtime` clock.
 * `aggregate_error` names why the interval is unusable -- the run failed, or
 * the endpoints are invalid -- and is null otherwise. The tabulator derives the
 * interval and the P*N headers and bytes it covers.
 */
inline x280::harness::output::RecordSpec aggregate_record(
    const char* member,
    std::vector<x280::harness::output::Field> identity,
    const Run& run,
    const x280::harness::measurement::WarmupSelection& warmup) {
    namespace hout = x280::harness::output;
    using x280::harness::measurement::Counter;
    const auto& r = run.result;
    const char* error = !run.successful                                                          ? "run_failed"
                        : !x280::harness::regions::packet_injection::aggregate_interval_valid(r) ? "invalid_interval"
                                                                                                 : nullptr;
    identity.insert(
        identity.end(),
        {
            {"record", "aggregate"},
            {"start_event", "release_issued"},
            {"aggregate_start_mtime", r.aggregate_start_mtime},
            {"aggregate_end_mtime", r.aggregate_end_mtime},
            {"aggregate_start_rdcycle", r.aggregate_start_rdcycle},
            {"aggregate_end_rdcycle", r.aggregate_end_rdcycle},
            {"empty_visits", r.empty_visits},
            {"aggregate_error", error ? hout::Value(error) : hout::Value()},
        });
    return {
        .suite = "packet_injection",
        .member = member,
        .fields = std::move(identity),
        .wall_counter = Counter::X280Mtime,
        .cpu_counter = Counter::X280Rdcycle,
        .warmup = warmup,
        .clock = hout::kX280MtimeClock};
}

}  // namespace x280::suites::packet_injection

#endif  // X280_SUITES_PACKET_INJECTION_RECORDS_HPP
