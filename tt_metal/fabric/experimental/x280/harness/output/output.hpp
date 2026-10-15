// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// harness/output/output.hpp -- the record builder and JSONL writer.
//
// This header centralizes:
//
//   - The record builder, record(), and the record contract it enforces.
//   - JSONL writer keyed by a deterministic file name
//     (<suite>_<member_or_direction>.<axis-values>.jsonl in
//      $TT_METAL_HOME/characterization or a caller override).

#ifndef X280_HARNESS_OUTPUT_HPP
#define X280_HARNESS_OUTPUT_HPP

#include "harness/measurement/measurement.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace x280::harness::output {

// Serialize a raw tick population: its count, the units and the samples in
// recorded order. An empty population carries reason no_samples. A sample at
// or above 2^63 throws std::runtime_error.
inline std::string sample_population_json(const std::vector<uint64_t>& samples) {
    for (auto sample : samples) {
        if (sample >= (1ULL << 63)) {
            throw std::runtime_error("sample exceeds half-range timer bound");
        }
    }
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "{\"count\":" << samples.size() << ",\"units\":\"ticks\",\"raw_ticks\":[";
    for (size_t i = 0; i < samples.size(); ++i) {
        if (i) {
            out << ',';
        }
        out << samples[i];
    }
    out << ']';
    if (samples.empty()) {
        out << ",\"reason\":\"no_samples\"";
    }
    out << '}';
    return out.str();
}

// A measured interval is usable only when it is nonzero and stays inside the
// timer half-range, so a wrapped or unwritten endpoint pair cannot become a
// rate. Callers that must fail a run on a bad interval share this predicate
// with the rate rendering below.
inline bool throughput_interval_valid(uint64_t start, uint64_t end) {
    const uint64_t ticks = end - start;
    return ticks != 0 && ticks < (1ULL << 63);
}

// Preserve raw endpoints and wire geometry. The offline measurement consumer
// derives rates only after validating the enclosing clock evidence.
//
// `useful_bytes` and `slot_bytes` are the payload and wire sizes of one
// request, which scale the rate into useful_header_gb_s and slot_gb_s. The
// caller owns them: this header cannot see any benchmark's wire ABI, so a
// hardcoded size here would keep reporting the old geometry after that ABI
// changed, with nothing to catch it. `requests` is the configured request
// count, rendered when given; a record whose own fields count the requests
// passes none.
inline std::string throughput_json(
    uint64_t start,
    uint64_t end,
    std::optional<uint64_t> requests,
    bool completed,
    uint64_t useful_bytes,
    uint64_t slot_bytes) {
    const char* error = !completed                               ? "run_failed"
                        : !throughput_interval_valid(start, end) ? "invalid_interval"
                        : requests == 0u                         ? "empty_workload"
                                                                 : nullptr;
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "{\"start_ticks\":" << start << ",\"end_ticks\":" << end;
    if (requests) {
        out << ",\"requests\":" << *requests;
    }
    out << ",\"useful_bytes_per_request\":" << useful_bytes << ",\"slot_bytes_per_request\":" << slot_bytes;
    if (error) {
        out << ",\"interval_error\":\"" << error << '"';
    } else {
        out << ",\"interval_error\":null";
    }
    out << '}';
    return out.str();
}

// Serialize self-contained AICLK conversion evidence. A zero conversion_hz is
// rendered as null and every rejection reason remains machine-readable.
inline std::string clock_json(const measurement::AiclkRunEvidence& clock) {
    auto sample_json = [](std::ostringstream& out, const measurement::HostClockSample& sample) {
        out << '[' << sample.host_before_ns << ',' << sample.counter_ticks << ',' << sample.host_after_ns << ']';
    };
    auto telemetry_json = [](std::ostringstream& out, const measurement::CmfwClockSample& sample) {
        out << "{\"host_before_ns\":" << sample.host_before_ns << ",\"host_after_ns\":" << sample.host_after_ns
            << ",\"aiclk_limit_max_mhz\":" << sample.limit_max_mhz << ",\"aiclk_mhz\":" << sample.current_mhz
            << ",\"aiclk_arb_max_raw\":" << sample.arb_max_raw
            << ",\"thermal_trip_count\":" << sample.thermal_trip_count
            << ",\"update_period_ms\":" << sample.update_period_ms << '}';
    };
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(17) << "{\"domain\":\"aiclk\",\"expected_hz\":" << clock.expected_hz
        << ",\"expected_source\":\"cmfw_aiclk_limit_max\""
        << ",\"reference_counter\":";
    if (clock.reference) {
        out << '"' << measurement::counter_name(*clock.reference) << '"';
    } else {
        out << "null";
    }
    out << ",\"reference_core\":";
    if (clock.reference_core) {
        out << '[' << (*clock.reference_core)[0] << ',' << (*clock.reference_core)[1] << ']';
    } else {
        out << "null";
    }
    out << ",\"sample_gap_ns\":" << measurement::kAiclkSampleGapNs << ",\"conversion_hz\":";
    if (clock.conversion_hz == 0) {
        out << "null";
    } else {
        out << clock.conversion_hz;
    }
    auto endpoint_json =
        [&](const char* name, const measurement::AiclkEndpoint& endpoint, const measurement::RateBounds& bounds) {
            out << ",\"" << name << "\":{\"samples\":[";
            sample_json(out, endpoint.first);
            out << ',';
            sample_json(out, endpoint.second);
            out << "],\"rate_lower_hz\":" << bounds.lower_hz << ",\"rate_upper_hz\":" << bounds.upper_hz
                << ",\"telemetry\":";
            telemetry_json(out, endpoint.telemetry);
            out << '}';
        };
    endpoint_json("start", clock.start, clock.start_rate);
    endpoint_json("end", clock.end, clock.end_rate);
    out << ",\"elapsed_rate_lower_hz\":" << clock.elapsed_rate.lower_hz
        << ",\"elapsed_rate_upper_hz\":" << clock.elapsed_rate.upper_hz << ",\"elapsed_window_ns\":";
    if (clock.elapsed_window_ns) {
        out << *clock.elapsed_window_ns;
    } else {
        out << "null";
    }
    out << ",\"rate_tolerance_ppm\":" << measurement::kAiclkRateTolerancePpm
        << ",\"maximum_endpoint_bound_width_ppm\":" << measurement::kAiclkMaximumEndpointBoundWidthPpm
        << ",\"maximum_elapsed_bound_width_ppm\":" << measurement::kAiclkMaximumElapsedBoundWidthPpm << ",\"errors\":[";
    for (size_t i = 0; i < clock.errors.size(); ++i) {
        if (i) {
            out << ',';
        }
        out << '"' << measurement::clock_error_name(clock.errors[i]) << '"';
    }
    out << "]}";
    return out.str();
}

// ---------------------------------------------------------------------------
// Record builder
//
// record() renders a whole JSONL record from typed fields. A record carries raw
// ticks or cycles, counts, byte counts, endpoints and clock evidence only; the
// offline tabulator derives every converted or computed figure (docs/measurement.md
// "Record contract"). The contract is enforced here: no field type holds a
// floating-point number.
// ---------------------------------------------------------------------------

// Why record() refused a key.
enum class RecordKeyFault : uint32_t {
    Duplicate,  // already in the same object, as an envelope key or another field
};

/**
 * Thrown by record() for a key that appears twice in one object, at any
 * depth. key() is the offending key as given.
 */
class RecordKeyError : public std::invalid_argument {
public:
    RecordKeyError(std::string key, RecordKeyFault fault) :
        std::invalid_argument(describe(key, fault)), key_(std::move(key)), fault_(fault) {}
    const std::string& key() const noexcept { return key_; }
    RecordKeyFault fault() const noexcept { return fault_; }

private:
    static std::string describe(const std::string& key, RecordKeyFault fault) {
        switch (fault) {
            case RecordKeyFault::Duplicate: return "record key \"" + key + "\" appears twice in one object";
        }
        return "record key \"" + key + "\" refused for an unknown reason " +
               std::to_string(static_cast<uint32_t>(fault));
    }
    std::string key_;
    RecordKeyFault fault_;
};

// A fixed-rate wall-clock domain: `counter` ticks at exactly `hz`.
struct FixedClock {
    measurement::Counter counter;
    uint64_t hz;
};

// X280 mtime's fixed 50 MHz contract.
inline constexpr FixedClock kX280MtimeClock{measurement::Counter::X280Mtime, measurement::kX280MtimeHz};

// A raw throughput interval between two endpoints of one counter, over the
// configured `requests` when given; `completed` is whether the run finished.
// The byte counts are one request's payload and wire sizes (throughput_json).
struct Throughput {
    uint64_t start_ticks = 0;
    uint64_t end_ticks = 0;
    std::optional<uint64_t> requests;
    bool completed = false;
    uint64_t useful_bytes_per_request = 0;
    uint64_t slot_bytes_per_request = 0;
};

// Raw timestamps with no conversion offered, measurement schema 1: core_sync's
// cross-core timestamps, whose counters share no origin.
struct NoConversion {};

// A measurement with no warm-up phase: core_sync and eth_credit. Rendered as
// `"warmup_source":"none"`.
struct NoWarmup {};

struct Field;

/**
 * One record field's value. Its kinds are the raw ones a record may carry:
 * null, a flag, an unsigned or signed integer, a string, an array or object of
 * values, a raw sample population, a raw throughput interval, and fixed or
 * AICLK clock evidence. There is no floating-point kind and constructing one
 * from a floating-point number does not compile, so a field cannot hold a
 * converted or averaged figure. A default-constructed Value is null.
 */
class Value {
public:
    Value() = default;
    template <typename T, std::enable_if_t<std::is_same_v<T, bool>, int> = 0>
    Value(T flag) : value_(flag) {}
    template <typename T, std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>, int> = 0>
    Value(T number);
    template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
    Value(T) = delete;
    // `text` must be non-null.
    Value(const char* text);
    Value(std::string text);

    static Value array(std::vector<Value> items);
    static Value object(std::vector<Field> fields);
    // A population whose samples are each below 2^63 (sample_population_json).
    static Value population(std::vector<uint64_t> samples);
    static Value throughput(const Throughput& interval);
    // A fixed clock with a nonzero rate; a zero rate throws std::invalid_argument.
    static Value clock(const FixedClock& clock);
    static Value clock(const measurement::AiclkRunEvidence& clock);

    // Appends this value's JSON to `out`. Throws RecordKeyError for a refused
    // key at any depth, and std::runtime_error for an invalid population.
    void render(std::ostream& out) const;

private:
    struct Population {
        std::vector<uint64_t> samples;
    };
    using AiclkClock = std::shared_ptr<const measurement::AiclkRunEvidence>;
    std::variant<
        std::monostate,
        bool,
        uint64_t,
        int64_t,
        std::string,
        std::vector<Value>,
        std::vector<Field>,
        Population,
        Throughput,
        FixedClock,
        AiclkClock>
        value_;
};

// One key and its value in a record or a nested object.
struct Field {
    std::string key;
    Value value;
};

// Defined once Field is complete: constructing the variant instantiates std::vector<Field>.
template <typename T, std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>, int>>
Value::Value(T number) {
    if constexpr (std::is_signed_v<T>) {
        value_ = static_cast<int64_t>(number);
    } else {
        value_ = static_cast<uint64_t>(number);
    }
}
inline Value::Value(const char* text) : value_(std::string(text)) {}
inline Value::Value(std::string text) : value_(std::move(text)) {}

namespace detail {

inline void render_string(std::ostream& out, const std::string& text) {
    out << '"';
    for (const unsigned char c : text) {
        if (c == '"' || c == '\\') {
            out << '\\' << c;
        } else if (c < 0x20) {
            static constexpr char hex[] = "0123456789abcdef";
            out << "\\u00" << hex[c >> 4] << hex[c & 0xf];
        } else {
            out << c;
        }
    }
    out << '"';
}

// Appends `fields` as the members of one object, without braces. Throws
// RecordKeyError for a key already in this object.
inline void render_fields(std::ostream& out, const std::vector<Field>& fields) {
    std::vector<std::string_view> seen;
    seen.reserve(fields.size());
    for (size_t i = 0; i < fields.size(); ++i) {
        const auto& field = fields[i];
        if (std::find(seen.begin(), seen.end(), field.key) != seen.end()) {
            throw RecordKeyError(field.key, RecordKeyFault::Duplicate);
        }
        seen.push_back(field.key);
        if (i) {
            out << ',';
        }
        render_string(out, field.key);
        out << ':';
        field.value.render(out);
    }
}

}  // namespace detail

inline Value Value::array(std::vector<Value> items) {
    Value value;
    value.value_ = std::move(items);
    return value;
}

inline Value Value::object(std::vector<Field> fields) {
    Value value;
    value.value_ = std::move(fields);
    return value;
}

inline Value Value::population(std::vector<uint64_t> samples) {
    Value value;
    value.value_ = Population{std::move(samples)};
    return value;
}

inline Value Value::throughput(const Throughput& interval) {
    Value value;
    value.value_ = interval;
    return value;
}

inline Value Value::clock(const FixedClock& clock) {
    if (clock.hz == 0) {
        throw std::invalid_argument("fixed clock requires a nonzero rate");
    }
    Value value;
    value.value_ = clock;
    return value;
}

inline Value Value::clock(const measurement::AiclkRunEvidence& clock) {
    Value value;
    value.value_ = std::make_shared<const measurement::AiclkRunEvidence>(clock);
    return value;
}

inline void Value::render(std::ostream& out) const {
    std::visit(
        [&out](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                out << "null";
            } else if constexpr (std::is_same_v<T, bool>) {
                out << (v ? "true" : "false");
            } else if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t>) {
                out << v;
            } else if constexpr (std::is_same_v<T, std::string>) {
                detail::render_string(out, v);
            } else if constexpr (std::is_same_v<T, std::vector<Value>>) {
                out << '[';
                for (size_t i = 0; i < v.size(); ++i) {
                    if (i) {
                        out << ',';
                    }
                    v[i].render(out);
                }
                out << ']';
            } else if constexpr (std::is_same_v<T, std::vector<Field>>) {
                out << '{';
                detail::render_fields(out, v);
                out << '}';
            } else if constexpr (std::is_same_v<T, Population>) {
                out << sample_population_json(v.samples);
            } else if constexpr (std::is_same_v<T, Throughput>) {
                out << throughput_json(
                    v.start_ticks,
                    v.end_ticks,
                    v.requests,
                    v.completed,
                    v.useful_bytes_per_request,
                    v.slot_bytes_per_request);
            } else if constexpr (std::is_same_v<T, FixedClock>) {
                out << "{\"domain\":\"" << measurement::counter_name(v.counter) << "\",\"expected_hz\":" << v.hz
                    << ",\"conversion_hz\":" << v.hz << ",\"errors\":[]}";
            } else {
                static_assert(std::is_same_v<T, AiclkClock>, "every Value kind renders");
                out << clock_json(*v);
            }
        },
        value_);
}

/**
 * The typed fields of one record. record() renders them in this order: suite
 * and member, `fields` as given, then the envelope -- measurement_schema,
 * wall_counter and cpu_counter, the batch, the warm-up selection, clock, and
 * tt_metal_slow_dispatch_mode.
 *
 * measurement_schema is 2 when `clock` holds fixed or AICLK evidence, 1 when it
 * is NoConversion, and absent when it is empty. A wall counter requires a
 * clock, NoConversion requires a wall counter, a batch requires both
 * counters, and a clock requires `warmup`: the selection behind the run's
 * warm-up count (explicit or calibrated), or NoWarmup. AICLK evidence with a
 * reference counter must have sampled the wall counter itself, or the Tensix
 * live timer for a DRISC's RiscDebugWallClock (execution::aiclk_reference).
 */
struct RecordSpec {
    std::string suite;
    std::string member;
    std::vector<Field> fields;
    std::optional<measurement::Counter> wall_counter;
    std::optional<measurement::Counter> cpu_counter;
    // Wall ticks and CPU cycles over `items` items of `logical_bytes` bytes.
    struct Batch {
        uint64_t wall_ticks = 0;
        uint64_t cpu_cycles = 0;
        uint64_t items = 0;
        uint64_t logical_bytes = 0;
    };
    std::optional<Batch> batch;
    std::variant<std::monostate, NoWarmup, measurement::WarmupSelection> warmup;
    std::variant<std::monostate, NoConversion, FixedClock, measurement::AiclkRunEvidence> clock;
};

/**
 * Renders `spec` as one JSONL record body (no braces) for write_jsonl. The
 * envelope ends with tt_metal_slow_dispatch_mode: the process's
 * TT_METAL_SLOW_DISPATCH_MODE as read now, or null when unset, the dispatch
 * mode the record was measured under.
 *
 * Throws std::invalid_argument for an empty suite or member, a clock,
 * counter, batch and warm-up combination RecordSpec does not admit, or a
 * warm-up selection carrying a refusal; RecordKeyError for
 * a refused key at any depth (including a field that repeats an envelope key);
 * and std::runtime_error for a zero or half-range batch counter, zero batch
 * items, or an invalid population.
 */
inline std::string record(RecordSpec spec) {
    const std::string name = spec.suite + "/" + spec.member;
    if (spec.suite.empty() || spec.member.empty()) {
        throw std::invalid_argument("record requires a suite and a member, got \"" + name + "\"");
    }
    const bool no_conversion = std::holds_alternative<NoConversion>(spec.clock);
    if (spec.wall_counter && std::holds_alternative<std::monostate>(spec.clock)) {
        throw std::invalid_argument("record " + name + ": a wall counter requires a clock or NoConversion");
    }
    if (no_conversion && !spec.wall_counter) {
        throw std::invalid_argument("record " + name + ": NoConversion requires a wall counter");
    }
    if (spec.batch && !(spec.wall_counter && spec.cpu_counter)) {
        throw std::invalid_argument("record " + name + ": a batch requires a wall and a CPU counter");
    }
    if (!std::holds_alternative<std::monostate>(spec.clock) && std::holds_alternative<std::monostate>(spec.warmup)) {
        throw std::invalid_argument("record " + name + ": a clock requires a warm-up selection or NoWarmup");
    }
    const auto* aiclk = std::get_if<measurement::AiclkRunEvidence>(&spec.clock);
    if (aiclk && aiclk->reference && spec.wall_counter) {
        const auto expected = *spec.wall_counter == measurement::Counter::RiscDebugWallClock
                                  ? measurement::Counter::TensixLiveTimer
                                  : *spec.wall_counter;
        if (*aiclk->reference != expected) {
            throw std::invalid_argument(
                "record " + name + ": " + measurement::counter_name(*spec.wall_counter) +
                " ticks with AICLK evidence sampled on " + measurement::counter_name(*aiclk->reference));
        }
    }
    const auto* selection = std::get_if<measurement::WarmupSelection>(&spec.warmup);
    if (selection && selection->refusal) {
        throw std::invalid_argument(
            "record " + name + ": the warm-up selection is a refusal, " +
            measurement::warmup_refusal_name(*selection->refusal));
    }
    std::vector<Field> fields{{"suite", std::move(spec.suite)}, {"member", std::move(spec.member)}};
    fields.insert(
        fields.end(), std::make_move_iterator(spec.fields.begin()), std::make_move_iterator(spec.fields.end()));
    if (no_conversion) {
        fields.push_back({"measurement_schema", 1u});
    } else if (!std::holds_alternative<std::monostate>(spec.clock)) {
        fields.push_back({"measurement_schema", 2u});
    }
    if (spec.wall_counter) {
        fields.push_back({"wall_counter", measurement::counter_name(*spec.wall_counter)});
    }
    if (spec.cpu_counter) {
        fields.push_back({"cpu_counter", measurement::counter_name(*spec.cpu_counter)});
    }
    if (const auto& batch = spec.batch) {
        if (batch->wall_ticks == 0 || batch->wall_ticks >= (1ULL << 63) || batch->cpu_cycles == 0 ||
            batch->cpu_cycles >= (1ULL << 63) || batch->items == 0) {
            throw std::runtime_error(
                "record " + name + ": invalid raw batch interval, wall " + std::to_string(batch->wall_ticks) + " cpu " +
                std::to_string(batch->cpu_cycles) + " items " + std::to_string(batch->items));
        }
        fields.insert(
            fields.end(),
            {{"raw_wall_ticks", batch->wall_ticks},
             {"cpu_cycles", batch->cpu_cycles},
             {"items", batch->items},
             {"logical_bytes", batch->logical_bytes}});
    }
    // How the run's unrecorded warm-up count was chosen; pilot figures only for
    // a calibrated count.
    if (selection && selection->calibration) {
        const auto& request = *selection->calibration;
        fields.insert(
            fields.end(),
            {{"warmup_source", "calibrated"},
             {"warmup_target_device_ns", request.target_device_ns},
             {"warmup_pilot_device_ns", request.pilot_device_ns},
             {"warmup_pilot_iterations", request.pilot_iterations}});
    } else if (selection) {
        fields.push_back({"warmup_source", "explicit"});
    } else if (std::holds_alternative<NoWarmup>(spec.warmup)) {
        fields.push_back({"warmup_source", "none"});
    }
    if (const auto* fixed = std::get_if<FixedClock>(&spec.clock)) {
        fields.push_back({"clock", Value::clock(*fixed)});
    } else if (aiclk) {
        fields.push_back({"clock", Value::clock(*aiclk)});
    }
    const char* dispatch = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    fields.push_back({"tt_metal_slow_dispatch_mode", dispatch ? Value(dispatch) : Value()});
    std::ostringstream out;
    out.imbue(std::locale::classic());
    detail::render_fields(out, fields);
    return out.str();
}

// Extract the "characterization" output directory, creating it if missing.
// X280_CHARACTERIZATION_OUTPUT_DIR when set, otherwise
// TT_METAL_HOME/characterization when set, else <cwd>/characterization.
// Filesystem errors propagate to the caller.
inline std::filesystem::path output_dir() {
    const char* selected = std::getenv("X280_CHARACTERIZATION_OUTPUT_DIR");
    if (selected && *selected) {
        std::filesystem::path out(selected);
        std::filesystem::create_directories(out);
        return out;
    }
    const char* home = std::getenv("TT_METAL_HOME");
    std::filesystem::path root = (home && *home) ? std::filesystem::path(home) : std::filesystem::current_path();
    std::filesystem::path out = root / "characterization";
    std::filesystem::create_directories(out);
    return out;
}

// Emit a single JSONL line into `<output_dir()>/<file_stem>.jsonl` (append).
// The caller supplies the fully rendered JSON object body (no braces, no
// trailing newline). The writer wraps it and closes.
// Open, write and close failures throw std::runtime_error naming the path.
inline void write_jsonl(const std::string& file_stem, const std::string& json_object_body) {
    auto path = output_dir() / (file_stem + ".jsonl");
    try {
        std::ofstream f;
        f.exceptions(std::ios::failbit | std::ios::badbit);
        f.open(path, std::ios::out | std::ios::app);
        f << '{' << json_object_body << "}\n";
        f.close();
    } catch (const std::ios_base::failure& error) {
        throw std::runtime_error("write_jsonl failed for " + path.string() + ": " + error.what());
    }
}

}  // namespace x280::harness::output

#endif  // X280_HARNESS_OUTPUT_HPP
