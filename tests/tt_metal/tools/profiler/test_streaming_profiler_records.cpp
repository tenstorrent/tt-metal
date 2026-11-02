// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DevicePublish records on the device, in one capture. The matching table lists, for every argument shape a kernel
// sends, the record types that receive it, and the test checks every type against every call, the ones that must not
// receive it too. The integrity sweep sends structs laid out every way a struct's fields can lie, every way a struct
// can be sent, each field filled from a seed the host recomputes, and compares each record with the struct it should
// be. Both check, through Batch::visit, that a record reaches each of its types exactly once, in its processor's order,
// and that visit and records<T>() agree. The records kernel, on a grid, sends records as real kernels do: metadata with
// Strings and enums, prints, metadata alone, options without metadata, C arrays and many scalars as arguments, call
// sites in helpers, lambdas and templates, zones, and Timestamps on both sides of a zone longer than the clock's low
// word. The host checks each value, its source location and its core, and that a kept record outlives its capture. Run
// with TT_METAL_STREAMING_PROFILER=1. Exits nonzero on any mismatch or missing record. With --errors it instead builds
// kernels that each break one record rule and checks that each build, or load, fails with that rule's message.
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <regex>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/device_observability.hpp>
#include <tt-metalium/experimental/streaming_profiler.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "records_types.hpp"

using namespace tt::tt_metal;
using namespace records_test;
using namespace semantics_test;
namespace api = tt::tt_metal::experimental::device_observability;
using api::Record;
using api::Variadic;
using experimental::streaming_profiler::TimestampedData;
using experimental::streaming_profiler::Zone;

namespace {

constexpr uint32_t kMatchingIterations = 10, kIntegrityIterations = 30, kRecordsIterations = 20;
constexpr std::pair<DataMovementProcessor, NOC> kDmRiscs[] = {
    {DataMovementProcessor::RISCV_0, NOC::RISCV_0_default}, {DataMovementProcessor::RISCV_1, NOC::RISCV_1_default}};
// integrity.cpp's groups: each struct's, by catalog index, and which the compute kernels send.
constexpr uint32_t kGroups = 4;
constexpr uint32_t kComputeGroup = kGroups;
constexpr bool sent_from_compute(size_t index) {
    return index == kCatalogIndex<AllScalars> || index == kCatalogIndex<SignedBits> || index == kCatalogIndex<Stamps>;
}
// How far after a Timestamp the record that carries it may publish: the kernel takes it just before.
constexpr uint64_t kPublishedWithin = 100'000;
const std::string kKernelDir = "tests/tt_metal/tools/profiler/kernels/";

std::mutex output_mutex;
bool failed = false;
void expect(bool ok, const std::string& what) {
    if (!ok) {
        const std::lock_guard lock(output_mutex);
        fmt::print("[records] FAIL {}\n", what);
        failed = true;
    }
}

// A core's processor, as its records name it.
using Lane = std::tuple<size_t, size_t, api::Processor>;
template <typename R>
Lane lane_of(const R& record) {
    const api::Core core = record.core();
    return {core.physical.x, core.physical.y, core.processor};
}

template <typename R>
struct RecordParts;
template <typename P, typename M>
struct RecordParts<Record<P, M>> {
    using Payload = P;
    using Metadata = M;
};

// Follows Batch::visit across batches. A processor's records come in the order it published them, and the copies of one
// record, one per type that takes it, come together: they share the processor and the time it was published, which no
// two records do. `check_record` gets each record's call and the types it reached.
class VisitOrder {
public:
    using CheckRecord = std::function<void(const Lane&, const std::string& call, const std::set<size_t>& types)>;
    explicit VisitOrder(CheckRecord check_record) : check_record_(std::move(check_record)) {}

    template <typename R>
    void visit(const R& record, size_t type, const std::string& call) {
        const Lane lane = lane_of(record);
        const uint64_t time = record.published_at().device_cycles();
        Copies& copies = open_[lane];
        if (copies.types.empty() || time > copies.time) {
            close(lane, copies);
            copies = {time, call, {type}};
            return;
        }
        expect(
            time == copies.time && call == copies.call && copies.types.insert(type).second,
            fmt::format(
                "visit gave {} (type {}) out of its processor's order, or twice, after {} published at {}",
                call,
                type,
                copies.call,
                copies.time));
    }
    void finish() {
        for (auto& [lane, copies] : open_) {
            close(lane, copies);
        }
    }

private:
    struct Copies {
        uint64_t time = 0;
        std::string call;
        std::set<size_t> types;
    };
    void close(const Lane& lane, Copies& copies) {
        if (!copies.types.empty()) {
            check_record_(lane, copies.call, copies.types);
        }
        copies.types.clear();
    }

    CheckRecord check_record_;
    std::map<Lane, Copies> open_;
};

// Visits the batch, giving `on_record` each record and its type's index, and checks that visit lists each type's
// records as records<T>() does.
template <typename F, typename... Rs>
void visit_checked(const api::Batch<Rs...>& batch, F on_record) {
    std::array<std::vector<const void*>, sizeof...(Rs)> visited;
    batch.visit([&](const auto& record) {
        using R = std::decay_t<decltype(record)>;
        constexpr size_t kType = index_of<R, Rs...>();
        visited[kType].push_back(&record);
        on_record(record, kType);
    });
    const auto check_type = [&]<typename R>(std::type_identity<R>) {
        constexpr size_t kType = index_of<R, Rs...>();
        expect(
            std::ranges::equal(
                batch.template records<R>(), visited[kType], [](const R& r, const void* at) { return &r == at; }),
            fmt::format("visit and records<T>() list type {} alike", kType));
    };
    (check_type(std::type_identity<Rs>()), ...);
}

// ---- The matching table ----

using MatchingBatch = api::Batch<
    Record<void, Call>,
    Record<uint32_t, Call>,
    Record<int32_t, Call>,
    Record<uint16_t, Call>,
    Record<uint64_t, Call>,
    Record<float, Call>,
    Record<double, Call>,
    Record<bool, Call>,
    Record<uint8_t, Call>,
    Record<int8_t, Call>,
    Record<char, Call>,
    Record<Color, Call>,
    Record<std::tuple<uint32_t, uint32_t>, Call>,
    Record<std::tuple<uint32_t, float>, Call>,
    Record<std::tuple<float, uint32_t>, Call>,
    Record<std::tuple<uint32_t, uint32_t, uint32_t>, Call>,
    Record<std::array<uint32_t, 3>, Call>,
    Record<std::array<uint32_t, 4>, Call>,
    Record<std::span<const uint32_t>, Call>,
    Record<std::variant<std::span<const uint32_t>, std::array<uint32_t, 3>>, Call>,
    Record<std::array<float, 3>, Call>,
    Record<PairA, Call>,
    Record<PairB, Call>,
    Record<std::variant<PairA, PairB>, Call>,
    Record<std::tuple<PairA, uint32_t>, Call>,
    Record<std::tuple<PairA, std::span<const uint32_t>>, Call>,
    Record<std::array<PairA, 2>, Call>,
    Record<std::span<const PairA>, Call>,
    Record<Timestamp, Call>,
    Record<std::variant<uint32_t, int32_t, Color>, Call>,
    Record<std::variant<uint32_t, uint64_t>, Call>,
    Record<Variadic<uint32_t>, Call>,
    Record<Variadic<std::variant<uint32_t, float>>, Call>,
    Record<Variadic<PairA>, Call>,
    Record<Variadic<std::span<const uint32_t>>, Call>,
    Record<uint32_t, Other>,
    Record<uint32_t, void>>;
// MatchingBatch's types, in its order.
constexpr std::string_view kMatchingTypes[] = {
    "void",
    "u32",
    "i32",
    "u16",
    "u64",
    "f32",
    "f64",
    "bool",
    "u8",
    "i8",
    "char",
    "Color",
    "(u32, u32)",
    "(u32, f32)",
    "(f32, u32)",
    "(u32, u32, u32)",
    "u32[3]",
    "u32[4]",
    "span u32",
    "span u32 | u32[3]",
    "f32[3]",
    "PairA",
    "PairB",
    "PairA | PairB",
    "(PairA, u32)",
    "(PairA, span u32)",
    "PairA[2]",
    "span PairA",
    "Timestamp",
    "u32 | i32 | Color",
    "u32 | u64",
    "Variadic u32",
    "Variadic u32 | f32",
    "Variadic PairA",
    "Variadic span u32",
    "u32 with Other",
    "u32 without metadata"};

// Every call matching_dm.cpp makes, by its metadata, and the types that receive it. No other type may.
const std::map<std::string, std::set<std::string_view>> kReaches{
    {"none", {"void", "Variadic u32", "Variadic u32 | f32", "Variadic PairA", "Variadic span u32"}},
    {"u32", {"u32", "u32 | i32 | Color", "u32 | u64", "Variadic u32", "Variadic u32 | f32"}},
    {"i32", {"i32", "u32 | i32 | Color"}},
    {"u16", {"u16"}},
    {"u64", {"u64", "u32 | u64"}},
    {"f32", {"f32", "Variadic u32 | f32"}},
    {"f64", {"f64"}},
    {"bool", {"bool"}},
    {"u8", {"u8"}},
    {"i8", {"i8"}},
    {"char", {"char"}},
    {"Color", {"Color", "u32 | i32 | Color"}},
    {"u32 u32", {"(u32, u32)", "Variadic u32", "Variadic u32 | f32"}},
    {"u32 f32", {"(u32, f32)", "Variadic u32 | f32"}},
    {"f32 u32", {"(f32, u32)", "Variadic u32 | f32"}},
    {"u32 u32 u32", {"(u32, u32, u32)", "Variadic u32", "Variadic u32 | f32"}},
    {"u32[3]", {"u32[3]", "span u32", "span u32 | u32[3]", "Variadic span u32"}},
    {"u32[4]", {"u32[4]", "span u32", "span u32 | u32[3]", "Variadic span u32"}},
    {"f32[3]", {"f32[3]"}},
    {"PairA", {"PairA", "PairA | PairB", "Variadic PairA"}},
    {"PairB", {"PairB", "PairA | PairB"}},
    {"PairA u32", {"(PairA, u32)"}},
    {"PairA u32[2]", {"(PairA, span u32)"}},
    {"PairA[2]", {"PairA[2]", "span PairA"}},
    {"Timestamp", {"Timestamp"}},
    {"u32 with Other", {"u32 with Other"}},
    {"u32 without metadata", {"u32 without metadata"}},
};
// For each variant type, the alternative each argument of a call it receives takes.
const std::map<std::pair<std::string_view, std::string>, std::vector<size_t>> kAlternatives{
    {{"span u32 | u32[3]", "u32[3]"}, {1}},
    {{"span u32 | u32[3]", "u32[4]"}, {0}},
    {{"PairA | PairB", "PairA"}, {0}},
    {{"PairA | PairB", "PairB"}, {1}},
    {{"u32 | i32 | Color", "u32"}, {0}},
    {{"u32 | i32 | Color", "i32"}, {1}},
    {{"u32 | i32 | Color", "Color"}, {2}},
    {{"u32 | u64", "u32"}, {0}},
    {{"u32 | u64", "u64"}, {1}},
    {{"Variadic u32 | f32", "f32"}, {1}},
    {{"Variadic u32 | f32", "u32"}, {0}},
    {{"Variadic u32 | f32", "u32 u32"}, {0, 0}},
    {{"Variadic u32 | f32", "u32 f32"}, {0, 1}},
    {{"Variadic u32 | f32", "f32 u32"}, {1, 0}},
    {{"Variadic u32 | f32", "u32 u32 u32"}, {0, 0, 0}},
};

template <typename R>
std::string matching_call(const R& record) {
    using Metadata = typename RecordParts<R>::Metadata;
    if constexpr (std::is_void_v<Metadata>) {
        return "u32 without metadata";
    } else if constexpr (std::is_same_v<Metadata, Other>) {
        return std::string(record.metadata().name) + " with Other";
    } else {
        return std::string(record.metadata().name);
    }
}
template <typename P>
std::vector<size_t> alternatives_of(const P& payload) {
    if constexpr (api::detail::is_specialization<P, std::variant>) {
        return {payload.index()};
    } else if constexpr (api::detail::is_array_value<P>) {
        std::vector<size_t> indices;
        for (const auto& argument : payload) {
            if constexpr (api::detail::is_specialization<std::decay_t<decltype(argument)>, std::variant>) {
                indices.push_back(argument.index());
            }
        }
        return indices;
    } else {
        return {};
    }
}

struct MatchingResults {
    std::mutex mutex;
    VisitOrder order{[this](const Lane& lane, const std::string& call, const std::set<size_t>& types) {
        std::set<std::string_view> reached;
        for (const size_t type : types) {
            reached.insert(kMatchingTypes[type]);
        }
        const auto want = kReaches.find(call);
        expect(want != kReaches.end(), "an unknown call " + call);
        if (want != kReaches.end()) {
            expect(
                reached == want->second,
                fmt::format("{} reached {}, want {}", call, fmt::join(reached, ", "), fmt::join(want->second, ", ")));
        }
        ++records[{lane, call}];
    }};
    std::map<std::pair<Lane, std::string>, uint64_t> records;
};

// ---- The integrity sweep ----

template <typename... Ss>
using IntegrityBatchOf =
    api::Batch<Record<Ss, Send>..., Record<std::tuple<uint8_t, Ss>, Send>..., Record<Variadic<Ss>, Send>...>;
using IntegrityBatch = IntegrityBatchOf<SEMANTICS_CATALOG>;
constexpr size_t kCatalogSize = std::tuple_size_v<std::tuple<SEMANTICS_CATALOG>>;

// The ways a struct is sent, and the types each reaches (0: Record<S>, 1: the pair, 2: Variadic<S>).
const std::map<std::string, std::set<size_t>> kSendsReach{
    {"alone", {0, 2}}, {"no inline", {0, 2}}, {"pair", {1}}, {"list", {2}}};

// A Stamps, StampedRun or StatRows as the kernel built it: its Timestamps at their distances, the other fields from the
// seed.
bool sent_as(const Stamps& stamps, uint64_t seed, uint32_t iteration, const Timestamp& published) {
    const uint64_t first = stamps.first.device_cycles();
    return stamps.x == uint8_t(seed) && stamps.second.device_cycles() == first - stamp_delta(iteration, 1) &&
           stamps.third.device_cycles() == first - stamp_delta(iteration, 2) && first <= published.device_cycles() &&
           published.device_cycles() - first < kPublishedWithin;
}
bool sent_as(const StampedRun& run, uint64_t seed, uint32_t iteration, const Timestamp& published) {
    const uint64_t back = published.device_cycles() - run.at.device_cycles();
    return std::ranges::equal(run.words, filled<StampedRun>(seed).words) && back >= stamp_delta(iteration, 1) &&
           back < stamp_delta(iteration, 1) + kPublishedWithin;
}
bool sent_as(const StatRows& rows, uint64_t seed, uint32_t iteration, const Timestamp& published) {
    const uint64_t back = published.device_cycles() - rows.at.device_cycles();
    return std::ranges::equal(rows.rows, filled<StatRows>(seed).rows) && back >= stamp_delta(iteration, 1) &&
           back < stamp_delta(iteration, 1) + kPublishedWithin;
}
template <typename S>
bool sent_as(const S& s, uint64_t seed, uint32_t, const Timestamp&) {
    return s == filled<S>(seed);
}

struct IntegrityResults {
    std::mutex mutex;
    VisitOrder order{[this](const Lane& lane, const std::string& call, const std::set<size_t>& types) {
        // The types of one struct: S, the pair and the list are kCatalogSize apart.
        const size_t index = *types.begin() % kCatalogSize;
        std::set<size_t> forms;
        for (const size_t type : types) {
            expect(type % kCatalogSize == index, fmt::format("one record reached two structs' types ({})", call));
            forms.insert(type / kCatalogSize);
        }
        const auto want = kSendsReach.find(call);
        expect(
            want != kSendsReach.end() && forms == want->second,
            fmt::format("struct {} sent as {} reached forms {}", index, call, fmt::join(forms, ", ")));
        ++records[{lane, index, call}];
    }};
    // Each type's records so far from each processor and call, which give the next one's iteration.
    std::map<std::tuple<size_t, Lane, std::string>, uint32_t> seen;
    std::map<std::tuple<Lane, size_t, std::string>, uint64_t> records;
};

// Checks a record of the integrity sweep against what its seed says it holds.
template <typename R>
void check_integrity(IntegrityResults& results, const R& record, size_t type) {
    using P = typename RecordParts<R>::Payload;
    const std::string call(record.metadata().name);
    const Lane lane = lane_of(record);
    const uint32_t iteration = results.seen[{type, lane, call}]++;
    const auto processor = uint32_t(std::get<2>(lane));
    const auto check = [&](auto index_tag, const auto& payload) {
        using S = typename decltype(index_tag)::type;
        const uint64_t seed = seed_of(kCatalogIndex<S>, processor, iteration);
        const Timestamp& published = record.published_at();
        bool ok = false;
        if constexpr (std::is_same_v<P, S>) {
            ok = sent_as(payload, seed, iteration, published);
        } else if constexpr (std::is_same_v<P, std::tuple<uint8_t, S>>) {
            ok = std::get<0>(payload) == uint8_t(seed) && sent_as(std::get<1>(payload), seed, iteration, published);
        } else {
            ok = call == "list" ? payload.size() == 2 && sent_as(payload[0], seed, iteration, published) &&
                                      sent_as(payload[1], mix(seed), iteration, published)
                                : payload.size() == 1 && sent_as(payload[0], seed, iteration, published);
        }
        expect(
            ok,
            fmt::format(
                "struct {} sent as {}, iteration {} from processor {}, holds other values than its seed's",
                kCatalogIndex<S>,
                call,
                iteration,
                processor));
    };
    if constexpr (api::detail::is_specialization<P, std::tuple>) {
        check(std::type_identity<std::tuple_element_t<1, P>>(), record.payload());
    } else if constexpr (api::detail::is_specialization<P, Variadic>) {
        check(std::type_identity<typename P::value_type>(), record.payload());
    } else {
        check(std::type_identity<P>(), record.payload());
    }
}

// ---- The records kernel ----

constexpr uint32_t kGridSide = 2;
constexpr uint64_t kCores = kGridSide * kGridSide;
// The records one call site sends from both data-movement RISCs across the grid.
constexpr uint64_t kRecordsPerSite = std::size(kDmRiscs) * kCores * kRecordsIterations;
// Each kind's count is its call sites times kRecordsPerSite, and once per kernel for those after the loop.
const std::map<std::string, uint64_t> kExpectedCounts{
    {"records-iteration zones", kRecordsPerSite},
    {"long zones", 2 * kCores},
    {"NocEvent + NocSite", kRecordsPerSite},
    {"TileStats + Name", kRecordsPerSite},
    {"Sample + no metadata", kRecordsPerSite},
    {"RowValue + Name", 3 * kRecordsPerSite},           // "row", "ids", "tag"
    {"(number, float array) + Name", kRecordsPerSite},  // "row"
    {"uint32_t[3] + Name", kRecordsPerSite},            // "ids"
    {"Bf16 array + Name", kRecordsPerSite},
    {"Scalars + Name", kRecordsPerSite},
    {"Skewed + Name", kRecordsPerSite},
    {"no payload + Hole", kRecordsPerSite},
    {"no payload + LaneFlags", kRecordsPerSite},
    {"no payload + From", 2 * kRecordsPerSite},            // "helper", "lambda"
    {"uint32_t + InTemplate", kRecordsPerSite},            // publish_from_template<uint32_t>
    {"uint64_t + InTemplate", kRecordsPerSite},            // publish_from_template<uint64_t>
    {"uint16_t + InTemplate", kRecordsPerSite},            // Publisher<uint16_t>::publish
    {"FormatValue + Format", 2 * kRecordsPerSite},         // EXAMPLE_PRINT, "iteration done"
    {"TimestampedData", kRecordsPerSite},                  // "pair"
    {"Stamped + Name", kRecordsPerSite + 3 * 2 * kCores},  // "since start", and three after the loop
    {"Interval + Name", kRecordsPerSite},
};
// A record publishes within this many ticks of any time it carries.
constexpr uint64_t kPublishWithinTicks = 1'000'000'000;
// More ticks than the whole kernel takes, including its long zone.
constexpr uint64_t kKernelTicksBound = uint64_t{1} << 34;
// A long zone starts within this many ticks of the time its processor takes just before opening it.
constexpr uint64_t kLongZoneStartWithinTicks = 1000;

using RowValue = std::variant<uint32_t, std::span<const float>, std::span<const uint32_t>, std::span<const char>>;
using FormatValue = std::variant<uint32_t, uint64_t>;
// "row" as exact arguments: its iteration as either number type, then its array.
using RowArguments = std::tuple<std::variant<uint32_t, float>, std::span<const float>>;
using Scalars = std::tuple<int8_t, int16_t, int32_t, uint16_t, double, char, uint64_t, bool>;
using RecordsBatch = api::Batch<
    Record<NocEvent, NocSite>,
    Record<TileStats, Name>,
    Record<Sample, void>,
    Record<Variadic<RowValue>, Name>,
    Record<RowArguments, Name>,
    Record<std::array<uint32_t, 3>, Name>,
    Record<std::span<const Bf16>, Name>,
    Record<Scalars, Name>,
    Record<Skewed, Name>,
    Record<void, Hole>,
    Record<void, LaneFlags>,
    Record<void, From>,
    Record<uint32_t, InTemplate>,
    Record<uint64_t, InTemplate>,
    Record<uint16_t, InTemplate>,
    Record<Variadic<FormatValue>, Format>,
    TimestampedData,
    Record<Stamped, Name>,
    Record<Interval, Name>,
    Zone>;

template <typename T, typename V>
const T* value_at(std::span<const V> values, size_t k) {
    return k < values.size() ? std::get_if<T>(&values[k]) : nullptr;
}

bool sent_as_row(uint32_t iteration, std::span<const float> row) {
    return row.size() == 4 && row[0] == float(iteration) && row[3] == float(iteration) + 0.75f;
}
bool sent_as_row(std::span<const RowValue> values) {
    const auto* iteration = value_at<uint32_t>(values, 0);
    const auto* row = value_at<std::span<const float>>(values, 1);
    return values.size() == 2 && iteration && row && sent_as_row(*iteration, *row);
}
bool sent_as_row(const RowArguments& arguments) {
    const auto& [iteration, row] = arguments;
    return std::holds_alternative<uint32_t>(iteration) && sent_as_row(std::get<uint32_t>(iteration), row);
}
// A call site in a template names the template, without the template arguments of the instantiation.
template <typename R>
bool sent_in_template(const R& record, std::string_view name, std::string_view function) {
    const std::string_view at = record.location().function;
    return record.metadata().name == name && at.ends_with(function) && at.find("[with") == std::string_view::npos;
}
bool before(const Timestamp& at, const Timestamp& published, uint64_t ticks = kPublishWithinTicks) {
    return at.device_cycles() <= published.device_cycles() && at.host_cycles() <= published.host_cycles() &&
           published.device_cycles() - at.device_cycles() < ticks;
}

struct RecordsResults {
    std::mutex mutex;
    std::optional<Record<Variadic<RowValue>, Name>> kept_row;
    std::optional<Record<RowArguments, Name>> kept_row_arguments;
    std::map<std::pair<std::string, CoreCoord>, uint64_t> core_counts;
    // A kernel's processor and launch.
    using Kernel = std::tuple<Lane, uint32_t>;
    template <typename R>
    static Kernel kernel_of(const R& record) {
        return {lane_of(record), record.runtime_id()};
    }
    std::map<Kernel, uint64_t> kernel_starts, long_starts, long_opens;

    template <typename R>
    void common(const std::string& kind, const R& record) {
        const api::Core core = record.core();
        ++core_counts[{kind, core.logical}];
        expect(
            core.logical.x < kGridSide && core.logical.y < kGridSide,
            fmt::format("{}: core {} outside the grid", kind, core.logical.str()));
        const std::string_view file = record.location().file;
        expect(!record.location().function.empty(), fmt::format("{}: no function", kind));
        expect(
            file.ends_with("records.cpp") && record.location().line > 0,
            fmt::format("{}: location {}:{}", kind, file, record.location().line));
    }
    template <typename R, typename... Ts, typename F>
    void each(const api::Batch<Ts...>& batch, const std::string& kind, F check) {
        for (const R& record : batch.template records<R>()) {
            common(kind, record);
            const api::Core core = record.core();
            expect(
                check(record),
                fmt::format(
                    "{}: wrong values from core {} processor {} line {}",
                    kind,
                    core.logical.str(),
                    int(core.processor),
                    record.location().line));
        }
    }
};

void check_records(RecordsResults& results, const RecordsBatch& batch) {
    for (const Zone& zone : batch.records<Zone>()) {
        if (zone.metadata().name == "records-iteration") {
            results.common("records-iteration zones", zone);
        } else if (zone.metadata().name == "long") {
            results.common("long zones", zone);
            const auto& [start, end] = zone.payload();
            const uint64_t ticks = end.device_cycles() - start.device_cycles();
            expect(
                ticks >= kLongZoneTicks && ticks < kLongZoneTicks + (1u << 20) &&
                    start.host_cycles() < end.host_cycles(),
                fmt::format("long zone of {} ticks", ticks));
            results.long_starts[RecordsResults::kernel_of(zone)] = start.device_cycles();
        }
    }
    results.each<Record<NocEvent, NocSite>>(batch, "NocEvent + NocSite", [](const auto& record) {
        const auto& payload = record.payload();
        return record.location().function.ends_with("kernel_main()") && payload.x < kRecordsIterations &&
               payload.y == (record.core().processor == api::Processor::BRISC ? 0 : 1) &&
               payload.bytes == uint32_t(payload.x) * 64 && record.metadata().api == "noc_async_read" &&
               record.metadata().type == NocType::Read && record.metadata().color == 0xFF8000;
    });
    results.each<Record<TileStats, Name>>(batch, "TileStats + Name", [](const auto& record) {
        const auto& payload = record.payload();
        return record.metadata().name == "tile stats" && payload.tiles < kRecordsIterations &&
               payload.bytes == payload.tiles * 2048 && payload.cycles == payload.tiles * 100 &&
               payload.stalls == payload.tiles % 3;
    });
    results.each<Record<Sample, void>>(batch, "Sample + no metadata", [](const auto& record) {
        const auto& payload = record.payload();
        return payload.value >= -5 && payload.value < int64_t(kRecordsIterations) - 5;
    });
    results.each<Record<Variadic<RowValue>, Name>>(batch, "RowValue + Name", [&](const auto& record) {
        const auto& payload = record.payload();
        const std::string_view name = record.metadata().name;
        if (name == "row") {
            if (!results.kept_row) {
                results.kept_row = record;
                const auto* row = value_at<std::span<const float>>(payload, 1);
                const auto* kept = value_at<std::span<const float>>(results.kept_row->payload(), 1);
                expect(row && kept && row->data() != kept->data(), "a kept record owns its array");
            }
            return sent_as_row(payload);
        }
        if (name == "ids") {
            const auto* ids = value_at<std::span<const uint32_t>>(payload, 0);
            return payload.size() == 1 && ids && ids->size() == 3 && (*ids)[0] < kRecordsIterations &&
                   (*ids)[1] == 2 * (*ids)[0] && (*ids)[2] == 3 * (*ids)[0];
        }
        // A char array that isn't const is data, not a string.
        const auto* tag = value_at<std::span<const char>>(payload, 0);
        return name == "tag" && payload.size() == 1 && tag && tag->size() == 4 && (*tag)[0] == 'i' &&
               (*tag)[1] == 'd' && (*tag)[3] == 0;
    });
    results.each<Record<RowArguments, Name>>(batch, "(number, float array) + Name", [&](const auto& record) {
        if (!results.kept_row_arguments) {
            results.kept_row_arguments = record;
            expect(
                std::get<1>(record.payload()).data() != std::get<1>(results.kept_row_arguments->payload()).data(),
                "a kept record owns the array in its tuple");
        }
        return record.metadata().name == "row" && sent_as_row(record.payload());
    });
    results.each<Record<std::span<const Bf16>, Name>>(batch, "Bf16 array + Name", [](const auto& record) {
        const auto& payload = record.payload();
        return record.metadata().name == "bf16 row" && payload.size() == 2 && payload[1].bits == payload[0].bits + 1;
    });
    // A C array argument is received as a std::array of its length.
    results.each<Record<std::array<uint32_t, 3>, Name>>(batch, "uint32_t[3] + Name", [](const auto& record) {
        const std::array<uint32_t, 3>& ids = record.payload();
        return record.metadata().name == "ids" && ids[0] < kRecordsIterations && ids[1] == 2 * ids[0] &&
               ids[2] == 3 * ids[0];
    });
    results.each<Record<Scalars, Name>>(batch, "Scalars + Name", [](const auto& record) {
        const auto& [i8, i16, i32, u16, f64, c, u64, odd] = record.payload();
        const int32_t iteration = -1 - i8;
        return record.metadata().name == "scalars" && iteration >= 0 && iteration < int32_t(kRecordsIterations) &&
               i16 == -300 - iteration && i32 == -70000 - iteration && u16 == 60000 + iteration &&
               f64 == iteration + 0.5 && c == 'a' + iteration &&
               u64 == (uint64_t(iteration) << 33 | uint64_t(iteration)) && odd == (iteration % 2 == 1);
    });
    results.each<Record<Skewed, Name>>(batch, "Skewed + Name", [](const auto& record) {
        const auto& payload = record.payload();
        const uint32_t a = payload.a, b = payload.b;
        return record.metadata().name == "skewed" && b == (a + 1) % 256;
    });
    results.each<Record<void, Hole>>(batch, "no payload + Hole", [](const auto& record) {
        const Hole& hole = record.metadata();
        return hole.iteration == 7 && std::string_view(hole.name) == "Ada" && hole.a == 5 && hole.b == 17;
    });
    results.each<Record<void, LaneFlags>>(batch, "no payload + LaneFlags", [](const auto& record) {
        const LaneFlags& flags = record.metadata();
        return flags.kind == 5 && flags.lane == 17 && flags.count == 1000;
    });
    results.each<Record<void, From>>(batch, "no payload + From", [](const auto& record) {
        const std::string_view function = record.location().function;
        return record.metadata().tag == "helper"
                   ? function.ends_with("publish_from_helper()")
                   : record.metadata().tag == "lambda" && function.starts_with("kernel_main()::<lambda");
    });
    results.each<Record<uint32_t, InTemplate>>(batch, "uint32_t + InTemplate", [](const auto& record) {
        return sent_in_template(record, "function", "publish_from_template(T)") &&
               record.payload() < kRecordsIterations;
    });
    results.each<Record<uint64_t, InTemplate>>(batch, "uint64_t + InTemplate", [](const auto& record) {
        return sent_in_template(record, "function", "publish_from_template(T)") &&
               record.payload() % (uint64_t{1} << 33) == 0 && record.payload() >> 33 < kRecordsIterations;
    });
    results.each<Record<uint16_t, InTemplate>>(batch, "uint16_t + InTemplate", [](const auto& record) {
        return sent_in_template(record, "member", "Publisher<T>::publish(T)") && record.payload() < kRecordsIterations;
    });
    results.each<Record<Variadic<FormatValue>, Format>>(batch, "FormatValue + Format", [](const auto& record) {
        const auto& payload = record.payload();
        if (payload.empty()) {
            return record.metadata().format == "iteration done" && record.metadata().arguments.empty();
        }
        const auto* iteration = value_at<uint32_t>(payload, 0);
        const auto* big = value_at<uint64_t>(payload, 1);
        return payload.size() == 2 && iteration && big && *iteration < kRecordsIterations &&
               *big == uint64_t(*iteration) << 33 && record.metadata().format == "hello Ada iteration {} big {}" &&
               record.metadata().arguments == "i, uint64_t(i) << 33";
    });
    // Dispatch kernels send TimestampedData of their own.
    for (const TimestampedData& record : batch.records<TimestampedData>()) {
        if (record.metadata().name == "pair") {
            results.common("TimestampedData", record);
            const auto& values = record.payload();
            expect(
                values.size() == 2 && values[0] < kRecordsIterations && values[1] == values[0] << 33,
                "TimestampedData: wrong values");
        }
    }
    results.each<Record<Stamped, Name>>(batch, "Stamped + Name", [&](const auto& record) {
        const auto& payload = record.payload();
        const std::string_view name = record.metadata().name;
        const uint64_t at = payload.at.device_cycles();
        if (name == "since start") {
            // Every record that carries the kernel's start time carries the same value, including the last, which the
            // kernel sends after the long zone.
            return results.kernel_starts.try_emplace(RecordsResults::kernel_of(record), at).first->second == at &&
                   payload.iteration <= kRecordsIterations &&
                   before(payload.at, record.published_at(), kKernelTicksBound);
        }
        if (name == "after full") {
            return payload.iteration == kRecordsIterations && before(payload.at, record.published_at());
        }
        if (name == "before long") {
            results.long_opens[RecordsResults::kernel_of(record)] = at;
            return record.published_at().device_cycles() - at > uint64_t{1} << 32 &&
                   before(payload.at, record.published_at(), kKernelTicksBound);
        }
        return false;
    });
    results.each<Record<Interval, Name>>(batch, "Interval + Name", [](const auto& record) {
        const auto& payload = record.payload();
        return record.metadata().name == "interval" && before(payload.start, payload.end) &&
               before(payload.end, record.published_at());
    });
}

// After the capture: the kept records, the long zones' starts, and every kind's count from each core.
void check_records_after(RecordsResults& results) {
    expect(
        results.kept_row && sent_as_row(results.kept_row->payload()), "a record with an array kept past its callback");
    // The device is closed and its capture's lanes have gone with its receiver, but for those a kept record holds.
    expect(
        results.kept_row && results.kept_row->metadata().name == "row" && !results.kept_row->location().file.empty() &&
            results.kept_row->core().logical.x < kGridSide && results.kept_row->core().logical.y < kGridSide,
        "a kept record's core, location and metadata outlive its capture");
    expect(
        results.kept_row_arguments && sent_as_row(results.kept_row_arguments->payload()),
        "a record with an array in its tuple kept past its callback");
    for (const auto& [kernel, start] : results.long_starts) {
        const auto open = results.long_opens.find(kernel);
        expect(
            open != results.long_opens.end() && open->second <= start &&
                start - open->second < kLongZoneStartWithinTicks,
            "long zone's start");
    }
    // Every record comes from a core in the grid (common() checks), so each core's count being right makes the total.
    for (const auto& [kind, want] : kExpectedCounts) {
        for (uint32_t x = 0; x < kGridSide; x++) {
            for (uint32_t y = 0; y < kGridSide; y++) {
                const CoreCoord core{x, y};
                const uint64_t seen_on_core = results.core_counts[{kind, core}];
                expect(
                    seen_on_core == want / kCores,
                    fmt::format("{}: {} records from core {}, want {}", kind, seen_on_core, core.str(), want / kCores));
            }
        }
    }
}

// ---- Running kernels ----

// A program of kernels/`file` on the first `riscs` data-movement RISCs of `cores`.
Program dm_program(
    const std::string& file,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& cores,
    const std::map<std::string, std::string>& defines,
    size_t riscs = std::size(kDmRiscs)) {
    Program program = CreateProgram();
    for (const auto& [processor, noc] : std::span(kDmRiscs).first(riscs)) {
        CreateKernel(
            program,
            kKernelDir + file,
            cores,
            DataMovementConfig{.processor = processor, .noc = noc, .defines = defines});
    }
    return program;
}
void enqueue(distributed::MeshDevice& mesh_device, Program program) {
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device.shape()), std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), workload, /*blocking=*/false);
}
// One capture: opens the device, enqueues the programs, waits for them and closes the device, which delivers every
// record before it returns. Returns the error a build, load or run threw, or nothing.
std::string run(const std::function<void(distributed::MeshDevice&)>& enqueue_programs) {
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    std::string error;
    try {
        enqueue_programs(*mesh_device);
        distributed::Finish(mesh_device->mesh_command_queue());
    } catch (const std::exception& e) {
        error = e.what();
    }
    mesh_device->close();
    return error;
}

// ---- The error cases ----

// A record_errors.cpp case, and the message its build, or its load, fails with.
struct ExpectedError {
    int case_id;
    bool fails_at_build;
    bool on_both_dm_riscs;
    std::string_view message;
};
constexpr std::string_view kUnsupportedType =
    "this type is not supported; use a fixed-width integer, char, float, double, bool, enum, String, Timestamp, "
    "struct or array";
constexpr std::string_view kNotSendable =
    "only aggregate structs, fixed-width scalars, enums, String, Timestamp and arrays of them are supported; this type "
    "has a constructor, private field or virtual function";
constexpr std::string_view kOverLimit = "this record is over the 2004-byte limit; split it into several records";
constexpr std::string_view kPayloadString =
    "strings are not supported in a payload; send the text as metadata, a DevicePublish call per string";
constexpr ExpectedError kErrors[] = {
    {1, true, false, kPayloadString},
    {2, true, false, "strings sent with DevicePublish need an _s suffix: \"text\"_s"},
    {3, true, false, kPayloadString},
    {4, true, false, kNotSendable},
    {5, true, false, "templates, function-local types and types in an anonymous namespace are not supported"},
    {6, true, false, "templates, function-local types and types in an anonymous namespace are not supported"},
    {7, true, false, "const and reference members are not supported"},
    {8, true, false, "enums without a fixed underlying type are not supported; use enum E : uint8_t"},
    {9, true, false, kUnsupportedType},
    {10, true, false, "arrays of arrays are not supported; put the inner array in a struct"},
    {11, true, false, "empty arrays are not supported"},
    {12, true, false, "templates, function-local types and types in an anonymous namespace are not supported"},
    {13, true, false, kOverLimit},
    {14,
     true,
     false,
     "unknown DevicePublish option; the options are kPublishBatched, kSendToExternalProcess and kNoInline"},
    {15, true, false, "types with a custom copy, move or destructor are not supported"},
    {16, true, false, kUnsupportedType},
    {17, true, false, "unions are not supported; send the member it holds"},
    {18, true, false, "unions are not supported; send the member it holds"},
    {19, true, false, kNotSendable},
    {20, false, true, "two kernels send different types named Clash"},
    {21, false, false, "records_test::LongField has different fields in kernels than on the host"},
    {22, true, false, kPayloadString},
    {23,
     true,
     false,
     "DevicePublish takes its metadata before its options: DevicePublish<Metadata{...}, kOption>(...)"},
    {24, true, false, "metadata is a struct; put the value in one"},
    {25,
     true,
     false,
     "kPublishBatched doesn't combine with kSendToExternalProcess, whose records are published at once"},
    {26, true, false, "arrays of empty structs are not supported"},
};

// Each case is one build of record_errors.cpp, on both data-movement RISCs when the case sets on_both_dm_riscs. A build
// must fail with one error, located at the case's call. A load must fail with the case's message.
int expect_errors() {
    // A callback that takes LongField, so the host compares its fields with the kernel's when the kernel loads.
    auto callback = api::RegisterCallback([](const api::Batch<Record<LongField, Name>>&) {}, "records-errors");
    int failures = 0;
    for (const auto& [case_id, fails_at_build, on_both_dm_riscs, message] : kErrors) {
        const std::string failure = run([case_id, on_both_dm_riscs](distributed::MeshDevice& mesh_device) {
            enqueue(
                mesh_device,
                dm_program(
                    "record_errors.cpp",
                    CoreCoord{0, 0},
                    {{"RECORD_ERROR", std::to_string(case_id)}},
                    on_both_dm_riscs ? 2 : 1));
        });
        const size_t e = failure.find(": error: ");
        const bool ok = failure.find(message) != std::string::npos &&
                        (!fails_at_build || (e != std::string::npos && e == failure.rfind(": error: ") &&
                                             std::regex_search(failure, std::regex(R"(record_errors\.cpp:\d+:\d+:)"))));
        fmt::print("[records] error {}: {}\n", case_id, ok ? "PASS" : "FAIL, got: " + failure);
        failures |= ok ? 0 : 1;
    }
    return failures;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--errors") {
        return expect_errors();
    }
    MatchingResults matching;
    IntegrityResults integrity;
    RecordsResults records;
    auto matching_callback = api::RegisterCallback(
        [&](const MatchingBatch& batch) {
            const std::lock_guard lock(matching.mutex);
            visit_checked(batch, [&](const auto& record, size_t type) {
                using P = typename RecordParts<std::decay_t<decltype(record)>>::Payload;
                const std::string call = matching_call(record);
                matching.order.visit(record, type, call);
                if constexpr (!std::is_void_v<P>) {
                    const std::vector<size_t> alternatives = alternatives_of(record.payload());
                    if (!alternatives.empty()) {
                        const auto want = kAlternatives.find({kMatchingTypes[type], call});
                        expect(
                            want != kAlternatives.end() && want->second == alternatives,
                            fmt::format(
                                "{} took alternatives {} for {}",
                                kMatchingTypes[type],
                                fmt::join(alternatives, ", "),
                                call));
                    }
                }
            });
        },
        "records-matching");
    auto integrity_callback = api::RegisterCallback(
        [&](const IntegrityBatch& batch) {
            const std::lock_guard lock(integrity.mutex);
            visit_checked(batch, [&](const auto& record, size_t type) {
                integrity.order.visit(record, type, std::string(record.metadata().name));
                check_integrity(integrity, record, type);
            });
        },
        "records-integrity");
    auto records_callback = api::RegisterCallback(
        [&](const RecordsBatch& batch) {
            const std::lock_guard lock(records.mutex);
            check_records(records, batch);
        },
        "records-kernel");

    const std::string error = run([](distributed::MeshDevice& mesh_device) {
        enqueue(
            mesh_device,
            dm_program("matching_dm.cpp", CoreCoord{0, 0}, {{"N_ITERS", std::to_string(kMatchingIterations) + "u"}}));
        enqueue(
            mesh_device,
            dm_program(
                "records.cpp",
                CoreRange(CoreCoord{0, 0}, CoreCoord{kGridSide - 1, kGridSide - 1}),
                {{"N_ITERS", std::to_string(kRecordsIterations) + "u"}}));
        for (uint32_t group = 0; group <= kComputeGroup; group++) {
            const std::map<std::string, std::string> defines{
                {"N_ITERS", std::to_string(kIntegrityIterations) + "u"},
                {"GROUP", std::to_string(group)},
                {"COMPUTE_GROUP", std::to_string(kComputeGroup)}};
            if (group == kComputeGroup) {
                Program program = CreateProgram();
                CreateKernel(program, kKernelDir + "integrity.cpp", CoreCoord{0, 0}, ComputeConfig{.defines = defines});
                enqueue(mesh_device, std::move(program));
            } else {
                enqueue(mesh_device, dm_program("integrity.cpp", CoreCoord{0, 0}, defines));
            }
        }
    });
    matching_callback.reset();
    integrity_callback.reset();
    records_callback.reset();
    expect(error.empty(), error);
    matching.order.finish();
    integrity.order.finish();
    check_records_after(records);

    // Every matching call reached its types once per iteration on each processor.
    std::set<Lane> matching_lanes;
    for (const auto& [key, count] : matching.records) {
        matching_lanes.insert(key.first);
    }
    expect(
        matching_lanes.size() == std::size(kDmRiscs),
        fmt::format("matching calls came from {} processors", matching_lanes.size()));
    for (const Lane& lane : matching_lanes) {
        for (const auto& [call, types] : kReaches) {
            const uint64_t count = matching.records[{lane, call}];
            expect(
                count == kMatchingIterations,
                fmt::format("{}: {} records from a processor, want {}", call, count, kMatchingIterations));
        }
    }
    std::set<Lane> integrity_lanes;
    for (const auto& [key, count] : integrity.records) {
        integrity_lanes.insert(std::get<0>(key));
    }
    for (size_t index = 0; index < kCatalogSize; index++) {
        for (const Lane& lane : integrity_lanes) {
            const bool sends = std::get<2>(lane) <= api::Processor::NCRISC || sent_from_compute(index);
            for (const auto& [call, forms] : kSendsReach) {
                // A record holds one Largest at most, so it goes alone only.
                const bool sent = sends && !(index == kCatalogIndex<Largest> && (call == "pair" || call == "list"));
                const uint64_t count = integrity.records[{lane, index, call}];
                expect(
                    count == (sent ? kIntegrityIterations : 0),
                    fmt::format(
                        "struct {} sent as {}: {} records from a processor, want {}",
                        index,
                        call,
                        count,
                        sent ? kIntegrityIterations : 0));
            }
        }
    }
    expect(
        integrity_lanes.size() == std::size(kDmRiscs) + 3,
        fmt::format("integrity records came from {} processors", integrity_lanes.size()));

    fmt::print("[records] {}\n", failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
}
