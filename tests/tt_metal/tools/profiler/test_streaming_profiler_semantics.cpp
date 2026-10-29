// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DevicePublish's semantics on the device. The matching table lists, for every argument shape a kernel sends, the
// record types that receive it, and the test checks every type against every call, the ones that must not receive it
// too. The integrity sweep sends structs laid out every way a struct's fields can lie, every way a struct can be sent,
// each field filled from a seed the host recomputes, and compares each record with the struct it should be. Both check,
// through Batch::visit, that a record reaches each of its types exactly once, in its processor's order, and that visit
// and records<T>() agree. Run with TT_METAL_STREAMING_PROFILER=1. Exits nonzero on any mismatch.
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
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
#include <tt-metalium/experimental/streaming_profiler.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "records_types.hpp"

using namespace tt::tt_metal;
using namespace semantics_test;
namespace api = tt::tt_metal::experimental::streaming_profiler;
using api::Record;
using api::Variadic;

namespace {

constexpr uint32_t kMatchingIterations = 10, kIntegrityIterations = 30;
constexpr std::pair<DataMovementProcessor, NOC> kDmRiscs[] = {
    {DataMovementProcessor::RISCV_0, NOC::RISCV_0_default}, {DataMovementProcessor::RISCV_1, NOC::RISCV_1_default}};
// integrity.cpp's groups: each struct's, by catalog index, and which the compute kernels send.
constexpr uint32_t kGroups = 7;
constexpr uint32_t kComputeGroup = kGroups;
constexpr bool sent_from_compute(size_t index) { return index == 0 || index == 6 || index == 17; }
// How far after a Timestamp the record that carries it may publish: the kernel takes it just before.
constexpr uint64_t kPublishedWithin = 100'000;
const std::string kKernelDir = "tests/tt_metal/tools/profiler/kernels/";

std::mutex output_mutex;
bool failed = false;
void expect(bool ok, const std::string& what) {
    if (!ok) {
        const std::lock_guard lock(output_mutex);
        fmt::print("[semantics] FAIL {}\n", what);
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

template <typename R, typename B>
struct IndexIn;
template <typename R, typename... Rs>
struct IndexIn<R, api::Batch<Rs...>> {
    static constexpr size_t value = [] {
        constexpr bool same[] = {std::is_same_v<R, Rs>...};
        return size_t(std::ranges::find(same, true) - std::begin(same));
    }();
};
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
        constexpr size_t kType = IndexIn<R, api::Batch<Rs...>>::value;
        visited[kType].push_back(&record);
        on_record(record, kType);
    });
    const auto check_type = [&]<typename R>(std::type_identity<R>) {
        constexpr size_t kType = IndexIn<R, api::Batch<Rs...>>::value;
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
using Catalog = std::tuple<SEMANTICS_CATALOG>;
constexpr size_t kCatalogSize = std::tuple_size_v<Catalog>;
template <typename S>
constexpr size_t kCatalogIndex = [] {
    return []<size_t... I>(std::index_sequence<I...>) {
        constexpr bool same[] = {std::is_same_v<S, std::tuple_element_t<I, Catalog>>...};
        return size_t(std::ranges::find(same, true) - std::begin(same));
    }(std::make_index_sequence<kCatalogSize>());
}();

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
    bool words = true;
    for (uint32_t k = 0; k < 12; k++) {
        words = words && run.words[k] == uint32_t(mix(seed + 1 + k));
    }
    return words && back >= stamp_delta(iteration, 1) && back < stamp_delta(iteration, 1) + kPublishedWithin;
}
bool sent_as(const StatRows& rows, uint64_t seed, uint32_t iteration, const Timestamp& published) {
    const uint64_t back = published.device_cycles() - rows.at.device_cycles();
    bool ok = back >= stamp_delta(iteration, 1) && back < stamp_delta(iteration, 1) + kPublishedWithin;
    for (uint32_t r = 0; r < 20; r++) {
        ok = ok && rows.rows[r].a == uint32_t(mix(seed + 1 + 3 * r)) &&
             rows.rows[r].b == uint32_t(mix(seed + 2 + 3 * r)) && rows.rows[r].c == uint16_t(mix(seed + 3 + 3 * r));
    }
    return ok;
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

// One capture of every kernel: the matching calls, then each integrity group.
std::string run() {
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    std::string error;
    try {
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        const auto launch = [&](Program program) {
            distributed::MeshWorkload workload;
            workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
            distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        };
        Program matching = CreateProgram();
        for (const auto& [processor, noc] : kDmRiscs) {
            CreateKernel(
                matching,
                kKernelDir + "matching_dm.cpp",
                CoreCoord{0, 0},
                DataMovementConfig{
                    .processor = processor,
                    .noc = noc,
                    .defines = {{"N_ITERS", std::to_string(kMatchingIterations) + "u"}}});
        }
        launch(std::move(matching));
        for (uint32_t group = 0; group <= kComputeGroup; group++) {
            const std::map<std::string, std::string> defines{
                {"N_ITERS", std::to_string(kIntegrityIterations) + "u"},
                {"GROUP", std::to_string(group)},
                {"COMPUTE_GROUP", std::to_string(kComputeGroup)}};
            Program program = CreateProgram();
            if (group == kComputeGroup) {
                CreateKernel(program, kKernelDir + "integrity.cpp", CoreCoord{0, 0}, ComputeConfig{.defines = defines});
            } else {
                for (const auto& [processor, noc] : kDmRiscs) {
                    CreateKernel(
                        program,
                        kKernelDir + "integrity.cpp",
                        CoreCoord{0, 0},
                        DataMovementConfig{.processor = processor, .noc = noc, .defines = defines});
                }
            }
            launch(std::move(program));
        }
        distributed::Finish(cq);
    } catch (const std::exception& e) {
        error = e.what();
    }
    mesh_device->close();
    return error;
}

}  // namespace

int main() {
    MatchingResults matching;
    IntegrityResults integrity;
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
        "semantics-matching");
    auto integrity_callback = api::RegisterCallback(
        [&](const IntegrityBatch& batch) {
            const std::lock_guard lock(integrity.mutex);
            visit_checked(batch, [&](const auto& record, size_t type) {
                integrity.order.visit(record, type, std::string(record.metadata().name));
                check_integrity(integrity, record, type);
            });
        },
        "semantics-integrity");

    const std::string error = run();
    matching_callback.reset();
    integrity_callback.reset();
    expect(error.empty(), error);
    matching.order.finish();
    integrity.order.finish();

    // Every call reached its types once per iteration on each processor.
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

    fmt::print("[semantics] {}\n", failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
}
