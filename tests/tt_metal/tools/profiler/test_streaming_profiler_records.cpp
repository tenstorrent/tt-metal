// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Kernels on all five RISCs of a small grid publish DevicePublish records (structs, passed values, no data, no
// metadata, and one data type with several metadata types). One callback receives them through exact, variant-data and
// any-metadata record types and checks every value, its metadata, its source location and its core, and a second
// callback checks the times kernels took. Run with TT_METAL_STREAMING_PROFILER=1. Exits nonzero on any mismatch or
// missing record. With --errors it instead builds kernels that each break one record rule and checks that each build,
// or load, fails with that rule's message.
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <regex>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>

#include <fmt/format.h>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/experimental/streaming_profiler.hpp>
#include <tt_stl/overloaded.hpp>

#include "records_types.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace records_test;
using experimental::streaming_profiler::Arguments;
using experimental::streaming_profiler::Batch;
using experimental::streaming_profiler::Core;
using experimental::streaming_profiler::Processor;
using experimental::streaming_profiler::Record;
using experimental::streaming_profiler::RegisterCallback;
using experimental::streaming_profiler::Zone;

namespace {

constexpr uint32_t kGridSide = 2, kIterations = 20;
constexpr uint64_t kCores = kGridSide * kGridSide;
// The records one call site sends from one RISC across the grid.
constexpr uint64_t kRecordsPerSite = kCores * kIterations;
const std::string kKernelDir = "tests/tt_metal/tools/profiler/kernels/";
constexpr std::pair<DataMovementProcessor, NOC> kDmRiscs[] = {
    {DataMovementProcessor::RISCV_0, NOC::RISCV_0_default}, {DataMovementProcessor::RISCV_1, NOC::RISCV_1_default}};
// Each kind's count is its call sites times the RISCs they run on. "tiles" and "start" run on all five RISCs and every
// other site on the two DM RISCs.
const std::map<std::string, uint64_t> kExpectedCounts{
    {"records-iteration zones", 2 * kRecordsPerSite},              // "records-iteration"
    {"long zones", 2 * kCores},                                    // "long", once per kernel
    {"NocEvent + NocSite", 2 * kRecordsPerSite},                   // "noc_async_read"
    {"NameValue + Name", (5 + 2 * 2) * kRecordsPerSite},           // "tiles", "mixed", "bf16"
    {"no data + From", 2 * 2 * kRecordsPerSite},                   // "helper", "lambda"
    {"Burst + Name", 2 * kRecordsPerSite},                         // "burst"
    {"RowValue + Name", 3 * 2 * kRecordsPerSite},                  // "row", "ids", "tag"
    {"Bf16 array + Name", 2 * kRecordsPerSite},                    // "bf16 row"
    {"uint32_t[3] + Name", 2 * kRecordsPerSite},                   // "ids"
    {"uint32_t[4] + Name", 0},                                     // no site sends four uint32_t
    {"Ragged + Name", 2 * kRecordsPerSite},                        // "ragged"
    {"Hole + Name", 2 * kRecordsPerSite},                          // "hole"
    {"no data + Hole", 2 * kRecordsPerSite},                       // the site with Hole metadata
    {"TightBits + Name", 2 * kRecordsPerSite},                     // "tight bits"
    {"LaneFlags + Name", 2 * kRecordsPerSite},                     // "flags"
    {"TaggedFlags + Name", 2 * kRecordsPerSite},                   // "tagged flags"
    {"Packed + Name", 2 * kRecordsPerSite},                        // "packed"
    {"Wide + Name", 2 * kRecordsPerSite},                          // "wide"
    {"WithEmpty + Name", 2 * kRecordsPerSite},                     // "with empty"
    {"Deep + Name", 2 * kRecordsPerSite},                          // "deep"
    {"Skewed + Name", 2 * kRecordsPerSite},                        // "skewed"
    {"BitsValue + Name", 6 * 2 * kRecordsPerSite},                 // each site sending a Hole, TightBits or LaneFlags
    {"no data + LaneFlags", 2 * kRecordsPerSite},                  // the site with LaneFlags metadata
    {"TileStats + Name", 2 * kRecordsPerSite},                     // "tile stats"
    {"no data + Name", 5 * kRecordsPerSite},                       // "start"
    {"FormatValue + Format", 2 * kRecordsPerSite},                 // EXAMPLE_PRINT
    {"Sample + PlotStyle", 2 * kRecordsPerSite},                   // "l1-used"
    {"Sample + no metadata", 2 * kRecordsPerSite},                 // the Sample sent without metadata
    {"Sample, any metadata", 2 * 2 * kRecordsPerSite},             // both Sample sites
    {"Stamped + Name", 2 * 2 * kRecordsPerSite + 2 * 2 * kCores},  // "stamped", "since start", two after the loop
    {"TimeValue + Name", 2 * kRecordsPerSite},                     // "now"
    {"Interval + Name", 2 * kRecordsPerSite},                      // "interval"
    {"TypedValue + Name", 2 * kRecordsPerSite},                    // "noc type"
    {"IdsValue + Name", 2 * kRecordsPerSite},                      // "ids"
};
// A record publishes within this many ticks of any time it carries.
constexpr uint64_t kPublishWithinTicks = 1'000'000'000;
// More ticks than the whole kernel takes, including its long zone.
constexpr uint64_t kKernelTicksBound = uint64_t{1} << 34;
// A long zone starts within this many ticks of the time its processor takes just before opening it.
constexpr uint64_t kLongZoneStartWithinTicks = 1000;

using NameValue = std::variant<uint8_t, uint32_t, uint64_t, float, bool, std::string_view, Bf16>;
using FormatValue = std::variant<std::string_view, uint32_t, uint64_t>;
using RowValue = std::variant<uint32_t, std::span<const float>, std::span<const uint32_t>, std::span<const char>>;
using BitsValue = std::variant<uint32_t, Hole, LaneFlags, TightBits>;
using TypedValue = std::variant<uint8_t, NocType>;
using IdsValue = std::variant<std::span<const uint32_t>, std::array<uint32_t, 3>>;
using TimeValue = std::variant<uint32_t, Timestamp>;

template <typename T, typename V>
const T* value_at(std::span<const V> values, size_t k) {
    return k < values.size() ? std::get_if<T>(&values[k]) : nullptr;
}

bool sent_as_row(std::span<const RowValue> values) {
    const auto* iteration = value_at<uint32_t>(values, 0);
    const auto* row = value_at<std::span<const float>>(values, 1);
    return values.size() == 2 && iteration && row && row->size() == 4 && (*row)[0] == float(*iteration) &&
           (*row)[3] == float(*iteration) + 0.75f;
}

std::optional<uint64_t> sent_in(const Hole& hole) {
    const uint32_t a = hole.a, b = hole.b;
    return hole.iteration < kIterations && a == hole.iteration % 8 && b == hole.iteration % 32 && hole.name == "Ada"
               ? std::optional(hole.iteration)
               : std::nullopt;
}
std::optional<uint64_t> sent_in(const LaneFlags& flags) {
    const uint32_t kind = flags.kind, lane = flags.lane, count = flags.count;
    return count < kIterations && kind == count % 8 && lane == count % 32 ? std::optional<uint64_t>(count)
                                                                          : std::nullopt;
}
std::optional<uint64_t> sent_in(const TightBits& tight) {
    const uint32_t a = tight.a, b = tight.b;
    return tight.iteration < kIterations && a == tight.iteration % 256 && b == (tight.iteration + 1) % 256
               ? std::optional<uint64_t>(tight.iteration)
               : std::nullopt;
}

struct Results {
    std::mutex mutex;
    std::optional<Record<Arguments<RowValue>, Name>> kept_row;
    std::map<std::string, uint64_t> counts;
    std::map<std::pair<std::string, CoreCoord>, uint64_t> core_counts;
    using Launch = std::tuple<size_t, size_t, Processor, uint32_t>;
    std::map<Launch, uint64_t> kernel_starts, long_starts, long_opens;
    template <typename R>
    static Launch launch(const R& record) {
        const Core core = record.core();
        return {core.physical.x, core.physical.y, core.processor, record.runtime_id()};
    }
    bool failed = false;

    template <typename... Ts>
    void check_order(const Batch<Ts...>& batch) {
        size_t visited = 0;
        std::map<std::tuple<size_t, size_t, Processor>, uint64_t> last;
        const auto in_order = [&](const Core& core, uint64_t written) {
            ++visited;
            uint64_t& previous = last[{core.physical.x, core.physical.y, core.processor}];
            expect(written >= previous, "visit() in each processor's order");
            previous = written;
        };
        batch.visit(ttsl::overloaded{
            [&](const Zone& zone) { in_order(zone.core(), zone.payload().end.device_cycles()); },
            [&](const auto& record) { in_order(record.core(), record.published_at().device_cycles()); }});
        expect(visited == (batch.template records<Ts>().size() + ...), "visit() reaches every record");
    }

    static bool before(const Timestamp& at, const Timestamp& published, uint64_t ticks = kPublishWithinTicks) {
        return at.device_cycles() <= published.device_cycles() && at.host_cycles() <= published.host_cycles() &&
               published.device_cycles() - at.device_cycles() < ticks;
    }

    void expect(bool ok, const std::string& what) {
        if (!ok) {
            fmt::print("[records] FAIL {}\n", what);
            failed = true;
        }
    }
    template <typename R>
    void common(const std::string& kind, const R& record) {
        const Core core = record.core();
        ++counts[kind];
        ++core_counts[{kind, core.logical}];
        expect(
            core.logical.x < kGridSide && core.logical.y < kGridSide,
            fmt::format("{}: core {} outside the grid", kind, core.logical.str()));
        const bool dm = core.processor == Processor::BRISC || core.processor == Processor::NCRISC;
        const std::string_view file = record.location().file;
        expect(!record.location().function.empty(), fmt::format("{}: no function", kind));
        expect(
            file.ends_with(dm ? "records_dm.cpp" : "records_compute.cpp") && record.location().line > 0,
            fmt::format("{}: location {}:{}", kind, file, record.location().line));
    }
    template <typename R, typename... Ts, typename F>
    void each(const Batch<Ts...>& batch, const std::string& kind, F check) {
        for (const R& record : batch.template records<R>()) {
            common(kind, record);
            const Core core = record.core();
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

std::string run(const std::function<Program()>& make) {
    auto mesh_device =
        distributed::MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1);
    std::string error;
    try {
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), make());
        distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        distributed::Finish(cq);
    } catch (const std::exception& e) {
        error = e.what();
    }
    mesh_device->close();
    return error;
}

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
constexpr std::string_view kOverLimit =
    "this record is over the 504-byte limit (500 with Timestamp fields); split it into several records";
constexpr ExpectedError kErrors[] = {
    {1, true, false, "strings sent with DevicePublish need an _s suffix: \"text\"_s"},
    {2, true, false, "strings sent with DevicePublish need an _s suffix: \"text\"_s"},
    {3, true, false, "strings sent with DevicePublish need an _s suffix: \"text\"_s"},
    {4, true, false, "types with a constructor, private field or virtual function are not supported"},
    {5, true, false, "templates, function-local types and types in an anonymous namespace are not supported"},
    {6, true, false, "templates, function-local types and types in an anonymous namespace are not supported"},
    {7, true, false, "const and reference members are not supported"},
    {8, true, false, "enums without a fixed underlying type are not supported; use enum E : uint8_t"},
    {9, true, false, kUnsupportedType},
    {10, true, false, "arrays of arrays are not supported; put the inner array in a struct"},
    {11, true, false, "empty arrays are not supported"},
    {12, true, false, "templates, function-local types and types in an anonymous namespace are not supported"},
    {13, true, false, kOverLimit},
    {14, true, false, "unknown DevicePublish option; the only one is kPublishImmediately"},
    {15, true, false, "types with a custom copy, move or destructor are not supported"},
    {16, true, false, kUnsupportedType},
    {17, true, false, "unions are not supported; send the member it holds"},
    {18, true, false, "unions are not supported; send the member it holds"},
    {19, true, false, "types with a constructor, private field or virtual function are not supported"},
    {20, false, true, "two kernels send different types named Clash"},
    {21, false, false, "records_test::LongField has different fields in kernels than on the host"},
    {22, false, false, "field 2 of records_test::LateField lies past the bytes a record carries"},
};

// Each case is one build of record_errors.cpp, on both data-movement RISCs when the case sets on_both_dm_riscs. A build
// must fail with one error, located at the case's call. A load must fail with the case's message.
int expect_errors() {
    // A callback that takes LongField, so the host compares its fields with the kernel's when the kernel loads.
    auto callback = RegisterCallback([](const Batch<Record<LongField, Name>>&) {}, "records-errors");
    int failed = 0;
    for (const auto& [case_id, fails_at_build, on_both_dm_riscs, message] : kErrors) {
        const std::string failure = run([case_id, on_both_dm_riscs] {
            Program program = CreateProgram();
            const std::map<std::string, std::string> defines{{"RECORD_ERROR", std::to_string(case_id)}};
            for (const auto& [processor, noc] : std::span(kDmRiscs).first(on_both_dm_riscs ? 2 : 1)) {
                CreateKernel(
                    program,
                    kKernelDir + "record_errors.cpp",
                    CoreCoord{0, 0},
                    DataMovementConfig{.processor = processor, .noc = noc, .defines = defines});
            }
            return program;
        });
        const size_t e = failure.find(": error: ");
        const bool ok = failure.find(message) != std::string::npos &&
                        (!fails_at_build || (e != std::string::npos && e == failure.rfind(": error: ") &&
                                             std::regex_search(failure, std::regex(R"(record_errors\.cpp:\d+:\d+:)"))));
        fmt::print("[records] error {}: {}\n", case_id, ok ? "PASS" : "FAIL, got: " + failure);
        failed |= ok ? 0 : 1;
    }
    return failed;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--errors") {
        return expect_errors();
    }
    Results results;
    auto callback = RegisterCallback(
        [&](const Batch<
            Record<NocEvent, NocSite>,
            Record<Arguments<NameValue>, Name>,
            Record<TileStats, Name>,
            Record<Burst, Name>,
            Record<Arguments<RowValue>, Name>,
            Record<Arguments<std::span<const Bf16>>, Name>,
            Record<Arguments<std::array<uint32_t, 3>>, Name>,
            Record<Arguments<std::array<uint32_t, 4>>, Name>,
            Record<Arguments<IdsValue>, Name>,
            Record<Arguments<Ragged>, Name>,
            Record<Hole, Name>,
            Record<void, Hole>,
            Record<TightBits, Name>,
            Record<LaneFlags, Name>,
            Record<TaggedFlags, Name>,
            Record<Packed, Name>,
            Record<Wide, Name>,
            Record<WithEmpty, Name>,
            Record<Deep, Name>,
            Record<Skewed, Name>,
            Record<Arguments<BitsValue>, Name>,
            Record<Arguments<TypedValue>, Name>,
            Record<void, LaneFlags>,
            Record<void, From>,
            Record<void, Name>,
            Record<Arguments<FormatValue>, Format>,
            Record<Sample, PlotStyle>,
            Record<Sample, void>,
            Record<Sample>,
            Zone>& batch) {
            std::lock_guard lock(results.mutex);
            for (const Zone& zone : batch.records<Zone>()) {
                if (zone.metadata().name == "records-iteration") {
                    results.common("records-iteration zones", zone);
                } else if (zone.metadata().name == "long") {
                    results.common("long zones", zone);
                    const auto& [start, end] = zone.payload();
                    const uint64_t ticks = end.device_cycles() - start.device_cycles();
                    results.expect(
                        ticks >= kLongZoneTicks && ticks < kLongZoneTicks + (1u << 20) &&
                            start.host_cycles() < end.host_cycles(),
                        fmt::format("long zone of {} ticks", ticks));
                    results.long_starts[Results::launch(zone)] = start.device_cycles();
                }
            }
            results.each<Record<NocEvent, NocSite>>(batch, "NocEvent + NocSite", [](const auto& record) {
                const auto& payload = record.payload();
                return record.location().function.ends_with("kernel_main()") && payload.x < kIterations &&
                       payload.y == (record.core().processor == Processor::BRISC ? 0 : 1) &&
                       payload.bytes == uint32_t(payload.x) * 64 && record.metadata().api == "noc_async_read" &&
                       record.metadata().type == NocType::Read && record.metadata().color == 0xFF8000;
            });
            // Only the Name records whose values are all plain values or Bf16s reach NameValue. The struct records go
            // to their own types.
            results.each<Record<Arguments<NameValue>, Name>>(batch, "NameValue + Name", [](const auto& record) {
                const auto& payload = record.payload();
                const std::string_view name = record.metadata().name;
                if (name == "tiles") {
                    const auto* count = value_at<uint32_t>(payload, 0);
                    const auto* ratio = value_at<float>(payload, 1);
                    return payload.size() == 2 && count && ratio && *count < kIterations &&
                           *ratio == float(*count) * 0.5f;
                }
                if (name == "mixed") {
                    const auto* iteration = value_at<uint8_t>(payload, 0);
                    const auto* who = value_at<std::string_view>(payload, 1);
                    const auto* big = value_at<uint64_t>(payload, 2);
                    const auto* odd = value_at<bool>(payload, 3);
                    return payload.size() == 4 && iteration && who && big && odd && *iteration < kIterations &&
                           *who == "Ada" && *big == uint64_t(*iteration) << 33 && *odd == (*iteration % 2 == 1);
                }
                const auto* bf16 = value_at<Bf16>(payload, 0);
                const auto* count = value_at<uint32_t>(payload, 1);
                return name == "bf16" && payload.size() == 2 && bf16 && count && bf16->bits == *count;
            });
            results.each<Record<void, From>>(batch, "no data + From", [](const auto& record) {
                const std::string_view function = record.location().function;
                return record.metadata().tag == "helper"
                           ? function.ends_with("publish_from_helper()")
                           : record.metadata().tag == "lambda" && function.starts_with("kernel_main()::<lambda");
            });
            results.each<Record<Burst, Name>>(batch, "Burst + Name", [](const auto& record) {
                const auto& payload = record.payload();
                const uint32_t iteration = payload.sizes[0];
                return record.metadata().name == "burst" && iteration < kIterations &&
                       payload.sizes[1] == iteration + 1 && payload.sizes[2] == iteration + 2 &&
                       payload.sizes[3] == iteration + 3 &&
                       std::string_view(payload.tag) == fmt::format("rd{}", iteration % 10);
            });
            results.each<Record<Arguments<RowValue>, Name>>(batch, "RowValue + Name", [&](const auto& record) {
                const auto& payload = record.payload();
                const std::string_view name = record.metadata().name;
                if (name == "row") {
                    if (!results.kept_row) {
                        results.kept_row = record;
                        const auto* row = value_at<std::span<const float>>(payload, 1);
                        const auto* kept = value_at<std::span<const float>>(results.kept_row->payload(), 1);
                        results.expect(row && kept && row->data() != kept->data(), "a kept record owns its array");
                    }
                    return sent_as_row(payload);
                }
                if (name == "ids") {
                    const auto* ids = value_at<std::span<const uint32_t>>(payload, 0);
                    return payload.size() == 1 && ids && ids->size() == 3 && (*ids)[0] < kIterations &&
                           (*ids)[1] == 2 * (*ids)[0] && (*ids)[2] == 3 * (*ids)[0];
                }
                const auto* tag = value_at<std::span<const char>>(payload, 0);
                return name == "tag" && payload.size() == 1 && tag && tag->size() == 4 && (*tag)[0] == 'i' &&
                       (*tag)[1] == 'd' && (*tag)[3] == 0;
            });
            results.each<Record<Arguments<std::span<const Bf16>>, Name>>(
                batch, "Bf16 array + Name", [](const auto& record) {
                    const auto& payload = record.payload();
                    return record.metadata().name == "bf16 row" && payload.size() == 1 && payload[0].size() == 2 &&
                           payload[0][1].bits == payload[0][0].bits + 1;
                });
            // "ids" sends three uint32_t, so only the array of three gets it.
            results.each<Record<Arguments<std::array<uint32_t, 3>>, Name>>(
                batch, "uint32_t[3] + Name", [](const auto& record) {
                    const auto& payload = record.payload();
                    if (payload.size() != 1) {
                        return false;
                    }
                    const std::array<uint32_t, 3>& ids = payload[0];
                    return record.metadata().name == "ids" && ids[0] < kIterations && ids[1] == 2 * ids[0] &&
                           ids[2] == 3 * ids[0];
                });
            results.each<Record<Arguments<std::array<uint32_t, 4>>, Name>>(
                batch, "uint32_t[4] + Name", [](const auto&) { return false; });
            // The exact-length array beats the span listed before it.
            results.each<Record<Arguments<IdsValue>, Name>>(batch, "IdsValue + Name", [](const auto& record) {
                const auto* ids = value_at<std::array<uint32_t, 3>>(record.payload(), 0);
                return record.metadata().name == "ids" && record.payload().size() == 1 && ids &&
                       (*ids)[2] == 3 * (*ids)[0];
            });
            results.each<Record<Arguments<Ragged>, Name>>(batch, "Ragged + Name", [](const auto& record) {
                const auto& payload = record.payload();
                return record.metadata().name == "ragged" && payload.size() == 2 && payload[0].b < kIterations &&
                       payload[1].b == payload[0].b && payload[0].a == (payload[0].b << 8 | 1) &&
                       payload[1].a == (payload[1].b << 8 | 2);
            });
            results.each<Record<Hole, Name>>(batch, "Hole + Name", [](const auto& record) {
                return record.metadata().name == "hole" && sent_in(record.payload()).has_value();
            });
            results.each<Record<void, Hole>>(batch, "no data + Hole", [](const auto& record) {
                const Hole& hole = record.metadata();
                return hole.iteration == 7 && hole.name == "Ada" && hole.a == 5 && hole.b == 17;
            });
            results.each<Record<TightBits, Name>>(batch, "TightBits + Name", [](const auto& record) {
                return record.metadata().name == "tight bits" && sent_in(record.payload()).has_value();
            });
            results.each<Record<LaneFlags, Name>>(batch, "LaneFlags + Name", [](const auto& record) {
                return record.metadata().name == "flags" && sent_in(record.payload()).has_value();
            });
            results.each<Record<TaggedFlags, Name>>(batch, "TaggedFlags + Name", [](const auto& record) {
                const auto& payload = record.payload();
                const uint32_t kind = payload.kind;
                const int32_t delta = payload.delta;
                return record.metadata().name == "tagged flags" && payload.name == "Ada" && delta >= -8 && delta < 8 &&
                       kind == uint32_t(delta + 8) % 8;
            });
            results.each<Record<Packed, Name>>(batch, "Packed + Name", [](const auto& record) {
                const auto& payload = record.payload();
                return record.metadata().name == "packed" && payload.value == payload.tag * 3u;
            });
            results.each<Record<Wide, Name>>(batch, "Wide + Name", [](const auto& record) {
                const auto& payload = record.payload();
                return record.metadata().name == "wide" && payload.a < kIterations && payload.b == payload.a + 1;
            });
            results.each<Record<WithEmpty, Name>>(batch, "WithEmpty + Name", [](const auto& record) {
                return record.metadata().name == "with empty" && record.payload().value < kIterations;
            });
            results.each<Record<Deep, Name>>(batch, "Deep + Name", [](const auto& record) {
                const auto& payload = record.payload();
                return record.metadata().name == "deep" && payload.x < kIterations && payload.y == payload.x + 1 &&
                       payload.z == payload.x + 2;
            });
            results.each<Record<Skewed, Name>>(batch, "Skewed + Name", [](const auto& record) {
                const auto& payload = record.payload();
                const uint32_t a = payload.a, b = payload.b;
                return record.metadata().name == "skewed" && b == (a + 1) % 256;
            });
            // An enum argument takes its own alternative rather than its underlying integer's, so NameValue, which
            // lists uint8_t but no enum, never gets "noc type".
            results.each<Record<Arguments<TypedValue>, Name>>(batch, "TypedValue + Name", [](const auto& record) {
                const auto& payload = record.payload();
                const auto* type = value_at<NocType>(payload, 0);
                const auto* iteration = value_at<uint8_t>(payload, 1);
                return record.metadata().name == "noc type" && payload.size() == 2 && type && *type == NocType::Write &&
                       iteration && *iteration < kIterations;
            });
            // Struct data is one value too, and a struct passed as a value goes beside the iteration it was sent in.
            results.each<Record<Arguments<BitsValue>, Name>>(batch, "BitsValue + Name", [](const auto& record) {
                const auto& payload = record.payload();
                std::optional<uint64_t> number, iteration;
                for (const BitsValue& value : payload) {
                    std::visit(
                        ttsl::overloaded{
                            [&](uint32_t plain) { number = plain; },
                            [&](const auto& data) { iteration = sent_in(data); }},
                        value);
                }
                return iteration &&
                       (record.metadata().name.ends_with(" value") ? payload.size() == 2 && number == iteration
                                                                   : payload.size() == 1);
            });
            results.each<Record<void, LaneFlags>>(batch, "no data + LaneFlags", [](const auto& record) {
                const LaneFlags& flags = record.metadata();
                return flags.kind == 5 && flags.lane == 17 && flags.count == 1000;
            });
            results.each<Record<TileStats, Name>>(batch, "TileStats + Name", [](const auto& record) {
                const auto& payload = record.payload();
                return record.metadata().name == "tile stats" && payload.tiles < kIterations &&
                       payload.bytes == payload.tiles * 2048 && payload.cycles == payload.tiles * 100 &&
                       payload.stalls == payload.tiles % 3;
            });
            results.each<Record<void, Name>>(
                batch, "no data + Name", [](const auto& record) { return record.metadata().name == "start"; });
            results.each<Record<Arguments<FormatValue>, Format>>(batch, "FormatValue + Format", [](const auto& record) {
                const auto& payload = record.payload();
                const auto* who = value_at<std::string_view>(payload, 0);
                const auto* iteration = value_at<uint32_t>(payload, 1);
                const auto* big = value_at<uint64_t>(payload, 2);
                return payload.size() == 3 && who && iteration && big && *who == "Ada" && *iteration < kIterations &&
                       *big == uint64_t(*iteration) << 33 &&
                       record.metadata().format == "hello {} iteration {} big {}" &&
                       record.metadata().arguments == R"("Ada"_s, i, uint64_t(i) << 33)";
            });
            results.each<Record<Sample, PlotStyle>>(batch, "Sample + PlotStyle", [](const auto& record) {
                const auto& payload = record.payload();
                return payload.value % 1000 == 0 && payload.value / 1000 < kIterations &&
                       record.metadata().name == "l1-used" && record.metadata().format == PlotFormat::Memory &&
                       record.metadata().color == 0x00FF00;
            });
            results.each<Record<Sample, void>>(batch, "Sample + no metadata", [](const auto& record) {
                const auto& payload = record.payload();
                return payload.value >= -5 && payload.value < int64_t(kIterations) - 5;
            });
            results.each<Record<Sample>>(batch, "Sample, any metadata", [](const auto&) { return true; });
            results.check_order(batch);
        },
        "records-test");
    auto times_callback = RegisterCallback(
        [&](const Batch<Record<Stamped, Name>, Record<Arguments<TimeValue>, Name>, Record<Interval, Name>>& batch) {
            std::lock_guard lock(results.mutex);
            results.each<Record<Stamped, Name>>(batch, "Stamped + Name", [&](const auto& record) {
                const auto& payload = record.payload();
                const std::string_view name = record.metadata().name;
                const uint64_t at = payload.at.device_cycles();
                if (name == "stamped") {
                    return payload.iteration < kIterations && Results::before(payload.at, record.published_at());
                }
                if (name == "since start") {
                    // Every record that carries the kernel's start time carries the same value, including the last,
                    // which the kernel sends after the long zone.
                    return results.kernel_starts.try_emplace(Results::launch(record), at).first->second == at &&
                           payload.iteration <= kIterations &&
                           Results::before(payload.at, record.published_at(), kKernelTicksBound);
                }
                if (name == "before long") {
                    results.long_opens[Results::launch(record)] = at;
                    return record.published_at().device_cycles() - at > uint64_t{1} << 32 &&
                           Results::before(payload.at, record.published_at(), kKernelTicksBound);
                }
                return false;
            });
            results.each<Record<Arguments<TimeValue>, Name>>(batch, "TimeValue + Name", [](const auto& record) {
                const auto& payload = record.payload();
                const auto* iteration = value_at<uint32_t>(payload, 0);
                const auto* at = value_at<Timestamp>(payload, 1);
                return record.metadata().name == "now" && payload.size() == 2 && iteration && at &&
                       *iteration < kIterations && Results::before(*at, record.published_at());
            });
            results.each<Record<Interval, Name>>(batch, "Interval + Name", [](const auto& record) {
                const auto& payload = record.payload();
                return record.metadata().name == "interval" && Results::before(payload.start, payload.end) &&
                       Results::before(payload.end, record.published_at());
            });
            results.check_order(batch);
        },
        "records-times");

    const std::string error = run([] {
        Program program = CreateProgram();
        const CoreRange cores(CoreCoord{0, 0}, CoreCoord{kGridSide - 1, kGridSide - 1});
        const std::map<std::string, std::string> defines{{"N_ITERS", std::to_string(kIterations) + "u"}};
        for (const auto& [processor, noc] : kDmRiscs) {
            CreateKernel(
                program,
                kKernelDir + "records_dm.cpp",
                cores,
                DataMovementConfig{.processor = processor, .noc = noc, .defines = defines});
        }
        CreateKernel(program, kKernelDir + "records_compute.cpp", cores, ComputeConfig{.defines = defines});
        return program;
    });
    callback.reset();
    times_callback.reset();
    results.expect(error.empty(), error);

    results.expect(
        results.kept_row && sent_as_row(results.kept_row->payload()), "a record with an array kept past its callback");
    for (const auto& [launch, start] : results.long_starts) {
        const auto open = results.long_opens.find(launch);
        results.expect(
            open != results.long_opens.end() && open->second <= start &&
                start - open->second < kLongZoneStartWithinTicks,
            "long zone's start");
    }
    for (const auto& [kind, want] : kExpectedCounts) {
        const uint64_t seen = results.counts[kind];
        results.expect(seen == want, fmt::format("{}: {} records, want {}", kind, seen, want));
        for (uint32_t x = 0; x < kGridSide; x++) {
            for (uint32_t y = 0; y < kGridSide; y++) {
                const CoreCoord core{x, y};
                const uint64_t seen_on_core = results.core_counts[{kind, core}];
                results.expect(
                    seen_on_core == want / kCores,
                    fmt::format("{}: {} records from core {}, want {}", kind, seen_on_core, core.str(), want / kCores));
            }
        }
    }
    fmt::print("[records] {}\n", results.failed ? "FAIL" : "PASS");
    return results.failed ? 1 : 0;
}
