// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Kernels on all five RISCs of a small grid publish DevicePublish records (structs, arguments, no payload, no metadata,
// and one payload type with several metadata types). One callback receives them through single-type, tuple, variant and
// Variadic record types and checks every value, its metadata, its source location and its core, and a second callback
// checks the times kernels took. Run with TT_METAL_STREAMING_PROFILER=1. Exits nonzero on any mismatch or missing
// record. With --errors it instead builds kernels that each break one record rule and checks that each build, or load,
// fails with that rule's message.
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
using experimental::streaming_profiler::Batch;
using experimental::streaming_profiler::Core;
using experimental::streaming_profiler::Processor;
using experimental::streaming_profiler::Record;
using experimental::streaming_profiler::RegisterCallback;
using experimental::streaming_profiler::TimestampedData;
using experimental::streaming_profiler::Variadic;
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
    {"NameValue + Name", (5 + 3 * 2 + 5) * kRecordsPerSite},       // "tiles", "mixed", "bf16", "no inline", "start"
    {"uint64_t + Name", 2 * kRecordsPerSite},                      // "no inline"
    {"(uint32_t, float) + Name", 5 * kRecordsPerSite},             // "tiles"
    {"no payload + From", 2 * 2 * kRecordsPerSite},                // "helper", "lambda"
    {"Burst + Name", 2 * kRecordsPerSite},                         // "burst"
    {"RowValue + Name", (3 * 2 + 5) * kRecordsPerSite},            // "row", "ids", "tag", "start"
    {"(number, float array) + Name", 2 * kRecordsPerSite},         // "row"
    {"Bf16 array + Name", 2 * kRecordsPerSite},                    // "bf16 row"
    {"uint32_t[3] + Name", 2 * kRecordsPerSite},                   // "ids"
    {"uint32_t[4] + Name", 0},                                     // no site sends four uint32_t
    {"(Ragged, Ragged) + Name", 2 * kRecordsPerSite},              // "ragged"
    {"Hole + Name", 2 * kRecordsPerSite},                          // "hole"
    {"no payload + Hole", 2 * kRecordsPerSite},                    // the site with Hole metadata
    {"TightBits + Name", 2 * kRecordsPerSite},                     // "tight bits"
    {"LaneFlags + Name", 2 * kRecordsPerSite},                     // "flags"
    {"TaggedFlags + Name", 2 * kRecordsPerSite},                   // "tagged flags"
    {"Packed + Name", 2 * kRecordsPerSite},                        // "packed"
    {"Wide + Name", 2 * kRecordsPerSite},                          // "wide"
    {"WithEmpty + Name", 2 * kRecordsPerSite},                     // "with empty"
    {"Deep + Name", 2 * kRecordsPerSite},                          // "deep"
    {"Skewed + Name", 2 * kRecordsPerSite},                        // "skewed"
    {"BitsValue + Name", (6 * 2 + 5) * kRecordsPerSite},           // each Hole, TightBits or LaneFlags site, "start"
    {"no payload + LaneFlags", 2 * kRecordsPerSite},               // the site with LaneFlags metadata
    {"TileStats + Name", 2 * kRecordsPerSite},                     // "tile stats"
    {"no payload + Name", 5 * kRecordsPerSite},                    // "start"
    {"FormatValue + Format", 2 * 2 * kRecordsPerSite},             // EXAMPLE_PRINT, "iteration done"
    {"TimestampedData", 2 * kRecordsPerSite},                      // "pair"
    {"Sample + PlotStyle", 2 * kRecordsPerSite},                   // "l1-used"
    {"Sample + no metadata", 2 * 2 * kRecordsPerSite},             // both Samples sent without metadata
    {"Stamped + Name", 2 * 3 * kRecordsPerSite + 2 * 3 * kCores},  // "stamped", "stamped no inline", "since start",
                                                                   // and three after the loop
    {"Big + Name", 2 * 2 * kRecordsPerSite},                       // "big", "big no inline"
    {"BigStamped + Name", 2 * kRecordsPerSite + 2 * kCores},       // "big stamped", and one after the loop
    {"Scalars + Name", 2 * kRecordsPerSite},                       // "scalars"
    {"uint32_t + InTemplate", 2 * kRecordsPerSite},                // publish_from_template<uint32_t>
    {"uint64_t + InTemplate", 2 * kRecordsPerSite},                // publish_from_template<uint64_t>
    {"uint16_t + InTemplate", 2 * kRecordsPerSite},                // Publisher<uint16_t>::publish
    {"TimeValue + Name", (2 + 5) * kRecordsPerSite},               // "now", "start"
    {"(uint32_t, Timestamp) + Name", 2 * kRecordsPerSite},         // "now"
    {"Interval + Name", 2 * kRecordsPerSite},                      // "interval"
    {"(NocType, uint8_t) + Name", 2 * kRecordsPerSite},            // "noc type"
    {"IdsValue + Name", 2 * kRecordsPerSite},                      // "ids"
};
// A record publishes within this many ticks of any time it carries.
constexpr uint64_t kPublishWithinTicks = 1'000'000'000;
// More ticks than the whole kernel takes, including its long zone.
constexpr uint64_t kKernelTicksBound = uint64_t{1} << 34;
// A long zone starts within this many ticks of the time its processor takes just before opening it.
constexpr uint64_t kLongZoneStartWithinTicks = 1000;

using NameValue = std::variant<uint8_t, uint32_t, uint64_t, float, bool, Bf16>;
using FormatValue = std::variant<uint32_t, uint64_t>;
using RowValue = std::variant<uint32_t, std::span<const float>, std::span<const uint32_t>, std::span<const char>>;
using BitsValue = std::variant<uint32_t, Hole, LaneFlags, TightBits>;
using IdsValue = std::variant<std::span<const uint32_t>, std::array<uint32_t, 3>>;
// "row" as exact arguments: its iteration as either number type, then its array.
using RowArguments = std::tuple<std::variant<uint32_t, float>, std::span<const float>>;
using Scalars = std::tuple<int8_t, int16_t, int32_t, uint16_t, double, char>;
using TimeValue = std::variant<uint32_t, Timestamp>;

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
bool sent_as_big(std::span<const uint32_t> words) {
    const uint32_t iteration = words[0] / 1000;
    for (size_t w = 0; w < words.size(); w++) {
        if (words[w] != iteration * 1000 + w) {
            return false;
        }
    }
    return iteration < kIterations;
}
// A call site in a template names the template, without the template arguments of the instantiation.
template <typename R>
bool sent_in_template(const R& record, std::string_view name, std::string_view function) {
    const std::string_view at = record.location().function;
    return record.metadata().name == name && at.ends_with(function) && at.find("[with") == std::string_view::npos;
}
bool sent_as_row(const RowArguments& arguments) {
    const auto& [iteration, row] = arguments;
    return std::holds_alternative<uint32_t>(iteration) && sent_as_row(std::get<uint32_t>(iteration), row);
}

std::optional<uint64_t> sent_in(const Hole& hole) {
    const uint32_t a = hole.a, b = hole.b;
    return hole.iteration < kIterations && a == hole.iteration % 8 && b == hole.iteration % 32 &&
                   std::string_view(hole.name) == "Ada"
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
    std::optional<Record<Variadic<RowValue>, Name>> kept_row;
    std::optional<Record<RowArguments, Name>> kept_row_arguments;
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
    {14, true, false, "unknown DevicePublish option; the options are kPublishBatched and kNoInline"},
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
            Record<Variadic<NameValue>, Name>,
            Record<uint64_t, Name>,
            Record<std::tuple<uint32_t, float>, Name>,
            Record<TileStats, Name>,
            Record<Burst, Name>,
            Record<Variadic<RowValue>, Name>,
            Record<RowArguments, Name>,
            Record<std::span<const Bf16>, Name>,
            Record<std::array<uint32_t, 3>, Name>,
            Record<std::array<uint32_t, 4>, Name>,
            Record<IdsValue, Name>,
            Record<std::tuple<Ragged, Ragged>, Name>,
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
            Record<Variadic<BitsValue>, Name>,
            Record<std::tuple<NocType, uint8_t>, Name>,
            Record<void, LaneFlags>,
            Record<void, From>,
            Record<void, Name>,
            Record<Variadic<FormatValue>, Format>,
            Record<Sample, PlotStyle>,
            Record<Sample, void>,
            TimestampedData,
            Record<Big, Name>,
            Record<Scalars, Name>,
            Record<uint32_t, InTemplate>,
            Record<uint64_t, InTemplate>,
            Record<uint16_t, InTemplate>,
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
            // Only the Name records whose arguments are all plain values or Bf16s reach NameValue, "start" as a list
            // of none. The struct records go to their own types.
            results.each<Record<Variadic<NameValue>, Name>>(batch, "NameValue + Name", [](const auto& record) {
                const auto& payload = record.payload();
                const std::string_view name = record.metadata().name;
                if (payload.empty()) {
                    return name == "start";
                }
                if (name == "tiles") {
                    const auto* count = value_at<uint32_t>(payload, 0);
                    const auto* ratio = value_at<float>(payload, 1);
                    return payload.size() == 2 && count && ratio && *count < kIterations &&
                           *ratio == float(*count) * 0.5f;
                }
                if (name == "mixed") {
                    const auto* iteration = value_at<uint8_t>(payload, 0);
                    const auto* big = value_at<uint64_t>(payload, 1);
                    const auto* odd = value_at<bool>(payload, 2);
                    return payload.size() == 3 && iteration && big && odd && *iteration < kIterations &&
                           *big == uint64_t(*iteration) << 33 && *odd == (*iteration % 2 == 1);
                }
                if (name == "no inline") {
                    const auto* iteration = value_at<uint64_t>(payload, 0);
                    return payload.size() == 1 && iteration && *iteration < kIterations;
                }
                const auto* bf16 = value_at<Bf16>(payload, 0);
                const auto* count = value_at<uint32_t>(payload, 1);
                return name == "bf16" && payload.size() == 2 && bf16 && count && bf16->bits == *count;
            });
            results.each<Record<uint64_t, Name>>(batch, "uint64_t + Name", [](const auto& record) {
                return record.metadata().name == "no inline" && record.payload() < kIterations;
            });
            results.each<Record<std::tuple<uint32_t, float>, Name>>(
                batch, "(uint32_t, float) + Name", [](const auto& record) {
                    const auto& [count, ratio] = record.payload();
                    return record.metadata().name == "tiles" && count < kIterations && ratio == float(count) * 0.5f;
                });
            results.each<Record<void, From>>(batch, "no payload + From", [](const auto& record) {
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
            results.each<Record<Variadic<RowValue>, Name>>(batch, "RowValue + Name", [&](const auto& record) {
                const auto& payload = record.payload();
                const std::string_view name = record.metadata().name;
                if (payload.empty()) {
                    return name == "start";
                }
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
            results.each<Record<RowArguments, Name>>(batch, "(number, float array) + Name", [&](const auto& record) {
                if (!results.kept_row_arguments) {
                    results.kept_row_arguments = record;
                    results.expect(
                        std::get<1>(record.payload()).data() !=
                            std::get<1>(results.kept_row_arguments->payload()).data(),
                        "a kept record owns the array in its tuple");
                }
                return record.metadata().name == "row" && sent_as_row(record.payload());
            });
            results.each<Record<std::span<const Bf16>, Name>>(batch, "Bf16 array + Name", [](const auto& record) {
                const auto& payload = record.payload();
                return record.metadata().name == "bf16 row" && payload.size() == 2 &&
                       payload[1].bits == payload[0].bits + 1;
            });
            // "ids" sends three uint32_t, so only the array of three gets it.
            results.each<Record<std::array<uint32_t, 3>, Name>>(batch, "uint32_t[3] + Name", [](const auto& record) {
                const std::array<uint32_t, 3>& ids = record.payload();
                return record.metadata().name == "ids" && ids[0] < kIterations && ids[1] == 2 * ids[0] &&
                       ids[2] == 3 * ids[0];
            });
            results.each<Record<std::array<uint32_t, 4>, Name>>(
                batch, "uint32_t[4] + Name", [](const auto&) { return false; });
            // The exact-length array beats the span listed before it.
            results.each<Record<IdsValue, Name>>(batch, "IdsValue + Name", [](const auto& record) {
                const auto* ids = std::get_if<std::array<uint32_t, 3>>(&record.payload());
                return record.metadata().name == "ids" && ids && (*ids)[2] == 3 * (*ids)[0];
            });
            results.each<Record<std::tuple<Ragged, Ragged>, Name>>(
                batch, "(Ragged, Ragged) + Name", [](const auto& record) {
                    const auto& [first, second] = record.payload();
                    return record.metadata().name == "ragged" && first.b < kIterations && second.b == first.b &&
                           first.a == (first.b << 8 | 1) && second.a == (second.b << 8 | 2);
                });
            results.each<Record<Hole, Name>>(batch, "Hole + Name", [](const auto& record) {
                return record.metadata().name == "hole" && sent_in(record.payload()).has_value();
            });
            results.each<Record<void, Hole>>(batch, "no payload + Hole", [](const auto& record) {
                const Hole& hole = record.metadata();
                return hole.iteration == 7 && std::string_view(hole.name) == "Ada" && hole.a == 5 && hole.b == 17;
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
                return record.metadata().name == "tagged flags" && std::string_view(payload.name) == "Ada" &&
                       delta >= -8 && delta < 8 && kind == uint32_t(delta + 8) % 8;
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
            // An enum argument matches its own type rather than its underlying integer, so NameValue, which lists
            // uint8_t but no enum, never gets "noc type".
            results.each<Record<std::tuple<NocType, uint8_t>, Name>>(
                batch, "(NocType, uint8_t) + Name", [](const auto& record) {
                    const auto& [type, iteration] = record.payload();
                    return record.metadata().name == "noc type" && type == NocType::Write && iteration < kIterations;
                });
            // A struct payload is one argument too, and a struct passed as an argument goes beside the iteration it was
            // sent in.
            results.each<Record<Variadic<BitsValue>, Name>>(batch, "BitsValue + Name", [](const auto& record) {
                const auto& payload = record.payload();
                if (payload.empty()) {
                    return record.metadata().name == "start";
                }
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
            results.each<Record<void, LaneFlags>>(batch, "no payload + LaneFlags", [](const auto& record) {
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
                batch, "no payload + Name", [](const auto& record) { return record.metadata().name == "start"; });
            results.each<Record<Variadic<FormatValue>, Format>>(batch, "FormatValue + Format", [](const auto& record) {
                const auto& payload = record.payload();
                if (payload.empty()) {
                    return record.metadata().format == "iteration done" && record.metadata().arguments.empty();
                }
                const auto* iteration = value_at<uint32_t>(payload, 0);
                const auto* big = value_at<uint64_t>(payload, 1);
                return payload.size() == 2 && iteration && big && *iteration < kIterations &&
                       *big == uint64_t(*iteration) << 33 &&
                       record.metadata().format == "hello Ada iteration {} big {}" &&
                       record.metadata().arguments == "i, uint64_t(i) << 33";
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
            // The largest payload, which the publish copies as word runs, inlined and through the out-of-line writer.
            results.each<Record<Big, Name>>(batch, "Big + Name", [](const auto& record) {
                const std::string_view name = record.metadata().name;
                return (name == "big" || name == "big no inline") && sent_as_big(record.payload().words);
            });
            results.each<Record<Scalars, Name>>(batch, "Scalars + Name", [](const auto& record) {
                const auto& [i8, i16, i32, u16, f64, c] = record.payload();
                const int32_t iteration = -1 - i8;
                return record.metadata().name == "scalars" && iteration >= 0 && iteration < int32_t(kIterations) &&
                       i16 == -300 - iteration && i32 == -70000 - iteration && u16 == 60000 + iteration &&
                       f64 == iteration + 0.5 && c == 'a' + iteration;
            });
            results.each<Record<uint32_t, InTemplate>>(batch, "uint32_t + InTemplate", [](const auto& record) {
                return sent_in_template(record, "function", "publish_from_template(T)") &&
                       record.payload() < kIterations;
            });
            results.each<Record<uint64_t, InTemplate>>(batch, "uint64_t + InTemplate", [](const auto& record) {
                return sent_in_template(record, "function", "publish_from_template(T)") &&
                       record.payload() % (uint64_t{1} << 33) == 0 && record.payload() >> 33 < kIterations;
            });
            results.each<Record<uint16_t, InTemplate>>(batch, "uint16_t + InTemplate", [](const auto& record) {
                return sent_in_template(record, "member", "Publisher<T>::publish(T)") && record.payload() < kIterations;
            });
            // Dispatch kernels send TimestampedData of their own.
            for (const TimestampedData& record : batch.records<TimestampedData>()) {
                if (record.metadata().name == "pair") {
                    results.common("TimestampedData", record);
                    const auto& values = record.payload();
                    results.expect(
                        values.size() == 2 && values[0] < kIterations && values[1] == values[0] << 33,
                        "TimestampedData: wrong values");
                }
            }
            results.check_order(batch);
        },
        "records-test");
    auto times_callback = RegisterCallback(
        [&](const Batch<
            Record<Stamped, Name>,
            Record<std::tuple<uint32_t, Timestamp>, Name>,
            Record<Variadic<TimeValue>, Name>,
            Record<Interval, Name>,
            Record<BigStamped, Name>>& batch) {
            std::lock_guard lock(results.mutex);
            results.each<Record<Stamped, Name>>(batch, "Stamped + Name", [&](const auto& record) {
                const auto& payload = record.payload();
                const std::string_view name = record.metadata().name;
                const uint64_t at = payload.at.device_cycles();
                if (name == "stamped" || name == "stamped no inline") {
                    return payload.iteration < kIterations && Results::before(payload.at, record.published_at());
                }
                if (name == "since start") {
                    // Every record that carries the kernel's start time carries the same value, including the last,
                    // which the kernel sends after the long zone.
                    return results.kernel_starts.try_emplace(Results::launch(record), at).first->second == at &&
                           payload.iteration <= kIterations &&
                           Results::before(payload.at, record.published_at(), kKernelTicksBound);
                }
                if (name == "after full") {
                    return payload.iteration == kIterations && Results::before(payload.at, record.published_at());
                }
                if (name == "before long") {
                    results.long_opens[Results::launch(record)] = at;
                    return record.published_at().device_cycles() - at > uint64_t{1} << 32 &&
                           Results::before(payload.at, record.published_at(), kKernelTicksBound);
                }
                return false;
            });
            results.each<Record<std::tuple<uint32_t, Timestamp>, Name>>(
                batch, "(uint32_t, Timestamp) + Name", [](const auto& record) {
                    const auto& [iteration, at] = record.payload();
                    return record.metadata().name == "now" && iteration < kIterations &&
                           Results::before(at, record.published_at());
                });
            // A Timestamp in a Variadic is built field by field rather than written directly.
            results.each<Record<Variadic<TimeValue>, Name>>(batch, "TimeValue + Name", [](const auto& record) {
                const auto& payload = record.payload();
                if (payload.empty()) {
                    return record.metadata().name == "start";
                }
                const auto* iteration = value_at<uint32_t>(payload, 0);
                const auto* at = value_at<Timestamp>(payload, 1);
                return record.metadata().name == "now" && payload.size() == 2 && iteration && at &&
                       *iteration < kIterations && Results::before(*at, record.published_at());
            });
            results.each<Record<BigStamped, Name>>(batch, "BigStamped + Name", [&](const auto& record) {
                const auto& payload = record.payload();
                const uint64_t at = payload.at.device_cycles();
                if (record.metadata().name != "big stamped" || !sent_as_big(payload.words)) {
                    return false;
                }
                // The one after the loop carries the kernel's start time, more than 32 bits back: a full record.
                if (record.published_at().device_cycles() - at > uint64_t{1} << 32) {
                    return results.kernel_starts.try_emplace(Results::launch(record), at).first->second == at &&
                           Results::before(payload.at, record.published_at(), kKernelTicksBound);
                }
                return Results::before(payload.at, record.published_at());
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
                kKernelDir + "records.cpp",
                cores,
                DataMovementConfig{.processor = processor, .noc = noc, .defines = defines});
        }
        CreateKernel(program, kKernelDir + "records.cpp", cores, ComputeConfig{.defines = defines});
        return program;
    });
    callback.reset();
    times_callback.reset();
    results.expect(error.empty(), error);

    results.expect(
        results.kept_row && sent_as_row(results.kept_row->payload()), "a record with an array kept past its callback");
    // The device is closed and its capture's lanes have gone with its receiver, but for those a kept record holds.
    results.expect(
        results.kept_row && results.kept_row->metadata().name == "row" && !results.kept_row->location().file.empty() &&
            results.kept_row->core().logical.x < kGridSide && results.kept_row->core().logical.y < kGridSide,
        "a kept record's core, location and metadata outlive its capture");
    results.expect(
        results.kept_row_arguments && sent_as_row(results.kept_row_arguments->payload()),
        "a record with an array in its tuple kept past its callback");
    for (const auto& [launch, start] : results.long_starts) {
        const auto open = results.long_opens.find(launch);
        results.expect(
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
                results.expect(
                    seen_on_core == want / kCores,
                    fmt::format("{}: {} records from core {}, want {}", kind, seen_on_core, core.str(), want / kCores));
            }
        }
    }
    fmt::print("[records] {}\n", results.failed ? "FAIL" : "PASS");
    return results.failed ? 1 : 0;
}
