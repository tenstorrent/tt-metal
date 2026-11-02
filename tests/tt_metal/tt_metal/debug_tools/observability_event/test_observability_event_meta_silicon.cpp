// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metadata on silicon: test_kernels/misc/observability_event_meta_silicon.cpp publishes one record per metadata shape
// on every RISC of one core, DevicePublish's descriptors carry each call site's metadata in its ELF, and this test
// reads every record's metadata() through the public streaming profiler API and compares it with the constants the
// kernel was built from (observability_event_meta_silicon.h).
//
// Needs a Blackhole device and TT_METAL_STREAMING_PROFILER=1 in the environment before the process starts; skips
// otherwise. Run it on its own (--gtest_filter=ObservabilityEventMetaSilicon.*), since the profiler is armed for the
// whole process.

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <set>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/streaming_profiler.hpp>
#include <tt-metalium/host_api.hpp>

#include "llrt/rtoptions.hpp"
#include "tt_metal/tt_metal/test_kernels/misc/observability_event_meta_silicon.h"

namespace observability_event_meta_silicon_test {

using namespace tt::tt_metal;
namespace sp = tt::tt_metal::experimental::streaming_profiler;
namespace m = observability_event_meta_silicon;

constexpr const char* kKernel = "tests/tt_metal/tt_metal/test_kernels/misc/observability_event_meta_silicon.cpp";
constexpr int kRiscs = 5;

using Nested = sp::Record<void, m::NestedMeta>;
using Deep = sp::Record<void, m::DeepMeta>;
using AllWidths = sp::Record<void, m::AllWidthsMeta>;
using Defaulted = sp::Record<void, m::DefaultedMeta>;
using TwoStrings = sp::Record<void, m::TwoStringsMeta>;
using Templated = sp::Record<void, m::TemplatedMeta>;

// Each metadata type's fields, leaves in order, so two values compare field by field.
auto fields(const m::NestedMeta& v) { return std::tuple(v.name, v.style.color, v.style.verbosity, v.flags); }
auto fields(const m::DeepMeta& v) { return std::tuple(v.name, v.mid.leaf.a, v.mid.leaf.b, v.mid.c, v.tail); }
auto fields(const m::AllWidthsMeta& v) { return std::tuple(v.name, v.on, v.i8, v.u16, v.i32, v.i64, v.wide, v.level); }
auto fields(const m::DefaultedMeta& v) { return std::tuple(v.name, v.color, v.verbosity); }
auto fields(const m::TwoStringsMeta& v) { return std::tuple(v.name, v.detail); }

bool streaming_profiler_enabled() {
    try {
        return tt::llrt::RunTimeOptions{}.get_streaming_profiler_enabled();
    } catch (const std::exception&) {
        return false;  // a build without Tracy rejects the variable
    }
}

// A record seen on one RISC: its name, its processor, and its call site's metadata, whose address tells call sites
// apart.
struct Seen {
    std::string_view name;
    int processor;
    const void* call_site_metadata;
    std::string_view file;
    uint32_t line;
};

template <typename R>
Seen seen(const R& record) {
    return {
        record.metadata().name,
        static_cast<int>(record.core().processor),
        &record.metadata(),
        record.location().file,
        record.location().line};
}

TEST(ObservabilityEventMetaSilicon, EveryShapeRoundTripsOnEveryRisc) {
    if (!streaming_profiler_enabled()) {
        GTEST_SKIP() << "needs TT_METAL_STREAMING_PROFILER=1 in the environment";
    }
    std::mutex mu;
    std::vector<Seen> all;
    std::multiset<std::pair<int, uint32_t>> templated_values;
    sp::Callback callback = sp::RegisterCallback(
        [&](const sp::Batch<sp::Zone, Nested, Deep, AllWidths, Defaulted, TwoStrings, Templated>& batch) {
            std::lock_guard lock(mu);
            for (const sp::Zone& zone : batch.records<sp::Zone>()) {
                const sp::ZoneMetadata& meta = zone.metadata();
                if (meta.name == "meta_plain") {
                    EXPECT_EQ(meta.color, 0u);
                    EXPECT_EQ(meta.unit, sp::TimeUnit::Cycles);
                } else if (meta.name == "meta_colored") {
                    EXPECT_EQ(meta.color, m::kColor);
                    EXPECT_EQ(meta.unit, sp::TimeUnit::Cycles);
                } else if (meta.name == "meta_unit") {
                    EXPECT_EQ(meta.color, m::kColor);
                    EXPECT_EQ(meta.unit, m::kUnit);
                } else {
                    continue;
                }
                all.push_back(seen(zone));
            }
            for (const Nested& r : batch.records<Nested>()) {
                EXPECT_EQ(fields(r.metadata()), fields(m::kNested));
                all.push_back(seen(r));
            }
            for (const Deep& r : batch.records<Deep>()) {
                EXPECT_EQ(fields(r.metadata()), fields(m::kDeep));
                all.push_back(seen(r));
            }
            for (const AllWidths& r : batch.records<AllWidths>()) {
                EXPECT_EQ(fields(r.metadata()), fields(m::kAllWidths));
                all.push_back(seen(r));
            }
            for (const Defaulted& r : batch.records<Defaulted>()) {
                EXPECT_EQ(fields(r.metadata()), fields(m::kDefaulted));
                EXPECT_EQ(r.metadata().color, 0x123456u);
                EXPECT_EQ(r.metadata().verbosity, 4u);
                all.push_back(seen(r));
            }
            for (const TwoStrings& r : batch.records<TwoStrings>()) {
                EXPECT_EQ(fields(r.metadata()), fields(m::kTwoStrings));
                all.push_back(seen(r));
            }
            for (const Templated& r : batch.records<Templated>()) {
                EXPECT_EQ(r.metadata().name, m::kTemplatedName);
                templated_values.insert({static_cast<int>(r.core().processor), r.metadata().value});
                all.push_back(seen(r));
            }
        },
        "observability-event-meta-test");

    {
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
        if (mesh_device->arch() != tt::ARCH::BLACKHOLE) {
            mesh_device->close();
            GTEST_SKIP() << "the streaming profiler runs on Blackhole";
        }
        Program program = CreateProgram();
        const CoreCoord core{0, 0};
        CreateKernel(
            program,
            kKernel,
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        CreateKernel(
            program,
            kKernel,
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        CreateKernel(program, kKernel, core, ComputeConfig{});
        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, /*blocking=*/false);
        distributed::Finish(mesh_device->mesh_command_queue());
        mesh_device->close();  // the capture is complete once the device closes
    }
    callback.reset();

    // Records per (RISC, name), and the distinct call sites behind each name.
    std::map<std::pair<int, std::string_view>, int> records;
    std::map<std::pair<int, std::string_view>, std::set<const void*>> call_sites;
    for (const Seen& s : all) {
        const auto key = std::make_pair(s.processor, s.name);
        records[key]++;
        call_sites[key].insert(s.call_site_metadata);
        EXPECT_TRUE(s.file.ends_with("observability_event_meta_silicon.cpp")) << s.file;
        EXPECT_GT(s.line, 0u);
    }

    const std::string_view once[] = {
        "meta_plain",
        "meta_colored",
        "meta_unit",
        "meta_nested",
        "meta_deep",
        "meta_all_widths",
        "meta_defaulted",
        "meta_two_strings",
    };
    for (int risc = 0; risc < kRiscs; risc++) {
        for (std::string_view name : once) {
            EXPECT_EQ((records[{risc, name}]), 1) << name << " on RISC " << risc;
        }
        // Three calls, two distinct values: the two calls with equal values share one call site.
        const auto key = std::make_pair(risc, std::string_view(m::kTemplatedName));
        EXPECT_EQ(records[key], 3) << "templated records on RISC " << risc;
        EXPECT_EQ(call_sites[key].size(), 2u) << "templated call sites on RISC " << risc;
        EXPECT_EQ(templated_values.count({risc, m::kTemplatedA}), 2u) << "RISC " << risc;
        EXPECT_EQ(templated_values.count({risc, m::kTemplatedB}), 1u) << "RISC " << risc;
    }
}

}  // namespace observability_event_meta_silicon_test
