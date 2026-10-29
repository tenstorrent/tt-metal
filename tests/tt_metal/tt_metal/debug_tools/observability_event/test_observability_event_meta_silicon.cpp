// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Debug event metadata, layer 3: the round trip on silicon. test_kernels/misc/observability_event_meta_silicon.cpp runs
// one zone per metadata shape on every RISC of one core; the device compiler marshals each site's value into its ELF,
// the host loader reads it back, and this test checks every zone's site through the public streaming profiler API
// (Zone::site().meta) against the constants the kernel was built from (observability_event_meta_silicon.h).
//
// Needs a Blackhole device and TT_METAL_STREAMING_PROFILER=1 in the environment before the process starts; skips
// otherwise. Run it on its own (--gtest_filter=DebugEventMetaSilicon.*), since the profiler is armed for the whole
// process.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <optional>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/streaming_profiler.hpp>
#include <tt-metalium/host_api.hpp>

#include "hostdev/observability_event_meta.h"
#include "llrt/rtoptions.hpp"
#include "tt_metal/tt_metal/test_kernels/misc/observability_event_meta_silicon.h"

namespace observability_event_meta_silicon_test {

using namespace tt::tt_metal;
namespace sp = tt::tt_metal::experimental::streaming_profiler;
namespace m = observability_event_meta_silicon;
using tt::debug_event::SiteMeta;
using tt::debug_event::ZoneColorMeta;
using tt::debug_event::ZoneMeta;
using tt::debug_event::ZoneUnitMeta;

constexpr const char* kKernel = "tests/tt_metal/tt_metal/test_kernels/misc/observability_event_meta_silicon.cpp";
constexpr int kRiscs = 5;

// A metadata value's leaf fields, as comparable values: integers by their bits, strings by their text.
struct LeafValue {
    char kind;
    uint64_t bits;
    std::string text;
    bool operator==(const LeafValue&) const = default;
};

template <class T>
std::vector<LeafValue> leaves(const T& value) {
    std::vector<LeafValue> out;
    tt::debug_event::detail::for_each_leaf(value, [&](const auto& field) {
        using F = tt::debug_event::detail::field_t<decltype(field)>;
        if constexpr (std::is_same_v<F, const char*>) {
            out.push_back({'s', 0, field != nullptr ? field : ""});
        } else {
            out.push_back({'i', tt::debug_event::detail::field_bits(field), {}});
        }
    });
    return out;
}

// SITE's metadata decoded as T and compared, leaf by leaf, with EXPECTED.
template <class T>
void expect_meta(const sp::ObservabilityEventInfo& site, const T& expected) {
    ASSERT_NE(site.meta, nullptr) << site.name;
    const std::optional<T> got = site.meta->as<T>();
    ASSERT_TRUE(got.has_value()) << site.name << " carries " << site.meta->signature << ", not "
                                 << tt::debug_event::signature_of<T>();
    EXPECT_EQ(leaves(*got), leaves(expected)) << site.name;
}

bool streaming_profiler_enabled() {
    try {
        return tt::llrt::RunTimeOptions{}.get_streaming_profiler_enabled();
    } catch (const std::exception&) {
        return false;  // a build without Tracy rejects the variable
    }
}

struct SeenZone {
    const sp::ObservabilityEventInfo* site;
    sp::Processor processor;
};

TEST(DebugEventMetaSilicon, EveryShapeRoundTripsOnEveryRisc) {
    if (!streaming_profiler_enabled()) {
        GTEST_SKIP() << "needs TT_METAL_STREAMING_PROFILER=1 in the environment";
    }
    std::mutex mu;
    std::vector<SeenZone> seen;
    sp::Callback callback = sp::RegisterCallback(
        [&](const sp::Batch<sp::Zone>& batch) {
            std::lock_guard lock(mu);
            for (const sp::Zone& zone : batch.records<sp::Zone>()) {
                if (zone.observability_event().name.starts_with("meta_")) {
                    seen.push_back({&zone.observability_event(), zone.core().processor});
                }
            }
        },
        "debug-event-meta-test");

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

    // Zones per (RISC, name), and the distinct sites behind each name.
    std::map<std::pair<int, std::string_view>, int> zones;
    std::map<std::pair<int, std::string_view>, std::set<const sp::ObservabilityEventInfo*>> sites;
    for (const SeenZone& z : seen) {
        const auto key = std::make_pair(static_cast<int>(z.processor), z.site->name);
        zones[key]++;
        sites[key].insert(z.site);

        EXPECT_TRUE(z.site->location.file.ends_with("observability_event_meta_silicon.cpp")) << z.site->location.file;
        EXPECT_GT(z.site->location.line, 0u);

        const std::string_view name = z.site->name;
        if (name == "meta_plain") {
            expect_meta(*z.site, ZoneMeta{"meta_plain"});
        } else if (name == "meta_colored") {
            expect_meta(*z.site, ZoneColorMeta{"meta_colored", m::kColor});
        } else if (name == "meta_unit") {
            expect_meta(*z.site, ZoneUnitMeta{"meta_unit", m::kColor, m::kUnit});
        } else if (name == "meta_nested") {
            expect_meta(*z.site, m::kNested);
        } else if (name == "meta_deep") {
            expect_meta(*z.site, m::kDeep);
            // The device compiler (GCC) and this one must spell the signature alike, nesting included.
            EXPECT_EQ(z.site->meta->signature, "observability_event_meta_silicon::DeepMeta:s{{bh}w}q");
        } else if (name == "meta_all_widths") {
            expect_meta(*z.site, m::kAllWidths);
        } else if (name == "meta_defaulted") {
            expect_meta(*z.site, m::kDefaulted);
            const auto d = z.site->meta->as<m::DefaultedMeta>();
            ASSERT_TRUE(d.has_value());
            EXPECT_EQ(d->color, 0x123456u);
            EXPECT_EQ(d->verbosity, 4u);
        } else if (name == "meta_two_strings") {
            expect_meta(*z.site, m::kTwoStrings);
        } else if (name == m::kTemplatedName) {
            const auto t = z.site->meta->as<m::TemplatedMeta>();
            ASSERT_TRUE(t.has_value());
            EXPECT_TRUE(t->value == m::kTemplatedA || t->value == m::kTemplatedB) << t->value;
        } else {
            ADD_FAILURE() << "unexpected site " << name;
        }
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
            EXPECT_EQ((zones[{risc, name}]), 1) << name << " on RISC " << risc;
        }
        // Three calls, two distinct values: the two calls with equal values share one site.
        const auto key = std::make_pair(risc, std::string_view(m::kTemplatedName));
        EXPECT_EQ(zones[key], 3) << "templated zones on RISC " << risc;
        ASSERT_EQ(sites[key].size(), 2u) << "templated sites on RISC " << risc;
        std::multiset<uint32_t> values;
        for (const SeenZone& z : seen) {
            if (static_cast<int>(z.processor) == risc && z.site->name == m::kTemplatedName) {
                values.insert(z.site->meta->as<m::TemplatedMeta>()->value);
            }
        }
        EXPECT_EQ(values.count(m::kTemplatedA), 2u) << "RISC " << risc;
        EXPECT_EQ(values.count(m::kTemplatedB), 1u) << "RISC " << risc;
    }
}

}  // namespace observability_event_meta_silicon_test
