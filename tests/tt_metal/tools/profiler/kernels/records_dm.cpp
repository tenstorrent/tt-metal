// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The data-movement RISCs publish every record the host checks, with values derived from the iteration so the host can
// verify every field.
#include <array>
#include <cstdint>
#include "api/debug/device_publish.h"
#include "../records_types.hpp"

#if defined(COMPILE_FOR_BRISC)
constexpr uint16_t kRisc = 0;
#else
constexpr uint16_t kRisc = 1;
#endif

using namespace records_test;
using tt::tt_metal::experimental::streaming_profiler::kPublishImmediately;

#define EXAMPLE_PRINT(fmt, ...) \
    DevicePublish<Format{TT_RECORD_STRING(fmt), TT_RECORD_STRING(#__VA_ARGS__)}>(__VA_ARGS__)

// The padding canary's pointer-free twin, laid out as the canary is. The language's own reading of its value bits must
// match the host's copy of the canary's layout (hostdev/record_site.h), which every image's canary is checked against.
struct PaddingCanaryTwin {
    uint8_t first;
    uint32_t text;
    uint32_t low : 3;
    uint32_t high : 10;
    uint8_t last;
};
constexpr bool matches_host_canary() {
    constexpr auto value_bits = kernel_profiler::records::value_bit_mask<PaddingCanaryTwin>();
    constexpr auto& host = tt::tt_metal::experimental::streaming_profiler::detail::kPaddingCanaryValueBits;
    for (size_t byte = 0; byte < host.size(); ++byte) {
        if (value_bits.at[byte] != host[byte]) {
            return false;
        }
    }
    return sizeof(value_bits) == host.size();
}
static_assert(matches_host_canary(), "the padding canary's layout differs from the host's copy");

// A call site in an inlined helper reports the helper, and one in a lambda reports kernel_main()::<lambda()>.
FORCE_INLINE void publish_from_helper() { DevicePublish<From{"helper"_s}>(); }

void kernel_main() {
    const Timestamp kernel_start = Timestamp::now();
    for (uint32_t i = 0; i < N_ITERS; i++) {
        DeviceZoneScopedN("records-iteration");
        DevicePublish<NocSite{"noc_async_read"_s, NocType::Read, 0xFF8000}>(NocEvent{{uint16_t(i), kRisc}, i * 64});
        DevicePublish<Name{"tiles"_s}>(i, float(i) * 0.5f);
        DevicePublish<Name{"mixed"_s}>(uint8_t(i), "Ada"_s, uint64_t(i) << 33, i % 2 == 1);
        DevicePublish<Name{"noc type"_s}>(NocType::Write, uint8_t(i));
        DevicePublish<Name{"tile stats"_s}, kPublishImmediately>(TileStats{i, i * 2048, i * 100, i % 3});
        DevicePublish<Name{"start"_s}>();
        DevicePublish<Name{"bf16"_s}>(Bf16{uint16_t(i)}, i);
        DevicePublish<Name{"burst"_s}>(Burst{{i, i + 1, i + 2, i + 3}, {'r', 'd', char('0' + i % 10), 0}});
        const std::array<float, 4> row{float(i), float(i) + 0.25f, float(i) + 0.5f, float(i) + 0.75f};
        DevicePublish<Name{"row"_s}>(i, row);
        const uint32_t ids[3] = {i, 2 * i, 3 * i};
        DevicePublish<Name{"ids"_s}>(ids);
        char tag[4] = {'i', 'd', char('0' + i % 10), 0};
        DevicePublish<Name{"tag"_s}>(tag);
        DevicePublish<Name{"bf16 row"_s}>(std::array<Bf16, 2>{Bf16{uint16_t(i)}, Bf16{uint16_t(i + 1)}});
        DevicePublish<Name{"ragged"_s}>(
            Ragged{uint16_t(i << 8 | 1), uint8_t(i)}, Ragged{uint16_t(i << 8 | 2), uint8_t(i)});
        const Hole hole{i, "Ada"_s, uint16_t(i % 8), uint16_t(i % 32)};
        DevicePublish<Name{"hole"_s}>(hole);
        DevicePublish<Name{"hole value"_s}>(hole, i);
        DevicePublish<Hole{7, "Ada"_s, 5, 17}>();
        DevicePublish<Name{"tight bits"_s}>(TightBits{i, uint8_t(i), uint16_t(i + 1)});
        DevicePublish<Name{"flags"_s}>(LaneFlags{i % 8, i % 32, i});
        DevicePublish<Name{"tagged flags"_s}>(TaggedFlags{"Ada"_s, i % 8, int32_t(i % 16) - 8});
        DevicePublish<Name{"packed"_s}>(Packed{uint8_t(i), i * 3});
        DevicePublish<Name{"wide"_s}>(Wide{i, i + 1});
        DevicePublish<Name{"with empty"_s}>(WithEmpty{{}, i});
        DevicePublish<Name{"deep"_s}>(Deep{{{i}, uint16_t(i + 1)}, uint16_t(i + 2)});
        DevicePublish<Name{"skewed"_s}>(Skewed{i % 256, (i + 1) % 256});
        DevicePublish<Name{"flags value"_s}>(i, LaneFlags{i % 8, i % 32, i});
        DevicePublish<Name{"tight value"_s}>(i, TightBits{i, uint8_t(i), uint16_t(i + 1)});
        DevicePublish<LaneFlags{5, 17, 1000}>();
        publish_from_helper();
        [] { DevicePublish<From{"lambda"_s}>(); }();
        EXAMPLE_PRINT("hello {} iteration {} big {}", "Ada"_s, i, uint64_t(i) << 33);
        const Timestamp start = Timestamp::now();
        DevicePublish<Name{"stamped"_s}>(Stamped{Timestamp::now(), i});
        DevicePublish<Name{"since start"_s}>(Stamped{kernel_start, i});
        DevicePublish<Name{"now"_s}>(i, Timestamp::now());
        DevicePublish<Name{"interval"_s}>(Interval{start, Timestamp::now()});
        DevicePublish<PlotStyle{"l1-used"_s, PlotFormat::Memory, 0x00FF00}>(Sample{int64_t(i) * 1000});
        DevicePublish(Sample{int64_t(i) - 5});
    }
    // The spin counts ticks from the clock's low word, whose reads can't tear as it wraps.
    const Timestamp before_long = Timestamp::now();
    {
        DeviceZoneScopedN("long");
        uint64_t elapsed = 0;
        for (uint32_t last = get_timestamp_32b(); elapsed < kLongZoneTicks;) {
            const uint32_t now = get_timestamp_32b();
            elapsed += now - last;
            last = now;
        }
    }
    DevicePublish<Name{"since start"_s}>(Stamped{kernel_start, N_ITERS});
    DevicePublish<Name{"before long"_s}>(Stamped{before_long, N_ITERS});
}
