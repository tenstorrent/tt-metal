// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The records test's end-to-end kernel, on both data-movement RISCs of a grid: records as real kernels send them, with
// values derived from the iteration so the host can verify every field, zones, and Timestamps on both sides of a zone
// longer than the clock's low word.
#include <array>
#include <cstdint>
#include "api/debug/device_publish.h"
#include "../records_types.hpp"

using namespace records_test;
using tt::tt_metal::experimental::device_observability::kPublishBatched;

#if defined(COMPILE_FOR_BRISC)
constexpr uint16_t kRisc = 0;
#else
constexpr uint16_t kRisc = 1;
#endif

#define EXAMPLE_PRINT(fmt, ...) \
    DevicePublish<Format{TT_RECORD_STRING(fmt), TT_RECORD_STRING(#__VA_ARGS__)}>(__VA_ARGS__)

// A call site in an inlined helper reports the helper, and one in a lambda reports kernel_main()::<lambda()>.
FORCE_INLINE void publish_from_helper() { DevicePublish<From{"helper"_s}>(); }
// A call site in a template is one call site per payload type its instantiations send, and reports the template.
template <typename T>
FORCE_INLINE void publish_from_template(T value) {
    DevicePublish<InTemplate{"function"_s}>(value);
}
template <typename T>
struct Publisher {
    static FORCE_INLINE void publish(T value) { DevicePublish<InTemplate{"member"_s}>(value); }
};

void kernel_main() {
    const Timestamp kernel_start = Timestamp::now();
    for (uint32_t i = 0; i < N_ITERS; i++) {
        DeviceZoneScopedN("records-iteration");
        DevicePublish<NocSite{"noc_async_read"_s, NocType::Read, 0xFF8000}>(NocEvent{{uint16_t(i), kRisc}, i * 64});
        DevicePublish<Name{"tile stats"_s}, kPublishBatched>(TileStats{i, i * 2048, i * 100, i % 3});
        DevicePublish<kPublishBatched>(Sample{int64_t(i) - 5});  // options without metadata
        const std::array<float, 4> row{float(i), float(i) + 0.25f, float(i) + 0.5f, float(i) + 0.75f};
        DevicePublish<Name{"row"_s}>(i, row);
        const uint32_t ids[3] = {i, 2 * i, 3 * i};
        DevicePublish<Name{"ids"_s}>(ids);
        char tag[4] = {'i', 'd', char('0' + i % 10), 0};  // not const, so data rather than a string
        DevicePublish<Name{"tag"_s}>(tag);
        DevicePublish<Name{"bf16 row"_s}>(std::array<Bf16, 2>{Bf16{uint16_t(i)}, Bf16{uint16_t(i + 1)}});
        const int32_t n = int32_t(i);
        DevicePublish<Name{"scalars"_s}>(
            int8_t(-1 - n),
            int16_t(-300 - n),
            -70000 - n,
            uint16_t(60000 + i),
            double(i) + 0.5,
            char('a' + i),
            uint64_t(i) << 33 | i,
            i % 2 == 1);
        DevicePublish<Name{"skewed"_s}>(Skewed{i % 256, (i + 1) % 256});
        DevicePublish<Hole{7, "Ada", 5, 17}>();
        DevicePublish<LaneFlags{5, 17, 1000}>();
        publish_from_helper();
        [] { DevicePublish<From{"lambda"_s}>(); }();
        publish_from_template(i);
        publish_from_template(uint64_t(i) << 33);
        Publisher<uint16_t>::publish(uint16_t(i));
        EXAMPLE_PRINT("hello Ada iteration {} big {}", i, uint64_t(i) << 33);
        DevicePublish<Format{"iteration done"_s, ""_s}>();  // a print with no arguments
        DevicePublish<TimestampedDataMetadata{"pair"_s}>(uint64_t(i), uint64_t(i) << 33);
        const Timestamp start = Timestamp::now();
        DevicePublish<Name{"since start"_s}>(Stamped{kernel_start, i});
        DevicePublish<Name{"interval"_s}>(Interval{start, Timestamp::now()});
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
    // Full records: their Timestamps are more than 32 bits back.
    DevicePublish<Name{"since start"_s}>(Stamped{kernel_start, N_ITERS});
    DevicePublish<Name{"before long"_s}>(Stamped{before_long, N_ITERS});
    // Short only because the full records above moved the cursor: the last short or medium record before them is more
    // than 32 bits back.
    DevicePublish<Name{"after full"_s}>(Stamped{Timestamp::now(), N_ITERS});
}
