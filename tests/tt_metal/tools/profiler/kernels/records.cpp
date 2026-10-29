// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Every RISC publishes records the host checks, with values derived from the iteration so the host can verify every
// field. The data-movement RISCs publish every kind of record; the compute RISCs publish two the data-movement RISCs
// also publish, so records from all five RISCs land in the same callback batches.
#include <array>
#include <cstdint>
#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#endif
#include "api/debug/device_publish.h"
#include "../records_types.hpp"

using namespace records_test;

#if defined(COMPILE_FOR_TRISC)
void kernel_main() {
    for (uint32_t i = 0; i < N_ITERS; i++) {
        DevicePublish<Name{"tiles"_s}>(i, float(i) * 0.5f);
        DevicePublish<Name{"start"_s}>();
    }
}
#else
#if defined(COMPILE_FOR_BRISC)
constexpr uint16_t kRisc = 0;
#else
constexpr uint16_t kRisc = 1;
#endif

using tt::tt_metal::experimental::streaming_profiler::kNoInline;
using tt::tt_metal::experimental::streaming_profiler::kPublishBatched;

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

// Too large for the stack.
Big big;
BigStamped big_stamped{};

void kernel_main() {
    const Timestamp kernel_start = Timestamp::now();
    for (uint32_t i = 0; i < N_ITERS; i++) {
        DeviceZoneScopedN("records-iteration");
        DevicePublish<NocSite{"noc_async_read"_s, NocType::Read, 0xFF8000}>(NocEvent{{uint16_t(i), kRisc}, i * 64});
        DevicePublish<Name{"tiles"_s}>(i, float(i) * 0.5f);
        DevicePublish<Name{"mixed"_s}>(uint8_t(i), uint64_t(i) << 33, i % 2 == 1);
        DevicePublish<Name{"no inline"_s}, kNoInline>(uint64_t(i));
        DevicePublish<Name{"noc type"_s}>(NocType::Write, uint8_t(i));
        DevicePublish<Name{"tile stats"_s}, kPublishBatched>(TileStats{i, i * 2048, i * 100, i % 3});
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
        const Hole hole{i, "Ada", uint16_t(i % 8), uint16_t(i % 32)};
        DevicePublish<Name{"hole"_s}>(hole);
        DevicePublish<Name{"hole value"_s}>(hole, i);
        DevicePublish<Hole{7, "Ada", 5, 17}>();
        DevicePublish<Name{"tight bits"_s}>(TightBits{i, uint8_t(i), uint16_t(i + 1)});
        DevicePublish<Name{"flags"_s}>(LaneFlags{i % 8, i % 32, i});
        DevicePublish<Name{"tagged flags"_s}>(TaggedFlags{"Ada", i % 8, int32_t(i % 16) - 8});
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
        EXAMPLE_PRINT("hello Ada iteration {} big {}", i, uint64_t(i) << 33);
        DevicePublish<Format{"iteration done"_s, ""_s}>();  // a print with no arguments
        DevicePublish<TimestampedDataMetadata{"pair"_s}>(uint64_t(i), uint64_t(i) << 33);
        const Timestamp start = Timestamp::now();
        DevicePublish<Name{"stamped"_s}>(Stamped{Timestamp::now(), i});
        DevicePublish<Name{"since start"_s}>(Stamped{kernel_start, i});
        DevicePublish<Name{"now"_s}>(i, Timestamp::now());
        DevicePublish<Name{"interval"_s}>(Interval{start, Timestamp::now()});
        DevicePublish<PlotStyle{"l1-used"_s, PlotFormat::Memory, 0x00FF00}>(Sample{int64_t(i) * 1000});
        DevicePublish<>(Sample{int64_t(i) - 5});
        DevicePublish<kPublishBatched>(Sample{int64_t(i) - 5});  // options without metadata
        for (uint32_t w = 0; w < 126; w++) {
            big.words[w] = i * 1000 + w;
        }
        DevicePublish<Name{"big"_s}>(big);
        DevicePublish<Name{"big no inline"_s}, kNoInline>(big);
        big_stamped.at = Timestamp::now();
        for (uint32_t w = 0; w < 123; w++) {
            big_stamped.words[w] = i * 1000 + w;
        }
        DevicePublish<Name{"big stamped"_s}>(big_stamped);
        DevicePublish<Name{"stamped no inline"_s}, kNoInline>(Stamped{Timestamp::now(), i});
        const int32_t n = int32_t(i);
        DevicePublish<Name{"scalars"_s}>(
            int8_t(-1 - n), int16_t(-300 - n), -70000 - n, uint16_t(60000 + i), double(i) + 0.5, char('a' + i));
        publish_from_template(i);
        publish_from_template(uint64_t(i) << 33);
        Publisher<uint16_t>::publish(uint16_t(i));
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
    big_stamped.at = kernel_start;  // a full record, whose words copy from the struct
    DevicePublish<Name{"big stamped"_s}>(big_stamped);
    // Short only because the full records above moved the cursor: the last short or medium record before them is more
    // than 32 bits back.
    DevicePublish<Name{"after full"_s}>(Stamped{Timestamp::now(), N_ITERS});
}
#endif
