// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metadata types and values shared by the kernel observability_event_meta_silicon.cpp and its host test,
// tests/tt_metal/tt_metal/debug_tools/observability_event/test_observability_event_meta_silicon.cpp. The kernel's
// records carry exactly these values, so the host compares what it decoded against the same constants.
#pragma once

#include <stdint.h>

#include <hostdevcommon/device_observability_types.hpp>
#include <hostdevcommon/streaming_profiler_types.hpp>

// A String constant: "..."_s on the device, the text itself on the host.
#ifdef TT_DEVICE_COMPILE
#define META_SILICON_STRING(text) text##_s
#else
#define META_SILICON_STRING(text) ::tt::tt_metal::experimental::device_observability::String(text)
#endif

namespace observability_event_meta_silicon {

using tt::tt_metal::experimental::device_observability::String;

enum class Level : int8_t { Low = -3, High = 9 };
enum Wide : uint64_t { kWideMax = 0xFFFFFFFFFFFFFFFFull };

struct Style {
    uint32_t color;
    uint8_t verbosity;
};
struct NestedMeta {
    String name;
    Style style;
    uint16_t flags;
};

struct Leaf {
    uint8_t a;
    uint16_t b;
};
struct Mid {
    Leaf leaf;
    uint32_t c;
};
struct DeepMeta {
    String name;
    Mid mid;
    uint64_t tail;
};

struct AllWidthsMeta {
    String name;
    bool on;
    int8_t i8;
    uint16_t u16;
    int32_t i32;
    int64_t i64;
    Wide wide;
    Level level;
};

struct DefaultedMeta {
    String name;
    uint32_t color = 0x123456;
    uint8_t verbosity = 4;
};

struct TwoStringsMeta {
    String name;
    String detail;
};

struct TemplatedMeta {
    String name;
    uint32_t value;
};

inline constexpr NestedMeta kNested{META_SILICON_STRING("meta_nested"), {0xA9A9A9, 2}, 0xBEEF};
inline constexpr DeepMeta kDeep{META_SILICON_STRING("meta_deep"), {{0x7F, 0x1234}, 0xDEADBEEF}, 0x0123456789ABCDEFull};
inline constexpr AllWidthsMeta kAllWidths{
    META_SILICON_STRING("meta_all_widths"),
    true,
    -128,
    0xFFFF,
    -123456,
    -0x7FFFFFFFFFFFFFFFll - 1,
    kWideMax,
    Level::Low};
inline constexpr DefaultedMeta kDefaulted{META_SILICON_STRING("meta_defaulted")};
// Quotes, a backslash, a newline, a tab, UTF-8 and DEL.
inline constexpr TwoStringsMeta kTwoStrings{
    META_SILICON_STRING("meta_two_strings"),
    META_SILICON_STRING("line one\nline two\t\"quotes\" \\ \xC3\xA9 \x7F end")};
inline constexpr uint32_t kColor = 0x42D4F4;
inline constexpr tt::tt_metal::experimental::streaming_profiler::TimeUnit kUnit =
    tt::tt_metal::experimental::streaming_profiler::TimeUnit::Milliseconds;

// The templated call site is instantiated with these values: equal values share one call site, so two call sites.
inline constexpr String kTemplatedName = META_SILICON_STRING("meta_templated");
inline constexpr uint32_t kTemplatedA = 11;
inline constexpr uint32_t kTemplatedB = 22;

}  // namespace observability_event_meta_silicon
