// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metadata types and values shared by the kernel debug_event_meta_silicon.cpp and its host test,
// tests/tt_metal/tt_metal/debug_tools/debug_event/test_debug_event_meta_silicon.cpp. The kernel's sites carry exactly
// these values, so the host compares what it decoded against the same constants.
#pragma once

#include <stdint.h>

namespace debug_event_meta_silicon {

enum class Level : int8_t { Low = -3, High = 9 };
enum Wide : uint64_t { kWideMax = 0xFFFFFFFFFFFFFFFFull };

struct Style {
    uint32_t color;
    uint8_t verbosity;
};
struct NestedMeta {
    const char* name;
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
    const char* name;
    Mid mid;
    uint64_t tail;
};

struct AllWidthsMeta {
    const char* name;
    bool on;
    int8_t i8;
    uint16_t u16;
    int32_t i32;
    int64_t i64;
    Wide wide;
    Level level;
};

struct DefaultedMeta {
    const char* name;
    uint32_t color = 0x123456;
    uint8_t verbosity = 4;
};

struct TwoStringsMeta {
    const char* name;
    const char* detail;
};

struct TemplatedMeta {
    const char* name;
    uint32_t value;
};

inline constexpr NestedMeta kNested{"meta_nested", {0xA9A9A9, 2}, 0xBEEF};
inline constexpr DeepMeta kDeep{"meta_deep", {{0x7F, 0x1234}, 0xDEADBEEF}, 0x0123456789ABCDEFull};
inline constexpr AllWidthsMeta kAllWidths{
    "meta_all_widths", true, -128, 0xFFFF, -123456, -0x7FFFFFFFFFFFFFFFll - 1, kWideMax, Level::Low};
inline constexpr DefaultedMeta kDefaulted{"meta_defaulted"};
// Every byte the assembler text has to escape: quotes, a backslash, a newline, a tab, UTF-8 and DEL.
inline constexpr TwoStringsMeta kTwoStrings{"meta_two_strings", "line one\nline two\t\"quotes\" \\ \xC3\xA9 \x7F end"};
inline constexpr uint32_t kColor = 0x42D4F4;

// The template site is instantiated with these values: equal values share one site, so two distinct sites.
inline constexpr const char* kTemplatedName = "meta_templated";
inline constexpr uint32_t kTemplatedA = 11;
inline constexpr uint32_t kTemplatedB = 22;

}  // namespace debug_event_meta_silicon
