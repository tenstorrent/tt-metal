// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <tracy/Tracy.hpp>
#include <type_traits>

// Opt-in Tracy macros (zones, messages, plots) for debug-verbosity profiling. A TTZone*D / TTMessage*D /
// TTPlotD macro's first argument is a category token (DISPATCH, RT_PROFILER, ...); it is compiled in only when
// that category is selected at build time via --build-perf-debug ('all' selects every category), and compiles
// to nothing otherwise. Categories are defined once in tt_metal/tools/profiler/tracy_debug_categories.txt; from
// it the build generates, per category, TT_TRACY_CATEGORY_<TOKEN> (1 if selected, 0 if not) and
// TT_TRACY_CATEGORY_INDEX_<TOKEN> (declaration order, used to auto-assign a color) into the header included here.
#if defined(TRACY_ENABLE)
#include <tracy_debug_categories_generated.hpp>
#endif

namespace tt::tracy_debug {
inline constexpr std::array<uint32_t, 20> kCategoryColors{
    0xE6194B,  // Red
    0x3CB44B,  // Green
    0xFFE119,  // Yellow
    0x4363D8,  // Blue
    0xF58231,  // Orange
    0x911EB4,  // Purple
    0x42D4F4,  // Cyan
    0xF032E6,  // Magenta
    0xBFEF45,  // Lime
    0xFABED4,  // Pink
    0x469990,  // Teal
    0xDCBEFF,  // Lavender
    0x9A6324,  // Brown
    0xFFFAC8,  // Beige
    0x800000,  // Maroon
    0xAAFFC3,  // Mint
    0x808000,  // Olive
    0xFFD8B1,  // Apricot
    0x000075,  // Navy
    0xA9A9A9,  // Grey
};
constexpr uint32_t category_color(std::size_t index) { return kCategoryColors[index % kCategoryColors.size()]; }
constexpr bool zone_name_valid(const char* name) { return name != nullptr; }
}  // namespace tt::tracy_debug

// TT_TRACY_EMIT(category, disabled, enabled...) selects `enabled...` when the category is selected, `disabled`
// when it is not, and the undeclared identifier TT_TRACY_ERROR_unknown_tracy_debug_category (a readable compile
// error at the call site) when the category token has no entry in tracy_debug_categories.txt. The category's define
// resolves to 1/0, or stays an unexpanded token when unknown; PROBE/SELECT then map 1/0/other onto the
// enabled/disabled/error branch. The two-step CONCAT forces the category define to expand before it is pasted.
#define TT_TRACY_CONCAT_(a, b) a##b
#define TT_TRACY_CONCAT(a, b) TT_TRACY_CONCAT_(a, b)
#define TT_TRACY_EMIT_1(disabled, ...) __VA_ARGS__
#define TT_TRACY_EMIT_0(disabled, ...) disabled
#define TT_TRACY_EMIT_BAD(disabled, ...) TT_TRACY_ERROR_unknown_tracy_debug_category
#define TT_TRACY_PROBE_0 ~, TT_TRACY_EMIT_0
#define TT_TRACY_PROBE_1 ~, TT_TRACY_EMIT_1
#define TT_TRACY_SECOND(_1, _2, ...) _2
#define TT_TRACY_SELECT(...) TT_TRACY_SECOND(__VA_ARGS__, TT_TRACY_EMIT_BAD, ~)
#define TT_TRACY_BRANCH_FOR(value) TT_TRACY_SELECT(TT_TRACY_CONCAT(TT_TRACY_PROBE_, value))
#if defined(TRACY_ENABLE)
#define TT_TRACY_EMIT(category, disabled, ...) TT_TRACY_BRANCH_FOR(TT_TRACY_CATEGORY_##category)(disabled, __VA_ARGS__)
#else
#define TT_TRACY_EMIT(category, disabled, ...) disabled
#endif

#define TT_TRACY_CATEGORY_COLOR(category) (::tt::tracy_debug::category_color(TT_TRACY_CATEGORY_INDEX_##category))

#define TT_TRACY_CONSTEXPR_STR(s) sizeof(::std::integral_constant<bool, ::tt::tracy_debug::zone_name_valid(s)>)
#define TT_TRACY_CONSTEXPR_COLOR(v) sizeof(::std::integral_constant<uint32_t, (v)>)

// A Tracy-only arg type (tracy::PlotFormatType) can't be checked under --disable-profiler.
#if defined(TRACY_ENABLE)
#define TT_TRACY_SIZEOF_IF_ENABLED(x) sizeof(x)
#else
#define TT_TRACY_SIZEOF_IF_ENABLED(x) 0u
#endif

// Category-gated Tracy macros; the first argument is a category token
// from tt_metal/tools/profiler/tracy_debug_categories.txt.
#define TTZoneScopedD(category) TT_TRACY_EMIT(category, , ZoneScopedC(TT_TRACY_CATEGORY_COLOR(category)))
#define TTZoneScopedDN(category, name) \
    TT_TRACY_EMIT(category, (void(TT_TRACY_CONSTEXPR_STR(name))), ZoneScopedNC(name, TT_TRACY_CATEGORY_COLOR(category)))
#define TTZoneScopedDC(category, color) \
    TT_TRACY_EMIT(category, (void(TT_TRACY_CONSTEXPR_COLOR(color))), ZoneScopedC(color))
#define TTZoneScopedDNC(category, name, color) \
    TT_TRACY_EMIT(                             \
        category, (void(TT_TRACY_CONSTEXPR_STR(name) + TT_TRACY_CONSTEXPR_COLOR(color))), ZoneScopedNC(name, color))
#define TTZoneScopedDS(category, depth) \
    TT_TRACY_EMIT(category, (void(sizeof(depth))), ZoneScopedCS(TT_TRACY_CATEGORY_COLOR(category), depth))
#define TTZoneScopedDNS(category, name, depth)                \
    TT_TRACY_EMIT(                                            \
        category,                                             \
        (void(TT_TRACY_CONSTEXPR_STR(name) + sizeof(depth))), \
        ZoneScopedNCS(name, TT_TRACY_CATEGORY_COLOR(category), depth))
#define TTZoneScopedDCS(category, color, depth) \
    TT_TRACY_EMIT(category, (void(TT_TRACY_CONSTEXPR_COLOR(color) + sizeof(depth))), ZoneScopedCS(color, depth))
#define TTZoneScopedDNCS(category, name, color, depth)                                          \
    TT_TRACY_EMIT(                                                                              \
        category,                                                                               \
        (void(TT_TRACY_CONSTEXPR_STR(name) + TT_TRACY_CONSTEXPR_COLOR(color) + sizeof(depth))), \
        ZoneScopedNCS(name, color, depth))
#define TTZoneTextD(category, txt, size) \
    TT_TRACY_EMIT(category, (void(sizeof(txt) + sizeof(size))), ZoneText(txt, size))
#define TTZoneNameD(category, txt, size) \
    TT_TRACY_EMIT(category, (void(sizeof(txt) + sizeof(size))), ZoneName(txt, size))
#define TTZoneColorD(category, color) TT_TRACY_EMIT(category, (void(sizeof(color))), ZoneColor(color))
#define TTZoneValueD(category, value) TT_TRACY_EMIT(category, (void(sizeof(value))), ZoneValue(value))
#define TTZoneIsActiveD(category) TT_TRACY_EMIT(category, false, ZoneIsActive)

#define TTMessageD(category, txt, size) \
    TT_TRACY_EMIT(category, (void(sizeof(txt) + sizeof(size))), TracyMessage(txt, size))
#define TTMessageDL(category, txt) TT_TRACY_EMIT(category, (void(sizeof(txt))), TracyMessageL(txt))
#define TTMessageDC(category, txt, size, color) \
    TT_TRACY_EMIT(category, (void(sizeof(txt) + sizeof(size) + sizeof(color))), TracyMessageC(txt, size, color))
#define TTMessageDLC(category, txt, color) \
    TT_TRACY_EMIT(category, (void(sizeof(txt) + sizeof(color))), TracyMessageLC(txt, color))
#define TTMessageDS(category, txt, size, depth) \
    TT_TRACY_EMIT(category, (void(sizeof(txt) + sizeof(size) + sizeof(depth))), TracyMessageS(txt, size, depth))
#define TTMessageDLS(category, txt, depth) \
    TT_TRACY_EMIT(category, (void(sizeof(txt) + sizeof(depth))), TracyMessageLS(txt, depth))
#define TTMessageDCS(category, txt, size, color, depth)                     \
    TT_TRACY_EMIT(                                                          \
        category,                                                           \
        (void(sizeof(txt) + sizeof(size) + sizeof(color) + sizeof(depth))), \
        TracyMessageCS(txt, size, color, depth))
#define TTMessageDLCS(category, txt, color, depth) \
    TT_TRACY_EMIT(category, (void(sizeof(txt) + sizeof(color) + sizeof(depth))), TracyMessageLCS(txt, color, depth))

// True only while a Tracy server is connected (always false when Tracy is compiled out). Not category-gated: it is a
// runtime check for gating Tracy emission that would otherwise buffer unboundedly while disconnected. That buffering
// drives Tracy's allocator into a continuous mmap storm that starves other threads (e.g. the RT-profiler receiver's
// D2H drain), so any high-rate emitter should skip its work unless this is true.
#if defined(TRACY_ENABLE)
#define TTTracyConnected() (::tracy::GetProfiler().IsConnected())
#else
#define TTTracyConnected() (false)
#endif

#define TTPlotD(category, name, val) TT_TRACY_EMIT(category, (void(sizeof(name) + sizeof(val))), TracyPlot(name, val))
#define TTPlotConfigD(category, name, type, step, fill, color)                                                 \
    TT_TRACY_EMIT(                                                                                             \
        category,                                                                                              \
        (void(sizeof(name) + TT_TRACY_SIZEOF_IF_ENABLED(type) + sizeof(step) + sizeof(fill) + sizeof(color))), \
        TracyPlotConfig(name, type, step, fill, color))
