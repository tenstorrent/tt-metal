// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Debug event metadata sites for tests/tt_metal/tt_metal/debug_tools/debug_event/test_debug_event_meta_compile.cpp,
// one case per DEBUG_EVENT_META_CASE. Case 0 is valid in every profiler mode; every other case breaks exactly one
// rule of hostdev/debug_event_meta.h and must fail to compile under the streaming profiler, with its own message.
// The zone macros take only their fixed library types, so a case with a metadata type of its own places its site
// through the metadata API, TT_DEBUG_SITE, directly.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "hostdev/observability_event_id.h"

#if DEBUG_EVENT_META_CASE == 0  // valid: built-in and nested types, an enum, a multi-line string, a template site

struct Style {
    uint32_t color;
    uint8_t verbosity;
};
struct NestedMeta {
    const char* name;
    Style style;
    uint16_t flags;
};
enum class Level : uint8_t { Low, High };
struct LevelMeta {
    const char* name;
    Level level;
};
struct NoteMeta {
    const char* name;
    const char* note;
};

template <uint32_t Color>
void colored_zone() {
    DeviceZoneScopedNC("templated", Color);
}

void kernel_main() {
    {
        DeviceZoneScopedN("plain");
    }
    {
        DeviceZoneScopedNC("colored", 0x42D4F4);
    }
    {
        DeviceZoneScopedNCU("unit", 0x42D4F4, kernel_profiler::TimeUnit::Microseconds);
    }
    {
        TT_DEBUG_SITE(site, NestedMeta{"nested", {0xA9A9A9, 2}, 3});
        (void)site::id();
    }
    {
        TT_DEBUG_SITE(site, LevelMeta{"level", Level::High});
        (void)site::id();
    }
    {
        TT_DEBUG_SITE(site, NoteMeta{"note", "two\nlines, a\ttab, \"quotes\", \\ and \xC3\xA9"});
        (void)site::id();
    }
    colored_zone<0xFF0000>();
    colored_zone<0x00FF00>();
}

#elif DEBUG_EVENT_META_CASE == 1  // a C array member

struct ArrayMeta {
    const char* name;
    uint8_t tags[3];
};
void kernel_main() {
    TT_DEBUG_SITE(site, ArrayMeta{"x", {1, 2, 3}});
    (void)site::id();
}

#elif DEBUG_EVENT_META_CASE == 2  // a C array inside a nested struct

struct Inner {
    uint8_t bytes[2];
};
struct NestedArrayMeta {
    const char* name;
    Inner inner;
};
void kernel_main() {
    TT_DEBUG_SITE(site, NestedArrayMeta{"x", {{1, 2}}});
    (void)site::id();
}

#elif DEBUG_EVENT_META_CASE == 3  // a const member

struct ConstMeta {
    const char* name;
    const uint32_t color;
};
void kernel_main() {
    TT_DEBUG_SITE(site, ConstMeta{"x", 1});
    (void)site::id();
}

#elif DEBUG_EVENT_META_CASE == 4  // a float field

struct FloatMeta {
    const char* name;
    float scale;
};
void kernel_main() {
    TT_DEBUG_SITE(site, FloatMeta{"x", 1.0f});
    (void)site::id();
}

#elif DEBUG_EVENT_META_CASE == 5  // a base class

struct Base {
    const char* name;
};
struct DerivedMeta : Base {
    uint32_t color;
};
void kernel_main() {
    TT_DEBUG_SITE(site, DerivedMeta{{"x"}, 1});
    (void)site::id();
}

#elif DEBUG_EVENT_META_CASE == 6  // the first field is not the name

struct ColorFirstMeta {
    uint32_t color;
    const char* name;
};
void kernel_main() {
    TT_DEBUG_SITE(site, ColorFirstMeta{1, "x"});
    (void)site::id();
}

#elif DEBUG_EVENT_META_CASE == 7  // an empty name

void kernel_main() { DeviceZoneScopedN(""); }

#elif DEBUG_EVENT_META_CASE == 8  // a type in an anonymous namespace

namespace {
struct HiddenMeta {
    const char* name;
};
}  // namespace
void kernel_main() {
    TT_DEBUG_SITE(site, HiddenMeta{"x"});
    (void)site::id();
}

#elif DEBUG_EVENT_META_CASE == 9  // a type local to a function

void kernel_main() {
    struct LocalMeta {
        const char* name;
    };
    TT_DEBUG_SITE(site, LocalMeta{"x"});
    (void)site::id();
}

#elif DEBUG_EVENT_META_CASE == 10  // a run-time value

void kernel_main() {
    const uint32_t color = get_arg_val<uint32_t>(0);
    DeviceZoneScopedNC("x", color);
}

#elif DEBUG_EVENT_META_CASE == 11  // more than kMaxFields fields

struct WideMeta {
    const char* name;
    uint8_t a, b, c, d, e, f, g, h;
};
void kernel_main() {
    TT_DEBUG_SITE(site, WideMeta{"x", 1, 2, 3, 4, 5, 6, 7, 8});
    (void)site::id();
}

#elif DEBUG_EVENT_META_CASE == 12  // strings longer than one site's assembler text

#define DEM_S10 "0123456789"
#define DEM_S100 DEM_S10 DEM_S10 DEM_S10 DEM_S10 DEM_S10 DEM_S10 DEM_S10 DEM_S10 DEM_S10 DEM_S10
#define DEM_S1000 DEM_S100 DEM_S100 DEM_S100 DEM_S100 DEM_S100 DEM_S100 DEM_S100 DEM_S100 DEM_S100 DEM_S100
void kernel_main() { DeviceZoneScopedN(DEM_S1000 DEM_S1000 DEM_S100); }

#else
#error "unknown DEBUG_EVENT_META_CASE"
#endif
