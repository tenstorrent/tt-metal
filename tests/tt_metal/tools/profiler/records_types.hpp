// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The data and metadata types the records test's kernels publish and its host callback reads. Compiled by the
// device compiler for the kernels and by the host compiler for the callback.
#pragma once

#include <cstdint>

#include <hostdevcommon/streaming_profiler_types.hpp>

namespace records_test {
using tt::tt_metal::experimental::streaming_profiler::Interval;
using tt::tt_metal::experimental::streaming_profiler::String;
using tt::tt_metal::experimental::streaming_profiler::Timestamp;

// The ticks the data-movement kernels' long zone spins for, past a wrap of the clock's low word.
inline constexpr uint64_t kLongZoneTicks = (uint64_t{1} << 32) + (1u << 20);

enum class NocType : uint8_t { Read, Write };
struct CoreXY {
    uint16_t x, y;
};
struct NocEvent : CoreXY {
    uint32_t bytes;
};
struct NocSite {
    String api;
    NocType type;
    uint32_t color;
};

struct Name {
    String name;
};
struct Format {
    String format;
    String arguments;
};

enum class PlotFormat : uint8_t { Number, Memory };
struct PlotStyle {
    String name;
    PlotFormat format;
    uint32_t color;
};
struct Sample {
    int64_t value;
};
struct TileStats {
    uint32_t tiles, bytes, cycles, stalls;
};

struct From {
    String tag;
};

struct Burst {
    uint32_t sizes[4];
    char tag[4];
};

struct Bf16 {
    uint16_t bits;
};

// Fields that end short of a multiple of the struct's alignment.
struct Ragged {
    uint16_t a;
    uint8_t b;
};

struct Stamped {
    Timestamp at;
    uint32_t iteration;
};

// Bit-fields that share a halfword, after a String.
struct Hole {
    uint64_t iteration;
    String name;
    uint16_t a : 3;
    uint16_t b : 5;
};

// A uint16_t bit-field in the byte right after a uint8_t.
struct TightBits {
    uint32_t iteration;
    uint8_t a;
    uint16_t b : 8;
};
// A field that alignas moves past where its field types alone would end the struct.
struct LateField {
    uint64_t t;
    uint8_t a;
    alignas(4) uint8_t b;
};
// A long is 32 bits on the device and 64 on the host, so the host refuses a kernel that sends one in a type it takes.
struct LongField {
    long v;
};

// Structs whose field types alone don't predict their layout.
struct LaneFlags {
    uint32_t kind : 3;
    uint32_t lane : 5;
    uint32_t count : 24;
};
struct TaggedFlags {
    String name;
    uint32_t kind : 3;
    int32_t delta : 5;
};
struct __attribute__((packed)) Packed {
    uint8_t tag;
    uint32_t value;
};
struct Wide {
    alignas(16) uint32_t a;
    uint32_t b;
};
struct Empty {};
struct WithEmpty {
    Empty empty;
    uint32_t value;
};
struct Base0 {
    uint32_t x;
};
struct Base1 : Base0 {
    uint16_t y;
};
struct Deep : Base1 {
    uint16_t z;
};
// Laid out differently by kernels and the host.
struct Skewed {
    uint32_t a : 8;
#if !defined(KERNEL_BUILD)
    uint32_t : 8;
#endif
    uint32_t b : 8;
};

}  // namespace records_test
