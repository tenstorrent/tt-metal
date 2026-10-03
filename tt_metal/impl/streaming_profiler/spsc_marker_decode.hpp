// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The vector kernels of the host-side decode and the per-lane state they run against; the frame walk that drives
// them is StreamDecoder::decode_frames (decode.cpp). A frame is the SPSC_SPAN_PREFIX_WORDS prefix (word 1 = payload
// length), the SPSC_SPAN_WIRE_CTRL_WORDS control block, then each RISC's live ring window packed flat with congruence
// pads and wraps resolved device-side. Packet formats: spsc_packet.h. The producer publishes its tail only on
// packet boundaries, so a window never ends mid-packet.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>

// SIMDe uses native AVX2 where available and supported host instructions otherwise.
#include <simde/x86/avx2.h>

#include "hostdev/streaming_profiler_common.h"
#include "spsc_packet.h"

static_assert(
    PP_BULK_SPAN == kernel_profiler::SPSC_SPAN_PACKET_TYPE,
    "spsc_packet.h (plain C, host decoder) and profiler_common.h (C++, relay and metal kernels) must agree on the "
    "BULK_SPAN wire code -- they cannot include each other, so this is the only thing holding them together");
// Same argument for the codes profiler_common.h also names.
static_assert(PP_STICKY_TIMER == kernel_profiler::SPSC_TYPE_STICKY_TIMER, "STICKY_TIMER wire code disagrees");
static_assert(PP_ZONE_L == kernel_profiler::SPSC_TYPE_ZONE_L, "ZONE_L wire code disagrees");
static_assert(PP_TYPE_SHIFT == kernel_profiler::SPSC_SPAN_TYPE_SHIFT, "packet type field moved");
// The relay kernel keeps its own copy of the PP_DATA packer (it cannot include kernel_profiler.hpp); a layout
// drift renders every relay marker under the wrong identity with no crash, and this is the only TU that sees
// both headers.
static_assert(PP_DATA == kernel_profiler::SPSC_TYPE_DATA, "PP_DATA wire code disagrees");
static_assert(PP_DATA_SIZE_SHIFT == kernel_profiler::SPSC_DATA_SIZE_SHIFT, "PP_DATA size field moved");

namespace tt::tt_metal::profiler {

inline constexpr uint32_t kSpscRingCap = kernel_profiler::PROFILER_L1_VECTOR_SIZE;
inline constexpr uint32_t kSpscRingMask = kSpscRingCap - 1;
inline constexpr uint32_t kSpscNRiscDecode = kernel_profiler::PROFILER_SPSC_TENSIX_RISC;
inline constexpr uint32_t kSpscStallZoneId = TT_ZONE_STALL_ID;

// Worst case: five full rings behind maximal pads. Bounds the bounce buffer and frame validation, not any
// device layout.
inline constexpr uint32_t kSpscMaxPayloadWords =
    kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE +
    kSpscNRiscDecode * (kSpscRingCap + kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS - 1);
inline constexpr uint32_t kSpscMaxFrameWords = kernel_profiler::spsc_span_frame_words(kSpscMaxPayloadWords);
inline constexpr uint32_t kSpscMaxFramePages = kSpscMaxFrameWords / kernel_profiler::SPSC_SPAN_PAGE_WORDS;
static_assert(kSpscMaxFrameWords == 2656 && kSpscMaxFramePages == 166);

// Every record a consumer sees is a public record (experimental::streaming_profiler::Zone / Event /
// TimestampedData), written straight into its kind's region of the batch buffer. Each starts with the kSpscSharedBytes
// every kind shares, {start|ts, zone id | runtime id << 32, coordinates | chip | processor, the lane's tile offset},
// one vector composed from the packet and the lane's constants. A zone follows it with its duration and an end_tsc_
// slot the service fills, Data with its value count and a pointer to its values, which go to the batch's values region.
// Stores are cached rather than NT because the consumer reads the buffers immediately. Block kernels write their last
// quad whole, so a region needs kSpscSinkSlackRecs of slack past cap.
inline constexpr uint32_t kSpscSharedBytes = 32;
inline constexpr uint32_t kSpscEventBytes = kSpscSharedBytes;
inline constexpr uint32_t kSpscDataBytes = 48;
inline constexpr uint32_t kSpscZoneBytes = 48;
inline constexpr uint32_t kSpscSinkSlackRecs = 8;
inline constexpr uint32_t kSpscQwTimestamp = 0, kSpscQwIds = 1, kSpscQwTsc = 3, kSpscQwDuration = 4, kSpscQwEndTsc = 5;
// A lane's record constants in record byte order: qword 2, its coordinates, chip and processor, then qword 3, its
// tile offset.
struct SpscRecConsts {
    uint32_t coords[2];
    uint64_t offset;
};

// A ZONE_L duration's high word when the zone's start read borrowed the next epoch.
inline constexpr uint32_t kSpscBorrowedDurHi = 0xFFFFFFFFu;

// What a block kernel reports: 16 bytes so it returns in registers; the caller knows the first record's timestamp
// from the words. The flags say only that something needs the slow path; the caller finds where.
struct SpscBlockResult {
    uint64_t ts_last;  // ZONE_S: the lane cursor after the block
    uint32_t n;
    uint8_t regress;   // some record's timestamp precedes the one before it
    uint8_t wrapped;   // ZONE_L only: some duration's high word is kSpscBorrowedDurHi
    uint16_t stalls;   // records whose id is kSpscStallZoneId
};

// Packet formats. Every packet the wire carries is one row below; word 0 of every packet is type | id27. The block
// kernel, the single-record composer and the walk's dispatch are instantiated from the row, so a new fixed-size
// packet is a new row and nothing else on the host.

inline constexpr uint8_t kAbsent = 0xFF;

struct PacketFormat {
    enum class Kind : uint8_t {
        Zone,    // a {start, duration} record
        Point,   // a {timestamp} record with no payload
        Data,    // a {timestamp} record with a payload of size_word's count of words behind the fixed head
        Sticky,  // sets a lane field (the timer high half or the program id); emits nothing
    };
    enum class Sets : uint8_t { Nothing, TimerHi, Prog };

    uint8_t type = 0;   // wire code (PP_*)
    uint8_t words = 0;  // the fixed part; a Data payload follows it
    Kind kind = Kind::Zone;
    // Field positions as word indices into the packet.
    uint8_t ts_lo = kAbsent;
    uint8_t ts_hi = kAbsent;  // absent: the lane's sticky high half
    uint8_t dur_lo = kAbsent;
    uint8_t dur_hi = kAbsent;  // absent: a 32-bit duration
    // A 2-word zone: word 1 = end_delta16 << 16 | dur16, the end relative to the lane cursor.
    bool delta16 = false;
    bool reanchor = false;  // a decoded end becomes the lane cursor
    // Data: the payload word count is (word[size_word] >> size_shift) & size_mask.
    uint8_t size_word = kAbsent;
    uint8_t size_shift = 0;
    uint8_t size_mask = 0;
    // Sticky: the field set from word `value_word` (word 0 contributes its low 27 bits, any other word all 32).
    Sets sets = Sets::Nothing;
    uint8_t value_word = 0;

    constexpr bool has_ts_hi() const { return ts_hi != kAbsent; }
    constexpr bool has_dur() const { return dur_lo != kAbsent; }
    constexpr bool has_dur_hi() const { return dur_hi != kAbsent; }
};
using Kind = PacketFormat::Kind;

inline constexpr PacketFormat kZoneS{
    .type = PP_ZONE_S, .words = 2, .kind = Kind::Zone, .delta16 = true, .reanchor = true};
inline constexpr PacketFormat kZoneAtomic{
    .type = PP_ZONE_ATOMIC, .words = 3, .kind = Kind::Zone, .ts_lo = 1, .dur_lo = 2, .reanchor = true};
inline constexpr PacketFormat kEvent{.type = PP_EVENT, .words = 2, .kind = Kind::Point, .ts_lo = 1};
inline constexpr PacketFormat kData{
    .type = PP_DATA,
    .words = 3,
    .kind = Kind::Data,
    .ts_lo = 1,
    .size_word = 2,
    .size_shift = PP_DATA_SIZE_SHIFT,
    .size_mask = PP_DATA_SIZE_MASK};
inline constexpr PacketFormat kZoneL{
    .type = PP_ZONE_L, .words = 5, .kind = Kind::Zone, .ts_lo = 1, .ts_hi = 2, .dur_lo = 3, .dur_hi = 4};
inline constexpr PacketFormat kStickyTimer{
    .type = PP_STICKY_TIMER, .words = 1, .kind = Kind::Sticky, .sets = PacketFormat::Sets::TimerHi, .value_word = 0};
inline constexpr PacketFormat kStickyProg{
    .type = PP_STICKY_PROG, .words = 1, .kind = Kind::Sticky, .sets = PacketFormat::Sets::Prog, .value_word = 0};
inline constexpr PacketFormat kStickyProgExt{
    .type = PP_STICKY_PROG_EXT, .words = 2, .kind = Kind::Sticky, .sets = PacketFormat::Sets::Prog, .value_word = 1};

// The walk tests types in this order, so the common ones come first.
inline constexpr std::array<PacketFormat, 8> kFormats = {
    kZoneS, kZoneAtomic, kEvent, kData, kZoneL, kStickyTimer, kStickyProg, kStickyProgExt};

constexpr bool spsc_formats_ok() {
    uint32_t seen = 0, data_rows = 0;
    uint8_t point_ts_lo = kAbsent;
    for (const PacketFormat& f : kFormats) {
        if (((seen >> f.type) & 1u) != 0) {
            return false;
        }
        seen |= 1u << f.type;
        const auto in_packet = [&](uint8_t w) { return w == kAbsent || (w > 0 && w < f.words); };
        if (!in_packet(f.ts_lo) || !in_packet(f.ts_hi) || !in_packet(f.dur_lo) || !in_packet(f.dur_hi)) {
            return false;
        }
        if (f.has_dur_hi() && !f.has_dur()) {
            return false;
        }
        switch (f.kind) {
            case Kind::Zone:
                // A 2-word zone can only be delta-encoded; a longer one carries its fields in place.
                if (f.delta16 != (f.words == 2) ||
                    (f.delta16 ? f.ts_lo != kAbsent : (f.ts_lo == kAbsent || !f.has_dur()))) {
                    return false;
                }
                break;
            case Kind::Point:
            case Kind::Data:
                // Points share one composer, so every point kind keeps its timestamp in the same word.
                if (f.ts_lo == kAbsent || f.has_dur() || f.delta16 ||
                    (point_ts_lo != kAbsent && point_ts_lo != f.ts_lo)) {
                    return false;
                }
                point_ts_lo = f.ts_lo;
                if (f.kind == Kind::Data) {
                    data_rows++;
                    if (f.size_word == kAbsent || f.size_word == 0 || f.size_word >= f.words || f.size_mask == 0) {
                        return false;
                    }
                }
                break;
            case Kind::Sticky:
                if (f.sets == PacketFormat::Sets::Nothing || f.value_word >= f.words) {
                    return false;
                }
                break;
        }
    }
    return data_rows == 1;
}
static_assert(spsc_formats_ok(), "kFormats: a row is inconsistent (see spsc_formats_ok)");

// Per-type lookups for the point path, which selects nothing by branching on the type.
inline constexpr PacketFormat kSpscDataFormat = [] {
    for (const PacketFormat& f : kFormats) {
        if (f.kind == Kind::Data) {
            return f;
        }
    }
    return PacketFormat{};
}();
inline constexpr uint32_t kSpscPointTypes = [] {
    uint32_t m = 0;
    for (const PacketFormat& f : kFormats) {
        if (f.kind == Kind::Point || f.kind == Kind::Data) {
            m |= 1u << f.type;
        }
    }
    return m;
}();
inline constexpr std::array<uint8_t, PP_TYPE_MASK + 1> kSpscWordsOfType = [] {
    std::array<uint8_t, PP_TYPE_MASK + 1> a{};
    for (const PacketFormat& f : kFormats) {
        a[f.type] = f.words;
    }
    return a;
}();
inline constexpr std::array<uint32_t, PP_TYPE_MASK + 1> kSpscDataMaskOfType = [] {
    std::array<uint32_t, PP_TYPE_MASK + 1> a{};
    for (const PacketFormat& f : kFormats) {
        if (f.kind == Kind::Data) {
            a[f.type] = 0xFFFFFFFFu;
        }
    }
    return a;
}();

constexpr bool spsc_is_point(Kind k) { return k == Kind::Point; }
constexpr bool spsc_is_zone_or_sticky(Kind k) { return k == Kind::Zone || k == Kind::Sticky; }
constexpr bool spsc_is_sticky(Kind k) { return k == Kind::Sticky; }

// Runs handle.template operator()<F>() for the row of kFormats whose wire type is t and whose kind Pred accepts;
// false when there is none. The chain unrolls in table order. always_inline, like the handlers it takes: a handler
// left out of line captures the walk's locals by address and pushes them all out of registers.
template <bool (*Pred)(Kind), size_t I = 0, typename H>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward) -- Reuse the callback as an lvalue through recursive calls.
__attribute__((always_inline)) inline bool spsc_for_format(uint32_t t, H&& handle) {
    if constexpr (I == kFormats.size()) {
        return false;
    } else if constexpr (!Pred(kFormats[I].kind)) {
        return spsc_for_format<Pred, I + 1>(t, handle);
    } else {
        if (t == kFormats[I].type) {
            handle.template operator()<kFormats[I]>();
            return true;
        }
        return spsc_for_format<Pred, I + 1>(t, handle);
    }
}
// Runs handle.template operator()<F>() for every row whose kind Pred accepts, in table order.
template <bool (*Pred)(Kind), size_t I = 0, typename H>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward) -- Reuse the callback as an lvalue through recursive calls.
__attribute__((always_inline)) inline void spsc_for_each_format(H&& handle) {
    if constexpr (I < kFormats.size()) {
        if constexpr (Pred(kFormats[I].kind)) {
            handle.template operator()<kFormats[I]>();
        }
        spsc_for_each_format<Pred, I + 1>(handle);
    }
}

// Record k's timestamp in a run of F packets starting at src.
template <PacketFormat F>
inline uint64_t spsc_ts_at(const uint32_t* src, uint32_t k, uint64_t th_hi) {
    const uint32_t* r = src + F.words * k;
    if constexpr (F.has_ts_hi()) {
        return (static_cast<uint64_t>(r[F.ts_hi]) << 32) | r[F.ts_lo];
    } else {
        return th_hi | r[F.ts_lo];
    }
}

// Everything a lane's records share: the constant part of the shared 32 bytes in dwords, {0, th, 0, prog, coords,
// offset}, which `shared_ts64` holds without th for the 64-bit-timestamp kinds, and the broadcasts the 2-word kernels
// compose from.
// Built at lane entry; a STICKY_TIMER re-blends only the th lanes, a STICKY_PROG only the prog lanes.
struct SpscLaneConsts {
    uint64_t th_hi;
    simde__m256i th_hi_v, prog_hi_v;
    simde__m256i tail;    // {coords, offset} in each 128-bit half
    simde__m256i shared, shared_ts64;
};
inline void spsc_lane_consts_th(SpscLaneConsts& c, uint32_t th) {
    c.th_hi = static_cast<uint64_t>(th) << 32;
    c.th_hi_v = simde_mm256_set1_epi64x(static_cast<long long>(c.th_hi));
    c.shared = simde_mm256_blend_epi32(c.shared, simde_mm256_set1_epi32(static_cast<int>(th)), 0x02);
}
inline void spsc_lane_consts_prog(SpscLaneConsts& c, uint32_t prog) {
    c.prog_hi_v = simde_mm256_set1_epi64x(static_cast<long long>(static_cast<uint64_t>(prog) << 32));
    const simde__m256i prog_v = simde_mm256_set1_epi32(static_cast<int>(prog));
    c.shared = simde_mm256_blend_epi32(c.shared, prog_v, 0x08);
    c.shared_ts64 = simde_mm256_blend_epi32(c.shared_ts64, prog_v, 0x08);
}
inline void spsc_lane_consts(SpscLaneConsts& c, const SpscRecConsts& r, uint32_t th, uint32_t prog) {
    const long long coords = static_cast<long long>(r.coords[0] | (static_cast<uint64_t>(r.coords[1]) << 32));
    const long long offset = static_cast<long long>(r.offset);
    c.tail = simde_mm256_setr_epi64x(coords, offset, coords, offset);
    c.shared = c.shared_ts64 = simde_mm256_setr_epi64x(0, 0, coords, offset);
    spsc_lane_consts_th(c, th);
    spsc_lane_consts_prog(c, prog);
}
// The constant part of an F record: its th lane is blended in only when the timestamp's high half comes from the
// lane's sticky.
template <PacketFormat F>
inline simde__m256i spsc_shared(const SpscLaneConsts& c) {
    return F.has_ts_hi() ? c.shared_ts64 : c.shared;
}
template <PacketFormat F>
inline constexpr uint32_t spsc_rec_bytes = F.kind == Kind::Zone ? kSpscZoneBytes : kSpscEventBytes;

// The vector constants the kernels share; all fold to immediates once inlined.
inline simde__m256i spsc_dw_type_mask() {
    return simde_mm256_set1_epi32(static_cast<int>((0xFFFFFFFFu >> PP_TYPE_SHIFT) << PP_TYPE_SHIFT));
}
inline simde__m256i spsc_dw_type(uint32_t type) {
    return simde_mm256_set1_epi32(static_cast<int>(type << PP_TYPE_SHIFT));
}
inline simde__m256i spsc_qw_type_mask() {
    return simde_mm256_set1_epi64x(static_cast<long long>((0xFFFFFFFFull >> PP_TYPE_SHIFT) << PP_TYPE_SHIFT));
}
inline simde__m256i spsc_qw_type(uint32_t type) {
    return simde_mm256_set1_epi64x(static_cast<long long>(static_cast<uint64_t>(type) << PP_TYPE_SHIFT));
}
inline simde__m256i spsc_lane_idx() { return simde_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7); }
inline simde__m256i spsc_lane0() { return simde_mm256_setr_epi32(-1, 0, 0, 0, 0, 0, 0, 0); }

// A 32 B load holds spsc_recs_per_load<F> whole F packets at word stride F.words; these are its lane constants.
template <PacketFormat F>
inline constexpr uint32_t spsc_recs_per_load = 8 / F.words;
// -1 at word `w` of each whole packet in the load.
template <PacketFormat F, uint8_t w>
inline simde__m256i spsc_lanes_at() {
    constexpr auto lane = [](int index) {
        return (index % F.words == w && index / F.words < static_cast<int>(spsc_recs_per_load<F>)) ? -1 : 0;
    };
    return simde_mm256_setr_epi32(lane(0), lane(1), lane(2), lane(3), lane(4), lane(5), lane(6), lane(7));
}
// Word 0 of each whole packet down to its 27-bit id, every other lane kept.
template <PacketFormat F>
inline simde__m256i spsc_w0_mask() {
    constexpr auto lane = [](int index) {
        return (index % F.words == 0 && index / F.words < static_cast<int>(spsc_recs_per_load<F>))
                   ? static_cast<int>(PP_LOW27_MASK)
                   : -1;
    };
    return simde_mm256_setr_epi32(lane(0), lane(1), lane(2), lane(3), lane(4), lane(5), lane(6), lane(7));
}
// The lanes of the shared 32 bytes that come from the constant part rather than the packet: prog, the coordinates and
// the offset always, plus the timestamp's high half when the format lacks it.
template <PacketFormat F>
constexpr int spsc_blend_s() {
    return 0xF8 | (F.has_ts_hi() ? 0 : 0x02);
}
template <PacketFormat F>
constexpr int spsc_blend_d() {
    return 0xFC | (F.has_dur_hi() ? 0 : 0x02);
}

// A composed record's 64-bit end, which ordering compares, and its duration, zero for a point; each in lane 0.
struct SpscComposed {
    simde__m256i end, duration;
};
// Composes the load's `Packet`-th packet into the public record at `record`: a lane permute puts {ts_lo, ts_hi, id} in
// place, a blend supplies the constant part, and subtracting the duration from lane 0 turns the end into the start; a
// zone's duration follows.
template <PacketFormat F, uint32_t Packet>
inline SpscComposed spsc_compose(simde__m256i loaded, simde__m256i shared, uint8_t* record) {
    constexpr int base = static_cast<int>(Packet * F.words);
    constexpr auto at = [](uint8_t word) { return word == kAbsent ? 0 : base + word; };
    const simde__m256i composed = simde_mm256_blend_epi32(
        simde_mm256_permutevar8x32_epi32(loaded, simde_mm256_setr_epi32(at(F.ts_lo), at(F.ts_hi), base, 0, 0, 0, 0, 0)),
        shared,
        spsc_blend_s<F>());
    simde__m256i duration = simde_mm256_setzero_si256();
    if constexpr (F.has_dur()) {
        duration = simde_mm256_blend_epi32(
            simde_mm256_permutevar8x32_epi32(
                loaded, simde_mm256_setr_epi32(at(F.dur_lo), at(F.dur_hi), 0, 0, 0, 0, 0, 0)),
            duration,
            spsc_blend_d<F>());
        simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(record), simde_mm256_sub_epi64(composed, duration));
    } else {
        simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(record), composed);
    }
    if constexpr (F.kind == Kind::Zone) {
        simde_mm_storel_epi64(
            reinterpret_cast<simde__m128i*>(record + kSpscSharedBytes), simde_mm256_castsi256_si128(duration));
    }
    return {composed, duration};
}

// Eight words from p with the lanes past `readable` zero; a count of zero or less loads nothing. A record's words
// are always in range, the load's tail may not be.
inline simde__m256i spsc_words(const uint32_t* p, int32_t readable) {
    if (readable >= 8) {
        return simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(p));
    }
    return simde_mm256_maskload_epi32(
        reinterpret_cast<const int*>(p), simde_mm256_cmpgt_epi32(simde_mm256_set1_epi32(readable), spsc_lane_idx()));
}
// Records of the type in a quad's compare result, counted from lane 0 to the first miss. A full quad is one test;
// only a partial one extracts its mask.
inline uint32_t spsc_quad_count(simde__m256i c) {
    if (simde_mm256_testc_si256(c, simde_mm256_cmpeq_epi64(c, c))) {
        return 4;
    }
    return std::countr_zero(~static_cast<uint32_t>(simde_mm256_movemask_pd(simde_mm256_castsi256_pd(c))) & 0x1Fu);
}
// Four records' shared 32 bytes at stride `Stride` from per-qword-lane halves: {times, ids} is a record's first 16
// bytes, start|ts and id | prog << 32, and the lane's `tail` its last 16. Written whole; a partial quad's spare records
// land in the sink's slack.
template <uint32_t Stride>
inline void spsc_store_quad(uint8_t* out, simde__m256i times, simde__m256i ids, simde__m256i tail) {
    const simde__m256i pairs_lo = simde_mm256_unpacklo_epi64(times, ids);
    const simde__m256i pairs_hi = simde_mm256_unpackhi_epi64(times, ids);
    simde_mm256_storeu_si256(
        reinterpret_cast<simde__m256i*>(out), simde_mm256_permute2x128_si256(pairs_lo, tail, 0x20));
    simde_mm256_storeu_si256(
        reinterpret_cast<simde__m256i*>(out + Stride), simde_mm256_permute2x128_si256(pairs_hi, tail, 0x20));
    simde_mm256_storeu_si256(
        reinterpret_cast<simde__m256i*>(out + 2 * Stride), simde_mm256_permute2x128_si256(pairs_lo, tail, 0x21));
    simde_mm256_storeu_si256(
        reinterpret_cast<simde__m256i*>(out + 3 * Stride), simde_mm256_permute2x128_si256(pairs_hi, tail, 0x21));
}
inline void spsc_store_durations(uint8_t* out, simde__m256i durations) {
    constexpr uint32_t kStride = kSpscZoneBytes;
    const simde__m128i lo = simde_mm256_castsi256_si128(durations);
    const simde__m128i hi = simde_mm256_extracti128_si256(durations, 1);
    simde_mm_storel_epi64(reinterpret_cast<simde__m128i*>(out + kSpscSharedBytes), lo);
    simde_mm_storeh_pd(reinterpret_cast<double*>(out + kStride + kSpscSharedBytes), simde_mm_castsi128_pd(lo));
    simde_mm_storel_epi64(reinterpret_cast<simde__m128i*>(out + 2 * kStride + kSpscSharedBytes), hi);
    simde_mm_storeh_pd(reinterpret_cast<double*>(out + 3 * kStride + kSpscSharedBytes), simde_mm_castsi128_pd(hi));
}

// One F record from its words, reported like a run of one.
template <PacketFormat F>
inline SpscBlockResult spsc_one(const uint32_t* p, uint32_t readable, const SpscLaneConsts& c, uint8_t* dst) {
    spsc_compose<F, 0>(
        simde_mm256_and_si256(spsc_words(p, static_cast<int32_t>(readable)), spsc_w0_mask<F>()),
        spsc_shared<F>(c),
        dst);
    SpscBlockResult out{spsc_ts_at<F>(p, 0, c.th_hi), 1, 0, 0, 0};
    if constexpr (F.kind == Kind::Zone) {
        out.stalls = (p[0] & PP_LOW27_MASK) == kSpscStallZoneId ? 1u : 0u;
    }
    if constexpr (F.has_dur_hi()) {
        out.wrapped = p[F.dur_hi] == kSpscBorrowedDurHi ? 1u : 0u;
    }
    return out;
}

// Four F records start at p (readable >= 8 words): the gate for the 2-word block kernel, so random single points
// never pay a block call and a run never pays the one-record path.
template <PacketFormat F>
inline bool spsc_run4(const uint32_t* p) {
    static_assert(F.words == 2);
    const simde__m256i v = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(p));
    const simde__m256i c = simde_mm256_cmpeq_epi64(simde_mm256_and_si256(v, spsc_qw_type_mask()), spsc_qw_type(F.type));
    return simde_mm256_testc_si256(c, simde_mm256_cmpeq_epi64(c, c));
}

// noinline on both block kernels: called once per run, and inlined into the walk they share its register file and
// spill inside their loops (+11-14% on the run shapes).
// A run of F records of three words or more, spsc_recs_per_load<F> per 32 B load at the packet's word stride so
// every field sits at a fixed dword; each record composes through spsc_compose. Consumes every consecutive record
// in one call, so a run pays the walk's per-call cost once. Compare results stay in vectors and are tested once; a
// mask is only extracted when a test fires, since each vector-to-scalar move costs more than the compose itself.
// `avail` authorizes loads, never emits.
template <PacketFormat F>
__attribute__((noinline)) inline SpscBlockResult spsc_block_strided(
    const uint32_t* p, uint32_t avail, uint32_t max_recs, const SpscLaneConsts& c, uint8_t* dst) {
    constexpr uint32_t W = F.words;
    constexpr uint32_t R = spsc_recs_per_load<F>;
    static_assert(R == 1 || R == 2);
    // Two records per load batch four loads ahead of their compares; one record per load composes as it goes, the
    // four-load batch measured 5% slower there.
    constexpr uint32_t kLoads = R == 2 ? 4 : 1;
    constexpr uint32_t kStride = R * W;
    constexpr uint32_t kBlockWords = kLoads * kStride;
    constexpr uint32_t kBlockRecs = kLoads * R;
    constexpr uint32_t kFullAvail = kBlockWords - kStride + 8;  // every load of the block whole
    SpscBlockResult out{0, 0, 0, 0, 0};
    const simde__m256i type_mask = spsc_dw_type_mask();
    const simde__m256i ftype = spsc_dw_type(F.type);
    const simde__m256i lanes_w0 = spsc_lanes_at<F, 0>();
    const simde__m256i lane_0 = spsc_lane0();
    const simde__m256i lane_1 = simde_mm256_setr_epi32(0, -1, 0, 0, 0, 0, 0, 0);
    const simde__m256i w0_mask = spsc_w0_mask<F>();
    const simde__m256i borrowed = simde_mm256_set1_epi32(static_cast<int>(kSpscBorrowedDurHi));
    const simde__m256i shared = spsc_shared<F>(c);
    constexpr uint32_t kRec = spsc_rec_bytes<F>;
    const simde__m256i z = simde_mm256_setzero_si256();
    const uint64_t th_hi = c.th_hi;
    const uint32_t* const p0 = p;
    const simde__m256i stall = simde_mm256_set1_epi32(static_cast<int>(kSpscStallZoneId));
    simde__m256i prev = z, back = z, stall_hit = z, wrap_hit = z;
    uint32_t total = 0;
    while (max_recs != 0 && avail >= W) {
        // One load at a time, stopping at the first whose records are not all F: a lone record costs one load.
        simde__m256i v[kLoads];
        uint32_t n = 0;
        for (uint32_t i = 0; i < kLoads; i++) {
            if (avail >= kFullAvail) {
                v[i] = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(p + i * kStride));
            } else {
                const simde__m256i mk = simde_mm256_cmpgt_epi32(
                    simde_mm256_set1_epi32(static_cast<int>(avail) - static_cast<int>(i * kStride)), spsc_lane_idx());
                v[i] = simde_mm256_maskload_epi32(reinterpret_cast<const int*>(p + i * kStride), mk);
            }
            const simde__m256i cm = simde_mm256_cmpeq_epi32(simde_mm256_and_si256(v[i], type_mask), ftype);
            if (!simde_mm256_testc_si256(cm, lanes_w0)) {
                if constexpr (R == 2) {
                    n += simde_mm256_testc_si256(cm, lane_0) ? 1u : 0u;
                }
                break;
            }
            n += R;
        }
        n = std::min(n, max_recs);
        if (n == 0) {
            break;
        }
        // Lane 0 of a composed record is its 64-bit end, so a signed 64-bit compare orders records, `prev` carrying
        // across loads and blocks. Dword 1 of its duration is the high word when the format has one.
        uint32_t i = 0;
        for (; (i + 1) * R <= n; i++) {
            const simde__m256i l = simde_mm256_and_si256(v[i], w0_mask);
            uint8_t* const record = dst + kRec * R * i;
            const SpscComposed first = spsc_compose<F, 0>(l, shared, record);
            if constexpr (R == 2) {
                const SpscComposed second = spsc_compose<F, 1>(l, shared, record + kRec);
                back = simde_mm256_or_si256(
                    back,
                    simde_mm256_or_si256(
                        simde_mm256_cmpgt_epi64(prev, first.end), simde_mm256_cmpgt_epi64(first.end, second.end)));
                prev = second.end;
                if constexpr (F.has_dur_hi()) {
                    wrap_hit = simde_mm256_or_si256(
                        wrap_hit,
                        simde_mm256_or_si256(
                            simde_mm256_cmpeq_epi32(first.duration, borrowed),
                            simde_mm256_cmpeq_epi32(second.duration, borrowed)));
                }
            } else {
                back = simde_mm256_or_si256(back, simde_mm256_cmpgt_epi64(prev, first.end));
                prev = first.end;
                if constexpr (F.has_dur_hi()) {
                    wrap_hit = simde_mm256_or_si256(wrap_hit, simde_mm256_cmpeq_epi32(first.duration, borrowed));
                }
            }
            stall_hit = simde_mm256_or_si256(stall_hit, simde_mm256_cmpeq_epi32(l, stall));
        }
        if constexpr (R == 2) {
            if (2 * i < n) {  // an odd last record: the load's second record is not ours
                const simde__m256i l = simde_mm256_and_si256(v[i], w0_mask);
                const SpscComposed last = spsc_compose<F, 0>(l, shared, dst + kRec * 2 * i);
                back = simde_mm256_or_si256(back, simde_mm256_cmpgt_epi64(prev, last.end));
                prev = last.end;
                if constexpr (F.has_dur_hi()) {
                    wrap_hit = simde_mm256_or_si256(wrap_hit, simde_mm256_cmpeq_epi32(last.duration, borrowed));
                }
                stall_hit =
                    simde_mm256_or_si256(stall_hit, simde_mm256_and_si256(simde_mm256_cmpeq_epi32(l, stall), lane_0));
            }
        }
        total += n;
        dst += kRec * n;
        if (n < kBlockRecs) {
            break;
        }
        p += kBlockWords;
        avail -= kBlockWords;
        max_recs -= kBlockRecs;
    }
    out.n = total;
    if (total != 0) {
        out.ts_last = spsc_ts_at<F>(p0, total - 1u, th_hi);
    }
    out.regress = simde_mm256_testz_si256(back, lane_0) ? 0u : 1u;
    if constexpr (F.has_dur_hi()) {
        out.wrapped = simde_mm256_testz_si256(wrap_hit, lane_1) ? 0u : 1u;
    }
    if (__builtin_expect(!simde_mm256_testz_si256(stall_hit, lanes_w0), 0)) {
        for (uint32_t k = 0; k < total; k++) {
            out.stalls += (p0[W * k] & PP_LOW27_MASK) == kSpscStallZoneId ? 1u : 0u;
        }
    }
    return out;
}

// A run of 2-word records on the wire's own layout: a record is one qword (w0 low, w1 high), so four records are one
// 256-bit load and each field is a shift or mask of its own lane; nothing deinterleaves or widens. A delta16 zone's
// ends are cursor-relative, so its end deltas take an inclusive prefix sum and the records normalize to the
// absolute form downstream sees; its ends are monotonic by construction, so only a plain (point) format tracks
// order. Consumes every consecutive full block in one call, so a dense lane pays the walk's per-call cost once per
// run rather than once per 16 records. `readable` authorizes loads, never emits, past the live run; `max_recs`
// bounds the emits.
template <PacketFormat F>
__attribute__((noinline)) inline SpscBlockResult spsc_block_qword(
    const uint32_t* p, uint32_t readable, uint32_t max_recs, uint64_t cursor, const SpscLaneConsts& c, uint8_t* dst) {
    static_assert(F.words == 2);
    constexpr bool kDelta = F.delta16;
    static_assert(kDelta ? F.kind == Kind::Zone : (F.kind == Kind::Point && F.ts_lo == 1));
    SpscBlockResult out{0, 0, 0, 0, 0};
    const simde__m256i type_mask = spsc_qw_type_mask();
    const simde__m256i ftype = spsc_qw_type(F.type);
    const simde__m256i z = simde_mm256_setzero_si256();
    const simde__m256i id_mask = simde_mm256_set1_epi64x(PP_LOW27_MASK);
    const simde__m256i dur_mask = simde_mm256_set1_epi64x(0xFFFF);
    const simde__m256i prog_hi_v = c.prog_hi_v, tail = c.tail;
    constexpr uint32_t kRec = spsc_rec_bytes<F>;
    // The cursor rides as a broadcast vector: the next block's starts need it as one, and the block total is already
    // a broadcast lane of the carry tree.
    simde__m256i cv = simde_mm256_set1_epi64x(static_cast<long long>(cursor));
    simde__m256i carry = z, back = z;
    uint32_t total = 0;
    while (max_recs != 0 && readable >= 2u) {
        simde__m256i v[4];
        if (readable >= 32u) {
            for (int i = 0; i < 4; i++) {
                v[i] = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(p + 8 * i));
            }
        } else {
            // Whole records only: a trailing odd word is never F, so its lane reads as zero and ends the scan.
            const int recs = static_cast<int>(readable / 2u);
            for (int i = 0; i < 4; i++) {
                const simde__m256i m =
                    simde_mm256_cmpgt_epi64(simde_mm256_set1_epi64x(recs - 4 * i), simde_mm256_setr_epi64x(0, 1, 2, 3));
                v[i] = simde_mm256_maskload_epi64(reinterpret_cast<const int64_t*>(p + 8 * i), m);
            }
        }
        if constexpr (kDelta) {
            // Both lines of the block four blocks ahead: the hardware prefetcher does not run far enough into
            // DMA-landed memory (-6%); farther ahead evicts before use once the sink's output stream shares L1, and
            // one line per block leaves every other line to the hardware.
            simde_mm_prefetch(reinterpret_cast<const char*>(p + 128), SIMDE_MM_HINT_T0);
            simde_mm_prefetch(reinterpret_cast<const char*>(p + 144), SIMDE_MM_HINT_T0);
        }
        uint32_t n = 0;
        for (const auto& value : v) {
            const simde__m256i cm = simde_mm256_cmpeq_epi64(simde_mm256_and_si256(value, type_mask), ftype);
            if constexpr (!kDelta) {
                // Timestamps as zero-extended qwords against the record before (`carry`: the previous load's
                // last); only F lanes count.
                const simde__m256i ts = simde_mm256_srli_epi64(value, 32);
                const simde__m256i before =
                    simde_mm256_blend_epi32(simde_mm256_permute4x64_epi64(ts, 0x90), carry, 0x03);
                back = simde_mm256_or_si256(back, simde_mm256_and_si256(simde_mm256_cmpgt_epi64(before, ts), cm));
                carry = simde_mm256_permute4x64_epi64(ts, 0xFF);
            }
            const uint32_t k = spsc_quad_count(cm);
            n += k;
            if (k < 4u) {
                break;
            }
        }
        n = std::min(n, max_recs);
        if (n == 0) {
            break;
        }
        // Inclusive prefix of the end deltas (each qword's top 16 bits) in record order: the four quads' in-lane
        // prefixes are independent and their totals carry as a tree, so no dependency chain spans the block.
        simde__m256i pfx[4];
        if constexpr (kDelta) {
            for (int i = 0; i < 4; i++) {
                simde__m256i d = simde_mm256_srli_epi64(v[i], 48);
                d = simde_mm256_add_epi64(d, simde_mm256_slli_si256(d, 8));
                pfx[i] =
                    simde_mm256_add_epi64(d, simde_mm256_blend_epi32(simde_mm256_permute4x64_epi64(d, 0x55), z, 0x0F));
            }
            const simde__m256i c1 = simde_mm256_permute4x64_epi64(pfx[0], 0xFF);
            const simde__m256i c2 = simde_mm256_add_epi64(c1, simde_mm256_permute4x64_epi64(pfx[1], 0xFF));
            const simde__m256i c3 = simde_mm256_add_epi64(c2, simde_mm256_permute4x64_epi64(pfx[2], 0xFF));
            pfx[1] = simde_mm256_add_epi64(pfx[1], c1);
            pfx[2] = simde_mm256_add_epi64(pfx[2], c2);
            pfx[3] = simde_mm256_add_epi64(pfx[3], c3);
        }
        for (uint32_t i = 0; i < 4 && 4 * i < n; i++) {
            if constexpr (kDelta) {
                const simde__m256i d64 = simde_mm256_and_si256(simde_mm256_srli_epi64(v[i], 32), dur_mask);
                const simde__m256i s64 = simde_mm256_sub_epi64(simde_mm256_add_epi64(cv, pfx[i]), d64);
                const simde__m256i m64 = simde_mm256_or_si256(simde_mm256_and_si256(v[i], id_mask), prog_hi_v);
                spsc_store_quad<kRec>(dst + 4 * kRec * i, s64, m64, tail);
                spsc_store_durations(dst + 4 * kRec * i, d64);
            } else {
                const simde__m256i ts64 = simde_mm256_or_si256(simde_mm256_srli_epi64(v[i], 32), c.th_hi_v);
                const simde__m256i m64 = simde_mm256_or_si256(simde_mm256_and_si256(v[i], id_mask), prog_hi_v);
                spsc_store_quad<kRec>(dst + 4 * kRec * i, ts64, m64, tail);
            }
        }
        dst += kRec * n;
        total += n;
        if constexpr (kDelta) {
            if (n < 16u) {
                // The prefix at record n-1 through a spill: a variable lane extract costs more than an aligned store
                // to hot stack.
                alignas(32) uint64_t arr[16];
                for (int i = 0; i < 4; i++) {
                    simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(arr + 4 * i), pfx[i]);
                }
                out.ts_last =
                    static_cast<uint64_t>(simde_mm_cvtsi128_si64(simde_mm256_castsi256_si128(cv))) + arr[n - 1];
                out.n = total;
                return out;  // the block ended on a non-F word or the emit budget
            }
            cv = simde_mm256_add_epi64(cv, simde_mm256_permute4x64_epi64(pfx[3], 0xFF));
        } else {
            // The carry for the next block is this block's last record, which the load loop may have run past.
            carry = simde_mm256_set1_epi64x(static_cast<long long>(p[2u * n - 1u]));
            out.ts_last = c.th_hi | p[2u * n - 1u];
            if (n < 16u) {
                break;
            }
        }
        p += 32;
        readable -= 32;
        max_recs -= 16;
    }
    out.n = total;
    if constexpr (kDelta) {
        if (total != 0) {
            out.ts_last = static_cast<uint64_t>(simde_mm_cvtsi128_si64(simde_mm256_castsi256_si128(cv)));
        }
    } else {
        out.regress = simde_mm256_testz_si256(back, back) ? 0u : 1u;
    }
    return out;
}

// The shortest delta16 run the block kernel takes: its setup costs more than the few records of a shorter one.
inline constexpr uint32_t kSpscDelta16BlockRun = 4;
// A shorter run of delta16 zones, one record at a time.
template <PacketFormat F>
inline SpscBlockResult spsc_delta16_short(
    const uint32_t* words, uint32_t max_recs, uint64_t cursor, const SpscLaneConsts& consts, uint8_t* dst) {
    static_assert(F.delta16 && F.words == 2);
    const simde__m128i tail = simde_mm256_castsi256_si128(consts.tail);
    const uint64_t prog = static_cast<uint64_t>(simde_mm_cvtsi128_si64(simde_mm256_castsi256_si128(consts.prog_hi_v)));
    uint32_t count = 0;
    for (; count < max_recs && count < kSpscDelta16BlockRun - 1 && pp_type(words[2 * count]) == F.type; count++) {
        const uint32_t deltas = words[2 * count + 1];
        cursor += deltas >> 16;
        const uint64_t duration = deltas & 0xFFFFu;
        const simde__m128i head = simde_mm_set_epi64x(
            static_cast<long long>((words[2 * count] & PP_LOW27_MASK) | prog),
            static_cast<long long>(cursor - duration));
        uint8_t* const record = dst + kSpscZoneBytes * count;
        simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(record), simde_mm256_set_m128i(tail, head));
        std::memcpy(record + kSpscSharedBytes, &duration, sizeof(duration));
    }
    return SpscBlockResult{.ts_last = cursor, .n = count};
}

// A run of F records through the layout its width selects. `cursor` is read by delta16 formats only.
template <PacketFormat F>
inline SpscBlockResult spsc_block(
    const uint32_t* p, uint32_t readable, uint32_t max_recs, uint64_t cursor, const SpscLaneConsts& c, uint8_t* dst) {
    if constexpr (F.words == 2) {
        return spsc_block_qword<F>(p, readable, max_recs, cursor, c, dst);
    } else {
        return spsc_block_strided<F>(p, readable, max_recs, c, dst);
    }
}

// One point packet, a Point kind or the Data kind: the shared 32 bytes at `dst`, then for Data the value count and
// `values`, where the payload words go as values, word 2k << 32 | word 2k+1 with the last zero-padded. `n` is the
// payload word count, 0 for a Point. Everything Data-only is masked by `n`, never selected by a branch, so random
// alternation costs no mispredicts; the count, the pointer and the first four payload words are stored unconditionally,
// a Point's landing in slack the next record overwrites, and only a payload beyond them takes the loop. Words past
// `readable` read as zero. Returns the values written.
inline uint32_t spsc_point(
    const uint32_t* p, uint32_t readable, uint32_t n, const SpscLaneConsts& c, uint8_t* dst, uint64_t* values) {
    constexpr int kTs = kSpscDataFormat.ts_lo;
    constexpr int kPayload = kSpscDataFormat.words;
    const uint32_t elems = (n + 1u) >> 1;
    const simde__m256i lane_idx = spsc_lane_idx();
    // The head and payload words 0-4 of the packet, the payload lanes zeroed past the count and past readable.
    const uint32_t words = std::min(readable, static_cast<uint32_t>(kPayload) + n);
    const simde__m256i l = simde_mm256_maskload_epi32(
        reinterpret_cast<const int*>(p),
        simde_mm256_cmpgt_epi32(simde_mm256_set1_epi32(static_cast<int>(words)), lane_idx));
    const simde__m256i head = simde_mm256_blend_epi32(
        simde_mm256_permutevar8x32_epi32(
            simde_mm256_and_si256(
                l, simde_mm256_setr_epi32(static_cast<int>(PP_LOW27_MASK), -1, -1, -1, -1, -1, -1, -1)),
            simde_mm256_setr_epi32(kTs, 0, 0, 0, 0, 0, 0, 0)),
        c.shared,
        0xFA);
    simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(dst), head);
    simde_mm_storeu_si128(
        reinterpret_cast<simde__m128i*>(dst + kSpscSharedBytes),
        simde_mm_set_epi64x(reinterpret_cast<long long>(values), static_cast<long long>(elems)));
    // Elements 0-1 from the packet's first four payload words, high word first.
    simde_mm_storeu_si128(
        reinterpret_cast<simde__m128i*>(values),
        simde_mm256_castsi256_si128(simde_mm256_permutevar8x32_epi32(
            l, simde_mm256_setr_epi32(kPayload + 1, kPayload, kPayload + 3, kPayload + 2, 0, 0, 0, 0))));
    if (__builtin_expect(n > 4u, 0)) {
        const uint32_t pw = readable > static_cast<uint32_t>(kPayload) ? std::min(readable - kPayload, n) : 0u;
        for (uint32_t k = 4; k < n; k += 8) {
            const uint32_t left = std::min(n - k, pw > k ? pw - k : 0u);
            const simde__m256i pl = simde_mm256_maskload_epi32(
                reinterpret_cast<const int*>(p + kPayload + k),
                simde_mm256_cmpgt_epi32(simde_mm256_set1_epi32(static_cast<int>(left)), lane_idx));
            simde_mm256_storeu_si256(
                reinterpret_cast<simde__m256i*>(values + k / 2),
                simde_mm256_permutevar8x32_epi32(pl, simde_mm256_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6)));
        }
    }
    return elems;
}

}  // namespace tt::tt_metal::profiler
