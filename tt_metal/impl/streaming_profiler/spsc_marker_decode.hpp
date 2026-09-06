// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The vector kernels of the host-side decode and the per-lane state they run against; the frame walk that drives
// them is StreamDecoder::decode_frame (streaming_profiler_decode.hpp). A frame is a 16-word prefix (word 1 = payload
// length), the SPSC_SPAN_WIRE_CTRL_WORDS control block, then each RISC's live ring window packed flat with congruence
// pads and wraps resolved device-side. Packet formats: spsc_packet.h. The producer publishes its tail only on
// packet boundaries, so a window never ends mid-packet.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <cstdint>
#include <span>
#include <vector>

#include <immintrin.h>

#if !defined(__AVX2__)
#error "the streaming profiler decode is AVX2 code; build for x86-64-v3 or newer"
#endif

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

// Packed NoC (y<<16)|x -> dense core index, direct-indexed so a frame's lookup is one load. 64x64 covers every
// supported grid; a coordinate outside it is unknown.
struct CoreTable {
    static constexpr uint16_t kNone = 0xFFFF;
    std::vector<uint16_t> slot = std::vector<uint16_t>(4096, kNone);
    static uint32_t idx(uint32_t xy) { return (((xy >> 16) & 63u) << 6) | (xy & 63u); }
    uint16_t& operator[](uint32_t xy) { return slot[idx(xy)]; }
    uint32_t find(uint32_t xy) const { return (xy & 0xFFC0FFC0u) != 0 ? kNone : slot[idx(xy)]; }
    void load(std::span<const uint32_t> core_xy) {
        slot.assign(4096, kNone);
        for (uint32_t core = 0; core < core_xy.size(); core++) {
            (*this)[core_xy[core]] = static_cast<uint16_t>(core);
        }
    }
};

// One (core, RISC) lane of a socket's frame stream.
struct SpscLane {
    // The end of the last ZONE_S/ZONE_ATOMIC zone, the base a ZONE_S's 16-bit end delta counts from. The producer
    // guarantees the first zone after a launch or rewind is an absolute ZONE_ATOMIC; a resync recovers at the next
    // one.
    uint64_t cursor = 0;
    uint64_t last_ts = 0;  // the last record's timestamp; lanes emit in end order, so a step back is a torn read
    // (batch_seq << 32) | byte offset just past the last timestamped record in the sink (a Data head, not its
    // Ext/Cont); 0 = none. A regression repairs that record while it is still in the scratch, so the offset is only
    // meaningful within its own batch.
    uint64_t last_rec = 0;
    uint32_t timer_hi = 0;  // sticky wall-clock high half
    uint32_t prog = 0;      // sticky runtime host-id (every RISC emits its own at launch)
    uint32_t head = 0;      // monotonic words-consumed mirror; head(N) == tail(N-1)
    uint32_t seeded = 0;
};

// The lanes of one socket's frame stream. Written only by the thread decoding that stream.
struct SpanDecodeState {
    std::vector<SpscLane> lanes;
    CoreTable core_of_xy;

    void reset(uint32_t num_cores) { lanes.assign(static_cast<size_t>(num_cores) * kSpscNRiscDecode, {}); }
};

// Every record a consumer sees is the public 32 B Rec {start|ts, duration, meta<<32 | id, prog}, composed straight
// into the sink's buffer. Stores are cached, not NT: the consumer re-reads the scratch immediately. ZONE_S and EVENT
// blocks write their last quad whole and a point packet writes its Ext and first Cont before knowing they count, so
// the buffer needs kSpscSinkSlackRecs of slack past cap.
inline constexpr uint32_t kSpscRecBytes = 32;
inline constexpr uint32_t kSpscSinkSlackRecs = 8;
// RecType codes, pinned by the receiver's layout probe.
inline constexpr uint32_t kSpscRecTypeZone = 1;
inline constexpr uint32_t kSpscRecTypeData = 2;
inline constexpr uint32_t kSpscRecTypeEvent = 3;
inline constexpr uint32_t kSpscRecTypeExt = 4;
inline constexpr uint32_t kSpscRecTypeCont = 5;
struct SpscRecSink {
    uint8_t* buf = nullptr;
    uint64_t off = 0;  // bytes written
};

// What a block kernel reports: 16 bytes so it returns in registers; the caller knows the first record's timestamp
// from the words. The flags say only that something needs the slow path; the caller finds where.
struct SpscBlockResult {
    uint64_t ts_last;  // ZONE_S: the lane cursor after the block
    uint32_t n;
    uint8_t regress;   // some record's timestamp precedes the one before it
    uint8_t wrapped;   // ZONE_L only: some duration's high word is all ones (the start read borrowed the next epoch)
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

// Runs handle.template operator()<F>() for the row of kFormats whose wire type is t and whose kind Pred accepts;
// false when there is none. The chain unrolls in table order. always_inline, like the handlers it takes: a handler
// left out of line captures the walk's locals by address and pushes them all out of registers.
template <bool (*Pred)(Kind), size_t I = 0, typename H>
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

// Everything a lane's records share: the constant half of each record kind, {0, th, 0, 0, 0, meta|type, prog, 0}
// (the 64-bit-timestamp kinds without th), and the qword broadcasts the 2-word kernels compose from. Built at lane
// entry; a STICKY_TIMER re-blends only the th lanes, a STICKY_PROG only the prog lanes.
struct SpscLaneConsts {
    uint64_t th_hi;
    __m256i tv, pv;             // th << 32 / prog, per qword
    __m256i mv_zone, mv_event;  // meta | type << 29, high dword of every qword
    __m256i zone_half, zone_half64, point_half64, ext_half, cont_half;
    __m256i point_half[2];  // [0] EVENT, [1] DATA
};
inline void spsc_lane_consts_th(SpscLaneConsts& c, uint32_t th) {
    c.th_hi = static_cast<uint64_t>(th) << 32;
    c.tv = _mm256_set1_epi64x(static_cast<long long>(c.th_hi));
    const __m256i thv = _mm256_set1_epi32(static_cast<int>(th));
    c.zone_half = _mm256_blend_epi32(c.zone_half, thv, 0x02);
    c.point_half[0] = _mm256_blend_epi32(c.point_half[0], thv, 0x02);
    c.point_half[1] = _mm256_blend_epi32(c.point_half[1], thv, 0x02);
}
inline void spsc_lane_consts_prog(SpscLaneConsts& c, uint32_t prog) {
    c.pv = _mm256_set1_epi64x(prog);
    const __m256i pgv = _mm256_set1_epi32(static_cast<int>(prog));
    for (__m256i* h :
         {&c.zone_half,
          &c.zone_half64,
          &c.point_half64,
          &c.ext_half,
          &c.cont_half,
          &c.point_half[0],
          &c.point_half[1]}) {
        *h = _mm256_blend_epi32(*h, pgv, 0x40);
    }
}
inline void spsc_lane_consts(SpscLaneConsts& c, uint32_t lane, uint32_t dev, uint32_t th, uint32_t prog) {
    const uint32_t meta = (lane << 16) | (dev << 26);
    const auto half = [meta](uint32_t type) {
        return _mm256_setr_epi32(0, 0, 0, 0, 0, static_cast<int>(meta | (type << 29)), 0, 0);
    };
    c.mv_zone =
        _mm256_set1_epi64x(static_cast<long long>(static_cast<uint64_t>(meta | (kSpscRecTypeZone << 29)) << 32));
    c.mv_event =
        _mm256_set1_epi64x(static_cast<long long>(static_cast<uint64_t>(meta | (kSpscRecTypeEvent << 29)) << 32));
    c.zone_half = half(kSpscRecTypeZone);
    c.zone_half64 = c.zone_half;
    c.point_half[0] = half(kSpscRecTypeEvent);
    c.point_half64 = c.point_half[0];
    c.point_half[1] = half(kSpscRecTypeData);
    c.ext_half = half(kSpscRecTypeExt);
    c.cont_half = half(kSpscRecTypeCont);
    spsc_lane_consts_th(c, th);
    spsc_lane_consts_prog(c, prog);
}
// The constant half of an F record: its th lane is blended in only when the timestamp's high half comes from the
// lane's sticky.
template <PacketFormat F>
inline __m256i spsc_half(const SpscLaneConsts& c) {
    if constexpr (F.kind == Kind::Zone) {
        return F.has_ts_hi() ? c.zone_half64 : c.zone_half;
    } else {
        return F.has_ts_hi() ? c.point_half64 : c.point_half[0];
    }
}

// The vector constants the kernels share; all fold to immediates once inlined.
inline __m256i spsc_dw_type_mask() {
    return _mm256_set1_epi32(static_cast<int>((0xFFFFFFFFu >> PP_TYPE_SHIFT) << PP_TYPE_SHIFT));
}
inline __m256i spsc_dw_type(uint32_t type) { return _mm256_set1_epi32(static_cast<int>(type << PP_TYPE_SHIFT)); }
inline __m256i spsc_qw_type_mask() {
    return _mm256_set1_epi64x(static_cast<long long>((0xFFFFFFFFull >> PP_TYPE_SHIFT) << PP_TYPE_SHIFT));
}
inline __m256i spsc_qw_type(uint32_t type) {
    return _mm256_set1_epi64x(static_cast<long long>(static_cast<uint64_t>(type) << PP_TYPE_SHIFT));
}
inline __m256i spsc_lane_idx() { return _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7); }
inline __m256i spsc_lane0() { return _mm256_setr_epi32(-1, 0, 0, 0, 0, 0, 0, 0); }

// A 32 B load holds spsc_recs_per_load<F> whole F packets at word stride F.words; these are its lane constants.
template <PacketFormat F>
inline constexpr uint32_t spsc_recs_per_load = 8 / F.words;
// -1 at word `w` of each whole packet in the load.
template <PacketFormat F, uint8_t w>
inline __m256i spsc_lanes_at() {
    constexpr auto L = [](int k) { return (k % F.words == w && k / F.words < 8 / F.words) ? -1 : 0; };
    return _mm256_setr_epi32(L(0), L(1), L(2), L(3), L(4), L(5), L(6), L(7));
}
// Word 0 of each whole packet down to its 27-bit id, every other lane kept.
template <PacketFormat F>
inline __m256i spsc_w0_mask() {
    constexpr auto L = [](int k) { return (k % F.words == 0 && k / F.words < 8 / F.words) ? 0x07FFFFFF : -1; };
    return _mm256_setr_epi32(L(0), L(1), L(2), L(3), L(4), L(5), L(6), L(7));
}
// The lanes of a composed record that come from the constant half rather than the packet: the meta, prog and pad
// lanes always, plus each field the format lacks.
template <PacketFormat F>
constexpr int spsc_blend_s() {
    return 0xE0 | (F.has_ts_hi() ? 0 : 0x02) | (F.has_dur() ? 0 : 0x04) | (F.has_dur_hi() ? 0 : 0x08);
}
template <PacketFormat F>
constexpr int spsc_blend_d() {
    return 0xFC | (F.has_dur_hi() ? 0 : 0x02);
}

// Composes packet `r` of a masked load into the public record at o and returns the composed vector: a lane permute
// puts {ts_lo, ts_hi, dur_lo, dur_hi, id} in place, a blend supplies the constant half, and subtracting
// {dur, 0, ...} leaves {start, dur, meta|id, prog} with the borrow in the high half.
template <PacketFormat F, uint32_t r>
inline __m256i spsc_compose(__m256i l, __m256i half, uint8_t* o) {
    constexpr int b = static_cast<int>(r * F.words);
    constexpr auto at = [](uint8_t f) { return f == kAbsent ? 0 : b + f; };
    const __m256i s = _mm256_blend_epi32(
        _mm256_permutevar8x32_epi32(
            l, _mm256_setr_epi32(at(F.ts_lo), at(F.ts_hi), at(F.dur_lo), at(F.dur_hi), b, 0, 0, 0)),
        half,
        spsc_blend_s<F>());
    if constexpr (F.has_dur()) {
        const __m256i d = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(l, _mm256_setr_epi32(at(F.dur_lo), at(F.dur_hi), 0, 0, 0, 0, 0, 0)),
            _mm256_setzero_si256(),
            spsc_blend_d<F>());
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(o), _mm256_sub_epi64(s, d));
    } else {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(o), s);
    }
    return s;
}

// Eight words from p with the lanes past `readable` zero; a count of zero or less loads nothing. A record's words
// are always in range, the load's tail may not be.
inline __m256i spsc_words(const uint32_t* p, int32_t readable) {
    if (readable >= 8) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    }
    return _mm256_maskload_epi32(
        reinterpret_cast<const int*>(p), _mm256_cmpgt_epi32(_mm256_set1_epi32(readable), spsc_lane_idx()));
}
// Four qword records from p, whole records only: a trailing odd word is never one, so its lane reads as zero.
inline __m256i spsc_qwords(const uint32_t* p, int32_t readable) {
    if (readable >= 8) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    }
    return _mm256_maskload_epi64(
        reinterpret_cast<const long long*>(p),
        _mm256_cmpgt_epi64(_mm256_set1_epi64x(readable / 2), _mm256_setr_epi64x(0, 1, 2, 3)));
}
// Records of the type in a quad's compare result, counted from lane 0 to the first miss. A full quad is one test;
// only a partial one extracts its mask.
inline uint32_t spsc_quad_count(__m256i c) {
    if (_mm256_testc_si256(c, _mm256_cmpeq_epi64(c, c))) {
        return 4;
    }
    return std::countr_zero(~static_cast<uint32_t>(_mm256_movemask_pd(_mm256_castsi256_pd(c))) & 0x1Fu);
}
// Four records from per-qword-lane halves: {a, b} is a record's first 16 bytes (start|ts, duration) and {m, pv}
// its second (meta|id, prog). Written whole; a partial quad's spare records land in the sink's slack.
inline void spsc_store_quad(uint8_t* o, __m256i a, __m256i b, __m256i m, __m256i pv) {
    const __m256i ab_lo = _mm256_unpacklo_epi64(a, b);
    const __m256i ab_hi = _mm256_unpackhi_epi64(a, b);
    const __m256i mp_lo = _mm256_unpacklo_epi64(m, pv);
    const __m256i mp_hi = _mm256_unpackhi_epi64(m, pv);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(o), _mm256_permute2x128_si256(ab_lo, mp_lo, 0x20));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(o + 32), _mm256_permute2x128_si256(ab_hi, mp_hi, 0x20));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(o + 64), _mm256_permute2x128_si256(ab_lo, mp_lo, 0x31));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(o + 96), _mm256_permute2x128_si256(ab_hi, mp_hi, 0x31));
}

// One F record from its words, reported like a run of one.
template <PacketFormat F>
inline SpscBlockResult spsc_one(const uint32_t* p, uint32_t readable, const SpscLaneConsts& c, SpscRecSink& sw) {
    spsc_compose<F, 0>(
        _mm256_and_si256(spsc_words(p, static_cast<int32_t>(readable)), spsc_w0_mask<F>()),
        spsc_half<F>(c),
        sw.buf + sw.off);
    sw.off += kSpscRecBytes;
    SpscBlockResult out{spsc_ts_at<F>(p, 0, c.th_hi), 1, 0, 0, 0};
    if constexpr (F.kind == Kind::Zone) {
        out.stalls = (p[0] & 0x07FFFFFFu) == kSpscStallZoneId ? 1u : 0u;
    }
    if constexpr (F.has_dur_hi()) {
        out.wrapped = p[F.dur_hi] == 0xFFFFFFFFu ? 1u : 0u;
    }
    return out;
}

// Four F records start at p (readable >= 8 words): the gate for the 2-word block kernel, so random single points
// never pay a block call and a run never pays the one-record path.
template <PacketFormat F>
inline bool spsc_run4(const uint32_t* p) {
    static_assert(F.words == 2);
    const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    const __m256i c = _mm256_cmpeq_epi64(_mm256_and_si256(v, spsc_qw_type_mask()), spsc_qw_type(F.type));
    return _mm256_testc_si256(c, _mm256_cmpeq_epi64(c, c));
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
    const uint32_t* p, uint32_t avail, uint32_t max_recs, const SpscLaneConsts& c, SpscRecSink& sw) {
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
    const __m256i type_mask = spsc_dw_type_mask();
    const __m256i ftype = spsc_dw_type(F.type);
    const __m256i lanes_w0 = spsc_lanes_at<F, 0>();
    const __m256i lane_0 = spsc_lane0();
    const __m256i lane_3 = _mm256_setr_epi32(0, 0, 0, -1, 0, 0, 0, 0);
    const __m256i w0_mask = spsc_w0_mask<F>();
    const __m256i stall = _mm256_set1_epi32(static_cast<int>(kSpscStallZoneId));
    const __m256i ones = _mm256_set1_epi32(-1);
    const __m256i half = spsc_half<F>(c);
    const __m256i z = _mm256_setzero_si256();
    const uint64_t th_hi = c.th_hi;
    const uint32_t* const p0 = p;
    __m256i prev = z, back = z, stall_hit = z, wrap_hit = z;
    uint32_t total = 0;
    while (max_recs != 0 && avail >= W) {
        // One load at a time, stopping at the first whose records are not all F: a lone record costs one load.
        __m256i v[kLoads];
        uint32_t n = 0;
        for (uint32_t i = 0; i < kLoads; i++) {
            if (avail >= kFullAvail) {
                v[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i * kStride));
            } else {
                const __m256i mk = _mm256_cmpgt_epi32(
                    _mm256_set1_epi32(static_cast<int>(avail) - static_cast<int>(i * kStride)), spsc_lane_idx());
                v[i] = _mm256_maskload_epi32(reinterpret_cast<const int*>(p + i * kStride), mk);
            }
            const __m256i cm = _mm256_cmpeq_epi32(_mm256_and_si256(v[i], type_mask), ftype);
            if (!_mm256_testc_si256(cm, lanes_w0)) {
                if constexpr (R == 2) {
                    n += _mm256_testc_si256(cm, lane_0) ? 1u : 0u;
                }
                break;
            }
            n += R;
        }
        if (n > max_recs) {
            n = max_recs;
        }
        if (n == 0) {
            break;
        }
        // Lane 0 of a composed record is its 64-bit end, so a signed 64-bit compare orders records, `prev` carrying
        // across loads and blocks. Lane 3 is the duration's high word when the format has one.
        uint32_t i = 0;
        for (; (i + 1) * R <= n; i++) {
            const __m256i l = _mm256_and_si256(v[i], w0_mask);
            uint8_t* const o = sw.buf + sw.off + kSpscRecBytes * R * i;
            const __m256i s0 = spsc_compose<F, 0>(l, half, o);
            if constexpr (R == 2) {
                const __m256i s1 = spsc_compose<F, 1>(l, half, o + kSpscRecBytes);
                back = _mm256_or_si256(back, _mm256_or_si256(_mm256_cmpgt_epi64(prev, s0), _mm256_cmpgt_epi64(s0, s1)));
                prev = s1;
                if constexpr (F.has_dur_hi()) {
                    wrap_hit = _mm256_or_si256(
                        wrap_hit, _mm256_or_si256(_mm256_cmpeq_epi32(s0, ones), _mm256_cmpeq_epi32(s1, ones)));
                }
            } else {
                back = _mm256_or_si256(back, _mm256_cmpgt_epi64(prev, s0));
                prev = s0;
                if constexpr (F.has_dur_hi()) {
                    wrap_hit = _mm256_or_si256(wrap_hit, _mm256_cmpeq_epi32(s0, ones));
                }
            }
            stall_hit = _mm256_or_si256(stall_hit, _mm256_cmpeq_epi32(l, stall));
        }
        if constexpr (R == 2) {
            if (2 * i < n) {  // an odd last record: the load's second record is not ours
                const __m256i l = _mm256_and_si256(v[i], w0_mask);
                const __m256i s0 = spsc_compose<F, 0>(l, half, sw.buf + sw.off + kSpscRecBytes * 2 * i);
                back = _mm256_or_si256(back, _mm256_cmpgt_epi64(prev, s0));
                prev = s0;
                if constexpr (F.has_dur_hi()) {
                    wrap_hit = _mm256_or_si256(wrap_hit, _mm256_cmpeq_epi32(s0, ones));
                }
                stall_hit = _mm256_or_si256(stall_hit, _mm256_and_si256(_mm256_cmpeq_epi32(l, stall), lane_0));
            }
        }
        total += n;
        sw.off += kSpscRecBytes * n;
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
    out.regress = _mm256_testz_si256(back, lane_0) ? 0u : 1u;
    if constexpr (F.has_dur_hi()) {
        out.wrapped = _mm256_testz_si256(wrap_hit, lane_3) ? 0u : 1u;
    }
    if (__builtin_expect(!_mm256_testz_si256(stall_hit, lanes_w0), 0)) {
        for (uint32_t k = 0; k < total; k++) {
            out.stalls += (p0[W * k] & 0x07FFFFFFu) == kSpscStallZoneId ? 1u : 0u;
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
    const uint32_t* p,
    uint32_t readable,
    uint32_t max_recs,
    uint64_t cursor,
    const SpscLaneConsts& c,
    SpscRecSink& sw) {
    static_assert(F.words == 2);
    constexpr bool kDelta = F.delta16;
    static_assert(kDelta ? F.kind == Kind::Zone : (F.kind == Kind::Point && F.ts_lo == 1));
    SpscBlockResult out{0, 0, 0, 0, 0};
    const __m256i type_mask = spsc_qw_type_mask();
    const __m256i ftype = spsc_qw_type(F.type);
    const __m256i z = _mm256_setzero_si256();
    const __m256i id_mask = _mm256_set1_epi64x(0x07FFFFFF);
    const __m256i dur_mask = _mm256_set1_epi64x(0xFFFF);
    // The cursor rides as a broadcast vector: the next block's starts need it as one, and the block total is already
    // a broadcast lane of the carry tree.
    __m256i cv = _mm256_set1_epi64x(static_cast<long long>(cursor));
    __m256i carry = z, back = z;
    uint32_t total = 0;
    while (max_recs != 0 && readable >= 2u) {
        __m256i v[4];
        if (readable >= 32u) {
            for (int i = 0; i < 4; i++) {
                v[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 8 * i));
            }
        } else {
            // Whole records only: a trailing odd word is never F, so its lane reads as zero and ends the scan.
            const int recs = static_cast<int>(readable / 2u);
            for (int i = 0; i < 4; i++) {
                const __m256i m = _mm256_cmpgt_epi64(_mm256_set1_epi64x(recs - 4 * i), _mm256_setr_epi64x(0, 1, 2, 3));
                v[i] = _mm256_maskload_epi64(reinterpret_cast<const long long*>(p + 8 * i), m);
            }
        }
        if constexpr (kDelta) {
            // Both lines of the block four blocks ahead: the hardware prefetcher does not run far enough into
            // DMA-landed memory (-6%); farther ahead evicts before use once the sink's output stream shares L1, and
            // one line per block leaves every other line to the hardware.
            _mm_prefetch(reinterpret_cast<const char*>(p + 128), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(p + 144), _MM_HINT_T0);
        }
        uint32_t n = 0;
        for (int i = 0; i < 4; i++) {
            const __m256i cm = _mm256_cmpeq_epi64(_mm256_and_si256(v[i], type_mask), ftype);
            if constexpr (!kDelta) {
                // Timestamps as zero-extended qwords against the record before (`carry`: the previous load's
                // last); only F lanes count.
                const __m256i ts = _mm256_srli_epi64(v[i], 32);
                const __m256i before = _mm256_blend_epi32(_mm256_permute4x64_epi64(ts, 0x90), carry, 0x03);
                back = _mm256_or_si256(back, _mm256_and_si256(_mm256_cmpgt_epi64(before, ts), cm));
                carry = _mm256_permute4x64_epi64(ts, 0xFF);
            }
            const uint32_t k = spsc_quad_count(cm);
            n += k;
            if (k < 4u) {
                break;
            }
        }
        if (n > max_recs) {
            n = max_recs;
        }
        if (n == 0) {
            break;
        }
        // Inclusive prefix of the end deltas (each qword's top 16 bits) in record order: the four quads' in-lane
        // prefixes are independent and their totals carry as a tree, so no dependency chain spans the block.
        __m256i pfx[4];
        if constexpr (kDelta) {
            for (int i = 0; i < 4; i++) {
                __m256i d = _mm256_srli_epi64(v[i], 48);
                d = _mm256_add_epi64(d, _mm256_slli_si256(d, 8));
                pfx[i] = _mm256_add_epi64(d, _mm256_blend_epi32(_mm256_permute4x64_epi64(d, 0x55), z, 0x0F));
            }
            const __m256i c1 = _mm256_permute4x64_epi64(pfx[0], 0xFF);
            const __m256i c2 = _mm256_add_epi64(c1, _mm256_permute4x64_epi64(pfx[1], 0xFF));
            const __m256i c3 = _mm256_add_epi64(c2, _mm256_permute4x64_epi64(pfx[2], 0xFF));
            pfx[1] = _mm256_add_epi64(pfx[1], c1);
            pfx[2] = _mm256_add_epi64(pfx[2], c2);
            pfx[3] = _mm256_add_epi64(pfx[3], c3);
        }
        uint8_t* const dst = sw.buf + sw.off;
        for (uint32_t i = 0; i < 4 && 4 * i < n; i++) {
            if constexpr (kDelta) {
                const __m256i d64 = _mm256_and_si256(_mm256_srli_epi64(v[i], 32), dur_mask);
                const __m256i s64 = _mm256_sub_epi64(_mm256_add_epi64(cv, pfx[i]), d64);
                const __m256i m64 = _mm256_or_si256(_mm256_and_si256(v[i], id_mask), c.mv_zone);
                spsc_store_quad(dst + 128 * i, s64, d64, m64, c.pv);
            } else {
                const __m256i ts64 = _mm256_or_si256(_mm256_srli_epi64(v[i], 32), c.tv);
                const __m256i m64 = _mm256_or_si256(_mm256_and_si256(v[i], id_mask), c.mv_event);
                spsc_store_quad(dst + 128 * i, ts64, z, m64, c.pv);
            }
        }
        sw.off += kSpscRecBytes * n;
        total += n;
        if constexpr (kDelta) {
            if (n < 16u) {
                // The prefix at record n-1 through a spill: a variable lane extract costs more than an aligned store
                // to hot stack.
                alignas(32) uint64_t arr[16];
                for (int i = 0; i < 4; i++) {
                    _mm256_store_si256(reinterpret_cast<__m256i*>(arr + 4 * i), pfx[i]);
                }
                out.ts_last = static_cast<uint64_t>(_mm_cvtsi128_si64(_mm256_castsi256_si128(cv))) + arr[n - 1];
                out.n = total;
                return out;  // the block ended on a non-F word or the emit budget
            }
            cv = _mm256_add_epi64(cv, _mm256_permute4x64_epi64(pfx[3], 0xFF));
        } else {
            // The carry for the next block is this block's last record, which the load loop may have run past.
            carry = _mm256_set1_epi64x(static_cast<long long>(p[2u * n - 1u]));
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
            out.ts_last = static_cast<uint64_t>(_mm_cvtsi128_si64(_mm256_castsi256_si128(cv)));
        }
    } else {
        out.regress = _mm256_testz_si256(back, back) ? 0u : 1u;
    }
    return out;
}

// A run of F records through the layout its width selects. `cursor` is read by delta16 formats only.
template <PacketFormat F>
inline SpscBlockResult spsc_block(
    const uint32_t* p,
    uint32_t readable,
    uint32_t max_recs,
    uint64_t cursor,
    const SpscLaneConsts& c,
    SpscRecSink& sw) {
    if constexpr (F.words == 2) {
        return spsc_block_qword<F>(p, readable, max_recs, cursor, c, sw);
    } else {
        return spsc_block_strided<F>(p, readable, max_recs, c, sw);
    }
}

// One point packet, a Point kind or the Data kind, with no branch on which: the head {ts, 0, meta|id, prog}, then an
// Ext (payload words 0-1 and the count) and a Cont (words 2-3) written unconditionally and only counted for a Data
// packet, so a Point's two spare records land in the slack the next record overwrites. Payload words past the count
// read as zero. Only a payload beyond four words takes the loop. Returns the records the packet counts for; `n` is
// the payload word count (0 for a Point).
// `dm` is all ones for a Data packet and zero for a Point; everything Data-only is masked by it, never selected by
// a branch, so random alternation costs no mispredicts.
inline uint32_t spsc_point(
    const uint32_t* p, uint32_t readable, uint32_t dm, uint32_t n, const SpscLaneConsts& c, SpscRecSink& sw) {
    constexpr int kTs = kSpscDataFormat.ts_lo;
    constexpr int kPayload = kSpscDataFormat.words;
    static_assert(kPayload == 3, "the Ext/Cont permutes below index a 3-word head");
    const uint32_t conts = n > 2u ? (n - 1u) / 2u : 0u;
    const uint32_t recs = 1u + (dm & (1u + conts));
    const __m256i lane_idx = spsc_lane_idx();
    // The head and payload 0-4 of the packet, the payload lanes zeroed past the count and past readable. Always a
    // masked load: a branch on the packet's size would follow the type and mispredict on alternation.
    const uint32_t words = std::min(readable, static_cast<uint32_t>(kPayload) + n);
    const __m256i l = _mm256_maskload_epi32(
        reinterpret_cast<const int*>(p), _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(words)), lane_idx));
    const __m256i type_half = c.point_half[dm & 1u];
    const __m256i head = _mm256_blend_epi32(
        _mm256_permutevar8x32_epi32(
            _mm256_and_si256(l, _mm256_setr_epi32(0x07FFFFFF, -1, -1, -1, -1, -1, -1, -1)),
            _mm256_setr_epi32(kTs, 0, 0, 0, 0, 0, 0, 0)),
        type_half,
        0xEE);
    uint8_t* const dst = sw.buf + sw.off;
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), head);
    const __m256i ext = _mm256_blend_epi32(
        _mm256_permutevar8x32_epi32(l, _mm256_setr_epi32(kPayload + 1, kPayload, 0, 0, 0, 0, 0, 0)),
        _mm256_blend_epi32(c.ext_half, _mm256_set1_epi32(static_cast<int>(n)), 0x10),
        0xFC);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 32), ext);
    const __m256i cont = c.cont_half;
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(dst + 64),
        _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(l, _mm256_setr_epi32(kPayload + 3, kPayload + 2, 0, 0, 0, 0, 0, 0)),
            cont,
            0xFC));
    if (__builtin_expect(n > 4u, 0)) {
        uint8_t* o = dst + 96;
        const uint32_t pw = readable > 3u ? std::min(readable - 3u, n) : 0u;
        for (uint32_t k = 4; k < n; k += 8) {
            const uint32_t left = std::min(n - k, pw > k ? pw - k : 0u);
            const __m256i pl = _mm256_maskload_epi32(
                reinterpret_cast<const int*>(p + kPayload + k),
                _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(left)), lane_idx));
            for (uint32_t j = 0; j < 4 && k + 2 * j < n; j++) {
                const __m256i idx =
                    _mm256_setr_epi32(static_cast<int>(2 * j + 1), static_cast<int>(2 * j), 0, 0, 0, 0, 0, 0);
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(o),
                    _mm256_blend_epi32(_mm256_permutevar8x32_epi32(pl, idx), cont, 0xFC));
                o += 32;
            }
        }
    }
    sw.off += kSpscRecBytes * recs;
    return recs;
}

}  // namespace tt::tt_metal::profiler
