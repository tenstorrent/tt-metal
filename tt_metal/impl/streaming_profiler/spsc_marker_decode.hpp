// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Host-side decode of the relay's wire, and its only definition. A frame is a 16-word prefix (word 1 = payload
// length), the SPSC_SPAN_WIRE_CTRL_WORDS control block, then each RISC's live ring window packed flat with congruence
// pads and wraps resolved device-side. Packet formats: spsc_packet.h. The producer publishes its tail only on
// packet boundaries, so a window never ends mid-packet.
#pragma once

#include <algorithm>
#include <bit>
#include <cstring>
#include <cstdint>
#include <unordered_map>
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
    void load(const std::unordered_map<uint32_t, uint32_t>& core_of_xy) {
        slot.assign(4096, kNone);
        for (const auto& [xy, core] : core_of_xy) {
            (*this)[xy] = static_cast<uint16_t>(core);
        }
    }
};

// Decode state for one socket's frame stream. Written only by that socket's decode thread.
struct SpanDecodeState {
    std::vector<uint32_t> timer_hi;  // per lane: sticky wall-clock high half
    // Per lane: the end of the last ZONE_S/ZONE_ATOMIC zone, the base a ZONE_S's 16-bit end delta counts from.
    // The producer guarantees the first zone after a launch or rewind is an absolute ZONE_ATOMIC; a resync
    // recovers at the next one.
    std::vector<uint64_t> cursor;
    std::vector<uint32_t> prog;  // per lane: sticky runtime host-id (every RISC emits its own at launch)
    std::vector<uint32_t> head;  // per lane: monotonic words-consumed mirror; head(N) == tail(N-1)
    std::vector<uint8_t> seeded;
    CoreTable core_of_xy;
    uint64_t live_words = 0;
    uint64_t resync_words = 0;
    uint64_t anomalies = 0;  // torn run / truncated run / undecodable word
    uint64_t unknown_core_frames = 0;

    void reset(uint32_t num_cores) {
        timer_hi.assign(static_cast<size_t>(num_cores) * kSpscNRiscDecode, 0);
        cursor.assign(static_cast<size_t>(num_cores) * kSpscNRiscDecode, 0);
        prog.assign(static_cast<size_t>(num_cores) * kSpscNRiscDecode, 0);
        head.assign(static_cast<size_t>(num_cores) * kSpscNRiscDecode, 0);
        seeded.assign(static_cast<size_t>(num_cores) * kSpscNRiscDecode, 0);
    }
};

// Every record a consumer sees is the public 32 B Rec {start|ts, duration, meta<<32 | id, prog}; blocks compose
// them as whole 64 B lines straight into the Sink's buffer. Stores are cached, not NT: the consumer re-reads the
// scratch immediately. The audit sink stores nothing (Sink::kStores). A partial block still writes its full
// half (4 lines), so the buffer needs kSpscSinkSlackRecs of slack past cap.
inline constexpr uint32_t kSpscRecBytes = 32;
inline constexpr uint32_t kSpscSinkSlackRecs = 8;
// RecType codes, pinned by the receiver's layout probe.
inline constexpr uint32_t kSpscRecTypeZone = 1;
inline constexpr uint32_t kSpscRecTypeData = 2;
inline constexpr uint32_t kSpscRecTypeEvent = 3;
inline constexpr uint32_t kSpscRecTypeExt = 4;
inline constexpr uint32_t kSpscRecTypeCont = 5;
struct SpscRecSink {
    static constexpr bool kStores = true;
    uint8_t* buf = nullptr;
    uint64_t off = 0;  // bytes written

    inline void put4(uint64_t q0, uint64_t q1, uint64_t q2, uint64_t q3) {
        uint64_t* p = reinterpret_cast<uint64_t*>(buf + off);
        p[0] = q0;
        p[1] = q1;
        p[2] = q2;
        p[3] = q3;
        off += kSpscRecBytes;
    }
};
struct SpscNullRecSink {
    static constexpr bool kStores = false;
    inline void put4(uint64_t, uint64_t, uint64_t, uint64_t) {}
};

// 16 bytes so it returns in registers; the caller knows the first record's timestamp from the words.
struct SpscA16Result {
    uint64_t ts_last;
    uint32_t n;
    uint16_t regress;  // nonzero: some record's timestamp precedes the one before it
    uint16_t stalls;   // records whose id is kSpscStallZoneId; storing sinks only, the audit has no use for it
};

// A wall-clock read whose low word is this close below a wrap may carry the next epoch's high word (the
// device-side latch race, kernel_profiler_streaming.hpp read_wall_clock); the decoder repairs it when the lane's
// next timestamp regresses. The gap is a few cycles; 1024 keeps the false-positive odds at ~2e-7 per regression.
inline constexpr uint32_t kLatchWindow = 0xFFFFFC00u;

struct SpscZoneS16Result {
    uint64_t ts_last;  // the lane cursor after the block
    uint32_t n;
};

// Everything a lane's records share. The type halves and meta broadcasts are fixed for the lane run; the timer and
// prog parts change on a sticky and are four broadcasts, blended into a type half by the kernel that uses it.
struct SpscLaneConsts {
    uint64_t th_hi;
    uint32_t th, prog;
    __m256i thv, pgv;                                // th / prog in every dword
    __m256i tv, pv;                                  // th << 32 / prog in every qword
    __m256i zone_t, event_t, data_t, ext_t, cont_t;  // {0, 0, 0, 0, 0, meta | type << 29, 0, 0}
    __m256i mv_zone, mv_event;                       // meta | type << 29, in the high dword of every qword
    // The type halves with th and prog in place, refreshed with them: {0, th, 0, 0, 0, meta|type, prog, 0}, the
    // 64-bit-end kinds without th. point_half[0] is the EVENT half, [1] the DATA half.
    __m256i zone_half, zone_l_half, point_half[2], ext_half, cont_half;
};
inline void spsc_lane_consts_lane(SpscLaneConsts& c, uint32_t lane, uint32_t dev) {
    const uint32_t meta = (lane << 16) | (dev << 26);
    const auto type_half = [meta](uint32_t type) {
        return _mm256_setr_epi32(0, 0, 0, 0, 0, static_cast<int>(meta | (type << 29)), 0, 0);
    };
    c.zone_t = type_half(kSpscRecTypeZone);
    c.event_t = type_half(kSpscRecTypeEvent);
    c.data_t = type_half(kSpscRecTypeData);
    c.ext_t = type_half(kSpscRecTypeExt);
    c.cont_t = type_half(kSpscRecTypeCont);
    c.mv_zone =
        _mm256_set1_epi64x(static_cast<long long>(static_cast<uint64_t>(meta | (kSpscRecTypeZone << 29)) << 32));
    c.mv_event =
        _mm256_set1_epi64x(static_cast<long long>(static_cast<uint64_t>(meta | (kSpscRecTypeEvent << 29)) << 32));
}
// A STICKY_TIMER touches only the th parts, a STICKY_PROG only the prog parts; the halves are re-blended in place.
inline void spsc_lane_consts_th(SpscLaneConsts& c, uint32_t th) {
    c.th_hi = static_cast<uint64_t>(th) << 32;
    c.th = th;
    c.thv = _mm256_set1_epi32(static_cast<int>(th));
    c.tv = _mm256_set1_epi64x(static_cast<long long>(c.th_hi));
    c.zone_half = _mm256_blend_epi32(c.zone_half, c.thv, 0x02);
    c.point_half[0] = _mm256_blend_epi32(c.point_half[0], c.thv, 0x02);
    c.point_half[1] = _mm256_blend_epi32(c.point_half[1], c.thv, 0x02);
}
inline void spsc_lane_consts_prog(SpscLaneConsts& c, uint32_t prog) {
    c.prog = prog;
    c.pgv = _mm256_set1_epi32(static_cast<int>(prog));
    c.pv = _mm256_set1_epi64x(prog);
    c.zone_half = _mm256_blend_epi32(c.zone_half, c.pgv, 0x40);
    c.zone_l_half = _mm256_blend_epi32(c.zone_l_half, c.pgv, 0x40);
    c.point_half[0] = _mm256_blend_epi32(c.point_half[0], c.pgv, 0x40);
    c.point_half[1] = _mm256_blend_epi32(c.point_half[1], c.pgv, 0x40);
    c.ext_half = _mm256_blend_epi32(c.ext_half, c.pgv, 0x40);
    c.cont_half = _mm256_blend_epi32(c.cont_half, c.pgv, 0x40);
}
inline void spsc_lane_consts_sticky(SpscLaneConsts& c, uint32_t th, uint32_t prog) {
    c.zone_half = c.zone_t;
    c.zone_l_half = c.zone_t;
    c.point_half[0] = c.event_t;
    c.point_half[1] = c.data_t;
    c.ext_half = c.ext_t;
    c.cont_half = c.cont_t;
    spsc_lane_consts_th(c, th);
    spsc_lane_consts_prog(c, prog);
}
// A record's constant half: {0, th, 0, 0, 0, meta|type, prog, 0}, or without th for the 64-bit-end types.
inline __m256i spsc_half(const SpscLaneConsts& c, __m256i type_half) {
    return _mm256_blend_epi32(_mm256_blend_epi32(type_half, c.thv, 0x02), c.pgv, 0x40);
}
inline __m256i spsc_half_nt(const SpscLaneConsts& c, __m256i type_half) {
    return _mm256_blend_epi32(type_half, c.pgv, 0x40);
}

// One record of a type, straight from its words: a lane permute puts the fields in place and a blend supplies the
// constant half. `readable` bounds the load; a record's words are always in range, the load's tail may not be.
inline __m256i spsc_words(const uint32_t* p, uint32_t readable) {
    if (readable >= 8u) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    }
    return _mm256_maskload_epi32(
        reinterpret_cast<const int*>(p),
        _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(readable)), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)));
}
template <typename Sink>
inline void spsc_atomic1(const uint32_t* p, uint32_t readable, const SpscLaneConsts& c, Sink& sw) {
    if constexpr (Sink::kStores) {
        const __m256i l =
            _mm256_and_si256(spsc_words(p, readable), _mm256_setr_epi32(0x07FFFFFF, -1, -1, -1, -1, -1, -1, -1));
        const __m256i s = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(l, _mm256_setr_epi32(1, 0, 2, 0, 0, 0, 0, 0)), c.zone_half, 0xEA);
        const __m256i d = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(l, _mm256_setr_epi32(2, 0, 0, 0, 0, 0, 0, 0)), _mm256_setzero_si256(), 0xFE);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(sw.buf + sw.off), _mm256_sub_epi64(s, d));
        sw.off += kSpscRecBytes;
    } else {
        (void)p;
        (void)readable;
        (void)c;
    }
}
template <typename Sink>
inline void spsc_zone_l1(const uint32_t* p, uint32_t readable, const SpscLaneConsts& c, Sink& sw) {
    if constexpr (Sink::kStores) {
        const __m256i l =
            _mm256_and_si256(spsc_words(p, readable), _mm256_setr_epi32(0x07FFFFFF, -1, -1, -1, -1, -1, -1, -1));
        const __m256i s = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(l, _mm256_setr_epi32(1, 2, 3, 4, 0, 0, 0, 0)), c.zone_l_half, 0xE0);
        const __m256i d = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(l, _mm256_setr_epi32(3, 4, 0, 0, 0, 0, 0, 0)), _mm256_setzero_si256(), 0xFC);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(sw.buf + sw.off), _mm256_sub_epi64(s, d));
        sw.off += kSpscRecBytes;
    } else {
        (void)p;
        (void)readable;
        (void)c;
    }
}

// ZONE_ATOMIC records, eight per block through four 32 B loads at a 24 B stride, so each load holds two whole
// records with every field at a fixed dword. A record composes with two lane permutes off its load:
// {end, th, dur, 0, id, meta, prog, 0} minus {dur, 0, ...} is {start, dur, meta|id, prog}, the borrow landing in
// the high half. Consumes every consecutive record in one call, so a run pays the walk's per-call cost once.
// Compare results stay in vectors and are tested once; a mask is only extracted when a test fires, since each
// vector-to-scalar move costs more than the compose itself. `avail` authorizes loads, never emits.
template <typename Sink>
inline SpscA16Result spsc_atomic8(
    const uint32_t* p, uint32_t avail, uint32_t max_recs, const SpscLaneConsts& c, Sink& sw) {
    SpscA16Result out{0, 0, 0, 0};
    const __m256i type_mask = _mm256_set1_epi32(static_cast<int>((0xFFFFFFFFu >> PP_TYPE_SHIFT) << PP_TYPE_SHIFT));
    const __m256i atype = _mm256_set1_epi32(static_cast<int>(PP_ZONE_ATOMIC << PP_TYPE_SHIFT));
    const __m256i lanes_w0 = _mm256_setr_epi32(-1, 0, 0, -1, 0, 0, 0, 0);
    const __m256i lane_0 = _mm256_setr_epi32(-1, 0, 0, 0, 0, 0, 0, 0);
    const __m256i lane_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i w0_mask = _mm256_setr_epi32(0x07FFFFFF, -1, -1, 0x07FFFFFF, -1, -1, -1, -1);
    const __m256i stall = _mm256_set1_epi32(static_cast<int>(kSpscStallZoneId));
    const __m256i consts = c.zone_half;
    const __m256i z = _mm256_setzero_si256();
    const __m256i idx_a = _mm256_setr_epi32(1, 0, 2, 0, 0, 0, 0, 0);
    const __m256i idx_b = _mm256_setr_epi32(4, 0, 5, 0, 3, 0, 0, 0);
    const __m256i idx_da = _mm256_setr_epi32(2, 0, 0, 0, 0, 0, 0, 0);
    const __m256i idx_db = _mm256_setr_epi32(5, 0, 0, 0, 0, 0, 0, 0);
    const uint64_t th_hi = c.th_hi;
    const uint32_t* const p0 = p;
    __m256i prev = z, back = z, stall_hit = z;
    uint32_t total = 0;
    while (max_recs != 0 && avail >= 3u) {
        // One load at a time, stopping at the first whose two records are not both ATOMIC: a lone record costs
        // one load.
        // Two lines per 96 B block, four blocks ahead, as the ZONE_S kernel does: the hardware prefetcher does not run
        // far enough into DMA-landed memory.
        _mm_prefetch(reinterpret_cast<const char*>(p + 96), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(p + 112), _MM_HINT_T0);
        __m256i v[4];
        uint32_t n = 0;
        for (int i = 0; i < 4; i++) {
            if (avail >= 26u) {
                v[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 6 * i));
            } else {
                const __m256i mk = _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(avail) - 6 * i), lane_idx);
                v[i] = _mm256_maskload_epi32(reinterpret_cast<const int*>(p + 6 * i), mk);
            }
            const __m256i c = _mm256_cmpeq_epi32(_mm256_and_si256(v[i], type_mask), atype);
            if (!_mm256_testc_si256(c, lanes_w0)) {
                n += _mm256_testc_si256(c, lane_0) ? 1u : 0u;
                break;
            }
            n += 2;
        }
        if (n > max_recs) {
            n = max_recs;
        }
        if (n == 0) {
            break;
        }
        // Lane 0 of a composed record is th<<32 | end, so a signed 64-bit compare orders records, `prev` carrying
        // across loads and blocks. The last load's second record may be past n; its compare is masked out.
        uint32_t i = 0;
        for (; 2 * i + 2 <= n; i++) {
            const __m256i l = _mm256_and_si256(v[i], w0_mask);
            const __m256i sa = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(l, idx_a), consts, 0xEA);
            const __m256i sb = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(l, idx_b), consts, 0xEA);
            back = _mm256_or_si256(back, _mm256_or_si256(_mm256_cmpgt_epi64(prev, sa), _mm256_cmpgt_epi64(sa, sb)));
            prev = sb;
            if constexpr (Sink::kStores) {
                stall_hit = _mm256_or_si256(stall_hit, _mm256_cmpeq_epi32(l, stall));
                const __m256i da = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(l, idx_da), z, 0xFE);
                const __m256i db = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(l, idx_db), z, 0xFE);
                uint8_t* const o = sw.buf + sw.off + 64 * i;
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(o), _mm256_sub_epi64(sa, da));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(o + 32), _mm256_sub_epi64(sb, db));
            }
        }
        if (2 * i < n) {  // an odd last record: the load's second record is not ours
            const __m256i l = _mm256_and_si256(v[i], w0_mask);
            const __m256i sa = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(l, idx_a), consts, 0xEA);
            back = _mm256_or_si256(back, _mm256_cmpgt_epi64(prev, sa));
            prev = sa;
            if constexpr (Sink::kStores) {
                stall_hit = _mm256_or_si256(stall_hit, _mm256_and_si256(_mm256_cmpeq_epi32(l, stall), lane_0));
                const __m256i da = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(l, idx_da), z, 0xFE);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(sw.buf + sw.off + 64 * i), _mm256_sub_epi64(sa, da));
            }
        }
        out.ts_last = th_hi | p[3u * n - 2u];
        total += n;
        if constexpr (Sink::kStores) {
            sw.off += kSpscRecBytes * n;
        }
        if (n < 8u) {
            break;
        }
        p += 24;
        avail -= 24;
        max_recs -= 8;
    }
    out.n = total;
    out.regress = _mm256_testz_si256(back, lane_0) ? 0u : 1u;
    if constexpr (Sink::kStores) {
        if (__builtin_expect(!_mm256_testz_si256(stall_hit, lanes_w0), 0)) {
            for (uint32_t k = 0; k < total; k++) {
                out.stalls += (p0[3u * k] & 0x07FFFFFFu) == kSpscStallZoneId ? 1u : 0u;
            }
        }
    }
    return out;
}

// A ZONE_S block on the wire's own layout: a record is one qword (w0 low, w1 high), so four records are one 256-bit
// load and each field is a shift or mask of its own lane; nothing deinterleaves or widens. A ZONE_S end is
// cursor-relative, so the end deltas take an inclusive prefix sum, and records normalize to ZONE_ATOMIC form so
// downstream never sees wire size classes. Consumes every consecutive full block in one call, so a dense lane pays
// the walk's per-call cost once per run rather than once per 16 records. `readable` authorizes loads, never emits,
// past the live run; `max_recs` bounds the emits.
template <typename Sink>
inline SpscZoneS16Result spsc_zone_s16(
    const uint32_t* p, uint32_t readable, uint32_t max_recs, uint64_t cursor, const SpscLaneConsts& c, Sink& sw) {
    SpscZoneS16Result out{cursor, 0};
    const __m256i type_mask =
        _mm256_set1_epi64x(static_cast<long long>((0xFFFFFFFFull >> PP_TYPE_SHIFT) << PP_TYPE_SHIFT));
    const __m256i stype = _mm256_set1_epi64x(static_cast<long long>(static_cast<uint64_t>(PP_ZONE_S) << PP_TYPE_SHIFT));
    const __m256i lane_idx = _mm256_setr_epi64x(0, 1, 2, 3);
    const __m256i z = _mm256_setzero_si256();
    const __m256i ones = _mm256_cmpeq_epi64(z, z);
    [[maybe_unused]] const __m256i mv = c.mv_zone;
    [[maybe_unused]] const __m256i pv = c.pv;
    [[maybe_unused]] const __m256i id_mask = _mm256_set1_epi64x(0x07FFFFFF);
    [[maybe_unused]] const __m256i dur_mask = _mm256_set1_epi64x(0xFFFF);
    // The storing path carries the cursor as a broadcast vector: the next block's starts need it as one, and the
    // block total is already a broadcast lane of the carry tree.
    [[maybe_unused]] __m256i cv = _mm256_set1_epi64x(static_cast<long long>(cursor));
    // Inclusive prefix of the end deltas (each qword's top 16 bits) in record order: the four quads' in-lane
    // prefixes are independent and their totals carry as a tree, so no dependency chain spans the block.
    const auto prefix = [&](const __m256i* v, __m256i* pfx) {
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
    };
    // The prefix at record n-1: the top lane for a full block, else through a spill (a variable lane extract
    // costs more than an aligned store to hot stack).
    const auto prefix_at = [](const __m256i* pfx, uint32_t n) -> uint64_t {
        if (n == 16u) {
            return static_cast<uint64_t>(_mm_extract_epi64(_mm256_extracti128_si256(pfx[3], 1), 1));
        }
        alignas(32) uint64_t arr[16];
        for (int i = 0; i < 4; i++) {
            _mm256_store_si256(reinterpret_cast<__m256i*>(arr + 4 * i), pfx[i]);
        }
        return arr[n - 1];
    };
    while (max_recs != 0 && readable >= 2u) {
        __m256i v[4];
        if (readable >= 32u) {
            for (int i = 0; i < 4; i++) {
                v[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 8 * i));
            }
        } else {
            // Whole records only: a trailing odd word is never ZONE_S, so its lane reads as zero and ends the scan.
            const int recs = static_cast<int>(readable / 2u);
            for (int i = 0; i < 4; i++) {
                const __m256i m = _mm256_cmpgt_epi64(_mm256_set1_epi64x(recs - 4 * i), lane_idx);
                v[i] = _mm256_maskload_epi64(reinterpret_cast<const long long*>(p + 8 * i), m);
            }
        }
        // Both lines of the block four blocks ahead: the audit reads DMA-landed memory the hardware prefetcher does
        // not run far enough into (-37% streaming, -6% for a storing sink); farther ahead evicts before use once
        // the sink's output stream shares L1, and one line per block leaves every other line to the hardware.
        _mm_prefetch(reinterpret_cast<const char*>(p + 128), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(p + 144), _MM_HINT_T0);
        // A full quad of ZONE_S is one test; only a partial quad extracts its mask.
        uint32_t n = 0;
        for (int i = 0; i < 4; i++) {
            const __m256i c = _mm256_cmpeq_epi64(_mm256_and_si256(v[i], type_mask), stype);
            if (!_mm256_testc_si256(c, ones)) {
                n += static_cast<uint32_t>(
                    std::countr_zero(~static_cast<uint32_t>(_mm256_movemask_pd(_mm256_castsi256_pd(c))) & 0x1Fu));
                break;
            }
            n += 4;
        }
        if (n > max_recs) {
            n = max_recs;
        }
        if (n == 0) {
            break;
        }
        if constexpr (Sink::kStores) {
            __m256i pfx[4];
            prefix(v, pfx);
            uint8_t* const dst = sw.buf + sw.off;
            // Every quad the emit reaches is written whole (the sink's slack covers the partial one).
            for (uint32_t i = 0; i < 4 && 4 * i < n; i++) {
                const __m256i d64 = _mm256_and_si256(_mm256_srli_epi64(v[i], 32), dur_mask);
                const __m256i s64 = _mm256_sub_epi64(_mm256_add_epi64(cv, pfx[i]), d64);
                const __m256i m64 = _mm256_or_si256(_mm256_and_si256(v[i], id_mask), mv);
                const __m256i sd_lo = _mm256_unpacklo_epi64(s64, d64);
                const __m256i sd_hi = _mm256_unpackhi_epi64(s64, d64);
                const __m256i mp_lo = _mm256_unpacklo_epi64(m64, pv);
                const __m256i mp_hi = _mm256_unpackhi_epi64(m64, pv);
                uint8_t* const o = dst + 128 * i;
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(o), _mm256_permute2x128_si256(sd_lo, mp_lo, 0x20));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(o + 32), _mm256_permute2x128_si256(sd_hi, mp_hi, 0x20));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(o + 64), _mm256_permute2x128_si256(sd_lo, mp_lo, 0x31));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(o + 96), _mm256_permute2x128_si256(sd_hi, mp_hi, 0x31));
            }
            sw.off += kSpscRecBytes * n;
            if (n == 16u) {
                cv = _mm256_add_epi64(cv, _mm256_permute4x64_epi64(pfx[3], 0xFF));
            } else {
                cursor = static_cast<uint64_t>(_mm_cvtsi128_si64(_mm256_castsi256_si128(cv))) + prefix_at(pfx, n);
            }
        } else if (n == 16u) {
            // Only the block's total moves the cursor: a tree of adds, no prefix.
            __m256i s = _mm256_add_epi64(
                _mm256_add_epi64(_mm256_srli_epi64(v[0], 48), _mm256_srli_epi64(v[1], 48)),
                _mm256_add_epi64(_mm256_srli_epi64(v[2], 48), _mm256_srli_epi64(v[3], 48)));
            s = _mm256_add_epi64(s, _mm256_permute4x64_epi64(s, 0x4E));
            s = _mm256_add_epi64(s, _mm256_shuffle_epi32(s, 0x4E));
            cursor += static_cast<uint64_t>(_mm_cvtsi128_si64(_mm256_castsi256_si128(s)));
        } else {
            __m256i pfx[4];
            prefix(v, pfx);
            cursor += prefix_at(pfx, n);
        }
        out.n += n;
        if (n < 16u) {
            out.ts_last = cursor;
            return out;  // the block ended on a non-S word or the emit budget
        }
        if constexpr (!Sink::kStores) {
            out.ts_last = cursor;
        }
        p += 32;
        readable -= 32;
        max_recs -= 16;
    }
    if constexpr (Sink::kStores) {
        if (out.n != 0) {
            out.ts_last = static_cast<uint64_t>(_mm_cvtsi128_si64(_mm256_castsi256_si128(cv)));
        }
    }
    return out;
}

// EVENT records on the wire's own layout: a record is one qword (w0 low, timer_low high), four per load, and
// composes as {ts, 0, meta|id, prog} from two unpacks and a lane permute. Consumes every consecutive record in one
// call; compare results stay in vectors and are tested once. `stalls` is always 0.
template <typename Sink>
inline SpscA16Result spsc_event16(
    const uint32_t* p, uint32_t readable, uint32_t max_recs, const SpscLaneConsts& c, Sink& sw) {
    SpscA16Result out{0, 0, 0, 0};
    const __m256i type_mask =
        _mm256_set1_epi64x(static_cast<long long>((0xFFFFFFFFull >> PP_TYPE_SHIFT) << PP_TYPE_SHIFT));
    const __m256i etype = _mm256_set1_epi64x(static_cast<long long>(static_cast<uint64_t>(PP_EVENT) << PP_TYPE_SHIFT));
    const __m256i z = _mm256_setzero_si256();
    const __m256i ones = _mm256_cmpeq_epi64(z, z);
    const __m256i lane_idx = _mm256_setr_epi64x(0, 1, 2, 3);
    const uint64_t th_hi = c.th_hi;
    [[maybe_unused]] const __m256i tv = c.tv;
    [[maybe_unused]] const __m256i mv = c.mv_event;
    [[maybe_unused]] const __m256i pv = c.pv;
    [[maybe_unused]] const __m256i id_mask = _mm256_set1_epi64x(0x07FFFFFF);
    __m256i carry = z, back = z;
    uint32_t total = 0;
    while (max_recs != 0 && readable >= 2u) {
        // One load at a time, stopping at the first that is not all EVENT: a lone record costs one load.
        _mm_prefetch(reinterpret_cast<const char*>(p + 128), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(p + 144), _MM_HINT_T0);
        __m256i v[4];
        uint32_t n = 0;
        const int recs = static_cast<int>(readable / 2u);
        for (int i = 0; i < 4; i++) {
            if (readable >= 32u) {
                v[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 8 * i));
            } else {
                const __m256i mk = _mm256_cmpgt_epi64(_mm256_set1_epi64x(recs - 4 * i), lane_idx);
                v[i] = _mm256_maskload_epi64(reinterpret_cast<const long long*>(p + 8 * i), mk);
            }
            const __m256i c = _mm256_cmpeq_epi64(_mm256_and_si256(v[i], type_mask), etype);
            // Timestamps as zero-extended qwords against the record before (`carry`: the previous load's last);
            // only EVENT lanes count.
            const __m256i ts = _mm256_srli_epi64(v[i], 32);
            const __m256i before = _mm256_blend_epi32(_mm256_permute4x64_epi64(ts, 0x90), carry, 0x03);
            back = _mm256_or_si256(back, _mm256_and_si256(_mm256_cmpgt_epi64(before, ts), c));
            carry = _mm256_permute4x64_epi64(ts, 0xFF);
            if (!_mm256_testc_si256(c, ones)) {
                n += static_cast<uint32_t>(
                    std::countr_zero(~static_cast<uint32_t>(_mm256_movemask_pd(_mm256_castsi256_pd(c))) & 0x1Fu));
                break;
            }
            n += 4;
        }
        if (n > max_recs) {
            n = max_recs;
        }
        if (n == 0) {
            break;
        }
        if constexpr (Sink::kStores) {
            uint8_t* const dst = sw.buf + sw.off;
            for (uint32_t i = 0; 4 * i < n; i++) {
                const __m256i ts64 = _mm256_or_si256(_mm256_srli_epi64(v[i], 32), tv);
                const __m256i m64 = _mm256_or_si256(_mm256_and_si256(v[i], id_mask), mv);
                const __m256i t_lo = _mm256_unpacklo_epi64(ts64, z);
                const __m256i t_hi = _mm256_unpackhi_epi64(ts64, z);
                const __m256i mp_lo = _mm256_unpacklo_epi64(m64, pv);
                const __m256i mp_hi = _mm256_unpackhi_epi64(m64, pv);
                uint8_t* const o = dst + 128 * i;
                const uint32_t left = n - 4 * i;
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(o), _mm256_permute2x128_si256(t_lo, mp_lo, 0x20));
                if (left > 1) {
                    _mm256_storeu_si256(
                        reinterpret_cast<__m256i*>(o + 32), _mm256_permute2x128_si256(t_hi, mp_hi, 0x20));
                }
                if (left > 2) {
                    _mm256_storeu_si256(
                        reinterpret_cast<__m256i*>(o + 64), _mm256_permute2x128_si256(t_lo, mp_lo, 0x31));
                }
                if (left > 3) {
                    _mm256_storeu_si256(
                        reinterpret_cast<__m256i*>(o + 96), _mm256_permute2x128_si256(t_hi, mp_hi, 0x31));
                }
            }
            sw.off += kSpscRecBytes * n;
        }
        // The carry for the next block is this block's last record, which the load loop may have run past.
        carry = _mm256_set1_epi64x(static_cast<long long>(p[2u * n - 1u]));
        out.ts_last = th_hi | p[2u * n - 1u];
        total += n;
        if (n < 16u) {
            break;
        }
        p += 32;
        readable -= 32;
        max_recs -= 16;
    }
    out.n = total;
    out.regress = _mm256_testz_si256(back, back) ? 0u : 1u;
    return out;
}

struct SpscL8Result {
    uint64_t ts_last;
    uint32_t n;
    uint8_t regress;  // nonzero: some record's end precedes the one before it
    uint8_t wrapped;  // nonzero: some duration's high word is all ones (the start read borrowed the next epoch)
    uint16_t stalls;  // storing sinks only
};

// ZONE_L records, one 32 B load per record at a 20 B stride so {id, end, dur} sit at fixed dwords; a record
// composes as {end, dur, meta|id, prog} minus {dur, 0, ...} like the atomic block, the 64-bit duration whole.
// Consumes every consecutive record in one call. Compare results stay in vectors and are tested once after the
// loop. `avail` authorizes loads, never emits.
template <typename Sink>
inline SpscL8Result spsc_zone_l8(
    const uint32_t* p, uint32_t avail, uint32_t max_recs, const SpscLaneConsts& c, Sink& sw) {
    SpscL8Result out{0, 0, 0, 0, 0};
    const __m256i type_mask = _mm256_set1_epi32(static_cast<int>((0xFFFFFFFFu >> PP_TYPE_SHIFT) << PP_TYPE_SHIFT));
    const __m256i ltype = _mm256_set1_epi32(static_cast<int>(PP_ZONE_L << PP_TYPE_SHIFT));
    const __m256i lane_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i w0_mask = _mm256_setr_epi32(0x07FFFFFF, -1, -1, -1, -1, -1, -1, -1);
    const __m256i consts = c.zone_l_half;
    const __m256i idx_s = _mm256_setr_epi32(1, 2, 3, 4, 0, 0, 0, 0);
    const __m256i idx_d = _mm256_setr_epi32(3, 4, 0, 0, 0, 0, 0, 0);
    const __m256i ones = _mm256_set1_epi32(-1);
    const __m256i stall = _mm256_set1_epi32(static_cast<int>(kSpscStallZoneId));
    const __m256i z = _mm256_setzero_si256();
    const __m256i lane_0 = _mm256_setr_epi32(-1, 0, 0, 0, 0, 0, 0, 0);
    const __m256i lane_3 = _mm256_setr_epi32(0, 0, 0, -1, 0, 0, 0, 0);
    const __m256i lane_4 = _mm256_setr_epi32(0, 0, 0, 0, -1, 0, 0, 0);
    __m256i prev = z, back = z, wrap_hit = z, stall_hit = z;
    uint32_t n = 0;
    while (n < max_recs && avail >= 5u) {
        _mm_prefetch(reinterpret_cast<const char*>(p + 160), _MM_HINT_T0);
        __m256i v;
        if (avail >= 8u) {
            v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        } else {
            v = _mm256_maskload_epi32(
                reinterpret_cast<const int*>(p),
                _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(avail)), lane_idx));
        }
        if (!_mm256_testc_si256(_mm256_cmpeq_epi32(_mm256_and_si256(v, type_mask), ltype), lane_0)) {
            break;
        }
        const __m256i l = _mm256_and_si256(v, w0_mask);
        const __m256i s = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(l, idx_s), consts, 0xE0);
        // Lane 0 is the end and lane 3 the duration's high word; both are 59-bit-or-less as signed 64-bit.
        back = _mm256_or_si256(back, _mm256_cmpgt_epi64(prev, s));
        wrap_hit = _mm256_or_si256(wrap_hit, _mm256_cmpeq_epi32(s, ones));
        if constexpr (Sink::kStores) {
            stall_hit = _mm256_or_si256(stall_hit, _mm256_cmpeq_epi32(s, stall));
            const __m256i d = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(l, idx_d), z, 0xFC);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(sw.buf + sw.off + 32 * n), _mm256_sub_epi64(s, d));
        }
        prev = s;
        n++;
        p += 5;
        avail -= 5;
    }
    if (n == 0) {
        return out;
    }
    p -= 5u * n;
    out.n = n;
    out.ts_last = (static_cast<uint64_t>(p[5u * n - 3u]) << 32) | p[5u * n - 4u];
    out.regress = _mm256_testz_si256(back, lane_0) ? 0u : 1u;
    out.wrapped = _mm256_testz_si256(wrap_hit, lane_3) ? 0u : 1u;
    if constexpr (Sink::kStores) {
        if (__builtin_expect(!_mm256_testz_si256(stall_hit, lane_4), 0)) {
            for (uint32_t k = 0; k < n; k++) {
                out.stalls += (p[5u * k] & 0x07FFFFFFu) == kSpscStallZoneId ? 1u : 0u;
            }
        }
        sw.off += kSpscRecBytes * n;
    }
    return out;
}

// One point packet, EVENT or DATA, with no branch on which: the head {ts, 0, meta|id, prog}, then an Ext (payload
// words 0-1 and the count) and a Cont (words 2-3) written unconditionally and only counted for a DATA packet, so an
// EVENT's two spare records land in the slack the next record overwrites. Payload words past the count read as
// zero. Only a payload beyond four words takes the loop. Returns the records the packet counts for; `n` is the
// payload word count (0 for an EVENT).
// `dm` is all ones for a DATA packet and zero for an EVENT; everything DATA-only is masked by it, never selected by a
// branch, so random alternation costs no mispredicts.
template <typename Sink>
inline uint32_t spsc_point(
    const uint32_t* p, uint32_t readable, uint32_t dm, uint32_t n, const SpscLaneConsts& c, Sink& sw) {
    const uint32_t conts = n > 2u ? (n - 1u) / 2u : 0u;
    const uint32_t recs = 1u + (dm & (1u + conts));
    if constexpr (Sink::kStores) {
        const __m256i lane_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        // Words 0-2 and payload 0-4 of the packet, the payload lanes zeroed past the count and past readable.
        const uint32_t words = std::min(readable, 3u + n);
        const __m256i l = _mm256_maskload_epi32(
            reinterpret_cast<const int*>(p), _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(words)), lane_idx));
        const __m256i type_half = c.point_half[dm & 1u];
        const __m256i head = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(
                _mm256_and_si256(l, _mm256_setr_epi32(0x07FFFFFF, -1, -1, -1, -1, -1, -1, -1)),
                _mm256_setr_epi32(1, 0, 0, 0, 0, 0, 0, 0)),
            type_half,
            0xEE);
        uint8_t* const dst = sw.buf + sw.off;
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), head);
        const __m256i ext = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(l, _mm256_setr_epi32(4, 3, 0, 0, 0, 0, 0, 0)),
            _mm256_blend_epi32(c.ext_half, _mm256_set1_epi32(static_cast<int>(n)), 0x10),
            0xFC);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 32), ext);
        const __m256i cont = c.cont_half;
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(dst + 64),
            _mm256_blend_epi32(_mm256_permutevar8x32_epi32(l, _mm256_setr_epi32(6, 5, 0, 0, 0, 0, 0, 0)), cont, 0xFC));
        if (__builtin_expect(n > 4u, 0)) {
            uint8_t* o = dst + 96;
            const uint32_t pw = readable > 3u ? std::min(readable - 3u, n) : 0u;
            for (uint32_t k = 4; k < n; k += 8) {
                const uint32_t left = std::min(n - k, pw > k ? pw - k : 0u);
                const __m256i pl = _mm256_maskload_epi32(
                    reinterpret_cast<const int*>(p + 3 + k),
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
    } else {
        (void)p;
        (void)readable;
        (void)c;
    }
    return recs;
}

// Four EVENT records start at p (readable >= 8 words): the gate for the block kernel, so random single points never
// pay a block call and a run never pays the one-record path.
inline bool spsc_event_run4(const uint32_t* p) {
    const __m256i type_mask =
        _mm256_set1_epi64x(static_cast<long long>((0xFFFFFFFFull >> PP_TYPE_SHIFT) << PP_TYPE_SHIFT));
    const __m256i etype = _mm256_set1_epi64x(static_cast<long long>(static_cast<uint64_t>(PP_EVENT) << PP_TYPE_SHIFT));
    const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    const __m256i c = _mm256_cmpeq_epi64(_mm256_and_si256(v, type_mask), etype);
    return _mm256_testc_si256(c, _mm256_cmpeq_epi64(c, c));
}

// Decode one packed BULK_SPAN frame in place. Every record composes through a vector kernel: a run of two or more
// same-type records through its block kernel (em.zone_s16 / em.atomic8 / em.event16 / em.zone_l8, which return the
// records consumed, 0 = truncated), a lone zone through its one-record composer (em.atomic1 / em.zone_l1), a
// point packet (EVENT or DATA) through em.point. Each takes the lane's SpscLaneConsts. enter_lane(lane) /
// leave_lane(lane) bracket each lane's run. Returns the payload words the control vector implies, which the caller
// checks against the frame's length field (a pack-rule disagreement desynchronizes every later lane), or 0 for an
// unknown core. Decode starts at the larger of the head mirror and the extent's start: the mirror runs behind after
// an upstream loss (adopt and count), the extent after a lagging head write-back (skip the overlap).
// always_inline: the emitters keep per-lane state in captured locals, which stay in registers only while the walk
// and its caller are one function.
template <typename Emitters, typename EnterLane, typename LeaveLane>
inline __attribute__((always_inline)) uint32_t spsc_decode_frame(
    SpanDecodeState& st,
    const uint32_t* frame,
    uint32_t dev,
    Emitters& em,
    EnterLane&& enter_lane,
    LeaveLane&& leave_lane,
    // Nonzero authorizes the kernels to load (never emit) past a lane's live run, up to the frame's end.
    uint32_t frame_words = 0) {
    const uint32_t* ctrl = frame + kernel_profiler::SPSC_SPAN_PREFIX_WORDS;
    const uint32_t core = st.core_of_xy.find(ctrl[kernel_profiler::SPSC_WIRE_XY]);
    if (core == CoreTable::kNone) {
        st.unknown_core_frames++;
        return 0;
    }
    // Folded into st once at the end: the emitters store through casted ring pointers, so the compiler must
    // assume those stores alias st and would reload per record.
    uint64_t lw = 0;
    uint32_t off = kernel_profiler::SPSC_SPAN_PREFIX_WORDS + kernel_profiler::SPSC_SPAN_WIRE_CTRL_WORDS;
    SpscLaneConsts lc;
    for (uint32_t r = 0; r < kSpscNRiscDecode; r++) {
        const uint32_t lane = core * kSpscNRiscDecode + r;
        const uint32_t tail = ctrl[kernel_profiler::SPSC_WIRE_TAIL_0 + r];
        const uint32_t frame_head = ctrl[kernel_profiler::SPSC_WIRE_HEAD_0 + r];
        const uint32_t extent = kernel_profiler::spsc_span_live(frame_head, tail, kSpscRingCap);
        if (extent != tail - frame_head) {
            st.anomalies++;  // torn snapshot; the clamped geometry still frames consistently on both sides
        }
        const uint32_t start = tail - extent;
        const uint32_t* p = nullptr;
        // A near-full wrapping run arrives as the whole ring image (predicate shared with the device,
        // spsc_span_wrap_image), the pad is phased for ring offset 0, and the payload advance is the full ring.
        const bool ring_ordered = extent != 0 && kernel_profiler::spsc_span_wrap_image(start, extent, kSpscRingCap);
        if (extent != 0) {
            off += kernel_profiler::spsc_span_pack_pad(ring_ordered ? 0u : start, off);
            p = frame + off;
            off += ring_ordered ? kSpscRingCap : extent;
            // The frame is read in place from memory the device may be rewriting, so a torn control vector must
            // not send the walk past the frame's own length.
            if (frame_words != 0 && off > frame_words) {
                st.anomalies++;
                break;
            }
        }
        uint32_t head;
        if (st.seeded[lane] == 0) {
            st.seeded[lane] = 1;
            head = start;
        } else {
            head = st.head[lane];
            const int32_t behind = static_cast<int32_t>(start - head);
            if (behind > 0) {
                st.resync_words += static_cast<uint32_t>(behind);
                head = start;
            }
        }
        st.head[lane] = tail;
        const uint32_t run = tail - head;
        if (run == 0) {
            continue;
        }
        if (run > extent) {
            st.anomalies++;
            continue;
        }
        lw += run;
        uint32_t th = st.timer_hi[lane];
        uint32_t pg = st.prog[lane];
        uint64_t cur = st.cursor[lane];
        // A wrap image is the whole ring in ring order, so its run is linearised before the walk.
        uint32_t lin[kSpscRingCap];
        if (ring_ordered) {
            const uint32_t hm = head & kSpscRingMask;
            const uint32_t first = kSpscRingCap - hm < run ? kSpscRingCap - hm : run;
            std::memcpy(lin, p + hm, first * sizeof(uint32_t));
            if (first < run) {
                std::memcpy(lin + first, p, (run - first) * sizeof(uint32_t));
            }
            p = lin;
        } else {
            p += extent - run;
        }
        // Loads may run to here (never emits): the frame's end, or the run's when the run was linearised (`lin` is
        // not comparable with the frame pointer).
        const uint32_t* const rd_end = (frame_words != 0 && !ring_ordered) ? frame + frame_words : p + run;
        spsc_lane_consts_lane(lc, lane, dev);
        spsc_lane_consts_sticky(lc, th, pg);
        enter_lane(lane);
        uint32_t i = 0;
        while (i < run) {
            const uint32_t w0 = p[i];
            const uint32_t t = pp_type(w0);
            const uint32_t readable = static_cast<uint32_t>(rd_end - (p + i));
            const uint32_t left = run - i;
            uint32_t got = 0;
            if (t == PP_ZONE_S) {
                const auto zs = em.zone_s16(lane, cur, lc, p + i, readable, left / 2u);
                cur = zs.n != 0 ? zs.ts_last : cur;
                got = 2u * zs.n;
            } else if (t == PP_ZONE_ATOMIC) {
                if (left >= 3u) {
                    if (left > 3u && pp_type(p[i + 3u]) == PP_ZONE_ATOMIC) {
                        const uint32_t n = em.atomic8(lane, lc, p + i, readable, left / 3u);
                        // A block is atomics only (a sticky ends it), so th is constant across it and the last
                        // end re-anchors the lane cursor.
                        cur = n != 0 ? pp_full_ts(th, p[i + 3u * (n - 1u) + 1u]) : cur;
                        got = 3u * n;
                    } else {
                        cur = em.atomic1(lane, lc, p + i, readable);  // absolute end re-anchors the lane cursor
                        got = 3;
                    }
                }
            } else if (t == PP_EVENT || t == PP_DATA) {
                // Both are points: the same head record, DATA with a payload behind a size word. One branch for the
                // pair keeps random alternation from mispredicting; only an EVENT run of four takes the block kernel.
                // Tested for both types so the branch does not follow the type: false for any DATA.
                if (left >= 8u && readable >= 8u && spsc_event_run4(p + i)) {
                    got = 2u * em.event16(lane, lc, p + i, readable, left / 2u);
                } else {
                    // No branch follows the type from here: the DATA-only quantities are masked, not selected.
                    const uint32_t dm = 0u - static_cast<uint32_t>(t == PP_DATA);
                    const uint32_t w2 = p[i + (readable >= 3u ? 2u : 1u)];
                    const uint32_t n = pp_data_size(w2) & dm;
                    const uint32_t words = 2u + (dm & (1u + n));
                    if (left >= words) {
                        em.point(lane, lc, p + i, readable, dm, n);
                        got = words;
                    }
                }
            } else if (t == PP_ZONE_L) {
                if (left >= 5u) {
                    if (left > 5u && pp_type(p[i + 5u]) == PP_ZONE_L) {
                        got = 5u * em.zone_l8(lane, lc, p + i, readable, left / 5u);  // cursor untouched
                    } else {
                        em.zone_l1(lane, lc, p + i, readable);
                        got = 5;
                    }
                }
            } else if (t == PP_STICKY_TIMER) {
                th = pp_timer_hi(w0);
                spsc_lane_consts_th(lc, th);
                got = 1;
            } else if (t == PP_STICKY_PROG) {
                pg = pp_low27(w0);
                spsc_lane_consts_prog(lc, pg);
                got = 1;
            } else if (t == PP_STICKY_PROG_EXT) {
                if (left >= 2u) {
                    pg = p[i + 1];
                    spsc_lane_consts_prog(lc, pg);
                    got = 2;
                }
            }
            if (got == 0) {
                st.anomalies++;  // undecodable word, or a record cut by the run's end
                break;
            }
            i += got;
        }
        leave_lane(lane);
        st.timer_hi[lane] = th;
        st.prog[lane] = pg;
        st.cursor[lane] = cur;
    }
    st.live_words += lw;
    return off - kernel_profiler::SPSC_SPAN_PREFIX_WORDS;
}

}  // namespace tt::tt_metal::profiler
