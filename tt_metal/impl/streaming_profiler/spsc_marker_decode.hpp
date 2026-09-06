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

// Everything a lane's records share: the constant half of each record kind, {0, th, 0, 0, 0, meta|type, prog, 0}
// (the 64-bit-end kinds without th), and the qword broadcasts the ZONE_S and EVENT kernels compose from. Built at
// lane entry; a STICKY_TIMER re-blends only the th lanes, a STICKY_PROG only the prog lanes.
struct SpscLaneConsts {
    uint64_t th_hi;
    __m256i tv, pv;                                                      // th << 32 / prog, per qword
    __m256i mv_zone, mv_event;                                           // meta | type << 29, high dword of every qword
    __m256i zone_half, zone_l_half, point_half[2], ext_half, cont_half;  // point_half: [0] EVENT, [1] DATA
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
    c.zone_half = _mm256_blend_epi32(c.zone_half, pgv, 0x40);
    c.zone_l_half = _mm256_blend_epi32(c.zone_l_half, pgv, 0x40);
    c.point_half[0] = _mm256_blend_epi32(c.point_half[0], pgv, 0x40);
    c.point_half[1] = _mm256_blend_epi32(c.point_half[1], pgv, 0x40);
    c.ext_half = _mm256_blend_epi32(c.ext_half, pgv, 0x40);
    c.cont_half = _mm256_blend_epi32(c.cont_half, pgv, 0x40);
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
    c.zone_l_half = c.zone_half;
    c.point_half[0] = half(kSpscRecTypeEvent);
    c.point_half[1] = half(kSpscRecTypeData);
    c.ext_half = half(kSpscRecTypeExt);
    c.cont_half = half(kSpscRecTypeCont);
    spsc_lane_consts_th(c, th);
    spsc_lane_consts_prog(c, prog);
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
inline __m256i spsc_w0_mask() { return _mm256_setr_epi32(0x07FFFFFF, -1, -1, -1, -1, -1, -1, -1); }

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

// One zone from its words: a lane permute puts {end, th, dur, ...} in place, a blend supplies the constant half, and
// subtracting {dur, 0, ...} leaves {start, dur, meta|id, prog} with the borrow in the high half.
template <int kBlendS, int kBlendD>
inline void spsc_one(
    const uint32_t* p, uint32_t readable, __m256i half, __m256i idx_s, __m256i idx_d, SpscRecSink& sw) {
    const __m256i l = _mm256_and_si256(spsc_words(p, static_cast<int32_t>(readable)), spsc_w0_mask());
    const __m256i s = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(l, idx_s), half, kBlendS);
    const __m256i d = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(l, idx_d), _mm256_setzero_si256(), kBlendD);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(sw.buf + sw.off), _mm256_sub_epi64(s, d));
    sw.off += kSpscRecBytes;
}
inline void spsc_atomic1(const uint32_t* p, uint32_t readable, const SpscLaneConsts& c, SpscRecSink& sw) {
    spsc_one<0xEA, 0xFE>(
        p,
        readable,
        c.zone_half,
        _mm256_setr_epi32(1, 0, 2, 0, 0, 0, 0, 0),
        _mm256_setr_epi32(2, 0, 0, 0, 0, 0, 0, 0),
        sw);
}
inline void spsc_zone_l1(const uint32_t* p, uint32_t readable, const SpscLaneConsts& c, SpscRecSink& sw) {
    spsc_one<0xE0, 0xFC>(
        p,
        readable,
        c.zone_l_half,
        _mm256_setr_epi32(1, 2, 3, 4, 0, 0, 0, 0),
        _mm256_setr_epi32(3, 4, 0, 0, 0, 0, 0, 0),
        sw);
}

// ZONE_ATOMIC records, eight per block through four 32 B loads at a 24 B stride, so each load holds two whole
// records with every field at a fixed dword. A record composes with two lane permutes off its load:
// {end, th, dur, 0, id, meta, prog, 0} minus {dur, 0, ...} is {start, dur, meta|id, prog}, the borrow landing in
// the high half. Consumes every consecutive record in one call, so a run pays the walk's per-call cost once.
// Compare results stay in vectors and are tested once; a mask is only extracted when a test fires, since each
// vector-to-scalar move costs more than the compose itself. `avail` authorizes loads, never emits.
inline SpscBlockResult spsc_atomic8(
    const uint32_t* p, uint32_t avail, uint32_t max_recs, const SpscLaneConsts& c, SpscRecSink& sw) {
    SpscBlockResult out{0, 0, 0, 0, 0};
    const __m256i type_mask = spsc_dw_type_mask();
    const __m256i atype = spsc_dw_type(PP_ZONE_ATOMIC);
    const __m256i lanes_w0 = _mm256_setr_epi32(-1, 0, 0, -1, 0, 0, 0, 0);
    const __m256i lane_0 = spsc_lane0();
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
        __m256i v[4];
        uint32_t n = 0;
        for (int i = 0; i < 4; i++) {
            if (avail >= 26u) {
                v[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 6 * i));
            } else {
                const __m256i mk =
                    _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(avail) - 6 * i), spsc_lane_idx());
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
            stall_hit = _mm256_or_si256(stall_hit, _mm256_cmpeq_epi32(l, stall));
            const __m256i da = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(l, idx_da), z, 0xFE);
            const __m256i db = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(l, idx_db), z, 0xFE);
            uint8_t* const o = sw.buf + sw.off + 64 * i;
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(o), _mm256_sub_epi64(sa, da));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(o + 32), _mm256_sub_epi64(sb, db));
        }
        if (2 * i < n) {  // an odd last record: the load's second record is not ours
            const __m256i l = _mm256_and_si256(v[i], w0_mask);
            const __m256i sa = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(l, idx_a), consts, 0xEA);
            back = _mm256_or_si256(back, _mm256_cmpgt_epi64(prev, sa));
            prev = sa;
            stall_hit = _mm256_or_si256(stall_hit, _mm256_and_si256(_mm256_cmpeq_epi32(l, stall), lane_0));
            const __m256i da = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(l, idx_da), z, 0xFE);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(sw.buf + sw.off + 64 * i), _mm256_sub_epi64(sa, da));
        }
        out.ts_last = th_hi | p[3u * n - 2u];
        total += n;
        sw.off += kSpscRecBytes * n;
        if (n < 8u) {
            break;
        }
        p += 24;
        avail -= 24;
        max_recs -= 8;
    }
    out.n = total;
    out.regress = _mm256_testz_si256(back, lane_0) ? 0u : 1u;
    if (__builtin_expect(!_mm256_testz_si256(stall_hit, lanes_w0), 0)) {
        for (uint32_t k = 0; k < total; k++) {
            out.stalls += (p0[3u * k] & 0x07FFFFFFu) == kSpscStallZoneId ? 1u : 0u;
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
inline SpscBlockResult spsc_zone_s16(
    const uint32_t* p,
    uint32_t readable,
    uint32_t max_recs,
    uint64_t cursor,
    const SpscLaneConsts& c,
    SpscRecSink& sw) {
    SpscBlockResult out{cursor, 0, 0, 0, 0};
    const __m256i type_mask = spsc_qw_type_mask();
    const __m256i stype = spsc_qw_type(PP_ZONE_S);
    const __m256i z = _mm256_setzero_si256();
    const __m256i id_mask = _mm256_set1_epi64x(0x07FFFFFF);
    const __m256i dur_mask = _mm256_set1_epi64x(0xFFFF);
    // The cursor rides as a broadcast vector: the next block's starts need it as one, and the block total is already
    // a broadcast lane of the carry tree.
    __m256i cv = _mm256_set1_epi64x(static_cast<long long>(cursor));
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
                const __m256i m = _mm256_cmpgt_epi64(_mm256_set1_epi64x(recs - 4 * i), _mm256_setr_epi64x(0, 1, 2, 3));
                v[i] = _mm256_maskload_epi64(reinterpret_cast<const long long*>(p + 8 * i), m);
            }
        }
        // Both lines of the block four blocks ahead: the hardware prefetcher does not run far enough into DMA-landed
        // memory (-6%); farther ahead evicts before use once the sink's output stream shares L1, and one line per
        // block leaves every other line to the hardware.
        _mm_prefetch(reinterpret_cast<const char*>(p + 128), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(p + 144), _MM_HINT_T0);
        uint32_t n = 0;
        for (int i = 0; i < 4; i++) {
            const uint32_t k = spsc_quad_count(_mm256_cmpeq_epi64(_mm256_and_si256(v[i], type_mask), stype));
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
        uint8_t* const dst = sw.buf + sw.off;
        for (uint32_t i = 0; i < 4 && 4 * i < n; i++) {
            const __m256i d64 = _mm256_and_si256(_mm256_srli_epi64(v[i], 32), dur_mask);
            const __m256i s64 = _mm256_sub_epi64(_mm256_add_epi64(cv, pfx[i]), d64);
            const __m256i m64 = _mm256_or_si256(_mm256_and_si256(v[i], id_mask), c.mv_zone);
            spsc_store_quad(dst + 128 * i, s64, d64, m64, c.pv);
        }
        sw.off += kSpscRecBytes * n;
        out.n += n;
        if (n < 16u) {
            // The prefix at record n-1 through a spill: a variable lane extract costs more than an aligned store to hot
            // stack.
            alignas(32) uint64_t arr[16];
            for (int i = 0; i < 4; i++) {
                _mm256_store_si256(reinterpret_cast<__m256i*>(arr + 4 * i), pfx[i]);
            }
            out.ts_last = static_cast<uint64_t>(_mm_cvtsi128_si64(_mm256_castsi256_si128(cv))) + arr[n - 1];
            return out;  // the block ended on a non-S word or the emit budget
        }
        cv = _mm256_add_epi64(cv, _mm256_permute4x64_epi64(pfx[3], 0xFF));
        p += 32;
        readable -= 32;
        max_recs -= 16;
    }
    if (out.n != 0) {
        out.ts_last = static_cast<uint64_t>(_mm_cvtsi128_si64(_mm256_castsi256_si128(cv)));
    }
    return out;
}

// EVENT records on the wire's own layout: a record is one qword (w0 low, timer_low high), four per load, and
// composes as {ts, 0, meta|id, prog} from two unpacks and a lane permute. Consumes every consecutive record in one
// call; compare results stay in vectors and are tested once. `stalls` is always 0.
inline SpscBlockResult spsc_event16(
    const uint32_t* p, uint32_t readable, uint32_t max_recs, const SpscLaneConsts& c, SpscRecSink& sw) {
    SpscBlockResult out{0, 0, 0, 0, 0};
    const __m256i type_mask = spsc_qw_type_mask();
    const __m256i etype = spsc_qw_type(PP_EVENT);
    const __m256i z = _mm256_setzero_si256();
    const __m256i id_mask = _mm256_set1_epi64x(0x07FFFFFF);
    const uint64_t th_hi = c.th_hi;
    __m256i carry = z, back = z;
    uint32_t total = 0;
    while (max_recs != 0 && readable >= 2u) {
        // One load at a time, stopping at the first that is not all EVENT: a lone record costs one load.
        __m256i v[4];
        uint32_t n = 0;
        for (int i = 0; i < 4; i++) {
            v[i] = spsc_qwords(p + 8 * i, static_cast<int32_t>(readable) - 8 * i);
            const __m256i c = _mm256_cmpeq_epi64(_mm256_and_si256(v[i], type_mask), etype);
            // Timestamps as zero-extended qwords against the record before (`carry`: the previous load's last);
            // only EVENT lanes count.
            const __m256i ts = _mm256_srli_epi64(v[i], 32);
            const __m256i before = _mm256_blend_epi32(_mm256_permute4x64_epi64(ts, 0x90), carry, 0x03);
            back = _mm256_or_si256(back, _mm256_and_si256(_mm256_cmpgt_epi64(before, ts), c));
            carry = _mm256_permute4x64_epi64(ts, 0xFF);
            const uint32_t k = spsc_quad_count(c);
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
        uint8_t* const dst = sw.buf + sw.off;
        for (uint32_t i = 0; 4 * i < n; i++) {
            const __m256i ts64 = _mm256_or_si256(_mm256_srli_epi64(v[i], 32), c.tv);
            const __m256i m64 = _mm256_or_si256(_mm256_and_si256(v[i], id_mask), c.mv_event);
            spsc_store_quad(dst + 128 * i, ts64, z, m64, c.pv);
        }
        sw.off += kSpscRecBytes * n;
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

// ZONE_L records, one 32 B load per record at a 20 B stride so {id, end, dur} sit at fixed dwords; a record
// composes as {end, dur, meta|id, prog} minus {dur, 0, ...} like the atomic block, the 64-bit duration whole.
// Consumes every consecutive record in one call. Compare results stay in vectors and are tested once after the
// loop. `avail` authorizes loads, never emits.
inline SpscBlockResult spsc_zone_l8(
    const uint32_t* p, uint32_t avail, uint32_t max_recs, const SpscLaneConsts& c, SpscRecSink& sw) {
    SpscBlockResult out{0, 0, 0, 0, 0};
    const __m256i type_mask = spsc_dw_type_mask();
    const __m256i ltype = spsc_dw_type(PP_ZONE_L);
    const __m256i w0_mask = spsc_w0_mask();
    const __m256i consts = c.zone_l_half;
    const __m256i idx_s = _mm256_setr_epi32(1, 2, 3, 4, 0, 0, 0, 0);
    const __m256i idx_d = _mm256_setr_epi32(3, 4, 0, 0, 0, 0, 0, 0);
    const __m256i ones = _mm256_set1_epi32(-1);
    const __m256i stall = _mm256_set1_epi32(static_cast<int>(kSpscStallZoneId));
    const __m256i z = _mm256_setzero_si256();
    const __m256i lane_0 = spsc_lane0();
    const __m256i lane_3 = _mm256_setr_epi32(0, 0, 0, -1, 0, 0, 0, 0);
    const __m256i lane_4 = _mm256_setr_epi32(0, 0, 0, 0, -1, 0, 0, 0);
    __m256i prev = z, back = z, wrap_hit = z, stall_hit = z;
    uint32_t n = 0;
    while (n < max_recs && avail >= 5u) {
        const __m256i v = spsc_words(p, static_cast<int32_t>(avail));
        if (!_mm256_testc_si256(_mm256_cmpeq_epi32(_mm256_and_si256(v, type_mask), ltype), lane_0)) {
            break;
        }
        const __m256i l = _mm256_and_si256(v, w0_mask);
        const __m256i s = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(l, idx_s), consts, 0xE0);
        // Lane 0 is the end and lane 3 the duration's high word; both are 59-bit-or-less as signed 64-bit.
        back = _mm256_or_si256(back, _mm256_cmpgt_epi64(prev, s));
        wrap_hit = _mm256_or_si256(wrap_hit, _mm256_cmpeq_epi32(s, ones));
        stall_hit = _mm256_or_si256(stall_hit, _mm256_cmpeq_epi32(s, stall));
        const __m256i d = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(l, idx_d), z, 0xFC);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(sw.buf + sw.off + 32 * n), _mm256_sub_epi64(s, d));
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
    if (__builtin_expect(!_mm256_testz_si256(stall_hit, lane_4), 0)) {
        for (uint32_t k = 0; k < n; k++) {
            out.stalls += (p[5u * k] & 0x07FFFFFFu) == kSpscStallZoneId ? 1u : 0u;
        }
    }
    sw.off += kSpscRecBytes * n;
    return out;
}

// One point packet, EVENT or DATA, with no branch on which: the head {ts, 0, meta|id, prog}, then an Ext (payload
// words 0-1 and the count) and a Cont (words 2-3) written unconditionally and only counted for a DATA packet, so an
// EVENT's two spare records land in the slack the next record overwrites. Payload words past the count read as
// zero. Only a payload beyond four words takes the loop. Returns the records the packet counts for; `n` is the
// payload word count (0 for an EVENT).
// `dm` is all ones for a DATA packet and zero for an EVENT; everything DATA-only is masked by it, never selected by a
// branch, so random alternation costs no mispredicts.
inline uint32_t spsc_point(
    const uint32_t* p, uint32_t readable, uint32_t dm, uint32_t n, const SpscLaneConsts& c, SpscRecSink& sw) {
    const uint32_t conts = n > 2u ? (n - 1u) / 2u : 0u;
    const uint32_t recs = 1u + (dm & (1u + conts));
    const __m256i lane_idx = spsc_lane_idx();
    // Words 0-2 and payload 0-4 of the packet, the payload lanes zeroed past the count and past readable. Always a
    // masked load: a branch on the packet's size would follow the type and mispredict on alternation.
    const uint32_t words = std::min(readable, 3u + n);
    const __m256i l = _mm256_maskload_epi32(
        reinterpret_cast<const int*>(p), _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(words)), lane_idx));
    const __m256i type_half = c.point_half[dm & 1u];
    const __m256i head = _mm256_blend_epi32(
        _mm256_permutevar8x32_epi32(_mm256_and_si256(l, spsc_w0_mask()), _mm256_setr_epi32(1, 0, 0, 0, 0, 0, 0, 0)),
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
    return recs;
}

// Four EVENT records start at p (readable >= 8 words): the gate for the block kernel, so random single points never
// pay a block call and a run never pays the one-record path.
inline bool spsc_event_run4(const uint32_t* p) {
    const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    const __m256i c = _mm256_cmpeq_epi64(_mm256_and_si256(v, spsc_qw_type_mask()), spsc_qw_type(PP_EVENT));
    return _mm256_testc_si256(c, _mm256_cmpeq_epi64(c, c));
}

}  // namespace tt::tt_metal::profiler
