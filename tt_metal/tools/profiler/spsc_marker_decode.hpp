// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// SINGLE SOURCE OF TRUTH for the host-side decode of the DRISC drainer's wire
// (producer: tt_metal/tools/profiler/kernels/drisc_profiler_drain.cpp).
//
// The wire carries only whole variable-length BULK_SPAN frames (layout and geometry rules in
// profiler_common.h): a 16-word prefix whose word 1 is the payload length, the worker's 64-word control
// vector verbatim, then each RISC's live ring window packed flat -- congruence-padded, wrap resolved
// device-side. Frames with SPSC_SPAN_RAW_FLAG in word 0 instead carry the whole raw span (five full
// rings at fixed offsets, windows circular) -- the drainer's high-fill fallback, where packing would
// cost write issues to save almost nothing. Inside each window is a packet run (spsc_packet.h):
// ZONE_ATOMIC packets (3 words: id | end timer_low | duration), legacy ZONE_START/END markers (2 words), STICKY_TIMER
// (1 word, per-lane wall-clock high half), STICKY_PROG (1 word, per-lane runtime host-id in low27; 2-word PROG_EXT
// escape past 2^27), EVENT (2 words, a payload-less flag) and DATA (3 + size words, self-describing -- the length lives
// in its word2). The producer publishes its tail only on packet boundaries, so a window never ends mid-packet.
#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <unordered_map>
#include <vector>

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "hostdevcommon/profiler_common.h"
#include "spsc_packet.h"

static_assert(
    PP_BULK_SPAN == kernel_profiler::SPSC_SPAN_PACKET_TYPE,
    "spsc_packet.h (plain C, drainer firmware) and profiler_common.h (C++, metal kernels) must agree on the "
    "BULK_SPAN wire code -- they cannot include each other, so this is the only thing holding them together");
// Same argument for the three codes the DRISC self-profiling packer in profiler_common.h emits directly.
static_assert(PP_ZONE_START == kernel_profiler::SPSC_TYPE_ZONE_START, "ZONE_START wire code disagrees");
static_assert(PP_ZONE_END == kernel_profiler::SPSC_TYPE_ZONE_END, "ZONE_END wire code disagrees");
static_assert(PP_ZONE_ATOMIC == kernel_profiler::SPSC_TYPE_ZONE_ATOMIC, "ZONE_ATOMIC wire code disagrees");
static_assert(PP_STICKY_TIMER == kernel_profiler::SPSC_TYPE_STICKY_TIMER, "STICKY_TIMER wire code disagrees");
static_assert(PP_TYPE_SHIFT == kernel_profiler::SPSC_SPAN_TYPE_SHIFT, "packet type field moved");
// The DRISC drain kernel keeps its OWN copy of the PP_DATA packer (it cannot include kernel_profiler.hpp),
// so its layout constants have to be pinned against this header's. Widening the id and moving the size
// word without updating that copy is not a crash -- it renders every one of its markers perfectly, with
// correct timestamps and nesting, under the WRONG identity. This translation unit is the only one that
// sees both headers, which makes it the only place the two can be held together.
static_assert(PP_DATA == kernel_profiler::SPSC_TYPE_DATA, "PP_DATA wire code disagrees");
static_assert(PP_DATA_SIZE_SHIFT == kernel_profiler::SPSC_DATA_SIZE_SHIFT, "PP_DATA size field moved");

namespace tt::tt_metal::profiler {

// Worker per-RISC SPSC ring depth (words) and RISC count -- MUST match the producer (kernel_profiler.hpp
// RING_CAPACITY, = kernel_profiler::PROFILER_L1_VECTOR_SIZE) so run clamps agree with the drainer's.
inline constexpr uint32_t kSpscRingCap = 512;
inline constexpr uint32_t kSpscRingMask = kSpscRingCap - 1;
inline constexpr uint32_t kSpscNRiscDecode = 5;

// Largest PP_DATA payload the 7-bit size field can express; bounds the raw-layout unwrap scratch.
inline constexpr uint32_t kSpscMaxDataWords = 127;

// Worst case: five full rings, each behind a maximal congruence pad. Larger than the raw 2,640-word span
// the drainer stages, so this bounds the bounce buffer and frame validation, not any device layout.
inline constexpr uint32_t kSpscMaxPayloadWords =
    kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE +
    kSpscNRiscDecode * (kSpscRingCap + kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS - 1);
inline constexpr uint32_t kSpscMaxFrameWords = kernel_profiler::spsc_span_frame_words(kSpscMaxPayloadWords);
inline constexpr uint32_t kSpscMaxFramePages = kSpscMaxFrameWords / kernel_profiler::SPSC_SPAN_PAGE_WORDS;
static_assert(kSpscMaxFrameWords == 2656 && kSpscMaxFramePages == 166);

// Decode state for one socket's frame stream. Written only by that socket's decode thread.
struct SpanDecodeState {
    std::vector<uint32_t> timer_hi;  // per lane: sticky wall-clock high half
    std::vector<uint32_t> prog;      // per lane: sticky runtime host-id (every RISC emits its own at launch)
    std::vector<uint32_t> head;      // per lane: monotonic words-consumed mirror; head(N) == tail(N-1)
    std::vector<uint8_t> seeded;
    std::unordered_map<uint32_t, uint32_t> core_of_xy;  // packed (y<<16)|x -> dense core index
    uint64_t live_words = 0;
    uint64_t resync_events = 0;
    uint64_t resync_words = 0;
    uint64_t head_lag = 0;
    uint64_t anomalies = 0;  // torn run / truncated run / undecodable word
    // Vector-block accounting: records taken by the 8-wide pair/atomic paths vs the scalar fallback, and
    // how often a would-be vector block was rejected by the type screen (record-type mix fragmenting SIMD).
    uint64_t vec_zone_recs = 0, vec_atomic_recs = 0, scalar_recs = 0, vec_block_rejects = 0;
    uint64_t unknown_core_frames = 0;

    void reset(uint32_t num_cores) {
        timer_hi.assign(static_cast<size_t>(num_cores) * kSpscNRiscDecode, 0);
        prog.assign(static_cast<size_t>(num_cores) * kSpscNRiscDecode, 0);
        head.assign(static_cast<size_t>(num_cores) * kSpscNRiscDecode, 0);
        seeded.assign(static_cast<size_t>(num_cores) * kSpscNRiscDecode, 0);
    }
};

struct SpscIgnoreProg {
    void operator()(uint32_t /*lane*/, uint32_t /*prog*/) const {}
};

// Sentinel defaults for the vectorized block sinks: without real ones the walk stays scalar.
struct SpscNoZones8 {};
struct SpscNoAtomic8 {};

#if defined(__AVX2__)
// Deinterleave EIGHT consecutive 3-word atomic records (24 contiguous words) into vectors of word0
// (type|id27), word1 (ts low) and word2 (duration). Output lane j belongs to record j; the stride-3
// scatter across three source vectors costs 9 lane permutes + 6 blends, against 24 scalar loads.
inline void spsc_load_atomic8(const uint32_t* src, __m256i& w0s, __m256i& w1s, __m256i& w2s) {
    const __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
    const __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 8));
    const __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 16));
    const __m256i iA0 = _mm256_setr_epi32(0, 3, 6, 0, 0, 0, 0, 0);
    const __m256i iB0 = _mm256_setr_epi32(0, 0, 0, 1, 4, 7, 0, 0);
    const __m256i iC0 = _mm256_setr_epi32(0, 0, 0, 0, 0, 0, 2, 5);
    const __m256i iA1 = _mm256_setr_epi32(1, 4, 7, 0, 0, 0, 0, 0);
    const __m256i iB1 = _mm256_setr_epi32(0, 0, 0, 2, 5, 0, 0, 0);
    const __m256i iC1 = _mm256_setr_epi32(0, 0, 0, 0, 0, 0, 3, 6);
    const __m256i iA2 = _mm256_setr_epi32(2, 5, 0, 0, 0, 0, 0, 0);
    const __m256i iB2 = _mm256_setr_epi32(0, 0, 0, 3, 6, 0, 0, 0);
    const __m256i iC2 = _mm256_setr_epi32(0, 0, 1, 0, 0, 1, 4, 7);
    w0s = _mm256_blend_epi32(
        _mm256_blend_epi32(_mm256_permutevar8x32_epi32(v0, iA0), _mm256_permutevar8x32_epi32(v1, iB0), 0b00111000),
        _mm256_permutevar8x32_epi32(v2, iC0),
        0b11000000);
    w1s = _mm256_blend_epi32(
        _mm256_blend_epi32(_mm256_permutevar8x32_epi32(v0, iA1), _mm256_permutevar8x32_epi32(v1, iB1), 0b00011000),
        _mm256_permutevar8x32_epi32(v2, iC1),
        0b11100000);
    w2s = _mm256_blend_epi32(
        _mm256_blend_epi32(_mm256_permutevar8x32_epi32(v0, iA2), _mm256_permutevar8x32_epi32(v1, iB2), 0b00011100),
        _mm256_permutevar8x32_epi32(v2, iC2),
        0b11100000);
}
#endif

inline void spsc_prefetch(const void* p) {
#if defined(__x86_64__)
    _mm_prefetch(static_cast<const char*>(p), _MM_HINT_T0);
#else
    (void)p;
#endif
}

// Decode ONE whole packed BULK_SPAN frame in place. For each zone packet calls
//   emit(lane, wire_type, zone_id27, full_ts, dur, prog)
// where for ZONE_ATOMIC full_ts is the zone END and dur its 32-bit duration (start = end - dur), and
// for the legacy ZONE_START/END markers (stall zone, >3.2s fallback) dur is 0.
// zone_id27 is the FULL 27-bit structural zone id (hostdevcommon/profiler_zone_id.h).
// For each PP_DATA/PP_EVENT it calls
//   emit_data(lane, wire_type, id, full_ts, prog, payload_words, n)   (payload in place, hi-word first)
// and emit_prog(lane, prog) whenever a lane's sticky host-id changes.
//
// Returns the payload words the control vector implies -- the caller checks it against the frame's own
// length field, since a pack-rule disagreement with the drainer desynchronizes every lane after the
// first -- or 0 for an unknown-core frame (decoded as nothing; the caller still owns the advance).
//
// Head adoption: the mirror and the extent's start are both monotonic and can each run behind -- the
// mirror after a frame was lost upstream (device credit-timeout drop), the extent when the drainer's
// head write-back lagged its snapshot, making the frame re-ship words the mirror already consumed -- so
// decode begins at the larger of the two: adopt-and-count on a loss, skip the overlap on a lag.
// The optional emit_zones8(lane, timer_hi, prog, w0s, w1s) sink receives EIGHT consecutive 2-word zone
// markers at once, deinterleaved into AVX2 vectors of word0s and word1s, whenever a 16-word block passes
// the all-zone type screen. emit_atomic8(lane, timer_hi, prog, w0s, w1s, w2s, n) is the same idea for up
// to EIGHT 3-word ZONE_ATOMIC records (w2s = durations): n <= 8 leading lanes are valid, the rest are
// speculative over-read and must not be emitted. Atomic zones are the dominant wire type, so without it
// the wire decodes essentially scalar. frame_words (linear path only) authorizes the 24-word block load
// to over-read past a lane's live window up to the frame end -- the FIFO always holds the whole page.
template <
    typename EmitMarker,
    typename EmitData,
    typename EmitProg = SpscIgnoreProg,
    typename EmitZones8 = SpscNoZones8,
    typename EmitAtomic8 = SpscNoAtomic8>
inline uint32_t spsc_decode_frame(
    SpanDecodeState& st,
    const uint32_t* frame,
    EmitMarker&& emit,
    EmitData&& emit_data,
    EmitProg&& emit_prog = SpscIgnoreProg{},
    EmitZones8&& emit_zones8 = SpscNoZones8{},
    EmitAtomic8&& emit_atomic8 = SpscNoAtomic8{},
    uint32_t frame_words = 0) {
    const uint32_t* ctrl = frame + kernel_profiler::SPSC_SPAN_PREFIX_WORDS;
    const auto xy_it = st.core_of_xy.find(ctrl[kernel_profiler::SPSC_CORE_XY]);
    if (xy_it == st.core_of_xy.end()) {
        st.unknown_core_frames++;
        return 0;
    }
    const uint32_t core = xy_it->second;
    const bool raw = (frame[0] & kernel_profiler::SPSC_SPAN_RAW_FLAG) != 0;
    uint32_t off = kernel_profiler::SPSC_SPAN_PREFIX_WORDS + kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE;
    for (uint32_t r = 0; r < kSpscNRiscDecode; r++) {
        const uint32_t lane = core * kSpscNRiscDecode + r;
        const uint32_t tail = ctrl[kernel_profiler::SPSC_RING_TAIL_0 + r];
        const uint32_t frame_head = ctrl[kernel_profiler::SPSC_RING_HEAD_0 + r];
        const uint32_t extent = kernel_profiler::spsc_span_live(frame_head, tail, kSpscRingCap);
        if (extent != tail - frame_head) {
            st.anomalies++;  // torn snapshot; the clamped geometry still frames consistently on both sides
        }
        const uint32_t start = tail - extent;
        const uint32_t* p = nullptr;
        if (!raw && extent != 0) {
            off += kernel_profiler::spsc_span_pack_pad(start, off);
            p = frame + off;
            off += extent;
        }
        uint32_t head;
        if (st.seeded[lane] == 0) {
            st.seeded[lane] = 1;
            head = start;
        } else {
            head = st.head[lane];
            const int32_t behind = static_cast<int32_t>(start - head);
            if (behind > 0) {
                st.resync_events++;
                st.resync_words += static_cast<uint32_t>(behind);
                head = start;
            } else if (behind < 0) {
                st.head_lag++;
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
        st.live_words += run;
        uint32_t th = st.timer_hi[lane];
        uint32_t pg = st.prog[lane];
        if (raw) {
            const uint32_t* ring = frame + kernel_profiler::SPSC_SPAN_PREFIX_WORDS +
                                   kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE + r * kSpscRingCap;
            const uint32_t hm = head & kSpscRingMask;
            for (uint32_t o = 0; o < run; o += 16) {
                spsc_prefetch(ring + ((hm + o) & kSpscRingMask));
            }
            uint32_t i = 0;
            while (i < run) {
#if defined(__AVX2__)
                if constexpr (!std::is_same_v<std::decay_t<EmitAtomic8>, SpscNoAtomic8>) {
                    const uint32_t idx = (hm + i) & kSpscRingMask;
                    if (i + 24 <= run && idx + 24 <= kSpscRingCap && pp_type(ring[idx]) == PP_ZONE_ATOMIC) {
                        __m256i w0s, w1s, w2s;
                        spsc_load_atomic8(ring + idx, w0s, w1s, w2s);
                        const __m256i types = _mm256_srli_epi32(w0s, PP_TYPE_SHIFT);
                        const uint32_t eq = static_cast<uint32_t>(
                            _mm256_movemask_epi8(_mm256_cmpeq_epi32(types, _mm256_set1_epi32(PP_ZONE_ATOMIC))));
                        const uint32_t n = std::countr_zero(~eq) / 4;
                        if (n != 0) {
                            emit_atomic8(lane, th, pg, w0s, w1s, w2s, n);
                            st.vec_atomic_recs += n;
                            i += 3 * n;
                            continue;
                        }
                        st.vec_block_rejects++;
                    }
                }
                if constexpr (!std::is_same_v<std::decay_t<EmitZones8>, SpscNoZones8>) {
                    const uint32_t idx = (hm + i) & kSpscRingMask;
                    if (i + 16 <= run && idx + 16 <= kSpscRingCap) {
                        const __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ring + idx));
                        const __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ring + idx + 8));
                        const __m256i even = _mm256_castps_si256(_mm256_shuffle_ps(
                            _mm256_castsi256_ps(v0), _mm256_castsi256_ps(v1), _MM_SHUFFLE(2, 0, 2, 0)));
                        const __m256i odd = _mm256_castps_si256(_mm256_shuffle_ps(
                            _mm256_castsi256_ps(v0), _mm256_castsi256_ps(v1), _MM_SHUFFLE(3, 1, 3, 1)));
                        const __m256i w0s = _mm256_permute4x64_epi64(even, _MM_SHUFFLE(3, 1, 2, 0));
                        const __m256i types = _mm256_srli_epi32(w0s, PP_TYPE_SHIFT);
                        if (_mm256_movemask_epi8(_mm256_cmpgt_epi32(types, _mm256_set1_epi32(1))) == 0) {
                            const __m256i w1s = _mm256_permute4x64_epi64(odd, _MM_SHUFFLE(3, 1, 2, 0));
                            emit_zones8(lane, th, pg, w0s, w1s);
                            st.vec_zone_recs += 8;
                            i += 16;
                            continue;
                        }
                        st.vec_block_rejects++;
                    }
                }
#endif
                const uint32_t w0 = ring[(hm + i) & kSpscRingMask];
                const uint32_t t = pp_type(w0);
                if (t == PP_ZONE_ATOMIC) {
                    if (i + 3 > run) {
                        st.anomalies++;
                        break;
                    }
                    const uint32_t w1 = ring[(hm + i + 1) & kSpscRingMask];
                    const uint32_t w2 = ring[(hm + i + 2) & kSpscRingMask];
                    emit(lane, t, pp_low27(w0), pp_full_ts(th, w1), w2, pg);  // ts = END, w2 = duration
                    st.scalar_recs++;
                    i += 3;
                } else if (t == PP_ZONE_START || t == PP_ZONE_END) {
                    if (i + 2 > run) {
                        st.anomalies++;
                        break;
                    }
                    const uint32_t w1 = ring[(hm + i + 1) & kSpscRingMask];
                    emit(lane, t, pp_low27(w0), pp_full_ts(th, w1), 0, pg);  // full 27-bit structural id
                    i += 2;
                } else if (t == PP_STICKY_TIMER) {
                    th = pp_timer_hi(w0);
                    i += 1;
                } else if (t == PP_STICKY_PROG) {
                    if (const uint32_t id = pp_low27(w0); id != pg) {
                        pg = id;
                        emit_prog(lane, pg);
                    }
                    i += 1;
                } else if (t == PP_STICKY_PROG_EXT) {
                    if (i + 2 > run) {
                        st.anomalies++;
                        break;
                    }
                    const uint32_t w1 = ring[(hm + i + 1) & kSpscRingMask];
                    if (w1 != pg) {
                        pg = w1;
                        emit_prog(lane, pg);
                    }
                    i += 2;
                } else if (t == PP_EVENT) {
                    // PP_EVENT: exactly 2 words -- a flag with a compile-time structural id, no payload.
                    if (i + 2 > run) {
                        st.anomalies++;
                        break;
                    }
                    const uint32_t w1 = ring[(hm + i + 1) & kSpscRingMask];
                    emit_data(lane, PP_EVENT, pp_point_id(w0), pp_full_ts(th, w1), pg, nullptr, 0);
                    i += 2;
                } else if (t == PP_DATA) {
                    // PP_DATA is 3 + size words and the length lives in word2, so the whole header must
                    // be inside the run before the packet can be sized at all.
                    if (i + 3 > run) {
                        st.anomalies++;
                        break;
                    }
                    const uint32_t n = pp_data_size(ring[(hm + i + 2) & kSpscRingMask]);
                    if (i + 3 + n > run) {
                        st.anomalies++;
                        break;
                    }
                    const uint32_t w1 = ring[(hm + i + 1) & kSpscRingMask];
                    // The payload can wrap the circular ring, so unwrap it into a flat scratch first.
                    uint32_t payload[kSpscMaxDataWords];
                    for (uint32_t k = 0; k < n; k++) {
                        payload[k] = ring[(hm + i + 3 + k) & kSpscRingMask];
                    }
                    emit_data(lane, PP_DATA, pp_point_id(w0), pp_full_ts(th, w1), pg, payload, n);
                    i += 3 + n;
                } else {
                    st.anomalies++;
                    break;
                }
            }
            st.timer_hi[lane] = th;
            st.prog[lane] = pg;
            continue;
        }
        p += extent - run;
        // Just-in-time prefetch of this lane's live window: small bursts consumed immediately fit the
        // core's fill-buffer budget -- issuing whole frames ahead was measured 30-40% SLOWER (the bulk
        // cold-line prefetches starve the walk's own demand loads).
        for (uint32_t o = 0; o < run; o += 16) {
            spsc_prefetch(p + o);
        }
        uint32_t i = 0;
        while (i < run) {
#if defined(__AVX2__)
            if constexpr (!std::is_same_v<std::decay_t<EmitAtomic8>, SpscNoAtomic8>) {
                const bool loadable = i + 24 <= run ||
                                      (frame_words != 0 &&
                                       static_cast<uint32_t>(p - frame) + i + 24 <= frame_words &&
                                       i + 3 <= run);
                if (loadable && pp_type(p[i]) == PP_ZONE_ATOMIC) {
                    __m256i w0s, w1s, w2s;
                    spsc_load_atomic8(p + i, w0s, w1s, w2s);
                    const __m256i types = _mm256_srli_epi32(w0s, PP_TYPE_SHIFT);
                    const uint32_t eq = static_cast<uint32_t>(
                        _mm256_movemask_epi8(_mm256_cmpeq_epi32(types, _mm256_set1_epi32(PP_ZONE_ATOMIC))));
                    uint32_t n = std::countr_zero(~eq) / 4;
                    const uint32_t avail = (run - i) / 3;
                    if (n > avail) {
                        n = avail;
                    }
                    if (n != 0) {
                        emit_atomic8(lane, th, pg, w0s, w1s, w2s, n);
                        st.vec_atomic_recs += n;
                        i += 3 * n;
                        continue;
                    }
                    st.vec_block_rejects++;
                }
            }
            if constexpr (!std::is_same_v<std::decay_t<EmitZones8>, SpscNoZones8>) {
                if (i + 16 <= run) {
                    const __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i));
                    const __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i + 8));
                    const __m256i even = _mm256_castps_si256(
                        _mm256_shuffle_ps(_mm256_castsi256_ps(v0), _mm256_castsi256_ps(v1), _MM_SHUFFLE(2, 0, 2, 0)));
                    const __m256i odd = _mm256_castps_si256(
                        _mm256_shuffle_ps(_mm256_castsi256_ps(v0), _mm256_castsi256_ps(v1), _MM_SHUFFLE(3, 1, 3, 1)));
                    const __m256i w0s = _mm256_permute4x64_epi64(even, _MM_SHUFFLE(3, 1, 2, 0));
                    const __m256i types = _mm256_srli_epi32(w0s, PP_TYPE_SHIFT);
                    if (_mm256_movemask_epi8(_mm256_cmpgt_epi32(types, _mm256_set1_epi32(1))) == 0) {
                        const __m256i w1s = _mm256_permute4x64_epi64(odd, _MM_SHUFFLE(3, 1, 2, 0));
                        emit_zones8(lane, th, pg, w0s, w1s);
                        st.vec_zone_recs += 8;
                        i += 16;
                        continue;
                    }
                    st.vec_block_rejects++;
                }
            }
#endif
            const uint32_t w0 = p[i];
            const uint32_t t = pp_type(w0);
            if (t == PP_ZONE_ATOMIC) {
                if (i + 3 > run) {
                    st.anomalies++;
                    break;
                }
                emit(lane, t, pp_low27(w0), pp_full_ts(th, p[i + 1]), p[i + 2], pg);  // ts = END, [2] = duration
                st.scalar_recs++;
                i += 3;
            } else if (t == PP_ZONE_START || t == PP_ZONE_END) {
                if (i + 2 > run) {
                    st.anomalies++;
                    break;
                }
                const uint32_t w1 = p[i + 1];
                emit(lane, t, pp_low27(w0), pp_full_ts(th, w1), 0, pg);  // full 27-bit structural id
                i += 2;
            } else if (t == PP_STICKY_TIMER) {
                th = pp_timer_hi(w0);
                i += 1;
            } else if (t == PP_STICKY_PROG) {
                if (const uint32_t id = pp_low27(w0); id != pg) {
                    pg = id;
                    emit_prog(lane, pg);
                }
                i += 1;
            } else if (t == PP_STICKY_PROG_EXT) {
                if (i + 2 > run) {
                    st.anomalies++;
                    break;
                }
                const uint32_t w1 = p[i + 1];
                if (w1 != pg) {
                    pg = w1;
                    emit_prog(lane, pg);
                }
                i += 2;
            } else if (t == PP_EVENT) {
                // PP_EVENT: exactly 2 words -- a flag with a compile-time structural id, no payload.
                if (i + 2 > run) {
                    st.anomalies++;
                    break;
                }
                emit_data(lane, PP_EVENT, pp_point_id(w0), pp_full_ts(th, p[i + 1]), pg, nullptr, 0);
                i += 2;
            } else if (t == PP_DATA) {
                // PP_DATA is 3 + size words with the length in word2; the packed window is flat, so the
                // payload is handed to the sink in place.
                if (i + 3 > run) {
                    st.anomalies++;
                    break;
                }
                const uint32_t n = pp_data_size(p[i + 2]);
                if (i + 3 + n > run) {
                    st.anomalies++;
                    break;
                }
                emit_data(lane, PP_DATA, pp_point_id(w0), pp_full_ts(th, p[i + 1]), pg, p + i + 3, n);
                i += 3 + n;
            } else {
                st.anomalies++;
                break;
            }
        }
        st.timer_hi[lane] = th;
        st.prog[lane] = pg;
    }
    if (raw) {
        return kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE + kSpscNRiscDecode * kSpscRingCap;
    }
    return off - kernel_profiler::SPSC_SPAN_PREFIX_WORDS;
}

}  // namespace tt::tt_metal::profiler
