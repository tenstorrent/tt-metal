// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// SINGLE SOURCE OF TRUTH for the host-side decode of the DRISC drainer's wire
// (producer: tt_metal/tools/profiler/kernels/drisc_profiler_drain.cpp).
//
// The wire carries only whole fixed-size BULK_SPAN frames: a 16-word prefix, the worker's 64-word
// control vector verbatim, and its five raw 512-word rings. Inside each ring is a variable-length
// packet run (spsc_packet.h): ZONE_START/END/TOTAL markers (2 words), STICKY_TIMER (1 word, per-lane
// wall-clock high half), STICKY_PROG (2 words, per-core runtime host-id), and DATA/EVENT (2 + size
// words, self-describing). The producer publishes its tail only on packet boundaries, so a run never
// ends mid-packet.
#pragma once

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
// RING_CAPACITY, = kernel_profiler::PROFILER_L1_VECTOR_SIZE) so the BULKCORE sub-ring walk indexes
// correctly.
inline constexpr uint32_t kSpscRingCap = 512;
inline constexpr uint32_t kSpscRingMask = kSpscRingCap - 1;
inline constexpr uint32_t kSpscNRiscDecode = 5;
static_assert((kSpscRingCap & kSpscRingMask) == 0);

// Largest PP_DATA payload the 7-bit size field can express; bounds the unwrap scratch.
inline constexpr uint32_t kSpscMaxDataWords = 127;

inline constexpr uint32_t kSpscFrameWords = kernel_profiler::SPSC_SPAN_PREFIX_WORDS +
                                            kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE +
                                            kSpscNRiscDecode * kSpscRingCap;
inline constexpr uint32_t kSpscFramePages = kSpscFrameWords * 4 / (kernel_profiler::SPSC_SPAN_PAGE_WORDS * 4);
static_assert(kSpscFrameWords == 2640 && kSpscFramePages == 165);

// Decode state for one socket's frame stream. Written only by that socket's decode thread.
struct SpanDecodeState {
    std::vector<uint32_t> timer_hi;  // per lane: sticky wall-clock high half
    std::vector<uint32_t> prog;      // per core: sticky runtime host-id (each core's BRISC emits its own)
    std::vector<uint32_t> head;      // per lane: monotonic words-consumed mirror; head(N) == tail(N-1)
    std::vector<uint8_t> seeded;
    std::unordered_map<uint32_t, uint32_t> core_of_xy;  // packed (y<<16)|x -> dense core index
    uint64_t live_words = 0;
    uint64_t resync_events = 0;
    uint64_t resync_words = 0;
    uint64_t head_lag = 0;
    uint64_t anomalies = 0;  // torn run / truncated run / undecodable word
    uint64_t unknown_core_frames = 0;

    void reset(uint32_t num_cores) {
        timer_hi.assign(static_cast<size_t>(num_cores) * kSpscNRiscDecode, 0);
        prog.assign(num_cores, 0);
        head.assign(static_cast<size_t>(num_cores) * kSpscNRiscDecode, 0);
        seeded.assign(static_cast<size_t>(num_cores) * kSpscNRiscDecode, 0);
    }
};

struct SpscIgnoreProg {
    void operator()(uint32_t /*core*/, uint32_t /*prog*/) const {}
};

// Sentinel default for the vectorized zone-block sink: without a real one the walk stays scalar.
struct SpscNoZones8 {};

inline void spsc_prefetch(const void* p) {
#if defined(__x86_64__)
    _mm_prefetch(static_cast<const char*>(p), _MM_HINT_T0);
#else
    (void)p;
#endif
}

// Decode ONE whole BULK_SPAN frame in place. For each marker calls
//   emit(lane, wire_type, hash16, full_ts, prog)     (ZONE_START/END; ZONE_TOTAL with full_ts = the sum)
// and for each PP_DATA/PP_EVENT
//   emit_data(lane, wire_type, id, full_ts, prog, payload_words, n)   (payload unwrapped, hi-word first)
// and emit_prog(core, prog) whenever a core's sticky host-id changes.
//
// Head adoption: the mirror and the frame's own head field are both monotonic and can each run behind --
// the mirror after a frame was lost upstream (device credit-timeout drop), the frame field when the
// drainer's write-back lagged the snapshot -- so the larger one is always the truth. Adopting it makes an
// upstream loss a counted resync instead of re-decoding overwritten ring words as markers.
// The optional emit_zones8(lane, timer_hi, prog, w0s, w1s) sink receives EIGHT consecutive 2-word zone
// markers at once, deinterleaved into AVX2 vectors of word0s and word1s, whenever a 16-word block passes
// the all-zone type screen -- the dominant case in a full span, and where the scalar walk's per-record
// cost lives.
template <
    typename EmitMarker,
    typename EmitData,
    typename EmitProg = SpscIgnoreProg,
    typename EmitZones8 = SpscNoZones8>
inline void spsc_decode_frame(
    SpanDecodeState& st,
    const uint32_t* frame,
    EmitMarker&& emit,
    EmitData&& emit_data,
    EmitProg&& emit_prog = SpscIgnoreProg{},
    EmitZones8&& emit_zones8 = SpscNoZones8{}) {
    const uint32_t* ctrl = frame + kernel_profiler::SPSC_SPAN_PREFIX_WORDS;
    const auto xy_it = st.core_of_xy.find(ctrl[kernel_profiler::SPSC_CORE_XY]);
    if (xy_it == st.core_of_xy.end()) {
        st.unknown_core_frames++;
        return;
    }
    const uint32_t core = xy_it->second;
    uint32_t pg = st.prog[core];
    const uint32_t* ring = ctrl + kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE;
    for (uint32_t r = 0; r < kSpscNRiscDecode; r++, ring += kSpscRingCap) {
        const uint32_t lane = core * kSpscNRiscDecode + r;
        const uint32_t tail = ctrl[kernel_profiler::SPSC_RING_TAIL_0 + r];
        const uint32_t frame_head = ctrl[kernel_profiler::SPSC_RING_HEAD_0 + r];
        uint32_t head;
        if (st.seeded[lane] == 0) {
            st.seeded[lane] = 1;
            head = frame_head;
        } else {
            head = st.head[lane];
            const int32_t behind = static_cast<int32_t>(frame_head - head);
            if (behind > 0) {
                st.resync_events++;
                st.resync_words += static_cast<uint32_t>(behind);
                head = frame_head;
            } else if (behind < 0) {
                st.head_lag++;
            }
        }
        uint32_t run = tail - head;
        if (run > kSpscRingCap) {
            st.anomalies++;
            head = tail - kSpscRingCap;  // only the newest ring-full of words still exists
            run = kSpscRingCap;
        }
        st.head[lane] = head + run;
        if (run == 0) {
            continue;
        }
        st.live_words += run;
        const uint32_t hm = head & kSpscRingMask;
        // The frame is DMA-fresh host memory the walk is about to miss on line by line; fetching the whole
        // live window up front overlaps those misses with the walk instead of serializing on them.
        for (uint32_t off = 0; off < run; off += 16) {
            spsc_prefetch(ring + ((hm + off) & kSpscRingMask));
        }
        uint32_t th = st.timer_hi[lane];
        uint32_t i = 0;
        while (i < run) {
#if defined(__AVX2__)
            if constexpr (!std::is_same_v<std::decay_t<EmitZones8>, SpscNoZones8>) {
                const uint32_t idx = (hm + i) & kSpscRingMask;
                if (i + 16 <= run && idx + 16 <= kSpscRingCap) {
                    const __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ring + idx));
                    const __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ring + idx + 8));
                    const __m256i even = _mm256_castps_si256(
                        _mm256_shuffle_ps(_mm256_castsi256_ps(v0), _mm256_castsi256_ps(v1), _MM_SHUFFLE(2, 0, 2, 0)));
                    const __m256i odd = _mm256_castps_si256(
                        _mm256_shuffle_ps(_mm256_castsi256_ps(v0), _mm256_castsi256_ps(v1), _MM_SHUFFLE(3, 1, 3, 1)));
                    const __m256i w0s = _mm256_permute4x64_epi64(even, _MM_SHUFFLE(3, 1, 2, 0));
                    const __m256i w1s = _mm256_permute4x64_epi64(odd, _MM_SHUFFLE(3, 1, 2, 0));
                    const __m256i types = _mm256_srli_epi32(w0s, PP_TYPE_SHIFT);
                    if (_mm256_movemask_epi8(_mm256_cmpgt_epi32(types, _mm256_set1_epi32(1))) == 0) {
                        emit_zones8(lane, th, pg, w0s, w1s);
                        i += 16;
                        continue;
                    }
                }
            }
#endif
            const uint32_t w0 = ring[(hm + i) & kSpscRingMask];
            const uint32_t t = pp_type(w0);
            if (t == PP_ZONE_START || t == PP_ZONE_END || t == PP_ZONE_TOTAL) {
                if (i + 2 > run) {
                    st.anomalies++;
                    break;
                }
                const uint32_t w1 = ring[(hm + i + 1) & kSpscRingMask];
                const uint64_t ts = (t == PP_ZONE_TOTAL) ? w1 : pp_full_ts(th, w1);
                emit(lane, t, pp_low27(w0) & 0xFFFFu, ts, pg);
                i += 2;
            } else if (t == PP_STICKY_TIMER) {
                th = pp_timer_hi(w0);
                i += 1;
            } else if (t == PP_STICKY_PROG) {
                if (i + 2 > run) {
                    st.anomalies++;
                    break;
                }
                const uint32_t w1 = ring[(hm + i + 1) & kSpscRingMask];
                if (w1 != pg) {
                    pg = w1;
                    emit_prog(core, pg);
                }
                i += 2;
            } else if (t == PP_DATA || t == PP_EVENT) {
                const uint32_t n = pp_data_size(w0);
                if (i + 2 + n > run) {
                    st.anomalies++;
                    break;
                }
                const uint32_t w1 = ring[(hm + i + 1) & kSpscRingMask];
                uint32_t payload[kSpscMaxDataWords];
                for (uint32_t k = 0; k < n; k++) {
                    payload[k] = ring[(hm + i + 2 + k) & kSpscRingMask];
                }
                emit_data(lane, t, pp_data_id(w0), pp_full_ts(th, w1), pg, payload, n);
                i += 2 + n;
            } else {
                st.anomalies++;
                break;
            }
        }
        st.timer_hi[lane] = th;
    }
    st.prog[core] = pg;
}

}  // namespace tt::tt_metal::profiler
