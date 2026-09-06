// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Host-side decode of the relay's wire, and its only definition. A frame is a 16-word prefix (word 1 = payload
// length), the SPSC_SPAN_WIRE_CTRL_WORDS control block, then each RISC's live ring window packed flat with congruence
// pads and wraps resolved device-side. Packet formats: spsc_packet.h. The producer publishes its tail only on
// packet boundaries, so a window never ends mid-packet.
#pragma once

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

// Worst case: five full rings behind maximal pads. Bounds the bounce buffer and frame validation, not any
// device layout.
inline constexpr uint32_t kSpscMaxPayloadWords =
    kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE +
    kSpscNRiscDecode * (kSpscRingCap + kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS - 1);
inline constexpr uint32_t kSpscMaxFrameWords = kernel_profiler::spsc_span_frame_words(kSpscMaxPayloadWords);
inline constexpr uint32_t kSpscMaxFramePages = kSpscMaxFrameWords / kernel_profiler::SPSC_SPAN_PAGE_WORDS;
static_assert(kSpscMaxFrameWords == 2656 && kSpscMaxFramePages == 166);

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
    std::unordered_map<uint32_t, uint32_t> core_of_xy;  // packed (y<<16)|x -> dense core index
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

// Attributed rather than -march: raising the build baseline changes codegen everywhere; clang inlines only
// across matching target attributes, so every function in the block carries it, and callers pass plain
// arguments because they cannot form a __m512i. Shuffle operands are file-scope data so the block loads them
// instead of rebuilding vectors per call.
// ZONE_ATOMIC operands: word0 (type|id27), word1 (end low) and word2 (duration) of sixteen 3-word records
// gathered from the three loaded vectors; indices 0-31 come from (v0, v1), the masked tail lanes from v2.
alignas(64) inline constexpr uint32_t kA16W0[16] = {0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 1, 4, 7, 10, 13};
alignas(64) inline constexpr uint32_t kA16W1[16] = {1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 2, 5, 8, 11, 14};
alignas(64) inline constexpr uint32_t kA16W2[16] = {2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 3, 6, 9, 12, 15};
inline constexpr __mmask16 kA16FromV2W0 = 0xF800, kA16FromV2W1 = 0xF800, kA16FromV2W2 = 0xFC00;

// ZONE_S operands: even/odd deinterleave of a 16-record (32-word) load pair into w0s/w1s.
alignas(64) inline constexpr uint32_t kZS16Even[16] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
alignas(64) inline constexpr uint32_t kZS16Odd[16] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};

// The build baseline is x86-64-v3, so AVX-512 exists only inside attributed functions and every call into one
// sits behind this check; a host without it runs the AVX2/scalar tier instead of faulting.
inline bool spsc_host_avx512() {
    static const bool v = [] {
        return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
               __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq");
    }();
    return v;
}

// Every record a consumer sees is the public 32 B Rec {start|ts, duration, meta<<32 | id, prog}; blocks compose
// them as whole 64 B lines straight into the Sink's buffer. Stores are cached, not NT: the consumer re-reads the
// scratch immediately. The audit sink stores nothing (Sink::kStores). A partial block still writes its full
// half (4 lines), so the buffer needs kSpscSinkSlackRecs of slack past cap.
inline constexpr uint32_t kSpscRecBytes = 32;
inline constexpr uint32_t kSpscSinkSlackRecs = 8;
inline constexpr uint32_t kSpscRecTypeZone = 1;  // RecType::Zone, pinned by the receiver's layout probe
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

#pragma clang attribute push(__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq"))), apply_to = function)

// Eight zone records {start, duration, meta|id, prog} as four lines stored straight to dst: lanes 0,1,4,5 of
// each line come from (start, duration), lanes 2,3,6,7 from (meta|id, prog); one index vector serves both
// permutes and advances by two records per line. Always writes all four lines (the sink's slack covers a
// partial block), so there is no per-line count to compute.
inline void spsc_zone_lines8(__m512i s64, __m512i d64, __m512i m64, __m512i pv, uint8_t* dst) {
    __m512i idx = _mm512_set_epi64(9, 1, 9, 1, 8, 0, 8, 0);
    const __m512i inc = _mm512_set1_epi64(2);
    for (int j = 0; j < 4; j++) {
        const __m512i sd = _mm512_permutex2var_epi64(s64, idx, d64);
        const __m512i mp = _mm512_permutex2var_epi64(m64, idx, pv);
        _mm512_storeu_si512(dst + 64 * j, _mm512_mask_blend_epi64(0xCC, sd, mp));
        idx = _mm512_add_epi64(idx, inc);
    }
}

struct SpscA16Result {
    uint32_t n;
    uint64_t ts_first;
    uint64_t ts_last;
    bool near_wrap;  // some end in the block lies within kLatchWindow of a low-word wrap
};

// A wall-clock read whose low word is this close below a wrap may carry the next epoch's high word (the
// device-side latch race, kernel_profiler_streaming.hpp read_wall_clock); the decoder repairs it when the lane's
// next timestamp regresses. The gap is a few cycles; 1024 keeps the false-positive odds at ~2e-7 per regression.
inline constexpr uint32_t kLatchWindow = 0xFFFFFC00u;

// Any n from 1 to 16: each output line comes straight off the wire through one two-source permute, with
// th/meta/prog on the constant operand.
template <typename Sink>
inline SpscA16Result spsc_atomic16_avx512(
    const uint32_t* p,
    uint32_t avail,
    uint32_t max_recs,
    uint32_t th,
    uint32_t prog,
    uint32_t lane,
    uint32_t dev,
    Sink& sw) {
    SpscA16Result out{0, 0, 0, false};
    __m512i v0, v1, v2;
    if (avail >= 48u) {  // frame interior: no mask math
        v0 = _mm512_loadu_si512(p);
        v1 = _mm512_loadu_si512(p + 16);
        v2 = _mm512_loadu_si512(p + 32);
    } else {
        const __mmask16 m0 = static_cast<__mmask16>(avail >= 16u ? 0xFFFFu : ((1u << avail) - 1u));
        const __mmask16 m1 =
            static_cast<__mmask16>(avail <= 16u ? 0u : (avail >= 32u ? 0xFFFFu : ((1u << (avail - 16u)) - 1u)));
        const __mmask16 m2 = static_cast<__mmask16>(avail <= 32u ? 0u : ((1u << (avail - 32u)) - 1u));
        v0 = _mm512_maskz_loadu_epi32(m0, p);
        v1 = _mm512_maskz_loadu_epi32(m1, p + 16);
        v2 = _mm512_maskz_loadu_epi32(m2, p + 32);
    }
    const __m512i atype = _mm512_set1_epi32(PP_ZONE_ATOMIC);
    const uint64_t k0 = _mm512_cmpeq_epi32_mask(_mm512_srli_epi32(v0, PP_TYPE_SHIFT), atype);
    const uint64_t k1 = _mm512_cmpeq_epi32_mask(_mm512_srli_epi32(v1, PP_TYPE_SHIFT), atype);
    const uint64_t k2 = _mm512_cmpeq_epi32_mask(_mm512_srli_epi32(v2, PP_TYPE_SHIFT), atype);
    // Only every third bit (bit 3r = record r's w0) is meaningful: any ts/dur word can match the type pattern.
    // Bit 48 terminates an all-hit scan.
    constexpr uint64_t kW0Bits = 0x249249249249ull;
    const uint64_t miss = (~(k0 | (k1 << 16) | (k2 << 32)) & kW0Bits) | (1ull << 48);
    uint32_t n = static_cast<uint32_t>(std::countr_zero(miss)) / 3u;
    if (n > max_recs) {
        n = max_recs;
    }
    if (n == 0) {
        return out;
    }
    const uint64_t th_hi = static_cast<uint64_t>(th) << 32;
    out.n = n;
    // Scalar reloads of L1-hot source lines: no store to forward from, no shuffle-port contention.
    out.ts_first = th_hi | p[1];
    out.ts_last = th_hi | p[3u * n - 2u];
    const __m512i lw = _mm512_set1_epi32(static_cast<int>(kLatchWindow));
    const uint64_t near = static_cast<uint64_t>(_mm512_cmpge_epu32_mask(v0, lw)) |
                          (static_cast<uint64_t>(_mm512_cmpge_epu32_mask(v1, lw)) << 16) |
                          (static_cast<uint64_t>(_mm512_cmpge_epu32_mask(v2, lw)) << 32);
    constexpr uint64_t kEndBits = 0x492492492492ull;  // bit 3r+1 = record r's end word
    out.near_wrap = (near & kEndBits & ((1ull << (3u * n)) - 1u)) != 0;
    if constexpr (Sink::kStores) {
        auto gather = [&](const uint32_t* idx, __mmask16 from_v2) {
            const __m512i iv = _mm512_load_si512(idx);
            return _mm512_mask_permutexvar_epi32(_mm512_permutex2var_epi32(v0, iv, v1), from_v2, iv, v2);
        };
        const __m512i ids = _mm512_and_si512(gather(kA16W0, kA16FromV2W0), _mm512_set1_epi32(0x07FFFFFF));
        const __m512i ends = gather(kA16W1, kA16FromV2W1);
        const __m512i durs = gather(kA16W2, kA16FromV2W2);
        const uint64_t meta64 = static_cast<uint64_t>((lane << 16) | (dev << 26) | (kSpscRecTypeZone << 29)) << 32;
        const __m512i mv = _mm512_set1_epi64(static_cast<long long>(meta64));
        const __m512i pv = _mm512_set1_epi64(prog);
        const __m512i tv = _mm512_set1_epi64(static_cast<long long>(th_hi));
        uint8_t* const dst = sw.buf + sw.off;
        const auto half = [&](__m256i end_h, __m256i dur_h, __m256i id_h, uint8_t* o) {
            const __m512i d64 = _mm512_cvtepu32_epi64(dur_h);
            const __m512i s64 = _mm512_sub_epi64(_mm512_or_si512(tv, _mm512_cvtepu32_epi64(end_h)), d64);
            spsc_zone_lines8(s64, d64, _mm512_or_si512(mv, _mm512_cvtepu32_epi64(id_h)), pv, o);
        };
        half(_mm512_castsi512_si256(ends), _mm512_castsi512_si256(durs), _mm512_castsi512_si256(ids), dst);
        if (n > 8) {
            half(
                _mm512_extracti64x4_epi64(ends, 1),
                _mm512_extracti64x4_epi64(durs, 1),
                _mm512_extracti64x4_epi64(ids, 1),
                dst + 256);
        }
        sw.off += kSpscRecBytes * n;
    } else {
        (void)lane;
        (void)dev;
        (void)prog;
    }
    return out;
}

struct SpscZoneS16Result {
    uint32_t n;
    uint64_t ts_first;
    uint64_t ts_last;  // the lane cursor after the block
};

// ZONE_S counterpart of the atomic block: a ZONE_S end is cursor-relative, an inclusive prefix sum (four
// shifted adds for 16 lanes), and records normalize to ZONE_ATOMIC form so downstream never sees wire size
// classes. Consumes every consecutive full block in one call, so a dense lane pays the walk's per-call cost
// once per run rather than once per 16 records. `readable` authorizes loads, never emits, past the live run;
// `max_recs` bounds the emits.
template <typename Sink>
inline SpscZoneS16Result spsc_zone_s16_avx512(
    const uint32_t* p,
    uint32_t readable,
    uint32_t max_recs,
    uint64_t cursor,
    uint32_t prog,
    uint32_t lane,
    uint32_t dev,
    Sink& sw) {
    SpscZoneS16Result out{0, 0, cursor};
    const __m512i stype = _mm512_set1_epi32(PP_ZONE_S);
    const __m512i odd = _mm512_load_si512(kZS16Odd);
    const __m512i z = _mm512_setzero_si512();
    [[maybe_unused]] const __m512i even = _mm512_load_si512(kZS16Even);
    [[maybe_unused]] const __m512i mv = _mm512_set1_epi64(
        static_cast<long long>(static_cast<uint64_t>((lane << 16) | (dev << 26) | (kSpscRecTypeZone << 29)) << 32));
    [[maybe_unused]] const __m512i pv = _mm512_set1_epi64(prog);
    while (max_recs != 0 && readable >= 2u) {
        __m512i v0, v1;
        if (readable >= 32u) {
            v0 = _mm512_loadu_si512(p);
            v1 = _mm512_loadu_si512(p + 16);
        } else {
            const __mmask16 m0 = static_cast<__mmask16>(readable >= 16u ? 0xFFFFu : ((1u << readable) - 1u));
            const __mmask16 m1 = static_cast<__mmask16>(readable <= 16u ? 0u : ((1u << (readable - 16u)) - 1u));
            v0 = _mm512_maskz_loadu_epi32(m0, p);
            v1 = _mm512_maskz_loadu_epi32(m1, p + 16);
        }
        const uint64_t k0 = _mm512_cmpeq_epi32_mask(_mm512_srli_epi32(v0, PP_TYPE_SHIFT), stype);
        const uint64_t k1 = _mm512_cmpeq_epi32_mask(_mm512_srli_epi32(v1, PP_TYPE_SHIFT), stype);
        // Only even bits (bit 2r = record r's w0) are meaningful; bit 32 terminates an all-hit scan.
        constexpr uint64_t kW0Bits = 0x55555555ull;
        const uint64_t miss = (~(k0 | (k1 << 16)) & kW0Bits) | (1ull << 32);
        uint32_t n = static_cast<uint32_t>(std::countr_zero(miss)) / 2u;
        if (n > max_recs) {
            n = max_recs;
        }
        if (n == 0) {
            break;
        }
        const __m512i w1s = _mm512_permutex2var_epi32(v0, odd, v1);
        __m512i pfx = _mm512_srli_epi32(w1s, 16);
        pfx = _mm512_add_epi32(pfx, _mm512_alignr_epi32(pfx, z, 15));
        pfx = _mm512_add_epi32(pfx, _mm512_alignr_epi32(pfx, z, 14));
        pfx = _mm512_add_epi32(pfx, _mm512_alignr_epi32(pfx, z, 12));
        pfx = _mm512_add_epi32(pfx, _mm512_alignr_epi32(pfx, z, 8));
        if (out.n == 0) {
            out.ts_first = cursor + (p[1] >> 16);
        }
        // Spilled, not lane-extracted: n-1 is runtime, and an aligned store to hot stack beats a variable-lane
        // compress.
        alignas(64) uint32_t pfx_arr[16];
        _mm512_store_si512(pfx_arr, pfx);
        if constexpr (Sink::kStores) {
            const __m512i w0s = _mm512_permutex2var_epi32(v0, even, v1);
            const __m512i durs = _mm512_and_si512(w1s, _mm512_set1_epi32(0xFFFF));
            const __m512i ids = _mm512_and_si512(w0s, _mm512_set1_epi32(0x07FFFFFF));
            const __m512i cv = _mm512_set1_epi64(static_cast<long long>(cursor));
            uint8_t* const dst = sw.buf + sw.off;
            const auto half = [&](__m256i pfx_h, __m256i dur_h, __m256i id_h, uint8_t* o) {
                const __m512i d64 = _mm512_cvtepu32_epi64(dur_h);
                const __m512i s64 = _mm512_sub_epi64(_mm512_add_epi64(cv, _mm512_cvtepu32_epi64(pfx_h)), d64);
                spsc_zone_lines8(s64, d64, _mm512_or_si512(mv, _mm512_cvtepu32_epi64(id_h)), pv, o);
            };
            half(_mm512_castsi512_si256(pfx), _mm512_castsi512_si256(durs), _mm512_castsi512_si256(ids), dst);
            if (n > 8) {
                half(
                    _mm512_extracti64x4_epi64(pfx, 1),
                    _mm512_extracti64x4_epi64(durs, 1),
                    _mm512_extracti64x4_epi64(ids, 1),
                    dst + 256);
            }
            sw.off += kSpscRecBytes * n;
        }
        cursor += pfx_arr[n - 1];
        out.n += n;
        out.ts_last = cursor;
        if (n < 16u) {
            break;  // the block ended on a non-S word or the emit budget
        }
        p += 32;
        readable -= 32;
        max_recs -= 16;
    }
    return out;
}

#pragma clang attribute pop

// Inclusive prefix sum of eight 32-bit lanes: two in-lane shifted adds, then the low lane's total carried into the
// high lane.
inline __m256i spsc_prefix8_avx2(__m256i x) {
    x = _mm256_add_epi32(x, _mm256_slli_si256(x, 4));
    x = _mm256_add_epi32(x, _mm256_slli_si256(x, 8));
    return _mm256_add_epi32(x, _mm256_shuffle_epi32(_mm256_permute2x128_si256(x, x, 0x08), 0xFF));
}

// Four zone records {start, duration, meta|id, prog} as four 32 B stores: 64-bit unpacks pair (start, duration) and
// (meta|id, prog), and each record's two pairs meet across the 128-bit lanes.
inline void spsc_zone_lines4_avx2(__m256i s64, __m256i d64, __m256i m64, __m256i pv, uint8_t* dst) {
    const __m256i sd_lo = _mm256_unpacklo_epi64(s64, d64);
    const __m256i sd_hi = _mm256_unpackhi_epi64(s64, d64);
    const __m256i mp_lo = _mm256_unpacklo_epi64(m64, pv);
    const __m256i mp_hi = _mm256_unpackhi_epi64(m64, pv);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), _mm256_permute2x128_si256(sd_lo, mp_lo, 0x20));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 32), _mm256_permute2x128_si256(sd_hi, mp_hi, 0x20));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 64), _mm256_permute2x128_si256(sd_lo, mp_lo, 0x31));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 96), _mm256_permute2x128_si256(sd_hi, mp_hi, 0x31));
}

// AVX2 tier of spsc_atomic16_avx512: any n from 1 to 8 out of three 256-bit loads. A field's every-third words
// gather with one permute per load and two blends; the permute indices name each output lane's source element.
template <typename Sink>
inline SpscA16Result spsc_atomic8_avx2(
    const uint32_t* p,
    uint32_t avail,
    uint32_t max_recs,
    uint32_t th,
    uint32_t prog,
    uint32_t lane,
    uint32_t dev,
    Sink& sw) {
    SpscA16Result out{0, 0, 0, false};
    __m256i v[3];
    if (avail >= 24u) {
        for (int i = 0; i < 3; i++) {
            v[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 8 * i));
        }
    } else {
        const __m256i lane_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        for (int i = 0; i < 3; i++) {
            const __m256i m = _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(avail) - 8 * i), lane_idx);
            v[i] = _mm256_maskload_epi32(reinterpret_cast<const int*>(p + 8 * i), m);
        }
    }
    const __m256i atype = _mm256_set1_epi32(PP_ZONE_ATOMIC);
    const __m256i lw = _mm256_set1_epi32(static_cast<int>(kLatchWindow));
    uint32_t hit = 0, near = 0;
    for (int i = 0; i < 3; i++) {
        hit |= static_cast<uint32_t>(_mm256_movemask_ps(
                   _mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_srli_epi32(v[i], PP_TYPE_SHIFT), atype))))
               << (8 * i);
        near |= static_cast<uint32_t>(
                    _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_max_epu32(v[i], lw), v[i]))))
                << (8 * i);
    }
    // Only every third bit (bit 3r = record r's w0) is meaningful: any ts/dur word can match the type pattern.
    // Bit 24 terminates an all-hit scan.
    constexpr uint32_t kW0Bits = 0x249249u;
    const uint32_t miss = (~hit & kW0Bits) | (1u << 24);
    uint32_t n = static_cast<uint32_t>(std::countr_zero(miss)) / 3u;
    if (n > max_recs) {
        n = max_recs;
    }
    if (n == 0) {
        return out;
    }
    const uint64_t th_hi = static_cast<uint64_t>(th) << 32;
    out.n = n;
    out.ts_first = th_hi | p[1];
    out.ts_last = th_hi | p[3u * n - 2u];
    constexpr uint32_t kEndBits = 0x492492u;  // bit 3r+1 = record r's end word
    out.near_wrap = (near & kEndBits & ((1u << (3u * n)) - 1u)) != 0;
    if constexpr (Sink::kStores) {
        const __m256i ids = _mm256_and_si256(
            _mm256_blend_epi32(
                _mm256_blend_epi32(
                    _mm256_permutevar8x32_epi32(v[0], _mm256_setr_epi32(0, 3, 6, 0, 0, 0, 0, 0)),
                    _mm256_permutevar8x32_epi32(v[1], _mm256_setr_epi32(0, 0, 0, 1, 4, 7, 0, 0)),
                    0x38),
                _mm256_permutevar8x32_epi32(v[2], _mm256_setr_epi32(0, 0, 0, 0, 0, 0, 2, 5)),
                0xC0),
            _mm256_set1_epi32(0x07FFFFFF));
        const __m256i ends = _mm256_blend_epi32(
            _mm256_blend_epi32(
                _mm256_permutevar8x32_epi32(v[0], _mm256_setr_epi32(1, 4, 7, 0, 0, 0, 0, 0)),
                _mm256_permutevar8x32_epi32(v[1], _mm256_setr_epi32(0, 0, 0, 2, 5, 0, 0, 0)),
                0x18),
            _mm256_permutevar8x32_epi32(v[2], _mm256_setr_epi32(0, 0, 0, 0, 0, 0, 3, 6)),
            0xE0);
        const __m256i durs = _mm256_blend_epi32(
            _mm256_blend_epi32(
                _mm256_permutevar8x32_epi32(v[0], _mm256_setr_epi32(2, 5, 0, 0, 0, 0, 0, 0)),
                _mm256_permutevar8x32_epi32(v[1], _mm256_setr_epi32(0, 0, 0, 3, 6, 0, 0, 0)),
                0x1C),
            _mm256_permutevar8x32_epi32(v[2], _mm256_setr_epi32(0, 0, 0, 0, 0, 1, 4, 7)),
            0xE0);
        const __m256i mv = _mm256_set1_epi64x(
            static_cast<long long>(static_cast<uint64_t>((lane << 16) | (dev << 26) | (kSpscRecTypeZone << 29)) << 32));
        const __m256i pv = _mm256_set1_epi64x(prog);
        const __m256i tv = _mm256_set1_epi64x(static_cast<long long>(th_hi));
        uint8_t* const dst = sw.buf + sw.off;
        const auto quad = [&](__m128i en, __m128i du, __m128i id, uint8_t* o) {
            const __m256i d64 = _mm256_cvtepu32_epi64(du);
            const __m256i s64 = _mm256_sub_epi64(_mm256_or_si256(tv, _mm256_cvtepu32_epi64(en)), d64);
            spsc_zone_lines4_avx2(s64, d64, _mm256_or_si256(mv, _mm256_cvtepu32_epi64(id)), pv, o);
        };
        quad(_mm256_castsi256_si128(ends), _mm256_castsi256_si128(durs), _mm256_castsi256_si128(ids), dst);
        quad(
            _mm256_extracti128_si256(ends, 1),
            _mm256_extracti128_si256(durs, 1),
            _mm256_extracti128_si256(ids, 1),
            dst + 128);
        sw.off += kSpscRecBytes * n;
    } else {
        (void)lane;
        (void)dev;
        (void)prog;
    }
    return out;
}

// AVX2 tier of spsc_zone_s16_avx512: the same contract and 16-record block on 256-bit vectors, so a host without
// AVX-512 pays the walk's per-call cost once per run too. A half block still writes all eight of its records
// (the sink's slack covers a partial block).
template <typename Sink>
inline SpscZoneS16Result spsc_zone_s16_avx2(
    const uint32_t* p,
    uint32_t readable,
    uint32_t max_recs,
    uint64_t cursor,
    uint32_t prog,
    uint32_t lane,
    uint32_t dev,
    Sink& sw) {
    SpscZoneS16Result out{0, 0, cursor};
    const __m256i stype = _mm256_set1_epi32(PP_ZONE_S);
    // shuffle_ps gathers a block's even (or odd) words as r0 r1 r4 r5 | r2 r3 r6 r7; this puts them in record order.
    const __m256i order = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    const __m256i lane_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    [[maybe_unused]] const __m256i mv = _mm256_set1_epi64x(
        static_cast<long long>(static_cast<uint64_t>((lane << 16) | (dev << 26) | (kSpscRecTypeZone << 29)) << 32));
    [[maybe_unused]] const __m256i pv = _mm256_set1_epi64x(prog);
    const auto evens = [&](__m256i a, __m256i b) {
        return _mm256_permutevar8x32_epi32(
            _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b), 0x88)), order);
    };
    const auto odds = [&](__m256i a, __m256i b) {
        return _mm256_permutevar8x32_epi32(
            _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b), 0xDD)), order);
    };
    while (max_recs != 0 && readable >= 2u) {
        __m256i v[4];
        if (readable >= 32u) {
            for (int i = 0; i < 4; i++) {
                v[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 8 * i));
            }
        } else {
            for (int i = 0; i < 4; i++) {
                const __m256i m = _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(readable) - 8 * i), lane_idx);
                v[i] = _mm256_maskload_epi32(reinterpret_cast<const int*>(p + 8 * i), m);
            }
        }
        uint32_t hit = 0;
        for (int i = 0; i < 4; i++) {
            hit |= static_cast<uint32_t>(_mm256_movemask_ps(
                       _mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_srli_epi32(v[i], PP_TYPE_SHIFT), stype))))
                   << (8 * i);
        }
        // Only even bits (bit 2r = record r's w0) are meaningful; bit 32 terminates an all-hit scan.
        const uint64_t miss = (~static_cast<uint64_t>(hit) & 0x55555555ull) | (1ull << 32);
        uint32_t n = static_cast<uint32_t>(std::countr_zero(miss)) / 2u;
        if (n > max_recs) {
            n = max_recs;
        }
        if (n == 0) {
            break;
        }
        const __m256i w1_lo = odds(v[0], v[1]);
        const __m256i w1_hi = odds(v[2], v[3]);
        const __m256i pfx_lo = spsc_prefix8_avx2(_mm256_srli_epi32(w1_lo, 16));
        const __m256i pfx_hi = _mm256_add_epi32(
            spsc_prefix8_avx2(_mm256_srli_epi32(w1_hi, 16)), _mm256_permutevar8x32_epi32(pfx_lo, _mm256_set1_epi32(7)));
        if (out.n == 0) {
            out.ts_first = cursor + (p[1] >> 16);
        }
        alignas(32) uint32_t pfx_arr[16];
        _mm256_store_si256(reinterpret_cast<__m256i*>(pfx_arr), pfx_lo);
        _mm256_store_si256(reinterpret_cast<__m256i*>(pfx_arr + 8), pfx_hi);
        if constexpr (Sink::kStores) {
            const __m256i cv = _mm256_set1_epi64x(static_cast<long long>(cursor));
            uint8_t* const dst = sw.buf + sw.off;
            const auto quad = [&](__m128i pf, __m128i du, __m128i id, uint8_t* o) {
                const __m256i d64 = _mm256_cvtepu32_epi64(du);
                const __m256i s64 = _mm256_sub_epi64(_mm256_add_epi64(cv, _mm256_cvtepu32_epi64(pf)), d64);
                spsc_zone_lines4_avx2(s64, d64, _mm256_or_si256(mv, _mm256_cvtepu32_epi64(id)), pv, o);
            };
            const auto half = [&](__m256i pfx, __m256i w0s, __m256i w1s, uint8_t* o) {
                const __m256i durs = _mm256_and_si256(w1s, _mm256_set1_epi32(0xFFFF));
                const __m256i ids = _mm256_and_si256(w0s, _mm256_set1_epi32(0x07FFFFFF));
                quad(_mm256_castsi256_si128(pfx), _mm256_castsi256_si128(durs), _mm256_castsi256_si128(ids), o);
                quad(
                    _mm256_extracti128_si256(pfx, 1),
                    _mm256_extracti128_si256(durs, 1),
                    _mm256_extracti128_si256(ids, 1),
                    o + 128);
            };
            half(pfx_lo, evens(v[0], v[1]), w1_lo, dst);
            if (n > 8) {
                half(pfx_hi, evens(v[2], v[3]), w1_hi, dst + 256);
            }
            sw.off += kSpscRecBytes * n;
        }
        cursor += pfx_arr[n - 1];
        out.n += n;
        out.ts_last = cursor;
        if (n < 16u) {
            break;  // the block ended on a non-S word or the emit budget
        }
        p += 32;
        readable -= 32;
        max_recs -= 16;
    }
    return out;
}

// Decode one packed BULK_SPAN frame in place. emit(lane, zone_id27, end, prog, duration, two_reads) is one whole
// zone; two_reads marks a ZONE_L, whose duration is the difference of two wall-clock reads rather than an exact
// count, which changes how the latch-race repair applies to it. emit_data(lane, wire_type, id, full_ts, prog,
// payload_words, n) for PP_DATA/PP_EVENT (payload in place, hi word first). emit_atomic16 / emit_zone_s16 take a
// block at a PP_ZONE_ATOMIC / PP_ZONE_S word and return the records consumed; 0 hands the word to the scalar arm.
// Returns the payload words the control vector implies, which the caller checks against the frame's length
// field (a pack-rule disagreement desynchronizes every later lane), or 0 for an unknown core. Decode starts at
// the larger of the head mirror and the extent's start: the mirror runs behind after an upstream loss (adopt
// and count), the extent after a lagging head write-back (skip the overlap). The walk is baseline code, so no
// AVX-512 can be emitted outside the gated kernels.
template <typename EmitZone, typename EmitData, typename EmitAtomic16, typename EmitZoneS16>
inline uint32_t spsc_decode_frame(
    SpanDecodeState& st,
    const uint32_t* frame,
    EmitZone&& emit,
    EmitData&& emit_data,
    EmitAtomic16&& emit_atomic16,
    EmitZoneS16&& emit_zone_s16,
    // Nonzero authorizes the atomic block to load (never emit) up to 24 words past a lane's live run.
    uint32_t frame_words = 0) {
    const uint32_t* ctrl = frame + kernel_profiler::SPSC_SPAN_PREFIX_WORDS;
    const auto xy_it = st.core_of_xy.find(ctrl[kernel_profiler::SPSC_WIRE_XY]);
    if (xy_it == st.core_of_xy.end()) {
        st.unknown_core_frames++;
        return 0;
    }
    const uint32_t core = xy_it->second;
    // Folded into st once at the end: the emitters store through casted ring pointers, so the compiler must
    // assume those stores alias st and would reload per record.
    uint64_t lw = 0;
    uint32_t off = kernel_profiler::SPSC_SPAN_PREFIX_WORDS + kernel_profiler::SPSC_SPAN_WIRE_CTRL_WORDS;
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
        // A linearised run lives in `lin`, where `frame + frame_words` is not a comparable pointer, so the over-read
        // vouch must be withdrawn or the gates admit reads past the scratch.
        const uint32_t fw_eff = ring_ordered ? 0u : frame_words;
        uint32_t i = 0;
        while (i < run) {
            const uint32_t w0 = p[i];
            const uint32_t t = pp_type(w0);
            if (t == PP_ZONE_S) {
                const size_t readable =
                    fw_eff != 0 ? static_cast<size_t>(frame + fw_eff - (p + i)) : static_cast<size_t>(run - i);
                const auto zs = emit_zone_s16(
                    lane,
                    cur,
                    pg,
                    p + i,
                    readable > 0xFFFFFFFFull ? 0xFFFFFFFFu : static_cast<uint32_t>(readable),
                    (run - i) / 2u);
                if (zs.n != 0) {
                    cur = zs.ts_last;
                    i += 2u * zs.n;
                    continue;
                }
            }
            // The block emits n records for any n, so there is no tail case.
            if (t == PP_ZONE_ATOMIC) {
                const size_t readable =
                    fw_eff != 0 ? static_cast<size_t>(frame + fw_eff - (p + i)) : static_cast<size_t>(run - i);
                const uint32_t got = emit_atomic16(
                    lane, th, pg, p + i, readable > 48u ? 48u : static_cast<uint32_t>(readable), (run - i) / 3u);
                if (got != 0) {
                    // A block is atomics only (a sticky ends it), so th is constant across it and the last end re-
                    // anchors the lane cursor.
                    cur = pp_full_ts(th, p[i + 3u * (got - 1u) + 1u]);
                    i += 3 * got;
                    continue;
                }
            }
            // Packets the vector paths cannot take: STICKY_TIMER redefines `th` and STICKY_PROG/_EXT redefine `pg` for
            // every later record, and PP_DATA's length is in its own word 2, so the next offset is unknown until read.
            if (t == PP_EVENT) {
                if (i + 2 > run) {
                    st.anomalies++;
                    break;
                }
                emit_data(lane, PP_EVENT, pp_point_id(w0), pp_full_ts(th, p[i + 1]), pg, nullptr, 0);
                i += 2;
            } else if (t == PP_DATA) {
                // PP_DATA is 3 + size words; the packed window is flat, so the payload is handed over in place.
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
            } else if (t == PP_ZONE_L) {
                if (i + 5 > run) {
                    st.anomalies++;
                    break;
                }
                const uint64_t lend = (static_cast<uint64_t>(p[i + 2]) << 32) | p[i + 1];
                const uint64_t ldur = (static_cast<uint64_t>(p[i + 4]) << 32) | p[i + 3];
                emit(lane, pp_low27(w0), lend, pg, ldur, true);  // does not move the cursor
                i += 5;
            } else if (t == PP_ZONE_ATOMIC) {
                if (i + 3 > run) {
                    st.anomalies++;
                    break;
                }
                cur = pp_full_ts(th, p[i + 1]);  // absolute end re-anchors the lane cursor
                emit(lane, pp_low27(w0), cur, pg, p[i + 2], false);
                i += 3;
            } else if (t == PP_ZONE_S) {
                if (i + 2 > run) {
                    st.anomalies++;
                    break;
                }
                const uint32_t w1 = p[i + 1];
                cur += pp_zone_s_delta(w1);  // 64-bit add: crosses the lo-wrap with no sticky
                emit(lane, pp_low27(w0), cur, pg, pp_zone_s_dur(w1), false);
                i += 2;
            } else if (t == PP_STICKY_TIMER) {
                th = pp_timer_hi(w0);
                i += 1;
            } else if (t == PP_STICKY_PROG) {
                pg = pp_low27(w0);
                i += 1;
            } else if (t == PP_STICKY_PROG_EXT) {
                if (i + 2 > run) {
                    st.anomalies++;
                    break;
                }
                pg = p[i + 1];
                i += 2;
            } else {
                st.anomalies++;
                break;
            }
        }
        st.timer_hi[lane] = th;
        st.prog[lane] = pg;
        st.cursor[lane] = cur;
    }
    st.live_words += lw;
    return off - kernel_profiler::SPSC_SPAN_PREFIX_WORDS;
}

}  // namespace tt::tt_metal::profiler
