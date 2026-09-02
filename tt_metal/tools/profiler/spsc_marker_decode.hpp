// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The host-side decode of the DRISC relay's wire (producer:
// tt_metal/tools/profiler/kernels/streaming_profiler_relay.cpp), and the only definition of it.
//
// The wire carries only whole variable-length BULK_SPAN frames (layout and geometry rules in
// profiler_common.h): a 16-word prefix whose word 1 is the payload length, the worker's 64-word control
// vector verbatim, then each RISC's live ring window packed flat, congruence-padded, wrap resolved
// device-side. Frames with SPSC_SPAN_RAW_FLAG in word 0 instead carry the whole raw span (five full
// rings at fixed offsets, windows circular), the relay's high-fill fallback. Inside each window is a
// packet run (spsc_packet.h): ZONE_START/END/TOTAL markers (2 words), STICKY_TIMER (1 word, per-lane
// wall-clock high half), STICKY_PROG (1 word, per-lane runtime host-id in low27; 2-word PROG_EXT escape
// past 2^27), EVENT (2 words, a payload-less flag) and DATA (3 + size words, whose length lives in its
// word2). The producer publishes its tail only on packet boundaries, so a window never ends mid-packet.
#pragma once

#include <algorithm>
#include <bit>
#include <cstring>
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
    "spsc_packet.h (plain C, relay firmware) and profiler_common.h (C++, metal kernels) must agree on the "
    "BULK_SPAN wire code -- they cannot include each other, so this is the only thing holding them together");
// Same argument for the three codes the DRISC self-profiling packer in profiler_common.h emits directly.
static_assert(PP_ZONE_START == kernel_profiler::SPSC_TYPE_ZONE_START, "ZONE_START wire code disagrees");
static_assert(PP_ZONE_END == kernel_profiler::SPSC_TYPE_ZONE_END, "ZONE_END wire code disagrees");
static_assert(PP_STICKY_TIMER == kernel_profiler::SPSC_TYPE_STICKY_TIMER, "STICKY_TIMER wire code disagrees");
static_assert(PP_ZONE_L == kernel_profiler::SPSC_TYPE_ZONE_L, "ZONE_L wire code disagrees");
static_assert(PP_TYPE_SHIFT == kernel_profiler::SPSC_SPAN_TYPE_SHIFT, "packet type field moved");
// The DRISC relay kernel keeps its own copy of the PP_DATA packer (it cannot include
// kernel_profiler.hpp), so its layout constants have to be pinned against this header's. Widening the id
// and moving the size word without updating that copy is not a crash: it renders every one of its
// markers perfectly, with correct timestamps and nesting, under the wrong identity. This translation
// unit is the only one that sees both headers.
static_assert(PP_DATA == kernel_profiler::SPSC_TYPE_DATA, "PP_DATA wire code disagrees");
static_assert(PP_DATA_SIZE_SHIFT == kernel_profiler::SPSC_DATA_SIZE_SHIFT, "PP_DATA size field moved");

namespace tt::tt_metal::profiler {

inline constexpr uint32_t kSpscRingCap = kernel_profiler::PROFILER_L1_VECTOR_SIZE;
inline constexpr uint32_t kSpscRingMask = kSpscRingCap - 1;
inline constexpr uint32_t kSpscNRiscDecode = kernel_profiler::PROFILER_SPSC_TENSIX_RISC;

// Largest PP_DATA payload the 7-bit size field can express; bounds the raw-layout unwrap scratch.
inline constexpr uint32_t kSpscMaxDataWords = 127;

// Worst case: five full rings, each behind a maximal congruence pad. Larger than the raw 2,640-word span
// the relay stages, so this bounds the bounce buffer and frame validation, not any device layout.
inline constexpr uint32_t kSpscMaxPayloadWords =
    kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE +
    kSpscNRiscDecode * (kSpscRingCap + kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS - 1);
inline constexpr uint32_t kSpscMaxFrameWords = kernel_profiler::spsc_span_frame_words(kSpscMaxPayloadWords);
inline constexpr uint32_t kSpscMaxFramePages = kSpscMaxFrameWords / kernel_profiler::SPSC_SPAN_PAGE_WORDS;
static_assert(kSpscMaxFrameWords == 2656 && kSpscMaxFramePages == 166);

// Decode state for one socket's frame stream. Written only by that socket's decode thread.
struct SpanDecodeState {
    std::vector<uint32_t> timer_hi;  // per lane: sticky wall-clock high half
    // Per lane: the end of the last ZONE_S/ZONE_ATOMIC zone, the base a ZONE_S's 16-bit end delta counts
    // from. Mirrors the producer's g_cursor: only those two types move it, and the producer guarantees
    // the first zone after any launch or rewind is an absolute ZONE_ATOMIC. A resync is the one
    // exception; timestamps recover at the next ZONE_ATOMIC, and resyncs are counted and flagged.
    std::vector<uint64_t> cursor;
    std::vector<uint32_t> prog;  // per lane: sticky runtime host-id (every RISC emits its own at launch)
    std::vector<uint32_t> head;  // per lane: monotonic words-consumed mirror; head(N) == tail(N-1)
    std::vector<uint8_t> seeded;
    std::unordered_map<uint32_t, uint32_t> core_of_xy;  // packed (y<<16)|x -> dense core index
    uint64_t live_words = 0;
    uint64_t resync_events = 0;
    uint64_t resync_words = 0;
    uint64_t head_lag = 0;
    uint64_t anomalies = 0;  // torn run / truncated run / undecodable word
    uint64_t unknown_core_frames = 0;
    // Diagnostic: records taken by the wide zone/atomic paths vs the scalar fallback, and how often the
    // per-lane type screen rejected a would-be vector block.
    uint64_t vec_zone_recs = 0, vec_atomic_recs = 0, scalar_recs = 0, vec_block_rejects = 0;
    uint64_t vec_atomic_calls = 0;
    uint64_t vec_zone_s_recs = 0, vec_zone_s_calls = 0;
    uint64_t scalar_by_type[16] = {};

    void reset(uint32_t num_cores) {
        timer_hi.assign(static_cast<size_t>(num_cores) * kSpscNRiscDecode, 0);
        cursor.assign(static_cast<size_t>(num_cores) * kSpscNRiscDecode, 0);
        prog.assign(static_cast<size_t>(num_cores) * kSpscNRiscDecode, 0);
        head.assign(static_cast<size_t>(num_cores) * kSpscNRiscDecode, 0);
        seeded.assign(static_cast<size_t>(num_cores) * kSpscNRiscDecode, 0);
    }
};

struct SpscIgnoreProg {
    void operator()(uint32_t /*lane*/, uint32_t /*prog*/) const {}
};

// Sentinel default for the vectorized zone-block sink: without a real one the walk stays scalar.
struct SpscNoZones8 {};
struct SpscNoAtomic8 {};
struct SpscNoAtomic16 {};
struct SpscNoZoneS16 {};

#if defined(__AVX2__)
// Deinterleave eight consecutive 3-word atomic records (24 contiguous words) into vectors of word0
// (type|id27), word1 (ts low) and word2 (duration). Output lane j belongs to record j.
//
// Attributed rather than built with -march: raising the whole build's baseline to x86-64-v4 changes
// codegen in every other translation unit too. The pragma below confines the ISA to this block, and
// every function in it shares the attribute, which is what lets them still inline into each other
// (clang only refuses to inline across differing target attributes). The kernels take plain arguments
// because the caller is compiled for the baseline ISA and cannot form a __m512i; everything a record
// needs beyond its own three words (th, prog, lane, dev) is block-invariant.
//
// Shuffle/permute operands live at file scope as plain aligned data, so the attributed block loads them
// from L1 instead of rebuilding the vectors on every call.
//
// vpermt2d index rows composing output lines straight from the packed wire. Record r's 24 B is six
// dwords {ts, th, id, meta, prog, dur} drawn from wire words {3r+1, -, 3r, -, -, 3r+2}; indices 0-15
// select the source vector's 16 wire words, 16/17/18 select th/meta/prog from the constant operand.
// Eight records = 24 words = 1.5 vectors, so the layout repeats every three lines and every 1.5 source
// vectors: rows 1-2 serve lines 1&4 and 2&5 (sources: words 8-23, then v2), row 3 is row 0 shifted by 8.
alignas(64) inline constexpr uint32_t kA16Lines[4][16] = {
    {1, 16, 0, 17, 18, 2, 4, 16, 3, 17, 18, 5, 7, 16, 6, 17},        // line 0 from v0 (words 0-15)
    {18, 0, 2, 16, 1, 17, 18, 3, 5, 16, 4, 17, 18, 6, 8, 16},        // lines 1, 4
    {7, 17, 18, 9, 11, 16, 10, 17, 18, 12, 14, 16, 13, 17, 18, 15},  // lines 2, 5
    {9, 16, 8, 17, 18, 10, 12, 16, 11, 17, 18, 13, 15, 16, 14, 17},  // line 3 from v1 (words 16-31)
};

// ZONE_S block operands, file-scope for the same reason as kA16Lines. Even/odd deinterleave a 16-record
// (32-word) load pair into w0s/w1s; the EM/D rows compose output lines from three COMPUTED 8-qword
// sources E (ends), M (meta|id), D (dur|prog) -- record r is qwords {E_r, M_r, D_r} -- via one
// two-source qword permute (indices 0-7 = E, 8-15 = M) plus one masked permute pulling D into the
// remaining lanes (kZS16DMask). Three lines per 8 records; both halves use the same rows.
alignas(64) inline constexpr uint32_t kZS16Even[16] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
alignas(64) inline constexpr uint32_t kZS16Odd[16] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};
alignas(64) inline constexpr uint64_t kZS16EM[3][8] = {
    {0, 8, 0, 1, 9, 0, 2, 10},
    {0, 3, 11, 0, 4, 12, 0, 5},
    {13, 0, 6, 14, 0, 7, 15, 0},
};
alignas(64) inline constexpr uint64_t kZS16D[3][8] = {
    {0, 0, 0, 0, 0, 1, 0, 0},
    {2, 0, 0, 3, 0, 0, 4, 0},
    {0, 5, 0, 0, 6, 0, 0, 7},
};
inline constexpr uint8_t kZS16DMask[3] = {0x24, 0x49, 0x92};

// The build baseline is x86-64-v3 (AVX2), so AVX-512 may exist only inside explicitly attributed
// functions (the two block kernels and the cached sink's line stores below) and every call into one sits
// behind this check. Keeping the attribute off the walk, the emitters and ingest makes the guarantee
// structural: outside the kernels the compiler cannot emit AVX-512 at all, so a host without it
// (Rome/Milan, post-ADL Intel client) runs the AVX2/scalar tier instead of faulting.
// TT_METAL_STREAMING_PROFILER_NO_AVX512=1 forces the fallback tier, for testing it on capable hosts.
inline bool spsc_host_avx512() {
    static const bool v = [] {
        if (std::getenv("TT_METAL_STREAMING_PROFILER_NO_AVX512") != nullptr) {
            return false;
        }
        return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
               __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq");
    }();
    return v;
}

// The blocks compose 24 B records as whole 64 B lines and hand them to a Sink. Delivery decodes into a
// consumer-private scratch buffer, so stores are cached: the buffer is re-read immediately by the same
// thread, and an NT store would evict the line and turn the read-back into a DRAM round trip. The audit
// path only wants the endpoints and counts, so its sink stores nothing and the blocks skip the compose
// entirely (Sink::kStores is a compile-time gate).
//
// put(lines, nq) writes nq qwords taken from lines[0..ceil(nq/8)-1]; a put may over-store up to a full
// vector past its nq qwords (overwritten by the next put), so the buffer needs 64 B of slack past cap.
// put/put_lines are reachable only from the attributed kernels; put3 is called from baseline emitters
// and stays scalar.
struct SpscCachedRecSink {
    static constexpr bool kStores = true;
    uint8_t* buf = nullptr;
    uint64_t off = 0;  // bytes written; 24 B per record

    __attribute__((target("avx512f"))) inline void put(const __m512i* lines, uint32_t nq) {
        for (uint32_t k = 0; 8 * k < nq; k++) {
            _mm512_storeu_si512(buf + off + 64ull * k, lines[k]);
        }
        off += 8ull * nq;
    }
    inline void put3(uint64_t a, uint64_t b, uint64_t c) {  // one 24 B record
        uint64_t* p = reinterpret_cast<uint64_t*>(buf + off);
        p[0] = a;
        p[1] = b;
        p[2] = c;
        off += 24;
    }
    __attribute__((target("avx512f"))) inline void put_lines(const __m512i* lines, uint32_t bytes) {
        put(lines, bytes >> 3);
    }
};
struct SpscNullRecSink {
    static constexpr bool kStores = false;
    inline void put(const __m512i*, uint32_t) {}
    inline void put3(uint64_t, uint64_t, uint64_t) {}
    inline void put_lines(const __m512i*, uint32_t) {}
};

// Public-record sink: the ZONE_S blocks compose the consumer-facing 32 B record directly, four qwords
// {start, duration, meta<<32|id, prog}, so a complete zone never round-trips through the 24 B raw form
// plus a scalar re-transform. Only genuine start/end pairs need pairing state, and those take the
// scalar arm.
struct SpscZone32Sink {
    static constexpr bool kStores = true;
    static constexpr bool kZone32 = true;
    uint8_t* buf = nullptr;
    uint64_t off = 0;  // bytes written; 32 B per record

    __attribute__((target("avx512f"))) inline void put_lines(const __m512i* lines, uint32_t bytes) {
        for (uint32_t k = 0; 64 * k < bytes; k++) {
            _mm512_storeu_si512(buf + off + 64ull * k, lines[k]);
        }
        off += bytes;
    }
    inline void put4(uint64_t q0, uint64_t q1, uint64_t q2, uint64_t q3) {
        uint64_t* p = reinterpret_cast<uint64_t*>(buf + off);
        p[0] = q0;
        p[1] = q1;
        p[2] = q2;
        p[3] = q3;
        off += 32;
    }
};

template <typename S>
inline constexpr bool kSpscSinkZone32 = requires { S::kZone32; };

#pragma clang attribute push(__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq"))), apply_to = function)

struct SpscA16Result {
    uint32_t n;
    uint64_t ts_first;
    uint64_t ts_last;
};

// Any n from 1 to 16, so run tails need no narrower path. Each output line comes straight off the wire
// words through one two-source permute (kA16Lines), with th/meta/prog riding the constant operand, so
// there is no SoA deinterleave and no unpack/re-interleave stage.
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
    SpscA16Result out{0, 0, 0};
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
    // The compare sees all 48 words and any ts/dur word can land on the type pattern, so only every
    // third bit (bit 3r = record r's w0) is meaningful. Bit 48 terminates an all-hit scan.
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
    // Endpoints as scalar reloads of L1-hot source lines: no store to forward from, and no contention
    // on the shuffle ports.
    out.ts_first = th_hi | p[1];
    out.ts_last = th_hi | p[3u * n - 2u];
    if constexpr (Sink::kStores) {
        const __m512i cvec = _mm512_zextsi128_si512(_mm_setr_epi32(
            static_cast<int>(th), static_cast<int>((lane << 16) | (dev << 26)), static_cast<int>(prog), 0));
        const __m512i idm = _mm512_set1_epi32(0x07FFFFFF);
        v0 = _mm512_mask_and_epi32(v0, 0x9249, v0, idm);
        v1 = _mm512_mask_and_epi32(v1, 0x4924, v1, idm);
        v2 = _mm512_mask_and_epi32(v2, 0x2492, v2, idm);
        const __m512i u1 = _mm512_alignr_epi32(v1, v0, 8);
        __m512i o[7];
        o[0] = _mm512_permutex2var_epi32(v0, _mm512_load_si512(kA16Lines[0]), cvec);
        o[1] = _mm512_permutex2var_epi32(u1, _mm512_load_si512(kA16Lines[1]), cvec);
        o[2] = _mm512_permutex2var_epi32(u1, _mm512_load_si512(kA16Lines[2]), cvec);
        if (n > 8) {  // the second compose half only when its lanes are live
            o[3] = _mm512_permutex2var_epi32(v1, _mm512_load_si512(kA16Lines[3]), cvec);
            o[4] = _mm512_permutex2var_epi32(v2, _mm512_load_si512(kA16Lines[1]), cvec);
            o[5] = _mm512_permutex2var_epi32(v2, _mm512_load_si512(kA16Lines[2]), cvec);
        }
        sw.put_lines(o, 24u * n);
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

// The ZONE_S counterpart of the atomic block above, for the wire's dense-zone hot type. A ZONE_S end is
// cursor-relative (end_r = cursor + sum of deltas 0..r), which looks serial but is an inclusive prefix
// sum, four shifted adds for 16 lanes. The block then composes exactly like the atomic one: records
// normalized to ZONE_ATOMIC form (type Zone, absolute end, duration), so downstream never sees wire size
// classes. Same load contract as spsc_atomic16_avx512: any n from 1 to 16, and `avail` authorizes vector
// loads, never emits, past the live run; the scalar endpoint read stays inside it because windows never
// end mid-packet.
template <typename Sink>
inline SpscZoneS16Result spsc_zone_s16_avx512(
    const uint32_t* p,
    uint32_t avail,
    uint32_t max_recs,
    uint64_t cursor,
    uint32_t prog,
    uint32_t lane,
    uint32_t dev,
    Sink& sw) {
    SpscZoneS16Result out{0, 0, 0};
    __m512i v0, v1;
    if (avail >= 32u) {
        v0 = _mm512_loadu_si512(p);
        v1 = _mm512_loadu_si512(p + 16);
    } else {
        const __mmask16 m0 = static_cast<__mmask16>(avail >= 16u ? 0xFFFFu : ((1u << avail) - 1u));
        const __mmask16 m1 = static_cast<__mmask16>(avail <= 16u ? 0u : ((1u << (avail - 16u)) - 1u));
        v0 = _mm512_maskz_loadu_epi32(m0, p);
        v1 = _mm512_maskz_loadu_epi32(m1, p + 16);
    }
    const __m512i stype = _mm512_set1_epi32(PP_ZONE_S);
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
        return out;
    }
    const __m512i w1s = _mm512_permutex2var_epi32(v0, _mm512_load_si512(kZS16Odd), v1);
    const __m512i z = _mm512_setzero_si512();
    __m512i pfx = _mm512_srli_epi32(w1s, 16);
    pfx = _mm512_add_epi32(pfx, _mm512_alignr_epi32(pfx, z, 15));
    pfx = _mm512_add_epi32(pfx, _mm512_alignr_epi32(pfx, z, 14));
    pfx = _mm512_add_epi32(pfx, _mm512_alignr_epi32(pfx, z, 12));
    pfx = _mm512_add_epi32(pfx, _mm512_alignr_epi32(pfx, z, 8));
    out.n = n;
    out.ts_first = cursor + (p[1] >> 16);
    // Spilled, not lane-extracted: n-1 is runtime, and one aligned store to hot stack beats a
    // variable-lane compress on the shuffle ports.
    alignas(64) uint32_t pfx_arr[16];
    _mm512_store_si512(pfx_arr, pfx);
    out.ts_last = cursor + pfx_arr[n - 1];
    if constexpr (kSpscSinkZone32<Sink>) {
        const __m512i w0s = _mm512_permutex2var_epi32(v0, _mm512_load_si512(kZS16Even), v1);
        const __m512i durs = _mm512_and_si512(w1s, _mm512_set1_epi32(0xFFFF));
        const __m512i ids = _mm512_and_si512(w0s, _mm512_set1_epi32(0x07FFFFFF));
        // Public meta: type = StreamingProfilerRecType::Zone (1).
        const uint64_t meta64 = static_cast<uint64_t>((lane << 16) | (dev << 26) | (1u << 29)) << 32;
        const __m512i mv = _mm512_set1_epi64(static_cast<long long>(meta64));
        const __m512i pv = _mm512_set1_epi64(prog);
        const __m512i cv = _mm512_set1_epi64(static_cast<long long>(cursor));
        // Record r is qwords {S_r, D_r, M_r, P_r}: one index vector serves both two-source permutes --
        // lanes 0,1,4,5 pick from (S,D), lanes 2,3,6,7 pick the same positions from (M,P), and the next
        // line is the same indices advanced by two records.
        const __m512i vbase = _mm512_set_epi64(9, 1, 9, 1, 8, 0, 8, 0);
        const __m512i inc = _mm512_set1_epi64(2);
        const auto half = [&](__m256i pfx_h, __m256i dur_h, __m256i id_h, __m512i* o) {
            const __m512i d64 = _mm512_cvtepu32_epi64(dur_h);
            const __m512i e64 = _mm512_add_epi64(cv, _mm512_cvtepu32_epi64(pfx_h));
            const __m512i s64 = _mm512_sub_epi64(e64, d64);
            const __m512i m64 = _mm512_or_si512(mv, _mm512_cvtepu32_epi64(id_h));
            __m512i idx = vbase;
            for (int j = 0; j < 4; j++) {
                const __m512i sd = _mm512_permutex2var_epi64(s64, idx, d64);
                const __m512i mp = _mm512_permutex2var_epi64(m64, idx, pv);
                o[j] = _mm512_mask_blend_epi64(0xCC, sd, mp);
                idx = _mm512_add_epi64(idx, inc);
            }
        };
        __m512i o[9];
        half(_mm512_castsi512_si256(pfx), _mm512_castsi512_si256(durs), _mm512_castsi512_si256(ids), o);
        if (n > 8) {
            half(
                _mm512_extracti64x4_epi64(pfx, 1),
                _mm512_extracti64x4_epi64(durs, 1),
                _mm512_extracti64x4_epi64(ids, 1),
                o + 4);
        }
        sw.put_lines(o, 32u * n);
    } else if constexpr (Sink::kStores) {
        const __m512i w0s = _mm512_permutex2var_epi32(v0, _mm512_load_si512(kZS16Even), v1);
        const __m512i durs = _mm512_and_si512(w1s, _mm512_set1_epi32(0xFFFF));
        const __m512i ids = _mm512_and_si512(w0s, _mm512_set1_epi32(0x07FFFFFF));
        const uint64_t meta64 = static_cast<uint64_t>((lane << 16) | (dev << 26)) << 32;  // type = Zone
        const __m512i mv = _mm512_set1_epi64(static_cast<long long>(meta64));
        const __m512i pv = _mm512_set1_epi64(prog);
        const __m512i cv = _mm512_set1_epi64(static_cast<long long>(cursor));
        const auto compose = [&](__m512i e, __m512i m, __m512i d, int r) {
            const __m512i em = _mm512_permutex2var_epi64(e, _mm512_load_si512(kZS16EM[r]), m);
            return _mm512_mask_permutexvar_epi64(em, kZS16DMask[r], _mm512_load_si512(kZS16D[r]), d);
        };
        const __m512i e_lo = _mm512_add_epi64(cv, _mm512_cvtepu32_epi64(_mm512_castsi512_si256(pfx)));
        const __m512i m_lo = _mm512_or_si512(mv, _mm512_cvtepu32_epi64(_mm512_castsi512_si256(ids)));
        const __m512i d_lo =
            _mm512_or_si512(pv, _mm512_slli_epi64(_mm512_cvtepu32_epi64(_mm512_castsi512_si256(durs)), 32));
        __m512i o[7];
        o[0] = compose(e_lo, m_lo, d_lo, 0);
        o[1] = compose(e_lo, m_lo, d_lo, 1);
        o[2] = compose(e_lo, m_lo, d_lo, 2);
        if (n > 8) {
            const __m512i e_hi = _mm512_add_epi64(cv, _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(pfx, 1)));
            const __m512i m_hi = _mm512_or_si512(mv, _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(ids, 1)));
            const __m512i d_hi =
                _mm512_or_si512(pv, _mm512_slli_epi64(_mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(durs, 1)), 32));
            o[3] = compose(e_hi, m_hi, d_hi, 0);
            o[4] = compose(e_hi, m_hi, d_hi, 1);
            o[5] = compose(e_hi, m_hi, d_hi, 2);
        }
        sw.put_lines(o, 24u * n);
    } else {
        (void)lane;
        (void)dev;
        (void)prog;
    }
    return out;
}

#pragma clang attribute pop

// AVX2 tier of the ZONE_S block, for hosts without AVX-512 (spsc_host_avx512() false): same contract, up
// to eight records (16 words). Everything here is baseline x86-64-v3, executable on any host this binary
// runs on. The compose spills to aligned scratch and emits through put3, three scalar stores per record.
template <typename Sink>
inline SpscZoneS16Result spsc_zone_s8_avx2(
    const uint32_t* p,
    uint32_t avail,
    uint32_t max_recs,
    uint64_t cursor,
    uint32_t prog,
    uint32_t lane,
    uint32_t dev,
    Sink& sw) {
    SpscZoneS16Result out{0, 0, 0};
    __m256i v0, v1;
    if (avail >= 16u) {
        v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 8));
    } else {
        alignas(32) int32_t m[16] = {};
        for (uint32_t i = 0; i < avail; i++) {
            m[i] = -1;
        }
        v0 = _mm256_maskload_epi32(
            reinterpret_cast<const int*>(p), _mm256_load_si256(reinterpret_cast<const __m256i*>(m)));
        v1 = _mm256_maskload_epi32(
            reinterpret_cast<const int*>(p + 8), _mm256_load_si256(reinterpret_cast<const __m256i*>(m + 8)));
    }
    const __m256i stype = _mm256_set1_epi32(PP_ZONE_S);
    const uint32_t k0 = static_cast<uint32_t>(
        _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_srli_epi32(v0, PP_TYPE_SHIFT), stype))));
    const uint32_t k1 = static_cast<uint32_t>(
        _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_srli_epi32(v1, PP_TYPE_SHIFT), stype))));
    const uint32_t miss = (~(k0 | (k1 << 8)) & 0x5555u) | (1u << 16);
    uint32_t n = static_cast<uint32_t>(std::countr_zero(miss)) / 2u;
    if (n > max_recs) {
        n = max_recs;
    }
    if (n == 0) {
        return out;
    }
    const __m256i idx_even = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
    const __m256i idx_odd = _mm256_setr_epi32(1, 3, 5, 7, 1, 3, 5, 7);
    const __m256i w0s = _mm256_permute2x128_si256(
        _mm256_permutevar8x32_epi32(v0, idx_even), _mm256_permutevar8x32_epi32(v1, idx_even), 0x20);
    const __m256i w1s = _mm256_permute2x128_si256(
        _mm256_permutevar8x32_epi32(v0, idx_odd), _mm256_permutevar8x32_epi32(v1, idx_odd), 0x20);
    // Hillis-Steele prefix sum: cross-lane shift via permutevar, low lanes zeroed by blend.
    const __m256i z = _mm256_setzero_si256();
    __m256i pfx = _mm256_srli_epi32(w1s, 16);
    pfx = _mm256_add_epi32(
        pfx, _mm256_blend_epi32(_mm256_permutevar8x32_epi32(pfx, _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6)), z, 0x01));
    pfx = _mm256_add_epi32(
        pfx, _mm256_blend_epi32(_mm256_permutevar8x32_epi32(pfx, _mm256_setr_epi32(0, 0, 0, 1, 2, 3, 4, 5)), z, 0x03));
    pfx = _mm256_add_epi32(
        pfx, _mm256_blend_epi32(_mm256_permutevar8x32_epi32(pfx, _mm256_setr_epi32(0, 0, 0, 0, 0, 1, 2, 3)), z, 0x0F));
    alignas(32) uint32_t pfx_arr[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(pfx_arr), pfx);
    out.n = n;
    out.ts_first = cursor + (p[1] >> 16);
    out.ts_last = cursor + pfx_arr[n - 1];
    if constexpr (Sink::kStores) {
        alignas(32) uint32_t id_arr[8];
        alignas(32) uint32_t dur_arr[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(id_arr), _mm256_and_si256(w0s, _mm256_set1_epi32(0x07FFFFFF)));
        _mm256_store_si256(reinterpret_cast<__m256i*>(dur_arr), _mm256_and_si256(w1s, _mm256_set1_epi32(0xFFFF)));
        const uint64_t meta64 = static_cast<uint64_t>((lane << 16) | (dev << 26)) << 32;  // type = Zone
        for (uint32_t k = 0; k < n; k++) {
            sw.put3(cursor + pfx_arr[k], meta64 | id_arr[k], (static_cast<uint64_t>(dur_arr[k]) << 32) | prog);
        }
    } else {
        (void)lane;
        (void)dev;
        (void)prog;
    }
    return out;
}

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

// Decode ONE whole packed BULK_SPAN frame in place. For each marker calls
//   emit(lane, wire_type, zone_id27, full_ts, prog, duration)
//     ZONE_START/END: full_ts is the marker's time, duration 0. ZONE_TOTAL: full_ts is the sum, duration 0.
//     ZONE_ATOMIC: full_ts is the zone's END and duration is its length, so start = full_ts - duration.
//     Duration is its own arg rather than riding the prog slot because op attribution keys on prog
//     (streaming_profiler_ops_csv drops records with prog == 0), so an atomic zone must keep both.
// where zone_id27 is the full 27-bit structural zone id (tu_id << TT_ZONE_LOCAL_BITS | local,
// hostdevcommon/profiler_zone_id.h), and for each PP_DATA/PP_EVENT
//   emit_data(lane, wire_type, id, full_ts, prog, payload_words, n)   (payload in place, hi-word first)
// and emit_prog(lane, prog) whenever a lane's sticky host-id changes.
//
// Returns the payload words the control vector implies -- the caller checks it against the frame's own
// length field, since a pack-rule disagreement with the relay desynchronizes every lane after the
// first -- or 0 for an unknown-core frame (decoded as nothing; the caller still owns the advance).
//
// Head adoption: the mirror and the extent's start are both monotonic and can each run behind. The
// mirror runs behind after a frame was lost upstream (device credit-timeout drop); the extent runs
// behind when the relay's head write-back lagged its snapshot, so the frame re-ships words the mirror
// already consumed. Decode therefore begins at the larger of the two: adopt-and-count on a loss, skip
// the overlap on a lag.
//
// The optional emit_zones8(lane, timer_hi, prog, w0s, w1s) sink receives eight consecutive 2-word zone
// markers at once, deinterleaved into AVX2 vectors of word0s and word1s, whenever a 16-word block passes
// the all-zone type screen. emit_atomic8(lane, timer_hi, prog, w0s, w1s, w2s, n) is the same for up to
// eight 3-word PP_ZONE_ATOMIC records (w2s = durations): the n leading lanes are valid and the rest are
// speculative over-read that must not be emitted. Without it the atomic wire decodes entirely scalar.
// The walk itself is baseline code: keeping the attribute off it means no AVX-512 can be emitted outside
// the spsc_host_avx512()-gated kernels, so the portability guarantee is structural rather than audited.
template <
    typename EmitMarker,
    typename EmitData,
    typename EmitProg = SpscIgnoreProg,
    typename EmitZones8 = SpscNoZones8,
    typename EmitAtomic8 = SpscNoAtomic8,
    typename EmitAtomic16 = SpscNoAtomic16,
    typename EmitZoneS16 = SpscNoZoneS16>
inline uint32_t spsc_decode_frame(
    SpanDecodeState& st,
    const uint32_t* frame,
    EmitMarker&& emit,
    EmitData&& emit_data,
    EmitProg&& emit_prog = SpscIgnoreProg{},
    EmitZones8&& emit_zones8 = SpscNoZones8{},
    EmitAtomic8&& emit_atomic8 = SpscNoAtomic8{},
    EmitAtomic16&& emit_atomic16 = SpscNoAtomic16{},
    EmitZoneS16&& emit_zone_s16 = SpscNoZoneS16{},
    // Total words in the frame buffer. Nonzero authorizes the atomic block path to LOAD (never emit) up
    // to 24 words past a lane's live run -- the bytes exist in the frame/bounce buffer -- which lets
    // run-tails and sticky-split blocks go through the vector path with a partial count.
    uint32_t frame_words = 0) {
    (void)emit_atomic16;
    (void)emit_atomic8;  // the atomic arm is 512-bit only
    (void)emit_zone_s16;
    const uint32_t* ctrl = frame + kernel_profiler::SPSC_SPAN_PREFIX_WORDS;
    const bool raw = (frame[0] & kernel_profiler::SPSC_SPAN_RAW_FLAG) != 0;
    const auto xy_it = st.core_of_xy.find(ctrl[raw ? +kernel_profiler::SPSC_CORE_XY : +kernel_profiler::SPSC_WIRE_XY]);
    if (xy_it == st.core_of_xy.end()) {
        st.unknown_core_frames++;
        return 0;
    }
    const uint32_t core = xy_it->second;
    // Frame-local counters, folded into st once at the end: the emitters store through casted ring
    // pointers, so the compiler must assume the NT stores alias a SpanDecodeState field and would turn
    // each update here into a load-add-store per record.
    uint64_t vz = 0, va = 0, vac = 0, sr = 0, lw = 0, vzs = 0, vzsc = 0;
    uint32_t off = kernel_profiler::SPSC_SPAN_PREFIX_WORDS + (raw ? kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE
                                                                  : kernel_profiler::SPSC_SPAN_WIRE_CTRL_WORDS);
    for (uint32_t r = 0; r < kSpscNRiscDecode; r++) {
        const uint32_t lane = core * kSpscNRiscDecode + r;
        const uint32_t tail = ctrl[(raw ? +kernel_profiler::SPSC_RING_TAIL_0 : +kernel_profiler::SPSC_WIRE_TAIL_0) + r];
        const uint32_t frame_head =
            ctrl[(raw ? +kernel_profiler::SPSC_RING_HEAD_0 : +kernel_profiler::SPSC_WIRE_HEAD_0) + r];
        const uint32_t extent = kernel_profiler::spsc_span_live(frame_head, tail, kSpscRingCap);
        if (extent != tail - frame_head) {
            st.anomalies++;  // torn snapshot; the clamped geometry still frames consistently on both sides
        }
        const uint32_t start = tail - extent;
        const uint32_t* p = nullptr;
        // A packed lane whose near-full run wraps the ring arrives as the whole ring image,
        // ring-ordered, in one device read; a small wrapping run arrives as the two-piece split, already
        // in run order. The predicate is shared with the device (spsc_span_wrap_image), the pad is then
        // phased for ring offset 0, and the payload advance is the full ring, not the extent.
        const bool ring_ordered =
            !raw && extent != 0 && kernel_profiler::spsc_span_wrap_image(start, extent, kSpscRingCap);
        if (!raw && extent != 0) {
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
        lw += run;
        uint32_t th = st.timer_hi[lane];
        uint32_t pg = st.prog[lane];
        uint64_t cur = st.cursor[lane];
        // One decode path for both frame layouts: a raw frame carries whole 512-word rings and is read
        // circularly, a packed frame is already contiguous, and that is a difference in addressing rather
        // than in decoding. Linearise and share the walk. Raw frames are the relays' own self frames, so
        // the copy is off the workload's path.
        uint32_t lin[kSpscRingCap];
        if (raw || ring_ordered) {
            const uint32_t* ring = raw ? frame + kernel_profiler::SPSC_SPAN_PREFIX_WORDS +
                                             kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE + r * kSpscRingCap
                                       : p;
            const uint32_t hm = head & kSpscRingMask;
            const uint32_t first = kSpscRingCap - hm < run ? kSpscRingCap - hm : run;
            std::memcpy(lin, ring + hm, first * sizeof(uint32_t));
            if (first < run) {
                std::memcpy(lin + first, ring, (run - first) * sizeof(uint32_t));
            }
            p = lin;
        } else {
            p += extent - run;
        }
        // The over-read gates below vouch for words past a lane's run using the frame buffer's extent. A
        // linearised run lives in `lin` instead, where no such slack exists and `frame + frame_words` is
        // not even a comparable pointer, so the frame vouch must be withdrawn here or the gates admit
        // reads past the scratch and the decode picks up garbage records.
        [[maybe_unused]] const uint32_t fw_eff = (raw || ring_ordered) ? 0u : frame_words;
        // Just-in-time prefetch of this lane's live window: small bursts consumed immediately fit the
        // core's fill-buffer budget, where issuing whole frames ahead is slower because the bulk cold-line
        // prefetches starve the walk's own demand loads. The second stream mirrors the same offsets one
        // frame ahead, since the device DMA lands in DRAM rather than cache and the just-in-time stream
        // alone would eat the full miss latency.
        for (uint32_t o = 0; o < run; o += 16) {
            spsc_prefetch(p + o);
            if (fw_eff != 0) {
                spsc_prefetch(p + o + frame_words);
            }
        }
        uint32_t i = 0;
        while (i < run) {
            const uint32_t w0 = p[i];
            const uint32_t t = pp_type(w0);
#if defined(__AVX2__)
            if constexpr (!std::is_same_v<std::decay_t<EmitZoneS16>, SpscNoZoneS16>) {
                if (t == PP_ZONE_S) {
                    const size_t readable =
                        fw_eff != 0 ? static_cast<size_t>(frame + fw_eff - (p + i)) : static_cast<size_t>(run - i);
                    const auto zs = emit_zone_s16(
                        lane, cur, pg, p + i, readable > 32u ? 32u : static_cast<uint32_t>(readable), (run - i) / 2u);
                    if (zs.n != 0) {
                        cur = zs.ts_last;
                        vzs += zs.n;
                        vzsc++;
                        i += 2u * zs.n;
                        continue;
                    }
                }
            }
#endif
#if defined(__AVX2__)
            // The screen matches 2-word markers only, so gate on the leading type rather than letting an
            // atomic stream fail the scan once per record.
            if constexpr (!std::is_same_v<std::decay_t<EmitZones8>, SpscNoZones8>) {
                if (t <= PP_ZONE_END && i + 16 <= run) {
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
                        vz += 8;
                        i += 16;
                        continue;
                    }
                    st.vec_block_rejects++;
                }
            }
#endif  // __AVX2__
#if defined(__AVX2__)
            // The atomic arm. The block emits n records for any n, so there is no tail case and no
            // narrower path behind it.
            if constexpr (!std::is_same_v<std::decay_t<EmitAtomic16>, SpscNoAtomic16>) {
                if (t == PP_ZONE_ATOMIC) {
                    const size_t readable =
                        fw_eff != 0 ? static_cast<size_t>(frame + fw_eff - (p + i)) : static_cast<size_t>(run - i);
                    const uint32_t got = emit_atomic16(
                        lane, th, pg, p + i, readable > 48u ? 48u : static_cast<uint32_t>(readable), (run - i) / 3u);
                    if (got != 0) {
                        va += got;
                        vac++;
                        // The block is atomics only (a sticky ends it), so th is constant across it and
                        // the last atomic's absolute end re-anchors the lane cursor.
                        cur = pp_full_ts(th, p[i + 3u * (got - 1u) + 1u]);
                        i += 3 * got;
                        continue;
                    }
                }
            }
#endif
            // Everything reaching here is a packet the vector paths above cannot take, and not for want
            // of a wider instruction: PP_STICKY_TIMER redefines `th` and PP_STICKY_PROG/_EXT redefine `pg`
            // for every record after them, so a lane cannot be decoded before its predecessors are, and
            // PP_DATA carries its length in its own word 2, so the next record's offset is unknown until
            // it is read.
            sr++;
            st.scalar_by_type[t & 15u]++;
            if (t == PP_EVENT) {
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
            } else if (t == PP_ZONE_L) {
                if (i + 5 > run) {
                    st.anomalies++;
                    break;
                }
                const uint64_t lend = (static_cast<uint64_t>(p[i + 2]) << 32) | p[i + 1];
                const uint64_t ldur = (static_cast<uint64_t>(p[i + 4]) << 32) | p[i + 3];
                // A 64-bit duration cannot ride the 32-bit dur argument: normalize to a synthetic
                // START/END pair for the downstream pairing stack. Does not move the cursor.
                emit(lane, PP_ZONE_START, pp_low27(w0), lend - ldur, pg, 0);
                emit(lane, PP_ZONE_END, pp_low27(w0), lend, pg, 0);
                i += 5;
            } else if (t == PP_ZONE_START || t == PP_ZONE_END || t == PP_ZONE_TOTAL) {
                if (i + 2 > run) {
                    st.anomalies++;
                    break;
                }
                const uint32_t w1 = p[i + 1];
                const uint64_t ts = (t == PP_ZONE_TOTAL) ? w1 : pp_full_ts(th, w1);
                emit(lane, t, pp_low27(w0), ts, pg, 0);  // full 27-bit structural id
                i += 2;
            } else if (t == PP_ZONE_ATOMIC) {
                if (i + 3 > run) {
                    st.anomalies++;
                    break;
                }
                cur = pp_full_ts(th, p[i + 1]);  // absolute end re-anchors the lane cursor
                emit(lane, t, pp_low27(w0), cur, pg, p[i + 2]);
                i += 3;
            } else if (t == PP_ZONE_S) {
                if (i + 2 > run) {
                    st.anomalies++;
                    break;
                }
                const uint32_t w1 = p[i + 1];
                cur += pp_zone_s_delta(w1);  // 64-bit add: crosses the lo-wrap with no sticky
                // Normalized at this boundary: an S emits as a ZONE_ATOMIC record with its end resolved
                // off the lane cursor, so nothing downstream knows the wire had size classes.
                emit(lane, PP_ZONE_ATOMIC, pp_low27(w0), cur, pg, pp_zone_s_dur(w1));
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
            } else {
                st.anomalies++;
                break;
            }
        }
        st.timer_hi[lane] = th;
        st.prog[lane] = pg;
        st.cursor[lane] = cur;
    }
    st.vec_zone_recs += vz;
    st.vec_atomic_recs += va;
    st.vec_atomic_calls += vac;
    st.vec_zone_s_recs += vzs;
    st.vec_zone_s_calls += vzsc;
    st.scalar_recs += sr;
    st.live_words += lw;
    if (raw) {
        return kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE + kSpscNRiscDecode * kSpscRingCap;
    }
    return off - kernel_profiler::SPSC_SPAN_PREFIX_WORDS;
}

}  // namespace tt::tt_metal::profiler
