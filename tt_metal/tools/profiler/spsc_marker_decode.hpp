// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// SINGLE SOURCE OF TRUTH for the host-side decode of the DRISC drainer's wire
// (producer: tt_metal/tools/profiler/kernels/drisc_profiler_filler.cpp).
//
// The wire carries only whole variable-length BULK_SPAN frames (layout and geometry rules in
// profiler_common.h): a 16-word prefix whose word 1 is the payload length, the worker's 64-word control
// vector verbatim, then each RISC's live ring window packed flat -- congruence-padded, wrap resolved
// device-side. Frames with SPSC_SPAN_RAW_FLAG in word 0 instead carry the whole raw span (five full
// rings at fixed offsets, windows circular) -- the drainer's high-fill fallback, where packing would
// cost write issues to save almost nothing. Inside each window is a packet run (spsc_packet.h):
// ZONE_START/END/TOTAL markers
// (2 words), STICKY_TIMER (1 word, per-lane wall-clock high half), STICKY_PROG (1 word, per-lane
// runtime host-id in low27; 2-word PROG_EXT escape past 2^27), EVENT (2 words, a payload-less flag)
// and DATA (3 + size words, self-describing -- the length lives in its word2). The producer publishes
// its tail only on packet boundaries, so a window never ends mid-packet.
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
    // Per lane: the end of the last ZONE_S/ZONE_ATOMIC zone -- the base a ZONE_S's 16-bit end delta
    // counts from. Mirrors the producer's g_cursor exactly: only those two types move it, and the
    // producer guarantees the first zone after any launch/rewind is an absolute ZONE_ATOMIC, so a
    // ZONE_S is never decoded against a cursor the producer didn't set (resync is the one exception;
    // timestamps recover at the next ZONE_ATOMIC, and resyncs are already counted and flagged).
    std::vector<uint64_t> cursor;
    std::vector<uint32_t> prog;      // per lane: sticky runtime host-id (every RISC emits its own at launch)
    std::vector<uint32_t> head;      // per lane: monotonic words-consumed mirror; head(N) == tail(N-1)
    std::vector<uint8_t> seeded;
    std::unordered_map<uint32_t, uint32_t> core_of_xy;  // packed (y<<16)|x -> dense core index
    uint64_t live_words = 0;
    uint64_t resync_events = 0;
    uint64_t resync_words = 0;
    uint64_t head_lag = 0;
    uint64_t anomalies = 0;  // torn run / truncated run / undecodable word
    uint64_t unknown_core_frames = 0;
    // Vector-block accounting (diagnostic): records taken by the 8-wide zone/atomic paths vs the scalar
    // fallback, and how often a would-be vector block was rejected by the per-lane type screen -- i.e. how
    // much the 2-word/3-word record mix on a stall-heavy wire is fragmenting the SIMD path.
    uint64_t vec_zone_recs = 0, vec_atomic_recs = 0, scalar_recs = 0, vec_block_rejects = 0;
    uint64_t vec_atomic_calls = 0;
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

#if defined(__AVX2__)
// Deinterleave EIGHT consecutive 3-word atomic records (24 contiguous words) into vectors of word0
// (type|id27), word1 (ts low) and word2 (duration). Output lane j belongs to record j; the stride-3
// scatter across three source vectors costs 9 lane permutes + 6 blends, against 24 scalar loads.
// ---- AVX-512 atomic block -------------------------------------------------------------------------
//
// Attributed, NOT built with -march: raising the whole build's baseline to x86-64-v4 to reach these
// intrinsics was measured as a 2.4x regression on an EVENT/DATA-interleaved profiler stream (23 -> 9.6
// GB/s), because every one of the ~450 other translation units gets different codegen too. The pragma
// below confines the ISA to this block; every function in it shares the attribute, which is what lets them
// still inline into EACH OTHER (clang only refuses to inline across DIFFERING target attributes).
//
// The signature takes PLAIN arguments on purpose: the caller is compiled for the baseline ISA and cannot
// form a __m512i, so the loads happen in here. Everything a record needs beyond its own three words --
// th, prog, lane, dev -- is block-invariant, which is what makes the 64-bit fields composable by unpack
// and the whole block layable-out in registers.
// Shuffle/permute operands as plain aligned data at file scope, so the attributed block loads them from
// L1 instead of rebuilding the vectors on every call (it cannot be inlined into a baseline-ISA caller,
// so nothing is hoisted for it).
//
// vpermt2d index rows composing output lines STRAIGHT from the packed wire. Record r's 24 B is six
// dwords {ts, th, id, meta, prog, dur} drawn from wire words {3r+1, -, 3r, -, -, 3r+2}; indices 0-15
// select the source vector's 16 wire words, 16/17/18 select th/meta/prog from the constant operand.
// Eight records = 24 words = 1.5 vectors, so the layout repeats every three lines and every 1.5 source
// vectors: rows 1-2 serve lines 1&4 and 2&5 (sources: words 8-23, then v2), row 3 is row 0 shifted by 8.
alignas(64) inline constexpr uint32_t kA16Lines[4][16] = {
    {1, 16, 0, 17, 18, 2, 4, 16, 3, 17, 18, 5, 7, 16, 6, 17},           // line 0 from v0 (words 0-15)
    {18, 0, 2, 16, 1, 17, 18, 3, 5, 16, 4, 17, 18, 6, 8, 16},           // lines 1, 4
    {7, 17, 18, 9, 11, 16, 10, 17, 18, 12, 14, 16, 13, 17, 18, 15},     // lines 2, 5
    {9, 16, 8, 17, 18, 10, 12, 16, 11, 17, 18, 13, 15, 16, 14, 17},     // line 3 from v1 (words 16-31)
};

// OFF by default, and measured that way. The 512-bit block below is correct and takes 100% of kimi's
// records, but it is not a win on this host: same-session A/B on an EVENT/DATA-interleaved synthetic gave
// 22.5 GB/s without it against 9.9 with, and it introduced 1290 order regressions on kimi that the AVX2
// path does not. Three delivery mechanisms were tried -- global -march=x86-64-v4 (which also cost 2.4x on
// the synthetic by changing codegen for all ~450 TUs), the same with the compose inlined, and this
// attribute -- and all three lost. Root-causing the regression needs a profile of the decode threads that
// this setup could not capture, so it stays gated rather than deleted.
// ---- One store discipline for the whole decode ----------------------------------------------------
//
// A ring record is 24 B, so a 16-record block is 64 B-aligned only when pos % 8 == 0. The previous version
// checked that and, when it failed, issued its stores CACHED -- and cached stores mixed with the movnti
// stores the other emitters keep issuing into the same lines cost a write-combining flush plus an RFO per
// collision (the same effect broadcast_ring.hpp::emit_store documents at ~4x). pos % 8 is driven by output
// ARITY -- PP_EVENT emits 2 records, PP_DATA emits 2+ceil(n/2) -- so an EVENT/DATA-interleaved stream lands
// misaligned ~7/8 of the time, while a stream whose atomic runs are multiples of 8 preserves the phase and
// never noticed. Measured: an identical mix with runs of 7 instead of 8 collapsed 17.9 -> 4.5 GB/s with no
// EVENT/DATA present at all.
//
// So the fix is not a better alignment test, it is removing alignment from the contract. The sub-line
// remainder lives in a register; every emission stitches carry+payload into whole 64 B lines and streams
// them. Every ring line is written exactly once, by one aligned NT store, on every stream shape -- which
// also means no cached store exists to mix with, and the alignment gate, the whole-groups-of-eight rule,
// the ring-wrap bailout and the entire AVX2 atomic tier all go away.
//
// EVERY emitter must route through this. A direct write at `pos` while the carry holds a partial line would
// be the same hazard wearing a different hat.
#pragma clang attribute push(__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq"))), apply_to = function)
struct SpscNtCarry {
    // Zero-init, NOT _mm512_setzero_si512(): a default member initializer is evaluated in the implicit
    // constructor, which the pragma above does not reach, so an intrinsic there needs a target feature
    // the constructor has not got.
    __m512i carry{};  // live qwords at lanes [8-cq, 8)
    uint32_t cq = 0;                         // qwords carried
    uint64_t line_off = 0;                   // ring byte the carry's line starts at; 64 B-aligned, < cap_bytes
    uint8_t* ring = nullptr;
    uint64_t cap_bytes = 0;                  // 24 B x a power-of-two slot count is always a multiple of 64

    // nq qwords taken from lines[0..ceil(nq/8)]; the caller must leave one extra vector readable.
    inline void put(const __m512i* lines, uint32_t nq) {
        const uint32_t total = cq + nq;
        const uint32_t L = total >> 3;
        const uint32_t r = total & 7u;
        const __m512i iota = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
        const __m512i idxA = _mm512_add_epi64(iota, _mm512_set1_epi64(8 - cq));
        uint64_t off = line_off;
        __m512i prev = carry;
        for (uint32_t k = 0; k < L; k++) {
            _mm512_stream_si512(
                reinterpret_cast<__m512i*>(ring + off), _mm512_permutex2var_epi64(prev, idxA, lines[k]));
            off += 64;
            if (off == cap_bytes) {
                off = 0;
            }
            prev = lines[k];
        }
        // One shuffle, not a window-then-shift pair: carry lane j wants (prev||lines[L])[j + r - cq], and
        // vpermt2q reads only the low index bits, so the negative indices in dead lanes wrap harmlessly.
        carry = _mm512_permutex2var_epi64(
            prev, _mm512_add_epi64(iota, _mm512_set1_epi64(static_cast<int64_t>(r) - cq)), lines[L]);
        cq = r;
        line_off = off;
    }
    inline void put3(uint64_t a, uint64_t b, uint64_t c) {  // one 24 B record
        __m512i v[2];
        v[0] = _mm512_set_epi64(
            0, 0, 0, 0, 0, static_cast<long long>(c), static_cast<long long>(b), static_cast<long long>(a));
        v[1] = v[0];
        put(v, 3);
    }
    inline void put_lines(const __m512i* lines, uint32_t bytes) { put(lines, bytes >> 3); }
    // Publish the partial line before the reader is told about those records. The bytes stay carried, so the
    // line is restreamed later with identical leading content -- benign for a reader that only looks at
    // committed slots.
    void flush_tail() {
        alignas(64) uint64_t tmp[8];
        _mm512_store_si512(tmp, carry);
        const uint64_t off = line_off;
        for (uint32_t k = 0; k < cq; k++) {
            _mm_stream_si64(reinterpret_cast<long long*>(ring + off) + k, static_cast<long long>(tmp[8 - cq + k]));
        }
        _mm_sfence();
    }
};

struct SpscA16Result {
    uint32_t n;
    uint64_t ts_first;
    uint64_t ts_last;
};

// Any n from 1 to 16, so run tails need no narrower path. Same masked loads as before; each output
// line then comes straight off the wire words through ONE two-source permute (kA16Lines), with
// th/meta/prog riding the constant operand -- no SoA deinterleave and no unpack/re-interleave stage.
inline SpscA16Result spsc_atomic16_avx512(
    const uint32_t* p,
    uint32_t avail,
    uint32_t max_recs,
    uint32_t th,
    uint32_t prog,
    uint32_t lane,
    uint32_t dev,
    SpscNtCarry& sw) {
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
    // The compare sees all 48 words, and any ts/dur word can land on the type pattern -- only every
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
    // Endpoints as scalar reloads of L1-hot source lines: unlike the spill-and-index this replaced,
    // a plain load has no store to forward from, and it competes with nothing on the shuffle ports.
    out.ts_first = th_hi | p[1];
    out.ts_last = th_hi | p[3u * n - 2u];
    const __m512i cvec = _mm512_zextsi128_si512(
        _mm_setr_epi32(static_cast<int>(th), static_cast<int>((lane << 16) | (dev << 26)), static_cast<int>(prog), 0));
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
    return out;
}

#pragma clang attribute pop

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
//     (perf_debug_ops_csv drops records with prog == 0), so an atomic zone must keep both.
// where zone_id27 is the FULL 27-bit structural zone id (tu_id << TT_ZONE_LOCAL_BITS | local --
// hostdevcommon/profiler_zone_id.h; it was a 16-bit source-location hash before, and the mask that
// truncated it here is gone),
// and for each PP_DATA/PP_EVENT
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
// the all-zone type screen -- the dominant case in a busy frame, and where the scalar walk's per-record
// cost lives. emit_atomic8(lane, timer_hi, prog, w0s, w1s, w2s, n) is the same idea for up to EIGHT
// 3-word PP_ZONE_ATOMIC records (w2s = durations): n <= 8 leading lanes are valid, the rest are
// speculative over-read and must not be emitted. Without it the atomic wire decodes entirely scalar.
// Attributed together with SpscNtCarry/spsc_atomic16_avx512 above and the emitters that call in: this
// walk inlines both, and clang will not inline across differing target attributes -- leaving it at the
// baseline turns every record into a real call (measured: knee 112 -> worse than 125).
#pragma clang attribute push(__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq"))), apply_to = function)
template <
    typename EmitMarker,
    typename EmitData,
    typename EmitProg = SpscIgnoreProg,
    typename EmitZones8 = SpscNoZones8,
    typename EmitAtomic8 = SpscNoAtomic8,
    typename EmitAtomic16 = SpscNoAtomic16>
inline uint32_t spsc_decode_frame(
    SpanDecodeState& st,
    const uint32_t* frame,
    EmitMarker&& emit,
    EmitData&& emit_data,
    EmitProg&& emit_prog = SpscIgnoreProg{},
    EmitZones8&& emit_zones8 = SpscNoZones8{},
    EmitAtomic8&& emit_atomic8 = SpscNoAtomic8{},
    EmitAtomic16&& emit_atomic16 = SpscNoAtomic16{},
    // Total words in the frame buffer. Nonzero authorizes the atomic block path to LOAD (never emit) up
    // to 24 words past a lane's live run -- the bytes exist in the frame/bounce buffer -- which lets
    // run-tails and sticky-split blocks go through the vector path with a partial count.
    uint32_t frame_words = 0) {
    (void)emit_atomic16;
    (void)emit_atomic8;  // the atomic arm is 512-bit only; the 8-wide tier is gone
    const uint32_t* ctrl = frame + kernel_profiler::SPSC_SPAN_PREFIX_WORDS;
    const auto xy_it = st.core_of_xy.find(ctrl[kernel_profiler::SPSC_CORE_XY]);
    if (xy_it == st.core_of_xy.end()) {
        st.unknown_core_frames++;
        return 0;
    }
    const uint32_t core = xy_it->second;
    const bool raw = (frame[0] & kernel_profiler::SPSC_SPAN_RAW_FLAG) != 0;
    // Frame-local counters, folded into st once at the end: the emitters store through casted ring
    // pointers, so a SpanDecodeState field update here is a forced load-add-store per record -- the
    // compiler must assume the NT stores alias it.
    uint64_t vz = 0, va = 0, vac = 0, sr = 0, lw = 0;
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
        lw += run;
        uint32_t th = st.timer_hi[lane];
        uint32_t pg = st.prog[lane];
        uint64_t cur = st.cursor[lane];
        // ONE DECODE PATH FOR BOTH FRAME LAYOUTS. A raw frame carries whole 512-word rings and is read
        // circularly; a packed frame is already contiguous. That is a difference in ADDRESSING, not in
        // decoding, and keeping two copies of the walk meant every improvement had to be made twice -- the
        // 16-wide path went into the packed copy only, so DRISC self zones (raw frames) kept decoding
        // scalar: 96 K records per stream on kimi. Linearise instead and share the walk. Raw frames are the
        // drainers' own self frames, so the copy is off the workload's path.
        uint32_t lin[kSpscRingCap];
        if (raw) {
            const uint32_t* ring = frame + kernel_profiler::SPSC_SPAN_PREFIX_WORDS +
                                   kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE + r * kSpscRingCap;
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
        // The over-read gates below vouch for words past a lane's run using the FRAME buffer's extent. A
        // linearised run lives in `lin` instead, where no such slack exists and `frame + frame_words` is not
        // even a comparable pointer -- so the frame vouch must be withdrawn here, or the gates admit reads
        // past the scratch and the decode picks up garbage records (measured: 1290 order regressions).
        [[maybe_unused]] const uint32_t fw_eff = raw ? 0u : frame_words;
        // Just-in-time prefetch of this lane's live window: small bursts consumed immediately fit the
        // core's fill-buffer budget -- issuing whole frames ahead was measured 30-40% SLOWER (the bulk
        // cold-line prefetches starve the walk's own demand loads). The second stream mirrors the same
        // offsets ONE FRAME ahead (the device DMA lands in DRAM, not cache, so the just-in-time stream
        // alone eats the full miss latency); spread across the lane walks it stays inside the same
        // fill-buffer budget the bulk variant blew.
        for (uint32_t o = 0; o < run; o += 16) {
            spsc_prefetch(p + o);
            if (!raw && frame_words != 0) {
                spsc_prefetch(p + o + frame_words);
            }
        }
        uint32_t i = 0;
        while (i < run) {
            const uint32_t w0 = p[i];
            const uint32_t t = pp_type(w0);
#if defined(__AVX2__)
            // See the ring-path copy above: screening matches 2-word markers only, so gate on the leading
            // type rather than letting an atomic stream fail the scan once per record.
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
            // THE atomic arm. Any run length from 1 up goes through here -- the block emits n records for
            // any n, so there is no tail case and no narrower path behind it.
            if constexpr (!std::is_same_v<std::decay_t<EmitAtomic16>, SpscNoAtomic16>) {
                if (t == PP_ZONE_ATOMIC) {
                    const size_t readable = fw_eff != 0 ? static_cast<size_t>(frame + fw_eff - (p + i))
                                                        : static_cast<size_t>(run - i);
                    const uint32_t got = emit_atomic16(
                        lane, th, pg, p + i, readable > 48u ? 48u : static_cast<uint32_t>(readable),
                        (run - i) / 3u);
                    if (got != 0) {
                        va += got;
                        vac++;
                        // The block is atomics only (a sticky ends it), so th is constant across it and
                        // the LAST atomic's absolute end re-anchors the lane cursor.
                        cur = pp_full_ts(th, p[i + 3u * (got - 1u) + 1u]);
                        i += 3 * got;
                        continue;
                    }
                }
            }
#endif
            // Everything reaching here is a packet the vector paths above CANNOT take, and not for want of
            // a wider instruction: PP_STICKY_TIMER redefines `th` and PP_STICKY_PROG/_EXT redefine `pg` for
            // every record after them, so a lane cannot be decoded before its predecessors are; PP_DATA
            // carries its length in its own word 2, so the next record's offset is unknown until it is read.
            // A gather walk over the width chain was built and measured for the fixed-width types instead --
            // it never fired, because a pure type never reaches this line. Measured on kimi: 99.6% of
            // records take the 16-wide path, and the remainder is one state transition per iteration.
            sr++;
            st.scalar_by_type[t & 15u]++;
            if (t == PP_EVENT) {
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
    st.scalar_recs += sr;
    st.live_words += lw;
    if (raw) {
        return kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE + kSpscNRiscDecode * kSpscRingCap;
    }
    return off - kernel_profiler::SPSC_SPAN_PREFIX_WORDS;
}
#pragma clang attribute pop

}  // namespace tt::tt_metal::profiler
