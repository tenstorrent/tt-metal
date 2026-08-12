// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared host-side decoder for the drainer `profzone` 2-word + split-sticky stream.
//
// This is the SINGLE SOURCE OF TRUTH for the host-side wire decode, so the standalone benchmark
// (the standalone drain harness) and the production RealtimeProfilerManager can never drift apart on the marker
// format (the drift -- manager decoding a stale 4-word layout while profzone emits the 2-word linearized
// stream -- is exactly what this module exists to prevent).
//
// The wire is a self-framed variable-length stream of packets (prof_packet.h):
//   STICKY_SRC   (1 word): sets the CURRENT lane (reader-injected on each source switch)
//   STICKY_TIMER (1 word): sets the current lane's wall-clock high half (timer_hi)
//   STICKY_PROG  (2 word): sets the program-global runtime host-id (prog)
//   BULKCORE     (variable): one core's NRISC sub-rings, each an inner variable-length packet run
//   marker       (2 word): ZONE_START/END/TOTAL -- emitted to the caller with its resolved lane/ts/prog
//
// D2HSocket/host pages are NOT packet-aligned, so a read can end mid-packet: the trailing partial packet is
// carried in SpscDecodeState::resid and prepended to the next call. Sticky state (cur_lane/cur_hi/prog)
// likewise persists across calls -- the stream is continuous.
#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#include "hostdevcommon/profiler_common.h"
#include "spsc_packet.h"

static_assert(
    PP_BULK_SPAN == kernel_profiler::SPSC_SPAN_PACKET_TYPE,
    "prof_packet.h (plain C, drainer firmware) and profiler_common.h (C++, metal kernels) must agree on the "
    "BULK_SPAN wire code -- they cannot include each other, so this is the only thing holding them together");

namespace tt::tt_metal::profiler {

// Worker per-RISC SPSC ring depth (words) and RISC count -- MUST match the producer (kernel_profiler.hpp
// RING_CAPACITY / profstream.c) so the BULKCORE sub-ring walk indexes correctly.
inline constexpr uint32_t kSpscRingCap = 512;
inline constexpr uint32_t kSpscNRiscDecode = 5;

// Sticky/framing state carried ACROSS decode calls for one continuous stream (one D2HSocket / host ring).
struct SpscDecodeState {
    uint32_t cur_lane = 0xFFFFFFFFu;  // set by STICKY_SRC
    uint32_t cur_prog = 0;            // set by STICKY_PROG (program-global)
    std::vector<uint32_t> cur_hi;     // per-lane wall-clock high half (set by STICKY_TIMER), size = nl
    std::vector<uint32_t> resid;      // trailing partial packet carried to the next call
    // BULK_SPAN identity: packed (y << 16) | x -> dense core index. Hot path is a flat grid indexed by
    // virtual (x,y) -- never a hash map. BH virtual coords are ~0..21; 32 covers with margin.
    static constexpr uint32_t kXyGrid = 32;
    uint32_t xy_core[kXyGrid * kXyGrid];  // UINT32_MAX = empty
    // Host-side head mirror, per lane. The drainer stopped patching heads into the frame (5 stores per core
    // per visit); the host reconstructs them instead: head(frame N) == tail(frame N-1) for that lane, which
    // is exact because the D2H FIFO is ordered and lossless. The span's own head field still arrives free
    // inside the control vector, so it seeds this on first sight and thereafter serves as a CONSISTENCY
    // CHECK -- `head_drift` counts disagreements rather than silently trusting either side.
    std::vector<uint32_t> host_head;
    std::vector<uint8_t> head_seeded;
    uint64_t head_drift = 0;

    SpscDecodeState() { clear_xy_core(); }

    void clear_xy_core() {
        for (uint32_t& c : xy_core) {
            c = 0xFFFFFFFFu;
        }
    }

    void reset(uint32_t nl) {
        cur_lane = 0xFFFFFFFFu;
        cur_prog = 0;
        cur_hi.assign(nl, 0);
        resid.clear();
        host_head.assign(nl, 0);
        head_seeded.assign(nl, 0);
        head_drift = 0;
        // xy_core is identity -- rebuilt at boot, not cleared here.
    }

    // Boot-time only. Returns false if (x,y) exceeds kXyGrid -- caller must TT_FATAL (part too big).
    bool set_core_xy(uint32_t xy, uint32_t core) {
        const uint32_t x = xy & 0xFFFFu;
        const uint32_t y = xy >> 16;
        if (x >= kXyGrid || y >= kXyGrid) {
            return false;
        }
        xy_core[y * kXyGrid + x] = core;
        return true;
    }

    // UINT32_MAX if unknown. Hot path: one indexed load, no hash.
    uint32_t lookup_core(uint32_t xy) const {
        const uint32_t x = xy & 0xFFFFu;
        const uint32_t y = xy >> 16;
        if (x >= kXyGrid || y >= kXyGrid) [[unlikely]] {
            return 0xFFFFFFFFu;
        }
        return xy_core[y * kXyGrid + x];
    }
};

// Decode `in[0..in_n)` (a fresh read), prepending any carried residual. For each MARKER packet, calls
//   emit(uint32_t lane, uint32_t type, uint32_t zone_hash, uint64_t full_ts, uint32_t prog)
// where type is PP_ZONE_START/END/TOTAL, zone_hash is the low-16 srcloc hash, and full_ts is the 59-bit
// device timestamp (timer_hi<<32 | timer_low). Sticky packets update `st` and are not emitted. A trailing
// partial packet is saved into st.resid for the next call. `nl` = number of lanes (num_cores * NRISC).
// No-op default for the optional drainer hart-zone sink (see the PP_drainer_ZONE branch below).
struct ProfzoneIgnoredrainer {
    void operator()(uint32_t /*hart*/, uint32_t /*meta*/, uint64_t /*rdcycle*/) const {}
};

// No-op default for the point-marker sink (PP_DATA / PP_EVENT), so a caller that only wants zones compiles
// unchanged. `type` is the wire type: PP_DATA ids are compile-time tags and can be name-resolved, PP_EVENT
// ids are runtime values and must NOT be.
struct SpscIgnoreData {
    void operator()(
        uint32_t /*lane*/,
        uint32_t /*type*/,
        uint32_t /*id*/,
        uint64_t /*full_ts*/,
        uint32_t /*prog*/,
        const uint32_t* /*payload*/,
        uint32_t /*n*/) const {}
};

// Largest PP_DATA payload the 7-bit size field can express; bounds the ring-unwrap scratch buffer.
inline constexpr uint32_t kSpscMaxDataWords = 127;

#if defined(__x86_64__)
inline bool spsc_have_avx2() {
    static const bool have = __builtin_cpu_supports("avx2") != 0;
    return have;
}

// Count the leading words of `lin` that are PURE 2-word zone markers, in 16-word (8-packet) blocks.
// A block is accepted iff all 8 even-offset words have type <= ZONE_END. This is alignment-safe with
// no wire change: a rare packet (1-word STICKY_TIMER, DATA, PROG) can only begin at a packet boundary,
// and every boundary inside a clean prefix is even -- so the packet that WOULD shift alignment is
// itself screened out first, and a payload word is never misread as a packet header.
__attribute__((target("avx2"))) inline uint32_t spsc_screen_zone_block(const uint32_t* lin, uint32_t n) {
    const __m256i one = _mm256_set1_epi32(1);
    uint32_t i = 0;
    while (i + 16 <= n) {
        const __m256i a = _mm256_loadu_si256(static_cast<const __m256i*>(static_cast<const void*>(lin + i)));
        const __m256i b = _mm256_loadu_si256(static_cast<const __m256i*>(static_cast<const void*>(lin + i + 8)));
        const uint32_t bad =
            static_cast<uint32_t>(
                _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_srli_epi32(a, 27), one)))) |
            (static_cast<uint32_t>(
                 _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_srli_epi32(b, 27), one))))
             << 8);
        if ((bad & 0x5555u) != 0) {
            break;
        }
        i += 16;
    }
    return i;
}
#endif

// Walk a CONTIGUOUS packet run, dispatching zone markers on the first branch (they are ~100% of the
// stream; every rare type used to cost them a failed compare). Returns words consumed: the walk stops
// before a packet that extends past `n`, so a caller decoding a wrapped ring can stitch the straddling
// packet and continue. timer_hi / prog are the caller's hoisted sticky registers.
//
// `emit_block(p, nwords, lane, hi, prog)` receives each screened ALL-MARKER block (nwords a multiple
// of 16) so a sink can materialize records with SIMD instead of a per-marker call; the scalar overload
// below adapts it back to per-marker emit.
template <typename Emit, typename EmitData, typename EmitBlock>
inline uint32_t spsc_walk_run(
    const uint32_t* lin,
    uint32_t n,
    uint32_t lane,
    uint32_t& hi,
    uint32_t& prog,
    Emit&& emit,
    EmitData&& emit_data,
    EmitBlock&& emit_block) {
    uint32_t i = 0;
    while (i + 1 < n) {
#if defined(__x86_64__)
        if (spsc_have_avx2()) {
            const uint32_t blk = spsc_screen_zone_block(lin + i, n - i);
            if (blk != 0) {
                emit_block(lin + i, blk, lane, hi, prog);
                i += blk;
            }
            if (i + 1 >= n) {
                break;
            }
        }
#endif
        const uint32_t rw0 = lin[i];
        const uint32_t rw1 = lin[i + 1];
        const uint32_t t = pp_type(rw0);
        if (t <= PP_ZONE_END) [[likely]] {
            emit(lane, t, rw0 & 0xFFFFu, pp_full_ts(hi, rw1), prog);
            i += 2;
            continue;
        }
        if (t == PP_STICKY_TIMER) {
            hi = pp_timer_hi(rw0);
            i += 1;
            continue;
        }
        if (t == PP_DATA || t == PP_EVENT) {
            const uint32_t sz = pp_data_size(rw0);
            if (i + 2u + sz > n) {
                break;
            }
            emit_data(lane, t, pp_data_id(rw0), pp_full_ts(hi, rw1), prog, &lin[i + 2], sz);
            i += 2u + sz;
            continue;
        }
        if (t == PP_STICKY_PROG) {
            prog = rw1;
        } else {
            emit(lane, t, rw0 & 0xFFFFu, pp_full_ts(hi, rw1), prog);
        }
        i += 2;
    }
    if (i < n && pp_type(lin[i]) == PP_STICKY_TIMER) {
        hi = pp_timer_hi(lin[i]);
        i += 1;
    }
    return i;
}

// Scalar-sink overload: adapts screened blocks back to per-marker emit.
template <typename Emit, typename EmitData>
inline uint32_t spsc_walk_run(
    const uint32_t* lin, uint32_t n, uint32_t lane, uint32_t& hi, uint32_t& prog, Emit&& emit, EmitData&& emit_data) {
    return spsc_walk_run(
        lin,
        n,
        lane,
        hi,
        prog,
        emit,
        emit_data,
        [&emit](const uint32_t* p, uint32_t nw, uint32_t bl, uint32_t bhi, uint32_t bprog) {
            for (uint32_t j = 0; j < nw; j += 2) {
                emit(bl, p[j] >> 27, p[j] & 0xFFFFu, pp_full_ts(bhi, p[j + 1]), bprog);
            }
        });
}

// Walk one RISC's live circular run as a flat packet stream. Contiguous runs decode in place. Wrapping
// runs decode as two in-place segments with only the straddling packet stitched through `scratch`
// (needs >= 2 + kSpscMaxDataWords words) -- at near-full rings almost every run wraps, so unwrapping
// the whole run was a memcpy of nearly the entire marker stream. timer_hi / prog are hoisted into
// registers for the run.
template <typename Emit, typename EmitData, typename EmitBlock>
inline void spsc_decode_ring_run(
    SpscDecodeState& st,
    const uint32_t* ring,
    uint32_t head_mod,
    uint32_t run,
    uint32_t lane,
    uint32_t nl,
    uint32_t* scratch,
    Emit&& emit,
    EmitData&& emit_data,
    EmitBlock&& emit_block) {
    if (run == 0 || lane >= nl) {
        return;
    }

    uint32_t hi = st.cur_hi[lane];
    uint32_t prog = st.cur_prog;
    if (head_mod + run <= kSpscRingCap) {
        spsc_walk_run(ring + head_mod, run, lane, hi, prog, emit, emit_data, emit_block);
    } else {
        const uint32_t first = kSpscRingCap - head_mod;
        const uint32_t c1 = spsc_walk_run(ring + head_mod, first, lane, hi, prog, emit, emit_data, emit_block);
        // Producer publishes tail only after a whole packet is in the ring, so the run ends on a packet
        // boundary and a packet cut by the wrap always has its remainder at the ring base.
        const uint32_t rem = first - c1;
        uint32_t second_off = 0;
        if (rem != 0) {
            constexpr uint32_t stitch_cap = 2u + kSpscMaxDataWords;
            const uint32_t second = run - first;
            const uint32_t take = second < stitch_cap - rem ? second : stitch_cap - rem;
            std::memcpy(scratch, ring + head_mod + c1, rem * sizeof(uint32_t));
            std::memcpy(scratch + rem, ring, take * sizeof(uint32_t));
            const uint32_t c2 = spsc_walk_run(scratch, rem + take, lane, hi, prog, emit, emit_data, emit_block);
            second_off = c2 - rem;
        }
        spsc_walk_run(ring + second_off, run - first - second_off, lane, hi, prog, emit, emit_data, emit_block);
    }
    st.cur_hi[lane] = hi;
    st.cur_prog = prog;
}

// Whole-packet word count of a top-level stream packet from its first two words (w1 is ignored for
// 1-word types). BULK frames need w1, so the caller must have 2 words before trusting the result.
inline size_t spsc_top_packet_words(uint32_t w0, uint32_t w1) {
    if (pp_is_bulkcore(w0)) {
        uint32_t prefix = 2u + kSpscNRiscDecode;
        if (prefix & 1u) {
            prefix++;
        }
        return static_cast<size_t>(prefix) + w1;
    }
    if (pp_is_bulkspan(w0)) {
        return kernel_profiler::spsc_span_frame_words(w1);
    }
    if (pp_is_src(w0) || pp_is_timer(w0)) {
        return 1;
    }
    if (pp_is_point(w0)) {
        return 2u + pp_data_size(w0);
    }
    return 2;
}

template <typename Emit, typename EmitData, typename EmitBlock>
inline void spsc_decode(
    SpscDecodeState& st,
    const uint32_t* in,
    size_t in_n,
    uint32_t nl,
    Emit&& emit,
    EmitData&& emit_data,
    EmitBlock&& emit_block) {
    // Scratch for wrapping ring runs (shared across RISCs / frames in this call).
    uint32_t ring_scratch[kSpscRingCap];

    // Decode a contiguous span of whole packets; returns words consumed (stops before a partial packet).
    auto decode_span = [&](const uint32_t* w, size_t sz) -> size_t {
        size_t p = 0;
        while (p < sz) {
            const uint32_t w0 = w[p];
            if (pp_is_bulkcore(w0)) {
                if (p + 1 >= sz) {
                    break;  // need the count word
                }
                const uint32_t core = pp_bulkcore_core(w0);
                const uint32_t rawn = w[p + 1];
                uint32_t prefix = 2u + kSpscNRiscDecode;  // {w0, count} + per-RISC {head,run} meta
                if (prefix & 1u) {
                    prefix++;  // meta padded to an even word count (matches the producer framing)
                }
                if (p + prefix + rawn > sz) {
                    break;  // incomplete bulk block -> carry to next call
                }
                const uint32_t* meta = &w[p + 2];
                const uint32_t* raw = &w[p + prefix];
                for (uint32_t r = 0; r < kSpscNRiscDecode; r++) {
                    const uint32_t head_mod = pp_bulk_head(meta[r]);
                    const uint32_t run = pp_bulk_run(meta[r]);
                    const uint32_t lane = core * kSpscNRiscDecode + r;
                    const uint32_t* ring = raw + (size_t)r * kSpscRingCap;
                    spsc_decode_ring_run(st, ring, head_mod, run, lane, nl, ring_scratch, emit, emit_data, emit_block);
                }
                p += prefix + rawn;
            } else if (pp_is_bulkspan(w0)) {
                // Identity-free whole-core frame. Everything BULK_CORE puts in a drainer-written header --
                // which core, how much is live -- is re-derived here from the worker's OWN control vector.
                // w0 bit 0 (SPSC_SPAN_PACKED_FLAG) selects the payload layout:
                //   set   -- each live ring's window, packed: a pad to the next 16 B frame offset, then
                //            `head & 3` lead words (the 16 B floor the filler had to ship from, see the
                //            NoC write-congruence note in stage_frame), then the run, UNWRAPPED. Empty
                //            rings contribute nothing. Everything here is re-derived from head/tail, so
                //            the walk below must mirror the filler's layout computation exactly.
                //   clear -- five WHOLE raw rings (the full-job drainer is a conduit and never repacks;
                //            a CPU repack cost it 45% of its cycles); the live window is the circular
                //            range [head, head+run) and the wrap is resolved per run in the walk.
                if (p + 1 >= sz) {
                    break;  // need the length word
                }
                const uint32_t frame = kernel_profiler::spsc_span_frame_words(w[p + 1]);
                if (p + frame > sz) {
                    break;  // incomplete frame -> carry to the next call
                }
                const bool packed = (w0 & kernel_profiler::SPSC_SPAN_PACKED_FLAG) != 0;
                const uint32_t* ctrl = &w[p + kernel_profiler::SPSC_SPAN_PREFIX_WORDS];
                const uint32_t* blk = ctrl + kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE;
                const uint32_t core = st.lookup_core(ctrl[kernel_profiler::SPSC_CORE_XY]);
                // A frame whose core is unknown (or whose lanes fall outside nl) is skipped WHOLE: a
                // packed payload cannot be advanced ring-by-ring without per-lane head state.
                if (core == 0xFFFFFFFFu || (core + 1) * kSpscNRiscDecode > nl) {
                    p += frame;
                    continue;
                }
                for (uint32_t r = 0; r < kSpscNRiscDecode; r++) {
                    const uint32_t tail = ctrl[kernel_profiler::SPSC_RING_TAIL_0 + r];
                    const uint32_t lane = core * kSpscNRiscDecode + r;
                    if (!st.head_seeded[lane]) {
                        st.host_head[lane] = ctrl[kernel_profiler::SPSC_RING_HEAD_0 + r];
                        st.head_seeded[lane] = 1;
                    }
                    // Head-drift check is diagnostic only (ORDER_CHECK era). The mirror is authoritative;
                    // comparing every frame on the packed hot path was pure host tax.
                    const uint32_t head = st.host_head[lane];
                    const uint32_t run = kernel_profiler::spsc_span_live(head, tail, kSpscRingCap);
                    st.host_head[lane] = head + run;
                    if (packed) {
                        if (run == 0) {
                            continue;
                        }
                        const uint32_t off = static_cast<uint32_t>(blk - &w[p]);
                        blk += ((4u - (off & 3u)) & 3u) + (head & 3u);  // dst pad + shipped lead words
                        // Packed runs are already unwrapped -- head_mod 0, no ring wrap stitch.
                        spsc_decode_ring_run(st, blk, 0, run, lane, nl, ring_scratch, emit, emit_data, emit_block);
                        blk += run;
                    } else {
                        spsc_decode_ring_run(
                            st, blk, head % kSpscRingCap, run, lane, nl, ring_scratch, emit, emit_data, emit_block);
                        blk += kSpscRingCap;
                    }
                }
                p += frame;
            } else if (pp_is_src(w0)) {  // 1-word: set the current lane
                st.cur_lane = pp_src_lane(w0);
                p += 1;
            } else if (pp_is_timer(w0)) {  // 1-word: refresh the current lane's timer_hi
                if (st.cur_lane < nl) {
                    st.cur_hi[st.cur_lane] = pp_timer_hi(w0);
                }
                p += 1;
            } else if (pp_is_point(w0)) {
                // 2 + size words: the unified EVENT/DATA packet (size 0 == a bare event). Self-describing
                // length, so an unknown payload shape can never desynchronize the walk.
                if (p + 1 >= sz) {
                    break;  // need the timestamp word to know we have the whole header
                }
                const uint32_t n = pp_data_size(w0);
                if (p + 2 + n > sz) {
                    break;  // partial payload -> carry
                }
                if (st.cur_lane < nl) {
                    const uint64_t ts = pp_full_ts(st.cur_hi[st.cur_lane], w[p + 1]);
                    emit_data(st.cur_lane, pp_type(w0), pp_data_id(w0), ts, st.cur_prog, &w[p + 2], n);
                }
                p += 2 + n;
            } else {  // 2-word: STICKY_PROG or a marker
                if (p + 1 >= sz) {
                    break;  // partial marker -> carry
                }
                const uint32_t w1 = w[p + 1];
                if (pp_type(w0) == PP_STICKY_PROG) {
                    st.cur_prog = w1;
                } else if (st.cur_lane < nl) {
                    const uint32_t hash = pp_low27(w0) & 0xFFFFu;
                    const uint64_t ts = pp_full_ts(st.cur_hi[st.cur_lane], w1);
                    emit(st.cur_lane, pp_type(w0), hash, ts, st.cur_prog);
                }
                p += 2;
            }
        }
        return p;
    };

    // The carried partial packet decodes via a SMALL stitch: top it up from `in` to exactly one whole
    // packet (bounded by one BULK frame), decode it alone, then decode the rest of `in` IN PLACE. The
    // old path appended the WHOLE read to resid whenever a partial frame was in flight -- i.e. on
    // nearly every call, silently re-copying the entire stream inside "decode".
    std::vector<uint32_t>& buf = st.resid;
    size_t in_off = 0;
    if (!buf.empty()) {
        for (;;) {
            if (buf.size() == 1) {
                const uint32_t w0 = buf[0];
                if (pp_is_src(w0) || pp_is_timer(w0)) {
                    break;
                }
                if (in_off >= in_n) {
                    return;
                }
                buf.push_back(in[in_off++]);
                continue;
            }
            const size_t need = spsc_top_packet_words(buf[0], buf[1]);
            if (buf.size() >= need) {
                break;
            }
            const size_t take = std::min(need - buf.size(), in_n - in_off);
            if (take == 0) {
                return;  // `in` exhausted; the packet stays carried
            }
            buf.insert(buf.end(), in + in_off, in + in_off + take);
            in_off += take;
        }
        decode_span(buf.data(), buf.size());  // exactly one whole packet by construction
        buf.clear();
    }

    const size_t p = in_off + decode_span(in + in_off, in_n - in_off);
    if (p < in_n) {
        buf.assign(in + p, in + in_n);  // trailing partial packet only
    }
}

// Scalar-sink overload: screened marker blocks fall back to per-marker emit.
template <typename Emit, typename EmitData = SpscIgnoreData>
inline void spsc_decode(
    SpscDecodeState& st,
    const uint32_t* in,
    size_t in_n,
    uint32_t nl,
    Emit&& emit,
    EmitData&& emit_data = SpscIgnoreData{}) {
    spsc_decode(
        st,
        in,
        in_n,
        nl,
        emit,
        emit_data,
        [&emit](const uint32_t* p, uint32_t nw, uint32_t bl, uint32_t bhi, uint32_t bprog) {
            for (uint32_t j = 0; j < nw; j += 2) {
                emit(bl, p[j] >> 27, p[j] & 0xFFFFu, pp_full_ts(bhi, p[j + 1]), bprog);
            }
        });
}

}  // namespace tt::tt_metal::profiler
