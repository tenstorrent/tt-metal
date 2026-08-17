// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared host-side decoder for the DRISC drainer's 2-word + split-sticky stream
// (producer: tt_metal/tools/profiler/kernels/drisc_profiler_drain.cpp).
//
// This is the SINGLE SOURCE OF TRUTH for the host-side wire decode, so the standalone benchmark
// (the standalone drain harness) and the production RealtimeProfilerManager can never drift apart on the marker
// format (the drift -- manager decoding a stale 4-word layout while the drainer emits the 2-word
// linearized stream -- is exactly what this module exists to prevent).
//
// The wire is a self-framed variable-length stream of packets (spsc_packet.h):
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
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <vector>

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

namespace tt::tt_metal::profiler {

// Worker per-RISC SPSC ring depth (words) and RISC count -- MUST match the producer (kernel_profiler.hpp
// RING_CAPACITY, = kernel_profiler::PROFILER_L1_VECTOR_SIZE) so the BULKCORE sub-ring walk indexes
// correctly.
inline constexpr uint32_t kSpscRingCap = 512;
inline constexpr uint32_t kSpscNRiscDecode = 5;

// Sticky/framing state carried ACROSS decode calls for one continuous stream (one D2HSocket / host ring).
struct SpscDecodeState {
    uint32_t cur_lane = 0xFFFFFFFFu;  // set by STICKY_SRC
    uint32_t cur_prog = 0;            // set by STICKY_PROG (program-global)
    std::vector<uint32_t> cur_hi;     // per-lane wall-clock high half (set by STICKY_TIMER), size = nl
    std::vector<uint32_t> resid;      // trailing partial packet carried to the next call
    // BULK_SPAN identity: packed (y << 16) | x -> dense core index, so lane = core*NRISC + risc keeps its
    // meaning downstream. The caller owns this map because only the host knows the grid; the drainer never
    // sees it and never puts a core id on the wire. A frame whose xy is absent is skipped whole.
    std::unordered_map<uint32_t, uint32_t> core_of_xy;
    // Host-side head mirror, per lane. The drainer stopped patching heads into the frame (5 stores per core
    // per visit); the host reconstructs them instead: head(frame N) == tail(frame N-1) for that lane, which
    // is exact because the D2H FIFO is ordered and lossless. The span's own head field still arrives free
    // inside the control vector, so it seeds this on first sight and thereafter serves as a CONSISTENCY
    // CHECK -- `head_drift` counts disagreements rather than silently trusting either side.
    std::vector<uint32_t> host_head;
    std::vector<uint8_t> head_seeded;
    uint64_t head_drift = 0;

    void reset(uint32_t nl) {
        cur_lane = 0xFFFFFFFFu;
        cur_prog = 0;
        cur_hi.assign(nl, 0);
        resid.clear();
        host_head.assign(nl, 0);
        head_seeded.assign(nl, 0);
        head_drift = 0;
    }
};

// Decode `in[0..in_n)` (a fresh read), prepending any carried residual. For each MARKER packet, calls
//   emit(uint32_t lane, uint32_t type, uint32_t zone_hash, uint64_t full_ts, uint32_t prog)
// where type is PP_ZONE_START/END/TOTAL, zone_hash is the low-16 srcloc hash, and full_ts is the 59-bit
// device timestamp (timer_hi<<32 | timer_low). Sticky packets update `st` and are not emitted. A trailing
// partial packet is saved into st.resid for the next call. `nl` = number of lanes (num_cores * NRISC).
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

template <typename Emit, typename EmitData = SpscIgnoreData>
inline void spsc_decode(
    SpscDecodeState& st,
    const uint32_t* in,
    size_t in_n,
    uint32_t nl,
    Emit&& emit,
    EmitData&& emit_data = SpscIgnoreData{}) {
    // Prepend the carried residual so packets that straddled the previous read are decoded whole.
    std::vector<uint32_t>& buf = st.resid;
    const size_t rn = buf.size();
    buf.resize(rn + in_n);
    for (size_t i = 0; i < in_n; i++) {
        buf[rn + i] = in[i];
    }
    const size_t sz = buf.size();
    const uint32_t* w = buf.data();

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
                uint32_t i = 0;
                while (i < run) {
                    const uint32_t rw0 = ring[(head_mod + i) % kSpscRingCap];
                    if (pp_is_timer(rw0)) {  // 1-word: refresh this lane's timer_hi
                        if (lane < nl) {
                            st.cur_hi[lane] = pp_timer_hi(rw0);
                        }
                        i += 1;
                        continue;
                    }
                    if (i + 1 >= run) {
                        break;  // partial trailing marker inside the run (shouldn't happen on a full frame)
                    }
                    const uint32_t rw1 = ring[(head_mod + i + 1) % kSpscRingCap];
                    if (pp_is_point(rw0)) {
                        // PP_DATA is VARIABLE length (2 + size). Its length is in the header, so the walk
                        // stays in sync without a per-type table -- the whole point of the unified packet.
                        const uint32_t n = pp_data_size(rw0);
                        if (i + 2u + n > run) {
                            break;  // payload not fully inside this frame -> carry via the next head
                        }
                        if (lane < nl) {
                            // The payload can wrap the circular ring, so unwrap it into a flat scratch buffer
                            // before handing it to the sink.
                            uint32_t payload[kSpscMaxDataWords];
                            for (uint32_t k = 0; k < n; k++) {
                                payload[k] = ring[(head_mod + i + 2 + k) % kSpscRingCap];
                            }
                            const uint64_t ts = pp_full_ts(st.cur_hi[lane], rw1);
                            emit_data(lane, pp_type(rw0), pp_data_id(rw0), ts, st.cur_prog, payload, n);
                        }
                        i += 2u + n;
                        continue;
                    }
                    if (pp_type(rw0) == PP_STICKY_PROG) {
                        st.cur_prog = rw1;
                    } else if (lane < nl) {
                        const uint32_t hash = pp_low27(rw0) & 0xFFFFu;
                        const uint64_t ts = pp_full_ts(st.cur_hi[lane], rw1);
                        emit(lane, pp_type(rw0), hash, ts, st.cur_prog);
                    }
                    i += 2;
                }
            }
            p += prefix + rawn;
        } else if (pp_is_bulkspan(w0)) {
            // Identity-free whole-core frame. Everything BULK_CORE puts in a drainer-written header --
            // which core, how much is live -- is re-derived here from the worker's OWN control vector.
            // Payload is each RISC's live run, packed exactly and already unwrapped device-side.
            if (p + 1 >= sz) {
                break;  // need the length word
            }
            const uint32_t frame = kernel_profiler::spsc_span_frame_words(w[p + 1]);
            if (p + frame > sz) {
                break;  // incomplete frame -> carry to the next call
            }
            const uint32_t* ctrl = &w[p + kernel_profiler::SPSC_SPAN_PREFIX_WORDS];
            const uint32_t* blk = ctrl + kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE;
            const auto xy_it = st.core_of_xy.find(ctrl[kernel_profiler::SPSC_CORE_XY]);
            const bool known = xy_it != st.core_of_xy.end();
            for (uint32_t r = 0; r < kSpscNRiscDecode; r++) {
                // The payload is five WHOLE raw rings -- the drainer is a conduit and never repacked them
                // (a CPU repack cost it 45% of its cycles). So the live window is the circular range
                // [head, head+run) and the wrap is resolved here, on a host that has cycles to spare.
                const uint32_t tail = ctrl[kernel_profiler::SPSC_RING_TAIL_0 + r];
                const uint32_t* ring = blk;
                blk += kSpscRingCap;  // rings are fixed-size and in RISC order, present even when empty
                const uint32_t lane = known ? xy_it->second * kSpscNRiscDecode + r : nl;
                if (lane >= nl) {
                    continue;
                }
                if (!st.head_seeded[lane]) {
                    st.host_head[lane] = ctrl[kernel_profiler::SPSC_RING_HEAD_0 + r];
                    st.head_seeded[lane] = 1;
                } else if (ctrl[kernel_profiler::SPSC_RING_HEAD_0 + r] != st.host_head[lane]) {
                    // Benign if the drainer's write-back was still in flight when the snapshot was taken;
                    // a real signal if a frame went missing. Counted, never trusted over the mirror.
                    st.head_drift++;
                }
                const uint32_t head = st.host_head[lane];
                const uint32_t run = kernel_profiler::spsc_span_live(head, tail, kSpscRingCap);
                st.host_head[lane] = head + run;
                const uint32_t head_mod = head % kSpscRingCap;
                if (run == 0) {
                    continue;
                }
                uint32_t i = 0;
                while (i < run) {
                    const uint32_t rw0 = ring[(head_mod + i) % kSpscRingCap];
                    if (pp_is_timer(rw0)) {  // 1-word: refresh this lane's timer_hi
                        st.cur_hi[lane] = pp_timer_hi(rw0);
                        i += 1;
                        continue;
                    }
                    // The producer publishes its tail only after a whole packet is in the ring, so a run
                    // never ends mid-packet. These bounds checks are assertions, not recovery.
                    if (i + 1 >= run) {
                        break;
                    }
                    const uint32_t rw1 = ring[(head_mod + i + 1) % kSpscRingCap];
                    if (pp_is_point(rw0)) {
                        const uint32_t n = pp_data_size(rw0);
                        if (i + 2u + n > run) {
                            break;
                        }
                        // The payload can wrap the ring, so unwrap it into a flat scratch buffer first.
                        uint32_t payload[kSpscMaxDataWords];
                        for (uint32_t k = 0; k < n && k < kSpscMaxDataWords; k++) {
                            payload[k] = ring[(head_mod + i + 2 + k) % kSpscRingCap];
                        }
                        emit_data(
                            lane,
                            pp_type(rw0),
                            pp_data_id(rw0),
                            pp_full_ts(st.cur_hi[lane], rw1),
                            st.cur_prog,
                            payload,
                            n);
                        i += 2u + n;
                        continue;
                    }
                    if (pp_type(rw0) == PP_STICKY_PROG) {
                        st.cur_prog = rw1;
                    } else {
                        emit(
                            lane, pp_type(rw0), pp_low27(rw0) & 0xFFFFu, pp_full_ts(st.cur_hi[lane], rw1), st.cur_prog);
                    }
                    i += 2;
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

    // Carry the trailing partial packet (if any) to the next call.
    if (p < sz) {
        buf.erase(buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(p));
    } else {
        buf.clear();
    }
}

}  // namespace tt::tt_metal::profiler
