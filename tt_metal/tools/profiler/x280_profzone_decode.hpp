// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared host-side decoder for the X280 `profzone` 2-word + split-sticky stream.
//
// This is the SINGLE SOURCE OF TRUTH for the host-side wire decode, so the standalone benchmark
// (test_x280_realprof) and the production RealtimeProfilerManager can never drift apart on the marker
// format (the drift -- manager decoding a stale 4-word layout while profzone emits the 2-word linearized
// stream -- is exactly what this module exists to prevent, mirroring x280_profzone_boot.hpp for the boot).
//
// The wire is a self-framed variable-length stream of packets (prof_packet.h):
//   STICKY_SRC   (1 word): sets the CURRENT lane (reader-injected on each source switch)
//   STICKY_TIMER (1 word): sets the current lane's wall-clock high half (timer_hi)
//   STICKY_PROG  (2 word): sets the program-global runtime host-id (prog)
//   BULKCORE     (variable): one core's NRISC sub-rings, each an inner variable-length packet run
//   marker       (2 word): ZONE_START/END/TOTAL -- emitted to the caller with its resolved lane/ts/prog
//
// D2HSocket/host pages are NOT packet-aligned, so a read can end mid-packet: the trailing partial packet is
// carried in ProfzoneDecodeState::resid and prepended to the next call. Sticky state (cur_lane/cur_hi/prog)
// likewise persists across calls -- the stream is continuous.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "prof_packet.h"

namespace tt::tt_metal::profiler {

// Worker per-RISC SPSC ring depth (words) and RISC count -- MUST match the producer (kernel_profiler.hpp
// RING_CAPACITY / profstream.c) so the BULKCORE sub-ring walk indexes correctly.
inline constexpr uint32_t kProfzoneRingCap = 512;
inline constexpr uint32_t kProfzoneNRiscDecode = 5;

// Sticky/framing state carried ACROSS decode calls for one continuous stream (one D2HSocket / host ring).
struct ProfzoneDecodeState {
    uint32_t cur_lane = 0xFFFFFFFFu;  // set by STICKY_SRC
    uint32_t cur_prog = 0;            // set by STICKY_PROG (program-global)
    std::vector<uint32_t> cur_hi;     // per-lane wall-clock high half (set by STICKY_TIMER), size = nl
    std::vector<uint32_t> resid;      // trailing partial packet carried to the next call

    void reset(uint32_t nl) {
        cur_lane = 0xFFFFFFFFu;
        cur_prog = 0;
        cur_hi.assign(nl, 0);
        resid.clear();
    }
};

// Decode `in[0..in_n)` (a fresh read), prepending any carried residual. For each MARKER packet, calls
//   emit(uint32_t lane, uint32_t type, uint32_t zone_hash, uint64_t full_ts, uint32_t prog)
// where type is PP_ZONE_START/END/TOTAL, zone_hash is the low-16 srcloc hash, and full_ts is the 59-bit
// device timestamp (timer_hi<<32 | timer_low). Sticky packets update `st` and are not emitted. A trailing
// partial packet is saved into st.resid for the next call. `nl` = number of lanes (num_cores * NRISC).
template <typename Emit>
inline void profzone_decode(ProfzoneDecodeState& st, const uint32_t* in, size_t in_n, uint32_t nl, Emit&& emit) {
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
            uint32_t prefix = 2u + kProfzoneNRiscDecode;  // {w0, count} + per-RISC {head,run} meta
            if (prefix & 1u) {
                prefix++;  // meta padded to an even word count (matches the producer framing)
            }
            if (p + prefix + rawn > sz) {
                break;  // incomplete bulk block -> carry to next call
            }
            const uint32_t* meta = &w[p + 2];
            const uint32_t* raw = &w[p + prefix];
            for (uint32_t r = 0; r < kProfzoneNRiscDecode; r++) {
                const uint32_t head_mod = pp_bulk_head(meta[r]);
                const uint32_t run = pp_bulk_run(meta[r]);
                const uint32_t lane = core * kProfzoneNRiscDecode + r;
                const uint32_t* ring = raw + (size_t)r * kProfzoneRingCap;
                uint32_t i = 0;
                while (i < run) {
                    const uint32_t rw0 = ring[(head_mod + i) % kProfzoneRingCap];
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
                    const uint32_t rw1 = ring[(head_mod + i + 1) % kProfzoneRingCap];
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
        } else if (pp_is_src(w0)) {  // 1-word: set the current lane
            st.cur_lane = pp_src_lane(w0);
            p += 1;
        } else if (pp_is_timer(w0)) {  // 1-word: refresh the current lane's timer_hi
            if (st.cur_lane < nl) {
                st.cur_hi[st.cur_lane] = pp_timer_hi(w0);
            }
            p += 1;
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
