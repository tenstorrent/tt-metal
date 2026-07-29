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
// No-op default for the optional X280 hart-zone sink (see the PP_X280_ZONE branch below).
struct ProfzoneIgnoreX280 {
    void operator()(uint32_t /*hart*/, uint32_t /*meta*/, uint64_t /*rdcycle*/) const {}
};

// No-op default for the point-marker sink (PP_DATA / PP_EVENT), so a caller that only wants zones compiles
// unchanged. `type` is the wire type: PP_DATA ids are compile-time tags and can be name-resolved, PP_EVENT
// ids are runtime values and must NOT be.
struct ProfzoneIgnoreData {
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
inline constexpr uint32_t kProfzoneMaxDataWords = 127;

template <typename Emit, typename EmitX280 = ProfzoneIgnoreX280, typename EmitData = ProfzoneIgnoreData>
inline void profzone_decode(
    ProfzoneDecodeState& st,
    const uint32_t* in,
    size_t in_n,
    uint32_t nl,
    Emit&& emit,
    EmitX280&& emit_x280 = ProfzoneIgnoreX280{},
    EmitData&& emit_data = ProfzoneIgnoreData{}) {
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
                            uint32_t payload[kProfzoneMaxDataWords];
                            for (uint32_t k = 0; k < n; k++) {
                                payload[k] = ring[(head_mod + i + 2 + k) % kProfzoneRingCap];
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
        } else if (pp_is_src(w0)) {  // 1-word: set the current lane
            st.cur_lane = pp_src_lane(w0);
            p += 1;
        } else if (pp_is_timer(w0)) {  // 1-word: refresh the current lane's timer_hi
            if (st.cur_lane < nl) {
                st.cur_hi[st.cur_lane] = pp_timer_hi(w0);
            }
            p += 1;
        } else if (pp_is_x280(w0)) {
            // 3-word: an X280 DRAIN-HART zone (--hartzones / ProfzoneBootCfg::hartzones), riding IN-BAND in
            // the marker stream: w0 = type|hart|kind|is_start, w1/w2 = the hart's 64-bit rdcycle.
            // This branch is MANDATORY whenever hart zones are enabled, even if the caller ignores them: the
            // generic tail below assumes a 2-WORD packet, so a 3-word X280 packet would desynchronize the
            // whole walk and corrupt every marker after it (silently -- the stream would still "decode").
            if (p + 2 >= sz) {
                break;  // partial -> carry
            }
            emit_x280(pp_x280_hart(w0), pp_low27(w0), (static_cast<uint64_t>(w[p + 2]) << 32) | w[p + 1]);
            p += 3;
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
