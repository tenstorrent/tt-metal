// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 14 in-program fabric all-reduce phase (tt/fabric_ar.py): shared layout of the persistent sync words.
// Every counter is MONOTONIC across launches (never reset; zeroed once by the host when the persistent tensor is
// created), so a fabric increment that lands while the receiver is still in its previous launch can never be lost
// or double counted: iteration g (1, 2, ...) waits for counter >= g * per_iteration_count.
#pragma once
#include <cstdint>

namespace far {
constexpr uint32_t WORD = 16;  // one counter per 16 B (NoC atomic / L1 alignment)
// lane core words
constexpr uint32_t OWN_READY = 0;   // producers -> lane (local NoC inc), n_prod per iteration
constexpr uint32_t ARR1 = 1;        // ARR1 + q: round-1 (mesh axis 0) packets landed from axis position q
constexpr uint32_t ARR2 = 9;        // ARR2 + q: round-2 (mesh axis 1) packets landed from axis position q
constexpr uint32_t BARRIER = 17;    // end-of-iteration incs from every chip this lane writes to
constexpr uint32_t RES_DONE = 18;   // lane NCRISC -> lane BRISC: final result packed (local), = g after iteration g
constexpr uint32_t GEN_BRISC = 19;  // per-RISC iteration counters (persist across launches)
constexpr uint32_t GEN_NCRISC = 20;
constexpr uint32_t PROBE = 24;  // probe words: lane BRISC 24..31, lane NCRISC 32..39
constexpr uint32_t PROBE_NC = 32;
// producer core words
constexpr uint32_t RESULT_READY = 0;  // lane -> producer, 1 per iteration
constexpr uint32_t GEN_PROD = 1;
constexpr uint32_t PROD_PROBE = 24;
constexpr uint32_t CB_ADDR = 40;  // far_probe.cpp: base address of each lane CB (megakernel "cb" mode), words 40..55
constexpr uint32_t SYNC_WORDS = 56;

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
inline volatile tt_l1_ptr uint32_t* word(uint32_t base, uint32_t idx) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + idx * WORD);
}
inline void wait_ge(volatile tt_l1_ptr uint32_t* p, uint32_t v) {
    WAYPOINT("FARW");
    do {
        invalidate_l1_cache();
    } while (static_cast<int32_t>(*p - v) < 0);
    WAYPOINT("FARD");
}
inline void probe(uint32_t base, uint32_t idx, uint64_t t) {
    // low 32 bits of the cycle counter (wraps every ~3 s at 1.35 GHz: differences only)
    *word(base, idx) = static_cast<uint32_t>(t);
}
// DRAIN, then close (2026-10-23 18:04:51Z ruling; the same helper as fs_common.hpp fs::drain_close): spin until the
// router freed every slot this connection filled, then close (close() does not wait for in-flight packets).
template <typename Conn>
inline void drain_close(Conn& conn) {
    WAYPOINT("FADW");
    while (conn.get_num_free_write_slots() != conn.num_buffers_per_channel) {
    }
    WAYPOINT("FADD");
    conn.close();
}
#endif
}  // namespace far
