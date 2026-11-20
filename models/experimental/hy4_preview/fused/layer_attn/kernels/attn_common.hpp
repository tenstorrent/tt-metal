// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Hand-overs of the attention half of a decoder layer on the 8x4 mesh (fused/layer_attn/program.py; PLAN.md section
// 7): a second set of persistent words beside layer_common's, 32 uint32 on every core in a host-zeroed L1 buffer at
// the same address on every core of every chip (HY4A_WORDS, a define). A count is taken off by an atomic of the
// waiting core on itself and never cleared, so a count that arrives before the core's program has started is kept.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#include "layer_common.hpp"

namespace hy4a {

// counted by the fabric (the last packet of a sender op): the collective's sender reports an op that was never
// handed over 12 words after its counter (collective.hpp CTRL_STATUS), so words 12 .. 17 are the sender's
constexpr uint32_t A_QA = 0;      // query hub: chips whose q_a tiles are in the hub's x buffer (Xa)
constexpr uint32_t A_LANES = 1;   // query hub: chips whose statistic tile is in the lane landing (Xa)
constexpr uint32_t A_POOL = 2;    // append core: chips whose pool tile has landed (Xa)
constexpr uint32_t A_QARR = 3;    // merge core g: chips of head group g whose absorbed queries have landed (X3)
constexpr uint32_t A_COLARR = 4;  // merge core r: the column partner of mesh row r has landed its partials (X4)
constexpr uint32_t A_ROWARR = 5;  // merge core c: the row partner of mesh column c has landed its partials (X4)
// counted inside the chip
constexpr uint32_t A_QGO = 6;      // attention core: its group's queries are complete on the landing core (relay)
constexpr uint32_t A_COLMASK = 7;  // merge core: bit r = the column partials of source row r are on core (5, r)
constexpr uint32_t A_ROWMASK = 8;  // merge core: bit c = the row partials of source column c are on core (5, c)
constexpr uint32_t A_LOCAL = 9;    // merge core u: this chip's own partial of (user u, own group) is packed
constexpr uint32_t A_CTX = 10;     // post hub: merge cores that have written their rows of ctx
constexpr uint32_t A_SERVED = 11;  // sender core: the sender's served word (collective_sender.cpp)
constexpr uint32_t A_STATUS = 18;  // any core: the last wait that ran out here (code << 24 | what it saw)
constexpr uint32_t A_RUNS = 19;    // hubs: launches
constexpr uint32_t C_GROUP = 20;   // .. 27, sender core: attention cores of head group g that have packed their partial
constexpr uint32_t C_ROW = 28;     // sender core: merge cores that have handed their column result to the row staging
constexpr uint32_t A_WORDS = 32;

constexpr uint32_t ST_QA = 0xD1u << 24;      // the query hub: q_a tiles missing
constexpr uint32_t ST_LANES = 0xD2u << 24;   // the query hub: statistic tiles missing
constexpr uint32_t ST_UNITS = 0xD3u << 24;   // a hub: its units did not all deliver
constexpr uint32_t ST_STAT = 0xD4u << 24;    // the pre hub: the statistic tile did not arrive
constexpr uint32_t ST_QARR = 0xD5u << 24;    // a relay: its group's queries did not all land
constexpr uint32_t ST_QGO = 0xD6u << 24;     // an attention core: no go for its queries
constexpr uint32_t ST_GROUP = 0xD7u << 24;   // the coordinator: a group's partials were not all packed
constexpr uint32_t ST_SERVED = 0xD8u << 24;  // the coordinator: the sender did not report an op served
constexpr uint32_t ST_COL = 0xD9u << 24;     // a merge core: column partials missing (low bits: the mask seen)
constexpr uint32_t ST_ROW = 0xDAu << 24;     // a merge core: row partials missing
constexpr uint32_t ST_LOCAL = 0xDBu << 24;   // a merge core: no local partial
constexpr uint32_t ST_CTX = 0xDCu << 24;     // the post hub: ctx incomplete
constexpr uint32_t ST_POOL = 0xDDu << 24;    // the append core: pool tiles missing

#ifdef HY4A_WORDS
constexpr uint32_t word_addr(uint32_t i) { return HY4A_WORDS + 4 * i; }
FORCE_INLINE volatile tt_l1_ptr uint32_t* word(uint32_t i) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(word_addr(i));
}

// Poll until this core's word i reaches `need`. False when the bound ran out: the code and what was seen go into
// this core's status word and the kernel carries on with what it has, so the program always ends.
inline bool wait_word(uint32_t i, uint32_t need, uint32_t code) {
    volatile tt_l1_ptr uint32_t* p = word(i);
    uint32_t seen = 0;
    for (uint32_t polls = 0; polls < HY4L_WAIT_LIMIT; ++polls) {
        invalidate_l1_cache();
        seen = p[0];
        if (seen >= need) {
            return true;
        }
    }
    word(A_STATUS)[0] = code | (seen & 0xFFFFFF);
    return false;
}

// Take `n` off this core's word i (an atomic on itself: a peer may be counting into it at this moment).
FORCE_INLINE void take_word(uint32_t i, uint32_t n) {
    noc_semaphore_inc(get_noc_addr(word_addr(i)), 0u - n);
    noc_async_atomic_barrier();
}

FORCE_INLINE void fail(uint32_t code, uint32_t seen) { word(A_STATUS)[0] = code | (seen & 0xFFFFFF); }
#endif  // HY4A_WORDS

}  // namespace hy4a
