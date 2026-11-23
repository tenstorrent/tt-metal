// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fabric ring microbench, RISC split (fr2_send.cpp / fr2_proc.cpp): the words of a lane core (16 B apart). ARR arrivals
// (the upstream router's atomic incs), READY arrivals the processor has examined, CPB / CX this lane's downstream
// credit (piggy-backed via the partner's processor / explicit inline write from the fabric), FREED this lane's freed
// prefix (sender), FREED_O / DONE_O the partner lane's (written by the partner's processor), DONE this sender done,
// TDONE this sender has stopped, ST_T sender stamps (8 words), ST_P processor stamps.
#pragma once
#include <cstdint>

namespace fr2 {
constexpr uint32_t ARR = 0, READY = 1, CPB = 2, CX = 3, FREED = 4, FREED_O = 5, DONE = 6, DONE_O = 7, TDONE = 8;
constexpr uint32_t ST_T = 10, ST_P = 18;
constexpr uint32_t ACT_FWD = 1, ACT_RETIRE = 2;

FORCE_INLINE volatile tt_l1_ptr uint32_t& word(uint32_t base, uint32_t i, uint32_t stride = 16) {
    return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + stride * i);
}
FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }
}  // namespace fr2
