// MKA2: the attention-type layer inside the megakernel engine (MK1 engine + the MKA1 SDPA phase), spec v0.3.
// Per layer L, every core of the full grid (STREAM rect time-multiplexed, plus the ATTN-only cores):
//   [STREAM] MIXER (qkv in-proj stand-in, real bytes)  ->  SDPA (phase type 5, ATTN CBs 9-12, all cores)
//   -> [STREAM] wo stand-in + AR stall -> MLP gate|up / down (fm11 engine) -> AR stall.
// ATTN synchronisation uses GLOBAL semaphores (spec v0.1): uniform L1 addresses passed in runtime args; all counters
// are cumulative across layers (value for layer L = (L + 1) x per-layer count), so nothing is reset inside a pass.
#pragma once
#include "../../kernels/fm11_common.hpp"
constexpr uint32_t CB_WM = 6, CB_YM = 7, CB_GO = 8;
constexpr uint32_t CB_Q = 9, CB_K = 10, CB_V = 11, CB_O = 12;
constexpr uint32_t CB_ONE = 24, CB_P = 25, CB_R = 26, CB_D = 27, CB_OA = 28, CB_SCR = 29;
#ifndef MKA_NU
#define MKA_NU 4
#endif
#ifndef MKA_FAN
#define MKA_FAN 6
#endif
constexpr uint32_t DH = 8, NO = DH + 1, QT = 8 * MKA_NU, FANOUT = MKA_FAN, NUSR = MKA_NU;   // own users per chip (u8: 4, u1: 1)
// global-semaphore slots (index into the runtime-arg address array)
constexpr uint32_t GS_QFLAG = 0, GS_LVL0 = 1, GS_LVL1 = 2, GS_DONE = 3, GS_MIXDONE = 4, GS_ADONE = 5, GS_ADFLAG = 6, NGS = 7;
FORCE_INLINE volatile tt_l1_ptr uint32_t* l1w(uint32_t addr) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr); }
FORCE_INLINE void spin_cycles(uint32_t cyc) {
    if (cyc == 0) return;
    const uint32_t t0 = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    while ((uint32_t)(reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L) - t0) < cyc) {}
}
// multicast an L1 word holding `val` to the same address on up to 2 rects (loopback on self_rect)
FORCE_INLINE void mcast_word(uint32_t addr, uint32_t val, const uint32_t (*rc)[5], uint32_t self_rect, uint8_t noc) {
    *l1w(addr) = val;
    for (uint32_t i = 0; i < 2; ++i) {
        if (rc[i][4] == 0) continue;
        const uint64_t m = noc == 0 ? get_noc_multicast_addr(rc[i][0], rc[i][1], rc[i][2], rc[i][3], addr, noc)
                                    : get_noc_multicast_addr(rc[i][2], rc[i][3], rc[i][0], rc[i][1], addr, noc);
        if (i == self_rect) noc_semaphore_set_multicast_loopback_src(addr, m, rc[i][4], false, noc);
        else noc_semaphore_set_multicast(addr, m, rc[i][4], false, noc);
    }
}
