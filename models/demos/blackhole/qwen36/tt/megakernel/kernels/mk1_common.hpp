// MK1 layer-skeleton megakernel: fm11 engine + a per-layer mixer projection stream and calibrated chain stalls.
#pragma once
#include "fm11_common.hpp"
constexpr uint32_t CB_WM = 6, CB_YM = 7, CB_GO = 8;
// -mk-tail (experimental): program semaphore 12 = cold-start "layer-0 mixer landed" flag (writer -> reader on the same
// core). DESIGN §6 reserves 12-15 for phase/layer barriers: agree an id with the engine owner before merging.
constexpr uint32_t SEM_COLD = 12;
// Busy-wait on the RISC wall clock (the same clock the device profiler uses), modelling a chain stall
// (GDN step / SDPA / all-reduce) while this core's DRAM prefetch for the next layer keeps flowing.
FORCE_INLINE void spin_cycles(uint32_t cyc) {
    if (cyc == 0) return;
    const uint32_t t0 = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    while ((uint32_t)(reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L) - t0) < cyc) {}
}
