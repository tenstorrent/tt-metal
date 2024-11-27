// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_ETH_API_H
#define TT_ETH_API_H

#include <stdint.h>

#include "tt_eth_ss_regs.h"

#define ETH_WRITE_REG(addr, val) ((*((volatile uint32_t*)((addr)))) = (val))
#define ETH_READ_REG(addr) (*((volatile uint32_t*)((addr))))

inline void eth_txq_reg_write(uint32_t qnum, uint32_t offset, uint32_t val) {
    ETH_WRITE_REG(ETH_TXQ0_REGS_START + (qnum * ETH_TXQ_REGS_SIZE) + offset, val);
}

inline uint32_t eth_txq_reg_read(uint32_t qnum, uint32_t offset) {
    return ETH_READ_REG(ETH_TXQ0_REGS_START + (qnum * ETH_TXQ_REGS_SIZE) + offset);
}

inline void eth_risc_reg_write(uint32_t offset, uint32_t val) { ETH_WRITE_REG(ETH_RISC_REGS_START + offset, val); }
inline uint32_t eth_risc_reg_read(uint32_t offset) { return ETH_READ_REG(ETH_RISC_REGS_START + offset); }
inline uint64_t eth_read_wall_clock() {
    uint32_t wall_clock_low = eth_risc_reg_read(ETH_RISC_WALL_CLOCK_0);
    uint32_t wall_clock_high = eth_risc_reg_read(ETH_RISC_WALL_CLOCK_1_AT);
    return (((uint64_t)wall_clock_high) << 32) | wall_clock_low;
}

inline void eth_wait_cycles(uint32_t wait_cycles) {
    if (wait_cycles == 0) {
        return;
    }
    uint64_t curr_timer = eth_read_wall_clock();
    uint64_t end_timer = curr_timer + wait_cycles;
    while (curr_timer < end_timer) {
        curr_timer = eth_read_wall_clock();
    }
}

#endif
