// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Data-movement helpers of the Hy4 decode projection programs: bounded waits on values another core writes into
// this core's L1, bounded NOC waits, the grid multicast, and the timeout report to the hub's status words. Derived
// from fused/moe/kernels/moe_dm.hpp.

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "proj_common.hpp"

namespace hy4proj {

// Spin until *p >= need. False when HY4_SPIN_LIMIT spins passed first.
FORCE_INLINE bool wait_ge(volatile tt_l1_ptr uint32_t* p, uint32_t need) {
    for (uint32_t spin = 0; spin < HY4_SPIN_LIMIT; ++spin) {
        invalidate_l1_cache();
        if (*p >= need) {
            return true;
        }
    }
    return false;
}

// The multicast rectangle (x0, y0) .. (x1, y1), corners in ascending order, for this kernel's NOC. NOC1 runs from
// high to low coordinates on Blackhole, so its rectangle is given end corner first; with the corners in ascending
// order a NOC1 multicast stalls its sender (fused/README.md; the same swap as in
// ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp).
FORCE_INLINE uint64_t grid_mcast_addr(uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1, uint32_t addr) {
    return noc_index == 0 ? get_noc_multicast_addr(x0, y0, x1, y1, addr) : get_noc_multicast_addr(x1, y1, x0, y0, addr);
}

// Bounded forms of the NOC waits a data-movement kernel would otherwise sit in for ever. False when HY4_SPIN_LIMIT
// spins passed: the caller reports and stops sending instead of hanging the chip.
FORCE_INLINE bool write_buf_free() {
    for (uint32_t spin = 0; spin < HY4_SPIN_LIMIT; ++spin) {
        if (noc_cmd_buf_ready(noc_index, write_cmd_buf)) {
            return true;
        }
    }
    return false;
}
FORCE_INLINE bool atomic_buf_free() {
    for (uint32_t spin = 0; spin < HY4_SPIN_LIMIT; ++spin) {
        if (noc_cmd_buf_ready(noc_index, write_at_cmd_buf)) {
            return true;
        }
    }
    return false;
}
FORCE_INLINE bool writes_flushed() {
    for (uint32_t spin = 0; spin < HY4_SPIN_LIMIT; ++spin) {
        if (ncrisc_noc_nonposted_writes_flushed(noc_index)) {
            invalidate_l1_cache();
            return true;
        }
    }
    return false;
}
FORCE_INLINE bool atomics_flushed() {
    for (uint32_t spin = 0; spin < HY4_SPIN_LIMIT; ++spin) {
        if (ncrisc_noc_nonposted_atomics_flushed(noc_index)) {
            invalidate_l1_cache();
            return true;
        }
    }
    return false;
}

// `bytes` from local L1 to `addr` on every core of the grid rectangle, one packet (at most 16 KB) at a time, each
// issued only once the command buffer is free. False when a packet could not be issued in time.
inline bool grid_mcast_write(
    uint32_t src, uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1, uint32_t addr, uint32_t bytes, uint32_t dests) {
    for (uint32_t done = 0; done < bytes;) {
        const uint32_t n = bytes - done < NOC_MAX_BURST_SIZE ? bytes - done : NOC_MAX_BURST_SIZE;
        if (!write_buf_free()) {
            return false;
        }
        noc_async_write_multicast_one_packet(src + done, grid_mcast_addr(x0, y0, x1, y1, addr + done), n, dests);
        done += n;
    }
    return true;
}

// One multicast atomic increment of the word at `addr` on every core of the grid rectangle.
inline bool grid_mcast_inc(uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1, uint32_t addr, uint32_t dests) {
    if (!atomic_buf_free()) {
        return false;
    }
    noc_semaphore_inc_multicast(grid_mcast_addr(x0, y0, x1, y1, addr), 1, dests);
    return true;
}

// Write (my noc x, y, code) into status word `word` of the hub (scratch: 4 writable bytes of this core's L1).
inline void report(
    uint32_t hub_x, uint32_t hub_y, uint32_t status_addr, uint32_t word, uint32_t code, uint32_t scratch) {
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    s[0] = (static_cast<uint32_t>(my_x[noc_index]) << 24) | (static_cast<uint32_t>(my_y[noc_index]) << 16) | code;
    if (write_buf_free()) {
        noc_async_write(scratch, get_noc_addr(hub_x, hub_y, status_addr + 4 * word), 4);
        writes_flushed();
    }
}

#ifdef HY4_TIMING
// Measurement only: wall-clock cycles of this core (low 32 bits), and two cycle counts into the hub's status tensor
// at words ST_TIMING + 2 * ((noc y - 2) * 16 + noc x).
FORCE_INLINE uint32_t now() { return *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }
inline void report_timing(
    uint32_t hub_x, uint32_t hub_y, uint32_t status_addr, uint32_t first, uint32_t done, uint32_t scratch) {
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    s[0] = first;
    s[1] = done;
    const uint32_t id = (static_cast<uint32_t>(my_y[noc_index]) - 2) * 16 + static_cast<uint32_t>(my_x[noc_index]);
    if (write_buf_free()) {
        noc_async_write(scratch, get_noc_addr(hub_x, hub_y, status_addr + 4 * (ST_TIMING + 2 * id)), 8);
        writes_flushed();
    }
}
#endif

}  // namespace hy4proj
