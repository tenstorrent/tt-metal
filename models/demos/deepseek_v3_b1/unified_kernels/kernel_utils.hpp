// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#endif
#if defined(COMPILE_FOR_TRISC)
#include "../kernel_includes/tt_metal/include/compute_kernel_api/deepseek_compute_kernel_hw_startup.h"
#endif

// Firmware-set logical coordinates (defined in brisc.cc, ncrisc.cc, trisc.cc)
extern uint8_t my_logical_x_;
extern uint8_t my_logical_y_;

namespace unified_kernels {

// ============================================================================
// Grid coordinate utilities
// ============================================================================

// Compute linear index within a grid defined by start/end coordinates
// RowMajor=true:  index = rel_y * grid_width + rel_x  (iterate x first, then y)
// RowMajor=false: index = rel_x * grid_height + rel_y (iterate y first, then x)
template <bool RowMajor>
uint32_t linear_id_in_grid(uint32_t grid_start_x, uint32_t grid_start_y, uint32_t grid_end_x, uint32_t grid_end_y) {
    uint32_t rel_x = my_logical_x_ - grid_start_x;
    uint32_t rel_y = my_logical_y_ - grid_start_y;
    if constexpr (RowMajor) {
        uint32_t grid_width = grid_end_x - grid_start_x + 1;
        return rel_y * grid_width + rel_x;
    } else {
        uint32_t grid_height = grid_end_y - grid_start_y + 1;
        return rel_x * grid_height + rel_y;
    }
}

struct SplitHalfCoreInfo {
    bool is_half0;
    uint32_t half_local_idx;
};

template <bool RowMajor>
SplitHalfCoreInfo get_split_half_core_info(
    uint32_t grid_start_x, uint32_t grid_start_y, uint32_t grid_end_x, uint32_t grid_end_y, uint32_t half_num_cores) {
    const uint32_t linear_idx = linear_id_in_grid<RowMajor>(grid_start_x, grid_start_y, grid_end_x, grid_end_y);
    const bool is_half0 = linear_idx < half_num_cores;
    const uint32_t half_local_idx = is_half0 ? linear_idx : (linear_idx - half_num_cores);
    return {is_half0, half_local_idx};
}

// ============================================================================
// Sharded persistent buffer setup utilities
// ============================================================================

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)

// Setup a sharded persistent buffer by reserving and pushing tiles
// This makes the buffer available for compute to read from
// Note: Can be called from either NCRISC or BRISC, whichever runs first
FORCE_INLINE void setup_sharded_buffer(uint32_t cb_id, uint32_t num_tiles) {
    cb_reserve_back(cb_id, num_tiles);
    cb_push_back(cb_id, num_tiles);
}

// Atomic semaphore decrement (for global semaphore reset across iterations)
FORCE_INLINE void semaphore_dec(volatile tt_l1_ptr uint32_t* sem_addr) {
    __atomic_fetch_sub(sem_addr, 1, __ATOMIC_RELAXED);
}

#endif

// ============================================================================
// Cross-RISC synchronization
// ============================================================================

// Two-phase cross-RISC synchronization barrier using atomic L1 semaphores.
// Used by reconfig_cb_interfaces to ensure NC resets stream regs only after
// all other RISCs (BR/TR0/TR2) have completed prior work.
//
// Phase 1 (enter): BR/TR0/TR2 signal done → NC waits for all 3, then proceeds.
// Phase 2 (exit):  NC signals done → BR/TR0/TR2 wait, then all proceed together.
//
// Safe for repeated use: the full MoE body executes between exit and the next
// enter, so sem[0] is always 0 when the next iteration's enter begins.
// sem[0] reset is non-atomic but safe because others are blocked on sem[1].

// Phase 1: BR/TR0/TR2 signal done → NC waits for all 3
FORCE_INLINE void sync_riscs_enter(volatile uint32_t tt_l1_ptr* sem_addr) {
#if defined(COMPILE_FOR_BRISC) || defined(UCK_CHLKC_UNPACK) || defined(UCK_CHLKC_PACK)
    __atomic_fetch_add(&sem_addr[0], 1, __ATOMIC_RELAXED);
#elif defined(COMPILE_FOR_NCRISC)
    while (__atomic_load_n(&sem_addr[0], __ATOMIC_RELAXED) < 3) {
    }
    sem_addr[0] = 0;
#endif
}

// Phase 2: NC signals done → BR/TR0/TR2 wait then proceed
FORCE_INLINE void sync_riscs_exit(volatile uint32_t tt_l1_ptr* sem_addr) {
#if defined(COMPILE_FOR_NCRISC)
    __atomic_fetch_add(&sem_addr[1], 3, __ATOMIC_RELAXED);
#elif defined(COMPILE_FOR_BRISC) || defined(UCK_CHLKC_UNPACK) || defined(UCK_CHLKC_PACK)
    while (__atomic_load_n(&sem_addr[1], __ATOMIC_RELAXED) == 0) {
    }
    __atomic_fetch_sub(&sem_addr[1], 1, __ATOMIC_RELAXED);
#endif
}

// ============================================================================
// CB reconfig utilities
// ============================================================================

// Minimal CB reconfig: reset fifo pointers from a config tensor in L1.
// Sets fifo_rd_ptr, fifo_wr_ptr, fifo_size, fifo_limit, fifo_page_size, fifo_num_pages,
// tiles_acked_received_init, and fifo_wr_tile_ptr. Also resets stream regs on NCRISC.
//
// Config tensor layout per core (264 uint32):
//   Words 0-255: 64 CB configs, 4 words each [addr, size, num_pages, page_size] (bytes)
//   Word 256:    cb_mask_low  (bits 0-31)
//   Word 257:    cb_mask_high (bits 32-63)
//
// read/write flags follow firmware convention per RISC:
//   NCRISC/BRISC:  read=true,  write=true
//   TRISC0/unpack: read=true,  write=false
//   TRISC2/pack:   read=false, write=true
template <bool do_read, bool do_write, bool do_reset_stream_regs>
FORCE_INLINE void reconfig_cbs_for_mask(uint32_t tt_l1_ptr* cb_config, uint32_t mask, uint32_t start_cb) {
    uint32_t cb = start_cb;
    while (mask) {
        if (mask & 1) {
            uint32_t base = cb * 4;
            uint32_t fifo_addr = cb_config[base + 0] >> cb_addr_shift;
            uint32_t fifo_size = cb_config[base + 1] >> cb_addr_shift;
            uint32_t fifo_num_pages = cb_config[base + 2];
            uint32_t fifo_page_size = cb_config[base + 3] >> cb_addr_shift;

            LocalCBInterface& iface = get_local_cb_interface(cb);
            if constexpr (do_read) {
                iface.fifo_rd_ptr = fifo_addr;
            }
            if constexpr (do_write) {
                iface.fifo_wr_ptr = fifo_addr;
                iface.fifo_num_pages = fifo_num_pages;
            }
            iface.fifo_size = fifo_size;
            iface.fifo_limit = fifo_addr + fifo_size;
            iface.fifo_page_size = fifo_page_size;
            iface.tiles_acked_received_init = 0;

            if constexpr (do_reset_stream_regs) {
                *get_cb_tiles_received_ptr(cb) = 0;
                *get_cb_tiles_acked_ptr(cb) = 0;
            }
        }
        mask >>= 1;
        cb++;
    }
}

FORCE_INLINE void reconfig_cb_interfaces(uint32_t tt_l1_ptr* cb_config) {
#if defined(COMPILE_FOR_NCRISC)
    constexpr bool do_read = true;
    constexpr bool do_write = true;
    constexpr bool do_reset_stream_regs = true;
#elif defined(COMPILE_FOR_BRISC)
    constexpr bool do_read = true;
    constexpr bool do_write = true;
    constexpr bool do_reset_stream_regs = false;
#elif defined(UCK_CHLKC_UNPACK)
    constexpr bool do_read = true;
    constexpr bool do_write = false;
    constexpr bool do_reset_stream_regs = false;
#elif defined(UCK_CHLKC_PACK)
    constexpr bool do_read = false;
    constexpr bool do_write = true;
    constexpr bool do_reset_stream_regs = false;
#else
    constexpr bool do_read = false;
    constexpr bool do_write = false;
    constexpr bool do_reset_stream_regs = false;
#endif

    volatile uint32_t tt_l1_ptr* reconfig_sem = reinterpret_cast<volatile uint32_t tt_l1_ptr*>(&cb_config[258]);
    sync_riscs_enter(reconfig_sem);

    reconfig_cbs_for_mask<do_read, do_write, do_reset_stream_regs>(cb_config, cb_config[256], 0);
    reconfig_cbs_for_mask<do_read, do_write, do_reset_stream_regs>(cb_config, cb_config[257], 32);

    sync_riscs_exit(reconfig_sem);
}

}  // namespace unified_kernels
