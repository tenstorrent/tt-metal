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

#endif

// ============================================================================
// CB reconfig utilities
// ============================================================================

// Minimal CB reconfig: reset fifo pointers from a config tensor in L1.
// Only sets fifo_rd_ptr, fifo_wr_ptr, fifo_size, fifo_limit, fifo_page_size, fifo_num_pages.
// Does NOT touch tiles_acked_received_init or fifo_wr_tile_ptr.
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
FORCE_INLINE void reconfig_cb_interfaces(uint32_t tt_l1_ptr* cb_config) {
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
    constexpr bool do_read = true;
    constexpr bool do_write = true;
#elif defined(UCK_CHLKC_UNPACK)
    constexpr bool do_read = true;
    constexpr bool do_write = false;
#elif defined(UCK_CHLKC_PACK)
    constexpr bool do_read = false;
    constexpr bool do_write = true;
#else
    constexpr bool do_read = false;
    constexpr bool do_write = false;
#endif
    uint32_t cb_mask_low = cb_config[256];
    uint32_t cb_mask_high = cb_config[257];

    // Process lower 32 CBs (0-31)
    uint32_t mask = cb_mask_low;
    uint32_t cb = 0;
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
        }
        mask >>= 1;
        cb++;
    }

    // Process upper 32 CBs (32-63)
    mask = cb_mask_high;
    cb = 32;
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
        }
        mask >>= 1;
        cb++;
    }
}

}  // namespace unified_kernels
