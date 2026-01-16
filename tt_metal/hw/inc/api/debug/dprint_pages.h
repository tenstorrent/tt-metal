// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// This file contains common kernel functions used for debugging
#pragma once
#include "api/debug/dprint.h"

#if (                                                            \
    defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC) || \
    (defined(COMPILE_FOR_TRISC) && (COMPILE_FOR_TRISC != 1)))
inline void print_cb_details(uint32_t cb_id) {
    DPRINT << "cb_id " << cb_id << ": { "
           << "size: " << get_local_cb_interface(cb_id).fifo_size << ", "
           << "limit: " << get_local_cb_interface(cb_id).fifo_limit << ", "
           << "page_size: " << get_local_cb_interface(cb_id).fifo_page_size << ", "
           << "num_pages: " << get_local_cb_interface(cb_id).fifo_num_pages << ", "
           << "rd_ptr: " << get_local_cb_interface(cb_id).fifo_rd_ptr << ", "
           << "wr_ptr: " << get_local_cb_interface(cb_id).fifo_wr_ptr << ", "
           << "wr_tile_ptr: " << get_local_cb_interface(cb_id).fifo_wr_tile_ptr << " }" << ENDL();
}
#endif

#if (defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)) && defined(DEBUG_PRINT_ENABLED) && \
    !defined(FORCE_DPRINT_OFF)

namespace tt::data_movement::common {

inline void print_bf16_pages(uint32_t l1_addr, uint32_t elts_per_page, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * elts_per_page;
    for (uint32_t page = 0; page < npages; ++page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < elts_per_page; ++j, ++ptr) {
            DPRINT << BF16(*ptr) << " ";
        }
        DPRINT << ENDL();
    }
}

inline void print_f32_pages(uint32_t l1_addr, uint32_t elts_per_page, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr) + start * elts_per_page;
    for (uint32_t page = 0; page < npages; ++page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < elts_per_page; ++j, ++ptr) {
            DPRINT << F32(*ptr) << " ";
        }
        DPRINT << ENDL();
    }
}

inline void print_u8_pages(uint32_t l1_addr, uint32_t bytes_per_page, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint8_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(l1_addr) + start * bytes_per_page;
    for (uint32_t page = 0; page < npages; ++page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < bytes_per_page; ++j, ++ptr) {
            DPRINT << SETW(2) << HEX() << "0x" << (uint32_t)*ptr << " ";
        }
        DPRINT << DEC();  // revert to decimal representation
        DPRINT << ENDL();
    }
}

inline void print_u16_pages(uint32_t l1_addr, uint32_t elts_per_page, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * elts_per_page;
    for (uint32_t page = 0; page < npages; ++page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < elts_per_page; ++j, ++ptr) {
            DPRINT << (uint32_t)*ptr << " ";
        }
        DPRINT << ENDL();
    }
}

}  // namespace tt::data_movement::common

#endif

#if defined(COMPILE_FOR_TRISC) && defined(DEBUG_PRINT_ENABLED) && !defined(FORCE_DPRINT_OFF)

namespace tt::compute::common {

void print_tile_rows(
    uint32_t cb_idx,
    uint32_t tile_idx,
    bool untilize = false,
    uint16_t start_row = 0,
    uint16_t end_row = 32,
    uint8_t start_col = 0,
    uint8_t end_col = 32) {
    DPRINT << "cb_idx: " << cb_idx << " tile_idx: " << tile_idx << ENDL();
    DPRINT << "======" << ENDL();
    for (uint16_t r = start_row; r < end_row; ++r) {
        DPRINT << (uint)r << " : "
               << TileSlice(
                      cb_idx,
                      tile_idx,
                      SliceRange{
                          .h0 = (uint8_t)r,
                          .h1 = (uint8_t)(r + 1),
                          .hs = (uint8_t)1,
                          .w0 = (uint8_t)start_col,
                          .w1 = (uint8_t)end_col,
                          .ws = (uint8_t)1},
                      true,
                      untilize)
               << ENDL();
    }
    DPRINT << "++++++" << ENDL();
}

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    DPRINT << "======" << ENDL();
    for (uint16_t r = 0; r < 32; ++r) {
        DPRINT << (uint)r << " : "
               << TileSlice(
                      cb_id,
                      tile_id,
                      SliceRange{
                          .h0 = (uint8_t)r,
                          .h1 = (uint8_t)(r + 1),
                          .hs = (uint8_t)1,
                          .w0 = (uint8_t)0,
                          .w1 = (uint8_t)32,
                          .ws = (uint8_t)1},
                      true,
                      untilize)
               << ENDL();
    }
    DPRINT << "++++++" << ENDL();
}

}  // namespace tt::compute::common

#endif
