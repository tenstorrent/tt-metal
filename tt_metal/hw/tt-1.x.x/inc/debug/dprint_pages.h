// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// This file contains common kernel functions used for debugging
#pragma once
#include "debug/dprint.h"

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

}  // namespace tt::data_movement::common

#endif

#if defined(COMPILE_FOR_TRISC) && defined(DEBUG_PRINT_ENABLED) && !defined(FORCE_DPRINT_OFF)

namespace tt::compute::common {

inline void print_tile_rows(uint32_t cb_id, uint32_t rows = 32, uint32_t tile_id = 0, bool untilize = false) {
    for (uint16_t r = 0; r < rows; ++r) {
        DPRINT << (uint)r << " :: "
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
                      untilize);
    }
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
