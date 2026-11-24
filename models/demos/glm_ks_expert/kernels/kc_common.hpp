// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Stage 2 (x chains) shared constants: named compile-time args of every kc_* kernel.
#pragma once
#include <cstdint>

namespace kc {
constexpr uint32_t MODE = get_named_compile_time_arg_val("mode");    // 0 today (one pair, all K), 1 K split in pairs
constexpr uint32_t MT = get_named_compile_time_arg_val("mt");        // row tiles a sub-block
constexpr uint32_t KB = get_named_compile_time_arg_val("kb");        // K tiles an x block
constexpr uint32_t NB = get_named_compile_time_arg_val("nb");        // x blocks a sub-block (this core's K)
constexpr uint32_t XS = get_named_compile_time_arg_val("xs");        // x ring slots (blocks)
constexpr uint32_t XP = get_named_compile_time_arg_val("xp");        // pieces an x block
constexpr uint32_t DEPTH = get_named_compile_time_arg_val("depth");  // pieces in flight a forwarder
constexpr uint32_t NSB = get_named_compile_time_arg_val("nsb");      // sub-blocks a launch
constexpr uint32_t WC = get_named_compile_time_arg_val("wc");        // weight tile columns (2 / 4)
constexpr uint32_t PS = get_named_compile_time_arg_val("ps");        // partial ring slots (sub-blocks)
constexpr uint32_t X_TILE = get_named_compile_time_arg_val("x_tile_bytes");
constexpr uint32_t P_TILE = get_named_compile_time_arg_val("p_tile_bytes");
constexpr uint32_t W_STREAM = get_named_compile_time_arg_val("w_stream");  // 1: weights streamed from DRAM a expert
constexpr uint32_t SBE = get_named_compile_time_arg_val("sbe");            // sub-blocks an expert (NSB / SBE experts)
constexpr uint32_t WS = get_named_compile_time_arg_val("ws");              // weight ring slots (blocks)
constexpr uint32_t WPIECES = get_named_compile_time_arg_val("wpieces");    // reads a weight block
constexpr uint32_t WDEPTH = get_named_compile_time_arg_val("wdepth");      // weight reads in flight
constexpr uint32_t W_TILE = get_named_compile_time_arg_val("w_tile_bytes");
constexpr uint32_t KT = NB * KB;                 // this core's K tiles
constexpr uint32_t X_BLK = MT * KB;              // tiles an x block
constexpr uint32_t X_BYTES = X_BLK * X_TILE;     // bytes an x block
constexpr uint32_t PIECE = X_BYTES / XP;         // bytes a piece
constexpr uint32_t P_TILES = MT * 2;             // partial tiles a sub-block (gate, up a row tile)
constexpr uint32_t P_BYTES = P_TILES * P_TILE;   // bytes a partial slot
constexpr uint32_t TOTAL_BLOCKS = NSB * NB;      // x blocks a core a launch
constexpr uint32_t NE = NSB / SBE;               // experts a launch
constexpr uint32_t WBLK = KB * WC;               // tiles a weight block
constexpr uint32_t WBLK_BYTES = WBLK * W_TILE;
// CBs
constexpr uint32_t X_CB = 0, W_CB = 1, PIN_CB = 2, ID_CB = 3, H_CB = 16, POUT_CB = 17;
// semaphores
constexpr uint32_t XARR = 0, SFREE = 1, XCONS = 2, XEND = 3, PARR = 4, PFREE = 5, RFREE = 6;
}  // namespace kc

#ifndef COMPILE_FOR_TRISC
FORCE_INLINE volatile tt_l1_ptr uint32_t* kc_sem(uint32_t id) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id));
}
FORCE_INLINE uint64_t kc_xy(uint32_t packed, uint32_t addr) { return get_noc_addr(packed >> 16, packed & 0xFFFF, addr); }
#endif
