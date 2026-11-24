// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Stage 4 down side (kd_*.cpp) shared constants.
#pragma once
#include <cstdint>

namespace kd {
constexpr uint32_t NSB = get_named_compile_time_arg_val("nsb");
constexpr uint32_t SBE = get_named_compile_time_arg_val("sbe");
constexpr uint32_t MT = get_named_compile_time_arg_val("mt");
constexpr uint32_t NGU = get_named_compile_time_arg_val("ngu");      // gate/up cores (h tiles a row tile)
constexpr uint32_t HBUF = get_named_compile_time_arg_val("hbuf");    // h buffers a down core
constexpr uint32_t HP = get_named_compile_time_arg_val("hp");        // pieces an h buffer (chain forwarding)
constexpr uint32_t HDEPTH = get_named_compile_time_arg_val("hdepth");
constexpr uint32_t DC = get_named_compile_time_arg_val("dc");        // y tile columns a down core
constexpr uint32_t DKB = get_named_compile_time_arg_val("dkb");      // K tiles a down weight block
constexpr uint32_t DPIECES = get_named_compile_time_arg_val("dpieces");
constexpr uint32_t DDEPTH = get_named_compile_time_arg_val("ddepth");
constexpr uint32_t ND = get_named_compile_time_arg_val("nd");        // down cores
constexpr uint32_t H_TILE = get_named_compile_time_arg_val("h_tile_bytes");
constexpr uint32_t W_TILE = get_named_compile_time_arg_val("w_tile_bytes");
constexpr uint32_t Y_TILE = get_named_compile_time_arg_val("y_tile_bytes");
constexpr uint32_t NE = NSB / SBE;
constexpr uint32_t HK = NGU;                       // h K tiles
constexpr uint32_t H_TILES = HK * MT;              // tiles an h buffer ([K][row])
constexpr uint32_t H_BYTES = H_TILES * H_TILE;
constexpr uint32_t H_PIECE = H_BYTES / HP;
constexpr uint32_t DNB = HK / DKB;                 // down weight blocks an expert
constexpr uint32_t DBLK = DKB * DC;                // tiles a down weight block
constexpr uint32_t DBLK_BYTES = DBLK * W_TILE;
constexpr uint32_t DWS = 2 * DNB;                  // down weight ring (blocks): two experts
constexpr uint32_t Y_TILES = MT * DC;               // y tiles a sub-block
// CBs on the down cores
constexpr uint32_t WD_CB = 4, HIN_CB = 5, OUT_CB = 18;
// semaphores
constexpr uint32_t GATH = 8, HARR = 9, DONE = 11, GO = 12;
}  // namespace kd
