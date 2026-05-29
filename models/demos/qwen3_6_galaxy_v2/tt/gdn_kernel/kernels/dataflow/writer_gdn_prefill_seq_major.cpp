// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Prefill GDN writer kernel — seq-major dense output variant.
//
// Output layout: [1, 1, num_tokens, num_pairs * Dv]   (TILE)
//   In tile coords: Yt = ceil(num_tokens / 32),  Xt = num_pairs * Vt
//   Tile-id at (token_y_tile, pair, vt) =
//      token_y_tile * (num_pairs * Vt) + pair * Vt + vt
//
// State layout: [num_pairs, Dk, Dv]  (unchanged from writer_gdn_prefill.cpp)
//
// The compute kernel pushes Vt tiles per token into cb_out (each tile has its
// 32 rows holding 32 copies of the same Dv slice — the Y dim is logical-1
// padded to 32). This writer L1-buffers up to 32 tokens' Vt-tiles and
// accumulates them into Vt "dense" tiles where row r holds token r's slice.
// After 32 tokens (or end-of-sequence), the dense tiles are NoC-written to
// DRAM at the right (token_y_tile, pair) coordinate.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t out_addr = get_arg_val<uint32_t>(0);    // [1,1,num_tokens, num_pairs*Dv]
    uint32_t state_addr = get_arg_val<uint32_t>(1);  // [num_pairs, Dk, Dv]
    uint32_t pair_start = get_arg_val<uint32_t>(2);
    uint32_t num_pairs_local = get_arg_val<uint32_t>(3);

    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t STATE_IN_L1 = get_compile_time_arg_val(3);
    constexpr uint32_t num_tokens = get_compile_time_arg_val(4);
    constexpr uint32_t num_pairs_total = get_compile_time_arg_val(5);  // NEW vs old writer
    constexpr uint32_t state_tiles = Kt * Vt;
    constexpr uint32_t num_blocks = (num_tokens + 31) / 32;  // outer Y-tile count

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_state_out = tt::CBIndex::c_8;

    constexpr bool is_dram = true;
    const InterleavedAddrGenFast<is_dram> out_wr = {
        .bank_base_address = out_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b};

    constexpr bool state_is_dram = (STATE_IN_L1 == 0);
    const InterleavedAddrGenFast<state_is_dram> state_wr = {
        .bank_base_address = state_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b};

    // Static L1 scratch: Vt tiles for assembling one block's dense Y-tiles.
    // 4 tiles × 2 KB = 8 KB. Sized at compile time via a pinned CB slot.
    constexpr uint32_t cb_scratch = tt::CBIndex::c_30;
    cb_reserve_back(cb_scratch, Vt);
    uint32_t scratch_base = get_write_ptr(cb_scratch);

    for (uint32_t pp = 0; pp < num_pairs_local; pp++) {
        uint32_t pair = pair_start + pp;

        for (uint32_t block = 0; block < num_blocks; block++) {
            uint32_t tok_in_block = (block + 1 < num_blocks) ? 32 : (num_tokens - block * 32);  // 1..32

            // Zero-init the scratch tiles for this block (covers tail-padding).
            uint32_t scratch_bytes = Vt * tile_bytes;
            volatile tt_l1_ptr uint32_t* sptr = (volatile tt_l1_ptr uint32_t*)scratch_base;
            for (uint32_t i = 0; i < scratch_bytes / 4; i++) {
                sptr[i] = 0;
            }

            for (uint32_t r = 0; r < tok_in_block; r++) {
                cb_wait_front(cb_out, Vt);
                uint32_t src_base = get_read_ptr(cb_out);

                // Each cb_out tile is 32x32 in TILE layout (4 faces of 16x16).
                // Row 0 of the logical Y dim sits in face 0/2 row 0 of the tile.
                // We copy that row (32 BF16 = 64 bytes) into row r of the dense
                // scratch tile (which has the same TILE layout). For r in 0..15
                // it goes into face 0; for r in 16..31 it goes into face 2.
                //
                // TILE layout face order (for 32x32 BF16): faces 0,1 are top
                // half (rows 0-15), faces 2,3 are bottom half (rows 16-31).
                // Each face is 16x16 in row-major.
                uint32_t face_offset = (r < 16) ? 0 : (2 * 16 * 16 * 2);  // bytes to bottom-half face 2
                uint32_t row_within_face = (r < 16) ? r : (r - 16);

                for (uint32_t vt = 0; vt < Vt; vt++) {
                    // Source: row 0 of source tile vt's face 0 (left half cols)
                    //         + row 0 of source tile vt's face 1 (right half cols)
                    uint32_t src_tile = src_base + vt * tile_bytes;
                    uint32_t dst_tile = scratch_base + vt * tile_bytes + face_offset;

                    // Face 0 (cols 0-15) row 0 -> dst face row r%16, cols 0-15
                    volatile tt_l1_ptr uint64_t* src_l = (volatile tt_l1_ptr uint64_t*)(src_tile + 0);
                    volatile tt_l1_ptr uint64_t* dst_l =
                        (volatile tt_l1_ptr uint64_t*)(dst_tile + row_within_face * 16 * 2);
                    for (uint32_t i = 0; i < 4; i++) {
                        dst_l[i] = src_l[i];
                    }

                    // Face 1 (cols 16-31) row 0 -> dst face row r%16, cols 16-31
                    volatile tt_l1_ptr uint64_t* src_r = (volatile tt_l1_ptr uint64_t*)(src_tile + 16 * 16 * 2);
                    volatile tt_l1_ptr uint64_t* dst_r =
                        (volatile tt_l1_ptr uint64_t*)(dst_tile + 16 * 16 * 2 + row_within_face * 16 * 2);
                    for (uint32_t i = 0; i < 4; i++) {
                        dst_r[i] = src_r[i];
                    }
                }

                cb_pop_front(cb_out, Vt);
            }

            // Write Vt scratch tiles to DRAM at this block's destination column strip.
            uint32_t dst_tile_base = block * (num_pairs_total * Vt) + pair * Vt;
            for (uint32_t vt = 0; vt < Vt; vt++) {
                noc_async_write_tile(dst_tile_base + vt, out_wr, scratch_base + vt * tile_bytes);
            }
            noc_async_write_barrier();
        }

        // State write — unchanged from old writer
        cb_wait_front(cb_state_out, state_tiles);
        uint32_t sp = get_read_ptr(cb_state_out);
        for (uint32_t s = 0; s < state_tiles; s++) {
            noc_async_write_tile(pair * state_tiles + s, state_wr, sp);
            sp += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_state_out, state_tiles);
    }
}
