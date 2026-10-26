// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the fused gather + combine-reduce (tt-blaze G53F-0190). Replaces the post_combine_reduce reader
// (deepseek_prefill/post_combine_reduce/device/kernels/deepseek_moe_post_combine_reduce_reader.cpp) and the
// ttnn.embedding gather in front of it: instead of reading comb[t, j] (a gathered copy of table[pos[t, j]]) it reads
// table[pos[t, j]] directly, and only the rows the compute kernel uses. The compute kernel (unchanged) always processes
// a token's pick 0 and skips every later pick whose routing weight is zero; in the regions MoE the weight is zero
// exactly at the non-local picks, whose slot is the dummy slot, so a pick j > 0 at the dummy slot is not read (its CB
// rows are pushed unfilled and popped unread). The rows the compute does read are byte-identical to comb's, so the
// result is bit-identical. A token's (up to k) row reads are in flight together, one barrier per token.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"

constexpr uint32_t cb_combine_input_id = tt::CBIndex::c_0;
constexpr uint32_t num_experts = get_compile_time_arg_val(0);
constexpr uint32_t emb_dim_cb_tiles = get_compile_time_arg_val(1);
constexpr uint32_t emb_dim_bytes = get_compile_time_arg_val(2);
constexpr uint32_t dummy_slot = get_compile_time_arg_val(3);
constexpr uint32_t cb_pos_id = get_compile_time_arg_val(4);
constexpr uint32_t pos_page_bytes = get_compile_time_arg_val(5);
constexpr uint32_t tile_bytes = get_compile_time_arg_val(6);
// POS_TILE (G53F-0193): pos is the [1,1,T,k] fp32 TILE slot table itself (one 32x32 tile per 32-token chunk), not its
// uint32 ROW_MAJOR copy -- the typecast + untilize that made the copy are skipped
constexpr bool POS_TILE = get_compile_time_arg_val(7) != 0;
constexpr auto table_args = TensorAccessorArgs<8>();
constexpr auto pos_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
// the chunk split (G53F-0206): TOTAL_CHUNKS chunks over NCORES cores of a GRID_X-wide grid, row-major
constexpr uint32_t TOTAL_CHUNKS = get_compile_time_arg_val(pos_args.next_compile_time_args_offset());
constexpr uint32_t NCORES = get_compile_time_arg_val(pos_args.next_compile_time_args_offset() + 1);
constexpr uint32_t GRID_X = get_compile_time_arg_val(pos_args.next_compile_time_args_offset() + 2);
constexpr uint32_t TOKENS_PER_CHUNK = 32;
constexpr uint32_t chunk_pos_bytes = POS_TILE ? pos_page_bytes : TOKENS_PER_CHUNK * num_experts * 4;

// the fp32 bit pattern of a non-negative integer value below 2^24 -> that integer (exact; no FPU on the data movers)
FORCE_INLINE uint32_t f32_to_u32(uint32_t bits) {
    const uint32_t e = (bits >> 23) & 0xFFu;
    if (e < 127u) {
        return 0u;
    }
    const uint32_t m = (bits & 0x7FFFFFu) | 0x800000u;
    return e >= 150u ? (m << (e - 150u)) : (m >> (150u - e));
}

FORCE_INLINE uint32_t slot_of(const CoreLocalMem<volatile uint32_t>& pos, uint32_t t, uint32_t j) {
    if constexpr (POS_TILE) {
        const uint32_t face = t < 16u ? 0u : 2u;  // rows 0-15 in faces 0/1, 16-31 in faces 2/3; j < 16: left face
        return f32_to_u32(pos[face * 256u + (t & 15u) * 16u + j]);
    } else {
        return pos[t * num_experts + j];
    }
}

void kernel_main() {
    // the buffer addresses are common runtime args (one list for every core), the token span from the core's position
    // (G53F-0206)
    const uint32_t table_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t pos_addr = get_common_arg_val<uint32_t>(1);
    // the core's chunk span: an even split of TOTAL_CHUNKS over NCORES cores (row-major, the first TOTAL_CHUNKS %
    // NCORES take one more) from the core's logical position -- no per-core runtime args (tt-blaze G53F-0206)
    const uint32_t core_i = get_absolute_logical_y() * GRID_X + get_absolute_logical_x();
    const uint32_t base_ = TOTAL_CHUNKS / NCORES, extra_ = TOTAL_CHUNKS % NCORES;
    const uint32_t num_chunks = base_ + (core_i < extra_ ? 1u : 0u);
    uint32_t token_start_idx = (core_i * base_ + (core_i < extra_ ? core_i : extra_)) * 32u;

    const auto s_table = TensorAccessor(table_args, table_addr, emb_dim_bytes);
    const auto s_pos = TensorAccessor(pos_args, pos_addr, pos_page_bytes);

    Noc noc;
    CircularBuffer cb_in(cb_combine_input_id), cb_pos(cb_pos_id);
    cb_pos.reserve_back(1);  // private scratch: reserved once, never pushed
    const uint32_t pos_base = get_write_ptr(cb_pos_id);
    const CoreLocalMem<volatile uint32_t> pos(pos_base);

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        // the chunk's 32 x k slots: one contiguous read (token_start is a multiple of 32: 1 KB aligned at k = 8)
        if constexpr (POS_TILE) {
            noc.async_read(
                s_pos,
                CoreLocalMem<uint32_t>(pos_base),
                chunk_pos_bytes,
                {.page_id = token_start_idx / TOKENS_PER_CHUNK},
                {});
        } else {
            noc.async_read(
                s_pos,
                CoreLocalMem<uint32_t>(pos_base),
                chunk_pos_bytes,
                {.page_id = 0, .offset_bytes = token_start_idx * num_experts * 4u},
                {});
        }
        noc.async_read_barrier();
        for (uint32_t t = 0; t < TOKENS_PER_CHUNK; ++t) {
            cb_in.reserve_back(num_experts * emb_dim_cb_tiles);
            for (uint32_t j = 0; j < num_experts; ++j) {
                const uint32_t slot = slot_of(pos, t, j);
                if (j == 0 || slot != dummy_slot) {
                    noc.async_read(
                        s_table,
                        cb_in,
                        emb_dim_bytes,
                        {.page_id = slot},
                        {.offset_bytes = j * emb_dim_cb_tiles * tile_bytes});
                }
            }
            noc.async_read_barrier();
            cb_in.push_back(num_experts * emb_dim_cb_tiles);
        }
        token_start_idx += TOKENS_PER_CHUNK;
    }
}
