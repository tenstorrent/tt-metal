// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The lane body's position-derived tensors from the uint32 position row P[0..31] (lane u = P_u), one core per lane
// row: derive.cpp's per-position arithmetic on row u of every row tensor (the indexer mask, the keep bits and the
// fill with lane u's KV region offset in its tail ids), lane u's kv_row_hit tile (lanes below the lane count), row u of
// the four RoPE tiles (the tables' rows P_u and P_u & ~3), and on core 0 the index row, the block-start row (P & ~3)
// and the kv_block_start row (P & ~31) of all 32 lanes.  The same NoC-assembled templates as derive.cpp (rows cut at
// the 64-byte DRAM read grain, boundary lanes stored by the RISC); no compute kernel.
// Named compile-time args: blocks, slots, block_topk, kv_row_mask, ring_mask, kv_block_start_mask, lane_block_mask,
// all_ones, one_bf16, rope_dim, cb_stage.  Compile-time args: TensorAccessorArgs for the 19 tensors below, chained
// from 0.  Runtime args: 0 position row, 1 lane offsets row (u * C for the active lanes), 2 bf16 templates [zeros |
// mask], 3 uint32 templates [zeros | ones], 4 tile templates [zero tile, ones-column tile], 5 cos table, 6 sin table,
// 7 kv_block_start row, 8 kv_row_hit tiles, 9 indexer_neg_mask rows, 10 row_keep_bits rows, 11 row_fill rows,
// 12 index_row, 13 block_start_row, 14 cos tiles, 15 sin tiles, 16 block-start cos tiles, 17 block-start sin tiles,
// 18 the rotary shift row (lane u's S_u; the RoPE table rows read are P_u - S_u and (P_u & ~3) - S_u, S_u <= P_u by
// the host's check) (buffer addresses), 19 lane, 20 lane count (lanes with a kv_row_hit tile).

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

constexpr uint32_t BLOCKS = get_named_compile_time_arg_val("blocks");
constexpr uint32_t SLOTS = get_named_compile_time_arg_val("slots");
constexpr uint32_t BLOCK_TOPK = get_named_compile_time_arg_val("block_topk");
constexpr uint32_t KV_ROW_MASK = get_named_compile_time_arg_val("kv_row_mask");
constexpr uint32_t RING_MASK = get_named_compile_time_arg_val("ring_mask");
constexpr uint32_t KV_BLOCK_START_MASK = get_named_compile_time_arg_val("kv_block_start_mask");
constexpr uint32_t LANE_BLOCK_MASK = get_named_compile_time_arg_val("lane_block_mask");
constexpr uint32_t ALL_ONES = get_named_compile_time_arg_val("all_ones");
constexpr uint16_t ONE_BF16 = get_named_compile_time_arg_val("one_bf16");
constexpr uint32_t ROPE_DIM = get_named_compile_time_arg_val("rope_dim");
constexpr uint32_t CB_STAGE = get_named_compile_time_arg_val("cb_stage");

constexpr uint32_t DRAM_READ_GRAIN = 64;
constexpr uint32_t TILE_BYTES = 2048;
constexpr uint32_t FACE_BYTES = 512;
constexpr uint32_t FACE_ROW_BYTES = 32;
constexpr uint32_t LANES = 32;
constexpr uint32_t BF16_ROW_BYTES = BLOCKS * 2;
constexpr uint32_t U32_ROW_BYTES = SLOTS * 4;
constexpr uint32_t ROPE_ROW_BYTES = ROPE_DIM * 2;
// staging layout (bytes from the reserved block)
constexpr uint32_t STAGE_BF16 = 0;
constexpr uint32_t STAGE_U32 = STAGE_BF16 + BF16_ROW_BYTES;
constexpr uint32_t STAGE_TILE = STAGE_U32 + ((U32_ROW_BYTES + 63) & ~63u);
constexpr uint32_t STAGE_ROPE = STAGE_TILE + TILE_BYTES;
constexpr uint32_t STAGE_ROWS =
    STAGE_ROPE + 4 * ROPE_ROW_BYTES;  // the position, offsets and shift rows, three output rows
constexpr uint32_t STAGE_BYTES = STAGE_ROWS + 6 * 128;

template <typename A>
FORCE_INLINE void read_bytes(
    Noc& noc,
    const A& src,
    DataflowBuffer& dfb,
    uint32_t bytes,
    uint32_t page,
    uint32_t page_offset,
    uint32_t l1_offset) {
    if (bytes) {
        noc.async_read(src, dfb, bytes, {.page_id = page, .offset_bytes = page_offset}, {.offset_bytes = l1_offset});
    }
}

template <typename A>
FORCE_INLINE void write_bytes(
    Noc& noc,
    DataflowBuffer& dfb,
    const A& dst,
    uint32_t bytes,
    uint32_t l1_offset,
    uint32_t page,
    uint32_t page_offset) {
    noc.async_write(dfb, dst, bytes, {.offset_bytes = l1_offset}, {.page_id = page, .offset_bytes = page_offset});
}

// row `lane` of the two-tile RoPE row tile [32, 64]: face f = (lane >> 4) * 2 + half at chunk (lane & 15) * 32
FORCE_INLINE uint32_t rope_face_offset(uint32_t lane, uint32_t half) {
    return (((lane >> 4) << 1) + half) * FACE_BYTES + (lane & 15) * FACE_ROW_BYTES;
}

void kernel_main() {
    constexpr auto a_p = TensorAccessorArgs<0>();
    constexpr auto a_off = TensorAccessorArgs<a_p.next_compile_time_args_offset()>();
    constexpr auto a_bf16 = TensorAccessorArgs<a_off.next_compile_time_args_offset()>();
    constexpr auto a_u32 = TensorAccessorArgs<a_bf16.next_compile_time_args_offset()>();
    constexpr auto a_tiles = TensorAccessorArgs<a_u32.next_compile_time_args_offset()>();
    constexpr auto a_cos = TensorAccessorArgs<a_tiles.next_compile_time_args_offset()>();
    constexpr auto a_sin = TensorAccessorArgs<a_cos.next_compile_time_args_offset()>();
    constexpr auto a_kvbs = TensorAccessorArgs<a_sin.next_compile_time_args_offset()>();
    constexpr auto a_kvhit = TensorAccessorArgs<a_kvbs.next_compile_time_args_offset()>();
    constexpr auto a_mask = TensorAccessorArgs<a_kvhit.next_compile_time_args_offset()>();
    constexpr auto a_keep = TensorAccessorArgs<a_mask.next_compile_time_args_offset()>();
    constexpr auto a_fill = TensorAccessorArgs<a_keep.next_compile_time_args_offset()>();
    constexpr auto a_irow = TensorAccessorArgs<a_fill.next_compile_time_args_offset()>();
    constexpr auto a_brow = TensorAccessorArgs<a_irow.next_compile_time_args_offset()>();
    constexpr auto a_ocos = TensorAccessorArgs<a_brow.next_compile_time_args_offset()>();
    constexpr auto a_osin = TensorAccessorArgs<a_ocos.next_compile_time_args_offset()>();
    constexpr auto a_obcos = TensorAccessorArgs<a_osin.next_compile_time_args_offset()>();
    constexpr auto a_obsin = TensorAccessorArgs<a_obcos.next_compile_time_args_offset()>();
    constexpr auto a_shift = TensorAccessorArgs<a_obsin.next_compile_time_args_offset()>();

    const auto p_in = TensorAccessor(a_p, get_arg_val<uint32_t>(0));
    const auto off_in = TensorAccessor(a_off, get_arg_val<uint32_t>(1));
    const auto bf16_tpl = TensorAccessor(a_bf16, get_arg_val<uint32_t>(2));
    const auto u32_tpl = TensorAccessor(a_u32, get_arg_val<uint32_t>(3));
    const auto tile_tpl = TensorAccessor(a_tiles, get_arg_val<uint32_t>(4));
    const auto cos_tbl = TensorAccessor(a_cos, get_arg_val<uint32_t>(5));
    const auto sin_tbl = TensorAccessor(a_sin, get_arg_val<uint32_t>(6));
    const auto o_kvbs = TensorAccessor(a_kvbs, get_arg_val<uint32_t>(7));
    const auto o_kvhit = TensorAccessor(a_kvhit, get_arg_val<uint32_t>(8));
    const auto o_mask = TensorAccessor(a_mask, get_arg_val<uint32_t>(9));
    const auto o_keep = TensorAccessor(a_keep, get_arg_val<uint32_t>(10));
    const auto o_fill = TensorAccessor(a_fill, get_arg_val<uint32_t>(11));
    const auto o_irow = TensorAccessor(a_irow, get_arg_val<uint32_t>(12));
    const auto o_brow = TensorAccessor(a_brow, get_arg_val<uint32_t>(13));
    const auto o_cos = TensorAccessor(a_ocos, get_arg_val<uint32_t>(14));
    const auto o_sin = TensorAccessor(a_osin, get_arg_val<uint32_t>(15));
    const auto o_bcos = TensorAccessor(a_obcos, get_arg_val<uint32_t>(16));
    const auto o_bsin = TensorAccessor(a_obsin, get_arg_val<uint32_t>(17));
    const auto shift_in = TensorAccessor(a_shift, get_arg_val<uint32_t>(18));
    const uint32_t lane = get_arg_val<uint32_t>(19);
    const uint32_t lane_count = get_arg_val<uint32_t>(20);

    Noc noc;
    DataflowBuffer stage(CB_STAGE);
    stage.reserve_back(1);
    const uint32_t base = stage.get_write_ptr();
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    volatile tt_l1_ptr uint16_t* halves = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base);
    constexpr uint32_t ROW_P = STAGE_ROWS, ROW_OFF = STAGE_ROWS + 128, ROW_I = STAGE_ROWS + 256,
                       ROW_B = STAGE_ROWS + 384, ROW_KV = STAGE_ROWS + 512, ROW_SH = STAGE_ROWS + 640;

    {
        FUSED_ZONE("fz_pd_rl_setup");
        // the position row, the offsets row, the shift row and the whole template rows / tile in one batch of reads
        read_bytes(noc, p_in, stage, 128, 0, 0, ROW_P);
        read_bytes(noc, off_in, stage, 128, 0, 0, ROW_OFF);
        read_bytes(noc, shift_in, stage, 128, 0, 0, ROW_SH);
        read_bytes(noc, bf16_tpl, stage, BF16_ROW_BYTES, 0, BF16_ROW_BYTES, STAGE_BF16);  // the MASK half
        read_bytes(noc, tile_tpl, stage, TILE_BYTES, 0, 0, STAGE_TILE);                   // zero tile
        noc.async_read_barrier();
    }

    const uint32_t p = words[ROW_P / 4 + lane];
    const uint32_t offset = words[ROW_OFF / 4 + lane];
    const uint32_t kv_row = p & KV_ROW_MASK;
    const uint32_t context = p + 1;
    const uint32_t complete_blocks = context >> 2;
    const uint32_t selected = complete_blocks < BLOCK_TOPK ? complete_blocks : BLOCK_TOPK;
    const uint32_t lo = selected << 2;
    const uint32_t hi = lo + (context & RING_MASK);
    const uint32_t tail_shift = (complete_blocks - selected) << 2;
    const uint32_t block_start = p & LANE_BLOCK_MASK;
    // the RoPE table rows of this lane: P_u - S_u and (P_u & ~3) - S_u
    const uint32_t shift = words[ROW_SH / 4 + lane];
    const uint32_t rope_p = p - shift;
    const uint32_t rope_block_start = block_start - shift;

    {
        FUSED_ZONE("fz_pd_rl_main");
        if (lane == 0) {  // the three 32-lane rows of the whole position row
            for (uint32_t u = 0; u < LANES; ++u) {
                const uint32_t pu = words[ROW_P / 4 + u];
                words[ROW_I / 4 + u] = pu;
                words[ROW_B / 4 + u] = pu & LANE_BLOCK_MASK;
                words[ROW_KV / 4 + u] = pu & KV_BLOCK_START_MASK;
            }
            write_bytes(noc, stage, o_irow, 128, ROW_I, 0, 0);
            write_bytes(noc, stage, o_brow, 128, ROW_B, 0, 0);
            write_bytes(noc, stage, o_kvbs, 128, ROW_KV, 0, 0);
        }

        // indexer_neg_mask row: MASK everywhere, +0.0 for blocks below complete_blocks (the chain's 0 * MASK is +0.0)
        {
            const uint32_t zero_bytes = complete_blocks * 2;
            const uint32_t bulk = zero_bytes & ~(DRAM_READ_GRAIN - 1);
            read_bytes(noc, bf16_tpl, stage, bulk, 0, 0, STAGE_BF16);
            noc.async_read_barrier();
            for (uint32_t k = bulk / 2; k < complete_blocks; ++k) {
                halves[STAGE_BF16 / 2 + k] = 0;
            }
        }
        write_bytes(noc, stage, o_mask, BF16_ROW_BYTES, STAGE_BF16, lane, 0);
        // kv_row_hit tile of this lane (lanes with a tile): the zero tile with lane (kv_row, 0) = 1.0
        if (lane < lane_count) {
            const uint32_t kv_lane = (kv_row >> 4) * 512 + (kv_row & 15) * 16;
            halves[STAGE_TILE / 2 + kv_lane] = ONE_BF16;
            write_bytes(noc, stage, o_kvhit, TILE_BYTES, STAGE_TILE, lane, 0);
        }

        // row_keep_bits row: zeros, ALL_ONES below lo
        read_bytes(noc, u32_tpl, stage, U32_ROW_BYTES, 0, 0, STAGE_U32);
        noc.async_read_barrier();
        {
            const uint32_t bulk = (lo * 4) & ~(DRAM_READ_GRAIN - 1);
            read_bytes(noc, u32_tpl, stage, bulk, 0, U32_ROW_BYTES, STAGE_U32);
            noc.async_read_barrier();
            for (uint32_t k = bulk / 4; k < lo; ++k) {
                words[STAGE_U32 / 4 + k] = ALL_ONES;
            }
        }
        noc.async_write_barrier();  // the mask row and the hit tile are out before their staging is reused
        write_bytes(noc, stage, o_keep, U32_ROW_BYTES, STAGE_U32, lane, 0);
        noc.async_write_barrier();

        // row_fill row: ALL_ONES from hi, slot + offset + tail_shift on [lo, hi), zeros below lo
        read_bytes(noc, u32_tpl, stage, U32_ROW_BYTES, 0, U32_ROW_BYTES, STAGE_U32);
        noc.async_read_barrier();
        {
            const uint32_t bulk = (hi * 4) & ~(DRAM_READ_GRAIN - 1);
            read_bytes(noc, u32_tpl, stage, bulk, 0, 0, STAGE_U32);
            noc.async_read_barrier();
            for (uint32_t k = bulk / 4; k < hi; ++k) {
                words[STAGE_U32 / 4 + k] = 0;
            }
            for (uint32_t k = lo; k < hi; ++k) {
                words[STAGE_U32 / 4 + k] = k + offset + tail_shift;
            }
        }
        write_bytes(noc, stage, o_fill, U32_ROW_BYTES, STAGE_U32, lane, 0);

        // RoPE rows: table rows P_u - S_u and (P_u & ~3) - S_u into row `lane` of the two output tiles (faces
        // (lane >> 4) * 2 and + 1)
        read_bytes(noc, cos_tbl, stage, ROPE_ROW_BYTES, rope_p, 0, STAGE_ROPE);
        read_bytes(noc, sin_tbl, stage, ROPE_ROW_BYTES, rope_p, 0, STAGE_ROPE + ROPE_ROW_BYTES);
        read_bytes(noc, cos_tbl, stage, ROPE_ROW_BYTES, rope_block_start, 0, STAGE_ROPE + 2 * ROPE_ROW_BYTES);
        read_bytes(noc, sin_tbl, stage, ROPE_ROW_BYTES, rope_block_start, 0, STAGE_ROPE + 3 * ROPE_ROW_BYTES);
        noc.async_read_barrier();
        const uint32_t rope_src[4] = {
            STAGE_ROPE, STAGE_ROPE + ROPE_ROW_BYTES, STAGE_ROPE + 2 * ROPE_ROW_BYTES, STAGE_ROPE + 3 * ROPE_ROW_BYTES};
        for (uint32_t face = 0; face < ROPE_DIM / 16;
             ++face) {  // face f: lanes 16f..16f+15 -> tile f/2, face row of half f%2
            const uint32_t page = face >> 1, offset_bytes = rope_face_offset(lane, face & 1);
            write_bytes(noc, stage, o_cos, FACE_ROW_BYTES, rope_src[0] + face * FACE_ROW_BYTES, page, offset_bytes);
            write_bytes(noc, stage, o_sin, FACE_ROW_BYTES, rope_src[1] + face * FACE_ROW_BYTES, page, offset_bytes);
            write_bytes(noc, stage, o_bcos, FACE_ROW_BYTES, rope_src[2] + face * FACE_ROW_BYTES, page, offset_bytes);
            write_bytes(noc, stage, o_bsin, FACE_ROW_BYTES, rope_src[3] + face * FACE_ROW_BYTES, page, offset_bytes);
        }
        noc.async_write_barrier();
    }
    stage.push_back(1);
}
