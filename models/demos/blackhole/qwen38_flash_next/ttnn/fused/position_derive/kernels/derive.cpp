// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The decode step's position-derived tensors from the uint32 position P, assembled in L1 from constant template rows
// (zeros | MASK bf16 lanes, zeros | ALL_ONES uint32 lanes, a zero tile and a ones-column tile) by NoC reads, a few
// boundary stores, and one NoC write per output page.  No compute kernel.  Row ranges copied over the NoC are cut at
// the 64-byte DRAM read granularity; the remainder lanes are stored by the RISC.
// Named compile-time args: blocks, slots, block_topk, kv_row_mask, ring_mask, kv_block_start_mask, lane_block_mask,
// all_ones, one_bf16, rope_dim, cb_stage.  Compile-time args: TensorAccessorArgs for the 22 tensors below, chained
// from 0.  Runtime args: 0 P, 1 bf16 templates [zeros | mask], 2 uint32 templates [zeros | ones], 3 tile templates
// [zero tile, ones-column tile], 4 cos table, 5 sin table, 6 kv_block_start, 7 kv_row_hit, 8 kv_row_keep, 9 ring_hit,
// 10 ring_keep, 11 block_index_i32, 12 indexer_neg_mask, 13 row_keep_bits, 14 row_fill, 15 index_row,
// 16 block_start_row, 17 cos, 18 sin, 19 block-start cos, 20 block-start sin, 21 the rotary shift S (uint32
// [1,1,1,1]; the RoPE table rows read are P - S and (P & ~3) - S, S <= P by the host's check) (buffer addresses).

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
constexpr uint32_t FACE_ROW_BYTES = 32;
constexpr uint32_t BF16_ROW_BYTES = BLOCKS * 2;
constexpr uint32_t U32_ROW_BYTES = SLOTS * 4;
constexpr uint32_t ROPE_ROW_BYTES = ROPE_DIM * 2;
// staging layout (bytes from the reserved block)
constexpr uint32_t STAGE_BF16 = 0;
constexpr uint32_t STAGE_U32 = STAGE_BF16 + BF16_ROW_BYTES;
constexpr uint32_t STAGE_TILES = STAGE_U32 + ((U32_ROW_BYTES + 63) & ~63u);
constexpr uint32_t STAGE_ROPE = STAGE_TILES + 2 * TILE_BYTES;
constexpr uint32_t STAGE_INDEX = STAGE_ROPE + 4 * ROPE_ROW_BYTES;
constexpr uint32_t STAGE_SCALARS = STAGE_INDEX + 2 * 128;
constexpr uint32_t STAGE_SHIFT = STAGE_SCALARS + 64;  // the rotary shift's 64-byte read grain
constexpr uint32_t STAGE_BYTES = STAGE_SHIFT + 64;

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

void kernel_main() {
    constexpr auto a_p = TensorAccessorArgs<0>();
    constexpr auto a_bf16 = TensorAccessorArgs<a_p.next_compile_time_args_offset()>();
    constexpr auto a_u32 = TensorAccessorArgs<a_bf16.next_compile_time_args_offset()>();
    constexpr auto a_tiles = TensorAccessorArgs<a_u32.next_compile_time_args_offset()>();
    constexpr auto a_cos = TensorAccessorArgs<a_tiles.next_compile_time_args_offset()>();
    constexpr auto a_sin = TensorAccessorArgs<a_cos.next_compile_time_args_offset()>();
    constexpr auto a_kvbs = TensorAccessorArgs<a_sin.next_compile_time_args_offset()>();
    constexpr auto a_kvhit = TensorAccessorArgs<a_kvbs.next_compile_time_args_offset()>();
    constexpr auto a_kvkeep = TensorAccessorArgs<a_kvhit.next_compile_time_args_offset()>();
    constexpr auto a_rhit = TensorAccessorArgs<a_kvkeep.next_compile_time_args_offset()>();
    constexpr auto a_rkeep = TensorAccessorArgs<a_rhit.next_compile_time_args_offset()>();
    constexpr auto a_bidx = TensorAccessorArgs<a_rkeep.next_compile_time_args_offset()>();
    constexpr auto a_mask = TensorAccessorArgs<a_bidx.next_compile_time_args_offset()>();
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
    const auto bf16_tpl = TensorAccessor(a_bf16, get_arg_val<uint32_t>(1));
    const auto u32_tpl = TensorAccessor(a_u32, get_arg_val<uint32_t>(2));
    const auto tile_tpl = TensorAccessor(a_tiles, get_arg_val<uint32_t>(3));
    const auto cos_tbl = TensorAccessor(a_cos, get_arg_val<uint32_t>(4));
    const auto sin_tbl = TensorAccessor(a_sin, get_arg_val<uint32_t>(5));
    const auto o_kvbs = TensorAccessor(a_kvbs, get_arg_val<uint32_t>(6));
    const auto o_kvhit = TensorAccessor(a_kvhit, get_arg_val<uint32_t>(7));
    const auto o_kvkeep = TensorAccessor(a_kvkeep, get_arg_val<uint32_t>(8));
    const auto o_rhit = TensorAccessor(a_rhit, get_arg_val<uint32_t>(9));
    const auto o_rkeep = TensorAccessor(a_rkeep, get_arg_val<uint32_t>(10));
    const auto o_bidx = TensorAccessor(a_bidx, get_arg_val<uint32_t>(11));
    const auto o_mask = TensorAccessor(a_mask, get_arg_val<uint32_t>(12));
    const auto o_keep = TensorAccessor(a_keep, get_arg_val<uint32_t>(13));
    const auto o_fill = TensorAccessor(a_fill, get_arg_val<uint32_t>(14));
    const auto o_irow = TensorAccessor(a_irow, get_arg_val<uint32_t>(15));
    const auto o_brow = TensorAccessor(a_brow, get_arg_val<uint32_t>(16));
    const auto o_cos = TensorAccessor(a_ocos, get_arg_val<uint32_t>(17));
    const auto o_sin = TensorAccessor(a_osin, get_arg_val<uint32_t>(18));
    const auto o_bcos = TensorAccessor(a_obcos, get_arg_val<uint32_t>(19));
    const auto o_bsin = TensorAccessor(a_obsin, get_arg_val<uint32_t>(20));
    const auto shift_in = TensorAccessor(a_shift, get_arg_val<uint32_t>(21));

    Noc noc;
    DataflowBuffer stage(CB_STAGE);
    stage.reserve_back(1);
    const uint32_t base = stage.get_write_ptr();
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    volatile tt_l1_ptr uint16_t* halves = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base);

    {
        FUSED_ZONE("fz_pd_r_setup");
        // P, the rotary shift S and the whole template rows / tiles in one batch of reads
        read_bytes(noc, p_in, stage, DRAM_READ_GRAIN, 0, 0, STAGE_SCALARS);
        read_bytes(noc, shift_in, stage, DRAM_READ_GRAIN, 0, 0, STAGE_SHIFT);
        read_bytes(noc, bf16_tpl, stage, BF16_ROW_BYTES, 0, BF16_ROW_BYTES, STAGE_BF16);  // the MASK half
        read_bytes(noc, tile_tpl, stage, TILE_BYTES, 0, 0, STAGE_TILES);                  // zero tile
        read_bytes(noc, tile_tpl, stage, TILE_BYTES, 1, 0, STAGE_TILES + TILE_BYTES);     // ones-column tile
        noc.async_read_barrier();
    }

    const uint32_t p = words[STAGE_SCALARS / 4];
    const uint32_t kv_row = p & KV_ROW_MASK;
    const uint32_t ring_row = p & RING_MASK;
    const uint32_t context = p + 1;
    const uint32_t complete_blocks = context >> 2;
    const uint32_t selected = complete_blocks < BLOCK_TOPK ? complete_blocks : BLOCK_TOPK;
    const uint32_t lo = selected << 2;
    const uint32_t hi = lo + (context & RING_MASK);
    const uint32_t tail_shift = (complete_blocks - selected) << 2;
    const uint32_t block_start = p & LANE_BLOCK_MASK;
    // the RoPE table rows: the token's rotary position P - S and its index block's first token's (P & ~3) - S
    const uint32_t shift = words[STAGE_SHIFT / 4];
    const uint32_t rope_p = p - shift;
    const uint32_t rope_block_start = block_start - shift;

    {
        FUSED_ZONE("fz_pd_r_main");
        // scalars and index rows
        words[STAGE_SCALARS / 4 + 4] = p & KV_BLOCK_START_MASK;
        words[STAGE_SCALARS / 4 + 8] = p >> 2;
        for (uint32_t lane = 0; lane < 32; ++lane) {
            words[STAGE_INDEX / 4 + lane] = p;
            words[STAGE_INDEX / 4 + 32 + lane] = block_start;
        }
        write_bytes(noc, stage, o_kvbs, 4, STAGE_SCALARS + 16, 0, 0);
        write_bytes(noc, stage, o_bidx, 4, STAGE_SCALARS + 32, 0, 0);
        write_bytes(noc, stage, o_irow, 128, STAGE_INDEX, 0, 0);
        write_bytes(noc, stage, o_brow, 128, STAGE_INDEX + 128, 0, 0);

        // indexer_neg_mask: MASK everywhere, +0.0 for blocks below complete_blocks (the chain's 0 * MASK is +0.0)
        {
            const uint32_t zero_bytes = complete_blocks * 2;
            const uint32_t bulk = zero_bytes & ~(DRAM_READ_GRAIN - 1);
            read_bytes(noc, bf16_tpl, stage, bulk, 0, 0, STAGE_BF16);
            for (uint32_t lane = bulk / 2; lane < complete_blocks; ++lane) {
                halves[STAGE_BF16 / 2 + lane] = 0;
            }
        }
        // the four one-hots: tile lane (row, 0) is 16-bit word (row >> 4) * 512 + (row & 15) * 16
        const uint32_t kv_lane = (kv_row >> 4) * 512 + (kv_row & 15) * 16;
        const uint32_t ring_lane = (ring_row >> 4) * 512 + (ring_row & 15) * 16;
        volatile tt_l1_ptr uint16_t* zero_tile = halves + STAGE_TILES / 2;
        volatile tt_l1_ptr uint16_t* ones_tile = halves + (STAGE_TILES + TILE_BYTES) / 2;
        noc.async_read_barrier();  // the zero-lane bulk read landed before the mask row is written
        write_bytes(noc, stage, o_mask, BF16_ROW_BYTES, STAGE_BF16, 0, 0);
        zero_tile[kv_lane] = ONE_BF16;
        write_bytes(noc, stage, o_kvhit, TILE_BYTES, STAGE_TILES, 0, 0);
        ones_tile[kv_lane] = 0;
        write_bytes(noc, stage, o_kvkeep, TILE_BYTES, STAGE_TILES + TILE_BYTES, 0, 0);
        noc.async_write_barrier();
        zero_tile[kv_lane] = 0;
        ones_tile[kv_lane] = ONE_BF16;
        zero_tile[ring_lane] = ONE_BF16;
        write_bytes(noc, stage, o_rhit, TILE_BYTES, STAGE_TILES, 0, 0);
        ones_tile[ring_lane] = 0;
        write_bytes(noc, stage, o_rkeep, TILE_BYTES, STAGE_TILES + TILE_BYTES, 0, 0);

        // row_keep_bits: zeros, ALL_ONES below lo
        read_bytes(noc, u32_tpl, stage, U32_ROW_BYTES, 0, 0, STAGE_U32);
        noc.async_read_barrier();
        {
            const uint32_t bulk = (lo * 4) & ~(DRAM_READ_GRAIN - 1);
            read_bytes(noc, u32_tpl, stage, bulk, 0, U32_ROW_BYTES, STAGE_U32);
            noc.async_read_barrier();
            for (uint32_t lane = bulk / 4; lane < lo; ++lane) {
                words[STAGE_U32 / 4 + lane] = ALL_ONES;
            }
        }
        noc.async_write_barrier();  // the ring tiles are out before the staging tiles change again (none do) and before
                                    // the u32 row is written
        write_bytes(noc, stage, o_keep, U32_ROW_BYTES, STAGE_U32, 0, 0);
        noc.async_write_barrier();

        // row_fill: ALL_ONES from hi, slot + tail_shift on [lo, hi), zeros below lo
        read_bytes(noc, u32_tpl, stage, U32_ROW_BYTES, 0, U32_ROW_BYTES, STAGE_U32);
        noc.async_read_barrier();
        {
            const uint32_t bulk = (hi * 4) & ~(DRAM_READ_GRAIN - 1);
            read_bytes(noc, u32_tpl, stage, bulk, 0, 0, STAGE_U32);
            noc.async_read_barrier();
            for (uint32_t lane = bulk / 4; lane < hi; ++lane) {
                words[STAGE_U32 / 4 + lane] = 0;
            }
            for (uint32_t lane = lo; lane < hi; ++lane) {
                words[STAGE_U32 / 4 + lane] = lane + tail_shift;
            }
        }
        write_bytes(noc, stage, o_fill, U32_ROW_BYTES, STAGE_U32, 0, 0);

        // RoPE rows: table rows P - S and (P & ~3) - S into row 0 of the two output tiles (faces 0 and 1 of each tile)
        read_bytes(noc, cos_tbl, stage, ROPE_ROW_BYTES, rope_p, 0, STAGE_ROPE);
        read_bytes(noc, sin_tbl, stage, ROPE_ROW_BYTES, rope_p, 0, STAGE_ROPE + ROPE_ROW_BYTES);
        read_bytes(noc, cos_tbl, stage, ROPE_ROW_BYTES, rope_block_start, 0, STAGE_ROPE + 2 * ROPE_ROW_BYTES);
        read_bytes(noc, sin_tbl, stage, ROPE_ROW_BYTES, rope_block_start, 0, STAGE_ROPE + 3 * ROPE_ROW_BYTES);
        noc.async_read_barrier();
        const uint32_t rope_src[4] = {
            STAGE_ROPE, STAGE_ROPE + ROPE_ROW_BYTES, STAGE_ROPE + 2 * ROPE_ROW_BYTES, STAGE_ROPE + 3 * ROPE_ROW_BYTES};
        for (uint32_t face = 0; face < ROPE_DIM / 16;
             ++face) {  // face f: lanes 16f..16f+15 -> tile f/2, face f%2 row 0
            const uint32_t page = face >> 1, offset = (face & 1) * 512;
            write_bytes(noc, stage, o_cos, FACE_ROW_BYTES, rope_src[0] + face * FACE_ROW_BYTES, page, offset);
            write_bytes(noc, stage, o_sin, FACE_ROW_BYTES, rope_src[1] + face * FACE_ROW_BYTES, page, offset);
            write_bytes(noc, stage, o_bcos, FACE_ROW_BYTES, rope_src[2] + face * FACE_ROW_BYTES, page, offset);
            write_bytes(noc, stage, o_bsin, FACE_ROW_BYTES, rope_src[3] + face * FACE_ROW_BYTES, page, offset);
        }
        noc.async_write_barrier();
    }
    stage.push_back(1);
}
