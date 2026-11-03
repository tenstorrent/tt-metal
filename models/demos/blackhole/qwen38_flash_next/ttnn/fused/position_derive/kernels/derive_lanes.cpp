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
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t blocks = get_named_compile_time_arg_val("blocks");
constexpr uint32_t slots = get_named_compile_time_arg_val("slots");
constexpr uint32_t block_topk = get_named_compile_time_arg_val("block_topk");
constexpr uint32_t kv_row_mask = get_named_compile_time_arg_val("kv_row_mask");
constexpr uint32_t ring_mask = get_named_compile_time_arg_val("ring_mask");
constexpr uint32_t kv_block_start_mask = get_named_compile_time_arg_val("kv_block_start_mask");
constexpr uint32_t lane_block_mask = get_named_compile_time_arg_val("lane_block_mask");
constexpr uint32_t all_ones = get_named_compile_time_arg_val("all_ones");
constexpr uint16_t one_bf16 = get_named_compile_time_arg_val("one_bf16");
constexpr uint32_t rope_dim = get_named_compile_time_arg_val("rope_dim");
constexpr uint32_t cb_stage = get_named_compile_time_arg_val("cb_stage");

constexpr uint32_t dram_read_grain = 64;
constexpr uint32_t bf16_tile_bytes = 2048;
constexpr uint32_t face_bytes = 512;
constexpr uint32_t face_row_bytes = 32;
constexpr uint32_t lanes = 32;
constexpr uint32_t bf16_row_bytes = blocks * 2;
constexpr uint32_t u32_row_bytes = slots * 4;
constexpr uint32_t rope_row_bytes = rope_dim * 2;
// staging layout (bytes from the reserved block)
constexpr uint32_t stage_bf16 = 0;
constexpr uint32_t stage_u32 = stage_bf16 + bf16_row_bytes;
constexpr uint32_t stage_tile = stage_u32 + ((u32_row_bytes + 63) & ~63u);
constexpr uint32_t stage_rope = stage_tile + bf16_tile_bytes;
constexpr uint32_t stage_rows =
    stage_rope + 4 * rope_row_bytes;  // the position, offsets and shift rows, three output rows
constexpr uint32_t stage_bytes = stage_rows + 6 * 128;

template <typename Accessor>
FORCE_INLINE void read_bytes(
    const Accessor& src,
    DataflowBuffer& buffer,
    uint32_t bytes,
    uint32_t page,
    uint32_t page_offset,
    uint32_t l1_offset) {
    const Noc noc;
    if (bytes) {
        noc.async_read(src, buffer, bytes, {.page_id = page, .offset_bytes = page_offset}, {.offset_bytes = l1_offset});
    }
}

template <typename Accessor>
FORCE_INLINE void write_bytes(
    DataflowBuffer& buffer,
    const Accessor& dst,
    uint32_t bytes,
    uint32_t l1_offset,
    uint32_t page,
    uint32_t page_offset) {
    const Noc noc;
    noc.async_write(buffer, dst, bytes, {.offset_bytes = l1_offset}, {.page_id = page, .offset_bytes = page_offset});
}

// row `lane` of the two-tile RoPE row tile [32, 64]: face f = (lane >> 4) * 2 + half at chunk (lane & 15) * 32
FORCE_INLINE uint32_t rope_face_offset(uint32_t lane, uint32_t half) {
    return (((lane >> 4) << 1) + half) * face_bytes + (lane & 15) * face_row_bytes;
}

void kernel_main() {
    constexpr auto position_args = TensorAccessorArgs<0>();
    constexpr auto offsets_args = TensorAccessorArgs<position_args.next_compile_time_args_offset()>();
    constexpr auto bf16_templates_args = TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();
    constexpr auto u32_templates_args = TensorAccessorArgs<bf16_templates_args.next_compile_time_args_offset()>();
    constexpr auto tile_templates_args = TensorAccessorArgs<u32_templates_args.next_compile_time_args_offset()>();
    constexpr auto cos_table_args = TensorAccessorArgs<tile_templates_args.next_compile_time_args_offset()>();
    constexpr auto sin_table_args = TensorAccessorArgs<cos_table_args.next_compile_time_args_offset()>();
    constexpr auto kv_block_start_args = TensorAccessorArgs<sin_table_args.next_compile_time_args_offset()>();
    constexpr auto kv_row_hit_args = TensorAccessorArgs<kv_block_start_args.next_compile_time_args_offset()>();
    constexpr auto indexer_neg_mask_args = TensorAccessorArgs<kv_row_hit_args.next_compile_time_args_offset()>();
    constexpr auto row_keep_bits_args = TensorAccessorArgs<indexer_neg_mask_args.next_compile_time_args_offset()>();
    constexpr auto row_fill_args = TensorAccessorArgs<row_keep_bits_args.next_compile_time_args_offset()>();
    constexpr auto index_row_args = TensorAccessorArgs<row_fill_args.next_compile_time_args_offset()>();
    constexpr auto block_start_row_args = TensorAccessorArgs<index_row_args.next_compile_time_args_offset()>();
    constexpr auto cos_args = TensorAccessorArgs<block_start_row_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr auto block_start_cos_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();
    constexpr auto block_start_sin_args = TensorAccessorArgs<block_start_cos_args.next_compile_time_args_offset()>();
    constexpr auto shift_args = TensorAccessorArgs<block_start_sin_args.next_compile_time_args_offset()>();

    const auto position_in = TensorAccessor(position_args, get_arg_val<uint32_t>(0));
    const auto offsets_in = TensorAccessor(offsets_args, get_arg_val<uint32_t>(1));
    const auto bf16_templates = TensorAccessor(bf16_templates_args, get_arg_val<uint32_t>(2));
    const auto u32_templates = TensorAccessor(u32_templates_args, get_arg_val<uint32_t>(3));
    const auto tile_templates = TensorAccessor(tile_templates_args, get_arg_val<uint32_t>(4));
    const auto cos_table = TensorAccessor(cos_table_args, get_arg_val<uint32_t>(5));
    const auto sin_table = TensorAccessor(sin_table_args, get_arg_val<uint32_t>(6));
    const auto kv_block_start_out = TensorAccessor(kv_block_start_args, get_arg_val<uint32_t>(7));
    const auto kv_row_hit_out = TensorAccessor(kv_row_hit_args, get_arg_val<uint32_t>(8));
    const auto indexer_neg_mask_out = TensorAccessor(indexer_neg_mask_args, get_arg_val<uint32_t>(9));
    const auto row_keep_bits_out = TensorAccessor(row_keep_bits_args, get_arg_val<uint32_t>(10));
    const auto row_fill_out = TensorAccessor(row_fill_args, get_arg_val<uint32_t>(11));
    const auto index_row_out = TensorAccessor(index_row_args, get_arg_val<uint32_t>(12));
    const auto block_start_row_out = TensorAccessor(block_start_row_args, get_arg_val<uint32_t>(13));
    const auto cos_out = TensorAccessor(cos_args, get_arg_val<uint32_t>(14));
    const auto sin_out = TensorAccessor(sin_args, get_arg_val<uint32_t>(15));
    const auto block_start_cos_out = TensorAccessor(block_start_cos_args, get_arg_val<uint32_t>(16));
    const auto block_start_sin_out = TensorAccessor(block_start_sin_args, get_arg_val<uint32_t>(17));
    const auto shift_in = TensorAccessor(shift_args, get_arg_val<uint32_t>(18));
    const uint32_t lane = get_arg_val<uint32_t>(19);
    const uint32_t lane_count = get_arg_val<uint32_t>(20);

    const Noc noc;
    DataflowBuffer stage(cb_stage);
    stage.reserve_back(1);
    const uint32_t base = stage.get_write_ptr();
    CoreLocalMem<volatile uint32_t> words(base);
    CoreLocalMem<volatile uint16_t> halves(base);
    constexpr uint32_t stage_position_row = stage_rows, stage_offsets_row = stage_rows + 128,
                       stage_index_row = stage_rows + 256, stage_block_start_row = stage_rows + 384,
                       stage_kv_block_start_row = stage_rows + 512, stage_shift_row = stage_rows + 640;

    {
        // the position row, the offsets row, the shift row and the whole template rows / tile in one batch of reads
        read_bytes(position_in, stage, 128, 0, 0, stage_position_row);
        read_bytes(offsets_in, stage, 128, 0, 0, stage_offsets_row);
        read_bytes(shift_in, stage, 128, 0, 0, stage_shift_row);
        read_bytes(bf16_templates, stage, bf16_row_bytes, 0, bf16_row_bytes, stage_bf16);  // the MASK half
        read_bytes(tile_templates, stage, bf16_tile_bytes, 0, 0, stage_tile);              // zero tile
        noc.async_read_barrier();
    }

    const uint32_t position = words[stage_position_row / 4 + lane];
    const uint32_t offset = words[stage_offsets_row / 4 + lane];
    const uint32_t kv_row = position & kv_row_mask;
    const uint32_t context = position + 1;
    const uint32_t complete_blocks = context >> 2;
    const uint32_t selected = complete_blocks < block_topk ? complete_blocks : block_topk;
    const uint32_t low = selected << 2;
    const uint32_t high = low + (context & ring_mask);
    const uint32_t tail_shift = (complete_blocks - selected) << 2;
    const uint32_t block_start = position & lane_block_mask;
    // the RoPE table rows of this lane: P_u - S_u and (P_u & ~3) - S_u
    const uint32_t shift = words[stage_shift_row / 4 + lane];
    const uint32_t rope_position = position - shift;
    const uint32_t rope_block_start = block_start - shift;

    {
        if (lane == 0) {  // the three 32-lane rows of the whole position row
            for (uint32_t u = 0; u < lanes; ++u) {
                const uint32_t position_u = words[stage_position_row / 4 + u];
                words[stage_index_row / 4 + u] = position_u;
                words[stage_block_start_row / 4 + u] = position_u & lane_block_mask;
                words[stage_kv_block_start_row / 4 + u] = position_u & kv_block_start_mask;
            }
            write_bytes(stage, index_row_out, 128, stage_index_row, 0, 0);
            write_bytes(stage, block_start_row_out, 128, stage_block_start_row, 0, 0);
            write_bytes(stage, kv_block_start_out, 128, stage_kv_block_start_row, 0, 0);
        }

        // indexer_neg_mask row: MASK everywhere, +0.0 for blocks below complete_blocks (matches 0 * MASK = +0.0)
        {
            const uint32_t zero_bytes = complete_blocks * 2;
            const uint32_t bulk = zero_bytes & ~(dram_read_grain - 1);
            read_bytes(bf16_templates, stage, bulk, 0, 0, stage_bf16);
            noc.async_read_barrier();
            for (uint32_t k = bulk / 2; k < complete_blocks; ++k) {
                halves[stage_bf16 / 2 + k] = 0;
            }
        }
        write_bytes(stage, indexer_neg_mask_out, bf16_row_bytes, stage_bf16, lane, 0);
        // kv_row_hit tile of this lane (lanes with a tile): the zero tile with lane (kv_row, 0) = 1.0
        if (lane < lane_count) {
            const uint32_t kv_lane = (kv_row >> 4) * 512 + (kv_row & 15) * 16;
            halves[stage_tile / 2 + kv_lane] = one_bf16;
            write_bytes(stage, kv_row_hit_out, bf16_tile_bytes, stage_tile, lane, 0);
        }

        // row_keep_bits row: zeros, all_ones below low
        read_bytes(u32_templates, stage, u32_row_bytes, 0, 0, stage_u32);
        noc.async_read_barrier();
        {
            const uint32_t bulk = (low * 4) & ~(dram_read_grain - 1);
            read_bytes(u32_templates, stage, bulk, 0, u32_row_bytes, stage_u32);
            noc.async_read_barrier();
            for (uint32_t k = bulk / 4; k < low; ++k) {
                words[stage_u32 / 4 + k] = all_ones;
            }
        }
        noc.async_write_barrier();  // the mask row and the hit tile are out before their staging is reused
        write_bytes(stage, row_keep_bits_out, u32_row_bytes, stage_u32, lane, 0);
        noc.async_write_barrier();

        // row_fill row: all_ones from high, slot + offset + tail_shift on [low, high), zeros below low
        read_bytes(u32_templates, stage, u32_row_bytes, 0, u32_row_bytes, stage_u32);
        noc.async_read_barrier();
        {
            const uint32_t bulk = (high * 4) & ~(dram_read_grain - 1);
            read_bytes(u32_templates, stage, bulk, 0, 0, stage_u32);
            noc.async_read_barrier();
            for (uint32_t k = bulk / 4; k < high; ++k) {
                words[stage_u32 / 4 + k] = 0;
            }
            for (uint32_t k = low; k < high; ++k) {
                words[stage_u32 / 4 + k] = k + offset + tail_shift;
            }
        }
        write_bytes(stage, row_fill_out, u32_row_bytes, stage_u32, lane, 0);

        // RoPE rows: table rows P_u - S_u and (P_u & ~3) - S_u into row `lane` of the two output tiles (faces
        // (lane >> 4) * 2 and + 1)
        read_bytes(cos_table, stage, rope_row_bytes, rope_position, 0, stage_rope);
        read_bytes(sin_table, stage, rope_row_bytes, rope_position, 0, stage_rope + rope_row_bytes);
        read_bytes(cos_table, stage, rope_row_bytes, rope_block_start, 0, stage_rope + 2 * rope_row_bytes);
        read_bytes(sin_table, stage, rope_row_bytes, rope_block_start, 0, stage_rope + 3 * rope_row_bytes);
        noc.async_read_barrier();
        const uint32_t rope_src[4] = {
            stage_rope, stage_rope + rope_row_bytes, stage_rope + 2 * rope_row_bytes, stage_rope + 3 * rope_row_bytes};
        for (uint32_t face = 0; face < rope_dim / 16;
             ++face) {  // face f: lanes 16f..16f+15 -> tile f/2, face row of half f%2
            const uint32_t page = face >> 1, offset_bytes = rope_face_offset(lane, face & 1);
            write_bytes(stage, cos_out, face_row_bytes, rope_src[0] + face * face_row_bytes, page, offset_bytes);
            write_bytes(stage, sin_out, face_row_bytes, rope_src[1] + face * face_row_bytes, page, offset_bytes);
            write_bytes(
                stage, block_start_cos_out, face_row_bytes, rope_src[2] + face * face_row_bytes, page, offset_bytes);
            write_bytes(
                stage, block_start_sin_out, face_row_bytes, rope_src[3] + face * face_row_bytes, page, offset_bytes);
        }
        noc.async_write_barrier();
    }
    stage.push_back(1);
}
