// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The decode step's position-derived tensors from the uint32 position P, assembled in L1 from constant template rows
// (zeros | MASK bf16 lanes, zeros | all_ones uint32 lanes, a zero tile and a ones-column tile) by NoC reads, a few
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
constexpr uint32_t face_row_bytes = 32;
constexpr uint32_t bf16_row_bytes = blocks * 2;
constexpr uint32_t u32_row_bytes = slots * 4;
constexpr uint32_t rope_row_bytes = rope_dim * 2;
// staging layout (bytes from the reserved block)
constexpr uint32_t stage_bf16 = 0;
constexpr uint32_t stage_u32 = stage_bf16 + bf16_row_bytes;
constexpr uint32_t stage_tiles = stage_u32 + ((u32_row_bytes + 63) & ~63u);
constexpr uint32_t stage_rope = stage_tiles + 2 * bf16_tile_bytes;
constexpr uint32_t stage_index = stage_rope + 4 * rope_row_bytes;
constexpr uint32_t stage_scalars = stage_index + 2 * 128;
constexpr uint32_t stage_shift = stage_scalars + 64;  // the rotary shift's 64-byte read grain
constexpr uint32_t stage_bytes = stage_shift + 64;

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

void kernel_main() {
    constexpr auto position_args = TensorAccessorArgs<0>();
    constexpr auto bf16_templates_args = TensorAccessorArgs<position_args.next_compile_time_args_offset()>();
    constexpr auto u32_templates_args = TensorAccessorArgs<bf16_templates_args.next_compile_time_args_offset()>();
    constexpr auto tile_templates_args = TensorAccessorArgs<u32_templates_args.next_compile_time_args_offset()>();
    constexpr auto cos_table_args = TensorAccessorArgs<tile_templates_args.next_compile_time_args_offset()>();
    constexpr auto sin_table_args = TensorAccessorArgs<cos_table_args.next_compile_time_args_offset()>();
    constexpr auto kv_block_start_args = TensorAccessorArgs<sin_table_args.next_compile_time_args_offset()>();
    constexpr auto kv_row_hit_args = TensorAccessorArgs<kv_block_start_args.next_compile_time_args_offset()>();
    constexpr auto kv_row_keep_args = TensorAccessorArgs<kv_row_hit_args.next_compile_time_args_offset()>();
    constexpr auto ring_hit_args = TensorAccessorArgs<kv_row_keep_args.next_compile_time_args_offset()>();
    constexpr auto ring_keep_args = TensorAccessorArgs<ring_hit_args.next_compile_time_args_offset()>();
    constexpr auto block_index_args = TensorAccessorArgs<ring_keep_args.next_compile_time_args_offset()>();
    constexpr auto indexer_neg_mask_args = TensorAccessorArgs<block_index_args.next_compile_time_args_offset()>();
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
    const auto bf16_templates = TensorAccessor(bf16_templates_args, get_arg_val<uint32_t>(1));
    const auto u32_templates = TensorAccessor(u32_templates_args, get_arg_val<uint32_t>(2));
    const auto tile_templates = TensorAccessor(tile_templates_args, get_arg_val<uint32_t>(3));
    const auto cos_table = TensorAccessor(cos_table_args, get_arg_val<uint32_t>(4));
    const auto sin_table = TensorAccessor(sin_table_args, get_arg_val<uint32_t>(5));
    const auto kv_block_start_out = TensorAccessor(kv_block_start_args, get_arg_val<uint32_t>(6));
    const auto kv_row_hit_out = TensorAccessor(kv_row_hit_args, get_arg_val<uint32_t>(7));
    const auto kv_row_keep_out = TensorAccessor(kv_row_keep_args, get_arg_val<uint32_t>(8));
    const auto ring_hit_out = TensorAccessor(ring_hit_args, get_arg_val<uint32_t>(9));
    const auto ring_keep_out = TensorAccessor(ring_keep_args, get_arg_val<uint32_t>(10));
    const auto block_index_out = TensorAccessor(block_index_args, get_arg_val<uint32_t>(11));
    const auto indexer_neg_mask_out = TensorAccessor(indexer_neg_mask_args, get_arg_val<uint32_t>(12));
    const auto row_keep_bits_out = TensorAccessor(row_keep_bits_args, get_arg_val<uint32_t>(13));
    const auto row_fill_out = TensorAccessor(row_fill_args, get_arg_val<uint32_t>(14));
    const auto index_row_out = TensorAccessor(index_row_args, get_arg_val<uint32_t>(15));
    const auto block_start_row_out = TensorAccessor(block_start_row_args, get_arg_val<uint32_t>(16));
    const auto cos_out = TensorAccessor(cos_args, get_arg_val<uint32_t>(17));
    const auto sin_out = TensorAccessor(sin_args, get_arg_val<uint32_t>(18));
    const auto block_start_cos_out = TensorAccessor(block_start_cos_args, get_arg_val<uint32_t>(19));
    const auto block_start_sin_out = TensorAccessor(block_start_sin_args, get_arg_val<uint32_t>(20));
    const auto shift_in = TensorAccessor(shift_args, get_arg_val<uint32_t>(21));

    const Noc noc;
    DataflowBuffer stage(cb_stage);
    stage.reserve_back(1);
    const uint32_t base = stage.get_write_ptr();
    CoreLocalMem<volatile uint32_t> words(base);
    CoreLocalMem<volatile uint16_t> halves(base);

    {
        // P, the rotary shift S and the whole template rows / tiles in one batch of reads
        read_bytes(position_in, stage, dram_read_grain, 0, 0, stage_scalars);
        read_bytes(shift_in, stage, dram_read_grain, 0, 0, stage_shift);
        read_bytes(bf16_templates, stage, bf16_row_bytes, 0, bf16_row_bytes, stage_bf16);         // the MASK half
        read_bytes(tile_templates, stage, bf16_tile_bytes, 0, 0, stage_tiles);                    // zero tile
        read_bytes(tile_templates, stage, bf16_tile_bytes, 1, 0, stage_tiles + bf16_tile_bytes);  // ones-column tile
        noc.async_read_barrier();
    }

    const uint32_t position = words[stage_scalars / 4];
    const uint32_t kv_row = position & kv_row_mask;
    const uint32_t ring_row = position & ring_mask;
    const uint32_t context = position + 1;
    const uint32_t complete_blocks = context >> 2;
    const uint32_t selected = complete_blocks < block_topk ? complete_blocks : block_topk;
    const uint32_t low = selected << 2;
    const uint32_t high = low + (context & ring_mask);
    const uint32_t tail_shift = (complete_blocks - selected) << 2;
    const uint32_t block_start = position & lane_block_mask;
    // the RoPE table rows: the token's rotary position P - S and its index block's first token's (P & ~3) - S
    const uint32_t shift = words[stage_shift / 4];
    const uint32_t rope_position = position - shift;
    const uint32_t rope_block_start = block_start - shift;

    {
        // scalars and index rows
        words[stage_scalars / 4 + 4] = position & kv_block_start_mask;
        words[stage_scalars / 4 + 8] = position >> 2;
        for (uint32_t lane = 0; lane < 32; ++lane) {
            words[stage_index / 4 + lane] = position;
            words[stage_index / 4 + 32 + lane] = block_start;
        }
        write_bytes(stage, kv_block_start_out, 4, stage_scalars + 16, 0, 0);
        write_bytes(stage, block_index_out, 4, stage_scalars + 32, 0, 0);
        write_bytes(stage, index_row_out, 128, stage_index, 0, 0);
        write_bytes(stage, block_start_row_out, 128, stage_index + 128, 0, 0);

        // indexer_neg_mask: MASK everywhere, +0.0 for blocks below complete_blocks (matches 0 * MASK = +0.0)
        {
            const uint32_t zero_bytes = complete_blocks * 2;
            const uint32_t bulk = zero_bytes & ~(dram_read_grain - 1);
            read_bytes(bf16_templates, stage, bulk, 0, 0, stage_bf16);
            for (uint32_t lane = bulk / 2; lane < complete_blocks; ++lane) {
                halves[stage_bf16 / 2 + lane] = 0;
            }
        }
        // the four one-hots: tile lane (row, 0) is 16-bit word (row >> 4) * 512 + (row & 15) * 16
        const uint32_t kv_lane = (kv_row >> 4) * 512 + (kv_row & 15) * 16;
        const uint32_t ring_lane = (ring_row >> 4) * 512 + (ring_row & 15) * 16;
        CoreLocalMem<volatile uint16_t> zero_tile = halves + stage_tiles / 2;
        CoreLocalMem<volatile uint16_t> ones_tile = halves + (stage_tiles + bf16_tile_bytes) / 2;
        noc.async_read_barrier();  // the zero-lane bulk read landed before the mask row is written
        write_bytes(stage, indexer_neg_mask_out, bf16_row_bytes, stage_bf16, 0, 0);
        zero_tile[kv_lane] = one_bf16;
        write_bytes(stage, kv_row_hit_out, bf16_tile_bytes, stage_tiles, 0, 0);
        ones_tile[kv_lane] = 0;
        write_bytes(stage, kv_row_keep_out, bf16_tile_bytes, stage_tiles + bf16_tile_bytes, 0, 0);
        noc.async_write_barrier();
        zero_tile[kv_lane] = 0;
        ones_tile[kv_lane] = one_bf16;
        zero_tile[ring_lane] = one_bf16;
        write_bytes(stage, ring_hit_out, bf16_tile_bytes, stage_tiles, 0, 0);
        ones_tile[ring_lane] = 0;
        write_bytes(stage, ring_keep_out, bf16_tile_bytes, stage_tiles + bf16_tile_bytes, 0, 0);

        // row_keep_bits: zeros, all_ones below low
        read_bytes(u32_templates, stage, u32_row_bytes, 0, 0, stage_u32);
        noc.async_read_barrier();
        {
            const uint32_t bulk = (low * 4) & ~(dram_read_grain - 1);
            read_bytes(u32_templates, stage, bulk, 0, u32_row_bytes, stage_u32);
            noc.async_read_barrier();
            for (uint32_t lane = bulk / 4; lane < low; ++lane) {
                words[stage_u32 / 4 + lane] = all_ones;
            }
        }
        noc.async_write_barrier();  // the ring tiles are out before the staging tiles change again (none do) and before
                                    // the u32 row is written
        write_bytes(stage, row_keep_bits_out, u32_row_bytes, stage_u32, 0, 0);
        noc.async_write_barrier();

        // row_fill: all_ones from high, slot + tail_shift on [low, high), zeros below low
        read_bytes(u32_templates, stage, u32_row_bytes, 0, u32_row_bytes, stage_u32);
        noc.async_read_barrier();
        {
            const uint32_t bulk = (high * 4) & ~(dram_read_grain - 1);
            read_bytes(u32_templates, stage, bulk, 0, 0, stage_u32);
            noc.async_read_barrier();
            for (uint32_t lane = bulk / 4; lane < high; ++lane) {
                words[stage_u32 / 4 + lane] = 0;
            }
            for (uint32_t lane = low; lane < high; ++lane) {
                words[stage_u32 / 4 + lane] = lane + tail_shift;
            }
        }
        write_bytes(stage, row_fill_out, u32_row_bytes, stage_u32, 0, 0);

        // RoPE rows: table rows P - S and (P & ~3) - S into row 0 of the two output tiles (faces 0 and 1 of each tile)
        read_bytes(cos_table, stage, rope_row_bytes, rope_position, 0, stage_rope);
        read_bytes(sin_table, stage, rope_row_bytes, rope_position, 0, stage_rope + rope_row_bytes);
        read_bytes(cos_table, stage, rope_row_bytes, rope_block_start, 0, stage_rope + 2 * rope_row_bytes);
        read_bytes(sin_table, stage, rope_row_bytes, rope_block_start, 0, stage_rope + 3 * rope_row_bytes);
        noc.async_read_barrier();
        const uint32_t rope_src[4] = {
            stage_rope, stage_rope + rope_row_bytes, stage_rope + 2 * rope_row_bytes, stage_rope + 3 * rope_row_bytes};
        for (uint32_t face = 0; face < rope_dim / 16;
             ++face) {  // face f: lanes 16f..16f+15 -> tile f/2, face f%2 row 0
            const uint32_t page = face >> 1, offset = (face & 1) * 512;
            write_bytes(stage, cos_out, face_row_bytes, rope_src[0] + face * face_row_bytes, page, offset);
            write_bytes(stage, sin_out, face_row_bytes, rope_src[1] + face * face_row_bytes, page, offset);
            write_bytes(stage, block_start_cos_out, face_row_bytes, rope_src[2] + face * face_row_bytes, page, offset);
            write_bytes(stage, block_start_sin_out, face_row_bytes, rope_src[3] + face * face_row_bytes, page, offset);
        }
        noc.async_write_barrier();
    }
    stage.push_back(1);
}
