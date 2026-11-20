// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of one scan core = one (user, DRAM bank) of one chip.
//
// The index-key cache is the prefill stack's: BFP8 tiles, one ND shard = one 32-position block of 4 tiles
// (4352 bytes), shards round-robin over the DRAM banks, slot-major. Bank b therefore holds blocks b, b + 8, ...
// of a slot back to back, and this core reads its share with plain large reads from ONE bank:
//     address in bank = buffer address + slot offset + (block / banks) * 4352.
// The positions a bank holds increase along its stream, so the positions the query may see are a prefix of it.
//
// Per run: read the user's step page (context length, score centre, head multipliers: a device tensor, so a trace
// follows it), derive how much of this bank's stream is visible, tell the compute kernel (meta page), load the
// query, set up the head-sum selectors, build the tiles that mask and centre the scores, stream the key blocks.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "scan_common.hpp"
#ifdef HY4_SCAN_FRAMES
#include "kv_frames.hpp"  // fused/kv_store/kernels: the frame tables of a paged key cache
#endif

namespace {

using namespace hy4_scan;

constexpr uint32_t blocks_per_pass = get_compile_time_arg_val(0);  // what the compute kernel takes at once
constexpr uint32_t blocks_per_read = get_compile_time_arg_val(1);  // one DRAM read; a multiple of the above
constexpr uint32_t log2_chip_block = get_compile_time_arg_val(2);  // tokens a chip holds per turn (128 -> 7)
constexpr uint32_t log2_chips = get_compile_time_arg_val(3);       // chips the context is striped over (32 -> 5)
constexpr uint32_t stripe_rows = get_compile_time_arg_val(4);      // positions per user on this chip
constexpr uint32_t sel_tile_bytes = get_compile_time_arg_val(5);
// 1: the selector tiles live in an L1 tensor behind cb_sel (zero outside the 32 x 32 multipliers, which are
// rewritten here every run); 0: they are read from a DRAM tensor.
constexpr bool selectors_in_l1 = get_compile_time_arg_val(7) != 0;
// Key stream. 0: one read of `blocks_per_read` blocks at a time, each awaited with the read barrier. N > 0: packets
// of `blocks_per_read` blocks (one NOC packet each, so at most 3 blocks = 13,056 bytes), up to N in flight, each
// with its own NOC transaction id and awaited on that id alone, on the response virtual channel the core was given
// (the pattern of models/demos/gpt_oss/tt/fused_decode/kernels/stream_down_reader.cpp).
constexpr uint32_t stream_depth = get_compile_time_arg_val(8);
constexpr uint32_t k_capacity_blocks = get_compile_time_arg_val(9);  // key blocks cb_k holds
#ifdef HY4_SCAN_FRAMES
// HY4_SCAN_FRAMES (fused/kv_store/PATCH.md 2.3): block j of this core's (user, bank) stream is the bank's one shard
// of frame j of the user, at table[j] (the frame tables: an L1 tensor under cb_frames, 128 frame bases per user),
// and runtime argument 2 is the layer's slot step in a bank, the same for every user. A packet holds only frames
// that lie back to back in the bank. With identity tables on a slot-major cache: the same bytes, the same packets.
constexpr uint32_t cb_frames = 7;
static_assert(stream_depth > 0, "the paged key stream is the pipelined one");
#endif
// Measurement only: push the key pages without reading them from DRAM.
constexpr bool skip_key_reads = (get_compile_time_arg_val(6) & 1) != 0;
// Measurement only: return at once (the launch floor of the program).
constexpr bool return_at_once = (get_compile_time_arg_val(6) & 4) != 0;

// Page `page` of an interleaved DRAM tensor at `base`.
FORCE_INLINE uint64_t interleaved_page(uint32_t base, uint32_t page, uint32_t page_bytes) {
    return get_noc_addr_from_bank_id<true>(page % NUM_DRAM_BANKS, base + (page / NUM_DRAM_BANKS) * page_bytes);
}

FORCE_INLINE uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }

}  // namespace

void kernel_main() {
    if constexpr (return_at_once) {
        return;
    }
    const uint32_t key_addr = get_arg_val<uint32_t>(0);
    const uint32_t bank = get_arg_val<uint32_t>(1);
    const uint32_t slot_offset = get_arg_val<uint32_t>(2);  // bytes from the buffer address to this slot, in a bank
    const uint32_t chip = get_arg_val<uint32_t>(3);         // this chip's index in the block-cyclic layout
    const uint32_t user = get_arg_val<uint32_t>(4);
    const uint32_t step_addr = get_arg_val<uint32_t>(5);
    const uint32_t qt_addr = get_arg_val<uint32_t>(6);
    const uint32_t sel_addr = get_arg_val<uint32_t>(7);
    const uint32_t stream_vc = get_arg_val<uint32_t>(8);  // response virtual channel of the pipelined key stream
#ifdef HY4_SCAN_FRAMES
    volatile tt_l1_ptr uint32_t* frame_table =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_frames)) + user * kv_frames::TABLE_FRAMES;
#endif

    // The user's step page: context length, score centre, head multipliers (a device tensor, so a trace follows).
    cb_reserve_back(cb_meta, 1);
    volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta));
    noc_async_read(interleaved_page(step_addr, user, step_page_bytes), get_write_ptr(cb_meta), step_page_bytes);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t context = meta[step_context];  // positions [0, context) are visible
    const uint32_t centre = meta[step_centre] & 0xFFFF;
    // minus the centre; +-0 means no centring
    const uint32_t minus_centre = (centre & 0x7FFF) ? (centre ^ 0x8000) : 0;

    // Rows of this chip's stripe that hold a visible position: position p sits on chip (p >> 7) % chips at row
    // (p >> 12) * 128 + p % 128, so the visible rows are a prefix.
    const uint32_t log2_cycle = log2_chip_block + log2_chips;
    const uint32_t chip_block = 1u << log2_chip_block;
    const uint32_t in_cycle = context & ((1u << log2_cycle) - 1);
    const uint32_t chip_start = chip << log2_chip_block;
    const uint32_t in_block = in_cycle > chip_start ? min_u32(in_cycle - chip_start, chip_block) : 0;
    const uint32_t rows = min_u32(((context >> log2_cycle) << log2_chip_block) + in_block, stripe_rows);

    // This bank's share: 32-row block j of the stripe is block j / banks of bank j % banks.
    const uint32_t whole_blocks = rows >> 5;
    const uint32_t tail_rows = rows & 31;
    const uint32_t turn = whole_blocks % NUM_DRAM_BANKS;
    const uint32_t full_blocks = whole_blocks / NUM_DRAM_BANKS + (turn > bank ? 1 : 0);
    const uint32_t partial = (turn == bank) ? tail_rows : 0;
    const uint32_t num_blocks = full_blocks + (partial ? 1 : 0);
    const uint32_t num_valid = full_blocks * 32 + partial;

    meta[meta_blocks] = num_blocks;
    meta[meta_valid] = num_valid;
    meta[meta_centre] = minus_centre;
    cb_push_back(cb_meta, 1);

    // The query, the selectors when they come from DRAM, and the first key blocks: issued now, awaited after the
    // tiles below are built. The compute kernel takes `blocks_per_pass` blocks at a time, so the last pass is
    // padded with pages that are pushed and never read from DRAM; their scores are never used.
    const uint32_t padded_blocks = (num_blocks + blocks_per_pass - 1) / blocks_per_pass * blocks_per_pass;
    const uint64_t bank_base = get_noc_addr_from_bank_id<true>(bank, key_addr + slot_offset);
    // state of the pipelined stream (stream_depth > 0), in blocks and packets
    const uint32_t k_base = get_write_ptr(cb_k);
    const uint32_t k_end = k_base + k_capacity_blocks * key_block_bytes;
    uint32_t k_write = k_base;
    uint32_t issued = 0, landed = 0, issued_packets = 0, landed_packets = 0, state_bytes = 0;
    constexpr uint32_t depth = stream_depth ? stream_depth : 1;
    uint32_t packet_blocks[depth];
    // Issue the next packet if a transaction id and room in cb_k are free. Room is asked for without blocking
    // while packets are in flight: they have to be pushed before the compute kernel can free anything.
    auto issue_packet = [&]() -> bool {
        if (issued == padded_blocks || issued_packets - landed_packets == depth) {
            return false;
        }
#ifndef HY4_SCAN_FRAMES
        const uint32_t blocks = min_u32(blocks_per_read, padded_blocks - issued);
#else
        // a packet holds frames that lie back to back in this bank, and does not run over the end of cb_k
        uint32_t blocks = min_u32(blocks_per_read, padded_blocks - issued);
        blocks = min_u32(blocks, (k_end - k_write) / key_block_bytes);
        if (issued < num_blocks) {
            uint32_t run = 1;
            while (run < blocks && issued + run < num_blocks &&
                   frame_table[issued + run] == frame_table[issued] + run * key_block_bytes) {
                ++run;
            }
            blocks = run;
        }
#endif
        if (!cb_pages_reservable_at_back(cb_k, (issued - landed + blocks) * key_block_tiles)) {
            if (issued_packets != landed_packets) {
                return false;
            }
            cb_reserve_back(cb_k, blocks * key_block_tiles);
        }
        const uint32_t read = issued < num_blocks ? min_u32(blocks, num_blocks - issued) : 0;
        const uint32_t slot = issued_packets % depth;
        if (read && !skip_key_reads) {
            if (read * key_block_bytes != state_bytes) {
                state_bytes = read * key_block_bytes;
                noc_async_read_one_packet_set_state<true>(bank_base, state_bytes, stream_vc);
            }
            noc_async_read_set_trid(slot + 1);
            noc_async_read_one_packet_with_state_with_trid(
#ifndef HY4_SCAN_FRAMES
                static_cast<uint32_t>(bank_base), issued * key_block_bytes, k_write, slot + 1);
#else
                static_cast<uint32_t>(bank_base), frame_table[issued], k_write, slot + 1);
#endif
        }
        packet_blocks[slot] = blocks;
        k_write += blocks * key_block_bytes;
        if (k_write >= k_end) {
            k_write = k_base;
        }
        issued += blocks;
        issued_packets += 1;
        return true;
    };
    auto land_packet = [&]() {
        const uint32_t slot = landed_packets % depth;
        noc_async_read_barrier_with_trid(slot + 1);
        cb_push_back(cb_k, packet_blocks[slot] * key_block_tiles);
        landed += packet_blocks[slot];
        landed_packets += 1;
    };

    constexpr uint32_t side_trid = 0xD;  // the query's reads: awaited on their own while key packets are in flight
    uint32_t bank_address = key_addr + slot_offset;
    uint32_t left = num_blocks;
    uint32_t padded_left = padded_blocks;
    uint32_t first_pushed = 0;
    if (num_blocks) {
        if constexpr (stream_depth > 0) {
            reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
            noc_async_read_set_trid(side_trid);
        }
        cb_reserve_back(cb_qt, key_block_tiles);
        const uint32_t qt_ptr = get_write_ptr(cb_qt);
        for (uint32_t t = 0; t < key_block_tiles; ++t) {
            noc_async_read(
                interleaved_page(qt_addr, user * key_block_tiles + t, bf16_tile_bytes),
                qt_ptr + t * bf16_tile_bytes,
                bf16_tile_bytes);
        }
        cb_reserve_back(cb_sel, 32);
        if constexpr (!selectors_in_l1) {
            const uint32_t sel_ptr = get_write_ptr(cb_sel);
            for (uint32_t t = 0; t < 32; ++t) {
                noc_async_read(
                    interleaved_page(sel_addr, user * 32 + t, sel_tile_bytes),
                    sel_ptr + t * sel_tile_bytes,
                    sel_tile_bytes);
            }
        }
        if constexpr (stream_depth > 0) {
            // the stateful packet reads own the read command buffer from here on: no other read is issued
            while (issue_packet()) {
            }
        } else {
            first_pushed = min_u32(blocks_per_read, padded_left);
            const uint32_t read = min_u32(first_pushed, left);
            cb_reserve_back(cb_k, first_pushed * key_block_tiles);
            if constexpr (!skip_key_reads) {
                noc_async_read(
                    get_noc_addr_from_bank_id<true>(bank, bank_address), get_write_ptr(cb_k), read * key_block_bytes);
            }
            bank_address += read * key_block_bytes;
            left -= read;
            padded_left -= first_pushed;
        }

        if constexpr (stream_depth > 0) {
            // The query first: the compute kernel starts on the key blocks as they land.
            noc_async_read_barrier_with_trid(side_trid);
            cb_push_back(cb_qt, key_block_tiles);
        }
        if constexpr (selectors_in_l1) {
            // Selector tile c holds the 32 multipliers in column c (row = head) and zeros elsewhere; the zeros
            // are the tensor's own and are never written. Element (h, c) is word (h / 16) * 512 + (c / 16) * 256
            // + (h % 16) * 16 + c % 16 of its tile. Written 8 tiles at a time, the key stream served in between.
            volatile tt_l1_ptr uint16_t* weights = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(meta + step_weights);
            uint16_t multiplier[32];
            for (uint32_t h = 0; h < 32; ++h) {
                multiplier[h] = weights[h];
            }
            uint16_t* sel = reinterpret_cast<uint16_t*>(get_write_ptr(cb_sel));
            for (uint32_t c = 0; c < 32; ++c) {
                uint16_t* column = sel + c * (bf16_tile_bytes / 2) + ((c >> 4) << 8) + (c & 15);
                for (uint32_t h = 0; h < 16; ++h) {
                    column[h << 4] = multiplier[h];
                    column[512 + (h << 4)] = multiplier[16 + h];
                }
                if constexpr (stream_depth > 0) {
                    if ((c & 7) == 7 && issued_packets != landed_packets) {
                        land_packet();
                        while (issue_packet()) {
                        }
                    }
                }
            }
        }
        if constexpr (stream_depth > 0) {
            cb_push_back(cb_sel, 32);
        }
    }

    // What is added to a score tile (row = position in block, col = block) after the head sum, as one more
    // matmul  scores += U @ M  of two tiles built here:
    //   * minus the user's centre on every visible score:        U[r][0] = 1,  M[0][c] = -centre
    //   * NEG on what the query may not see, in the tile the context ends in: its columns past the last visible
    //     block, M[0][c] = NEG, and the rows of a part-visible block past its last visible position,
    //     U[r][1] = (r >= partial), M[1][partial block] = NEG.
    // A bank with nothing visible is one score tile that is all NEG. cb_mask is the pair of the tile the context
    // ends in, cb_centre the pair of the whole tiles before it (only when there is a centre).
    const uint32_t in_tile = num_valid & 1023;
    const bool has_masked = in_tile != 0 || num_valid == 0;
    const bool has_centred = minus_centre != 0 && (num_valid >> 10) != 0;
    if (has_masked) {
        cb_reserve_back(cb_mask, 2);
        uint32_t* words = reinterpret_cast<uint32_t*>(get_write_ptr(cb_mask));
        for (uint32_t i = 0; i < 2 * bf16_tile_bytes / 4; ++i) {
            words[i] = 0;
        }
        uint16_t* u_tile = reinterpret_cast<uint16_t*>(get_write_ptr(cb_mask));
        uint16_t* m_tile = u_tile + bf16_tile_bytes / 2;
        const uint32_t whole_cols = in_tile >> 5;
        const uint32_t part_rows = in_tile & 31;
        const uint32_t seen_cols = whole_cols + (part_rows ? 1 : 0);
        for (uint32_t r = 0; r < 32; ++r) {
            u_tile[tile_word(r, 0)] = bf16_one;
            if (part_rows && r >= part_rows) {
                u_tile[tile_word(r, 1)] = bf16_one;
            }
        }
        for (uint32_t c = 0; c < 32; ++c) {
            m_tile[tile_word(0, c)] = c < seen_cols ? minus_centre : bf16_neg;
        }
        if (part_rows) {
            m_tile[tile_word(1, whole_cols)] = bf16_neg;
        }
    }
    if (has_centred) {
        cb_reserve_back(cb_centre, 2);
        uint32_t* words = reinterpret_cast<uint32_t*>(get_write_ptr(cb_centre));
        for (uint32_t i = 0; i < 2 * bf16_tile_bytes / 4; ++i) {
            words[i] = 0;
        }
        uint16_t* u_tile = reinterpret_cast<uint16_t*>(get_write_ptr(cb_centre));
        uint16_t* m_tile = u_tile + bf16_tile_bytes / 2;
        for (uint32_t i = 0; i < 32; ++i) {
            u_tile[tile_word(i, 0)] = bf16_one;
            m_tile[tile_word(0, i)] = minus_centre;
        }
    }

    if constexpr (stream_depth == 0) {
        if (num_blocks) {
            noc_async_read_barrier();
            cb_push_back(cb_qt, key_block_tiles);
            cb_push_back(cb_sel, 32);
            cb_push_back(cb_k, first_pushed * key_block_tiles);
        }
    }
    if (has_masked) {
        cb_push_back(cb_mask, 2);
    }
    if (has_centred) {
        cb_push_back(cb_centre, 2);
    }

    // The rest of the key blocks.
    if constexpr (stream_depth > 0) {
        while (landed < padded_blocks) {
            if (!issue_packet()) {
                land_packet();
            }
        }
        noc_async_read_set_trid(0);
    } else {
        while (padded_left) {
            const uint32_t pushed = min_u32(blocks_per_read, padded_left);
            const uint32_t read = min_u32(pushed, left);
            cb_reserve_back(cb_k, pushed * key_block_tiles);
            if (read && !skip_key_reads) {
                noc_async_read(
                    get_noc_addr_from_bank_id<true>(bank, bank_address), get_write_ptr(cb_k), read * key_block_bytes);
                noc_async_read_barrier();
            }
            cb_push_back(cb_k, pushed * key_block_tiles);
            bank_address += read * key_block_bytes;
            left -= read;
            padded_left -= pushed;
        }
    }
}
