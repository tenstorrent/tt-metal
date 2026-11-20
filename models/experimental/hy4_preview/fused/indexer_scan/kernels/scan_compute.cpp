// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of one scan core = one (user, DRAM bank) of one chip.
//
// Scoring, per 32-position block of keys K [32 x 128] (BFP8, 4 tiles) against the query Qt [128 x 32 heads]:
//     P      = relu(K @ Qt)                [32 positions x 32 heads], FP32 in DEST, relu in the packer, BF16 in L1
//     scores += P @ Sel[block]             Sel[block] holds the heads' signs / weights in column `block` only
// so 32 blocks accumulate into ONE score tile in DEST (row = position in block, col = block) and are packed once.
// The reader's addend goes in with one more matmul (U @ M): minus the user's score centre, and NEG on what the
// query may not see in the tile the context ends in. Scores leave as BF16(score - centre): BF16 resolves a value
// relative to its own size, so a centre near the cut keeps the order around the cut far finer than BF16(score).
//
// Top-2048 with positions, the fused form of the topk_xl LLK ([bf16 score | u16 stamp] in one FP32 word): each
// 2048 scores (2 score tiles) are copied to DEST, stamped, sorted and merged into the running survivor. Then
// the binary tree over the user's banks: a core merges the survivors that land in cb_landing (level by level)
// and either sends its own to its partner (cb_send) or, as the root, hands it to the writer (cb_out).
// Derived from ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/kernels/compute.cpp
// (reduce_fused_row, merge_landed_fused_survivor, settle_next_program).

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/topk_xl.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "scan_common.hpp"

namespace {

using namespace hy4_scan;

constexpr uint32_t blocks_per_pass = get_compile_time_arg_val(0);  // key blocks per DEST pass, at most 8
constexpr bool scores_only = get_compile_time_arg_val(1) != 0;     // stop after scoring (the writer stores the tiles)
constexpr uint32_t score_tiles_max = get_compile_time_arg_val(2);  // score tiles of a whole bank stream
// Measurement only: consume the keys and push score tiles without computing them.
constexpr bool skip_scoring = (get_compile_time_arg_val(3) & 2) != 0;
// Measurement only: return at once (the launch floor of the program).
constexpr bool return_at_once = (get_compile_time_arg_val(3) & 4) != 0;
constexpr uint32_t K = topk;
constexpr uint32_t blocks_per_tile = 32;
constexpr uint32_t survivor_slot = 0;
constexpr uint32_t incoming_slot = survivor_tiles;

static_assert(blocks_per_pass >= 1 && blocks_per_pass <= 8 && blocks_per_tile % blocks_per_pass == 0);

// srcB <- keys, srcA <- query; relu(K @ Qt) leaves through the packer.
FORCE_INLINE void enter_qk() {
    reconfig_data_format(cb_sel, cb_qt, cb_p, cb_k);
#ifdef HY4_SCAN_QT_ROWS
    // the query's tiles hold [32 heads x 32 dims] (a head a row, as a landing of per-head pieces has it): the
    // unpacker transposes them
    matmul_block_init(cb_k, cb_qt, 1, 1, blocks_per_pass, key_block_tiles);
#else
    matmul_block_init(cb_k, cb_qt, 0, 1, blocks_per_pass, key_block_tiles);
#endif
    pack_relu_config(ReluConfig::zero());
}

// srcB <- P (or the mask's U), srcA <- selector (or the mask's M): all BF16, one tile each.
FORCE_INLINE void enter_head_sum() {
    reconfig_data_format(cb_qt, cb_sel, cb_k, cb_p);
    matmul_block_init(cb_p, cb_sel, 0, 1, 1, 1);
    pack_relu_config(ReluConfig::none());
}

// One score tile: `blocks` key blocks (0 for a bank with nothing visible), plus the reader's addend: the mask
// pair for the tile the context ends in (`masked`), else the centre pair when the user has a centre (`centred`).
FORCE_INLINE void score_tile(uint32_t blocks, bool masked, bool centred) {
    const uint32_t padded = (blocks + blocks_per_pass - 1) / blocks_per_pass * blocks_per_pass;
    if constexpr (skip_scoring) {
        for (uint32_t first = 0; first < padded; first += blocks_per_pass) {
            cb_wait_front(cb_k, blocks_per_pass * key_block_tiles);
            cb_pop_front(cb_k, blocks_per_pass * key_block_tiles);
        }
        if (masked) {
            cb_wait_front(cb_mask, 2);
        }
        cb_reserve_back(cb_scores, 1);
        cb_push_back(cb_scores, 1);
        return;
    }
    if (blocks) {
        enter_qk();
        cb_reserve_back(cb_p, padded);
        for (uint32_t first = 0; first < padded; first += blocks_per_pass) {
            cb_wait_front(cb_k, blocks_per_pass * key_block_tiles);
            tile_regs_acquire();
            // rt_dim > ct_dim: the query tile stays in srcA while the pass's key tiles go through srcB, and
            // block i accumulates in DEST tile i.
            for (uint32_t kt = 0; kt < key_block_tiles; ++kt) {
#ifdef HY4_SCAN_QT_ROWS
                matmul_block(cb_k, cb_qt, kt, kt, 0, true, 1, blocks_per_pass, key_block_tiles);
#else
                matmul_block(cb_k, cb_qt, kt, kt, 0, false, 1, blocks_per_pass, key_block_tiles);
#endif
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < blocks_per_pass; ++i) {
                pack_tile<true>(i, cb_p, first + i);
            }
            tile_regs_release();
            cb_pop_front(cb_k, blocks_per_pass * key_block_tiles);
        }
        cb_push_back(cb_p, padded);
        cb_wait_front(cb_p, padded);
    }

    enter_head_sum();
    tile_regs_acquire();
    for (uint32_t block = 0; block < blocks; ++block) {
        matmul_block(cb_p, cb_sel, block, block, 0, false, 1, 1, 1);
    }
    if (masked) {
        cb_wait_front(cb_mask, 2);
        matmul_block(cb_mask, cb_mask, 0, 1, 0, false, 1, 1, 1);
    } else if (centred) {
        cb_wait_front(cb_centre, 2);
        matmul_block(cb_centre, cb_centre, 0, 1, 0, false, 1, 1, 1);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_scores, 1);
    pack_tile(0, cb_scores);
    cb_push_back(cb_scores, 1);
    tile_regs_release();
    if (blocks) {
        cb_pop_front(cb_p, padded);
    }
}

// Copy one chunk (2 score tiles, `active` of its 2048 lanes) into DEST at `dst`, stamp it and sort it.
template <bool FullInit>
FORCE_INLINE void sort_fused_chunk(uint32_t dst, uint32_t active, bool ascending, uint32_t chunk_id) {
    cb_wait_front(cb_scores, survivor_tiles);
    if constexpr (FullInit) {
        topk_xl_copy_tile_init(cb_scores);
    } else {
        topk_xl_copy_tile_init_short(cb_scores);
    }
    topk_xl_copy_tile<K>(cb_scores, dst, 0, active);
    cb_pop_front(cb_scores, survivor_tiles);

    topk_xl_add_lsb_indices_init();
    topk_xl_add_lsb_indices_rt<K>(dst, chunk_id);
    if constexpr (FullInit) {
        topk_xl_init<K, true>();
    } else {
        topk_large_indices_reinit_fused_after_stamp();
    }
    topk_xl_local_sort<K>(dst, ascending);
}

// A partner's survivor (ascending) lands as raw 32-bit words and merges into ours (descending).
FORCE_INLINE void merge_landed(bool final_ascending) {
    cb_wait_front(cb_landing, survivor_tiles);
    reconfig_data_format_srca(cb_landing);
    copy_init(cb_landing);
    for (uint32_t tile = 0; tile < survivor_tiles; ++tile) {
        copy_tile(cb_landing, tile, incoming_slot + tile);
    }
    cb_pop_front(cb_landing, survivor_tiles);
    // The integer landing format leaves the FPU in int math; the rebuild transposes need the float path.
    reconfig_data_format_srca(cb_scores);

    topk_xl_init<K, true>();
    topk_xl_merge<K, true>(survivor_slot);
    topk_xl_rebuild<K, true>(survivor_slot, final_ascending);
}

// A raw pack and unpack round trip can leave unpacker state a later program's init does not reset; one discarded
// copy resets it (the value is thrown away, the address is inside the CB's own L1).
FORCE_INLINE void settle_next_program() {
    reconfig_data_format_srca(cb_scores);
    copy_init(cb_scores);
    tile_regs_acquire();
    copy_tile(cb_scores, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    tile_regs_release();
}

}  // namespace

void kernel_main() {
    if constexpr (return_at_once) {
        return;
    }
    const uint32_t first_chunk = get_arg_val<uint32_t>(0);   // this bank's first chunk id in the user's stamps
    const uint32_t num_receives = get_arg_val<uint32_t>(1);  // tree levels at which a partner's survivor lands here
    const bool sends = get_arg_val<uint32_t>(2) != 0;        // not a root: the survivor leaves for a partner

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_k, cb_qt, cb_p);

    cb_wait_front(cb_meta, 1);
    const uint32_t num_blocks = read_tile_value(cb_meta, 0, meta_blocks);
    const uint32_t num_valid = read_tile_value(cb_meta, 0, meta_valid);
    const bool centred = read_tile_value(cb_meta, 0, meta_centre) != 0;
    cb_pop_front(cb_meta, 1);

    if (num_blocks) {
        cb_wait_front(cb_qt, key_block_tiles);
        cb_wait_front(cb_sel, 32);
    }

    // Score tiles: the whole ones, then the one the context ends in (masked). A bank with nothing visible has
    // one all-NEG tile, so every core has at least one chunk to sort.
    const uint32_t whole_tiles = num_valid >> 10;
    const bool has_masked = (num_valid & 1023) != 0 || num_valid == 0;
    const uint32_t used_tiles = whole_tiles + (has_masked ? 1 : 0);
    uint32_t blocks_left = num_blocks;
    for (uint32_t tile = 0; tile < used_tiles; ++tile) {
        const uint32_t blocks = blocks_left < blocks_per_tile ? blocks_left : blocks_per_tile;
        score_tile(blocks, tile == whole_tiles, centred);
        blocks_left -= blocks;
    }
    if (num_blocks) {
        cb_pop_front(cb_qt, key_block_tiles);
        cb_pop_front(cb_sel, 32);
    }
    if (has_masked) {
        cb_pop_front(cb_mask, 2);
    }
    if (centred && whole_tiles) {
        cb_pop_front(cb_centre, 2);
    }

    if constexpr (scores_only) {
        // The writer stores a fixed number of tiles; the ones past the context are never looked at.
        if (used_tiles < score_tiles_max) {
            cb_reserve_back(cb_scores, score_tiles_max - used_tiles);
            cb_push_back(cb_scores, score_tiles_max - used_tiles);
        }
        return;
    }

    // Chunks of 2 score tiles; the last one may hold one tile (its other 1024 lanes are filled with -inf).
    const uint32_t num_chunks = (used_tiles + 1) / 2;
    const uint32_t tail_elements = (used_tiles - 2 * (num_chunks - 1)) * 1024;
    if (used_tiles & 1) {
        cb_reserve_back(cb_scores, 1);
        cb_push_back(cb_scores, 1);
    }

    tile_regs_acquire();

    const bool leaf_sends = sends && num_receives == 0;
    sort_fused_chunk<true>(survivor_slot, num_chunks == 1 ? tail_elements : K, false, first_chunk);
    if (num_chunks == 1) {
        topk_xl_rebuild<K, true>(survivor_slot, leaf_sends);
    } else {
        for (uint32_t chunk = 1; chunk < num_chunks; ++chunk) {
            const bool last = chunk + 1 == num_chunks;
            sort_fused_chunk<false>(incoming_slot, last ? tail_elements : K, true, first_chunk + chunk);
            topk_xl_merge<K, true>(survivor_slot);
            topk_xl_rebuild<K, true>(survivor_slot, leaf_sends && last);
        }
    }

    for (uint32_t level = 0; level < num_receives; ++level) {
        merge_landed(sends && level + 1 == num_receives);
    }

    tile_regs_commit();
    tile_regs_wait();
    // Raw 32-bit tiles: the fused words must not pass through the float packer.
    if (sends) {
        cb_reserve_back(cb_send, survivor_tiles);
        pack_reconfig_data_format(cb_send);
        for (uint32_t tile = 0; tile < survivor_tiles; ++tile) {
            pack_tile(survivor_slot + tile, cb_send);
        }
        cb_push_back(cb_send, survivor_tiles);
    } else {
        cb_reserve_back(cb_out, survivor_tiles);
        pack_reconfig_data_format(cb_out);
        for (uint32_t tile = 0; tile < survivor_tiles; ++tile) {
            pack_tile(survivor_slot + tile, cb_out);
        }
        cb_push_back(cb_out, survivor_tiles);
    }
    tile_regs_release();

    settle_next_program();
}
