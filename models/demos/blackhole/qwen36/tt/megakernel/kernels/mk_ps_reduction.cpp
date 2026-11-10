// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Megakernel AR step 3 (oneshot_ar.py OneShotARP): persistent copy of all_reduce_async's compute/reduction.cpp --
// ITERS reductions (+ residual) in one program; the input slots and the residual are popped per iteration.
#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reduce.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out0 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in1 = cb_in0;
    constexpr bool fuse_residual = get_compile_time_arg_val(2) != 0;
    constexpr uint32_t cb_residual = get_compile_time_arg_val(3);
    constexpr uint32_t ITERS = get_compile_time_arg_val(4);
    // STATS: per iteration, every output core (owner or all-gather receiver) also takes the row statistic of its chunk
    // (sum over its PT tiles of x^2 x 2^-12, REDUCE_ROW -> CB 25 column 0) from the output CB, before the receiver frees it
    constexpr bool STATS = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t CB_SQ = 24, CB_PART = 25, CB_SCALER = 30;
    // SRED: reduce on arrival. rt args 3.. = the slot order (own slot first); acc = slot o0 + slot o1, then acc += slot
    // o_k as each lands (fp32 accumulators CB 26 / 27, ping-pong), the last step adding the residual into the output
    constexpr bool SRED = get_compile_time_arg_val(6) != 0;
    constexpr uint32_t CB_ACC0 = 26, CB_ACC1 = 27;
    const uint32_t has_work = get_arg_val<uint32_t>(0);
    if (!has_work && !STATS) {
        return;
    }
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);
    const uint32_t block_num_tiles = get_arg_val<uint32_t>(2);
    const uint32_t copy_first_block = num_blocks % 2 != 0;
    constexpr uint32_t max_dst_tiles = DST_ACCUM_MODE ? 4 : 8;  // fp32 dest (STATS) halves DST
    CircularBuffer cb_in(cb_in0);
    CircularBuffer cb_out(cb_out0);
    CircularBuffer cb_res(cb_residual);
    compute_kernel_hw_startup(cb_in0, cb_in1, cb_out0);
    for (uint32_t it = 0; it < ITERS; ++it) {
        if (has_work && SRED) {
            const uint32_t CT = block_num_tiles;
            uint32_t acc = CB_ACC0;
            for (uint32_t k = 1; k < num_blocks; ++k) {
                const uint32_t ok = get_arg_val<uint32_t>(3 + k);
                const bool first = k == 1, last = k == num_blocks - 1;
                cb_in.wait_front((k + 1) * CT);
                const uint32_t a_cb = first ? cb_in0 : acc;
                const uint32_t a_base = first ? get_arg_val<uint32_t>(3) * CT : 0;
                const uint32_t dst = last ? cb_out0 : (acc == CB_ACC0 ? CB_ACC1 : CB_ACC0);
                if (!first) {
                    cb_wait_front(a_cb, CT);
                }
                if (last && fuse_residual) {
                    cb_res.wait_front(CT);
                }
                cb_reserve_back(dst, CT);
                reconfig_data_format(a_cb, cb_in0);
                pack_reconfig_data_format(dst);
                for (uint32_t t0 = 0; t0 < CT; t0 += max_dst_tiles) {
                    const uint32_t n = std::min(max_dst_tiles, CT - t0);
                    add_init(a_cb, cb_in0);
                    tile_regs_acquire();
                    for (uint32_t i = 0; i < n; ++i) {
                        add_tiles(a_cb, cb_in0, a_base + t0 + i, ok * CT + t0 + i, i);
                    }
                    if (last && fuse_residual) {
                        reconfig_data_format_srca(a_cb, cb_residual);
                        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_residual);
                        for (uint32_t i = 0; i < n; ++i) {
                            add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_residual, t0 + i, i);
                        }
                        reconfig_data_format_srca(cb_residual, a_cb);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < n; ++i) {
                        pack_tile(i, dst, t0 + i);
                    }
                    tile_regs_release();
                }
                cb_push_back(dst, CT);
                if (!first) {
                    cb_pop_front(a_cb, CT);
                }
                acc = dst;
            }
            cb_in.pop_front(num_blocks * CT);
            if constexpr (fuse_residual) {
                cb_res.pop_front(CT);
            }
            reconfig_data_format(cb_in0, cb_in1);
            pack_reconfig_data_format(cb_out0);
        } else if (has_work) {
        cb_in.wait_front(num_blocks * block_num_tiles);
        if constexpr (fuse_residual) {
            cb_res.wait_front(block_num_tiles);
        }
        cb_out.reserve_back(block_num_tiles);
        add_init(cb_in0, cb_in1, true);
        const uint32_t num_pack_iters = (block_num_tiles + max_dst_tiles - 1) / max_dst_tiles;
        uint32_t cnt = 0;
        for (uint32_t p = 0; p < num_pack_iters; ++p) {
            const uint32_t n = std::min(max_dst_tiles, block_num_tiles - cnt);
            if constexpr (fuse_residual) {
                if (p > 0) {
                    add_init(cb_in0, cb_in1, true);
                }
            }
            tile_regs_acquire();
            for (uint32_t block = 0; block < num_blocks; block += 2) {
                if (!(copy_first_block && block == 0)) {
                    for (uint32_t i = 0; i < n; ++i) {
                        add_tiles(cb_in0, cb_in1, block * block_num_tiles + p * max_dst_tiles + i,
                                  (block + 1) * block_num_tiles + p * max_dst_tiles + i, i);
                    }
                }
            }
            if constexpr (fuse_residual) {
                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_residual);
                for (uint32_t i = 0; i < n; ++i) {
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_residual, p * max_dst_tiles + i, i);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < n; ++i) {
                pack_tile(i, cb_out0, p * max_dst_tiles + i);
            }
            tile_regs_release();
            cnt += n;
        }
        cb_out.push_back(block_num_tiles);
        cb_in.pop_front(num_blocks * block_num_tiles);
        if constexpr (fuse_residual) {
            cb_res.pop_front(block_num_tiles);
        }
        }  // has_work
        if constexpr (STATS) {
            const uint32_t PT = block_num_tiles;
            cb_wait_front(cb_out0, PT);
            cb_reserve_back(CB_SQ, PT);
            reconfig_data_format(cb_out0, cb_out0);
            pack_reconfig_data_format(CB_SQ);
            mul_init(cb_out0, cb_out0);
            for (uint32_t t = 0; t < PT; ++t) {
                tile_regs_acquire();
                mul_tiles(cb_out0, cb_out0, t, t, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, CB_SQ);
                tile_regs_release();
            }
            cb_push_back(CB_SQ, PT);
            cb_wait_front(CB_SQ, PT);
            cb_wait_front(CB_SCALER, 1);
            cb_reserve_back(CB_PART, 1);
            reconfig_data_format(CB_SQ, CB_SCALER);
            pack_reconfig_data_format(CB_PART);
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_SQ, CB_SCALER, CB_PART);
            tile_regs_acquire();
            for (uint32_t t = 0; t < PT; ++t) {
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_SQ, CB_SCALER, t, 0, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_PART);
            tile_regs_release();
            reduce_uninit();
            cb_push_back(CB_PART, 1);
            cb_pop_front(CB_SQ, PT);
            // back to the reduction formats for the next iteration
            reconfig_data_format(cb_in0, cb_in1);
            pack_reconfig_data_format(cb_out0);
        }
    }
}
