// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Megakernel AR, owner-core compute phase (AR_INTERFACE.md): one all-reduce per iter(). Owner (has_work): sum the
// ring partial slots of chunk j (+ the residual) into the output chunk; every output core (STATS): its row statistic
// sum(x^2) x 2^-12 (REDUCE_ROW, fp32) for the receiver's CB_RSTD send. Used by the standalone persistent compute
// (mk_ps_reduction_h.cpp) and by the engine compute right after the v12 owner sum (C-OSUM).
// Config macros: MKAR_CB_RED, MKAR_CB_OUT, MKAR_FUSE_RES, MKAR_CB_RES, MKAR_STATS, MKAR_CB_SQ, MKAR_CB_PART,
// MKAR_CB_SCALER; MKAR_NOOWN + MKAR_CB_OWN: no own slot in CB_RED, the own partial is read from CB MKAR_CB_OWN
// (tile layout, R = 32 only; never popped here). Runtime (from rbase): [has_work, num_blocks (= ring), block_tiles (chunk tiles / pseudo-tiles)].
// The engine runs this in the middle of other phases, so iter() sets its own unpack / pack formats on entry.
#pragma once
#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reduce.h"
#include "api/dataflow/circular_buffer.h"

namespace mkar {

struct Reduce {
    uint32_t has_work = 0, num_blocks = 0, block_tiles = 0;

    FORCE_INLINE bool init(uint32_t rbase) {
        has_work = get_arg_val<uint32_t>(rbase);
        num_blocks = get_arg_val<uint32_t>(rbase + 1);
        block_tiles = get_arg_val<uint32_t>(rbase + 2);
        return has_work || MKAR_STATS;
    }

    FORCE_INLINE void iter() const {
        constexpr uint32_t cb_in0 = MKAR_CB_RED, cb_in1 = MKAR_CB_RED, cb_out0 = MKAR_CB_OUT;
        constexpr uint32_t max_dst_tiles = DST_ACCUM_MODE ? 4 : 8;
        CircularBuffer cb_in(cb_in0), cb_out(cb_out0), cb_res(MKAR_CB_RES);
        if (has_work) {
            const uint32_t copy_first_block = num_blocks % 2 != 0;
            cb_in.wait_front(num_blocks * block_tiles);
            if constexpr (MKAR_FUSE_RES) {
                cb_res.wait_front(block_tiles);
            }
            cb_out.reserve_back(block_tiles);
            reconfig_data_format(cb_in0, cb_in1);
            pack_reconfig_data_format(cb_out0);
            add_init(cb_in0, cb_in1, true);
            const uint32_t num_pack_iters = (block_tiles + max_dst_tiles - 1) / max_dst_tiles;
            uint32_t cnt = 0;
            for (uint32_t p = 0; p < num_pack_iters; ++p) {
                const uint32_t n = std::min(max_dst_tiles, block_tiles - cnt);
#if defined(MKAR_NOOWN)
                if (p > 0) {
                    add_init(cb_in0, cb_in1, true);
                }
                tile_regs_acquire();
                // own slot elided (v0.9.19, engine at R = 32): num_blocks = ring - 1 remote slots, pairs (0,1)..(4,5),
                // then (last remote slot, this device's own partial, read in place from the producer CB MKAR_CB_OWN)
                for (uint32_t block = 0; block + 1 < num_blocks; block += 2) {
                    for (uint32_t i = 0; i < n; ++i) {
                        add_tiles(cb_in0, cb_in1, block * block_tiles + p * max_dst_tiles + i,
                                  (block + 1) * block_tiles + p * max_dst_tiles + i, i);
                    }
                }
                add_init(cb_in0, MKAR_CB_OWN, true);
                for (uint32_t i = 0; i < n; ++i) {
                    add_tiles(cb_in0, MKAR_CB_OWN, (num_blocks - 1) * block_tiles + p * max_dst_tiles + i, p * max_dst_tiles + i, i);
                }
#else
                if constexpr (MKAR_FUSE_RES) {
                    if (p > 0) {
                        add_init(cb_in0, cb_in1, true);
                    }
                }
                tile_regs_acquire();
                for (uint32_t block = 0; block < num_blocks; block += 2) {
                    if (!(copy_first_block && block == 0)) {
                        for (uint32_t i = 0; i < n; ++i) {
                            add_tiles(cb_in0, cb_in1, block * block_tiles + p * max_dst_tiles + i,
                                      (block + 1) * block_tiles + p * max_dst_tiles + i, i);
                        }
                    }
                }
#endif
                if constexpr (MKAR_FUSE_RES) {
                    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(MKAR_CB_RES);
                    for (uint32_t i = 0; i < n; ++i) {
                        add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(MKAR_CB_RES, p * max_dst_tiles + i, i);
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
            cb_out.push_back(block_tiles);
            cb_in.pop_front(num_blocks * block_tiles);
            if constexpr (MKAR_FUSE_RES) {
                cb_res.pop_front(block_tiles);
            }
        }
        if constexpr (MKAR_STATS) {
            const uint32_t PT = block_tiles;
            cb_wait_front(cb_out0, PT);
            cb_reserve_back(MKAR_CB_SQ, PT);
            reconfig_data_format(cb_out0, cb_out0);
            pack_reconfig_data_format(MKAR_CB_SQ);
            mul_init(cb_out0, cb_out0);
            for (uint32_t t = 0; t < PT; ++t) {
                tile_regs_acquire();
                mul_tiles(cb_out0, cb_out0, t, t, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, MKAR_CB_SQ);
                tile_regs_release();
            }
            cb_push_back(MKAR_CB_SQ, PT);
            cb_wait_front(MKAR_CB_SQ, PT);
            cb_wait_front(MKAR_CB_SCALER, 1);
            cb_reserve_back(MKAR_CB_PART, 1);
            reconfig_data_format(MKAR_CB_SQ, MKAR_CB_SCALER);
            pack_reconfig_data_format(MKAR_CB_PART);
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(MKAR_CB_SQ, MKAR_CB_SCALER, MKAR_CB_PART);
            tile_regs_acquire();
            for (uint32_t t = 0; t < PT; ++t) {
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(MKAR_CB_SQ, MKAR_CB_SCALER, t, 0, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, MKAR_CB_PART);
            tile_regs_release();
            reduce_uninit();
            cb_push_back(MKAR_CB_PART, 1);
            cb_pop_front(MKAR_CB_SQ, PT);
        }
    }
};

}  // namespace mkar
