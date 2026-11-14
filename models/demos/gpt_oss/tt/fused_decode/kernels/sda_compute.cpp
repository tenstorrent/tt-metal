// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Split decode attention, compute (TRISC) (fused_decode/split_attn.py, sda_common.hpp): kv head g's 8 query heads
// (rows 0..7 of the Q tile) over this slice's chunks, online softmax over groups of G chunks with the running state
// m (CB_M) and [O_0, O_1, L] (CB_O; FP32, O unnormalised, L = rowsum P in every column):
//   S_j = Q . K_j^T (+ the window's first-chunk / the position's last-chunk mask; unscaled: the softmax scale c is
//   applied inside every exp); slice 0's first group also takes the sink "chunk" (column 0 = each head's sink / c:
//   in the max and in L, no V);
//   m' = max(m, rowmax S); a = exp(c (m - m')); P_j = exp(c (S_j - m')) (P in BF16);
//   [O, L]' = a [O, L] + sum_j P_j . [V_j, ones].
// Then, with M = the max of the kv head's slices' m (all-gathered by the data movement kernels, CB_MX): the state
// scaled by w = exp(c (m - M)) goes to the writer (CB_PART), which writes it into the leader's slot. The leader
// (slice 0) sums the slots: F . pack tile k maps rows 8 i + r (slice 4 k + i, head r) onto row r; out = sum O / sum L.

// (not size-optimised: -Os made the softmax loop ~1.5x slower)

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/debug/kernel_profiler.hpp"
#include "sda_common.hpp"

using namespace sda;

namespace {
constexpr uint32_t scale_bits = get_compile_time_arg_val(0);  // FP32 bits of the softmax scale c

// formats for the next op: SrcA, SrcB operands and the pack output (compute_kernel_hw_startup runs once)
inline void cfg(uint32_t a, uint32_t b, uint32_t out) {
    reconfig_data_format(a, b);
    pack_reconfig_data_format(out);
}

// FP32-accurate exp(c x) of a state tile (values in column 0: faces 0 and 2, VectorMode::C). The accurate exp's own
// scale is an instruction immediate: c is applied by a multiply first.
inline void exp_c(uint32_t i) {
    binop_with_scalar_tile_init();
    mul_unary_tile(i, scale_bits);
    exp_tile_init<false, 0x3F800000, InputClamping::ClampToNegative, true>();
    exp_tile<false, false, InputClamping::ClampToNegative, 8, true>(i, VectorMode::C);
}
// P = exp(c x) of DST tiles 0 .. n - 1 over rows 0..7 (the heads; 4 of the 8 SFPU iterations per face)
constexpr int exp_iters = 4;
inline void exp_p(uint32_t n) {
    binop_with_scalar_tile_init();
    for (uint32_t b = 0; b < n; ++b) {
        mul_unary_tile(b, scale_bits);
    }
    exp_tile_init<false, 0x3F800000, InputClamping::ClampToNegative, false>();
    for (uint32_t b = 0; b < n; ++b) {
        exp_tile<false, false, InputClamping::ClampToNegative, exp_iters, false>(b, VectorMode::R);
    }
}
}  // namespace

void kernel_main() {
    uint32_t info = 0;
    UNPACK(info = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    MATH(info = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    PACK(info = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    const uint32_t nc = info & 0x1FFF;
    if (nc == 0) {
        return;
    }
    const bool sink = (info >> 13) & 1;
    const bool leader = (info >> 14) & 1;
    const uint32_t active = (info >> 15) & 0x3F;
    const bool first_cut = (info >> 21) & 1;
    const bool last_cut = (info >> 22) & 1;
#ifdef SDA_ONE_GROUP
    constexpr uint32_t n_groups = 1;  // the host knows a slice never holds more than G chunks (no rescale code)
#else
    const uint32_t n_groups = (nc + G - 1) / G;
#endif
    // mask tile of the slice's chunk j (AUX_ZERO: no cut)
    auto mask_of = [&](uint32_t j) -> uint32_t {
        const bool f = first_cut && j == 0, l = last_cut && j + 1 == nc;
        return f && l ? AUX_SINGLE : (f ? AUX_FIRST : (l ? AUX_LAST : AUX_ZERO));
    };

    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_Q, CB_K, CB_S);
    {
        DeviceZoneScopedN("sda-qwait");
        cb_wait_front(CB_Q, 2);
    }
    cb_wait_front(CB_AUX, AUX_TILES);
    cb_wait_front(CB_ONES, 1);

    for (uint32_t gi = 0; gi < n_groups; ++gi) {
        DeviceZoneScopedN("sda-group");
        const uint32_t j0 = gi * G;
        const uint32_t n = j0 + G < nc ? G : nc - j0;
        const bool with_sink = sink && gi == 0;
        const uint32_t ns = n + (with_sink ? 1 : 0);
#ifdef SDA_ONE_GROUP
        constexpr bool first = true;
#else
        const bool first = gi == 0;
#endif
        const bool last = gi + 1 == n_groups;

        // S_j = Q . K_j^T + mask, 4 chunks per DST block. The group CBs step by whole groups (K / V: 2 G pages,
        // S / P: G + 1) whatever the group's size: a tile index never runs past the end of a CB.
        cb_wait_front(CB_K, 2 * G);
        cb_reserve_back(CB_S, G + 1);
        for (uint32_t jb = 0; jb < n; jb += 4) {
            const uint32_t nb = jb + 4 < n ? 4 : n - jb;
            cfg(CB_K, CB_Q, CB_S);
            tile_regs_acquire();
            matmul_init(CB_Q, CB_K, 1);
            bool masked = false;
            for (uint32_t b = 0; b < nb; ++b) {
                matmul_tiles(CB_Q, CB_K, 0, 2 * (jb + b), b);
                matmul_tiles(CB_Q, CB_K, 1, 2 * (jb + b) + 1, b);
                masked |= mask_of(j0 + jb + b) != AUX_ZERO;
            }
            if (masked) {
                // only the cut chunks (the window's first, the position's last) pass through the mask add
                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_AUX);
                for (uint32_t b = 0; b < nb; ++b) {
                    const uint32_t mk = mask_of(j0 + jb + b);
                    if (mk != AUX_ZERO) {
                        add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_AUX, mk, b);
                    }
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t b = 0; b < nb; ++b) {
                pack_tile(b, CB_S);
            }
            tile_regs_release();
        }
        if (with_sink) {
            // the sink tile: column 0 = sink / c (S is unscaled), columns 1..31 masked
            cfg(CB_AUX, CB_AUX, CB_S);
            tile_regs_acquire();
            copy_init(CB_AUX);
            copy_tile(CB_AUX, AUX_SINK, 0);
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_AUX);
            add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_AUX, AUX_SINKCOL, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_S);
            tile_regs_release();
        }
        cb_push_back(CB_S, G + 1);
        cb_pop_front(CB_K, 2 * G);

        // m' = max(m, rowmax S); a = exp(c (m - m')); the last group's m' also to the writer (the all-gather)
        cb_wait_front(CB_S, G + 1);
        cfg(CB_S, CB_AUX, CB_M);
        tile_regs_acquire();
        reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(CB_S, CB_AUX, CB_M);
        for (uint32_t j = 0; j < ns; ++j) {
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(CB_S, CB_AUX, j, AUX_SCALER, 0);
        }
        reduce_uninit(CB_S);
        if (!first) {
            cb_wait_front(CB_M, 1);
            reconfig_data_format_srca(CB_M);
            copy_init(CB_M);
            copy_tile(CB_M, 0, 1);
            binary_max_tile_init();
            binary_max_tile(0, 1, 0, VectorMode::C);
            sub_binary_tile_init();
            sub_binary_tile(1, 0, 1);
            exp_c(1);
        }
        tile_regs_commit();
        if (!first) {
            cb_pop_front(CB_M, 1);
            cb_reserve_back(CB_AB, 1);
        }
        cb_reserve_back(CB_M, 1);
        if (last) {
            cb_reserve_back(CB_MOUT, 1);
        }
        tile_regs_wait();
        pack_reconfig_data_format(CB_M);
        pack_tile(0, CB_M);
        if (last) {
            pack_tile(0, CB_MOUT);
        }
        if (!first) {
            pack_tile(1, CB_AB);
        }
        tile_regs_release();
        cb_push_back(CB_M, 1);
        if (last) {
            cb_push_back(CB_MOUT, 1);
        }
        if (!first) {
            cb_push_back(CB_AB, 1);
        }

        // P_j = exp(c (S_j - m')) (BF16; rows 0..7 hold the heads)
        cb_wait_front(CB_M, 1);
        cb_reserve_back(CB_P, G + 1);
        for (uint32_t jb = 0; jb < ns; jb += 4) {
            const uint32_t nb = jb + 4 < ns ? 4 : ns - jb;
            cfg(CB_S, CB_M, CB_P);
            sub_bcast_cols_init(CB_S, CB_M);
            tile_regs_acquire();
            for (uint32_t b = 0; b < nb; ++b) {
                sub_tiles_bcast_cols(CB_S, CB_M, jb + b, 0, b);
            }
            exp_p(nb);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t b = 0; b < nb; ++b) {
                pack_tile(b, CB_P);
            }
            tile_regs_release();
        }
        cb_push_back(CB_P, G + 1);
        cb_pop_front(CB_S, G + 1);

        // [O, L]' = a [O, L] + sum_j P_j . [V_j, ones] (the sink tile, P's last on the sink group: ones only)
        cb_wait_front(CB_P, G + 1);
        cb_wait_front(CB_V, 2 * G);
        tile_regs_acquire();
        if (!first) {
            cb_wait_front(CB_O, 3);
            cb_wait_front(CB_AB, 1);
            cfg(CB_O, CB_AB, CB_O);
            mul_bcast_cols_init(CB_O, CB_AB);
            mul_tiles_bcast_cols(CB_O, CB_AB, 0, 0, 0);
            mul_tiles_bcast_cols(CB_O, CB_AB, 1, 0, 1);
            mul_tiles_bcast_cols(CB_O, CB_AB, 2, 0, 2);
        }
        cfg(CB_V, CB_P, CB_O);
        matmul_init(CB_P, CB_V, 0);
        for (uint32_t j = 0; j < n; ++j) {
            matmul_tiles(CB_P, CB_V, j, 2 * j, 0);
            matmul_tiles(CB_P, CB_V, j, 2 * j + 1, 1);
            matmul_tiles(CB_P, CB_ONES, j, 0, 2);
        }
        if (with_sink) {
            matmul_tiles(CB_P, CB_ONES, n, 0, 2);
        }
        tile_regs_commit();
        if (!first) {
            cb_pop_front(CB_O, 3);
            cb_pop_front(CB_AB, 1);
        }
        cb_reserve_back(CB_O, 3);
        tile_regs_wait();
        pack_tile(0, CB_O);
        pack_tile(1, CB_O);
        pack_tile(2, CB_O);
        tile_regs_release();
        cb_push_back(CB_O, 3);
        cb_pop_front(CB_P, G + 1);
        cb_pop_front(CB_V, 2 * G);
    }

    // w = exp(c (m - M)) (M: the kv head's slices' max, from the reader), the scaled state to the writer
    {
        DeviceZoneScopedN("sda-mxwait");
        cb_wait_front(CB_MX, 1);
    }
    cb_wait_front(CB_M, 1);
    cb_reserve_back(CB_W, 1);
    cfg(CB_M, CB_MX, CB_W);
    sub_init(CB_M, CB_MX);
    tile_regs_acquire();
    sub_tiles(CB_M, CB_MX, 0, 0, 0);
    exp_c(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, CB_W);
    tile_regs_release();
    cb_push_back(CB_W, 1);
    cb_pop_front(CB_M, 1);
    cb_pop_front(CB_MX, 1);
    cb_wait_front(CB_W, 1);
    cb_wait_front(CB_O, 3);
    cb_reserve_back(CB_PART, 3);
    cfg(CB_O, CB_W, CB_PART);
    mul_bcast_cols_init(CB_O, CB_W);
    tile_regs_acquire();
    mul_tiles_bcast_cols(CB_O, CB_W, 0, 0, 0);
    mul_tiles_bcast_cols(CB_O, CB_W, 1, 0, 1);
    mul_tiles_bcast_cols(CB_O, CB_W, 2, 0, 2);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, CB_PART);
    pack_tile(1, CB_PART);
    pack_tile(2, CB_PART);
    tile_regs_release();
    cb_push_back(CB_PART, 3);
    cb_pop_front(CB_O, 3);
    cb_pop_front(CB_W, 1);

    if (!leader) {
        return;
    }
    // leader: sum the active slices' scaled states (pack tiles k < K: slot i = rows 8 i .. 8 i + 7 of slice 4 k + i)
    const uint32_t K = (active + 3) / 4;
    {
        DeviceZoneScopedN("sda-mwait");
        cb_wait_front(CB_OPACK, 3 * K);
    }
    DeviceZoneScopedN("sda-merge");
    cb_wait_front(CB_FOLD, 1);
    cb_reserve_back(CB_ACC, 3);
    cfg(CB_OPACK, CB_FOLD, CB_ACC);
    tile_regs_acquire();
    matmul_init(CB_FOLD, CB_OPACK, 0);
    for (uint32_t k = 0; k < K; ++k) {
        matmul_tiles(CB_FOLD, CB_OPACK, 0, 3 * k, 0);
        matmul_tiles(CB_FOLD, CB_OPACK, 0, 3 * k + 1, 1);
        matmul_tiles(CB_FOLD, CB_OPACK, 0, 3 * k + 2, 2);
    }
    recip_tile_init();
    recip_tile(2);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, CB_ACC);
    pack_tile(1, CB_ACC);
    pack_tile(2, CB_ACC);
    tile_regs_release();
    cb_push_back(CB_ACC, 3);
    cb_pop_front(CB_OPACK, 3 * K);
    cb_pop_front(CB_FOLD, 1);
    // out = O / L (BF16)
    cb_wait_front(CB_ACC, 3);
    cb_reserve_back(CB_OUT, 2);
    cfg(CB_ACC, CB_ACC, CB_OUT);
    mul_bcast_cols_init(CB_ACC, CB_ACC);
    tile_regs_acquire();
    mul_tiles_bcast_cols(CB_ACC, CB_ACC, 0, 2, 0);
    mul_tiles_bcast_cols(CB_ACC, CB_ACC, 1, 2, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, CB_OUT);
    pack_tile(1, CB_OUT);
    tile_regs_release();
    cb_push_back(CB_OUT, 2);
    cb_pop_front(CB_ACC, 3);
}
