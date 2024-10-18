// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reduce.h"
#include "debug/dprint.h"
#include "debug/assert.h"

//#define DEBUG 1

namespace NAMESPACE {
inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    PACK( DPRINT << "======" << ENDL() );
    for (uint16_t r = 0; r < 32; ++ r) {
        SliceRange sr = SliceRange{.h0 = r, .h1 = (uint16_t)(r+1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        PACK( DPRINT << (uint)r << TileSlice(cb_id, tile_id, sr, true, untilize) << ENDL() );
    }
    PACK( DPRINT << "++++++" << ENDL() );
}

inline void add_nops(const int num_nops) {
	for(int i = 0; i < num_nops; i++) {
		TTI_NOP;
        }
}

template<EltwiseBinaryType EltOp, uint32_t in0_cb, uint32_t scale_cb, uint32_t out_cb>
void eltwise_op() {

   MATH(( llk_math_eltwise_binary_init<EltOp, NONE, MATH_FIDELITY>() ));
   UNPACK(( llk_unpack_AB_init<BroadcastType::NONE>(in0_cb, scale_cb) ));
   cb_wait_front(in0_cb, 1);

    // We do not use the result
   acquire_dst(tt::DstMode::Half);
   MATH(( llk_math_eltwise_binary<EltOp, NONE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE, DST_ACCUM_MODE>(0, 0, 0) ));
   UNPACK(( llk_unpack_AB(in0_cb, scale_cb, 0, 0) ));
   cb_reserve_back(out_cb, 1);
   pack_tile(0, out_cb);
   cb_push_back(out_cb, 1);
   release_dst(tt::DstMode::Half);

   #ifdef MM_ADD_NOPS
   PACK(add_nops(MM_NUM_NOPS));
   #endif

   reduce_revert_delta<ReduceDim::REDUCE_ROW>(out_cb);
}

void matmul_blocks(const uint32_t& in0_cb, const uint32_t& in1_cb, const uint32_t& out_cb) {
    mm_block_init_short(in0_cb, in1_cb, false, 1, 1, 1);
    unpack_reconfig_data_format(in1_cb, in0_cb);
    tile_regs_acquire();
    matmul_block(in0_cb, in1_cb, 0, 0, 0, false, 1, 1, 1);
    tile_regs_commit();
    cb_reserve_back(out_cb, 1);
    tile_regs_wait();
    pack_tile(0, out_cb);
    tile_regs_release();
    cb_push_back(out_cb, 1);
    cb_pop_front(in1_cb, 1);
}

void MAIN {
    constexpr uint32_t cb_q_in = tt::CB::c_in0;
    constexpr uint32_t cb_k_in = tt::CB::c_in1;
    constexpr uint32_t cb_v_in = tt::CB::c_in2;
    constexpr uint32_t cb_identity_scale_in = tt::CB::c_in5;
    constexpr uint32_t cb_qk_im = tt::CB::c_intermed0;
    constexpr uint32_t cb_cur_sum = tt::CB::c_intermed5;
    constexpr uint32_t cb_out = tt::CB::c_out0;


    mm_init();
    cb_wait_front(cb_q_in,1);
    cb_wait_front(cb_k_in,1);
    cb_wait_front(cb_v_in,1);
    cb_wait_front(cb_identity_scale_in, 1);
    // Make sure unpacker is always behind
    UNPACK(add_nops(1000));

    /* QK = Q @ K */
    unpack_reconfig_data_format(cb_k_in, cb_q_in);
    pack_reconfig_data_format(cb_qk_im);
    matmul_blocks(cb_q_in, cb_k_in, cb_qk_im);

    eltwise_op<ELWADD, cb_qk_im, cb_identity_scale_in, cb_cur_sum>();

    /* OUT_IM = QK @ V */
    cb_wait_front(cb_qk_im, 1);
    unpack_reconfig_data_format(cb_v_in, cb_qk_im);
    pack_reconfig_data_format(cb_out);
    matmul_blocks(cb_qk_im, cb_v_in, cb_out);

    cb_pop_front(cb_qk_im, 1);
    cb_pop_front(cb_cur_sum, 1);
    cb_pop_front(cb_q_in, 1);

    // Even if stale Q is in SrcB, the result of reduce_c is correct.
    // Indicating it is an issue where the stale buffer from double buffering
    // contributes. SrcB has Q, then cb_identity_scale_in and then QK.
    // In cb_out we either see QK * V or Q * V or 0 * V
    // print_full_tile(cb_cur_sum);
}
}
