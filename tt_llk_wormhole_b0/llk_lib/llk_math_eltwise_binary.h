#pragma once
#include "ckernel_include.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_param_structs.h"

using namespace ckernel;

// local function declarations
inline void eltwise_binary_configure_addrmod();
inline void eltwise_binary_configure_mop(uint total_rows);

template <EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void eltwise_binary_reuse_dest_as_src() {
    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA) {
        move_d2a_fixed_face(ADDR_MOD_1);
    } else if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB) {
        move_d2b_fixed_face(ADDR_MOD_1);
    }
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    DstSync Dst = DstSync::SyncFull,
    int NUM_FIDELITY_PHASES = 0,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool is_fp32_dest_acc_en = false>
inline void llk_math_eltwise_binary(uint dst_index, bool clear_dest_acc = false) {
    constexpr bool high_fidelity = (NUM_FIDELITY_PHASES > 0);
    constexpr uint32_t ZERO_ACC_MODE = p_zeroacc::CLR_16;
    
    if constexpr ((Dst == DstSync::SyncTile16) || (Dst == DstSync::SyncTile2)) {
        math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(math_sync_tile_dst_index);

        if constexpr (eltwise_binary_type == ELWMUL) {

            if constexpr (is_fp32_dest_acc_en) {
                #pragma GCC unroll 0
                for(std::uint32_t i = 0; i < 8; i++) {
                    TT_ZEROACC(ZERO_ACC_MODE, ADDR_MOD_1, (math_sync_tile_dst_index << 3) + i);
                }    
            } else {
                #pragma GCC unroll 0
                for(std::uint32_t i = 0; i < 4; i++) {
                    TT_ZEROACC(ZERO_ACC_MODE, ADDR_MOD_1, (math_sync_tile_dst_index << 2) + i);
                }   
            }
        } else if constexpr (binary_reuse_dest != EltwiseBinaryReuseDestType::NONE) {

            if (clear_dest_acc) {
                if constexpr (is_fp32_dest_acc_en) {
                    #pragma GCC unroll 0
                    for(std::uint32_t i = 0; i < 8; i++) {
                        TT_ZEROACC(ZERO_ACC_MODE, ADDR_MOD_1, (math_sync_tile_dst_index << 3) + i);
                    }    
                } else {
                    #pragma GCC unroll 0
                    for(std::uint32_t i = 0; i < 4; i++) {
                        TT_ZEROACC(ZERO_ACC_MODE, ADDR_MOD_1, (math_sync_tile_dst_index << 2) + i);
                    }   
                }
            }
        }

    } else {
        math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(dst_index);
    }

    if constexpr ((eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB)) {
        if constexpr (src_b_bcast_type == BroadcastType::COL) {
            // Mop for col broadcast only does 2 outerloops.  Needs to clear B manually and call twice
            constexpr uint32_t outerloop = (binary_reuse_dest != EltwiseBinaryReuseDestType::NONE) ? 2 : 1;
#pragma GCC unroll 0
            for (std::uint32_t n = 0; n < outerloop; n++) {  // N-num faces
                eltwise_binary_reuse_dest_as_src<binary_reuse_dest>();
                ckernel_template::run(instrn_buffer);
            }
            TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
#pragma GCC unroll 0
            for (std::uint32_t n = 0; n < outerloop; n++) {  // N-num faces
                eltwise_binary_reuse_dest_as_src<binary_reuse_dest>();
                ckernel_template::run(instrn_buffer);
            }
            TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
        } else {
            constexpr uint32_t outerloop = (binary_reuse_dest != EltwiseBinaryReuseDestType::NONE) ? 4 : 1;
#pragma GCC unroll 0
            for (std::uint32_t n = 0; n < outerloop; n++) {  // N-num faces
                eltwise_binary_reuse_dest_as_src<binary_reuse_dest>();
                ckernel_template::run(instrn_buffer);
            }
            // Manually clear B once mop is done for scaler bcast
            if constexpr (src_b_bcast_type == BroadcastType::SCALAR) {
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_D);
            }
        }
    } else if constexpr (eltwise_binary_type == ELWMUL) {
        if constexpr (src_b_bcast_type == BroadcastType::COL) {
            // Mop for col broadcast only does 2 outerloops.  Needs to clear B manually and call twice
            constexpr uint32_t outerloop = (binary_reuse_dest != EltwiseBinaryReuseDestType::NONE) ? 2 : 1;
            if constexpr (high_fidelity) {
#pragma GCC unroll 0
                for (std::uint32_t n = 0; n < 2; n++) {  // N-num faces
                    eltwise_binary_reuse_dest_as_src<binary_reuse_dest>();
                        //fp32 zeroacc can only clear 8x16 datums at a time, need to call twice per 16x16 face
                        if constexpr (is_fp32_dest_acc_en) {
                            TT_ZEROACC(ZERO_ACC_MODE, ADDR_MOD_1, ((get_dest_buffer_base() >> 4) + (dst_index << 3)) + n*2);
                            TT_ZEROACC(ZERO_ACC_MODE, ADDR_MOD_1, ((get_dest_buffer_base() >> 4) + (dst_index << 3)) + ((n*2)+1));
                        } else {
                            TT_ZEROACC(ZERO_ACC_MODE, ADDR_MOD_1, ((get_dest_buffer_base() >> 4) + (dst_index << 2)) + n);
                        }
                    ckernel_template::run(instrn_buffer);
                }
            } else {
#pragma GCC unroll 0
                for (std::uint32_t n = 0; n < outerloop; n++) {  // N-num faces
                    eltwise_binary_reuse_dest_as_src<binary_reuse_dest>();
                    if constexpr (binary_reuse_dest != EltwiseBinaryReuseDestType::NONE) {
                        if constexpr (is_fp32_dest_acc_en) {
                            TT_ZEROACC(ZERO_ACC_MODE, ADDR_MOD_1, ((get_dest_buffer_base() >> 4) + (dst_index << 3)) + n*2);
                            TT_ZEROACC(ZERO_ACC_MODE, ADDR_MOD_1, ((get_dest_buffer_base() >> 4) + (dst_index << 3)) + ((n*2)+1));
                        } else {
                            TT_ZEROACC(ZERO_ACC_MODE, ADDR_MOD_1, ((get_dest_buffer_base() >> 4) + (dst_index << 2)) + n);
                        }
                    }
                    ckernel_template::run(instrn_buffer);
                }
            }
            TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
            if constexpr (high_fidelity) {
#pragma GCC unroll 0
                for (std::uint32_t n = 0; n < 2; n++) {  // N-num faces
                    eltwise_binary_reuse_dest_as_src<binary_reuse_dest>();
                    if constexpr (binary_reuse_dest != EltwiseBinaryReuseDestType::NONE) {
                        if constexpr (is_fp32_dest_acc_en) {
                            TT_ZEROACC(ZERO_ACC_MODE, ADDR_MOD_1, ((get_dest_buffer_base() >> 4) + (dst_index << 3)) + 2 + n*2);
                            TT_ZEROACC(ZERO_ACC_MODE, ADDR_MOD_1, ((get_dest_buffer_base() >> 4) + (dst_index << 3)) + 2 + ((n*2)+1));
                        } else {
                            TT_ZEROACC(ZERO_ACC_MODE, ADDR_MOD_1, ((get_dest_buffer_base() >> 4) + (dst_index << 2)) + 2 + n);
                        }
                    }
                    ckernel_template::run(instrn_buffer);
                }
            } else {
#pragma GCC unroll 0
                for (std::uint32_t n = 0; n < outerloop; n++) {  // N-num faces
                    eltwise_binary_reuse_dest_as_src<binary_reuse_dest>();
                    if constexpr (binary_reuse_dest != EltwiseBinaryReuseDestType::NONE) {
                        if constexpr (is_fp32_dest_acc_en) {
                            TT_ZEROACC(ZERO_ACC_MODE, ADDR_MOD_1, ((get_dest_buffer_base() >> 4) + (dst_index << 3)) + 2 + n*2);
                            TT_ZEROACC(ZERO_ACC_MODE, ADDR_MOD_1, ((get_dest_buffer_base() >> 4) + (dst_index << 3)) + 2 + ((n*2)+1));
                        } else {
                            TT_ZEROACC(ZERO_ACC_MODE, ADDR_MOD_1, ((get_dest_buffer_base() >> 4) + (dst_index << 2)) + 2 + n);
                        }
                    }
                    ckernel_template::run(instrn_buffer);
                }
            }
            TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
        } else {
            // Row and no broadcasted behaves similarly
            constexpr uint32_t outerloop = (binary_reuse_dest != EltwiseBinaryReuseDestType::NONE) ? 4 : 1;
            if constexpr (high_fidelity) {
#pragma GCC unroll 0
                for (std::uint32_t n = 0; n < 4; n++) {  // N-num faces
                    eltwise_binary_reuse_dest_as_src<binary_reuse_dest>();
                    if constexpr (binary_reuse_dest != EltwiseBinaryReuseDestType::NONE) {
                        if constexpr (is_fp32_dest_acc_en) {
                            TT_ZEROACC(ZERO_ACC_MODE, ADDR_MOD_1, ((get_dest_buffer_base() >> 4) + (dst_index << 3)) + n*2);
                            TT_ZEROACC(ZERO_ACC_MODE, ADDR_MOD_1, ((get_dest_buffer_base() >> 4) + (dst_index << 3)) + ((n*2)+1));
                        } else {
                            TT_ZEROACC(ZERO_ACC_MODE, ADDR_MOD_1, ((get_dest_buffer_base() >> 4) + (dst_index << 2)) + n);
                        }
                    }
                    ckernel_template::run(instrn_buffer);
                }
            } else {
#pragma GCC unroll 0
                for (std::uint32_t n = 0; n < outerloop; n++) {  // N-num faces
                    eltwise_binary_reuse_dest_as_src<binary_reuse_dest>();
                    if constexpr (binary_reuse_dest != EltwiseBinaryReuseDestType::NONE) {
                        if constexpr (is_fp32_dest_acc_en) {
                            TT_ZEROACC(ZERO_ACC_MODE, ADDR_MOD_1, ((get_dest_buffer_base() >> 4) + (dst_index << 3)) + n*2);
                            TT_ZEROACC(ZERO_ACC_MODE, ADDR_MOD_1, ((get_dest_buffer_base() >> 4) + (dst_index << 3)) + ((n*2)+1));
                        } else {
                            TT_ZEROACC(ZERO_ACC_MODE, ADDR_MOD_1, ((get_dest_buffer_base() >> 4) + (dst_index << 2)) + n);
                        }
                    }
                    ckernel_template::run(instrn_buffer);
                }
            }
            if constexpr (src_b_bcast_type == BroadcastType::SCALAR) {
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_D);
            }
        }
    } else {
        FWASSERT("Unsupported op!", false);
    }
    math::clear_dst_reg_addr();
}

template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type>
inline void eltwise_binary_configure_addrmod() {
    // Use srcA for data movement
    if constexpr (
        (eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB) || (eltwise_binary_type == ELWMUL)) {
        if constexpr (bcast_type == BroadcastType::NONE || bcast_type == BroadcastType::COL) {
            addr_mod_t{
                .srca = {.incr = 8},
                .srcb = {.incr = 8},
                .dest = {.incr = 8},
            }
                .set(ADDR_MOD_0);
        } else if constexpr (bcast_type == BroadcastType::ROW || bcast_type == BroadcastType::SCALAR) {
            addr_mod_t{
                .srca = {.incr = 8},
                .srcb = {.incr = 0},
                .dest = {.incr = 8},
            }
                .set(ADDR_MOD_0);
        }
        addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 0},
        }
            .set(ADDR_MOD_1);

        addr_mod_t{
            .srca = {.incr = 0, .clr = 1},
            .srcb = {.incr = 0, .clr = 1},
            .dest = {.incr = 0, .clr = 0, .cr = 1},
            .fidelity = {.incr = 1}}
            .set(ADDR_MOD_2);

        addr_mod_t{
            .srca = {.incr = 0, .clr = 1},
            .srcb = {.incr = 0, .clr = 1},
            .dest = {.incr = 8, .clr = 0, .cr = 0, .c_to_cr = 1},
            .fidelity = {.incr = 0, .clr = 1}}
            .set(ADDR_MOD_3);
    }
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType bcast_type,
    int NUM_FIDELITY_PHASES = 0,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void eltwise_binary_configure_mop() {
    constexpr bool high_fidelity = (NUM_FIDELITY_PHASES > 0);
    const uint addr_mod = ADDR_MOD_0;
    constexpr uint innerloop = 16 >> 3;  // 8 rows per eltwise op at a time.
    uint outerloop = 4;
    auto broadcast_type = p_elwise::SRCB_NO_BCAST;
    if constexpr (bcast_type == BroadcastType::COL) {
        // The mop only runs for 2 outer loops and mop is called twice for col broadcast
        outerloop = 2;
        broadcast_type = p_elwise::SRCB_BCAST_COL;
    } else if constexpr (bcast_type == BroadcastType::ROW) {
        broadcast_type = p_elwise::SRCB_BCAST_ROW;
    } else if constexpr (bcast_type == BroadcastType::SCALAR) {
        broadcast_type = p_elwise::SRCB_BCAST_ALL;
    }

    if constexpr (binary_reuse_dest != EltwiseBinaryReuseDestType::NONE) {
        outerloop = 1;
    }

    // Scalar and Col broadcast should not Clear B within a mop.  This is controlled outside of MOP.
    if constexpr (bcast_type == BroadcastType::COL || bcast_type == BroadcastType::SCALAR) {
        if constexpr (eltwise_binary_type == ELWADD) {
            ckernel_template tmp(outerloop, innerloop, TT_OP_ELWADD(0, 0, broadcast_type, addr_mod, 0));
            tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_A, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
            tmp.program(instrn_buffer);
        } else if constexpr (eltwise_binary_type == ELWSUB) {
            ckernel_template tmp(outerloop, innerloop, TT_OP_ELWSUB(0, 0, broadcast_type, addr_mod, 0));
            tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_A, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
            tmp.program(instrn_buffer);
        } else if constexpr (eltwise_binary_type == ELWMUL) {
            ckernel_template tmp(
                high_fidelity ? NUM_FIDELITY_PHASES : outerloop,
                innerloop,
                TT_OP_ELWMUL(0, 0, broadcast_type, addr_mod, 0));
            if constexpr (high_fidelity) {
                tmp.set_last_inner_loop_instr(
                    TT_OP_ELWMUL(0, 0, broadcast_type, ADDR_MOD_2, 0));  // Incr fidelity last inst of inner loop
                tmp.set_last_outer_loop_instr(TT_OP_ELWMUL(p_setrwc::CLR_A, 0, broadcast_type, ADDR_MOD_3, 0));
            } else {
                tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_A, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
            }
            tmp.program(instrn_buffer);
        }
    } else {
        if constexpr (eltwise_binary_type == ELWADD) {
            ckernel_template tmp(outerloop, innerloop, TT_OP_ELWADD(0, 0, broadcast_type, addr_mod, 0));
            tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
            tmp.program(instrn_buffer);
        } else if constexpr (eltwise_binary_type == ELWSUB) {
            ckernel_template tmp(outerloop, innerloop, TT_OP_ELWSUB(0, 0, broadcast_type, addr_mod, 0));
            tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
            tmp.program(instrn_buffer);
        } else if constexpr (eltwise_binary_type == ELWMUL) {
            ckernel_template tmp(
                high_fidelity ? NUM_FIDELITY_PHASES : outerloop,
                innerloop,
                TT_OP_ELWMUL(0, 0, broadcast_type, addr_mod, 0));
            if constexpr (high_fidelity) {
                tmp.set_last_inner_loop_instr(
                    TT_OP_ELWMUL(0, 0, broadcast_type, ADDR_MOD_2, 0));  // Incr fidelity last inst of inner loop
                tmp.set_last_outer_loop_instr(TT_OP_ELWMUL(p_setrwc::CLR_AB, 0, broadcast_type, ADDR_MOD_3, 0));
            } else {
                tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
            }
            tmp.program(instrn_buffer);
        }
    }
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    int NUM_FIDELITY_PHASES = 0,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_eltwise_binary_init() {
    eltwise_binary_configure_addrmod<eltwise_binary_type, src_b_bcast_type>();

    if constexpr (
        (eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB) || (eltwise_binary_type == ELWMUL)) {
        eltwise_binary_configure_mop<eltwise_binary_type, src_b_bcast_type, NUM_FIDELITY_PHASES, binary_reuse_dest>();
    } else {
        FWASSERT("Unsupported op!", false);
    }

    math::reset_counters(p_setrwc::SET_ABD_F);
}
