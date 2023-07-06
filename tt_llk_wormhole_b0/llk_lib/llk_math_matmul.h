#pragma once
#include "llk_param_structs.h"

#include "ckernel_include.h"
#include "ckernel_template.h"

#include "cmath_common.h"
#include "llk_math_common.h"

#ifndef HF
#define HF 0
#endif

using namespace ckernel;

template <int NUM_FIDELITY_PHASES, DstTileFaceLayout FaceLayout=DstTileFaceLayout::ColMajor>
inline void matmul_configure_addrmod(const bool transpose, const std::uint32_t ct_dim, const std::uint32_t rt_dim, const std::uint32_t kt_dim, const std::uint32_t in0_tile_dims[2] = default_tile_dims, const std::uint32_t in1_tile_dims[2] = default_tile_dims) {

    constexpr bool high_fidelity = (NUM_FIDELITY_PHASES > 0);

    // MVMUL does D = B*A

    // Inner Loop --> 32/8 = 4 times for the full 32x16 face
    // DEST -- 8 rows are calculated each time
    // SRCB -- 8 rows are needed
    // SRCA -- full 16x16 gets used -- hardware will pair cols of A with rows of B
    // D[8,16] = B[8,16] * A[16,16]
    addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 8, .clr = 0, .cr = 0},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0);

    // copy of addr_mod_0
    addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 8, .clr = 0, .cr = 0},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
        .bias = {.incr = 1},
    }
        .set(ADDR_MOD_3);

    // reset all, increment fidelity if we have more fidelity phases
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 1},
        .srcb = {.incr = 0, .clr = 1, .cr = 1},
        .dest = {.incr = 0, .clr = 1, .cr = 1},
        .fidelity = {.incr = high_fidelity, .clr = 0},
        .bias = {.incr = 1},
    }
        .set(ADDR_MOD_5);

    if constexpr (FaceLayout == DstTileFaceLayout::ColMajor) {
    
        uint8_t srca_increment = transpose == false ? 16 : 32;
        uint8_t srca_set = transpose == false ? 32 : 16;

        addr_mod_t{
            .srca = {.incr = 0, .clr = 0, .cr = 0},
            .srcb = {.incr = 24, .clr = 0, .cr = 0},
            .dest = {.incr = 8, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_1);

        if (in1_tile_dims[TileDim::C_IDX]>FACE_C_DIM) { 
            addr_mod_t{
                .srca = {.incr =srca_increment, .clr = 0, .cr = 0},
                .srcb = {.incr = 0, .clr = 0, .cr = 1},
                .dest = {.incr = 8, .clr = 0, .cr = 0},
            }
                .set(ADDR_MOD_2);
        } else {
            addr_mod_t{
                .srca = {.incr = 32, .clr = 0, .cr = 1}, //FIXME: transpose???
                .srcb = {.incr = 16, .clr = 0, .cr = 1},
                .dest = {.incr = 0, .clr = 0, .cr = 1},
            }
                .set(ADDR_MOD_2);
        }    

        addr_mod_t{
            .srca = {.incr =srca_set, .clr = 0, .cr = 1},
            .srcb = {.incr =16, .clr = 0, .cr = 1},
            .dest = {.incr = 0, .clr = 0, .cr = 1},
            .bias = {.incr = 1},
        }
            .set(ADDR_MOD_4);

    } else {

        addr_mod_t{
            .srca = {.incr = 16, .clr = 0, .cr = 0},
            .srcb = {.incr =  0, .clr = 0, .cr = 1},
            .dest = {.incr =  8, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_1);

        addr_mod_t{
            .srca = {.incr =  0, .clr = 0, .cr = 1},
            .srcb = {.incr = 32, .clr = 0, .cr = 1},
            .dest = {.incr =  8, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_2);

        addr_mod_t{
            .srca = {.incr = 32, .clr = 0, .cr = 1},
            .srcb = {.incr = 48, .clr = 0, .cr = 1}, // cr=32 before, cr+48=16 after wrapping
            .dest = {.incr = 0, .clr = 0, .cr = 1},
            .bias = {.incr = 1},
        }
            .set(ADDR_MOD_4);
    }
}

template <int NUM_FIDELITY_PHASES, DstTileFaceLayout FaceLayout=DstTileFaceLayout::ColMajor>
inline void matmul_configure_mop(bool transpose, const std::uint32_t ct_dim, const std::uint32_t rt_dim, const std::uint32_t kt_dim, const std::uint32_t in0_tile_dims[2] = default_tile_dims, const std::uint32_t in1_tile_dims[2] = default_tile_dims) {

    // NOTE1: srca increment values are different for transpose=true
    // NOTE2: same instructions are run for RowMajor with different addr_mods
    // row-major:
    // TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A0
    // TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B0A0 // srca+=16, reset srcb
    // TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A1
    // TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B0A1 // reset srca, srcb+=32
    // TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B2A0
    // TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B2A0 // srca+=16, reset srcb
    // TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_3, 0); // B2A1
    // TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B2A1 // srca=32, srcb=16, dest=0 (addr_mod_4)

    // TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A2
    // TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B1A2 // srca+=16, reset srcb
    // TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A3
    // TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B1A3 // reset srca, srcb+=32
    // TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B3A2
    // TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B3A2 // srca+=16, reset srcb
    // TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_3, 0); // B3A3
    // TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B3A3 // reset srca/srcb/dest, increment phase (addr_mod_5)   

    // in0 - loaded to SrcB
    // in1 - loaded to SrcA

    constexpr uint replay_buf_len = 16;
    constexpr bool high_fidelity = NUM_FIDELITY_PHASES > 0;
    const bool reuse_a = ct_dim>=rt_dim;
    const std::uint32_t t_dim = reuse_a ? rt_dim : ct_dim;

    if (in1_tile_dims[TileDim::C_IDX]>FACE_C_DIM) { 
        TTI_REPLAY(replay_buf_offset, replay_buf_len, 0, 1);
    } else {
        TTI_REPLAY(replay_buf_offset, replay_buf_len/2, 0, 1);
    }
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A0
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B0A0 // srcb+=32
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B2A0
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B2A0 // srca+=16, reset srcb or srca+=32, srcb=16 

    if (in1_tile_dims[TileDim::C_IDX]>FACE_C_DIM) { 
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A1
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B0A1 // srcb+=32
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_3, 0); // B2A1
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B2A1 // srca+=16, srcb=16, dest=0 (addr_mod_4)
    }

    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A2
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B1A2 // srcb+=32

    if (in1_tile_dims[TileDim::C_IDX]>FACE_C_DIM) { 
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B3A2
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B3A2 // srca+=16, reset srcb
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A3
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B1A3 // srcb+=32
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_3, 0); // B3A3 or B2A1
    } else {
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_3, 0); // B3A2
    }
       
    if constexpr(high_fidelity) {
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B3A3 or B3A2 // reset srca/srcb/dest, increment phase (addr_mod_5)
    } else {
        if (reuse_a) {
            if (t_dim>1) {
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B3A3 or B3A2 // reset srca/srcb/dest, increment phase (addr_mod_5)
            } else {
                TTI_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_1, 0); // B3A3 or B3A2 // reset srca/srcb/dest, increment phase (addr_mod_5), clear src A
            }

        } else {
            if (t_dim>1) {
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B3A3 or B2A1 // reset srca/srcb/dest, increment phase (addr_mod_5)
            } else {
                TTI_MVMUL(p_setrwc::CLR_B, 0, ADDR_MOD_1, 0); // B3A3 or B2A1 // reset srca/srcb/dest, increment phase (addr_mod_5), clear src A
            }
        }    
    }

    // TODO: can we commonize this?
    constexpr uint inner_loops = high_fidelity ? NUM_FIDELITY_PHASES : 1;
    ckernel_template tmp(1 /* outer loop */, inner_loops, TT_OP_REPLAY(replay_buf_offset, (in1_tile_dims[TileDim::C_IDX]>FACE_C_DIM) ? replay_buf_len : replay_buf_len/2, 0, 0));

    if constexpr(high_fidelity) {
        if (t_dim>1) { //
            tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_F)); // Reset fidelity phase
        } else {
            if (reuse_a) {
                tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD_F));
            } else {
                tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F));
            }    

        }    

    }
    tmp.program(instrn_buffer);
}

template <int NUM_FIDELITY_PHASES, DstTileFaceLayout FaceLayout=DstTileFaceLayout::ColMajor>
inline void llk_math_matmul_init(std::uint32_t transpose=0, const std::uint32_t in0_tile_dims[2] = default_tile_dims, const std::uint32_t in1_tile_dims[2] = default_tile_dims, const std::uint32_t ct_dim=1, const std::uint32_t rt_dim=1, const std::uint32_t kt_dim=1) {
    matmul_configure_addrmod<NUM_FIDELITY_PHASES, FaceLayout>(transpose, ct_dim, rt_dim, kt_dim, in0_tile_dims, in1_tile_dims);
    // Todo: do something with input tile dims

    matmul_configure_mop<NUM_FIDELITY_PHASES, FaceLayout>(transpose>0, ct_dim, rt_dim, kt_dim, in0_tile_dims, in1_tile_dims);
    const bool reuse_a = ct_dim>=rt_dim;
    const std::uint32_t t_dim = reuse_a ? rt_dim : ct_dim;
    if (t_dim>1) {
        if (reuse_a) {
            TTI_SETC16(CLR_DVALID_SrcB_Disable_ADDR32, CLR_DVALID_SrcB_Disable_MASK); // Disable srcB valid clear. Has to be done via dedicated instruction 
        } else {
            TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, CLR_DVALID_SrcA_Disable_MASK); // Disable srcA valid clear. Has to be done via dedicated instruction
        }    
    } else {
        TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0); 
    }
    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <int NUM_FIDELITY_PHASES, DstTileFaceLayout FaceLayout=DstTileFaceLayout::ColMajor>
inline void llk_math_matmul(uint dst_index, const bool transpose=false, const std::uint32_t in0_tile_dims[2] = default_tile_dims, const std::uint32_t in1_tile_dims[2] = default_tile_dims, const std::uint32_t ct_dim=1, const std::uint32_t rt_dim=1, const std::uint32_t kt_dim=1) {
    const bool reuse_a = ct_dim>=rt_dim;
    const std::uint32_t t_dim = reuse_a ? rt_dim : ct_dim;
    const std::uint32_t rut_dim = reuse_a ? ct_dim : rt_dim; //reuse-dim

    //Todo: do something with input tile dims

    for (uint t = 0; t < t_dim; t++) {
        for (uint rut=0; rut<rut_dim; rut++) {
            math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(dst_index+(reuse_a ? ct_dim*t+rut : t+rut*ct_dim));

            if  (t_dim == 1) {
                ckernel_template::run(instrn_buffer);

                // Done with reuse. Clear srcA or srcB valid
                if(rut == (rut_dim-1)) {
                    if (reuse_a) {
                        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD);
                    } else {
                        TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD);
                    }
                }
            } else {
                ckernel_template::run(instrn_buffer);

                if ((t+1)<t_dim) {

                    // Move to the next srcA or srcB bank 
                    if (reuse_a) {
                        if(rut == (rut_dim-1)) {
                            // Clear srcB valid as reuse is done and move to next srcB bank
                            TTI_CLEARDVALID(p_setrwc::CLR_B, 0);
                        } else {
                            // Move to the next srcB bank
                            TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD);
                        }

                    } else {
                        if(rut == (rut_dim-1)) {
                            // Clear srcA valid as reuse is done and move to next srcA bank
                            TTI_CLEARDVALID(p_setrwc::CLR_A, 0);
                        } else {
                            // Move to the next srcB bank
                            TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD);
                        }    
                    }    

                    math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(dst_index+(reuse_a ? ct_dim*(t+1)+rut : t+1+rut*ct_dim));
                    ckernel_template::run(instrn_buffer);
                } 

                if (reuse_a) {
                    // Clear srcA&B valid
                    if(rut == (rut_dim-1)) {
                        TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD); // Clear srcA valid as reuse is done and move to next srcA bank
                        TTI_CLEARDVALID(p_setrwc::CLR_B, 0); // Clear srcB valid as reuse is done and move to next srcB bank 
                    } else {
                        TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD); // Move to the next srcB and srcA bank
                    }    
                } else{
                    // Clear srcB&A valid
                    if(rut == (rut_dim-1)) {
                        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD); // Clear srcB valid as reuse is done and move to next srcB bank
                        TTI_CLEARDVALID(p_setrwc::CLR_A, 0); // Clear srcA valid as reuse is done and move to next srcA bank
                    } else {
                        TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD); // Move to the next srcA and srcB bank
                    }    
                }    
            }    

        }    
        t++;
    }
}
