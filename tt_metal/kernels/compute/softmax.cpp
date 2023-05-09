#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "llk_3c.h"
#include "../op_config.hpp"

#include "debug_print.h"

ALWI void ACQ() { acquire_dst(DstMode::Half); }
ALWI void REL() { release_dst(DstMode::Half); }

// for scale+mask+softmax:
// bcast HW (mul by 1 tile)  example: (  [2,1,1024,64] * [1,1,32,32]  )
// bcast add H               example: ( [2,1,1024,64] + [2,1,32,64] ) (bcast W -> H)
// Note that the attention mask will not fit in L1 for the entire tensor
// The buffer for the att mask is currently sized as (1t,Wt) so we only reuse it for one HtWt-sized batch of x
// then read another Wt tiles of mask for the next batch

namespace NAMESPACE {
void MAIN {

    uint32_t NCHt = get_compile_time_arg_val(0);
    uint32_t Ht = get_compile_time_arg_val(1);
    uint32_t Wt = get_compile_time_arg_val(2);

    binary_op_init_common(CB::c_in0, CB::c_in2);
    //UNPACK(( DPRINT << "NCHt=" << NCHt << " Wt=" << Wt << ENDL() ));

    constexpr uint32_t onetile = 1;
    // reserve one tile for zeros on cb_in2
    // We only do the reserve for the intermediates once and use pack_tile
    // So effectively these are used as pre-allocated arrays
    // Note that the entire W dimension must fit in the intermed0 CB for this kernel to be correct
    constexpr auto cb_bcast_scaler = CB::c_in2;
    constexpr auto cb_fused_scale = CB::c_in3;
    constexpr auto cb_fused_attn = CB::c_in4;
    constexpr auto cb_exps = CB::c_intermed0;
    constexpr auto cb_sumexps = CB::c_intermed1;
    constexpr auto cb_recips = CB::c_intermed2;
    constexpr auto ndst = BLOCK_SIZE;
    constexpr auto cb_in0 = CB::c_in0;
    constexpr auto cb_out0 = CB::c_out0;

    //UNPACK(( { DPRINT << "Initial rd ptr(cb_exps): " << CB_RD_PTR(cb_exps) << ENDL(); } ));

    cb_wait_front(cb_bcast_scaler, 1); // comes from the reader
            //UNPACK(( { DPRINT  << TSLICE(cb_bcast_scaler, 0, 32, 0, 1) << ENDL(); } ));
            //MATH(( DPRINT << "APPROX=" << U32(APPROX) << ENDL() ));
    #if FUSED_SCALE_MASK
    cb_wait_front(cb_fused_scale, 1);
    #endif

    constexpr int dst0 = 0;
    uint32_t ht = 0;
    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
            //auto s8 = SliceRange::hw0_32_8();
            //auto s16 = SliceRange::hw0_32_16();
            //auto shw04 = SliceRange::hw041();
            //auto h032 = SliceRange::h0_32_w0();
            //DPRINT << FIXP() << SETW(4) << SETP(3);
            //kernel_profiler::mark_time(7);
            int cb_in = cb_in0;
            #if FUSED_SCALE_MASK
            for (uint32_t wt = 0; wt < Wt; wt+=ndst) {
                // apply fused scale [*= 1/sqrt(...)]
                ACQ();
                mul_tiles_bcast_scalar_init_short();
                cb_wait_front(cb_in0, ndst);
                cb_reserve_back(cb_exps, ndst);
                for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                    mul_tiles_bcast_scalar(cb_in0, cb_fused_scale, wt8, 0, wt8); // mul bcast-HW -> DST[wt8]
                    pack_tile(wt8, cb_exps); // reuse exps buffer
                    //if (ht == 0 && wt == 0) UNPACK(( DPRINT  << TSLICE(cb_in0, wt8, s16) << ENDL() ));
                    //if (ht == 0 && wt == 0) UNPACK(( DPRINT  << TSLICE(cb_fused_scale, 0, s16) << ENDL() ));
                    //if (ht == 2 && wt <= 1000) PACK(( DPRINT  << TSLICE(cb_exps, wt8, s16) << ENDL() ));
                }
                cb_push_back(cb_exps, ndst);
                cb_pop_front(cb_in0, ndst);
                REL();
            }

            for (uint32_t wt = 0; wt < Wt; wt+=ndst) {
                ACQ();
                if (ht == 0) {
                    cb_wait_front(cb_fused_attn, wt+ndst); // cumulative wait for up to Wt tiles, only at first ht
                }
                cb_wait_front(cb_exps, ndst);
                cb_reserve_back(cb_exps, ndst);
                add_bcast_rows_init_short();
                for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                    //if (ht == 0 && wt+wt8 == 0) UNPACK(( DPRINT  << TSLICE(cb_fused_attn, wt+wt8, shw04) << ENDL() ));
                    //if (ht == 0 && wt+wt8 == 0) UNPACK(( DPRINT  << TSLICE(cb_exps, wt8, shw04) << ENDL() ));
                    add_tiles_bcast_rows(cb_exps, cb_fused_attn, wt8, wt+wt8, wt8); // tile *= 1/(sum(exp(x)))
                    pack_tile(wt8, cb_exps); // reuse the exps buffer again, this time in a circular manner
                    //if (ht == 0 && wt+wt8 == 0) PACK(( { DPRINT  << TSLICE(cb_exps, wt8, shw04) << ENDL(); } ));
                }
                if (ht == Ht-1)
                    cb_pop_front(cb_fused_attn, Wt);
                cb_pop_front(cb_exps, ndst);
                cb_push_back(cb_exps, ndst);
                REL();
            }

            //UNPACK(( DPRINT << "UNP post ncht=" << ncht << ENDL() ));
            cb_in = cb_exps; // switch to cb_exps as input if we are in fused_scale_mask variant
            #endif // #if FUSED_SCALE_MASK

            for (uint32_t wt = 0; wt < Wt; wt+=ndst) {
                //UNPACK(( DPRINT << "wt=" << wt << " " ));
                //UNPACK(( DPRINT << "ndst=" << ndst << ENDL() ));
                ACQ();
                cb_wait_front(cb_in, ndst);
                copy_tile_init(); // need to copy from CB to DST to be able to run sfpu math
                for (uint32_t wt8 = 0; wt8 < ndst; ++wt8) {
                    //UNPACK(( { DPRINT  << CB_RD_PTR(cb_in) << ENDL(); } ));
                    //if (ht == 2 && wt+wt8 < 1000) UNPACK(( { DPRINT  << TSLICE8(cb_in, wt8, s16) << ENDL(); } ));
                    copy_tile(cb_in, wt8, wt8); // copy from c_in[0] to DST[0]
                }
                cb_pop_front(cb_in, ndst);

                cb_reserve_back(cb_exps, ndst);
                exp_tile_init();
                for (uint32_t wt8 = 0; wt8 < ndst; ++wt8) {
                    exp_tile(wt8); // exp on DST[0]
                    // make a copy of the exp tile in cb_exps since we'll need it in second pass to compute exp(x)/sum(exp(x))
                    pack_tile(wt8, cb_exps); // DST[0]->cb_id[wt]
                            //if (ht == 2 && wt+wt8 == 0) PACK(( { DPRINT  << "Exps1 [" << ht << "," << U32(wt+wt8) << "]" << ENDL(); } ));
                            //if (ht == 2 && wt+wt8 == 0) PACK(( { DPRINT  << TSLICE8(cb_exps, wt8, s16) << ENDL(); } ));
                }
                cb_push_back(cb_exps, ndst);
                REL();
            }
            //kernel_profiler::mark_time(8);

            ACQ();
            cb_reserve_back(cb_sumexps, 1*onetile);
            reduce_init_delta_v2<false>(REDUCE_OP, REDUCE_DIM);
            for (uint32_t wt = 0; wt < Wt; wt++) {
                cb_wait_front(cb_exps, wt+1); // must be a cumulative wait for correctness
                constexpr uint32_t bcast_scaler0 = 0; // 0th index from bcast_scaler CB
                    //UNPACK((  DPRINT << TSLICE(cb_scaler, scaler0, s8) << ENDL()  ));
                    //UNPACK((  DPRINT  << "Exps2 wt=" << U32(wt) << ENDL() ));
                    //UNPACK((  DPRINT << TSLICE(cb_exps, wt, s16) << ENDL()  ));
                reduce_tile_v2(REDUCE_OP, REDUCE_DIM, cb_exps, cb_bcast_scaler, wt, bcast_scaler0, dst0);
            }
            pack_tile(dst0, cb_sumexps);
                    //PACK((   DPRINT  << "SumExps:" << ENDL() ));
                    //PACK(( { DPRINT  << TSLICE(cb_sumexps, 0, h032, false) << ENDL(); } ));
            cb_push_back(cb_sumexps, 1);
            reduce_revert_delta_v2();
            REL();

            //kernel_profiler::mark_time(9);
            ACQ();
            cb_wait_front(cb_sumexps, 1);
            cb_reserve_back(cb_recips, 1);
            {
                copy_tile_init();
                copy_tile(cb_sumexps, 0, dst0);
                cb_pop_front(cb_sumexps, 1);

                recip_tile_init();
                recip_tile(dst0); // DST[2] = 1/sum(exp(x))
                pack_tile(dst0, cb_recips); // DST[2] -> cb_sumexps[1]
                    //PACK(( DPRINT  << "RecipSumExps:" << ENDL() ));
                    //PACK(( DPRINT  << TILESAMPLES32(cb_recips, 0, 16, 0, 16) << ENDL() ));
                cb_push_back(cb_recips, 1);
            }
            REL();
            cb_wait_front(cb_recips, 1); // will reuse Wt times for bcast

            // now cb_sumexps has exp tiles, need to multiply by our DST[2]
            // by now we already did a cumulative wait for Wt tiles in cb_exps
            mul_bcast_cols_init_short();
            for (uint32_t wt = 0; wt < Wt; wt += ndst) {
                            //if (ht == 1) UNPACK(( DPRINT << "wt_2=" << wt << " " ));
                            //if (ht == 1) UNPACK(( DPRINT << "rem8_2=" << rem8 << ENDL() ));
                ACQ();
                cb_reserve_back(CB::c_out0, ndst);
                for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                        //UNPACK(( DPRINT << "ExpsInBcast:[" << ht << "," << wt << "]" << ENDL() ));
                        //UNPACK(( DPRINT << TSLICE(cb_exps, wt+wt8, s8) << ENDL() ));
                        //UNPACK(( DPRINT << "RecipsInBcast:" << ENDL() ));
                        //UNPACK(( DPRINT << TSLICE(cb_recips, 0, s8) << ENDL() ));
                    // wt+wt8 since we pop Wt after the entire loop
                    mul_tiles_bcast(tt::Dim::R, cb_exps, cb_recips, wt+wt8, 0, wt8); // tile *= 1/(sum(exp(x)))
                    pack_tile(wt8, CB::c_out0);
                        //if (ht == 1) PACK(( DPRINT << "exp*RecipSumExps[" << ht << "," << wt+wt8 << "]" << ENDL() ));
                        //if (ht == 1) PACK(( DPRINT << TSLICE(CB::c_out0, 0, s8) << ENDL() ));
                }
                cb_push_back(CB::c_out0, ndst);
                REL();
            }
            cb_pop_front(cb_recips, 1);
            cb_pop_front(cb_exps, Wt);
            //kernel_profiler::mark_time(11);
            ht ++;
            if (ht == Ht)
                ht = 0;
    } // NCHt loop
    //cb_pop_front(cb_bcast_scaler, 1); // we don't actually have to do this
    //cb_pop_front(cb_fused_scale, 1); // we don't actually have to do this
}
}
