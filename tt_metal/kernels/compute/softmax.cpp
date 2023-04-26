#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "llk_3c.h"

#include "debug_print.h"

#include "tt_metal/tools/profiler/kernel_profiler.hpp"

namespace NAMESPACE {
void MAIN {

    uint32_t NC = get_compile_time_arg_val(0);
    uint32_t Ht = get_compile_time_arg_val(1);
    uint32_t Wt = get_compile_time_arg_val(2);

    kernel_profiler::init_profiler();

    //UNPACK(( DPRINT << "NC=" << NC << " Ht=" << Ht << " Wt=" << Wt << ENDL() ));

    binary_op_init_common(CB::c_in0, CB::c_in2);

    constexpr uint32_t onetile = 1;
    // reserve one tile for zeros on cb_in2
    // TODO(AP): check that if DST is indeed zeroed by release_dst (and initially), we can use it as zeroes

    // We only do the reserve for the intermediates once and use pack_tile
    // So effectively these are used as pre-allocated arrays
    // Note that the entire W dimension must fit in the intermed0 CB for this kernel to be correct
    constexpr auto cb_scaler = CB::c_in2;
    constexpr auto cb_exps = CB::c_intermed0;
    constexpr auto cb_sumexps = CB::c_intermed1;
    constexpr auto cb_recips = CB::c_intermed2;
    constexpr auto ndst = 8; // configurable size of DST block, use 1,2 for stress testing
    //constexpr auto cb_exps1 = CB::c_intermed4;

    cb_wait_front(cb_scaler, 1); // comes from the reader
            //UNPACK(( { DPRINT  << TSLICE(cb_scaler, 0, 32, 0, 1) << ENDL(); } ));
    //MATH(( DPRINT << "APPROX=" << U32(APPROX) << ENDL() ));


    constexpr uint32_t CO = 1;

    for (uint32_t nc = 0; nc < NC; nc++) {

        constexpr int onetile = 1;
        constexpr int dst0 = 0;
        //auto s8 = SliceRange::hw0_32_8();
        //auto h032 = SliceRange::h0_32_w0();
        //DPRINT << FIXP() << SETW(4) << SETP(3);
        for (uint32_t ht = 0; ht < Ht; ++ht) {

            //kernel_profiler::mark_time(7);
            cb_reserve_back(cb_exps, Wt);
            for (uint32_t wt = 0; wt < Wt; wt+=ndst) {
                //UNPACK(( DPRINT << "wt=" << wt << " " ));
                //UNPACK(( DPRINT << "ndst=" << ndst << ENDL() ));
                acquire_dst(DstMode::Half);
                cb_wait_front(CB::c_in0, ndst);
                copy_tile_init(); // need to copy from CB to DST to be able to run sfpu math
                for (uint32_t wt8 = 0; wt8 < ndst; ++wt8)
                    //UNPACK(( { DPRINT  << TILESAMPLES8(CB::c_in0, 0, 8, 0, 8) << ENDL(); } ));
                    copy_tile(CB::c_in0, wt8, wt8); // copy from c_in[0] to DST[0]

                exp_tile_init();
                for (uint32_t wt8 = 0; wt8 < ndst; ++wt8) {
                    exp_tile(wt8); // exp on DST[0]
                    // make a copy of the exp tile in cb_exps since we'll need it in second pass to compute exp(x)/sum(exp(x))
                    pack_tile(wt8, cb_exps); // DST[0]->cb_id[wt]
                }
                cb_pop_front(CB::c_in0, ndst);
                            //if (ht == 1) PACK(( DPRINT  << "Exps1 [" << ht << "," << U32(wt+wt8) << "]" << ENDL() ));
                            //if (ht == 1) PACK(( DPRINT  << TSLICE(cb_exps, wt8, s8) << ENDL() ));

                cb_push_back(cb_exps, ndst);
                release_dst(DstMode::Half);
            }
            //kernel_profiler::mark_time(8);

            acquire_dst(DstMode::Half);
            cb_reserve_back(cb_sumexps, 1*onetile);
            reduce_init_delta_v2<false>(REDUCE_OP, REDUCE_DIM);
            for (uint32_t wt = 0; wt < Wt; wt++) {
                cb_wait_front(cb_exps, wt+1); // must be a cumulative wait for correctness
                constexpr uint32_t scaler0 = 0;
                    //UNPACK((  DPRINT << TSLICE(cb_scaler, scaler0, s8) << ENDL()  ));
                    //UNPACK((  DPRINT  << "Exps2 wt=" << U32(wt) << ENDL() ));
                    //UNPACK((  DPRINT << TSLICE(cb_exps, wt, s8) << ENDL()  ));
                reduce_tile_v2(REDUCE_OP, REDUCE_DIM, cb_exps, cb_scaler, wt, scaler0, dst0);
            }
            pack_tile(dst0, cb_sumexps);
                    //PACK((   DPRINT  << "SumExps:" << ENDL() ));
                    //PACK(( { DPRINT  << TSLICE(cb_sumexps, 0, h032, false) << ENDL(); } ));
            cb_push_back(cb_sumexps, 1);
            reduce_revert_delta_v2();
            release_dst(DstMode::Half);

            //kernel_profiler::mark_time(9);
            acquire_dst(DstMode::Half);
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
            release_dst(DstMode::Half);
            cb_wait_front(cb_recips, 1); // will reuse Wt times for bcast

            //kernel_profiler::mark_time(10);

            // now cb_sumexps has exp tiles, need to multiply by our DST[2]
            // by now we already did a cumulative wait for Wt tiles in cb_exps
            mul_bcast_cols_init_short();
            for (uint32_t wt = 0; wt < Wt; wt += ndst) {
                            //if (ht == 1) UNPACK(( DPRINT << "wt_2=" << wt << " " ));
                            //if (ht == 1) UNPACK(( DPRINT << "rem8_2=" << rem8 << ENDL() ));
                acquire_dst(DstMode::Half);
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
                release_dst(DstMode::Half);
            }
            cb_pop_front(cb_recips, 1);
            cb_pop_front(cb_exps, Wt);
            //kernel_profiler::mark_time(11);
        }
    }
    //kernel_profiler::mark_time(8);
    cb_pop_front(cb_scaler, 1); // we don't actually have to do this
}
}
