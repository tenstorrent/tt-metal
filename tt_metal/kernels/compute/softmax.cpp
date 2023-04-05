#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_COL

#define BCAST_LLKOP EltwiseBinaryType::ELWADD
#define BCAST_DIM BroadcastType::COL

#include "llk_3c.h"

#include "debug_print.h"

// Synopsis of CBs used:
// c_in0 : 2 tiles buffer (input)
// c_out0: 2 tiles buffer (output)
// c_im0 : 32 tiles
// c_im1 : 32 tiles
// c_im2 : 2 tiles
// c_im3 : 1 tile (zeros)
//
namespace NAMESPACE {
void MAIN {

    uint32_t NC = get_compile_time_arg_val(0);
    uint32_t Ht = get_compile_time_arg_val(1);
    uint32_t Wt = get_compile_time_arg_val(2);

    DPRINT << "NC=" << NC << " Ht=" << Ht << " Wt=" << Wt << ENDL();

    binary_op_init_common(CB::c_in0, CB::c_in1);

    constexpr uint32_t onetile = 1;
    // reserve one tile for zeros on cb_in2
    // TODO(AP): check that if DST is indeed zeroed by release_dst (and initially), we can use it as zeroes

    // We only do the reserve for the intermediates once and use pack_tile
    // So effectively these are used as pre-allocated arrays
    // Note that the entire W dimension must fit in the intermed0 CB for this kernel to be correct
    //- cb_reserve_back(CB::c_intermed0, Wt);

    //- cb_reserve_back(CB::c_intermed3, 1);
    //- pack_tile(0, CB::c_intermed3, 0); // DST[0] should contain zeros at kernel startup, copy these zeros to c_intermed3[0]

    //- cb_reserve_back(CB::c_intermed1, 2*onetile); // just need space for 2 tiles in intermed1

    for (uint32_t nc = 0; nc < NC; nc++) {

        constexpr int onetile = 1;
        int reduce_dst_idx = 1;
        for (uint32_t ht = 0; ht < Ht; ++ht) {

            acquire_dst(DstMode::Half);
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                cb_wait_front(CB::c_in0, onetile);
                //DPRINT  << TILESAMPLES8(true, 0, 0, 4, 0, 8) << ENDL();
                //- copy_tile_init();
                //- copy_tile(CB::c_in0, 0, 0); // copy from c_in[0] to DST[0]

                //- exp_tile_init();
                //- exp_tile(0); // exp on DST[0]

                // make a copy of the exp tile in intermed0
                //- pack_tile(0, CB::c_intermed0, wt); // overwrite c_in0[0] back from DST[0]
                //DPRINT  << TILESAMPLES8(true, 0, 0, 4, 0, 8) << ENDL();
                //DPRINT  << TILESAMPLES8(false, 0, 0, 4, 0, 8) << ENDL(); // c_in0
                //DPRINT  << TILESAMPLES8(false, 26, 0, 4, 0, 8) << ENDL(); // intermed2


                // DST[1] += reduce(interm0[wt])
                //- reduce_init(PoolType::SUM, ReduceDim::REDUCE_COL, CB::c_intermed0, 1.0f);
                //- reduce_tile(PoolType::SUM, ReduceDim::REDUCE_COL, CB::c_intermed0, wt, reduce_dst_idx, 1.0f);
                cb_pop_front(CB::c_in0, onetile);
            }

            // Now we have the reduce-W in column 0 of DST[1], we need to broadcast the reduce to the entire tile (all columns)
            // Dim::C is in row direction
            // c_intermed3[0] is zeroes
            //- pack_tile(reduce_dst_idx, CB::c_intermed1, 0); // DST[1] -> intermed1[0]

            //- init_bcast(CB::c_intermed1, CB::c_intermed3);
            // TODO(AP): first arg is actually ignored right now, instead the BCAST_DIM macro is used in llk_3c.h
            //- add_tiles_bcast(tt::Dim::C, CB::c_intermed1, CB::c_intermed3, 0, 0, 2); // (DST[1] + 0) bcast_w-> DST[2]

            //- recip_tile_init();
            //- recip_tile(2); // DST[2] = 1/sum(exp(x))

            //- pack_tile(2, CB::c_intermed1, 1); // DST[2] -> intermed1[1]

            // now c_intermed0 has exp tiles, need to multiply by our DST[2]
            for(uint32_t wt = 0; wt < Wt; ++wt) {
                cb_reserve_back(CB::c_out0, onetile);

                uint32_t dst_idx = 0;
                //- mul_tiles_init();
                //- mul_tiles(CB::c_intermed0, CB::c_intermed1, wt, 1, dst_idx); // tile *= 1/(sum(exp(x)))

                //- pack_tile(dst_idx, CB::c_out0);
                cb_push_back(CB::c_out0, onetile);
            }
            release_dst(DstMode::Full);
        }
    }
}
}
