#include <cstdint>

#include "debug_print.h"

#include "llk_3c.h"

namespace NAMESPACE {
void MAIN {

    uint32_t scaler = get_compile_time_arg_val(0);
    uint32_t Ht = get_compile_time_arg_val(1);
    uint32_t Wt = get_compile_time_arg_val(2);
    uint32_t NC = get_compile_time_arg_val(3);

    union { float f; uint32_t u; } u; u.u = scaler;

    if (REDUCE_OP == PoolType::MAX)
        reduce_init(REDUCE_OP, REDUCE_DIM, CB::c_in0, u.f);
    else
        reduce_init_v2<true>(REDUCE_OP, REDUCE_DIM, CB::c_in0, CB::c_in2);

    cb_wait_front(CB::c_in2, 1); // scaler tile from the reader

    for (uint32_t nc = 0; nc < NC; nc++) {

        constexpr int onetile = 1;
        int reduce_dst_idx = 0;
        for(uint32_t wt = 0; wt < Wt; ++wt) {
            // tiles are expected to be coming in in NCWH order (H-contiguous)
            // reducing in W means out[0][w] = sum(h=0..H-1, in[h][w])
            // in this case we just sequentially add to accumulator all the H-tiles in a column
            acquire_dst(DstMode::Half);
            for(uint32_t ht = 0; ht < Ht; ++ht) {
                cb_wait_front(CB::c_in0, onetile);
                // REDUCE_OP is expected to come from add_define
                //UNPACK(( DPRINT << "Scaler rd_ptr=" << U32(cb_read_interface[get_operand_id(CB::c_in2)].fifo_rd_ptr << 4) << ENDL{} ));
                //for (uint32_t offs = 0; offs < 1024; offs += 16) {
                //    UNPACK(( DPRINT << offs ));
                //    UNPACK(( DPRINT << TILESAMPLES32(false, CB::c_in2, 0, 16, offs, 1) << ENDL{} ));
                //}
                // TODO(AP): need to fix reduce_max with _v2 API
                if (REDUCE_OP == PoolType::MAX)
                    reduce_tile(REDUCE_OP, REDUCE_DIM, CB::c_in0, 0, reduce_dst_idx, scaler);
                else
                    reduce_tile_v2(REDUCE_OP, REDUCE_DIM, CB::c_in0, CB::c_in2, 0, 0, reduce_dst_idx);
                cb_pop_front(CB::c_in0, onetile);
            }

            cb_reserve_back(CB::c_out0, onetile);
            pack_tile(reduce_dst_idx, CB::c_out0);
            cb_push_back(CB::c_out0, onetile);
            release_dst(DstMode::Half);
        }
    }
}
}
