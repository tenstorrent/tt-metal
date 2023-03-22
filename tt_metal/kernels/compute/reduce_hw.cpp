#include <cstdint>

#include "llk_3c.h"

namespace NAMESPACE {
void MAIN {
    uint32_t scaler = get_compile_time_arg_val(0);
    uint32_t Ht = get_compile_time_arg_val(1);
    uint32_t Wt = get_compile_time_arg_val(2);
    uint32_t NC = get_compile_time_arg_val(3);
    union { float f; uint32_t u; } u; u.u = scaler;
    reduce_init(REDUCE_OP, REDUCE_DIM, CB::c_in0, u.f);

    for (uint32_t nc = 0; nc < NC; nc++) {
        constexpr int onetile = 1;
        int reduce_dst_idx = 0;
        acquire_dst(DstMode::Full);
        for(uint32_t ht = 0; ht < Ht; ++ht) {
            // tiles are expected to be coming in in NCHW order (W-contiguous)
            // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
            // in this case we just sequentially add to accumulator all the W-tiles in a row
            for(uint32_t wt = 0; wt < Wt; ++wt) {
                cb_wait_front(CB::c_in0, onetile);
                // REDUCE_OP/DIM is expected to come from add_define
                reduce_tile(REDUCE_OP, REDUCE_DIM, CB::c_in0, 0, reduce_dst_idx, scaler);
                cb_pop_front(CB::c_in0, onetile);
            }

        }
        cb_reserve_back(CB::c_out0, onetile);
        pack_tile(reduce_dst_idx, CB::c_out0);
        cb_push_back(CB::c_out0, onetile);
        release_dst(DstMode::Full);
    }
}
}
