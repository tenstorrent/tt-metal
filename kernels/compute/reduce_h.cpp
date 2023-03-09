#include <cstdint>

#include "compute_hlk_api.h"

void compute_main() {

    uint32_t scaler = get_compile_time_arg_val(0);
    // float scaler = *reinterpret_cast<float*>(&int_scaler);
    uint32_t Ht = get_compile_time_arg_val(1);
    uint32_t Wt = get_compile_time_arg_val(2);
    uint32_t NC = get_compile_time_arg_val(3);

    for (uint32_t nc = 0; nc < NC; nc++) {

        constexpr int onetile = 1;
        int reduce_dst_idx = 0;
        for(uint32_t wt = 0; wt < Wt; ++wt) {
            // tiles are expected to be coming in in NCWH order (H-contiguous)
            // reducing in W means out[0][w] = sum(h=0..H-1, in[h][w])
            // in this case we just sequentially add to accumulator all the H-tiles in a column
            acquire_dst(DstMode::Full);
            for(uint32_t ht = 0; ht < Ht; ++ht) {
                cb_wait_front(CB::c_in0, onetile);
                // REDUCE_OP is expected to come from add_define
                reduce_tile(REDUCE_OP, (int)Dim::R, CB::c_in0, 0, reduce_dst_idx, scaler);
                cb_pop_front(CB::c_in0, onetile);
            }

            cb_reserve_back(CB::c_out0, onetile);
            pack_tile(reduce_dst_idx, CB::c_out0);
            cb_push_back(CB::c_out0, onetile);
            release_dst(DstMode::Full);
        }
    }
}
