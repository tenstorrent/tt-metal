#include <cstdint>

#include "compute_hlk_api.h"

struct hlk_args_t {
    // per-batch params
    int Ht; // number of tiles in H to expect (expected to be a full tensor by this kernel)
    int Wt; // number of tiles in W to expect (can be a partial tensor), always <= DSTt
    int NC;
    float scaler;
};

void compute_main(const hlk_args_t *args) {

    for (int nc = 0; nc < args->NC; nc++) {

        constexpr int onetile = 1;
        int reduce_dst_idx = 0;
        for(int ht = 0; ht < args->Ht; ++ht) {
            // tiles are expected to be coming in in NCHW order (W-contiguous)
            // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
            // in this case we just sequentially add to accumulator all the W-tiles in a row
            acquire_dst(DstMode::Full);
            for(int wt = 0; wt < args->Wt; ++wt) {
                cb_wait_front(CB::c_in0, onetile);
                // REDUCE_OP is expected to come from add_define
                reduce_tile(REDUCE_OP, (int)Dim::C, CB::c_in0, 0, reduce_dst_idx, args->scaler);
                cb_pop_front(CB::c_in0, onetile);
            }

            cb_reserve_back(CB::c_out0, onetile);
            pack_tile(reduce_dst_idx, CB::c_out0);
            cb_push_back(CB::c_out0, onetile);
            release_dst(DstMode::Full);
        }
    }
}
