#include <cstdint>

#include "compute_hlk_api.h"

struct hlk_args_t {
    std::int32_t NHtWt; // can be interpreted as "num blocks", "num tensors", "z-dim", or "batch" loop
};

void compute_main(const hlk_args_t *args) {

    // transpose a row-major block:
    // - assumes the tiles come in in column major order from reader
    // - uses reader_unary_transpose_wh 
    // - transpose_wh each tile 
    for (int n = 0; n < args->NHtWt; n++) {
        cb_wait_front(CB::c_in0, 1);
        cb_reserve_back(CB::c_out0, 1);

        acquire_dst(DstMode::Half);
        transpose_wh_tile(CB::c_in0, 0, 0);
        pack_tile(0, CB::c_out0);
        release_dst(DstMode::Half);

        cb_push_back(CB::c_out0, 1);
        cb_pop_front(CB::c_in0, 1);
    }
}
