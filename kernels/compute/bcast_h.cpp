#include <cstdint>

#include "compute_hlk_api.h"

void compute_main() {
    constexpr uint32_t onetile = 1;
    uint32_t B = get_compile_time_arg_val(0);
    uint32_t Ht = get_compile_time_arg_val(1);
    uint32_t Wt = get_compile_time_arg_val(2);

    for (uint32_t b = 0; b < B; b++) {
    for (uint32_t h = 0; h < Ht; h++) {
    for (uint32_t w = 0; w < Wt; w++) {
        // For this bcast-h op the reader will wrap the RHS source tile around at Wt
        // so here we just linearly read 2 parallel arrays and apply bcast op per tile
        // (bcast_h propagates the op down the H dimension, so it can be though of as bcast to H)
        cb_wait_front(CB::c_in1, onetile);

        cb_reserve_back(CB::c_out0, onetile);

        acquire_dst(DstMode::Half);

        cb_wait_front(CB::c_in0, onetile);

        BCAST_OP((int) Dim::R, CB::c_in0, CB::c_in1, 0, 0, 0);
        pack_tile(0, CB::c_out0);

        cb_pop_front(CB::c_in0, onetile);

        release_dst(DstMode::Half);

        cb_push_back(CB::c_out0, onetile);
        cb_pop_front(CB::c_in1, onetile);
    } } }
}
