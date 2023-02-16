#include <cstdint>

#include "compute_hlk_api.h"

struct hlk_args_t {
    uint32_t B;
    uint32_t Ht;
    uint32_t Wt;
};

void compute_main(const hlk_args_t *args) {
    constexpr uint32_t onetile = 1;

    cb_wait_front(CB::c_in1, onetile);

    for (uint32_t b = 0; b < args->B; b++) {
    for (uint32_t h = 0; h < args->Ht; h++) {
    for (uint32_t w = 0; w < args->Wt; w++) {
        cb_reserve_back(CB::c_out0, onetile);

        acquire_dst(DstMode::Half);

        cb_wait_front(CB::c_in0, onetile);

        BCAST_OP((int) Dim::RC, CB::c_in0, CB::c_in1, 0, 0, 0);
        pack_tile(0, CB::c_out0);

        cb_pop_front(CB::c_in0, onetile);

        release_dst(DstMode::Half);

        cb_push_back(CB::c_out0, onetile);
    } } }

    cb_pop_front(CB::c_in1, onetile);
}
