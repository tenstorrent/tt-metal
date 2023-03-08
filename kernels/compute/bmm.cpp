#include <cstdint>

#include "compute_hlk_api.h"

using std::uint32_t;

// matmul C=A*B using dims MK*KN = MN (row major order)
//
struct hlk_args_t {
    uint32_t batch; // batch
    uint32_t Mt; // number of tiles in M
    uint32_t Kt; // number of tiles in K
    uint32_t Nt; // number of tiles in N
};

void compute_main(const hlk_args_t *args) {

    constexpr int onetile = 1;

    int dst_tile_index = 0;
    int in0_block_tile_index = 0;
    int Mt = args->Mt;
    int Kt = args->Kt;
    int Nt = args->Nt;

    // the simplest possible version of outer product blocked matmul
    // the reader is expected to read the A's and B's tile rows and tile columns for each output tile
    for (uint32_t nb = 0; nb < args->batch; nb++)
    for (uint32_t mt_C = 0; mt_C < args->Mt; ++mt_C) // output tile of C
    for (uint32_t nt_C = 0; nt_C < args->Nt; ++nt_C) // output tile index of C
    {
        acquire_dst(DstMode::Full);
        for (uint32_t kt = 0; kt < args->Kt; kt++) {
            cb_wait_front(CB::c_in0, onetile);
            cb_wait_front(CB::c_in1, onetile);

            matmul_tiles(CB::c_in0, CB::c_in1, 0, 0, 0, false);

            cb_pop_front(CB::c_in0, onetile);
            cb_pop_front(CB::c_in1, onetile);
        }

        cb_reserve_back(CB::c_out0, onetile);
        pack_tile(0, CB::c_out0);
        cb_push_back(CB::c_out0, onetile);

        release_dst(DstMode::Full);
    }


}
