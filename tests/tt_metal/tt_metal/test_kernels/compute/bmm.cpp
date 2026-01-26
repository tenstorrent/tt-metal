// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "experimental/circular_buffer.h"

using std::uint32_t;

// matmul C=A*B using dims MK*KN = MN (row major order)
//
namespace NAMESPACE {
void MAIN {
    constexpr int onetile = 1;

    int dst_tile_index = 0;
    int in0_block_tile_index = 0;

    uint32_t batch = get_compile_time_arg_val(0);
    uint32_t Mt = get_compile_time_arg_val(1);
    uint32_t Kt = get_compile_time_arg_val(2);
    uint32_t Nt = get_compile_time_arg_val(3);

    experimental::CircularBuffer cb0(tt::CBIndex::c_0);
    experimental::CircularBuffer cb1(tt::CBIndex::c_1);
    experimental::CircularBuffer cb16(tt::CBIndex::c_16);

    mm_init(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);

    // the simplest possible version of outer product blocked matmul
    // the reader is expected to read the A's and B's tile rows and tile columns for each output tile
    for (uint32_t nb = 0; nb < batch; nb++) {
        for (uint32_t mt_C = 0; mt_C < Mt; ++mt_C) {    // output tile of C
            for (uint32_t nt_C = 0; nt_C < Nt; ++nt_C)  // output tile index of C
            {
                acquire_dst();
                for (uint32_t kt = 0; kt < Kt; kt++) {
                    cb0.wait_front(onetile);
                    cb1.wait_front(onetile);

                    matmul_tiles(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0);

                    cb0.pop_front(onetile);
                    cb1.pop_front(onetile);
                }

                cb16.reserve_back(onetile);
                pack_tile(0, tt::CBIndex::c_16);
                cb16.push_back(onetile);

                release_dst();
            }
        }
    }
}
}  // namespace NAMESPACE
