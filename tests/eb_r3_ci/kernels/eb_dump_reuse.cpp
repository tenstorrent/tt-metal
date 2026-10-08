// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Round 3 eltwise binary (#58818 review) dump kernel: DEST <- tile of c_0 (copy_tile), then the dest-reuse binary op with the
// tile of c_1, packed to c_16. EB_DUMP_PER_TILE (host define, 0 or 1) is the dest-reuse forms' hand-off.
// CT args: n tiles, op (0 add, 1 sub, 2 mul), form (0 DEST_TO_SRCA, 1 DEST_TO_SRCB, 2 row broadcast DEST_TO_SRCA as in
// bge_m3's balanced layernorm).
#define ELTWISE_BINARY_PER_TILE_HANDOFF_DEST_REUSE EB_DUMP_PER_TILE
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"

using namespace ckernel;

// bge_m3 balanced layernorm's row-broadcast dest-reuse form (Blackhole branch of the PR)
template <EltwiseBinaryType op>
ALWI void row_bcast_reuse_init(uint32_t cb) {
    constexpr auto src_dvalid = ckernel::detail::binary_src_dvalid<EltwiseBinaryReuseDestType::DEST_TO_SRCA>;
    UNPACK((llk_unpack_A_init<BroadcastType::ROW, true, EltwiseBinaryReuseDestType::DEST_TO_SRCA, false, src_dvalid>(
        false, false, cb)));
    MATH((llk_math_eltwise_binary_init<
          op,
          BroadcastType::ROW,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::DEST_TO_SRCA,
          src_dvalid>(cb, cb, false)));
}

template <EltwiseBinaryType op>
ALWI void row_bcast_reuse_tile(uint32_t cb, uint32_t itile, uint32_t idst) {
    UNPACK((llk_unpack_A<BroadcastType::ROW, true, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb, itile)));
    MATH((llk_math_eltwise_binary<
          op,
          BroadcastType::ROW,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::DEST_TO_SRCA,
          ckernel::detail::binary_src_dvalid<EltwiseBinaryReuseDestType::DEST_TO_SRCA>>(cb, cb, idst, true)));
}

void kernel_main() {
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t op = get_compile_time_arg_val(1);
    constexpr uint32_t form = get_compile_time_arg_val(2);
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr EltwiseBinaryType ET = op == 0   ? EltwiseBinaryType::ELWADD
                                     : op == 1 ? EltwiseBinaryType::ELWSUB
                                               : EltwiseBinaryType::ELWMUL;
    constexpr EltwiseBinaryReuseDestType RD =
        form == 1 ? EltwiseBinaryReuseDestType::DEST_TO_SRCB : EltwiseBinaryReuseDestType::DEST_TO_SRCA;

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);
    for (uint32_t i = 0; i < n; ++i) {
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);
        cb_reserve_back(cb_out, 1);
        tile_regs_acquire();
        copy_init(cb_a);
        copy_tile(cb_a, 0, 0);
        if constexpr (form == 2) {
            row_bcast_reuse_init<ET>(cb_b);
            row_bcast_reuse_tile<ET>(cb_b, 0, 0);
        } else {
            ckernel::detail::binary_reuse_dest_init<ET, RD>(cb_b, __LINE__);
            ckernel::detail::binary_reuse_dest_tiles<ET, RD>(cb_b, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);
    }
}
