// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// TRISCs of the SWA decode qkv split + RoPE + KV-cache update. Q and K cores: RoPE of the six cb_x tiles as
// rotary_embedding_llama does it, out = x * cos + (x @ T) * sin (cos / sin rows are the users' positions, T the
// pair-rotation tile; identity past the RoPE dims) -> cb_rot. K and V cores: their user's six BFP8 cache tiles to
// bf16 (cb_c16) for the writer to patch the new row in, then, once it signals cb_sig, back to BFP8 (cb_out8).
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"

constexpr uint32_t kRoleQ = 0, kRoleK = 1, kRoleV = 2;

namespace {

FORCE_INLINE void pack_to(uint32_t cb) {
    tile_regs_commit();
    cb_reserve_back(cb, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    pack_tile(0, cb);
    tile_regs_release();
    cb_push_back(cb, 1);
}

// Each tile of `in` copied through Dest into `out` (format conversion).
template <uint32_t tiles>
FORCE_INLINE void convert(uint32_t in, uint32_t out) {
    cb_wait_front(in, tiles);
    reconfig_data_format_srca(in);
    copy_tile_to_dst_init_short(in);
    for (uint32_t j = 0; j < tiles; j++) {
        tile_regs_acquire();
        copy_tile(in, j, 0);
        pack_to(out);
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = get_compile_time_arg_val(1);
    constexpr uint32_t cb_cos = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sin = get_compile_time_arg_val(3);
    constexpr uint32_t cb_trans = get_compile_time_arg_val(4);
    constexpr uint32_t cb_tmp = get_compile_time_arg_val(5);
    constexpr uint32_t cb_rot = get_compile_time_arg_val(6);
    constexpr uint32_t cb_cache = get_compile_time_arg_val(7);
    constexpr uint32_t cb_c16 = get_compile_time_arg_val(8);
    constexpr uint32_t cb_sig = get_compile_time_arg_val(9);
    constexpr uint32_t cb_out8 = get_compile_time_arg_val(10);
    constexpr uint32_t d_tiles = get_compile_time_arg_val(11);

    compute_kernel_hw_startup(cb_trans, cb_x, cb_rot);
    if constexpr (role != kRoleV) {
        cb_wait_front(cb_x, d_tiles);
        cb_wait_front(cb_cos, d_tiles);
        cb_wait_front(cb_sin, d_tiles);
        cb_wait_front(cb_trans, 1);
        for (uint32_t j = 0; j < d_tiles; j++) {
            // tmp = x @ T
            reconfig_data_format(cb_trans, cb_x);
            matmul_init(cb_x, cb_trans);
            tile_regs_acquire();
            matmul_tiles(cb_x, cb_trans, j, 0, 0);
            pack_to(cb_tmp);
            // rot = x * cos + tmp * sin
            cb_wait_front(cb_tmp, 1);
            reconfig_data_format(cb_x, cb_cos);
            mul_init(cb_x, cb_cos, false);
            tile_regs_acquire();
            mul_tiles(cb_x, cb_cos, j, j, 0);
            reconfig_data_format(cb_tmp, cb_sin);
            mul_init(cb_tmp, cb_sin, true);
            mul_tiles(cb_tmp, cb_sin, 0, j, 0);
            pack_to(cb_rot);
            cb_pop_front(cb_tmp, 1);
        }
        cb_pop_front(cb_x, d_tiles);
        cb_pop_front(cb_cos, d_tiles);
        cb_pop_front(cb_sin, d_tiles);
        cb_pop_front(cb_trans, 1);
    }
    if constexpr (role != kRoleQ) {
        convert<d_tiles>(cb_cache, cb_c16);
        cb_pop_front(cb_cache, d_tiles);
        cb_wait_front(cb_sig, 1);
        convert<d_tiles>(cb_c16, cb_out8);
        cb_pop_front(cb_c16, d_tiles);
        cb_pop_front(cb_sig, 1);
    }
}
