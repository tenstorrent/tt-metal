// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// SigLIP residual add kernel: out[m, k] = a[m, k] + b[m, k].
//
// Elementwise add across (M=256, D=1152) bf16 tensors. 8 cores × 1 M-tile
// (32 rows) each. K_TILES = D/32 = 36 tiles per row.
//
// Pattern: standard binary_op_init_common + add_tiles_init + add_tiles loop.
// Identical to ttnn's eltwise_binary_kernel.cpp shape (which was the
// canonical reference for the LN binary_op_init_common lesson).

#include "../../../../demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#include "../../../../demos/deepseek_v3_b1/unified_kernels/kernel_utils.hpp"
#include "api/debug/dprint.h"

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#endif

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t a_cb = get_named_compile_time_arg_val("a_cb");
    constexpr uint32_t b_cb = get_named_compile_time_arg_val("b_cb");
    constexpr uint32_t out_cb = get_named_compile_time_arg_val("out_cb");
    constexpr uint32_t in_tiles = get_named_compile_time_arg_val("in_tiles");

    unified_kernels::setup_sharded_buffer(a_cb, in_tiles);
    unified_kernels::setup_sharded_buffer(b_cb, in_tiles);
    unified_kernels::setup_sharded_buffer(out_cb, in_tiles);
    DPRINT << "NC: setup. in_tiles=" << in_tiles << ENDL();

#elif defined(COMPILE_FOR_BRISC)
    // no-op

#elif defined(COMPILE_FOR_TRISC)
    constexpr uint32_t a_cb = get_named_compile_time_arg_val("a_cb");
    constexpr uint32_t b_cb = get_named_compile_time_arg_val("b_cb");
    constexpr uint32_t out_cb = get_named_compile_time_arg_val("out_cb");
    constexpr uint32_t IN_TILES = get_named_compile_time_arg_val("in_tiles");

    cb_wait_front(a_cb, IN_TILES);
    cb_wait_front(b_cb, IN_TILES);

    binary_op_init_common(a_cb, b_cb, out_cb);
    add_tiles_init(a_cb, b_cb);

    DPRINT << "TR: ready. IN_TILES=" << IN_TILES << ENDL();

    // Tile-by-tile add: dst[0] = a[i] + b[i]; pack to out_cb at i.
    cb_reserve_back(out_cb, IN_TILES);
    for (uint32_t i = 0; i < IN_TILES; ++i) {
        tile_regs_acquire();
        add_tiles(a_cb, b_cb, i, i, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, out_cb, i);
        tile_regs_release();
    }
    cb_push_back(out_cb, IN_TILES);
    cb_pop_front(a_cb, IN_TILES);
    cb_pop_front(b_cb, IN_TILES);
    DPRINT << "TR: residual add done." << ENDL();
#endif
}
