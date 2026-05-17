// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// SigLIP GELU activation kernel: out[m, k] = GELU(x[m, k]).
//
// SigLIP-So400m uses gelu_pytorch_tanh — the tanh-approximation GELU,
// matched by the LLK's gelu_tile (fast_and_approx=true, the default).
//
// Shape: (M=256, D=4320) bf16 for the FC1→GELU→FC2 path. 8 cores × 1 M-tile
// (32 rows) each. K_TILES = 4320/32 = 135 per row per core.

#include "../../../../demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#include "../../../../demos/deepseek_v3_b1/unified_kernels/kernel_utils.hpp"
#include "api/debug/dprint.h"

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"  // binary_op_init_common
#endif

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t in_cb = get_named_compile_time_arg_val("in_cb");
    constexpr uint32_t out_cb = get_named_compile_time_arg_val("out_cb");
    constexpr uint32_t in_tiles = get_named_compile_time_arg_val("in_tiles");

    unified_kernels::setup_sharded_buffer(in_cb, in_tiles);
    unified_kernels::setup_sharded_buffer(out_cb, in_tiles);
    DPRINT << "NC: setup. in_tiles=" << in_tiles << ENDL();

#elif defined(COMPILE_FOR_BRISC)
    // no-op

#elif defined(COMPILE_FOR_TRISC)
    constexpr uint32_t in_cb = get_named_compile_time_arg_val("in_cb");
    constexpr uint32_t out_cb = get_named_compile_time_arg_val("out_cb");
    constexpr uint32_t IN_TILES = get_named_compile_time_arg_val("in_tiles");

    cb_wait_front(in_cb, IN_TILES);

    // LN-debug lesson: binary_op_init_common at the top, before any SFPU init.
    binary_op_init_common(in_cb, in_cb, out_cb);
    copy_tile_to_dst_init_short(in_cb);
    gelu_tile_init();

    DPRINT << "TR: ready. IN_TILES=" << IN_TILES << ENDL();

    cb_reserve_back(out_cb, IN_TILES);
    for (uint32_t i = 0; i < IN_TILES; ++i) {
        tile_regs_acquire();
        copy_tile(in_cb, i, 0);  // dst[0] = in[i]
        gelu_tile(0);            // dst[0] = GELU(dst[0])
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, out_cb, i);
        tile_regs_release();
    }
    cb_push_back(out_cb, IN_TILES);
    cb_pop_front(in_cb, IN_TILES);
    DPRINT << "TR: GELU done." << ENDL();
#endif
}
