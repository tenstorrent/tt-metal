// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr uint32_t tile_count = get_compile_time_arg_val(1);
    constexpr uint32_t pack_width = get_compile_time_arg_val(2);
    constexpr bool bfp_output = get_compile_time_arg_val(3);
    static_assert(batch == (DST_ACCUM_MODE ? 4 : 8));
    static_assert(pack_width == 1 || pack_width == 4);
    static_assert(batch % pack_width == 0);
    static_assert(
        !bfp_output || pack_width == 1,
        "Multi-tile default pack MOP does not frame separate BFP exponent sections; use width 1");
    compute_kernel_hw_startup(0, 16);
    copy_init(0);
    // FP32-DST datacopy uses FPU ELWADD; its synthesized SrcB zero must be
    // numeric zero, not a stale source value. This is also harmless for BF16.
    MATH((ckernel::math::_configure_src_zero_flag_(false)));
    if constexpr (pack_width == 4) {
        // The standard pack API exposes no width parameter. This one LLK
        // initialization programs a four-tile MOP; all copies/packs below use
        // standard APIs. No attention headers, SFPU, or format changes.
        // Headerless outputs ONLY: the default LLK MOP closes once after
        // num_faces * pack_width faces, while BFP exp_section_size remains
        // num_faces. It cannot emit four separately framed BFP tiles here.
        PACK((llk_pack_init<PackMode::Default>(16, pack_width)));
    }
    {
        for (uint32_t tile = 0; tile < tile_count; tile += batch) {
            cb_wait_front(0, batch);
            cb_reserve_back(16, batch);
            tile_regs_acquire();
            for (uint32_t j = 0; j < batch; ++j) {
                copy_tile(0, j, j);
            }
            tile_regs_commit();
            cb_pop_front(0, batch);
            tile_regs_wait();
            for (uint32_t j = 0; j < batch; j += pack_width) {
                // Explicit output positions work for both scalar and blocked
                // MOPs; no assumption about sequential pack-pointer advances.
                pack_tile<true>(j, 16, j);
            }
            tile_regs_release();
            cb_push_back(16, batch);
        }
    }
}
