// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// One unified generic-op kernel for packed [gate | up] SwiGLU.
// NCRISC reads paired tiles, TRISC computes with a BF16 activation boundary,
// and BRISC writes the half-width result.

#if defined(COMPILE_FOR_TRISC)
#include <cstdint>

#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/circular_buffer.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_silu.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif
#else
#include "api/dataflow/dataflow_api.h"
#endif

#if defined(COMPILE_FOR_TRISC)
namespace ckernel {

// SiLU still lives in the legacy umbrella API.  Define its two small public
// wrappers locally so this unified kernel can use the split compute APIs
// without including compute_kernel_api.h (which recursively includes the
// unified kernel through chlkc_list.h).
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void packed_silu_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_silu, (is_fp32_dest_acc_en, 8), idst, VectorMode::RC));
}

ALWI void packed_silu_tile_init() { MATH(SFPU_UNARY_INIT_FN(silu, sfpu::silu_init, (APPROX))); }

}  // namespace ckernel
#endif

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t half_width_tiles = get_named_compile_time_arg_val("half_width_tiles");
    constexpr auto accessor_args = TensorAccessorArgs<0>();
    const auto input = TensorAccessor(accessor_args, get_common_arg_val<uint32_t>(0));
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    constexpr uint32_t io_batch = 8;
    constexpr uint32_t tile_bytes = get_tile_size(tt::CBIndex::c_0);

    for (uint32_t offset = 0; offset < count; offset += io_batch) {
        const uint32_t remaining = count - offset;
        const uint32_t n = remaining < io_batch ? remaining : io_batch;
        cb_reserve_back(tt::CBIndex::c_0, n);
        cb_reserve_back(tt::CBIndex::c_1, n);
        const uint32_t gate_l1 = get_write_ptr(tt::CBIndex::c_0);
        const uint32_t up_l1 = get_write_ptr(tt::CBIndex::c_1);
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t output_tile = start + offset + i;
            const uint32_t row = output_tile / half_width_tiles;
            const uint32_t column = output_tile % half_width_tiles;
            const uint32_t gate_tile = row * (2 * half_width_tiles) + column;
            const uint32_t up_tile = gate_tile + half_width_tiles;
            noc_async_read_page(gate_tile, input, gate_l1 + i * tile_bytes);
            noc_async_read_page(up_tile, input, up_l1 + i * tile_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(tt::CBIndex::c_0, n);
        cb_push_back(tt::CBIndex::c_1, n);
    }

#elif defined(COMPILE_FOR_BRISC)
    constexpr auto accessor_args = TensorAccessorArgs<0>();
    const auto output = TensorAccessor(accessor_args, get_common_arg_val<uint32_t>(0));
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    constexpr uint32_t io_batch = 8;
    constexpr uint32_t tile_bytes = get_tile_size(tt::CBIndex::c_2);

    for (uint32_t offset = 0; offset < count; offset += io_batch) {
        const uint32_t remaining = count - offset;
        const uint32_t n = remaining < io_batch ? remaining : io_batch;
        cb_wait_front(tt::CBIndex::c_2, n);
        const uint32_t output_l1 = get_read_ptr(tt::CBIndex::c_2);
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_write_page(start + offset + i, output, output_l1 + i * tile_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(tt::CBIndex::c_2, n);
    }

#elif defined(COMPILE_FOR_TRISC)
    constexpr uint32_t gate_cb = tt::CBIndex::c_0;
    constexpr uint32_t up_cb = tt::CBIndex::c_1;
    constexpr uint32_t output_cb = tt::CBIndex::c_2;
    constexpr uint32_t activated_cb = tt::CBIndex::c_3;
    // SFPU binary ops are currently reliable for at most two tiles per cycle
    // (the binary_ng factory enforces the same limit).
    constexpr uint32_t tiles_per_cycle = 2;
    const uint32_t count = get_arg_val<uint32_t>(0);

    // ttnn.multiply(fast_and_approximate_mode=False) uses binary_ng's SFPU
    // multiply path.  Match that path, including its BF16 activation CB.
    compute_kernel_hw_startup(activated_cb, output_cb);
    copy_init(activated_cb);
    for (uint32_t offset = 0; offset < count; offset += tiles_per_cycle) {
        const uint32_t remaining = count - offset;
        const uint32_t n = remaining < tiles_per_cycle ? remaining : tiles_per_cycle;
        // binary_ng implements operand activations as a separate copy/pack
        // pass.  Keeping that BF16 pack here is required for exact model state.
        cb_wait_front(gate_cb, n);
        cb_reserve_back(activated_cb, n);
        pack_reconfig_data_format(output_cb, activated_cb);
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            copy_init(gate_cb);
            copy_tile(gate_cb, i, i);
            packed_silu_tile_init();
            packed_silu_tile(i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, activated_cb);
        }
        tile_regs_release();
        cb_pop_front(gate_cb, n);
        cb_push_back(activated_cb, n);
        pack_reconfig_data_format(activated_cb, output_cb);

        cb_wait_front(activated_cb, n);
        cb_wait_front(up_cb, n);
        cb_reserve_back(output_cb, n);
        mul_binary_tile_init();
        tile_regs_acquire();
        reconfig_data_format_srca(up_cb, activated_cb);
        copy_init(activated_cb);
        for (uint32_t i = 0; i < n; ++i) {
            copy_tile(activated_cb, i, i * 2);
        }
        reconfig_data_format_srca(activated_cb, up_cb);
        copy_init(up_cb);
        for (uint32_t i = 0; i < n; ++i) {
            copy_tile(up_cb, i, i * 2 + 1);
            mul_binary_tile(i * 2, i * 2 + 1, i * 2);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i * 2, output_cb);
        }
        tile_regs_release();
        cb_pop_front(activated_cb, n);
        cb_pop_front(up_cb, n);
        cb_push_back(output_cb, n);
    }
#endif
}
