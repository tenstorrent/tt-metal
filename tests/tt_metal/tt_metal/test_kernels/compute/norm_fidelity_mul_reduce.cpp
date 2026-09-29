// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/experimental/add_rsqrt.h"
#include "api/compute/experimental/mul_reduce_scalar.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr std::uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr bool explicit_fidelity = get_compile_time_arg_val(1) != 0;
    constexpr auto mul_fidelity = static_cast<MathFidelity>(get_compile_time_arg_val(2));
    constexpr auto reduce_fidelity = static_cast<MathFidelity>(get_compile_time_arg_val(3));
    constexpr bool accumulate_in_one_tile = get_compile_time_arg_val(4) != 0;
    constexpr bool apply_rsqrt = get_compile_time_arg_val(5) != 0;
    constexpr std::uint32_t rsqrt_input_scale = get_compile_time_arg_val(6);
    constexpr std::uint32_t rsqrt_epsilon = get_compile_time_arg_val(7);
    CircularBuffer input_a(tt::CBIndex::c_0);
    CircularBuffer input_b(tt::CBIndex::c_1);
    CircularBuffer output(tt::CBIndex::c_16);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);
    input_a.wait_front(num_tiles);
    input_b.wait_front(num_tiles);
    output.reserve_back(1);

    if constexpr (explicit_fidelity) {
        mul_reduce_scalar_init_fidelity<mul_fidelity>(tt::CBIndex::c_0, tt::CBIndex::c_1);
    } else {
        mul_reduce_scalar_init(tt::CBIndex::c_0, tt::CBIndex::c_1);
    }
    if constexpr (apply_rsqrt) {
        add_rsqrt_tile_init();
    }
    tile_regs_acquire();
    if constexpr (explicit_fidelity) {
        mul_reduce_scalar_tile_fidelity<mul_fidelity, reduce_fidelity, accumulate_in_one_tile>(
            tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16, num_tiles);
    } else {
        mul_reduce_scalar_tile<PoolType::SUM, DST_ACCUM_MODE, accumulate_in_one_tile>(
            tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16, num_tiles);
    }
    mul_reduce_scalar_uninit();
    if constexpr (apply_rsqrt) {
        add_rsqrt_tile<false, VectorMode::RC_custom, 1, false, rsqrt_input_scale>(0, rsqrt_epsilon);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, tt::CBIndex::c_16);
    tile_regs_release();

    input_a.pop_front(num_tiles);
    input_b.pop_front(num_tiles);
    output.push_back(1);
}
