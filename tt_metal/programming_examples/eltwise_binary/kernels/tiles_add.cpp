// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api.h"

#include "universal_common.h"

KERNEL_MAIN {
    uint32_t in0_addr = get_arg_val<uint32_t>(0);
    uint32_t in1_addr = get_arg_val<uint32_t>(1);
    uint32_t out_addr = get_arg_val<uint32_t>(2);
    uint32_t n_tiles = get_arg_val<uint32_t>(3);

    constexpr uint32_t tile_size_bytes = get_compile_time_arg_val(0);
    constexpr auto in0_args = TensorAccessorArgs<1>();
    const auto in0 = TensorAccessor(in0_args, in0_addr, tile_size_bytes);
    constexpr auto in1_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto in1 = TensorAccessor(in1_args, in1_addr, tile_size_bytes);
    constexpr auto out_args = TensorAccessorArgs<in1_args.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(out_args, out_addr, tile_size_bytes);

    binary_op_init_common(cb_in, cb_in, cb_out);
    add_tiles_init(cb_in, cb_in);

    for (uint32_t i = 0; i < n_tiles; i++) {
        read_tile(i, in0, tile_size_bytes);
        read_tile(i, in1, tile_size_bytes);

        tile_regs_acquire();
        add_tiles(cb_in, cb_in, 0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();

        write_packed_tile(0, i, out, tile_size_bytes);

        release_write_tiles(1);
        release_read_tiles(2);
        tile_regs_release();
    }
}
