// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#if defined(READER)
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto accessor_args = TensorAccessorArgs<0>();
    const auto packed = TensorAccessor(accessor_args, get_arg_val<uint32_t>(0), 2048);
    const uint32_t first_tile = get_arg_val<uint32_t>(1);
    for (uint32_t tile = 0; tile < 7; ++tile) {
        cb_reserve_back(0, 1);
        cb_reserve_back(1, 1);
        noc_async_read_page(first_tile + tile, packed, get_write_ptr(0));
        noc_async_read_page(112 + first_tile + tile, packed, get_write_ptr(1));
        noc_async_read_barrier();
        cb_push_back(0, 1);
        cb_push_back(1, 1);
    }
}
#elif defined(COMPUTE)
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
using namespace ckernel;

void kernel_main() {
    compute_kernel_hw_startup(0, 16);
    for (uint32_t tile = 0; tile < 7; ++tile) {
        // Match binary_ng's operand-activation path: BF16 pack and reload
        // between SiLU and multiply, even though both run in this program.
        cb_wait_front(0, 1);
        cb_reserve_back(2, 1);
        copy_init(0);
        tile_regs_acquire();
        copy_tile(0, 0, 0);
        silu_tile_init();
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, 2);
        tile_regs_release();
        cb_pop_front(0, 1);
        cb_push_back(2, 1);

        cb_wait_front(2, 1);
        cb_wait_front(1, 1);
        cb_reserve_back(16, 1);
        tile_regs_acquire();
        copy_init(2);
        copy_tile(2, 0, 0);
        copy_init(1);
        copy_tile(1, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, 16);
        tile_regs_release();
        cb_pop_front(2, 1);
        cb_pop_front(1, 1);
        cb_push_back(16, 1);
    }
}
#endif
