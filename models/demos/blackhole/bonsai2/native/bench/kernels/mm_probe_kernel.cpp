// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Probe: one 32x32 bf16 matmul tile (cb 0 @ cb 1 -> fp32 cb 16), optionally after deepseek_compute_kernel_init.
#include "../../../../../deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#endif
#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/experimental/deepseek_compute_kernel_hw_startup.h"
#include "api/compute/experimental/pack_block.h"
#endif

void kernel_main() {
    constexpr uint32_t mode = get_named_compile_time_arg_val("pr_mode");  // bit 0: deepseek init; bit 1: mm_block_init-style hw startup
#if defined(COMPILE_FOR_NCRISC)
    const InterleavedAddrGen<true> a{.bank_base_address = get_arg_val<uint32_t>(0), .page_size = 2048};
    const InterleavedAddrGen<true> b{.bank_base_address = get_arg_val<uint32_t>(1), .page_size = 2048};
    cb_reserve_back(0, 1);
    cb_reserve_back(1, 1);
    noc_async_read(get_noc_addr(0, a, 0), get_write_ptr(0), 2048);
    noc_async_read(get_noc_addr(0, b, 0), get_write_ptr(1), 2048);
    noc_async_read_barrier();
    cb_push_back(0, 1);
    cb_push_back(1, 1);
#elif defined(COMPILE_FOR_BRISC)
    const InterleavedAddrGen<true> y{.bank_base_address = get_arg_val<uint32_t>(0), .page_size = 4096};
    cb_wait_front(16, 1);
    noc_async_write(get_read_ptr(16), get_noc_addr(0, y, 0), 4096);
    noc_async_write_barrier();
    cb_pop_front(16, 1);
#elif defined(COMPILE_FOR_TRISC)
    if constexpr (mode & 1) {
        deepseek_compute_kernel_init();
    } else {
        compute_kernel_hw_startup(0, 1, 16);
    }
    matmul_init(0, 1);
    cb_wait_front(0, 1);
    cb_wait_front(1, 1);
    tile_regs_acquire();
    matmul_tiles(0, 1, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(16, 1);
    if constexpr (mode & 2) {
        pack_reconfig_data_format(16);
        pack_block_contiguous_init(16);
        pack_block_contiguous(0, 16, 1);
    } else {
        pack_reconfig_data_format(16);
        pack_tile(0, 16);
    }
    cb_push_back(16, 1);
    tile_regs_release();
#endif
}
