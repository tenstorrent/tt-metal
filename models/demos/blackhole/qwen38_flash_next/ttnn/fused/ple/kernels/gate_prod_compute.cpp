// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// PLE gate split, products program compute (32-bit dest): per tile exactly gate_compute.cpp's step 1 -- bf16 key and
// query tiles unpacked into the fp32 dest (exact), mul_binary_tile, packed fp32.  CBs: 0 key, 1 query (bf16, N),
// 2 products (fp32, N).  Compile-time args: 0 N.
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reg_api.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/dataflow_buffer.h"
#include "../../kernels/zones.h"

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t c_key = 0, c_query = 1, c_prod = 2;
    compute_kernel_hw_startup(c_key, c_query, c_prod);
    DataflowBuffer key(c_key), query(c_query), prod(c_prod);
    FUSED_ZONE("fz_pl_gp_c");
    key.wait_front(N);
    query.wait_front(N);
    reconfig_data_format(c_key, c_query);
    pack_reconfig_data_format(c_prod);
    copy_init(c_key);
    mul_binary_tile_init();
    for (uint32_t t = 0; t < N; ++t) {
        tile_regs_acquire();
        reconfig_data_format_srca(c_query, c_key);
        copy_init(c_key);
        copy_tile(c_key, t, 0);
        reconfig_data_format_srca(c_key, c_query);
        copy_init(c_query);
        copy_tile(c_query, t, 1);
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        prod.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, c_prod);
        tile_regs_release();
        prod.push_back(1);
    }
    key.pop_front(N);
    query.pop_front(N);
}
