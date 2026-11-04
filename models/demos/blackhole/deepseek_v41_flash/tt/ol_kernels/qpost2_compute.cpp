// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// otherlayers: compute of the indexer-q RoPE program, variant 2 (see qpost2_reader.cpp): the UNF formula of
// qpost_compute.cpp with the cos / sin operands broadcast from row 0 by the unpacker:
//   p = x @ P (exact), m = x * bcast_rows(c) (SFPU mul, bf16 DEST), out = addcmul(m, p, bcast_rows(s)).
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "ol_zones.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_r = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t cb_r2 = get_compile_time_arg_val(4);
    constexpr uint32_t cb_ri = get_compile_time_arg_val(5);
    constexpr uint32_t cb_p = get_compile_time_arg_val(7);
    OLZ("Z_CALL");
    compute_kernel_hw_startup(cb_x, cb_r, cb_out);
    CircularBuffer x(cb_x), r(cb_r), out(cb_out), r2(cb_r2), sfull(cb_ri), p(cb_p);
    {
        OLZ("Z_CWAIT");
        x.wait_front(1);
        r.wait_front(1);
        r2.wait_front(1);
        sfull.wait_front(1);
    }
    {
        OLZ("Z_MM");
        reconfig_data_format(cb_x, cb_r2);
        matmul_init(cb_x, cb_r2);
        tile_regs_acquire();
        matmul_tiles(cb_x, cb_r2, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        p.reserve_back(1);
        pack_tile(0, cb_p);
        p.push_back(1);
        tile_regs_release();
        p.wait_front(1);
    }
    {
        OLZ("Z_SFPU");
        tile_regs_acquire();
        reconfig_data_format_srca(cb_r2, cb_x);
        copy_init(cb_x);
        copy_tile(cb_x, 0, 0);
        reconfig_data_format_srca(cb_x, cb_r);
        unary_bcast_init<BroadcastType::ROW>(cb_r);
        unary_bcast<BroadcastType::ROW>(cb_r, 0, 1);  // c replicated over the rows
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);  // m = x * c (SFPU, rounded to bf16 in DEST like the stock multiply)
        reconfig_data_format_srca(cb_r, cb_p);
        copy_init(cb_p);
        copy_tile(cb_p, 0, 1);
        reconfig_data_format_srca(cb_p, cb_ri);
        unary_bcast_init<BroadcastType::ROW>(cb_ri);
        unary_bcast<BroadcastType::ROW>(cb_ri, 0, 2);  // s replicated over the rows
        addcmul_tile_init();
        addcmul_tile<DataFormat::Float16_b>(0, 1, 2, 0, 0x3F800000u);
        tile_regs_commit();
        tile_regs_wait();
        out.reserve_back(1);
        pack_tile(0, cb_out);
        out.push_back(1);
        tile_regs_release();
    }
}
