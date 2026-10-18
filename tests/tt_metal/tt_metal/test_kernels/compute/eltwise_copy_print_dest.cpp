// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/debug/dprint.h"
#include "api/debug/dprint_tensix.h"
#ifdef ARCH_QUASAR
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#endif

void kernel_main() {
#ifdef ARCH_QUASAR
    // Quasar declares the operands as dataflow buffers and passes compile args by name.
    constexpr uint32_t per_core_tile_cnt = get_arg(args::per_core_tile_cnt);
    constexpr uint32_t in_id = dfb::in;
    constexpr uint32_t out_id = dfb::out;
    DataflowBuffer in_buf(in_id);
    DataflowBuffer out_buf(out_id);
#else
    uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);
    bool remap = get_compile_time_arg_val(1) != 0;
    bool swizzle = get_compile_time_arg_val(2) != 0;
    constexpr uint32_t in_id = tt::CBIndex::c_0;
    constexpr uint32_t out_id = tt::CBIndex::c_16;
#endif

    compute_kernel_hw_startup(in_id, out_id);
    copy_init(in_id);
#ifdef ARCH_BLACKHOLE
    cfg_reg_rmw_tensix<DEST_ACCESS_CFG_remap_addrs_RMW>(remap);
    cfg_reg_rmw_tensix<DEST_ACCESS_CFG_swizzle_32b_RMW>(swizzle);
#endif
    tile_regs_acquire();
#ifdef ARCH_QUASAR
    in_buf.wait_front(per_core_tile_cnt);
    out_buf.reserve_back(per_core_tile_cnt);
#else
    cb_wait_front(in_id, per_core_tile_cnt);
    cb_reserve_back(out_id, per_core_tile_cnt);
#endif

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        copy_tile(in_id, b, b);
#ifdef ARCH_QUASAR
        // Quasar's config-read wrappers are unwired, so the host tells the kernel the dest format.
        dprint_tensix_dest_reg(static_cast<DataFormat>(get_arg(args::dest_data_format)), b);
#else
        dprint_tensix_dest_reg(b);
#endif
    }

    tile_regs_commit();
    tile_regs_wait();

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        pack_tile(b, out_id);
#ifdef ARCH_QUASAR
        in_buf.pop_front(1);
        out_buf.push_back(1);
#else
        cb_pop_front(in_id, 1);
        cb_push_back(out_id, 1);
#endif
    }

    tile_regs_release();
}
