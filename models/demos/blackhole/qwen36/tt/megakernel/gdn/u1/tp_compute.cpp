// GDN_STEP_U1 op-layout prefix compute: kind 0 = gdn_spec_step phase A3 + A4 on one chunk (DST_j = SH_j * tap row j,
// DST_0 += DST_1..3, silu -> cv; P @ cv one-hot placement, same operand roles); kind 1 = silu_tiles on one z tile.
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "tp_common.hpp"

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    const uint32_t kind = get_arg_val<uint32_t>(0);
    DeviceZoneScopedN("TP-C");
    if (kind == 1) {
        compute_kernel_hw_startup(TP_Z, TP_Z, TP_OUT);
        cb_wait_front(TP_Z, 1);
        cb_reserve_back(TP_OUT, 1);
        pack_reconfig_data_format(TP_OUT);
        reconfig_data_format_srca(TP_Z);
        copy_init(TP_Z);
        silu_tile_init();
        tile_regs_acquire();
        copy_tile(TP_Z, 0, 0);
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, TP_OUT, 0);
        tile_regs_release();
        cb_push_back(TP_OUT, 1);
        cb_pop_front(TP_Z, 1);
        return;
    }
    compute_kernel_hw_startup(TP_SH, TP_TAP, TP_CV);
    cb_wait_front(TP_SH, K);
    cb_wait_front(TP_TAP, 1);
    cb_reserve_back(TP_CV, 1);
    pack_reconfig_data_format(TP_CV);
    reconfig_data_format(TP_SH, TP_TAP);
    mul_bcast_rows_init(TP_SH, TP_TAP);
    tile_regs_acquire();
    for (uint32_t j = 0; j < K; ++j) mul_tiles_bcast_rows(TP_SH, TP_TAP, j, 0, j, j);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    add_binary_tile(0, 2, 0);
    add_binary_tile(0, 3, 0);
    silu_tile_init();
    silu_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, TP_CV, 0);
    tile_regs_release();
    cb_push_back(TP_CV, 1);
    cb_pop_front(TP_SH, K);
    cb_pop_front(TP_TAP, 1);
    cb_wait_front(TP_CV, 1);
    cb_wait_front(TP_P, 1);
    cb_reserve_back(TP_OUT, 1);
    pack_reconfig_data_format(TP_OUT);
    reconfig_data_format<SrcOrder::Reverse>(TP_P, TP_CV);
    matmul_init(TP_P, TP_CV);
    tile_regs_acquire();
    matmul_tiles(TP_P, TP_CV, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, TP_OUT, 0);
    tile_regs_release();
    cb_push_back(TP_OUT, 1);
    cb_pop_front(TP_CV, 1);
    cb_pop_front(TP_P, 1);
    cb_wait_front(TP_ST, 2);
    cb_pop_front(TP_ST, 2);
}
