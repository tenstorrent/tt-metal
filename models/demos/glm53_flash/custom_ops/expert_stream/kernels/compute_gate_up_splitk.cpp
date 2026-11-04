// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Split-K gate|up compute: for every active expert this core multiplies its K slice of x (ks tiles) with its slice of
// the gate and up columns (fp32 accumulation in DST 0 / 1) and packs the two fp32 partial tiles for the expert's
// owner (bank core rank i % G); for the experts it owns it sums the G partials in DST (fp32, SFPU adds), applies
// the clamped SwiGLU and the routing diagonal as compute_gate_up.cpp does, and emits the output tile.
// Runtime args: [group (this core's rank in its bank), ks]. Defines: SPLITK_KS_MAX, SPLITK_CB_PART, SPLITK_CB_RECV.
#include <cstdint>

#include "api/compute/clamped_silu_glu.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "ckernel.h"

// profiler zones (tracy -p builds only; each DeviceZoneScopedN needs its own scope)
#if defined(PROFILE_KERNEL) && defined(DeviceZoneScopedN)
#define EX_ZONE(name) DeviceZoneScopedN(name)
#else
#define EX_ZONE(name)
#endif

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_d = get_compile_time_arg_val(2);
    constexpr uint32_t cb_h = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    constexpr uint32_t groups = get_compile_time_arg_val(7);
    constexpr uint32_t KS_MAX = SPLITK_KS_MAX;
    constexpr uint32_t cb_part = SPLITK_CB_PART;
    constexpr uint32_t cb_recv = SPLITK_CB_RECV;
    const uint32_t group = get_arg_val<uint32_t>(0);
    const uint32_t ks = get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    matmul_init(cb_x, cb_w);
    uint32_t n = 0;
    UNPACK(n = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
    MATH(n = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
    PACK(n = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);) {
        EX_ZONE("GU_C_X");
        cb_wait_front(cb_x, KS_MAX);
    }
    for (uint32_t i = 0; i < n; ++i) {
        EX_ZONE("GU_C_EXPERT");
        // ---- this core's partial products of expert i (gate -> DST 0, up -> DST 1)
        tile_regs_acquire();
        fill_tile_init();
        fill_tile(0, 0.0f);
        fill_tile(1, 0.0f);
        matmul_init(cb_x, cb_w);
        for (uint32_t c = 0; c < 2; ++c) {
            cb_wait_front(cb_w, KS_MAX);
            for (uint32_t k = 0; k < ks; ++k) {
                matmul_tiles(cb_x, cb_w, k, k, c);
            }
            cb_pop_front(cb_w, KS_MAX);
        }
        tile_regs_commit();
        cb_reserve_back(cb_part, 2);
        tile_regs_wait();
        pack_reconfig_data_format(cb_part);
        pack_tile(0, cb_part);
        pack_tile(1, cb_part);
        tile_regs_release();
        cb_push_back(cb_part, 2);
        if (i % groups != group) {
            continue;
        }
        // ---- the owner: sum the G partials (fp32 in DST), SwiGLU, the routing diagonal, the output tile
        {
            EX_ZONE("GU_C_OWNER_WAIT");
            cb_wait_front(cb_d, 1);
            cb_wait_front(cb_recv, 2 * groups);
        }
        EX_ZONE("GU_C_OWNER");
        tile_regs_acquire();
        reconfig_data_format_srca(cb_w, cb_recv);  // fp32 partials (the unpacker held the weight CB's format)
        copy_init(cb_recv);
        copy_tile(cb_recv, 0, 0);
        copy_tile(cb_recv, 1, 1);
        add_binary_tile_init();
        for (uint32_t j = 1; j < groups; ++j) {
            copy_tile(cb_recv, 2 * j, 2);
            copy_tile(cb_recv, 2 * j + 1, 3);
            add_binary_tile(0, 2, 0);
            add_binary_tile(1, 3, 1);
        }
        clamped_silu_glu_tile_init();
        clamped_silu_glu_tile(0, 1, 2);
        tile_regs_commit();
        cb_pop_front(cb_recv, 2 * groups);
        cb_reserve_back(cb_h, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_h);
        pack_tile(2, cb_h);
        tile_regs_release();
        cb_push_back(cb_h, 1);
        // routing weights: out = D_e @ h (BF16 x BF16 operands; srcA = in1, srcB = in0 as in the matmul kernels)
        reconfig_data_format(cb_recv, cb_h, cb_x, cb_d);
        matmul_init(cb_d, cb_h);
        tile_regs_acquire();
        cb_wait_front(cb_h, 1);
        matmul_tiles(cb_d, cb_h, 0, 0, 3);
        cb_pop_front(cb_h, 1);
        tile_regs_commit();
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        pack_tile(3, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        reconfig_data_format(cb_h, cb_w, cb_d, cb_x);
        cb_pop_front(cb_d, 1);
    }
    cb_pop_front(cb_x, KS_MAX);
}
