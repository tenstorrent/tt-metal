// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/compute/compute_kernel_hw_startup.h"
#define EXP_APPROX_MODE 1
#include "candidate/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"
#include "hybrid.hpp"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
namespace ckernel::sfpu {
template <int repeats>
inline void hybrid_probe_update() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_6);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_7);
    for (int i = 0; i < 32; ++i) {
        TTI_SFPLOAD(0, InstrModLoadStore::FP32, ADDR_MOD_6, 0);
        TTI_SFPLOAD(1, InstrModLoadStore::FP32, ADDR_MOD_6, 64);
        TTI_SFPLOAD(2, InstrModLoadStore::FP32, ADDR_MOD_6, 128);
        TTI_SFPNOP;
        for (int j = 0; j < repeats; ++j) {
            TTI_SFPMAD(0, 2, 1, 0, 0);
            TTI_SFPNOP;
            TTI_SFPNOP;
        }
        TTI_SFPSTORE(0, InstrModLoadStore::FP32, ADDR_MOD_7, 0);
    }
}
}  // namespace ckernel::sfpu
#endif
void kernel_main() {
    constexpr int repeats = get_compile_time_arg_val(0);
    compute_kernel_hw_startup(1, 1, 16);
    cb_wait_front(0, 3);
    cb_reserve_back(16, 1);
    enable_fp32_dest_acc();
    tile_regs_acquire();
    hybrid_load(0, 0, 0);
    if constexpr (repeats != 0) {
        hybrid_load(0, 1, 1);
        hybrid_load(0, 2, 2);
    }
    tile_regs_commit<true>();
    tile_regs_wait();
    PACK((llk_math_sfpu_init_once()));
    if constexpr (repeats != 0) {
        PACK((SFPU_UNARY_CALL(DST_SYNC_MODE, true, hybrid_probe_update, (repeats), 0, VectorMode::None)));
    }
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    hybrid_pack_config(16);
    PACK((llk_pack_reconfig_l1_acc(0)));
    hybrid_pack(0, 16);
    tile_regs_release<true>();
    disable_fp32_dest_acc();
    cb_push_back(16, 1);
}
