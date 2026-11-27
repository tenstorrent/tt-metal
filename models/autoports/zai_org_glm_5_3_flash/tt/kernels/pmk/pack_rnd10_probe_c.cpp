// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// tests/s16/jobs/pack_rnd10_probe.py -- compute: every fp32 tile of CB 0 read EXACTLY (UnpackToDestFp32 on CB 0) and
// packed fp32 to CB 16, under PROBE_MODE:
//   0 = plain pack (the control: identity);
//   1 = the packer gasket's 10-bit mantissa rounding on (PCK_DEST_RD_CTRL_Round_10b_mant: HAL "rounds to 10b mantissa,
//       regardless of fp_pack format"; llk sets it only for fp16 packs) -- the question: is it RNE, half-away or other;
//   2 = the SFPU tf32 RNE in DST (rne_tf32.hpp, GLM53_PMK_KDA_RNE=1's path) -- the reference rounding.
// Runtime: tile count.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#if PROBE_MODE == 2
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/rne_tf32.hpp"
#endif

void kernel_main() {
    constexpr uint32_t CB_IN = 0;
    constexpr uint32_t CB_OUT = 16;
    const uint32_t n = get_arg_val<uint32_t>(0);
    if (n == 0) {
        return;
    }
    compute_kernel_hw_startup(CB_IN, CB_OUT);
    copy_init(CB_IN);
#if PROBE_MODE == 1
    PACK((ckernel::cfg_reg_rmw_tensix<PCK_DEST_RD_CTRL_Round_10b_mant_RMW>(1)));
#elif PROBE_MODE == 2
    kda_rne_tf32_init();
#endif
    for (uint32_t i = 0; i < n; ++i) {
        cb_wait_front(CB_IN, 1);
        cb_reserve_back(CB_OUT, 1);
        tile_regs_acquire();
        copy_tile(CB_IN, 0, 0);
#if PROBE_MODE == 2
        kda_rne_tf32_tile(0);
#endif
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_OUT);
        tile_regs_release();
        cb_pop_front(CB_IN, 1);
        cb_push_back(CB_OUT, 1);
    }
#if PROBE_MODE == 1
    // the last pack done before the gasket mode goes back to the configured value
    PACK((TTI_STALLWAIT(ckernel::p_stall::STALL_CFG, ckernel::p_stall::PACK)));
    PACK((ckernel::cfg_reg_rmw_tensix<PCK_DEST_RD_CTRL_Round_10b_mant_RMW>(0)));
#endif
}
