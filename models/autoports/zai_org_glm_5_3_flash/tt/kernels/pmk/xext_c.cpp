// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// W7 step 5, D3 (P11x; coordinator 2026-10-25 18:48:11Z, design pmk/w7/step5_gang_design.md §8): the member's x-hat_f
// block -- XS, the AG's landing CB: KT bf16 tiles of this member's 32-row tile row -- untilized into ROW_MAJOR row
// segments for x_ext [C + 32, H] (the EP gang's DRAM gather input). pfs_member_c.cpp's EPI 2 path applied to a copy of
// XS: BW tiles per DST block (half-sync fp32 DEST holds 4), each block packed as BW tile-sized pages = 32 rows x BW
// tiles. XS is the stationary in0 of every phase: the member DM1 part re-pushes it after the phase's arrival wait, this
// part pops it. CT at PH_CT0: [cb_xs, cb_rm, kt, bw]. RT at ph_rt0: [run (0 = skip)].
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace xec {
constexpr uint32_t CB_XS = get_compile_time_arg_val(PH_CT0 + 0);
constexpr uint32_t CB_RM = get_compile_time_arg_val(PH_CT0 + 1);
constexpr uint32_t KT = get_compile_time_arg_val(PH_CT0 + 2);
constexpr uint32_t BW = get_compile_time_arg_val(PH_CT0 + 3);
static_assert(KT % BW == 0 && BW >= 1 && BW <= 4, "x_ext untilize: BW tiles per DST block (fp32 DEST: <= 4)");
}  // namespace xec

void kernel_main() {
    using namespace xec;
    if (get_arg_val<uint32_t>(ph_rt0) == 0) {
        return;
    }
#if !defined(MK_PHASED)
    compute_kernel_hw_startup(CB_XS, CB_RM);
#else
    reconfig_data_format_srca(CB_XS);  // the previous part left the unpackers / packer on its own CBs
    pack_reconfig_data_format(CB_RM);
#endif
    copy_init(CB_XS);
    pack_untilize_dest_init<BW, BW>(CB_RM);
    cb_wait_front(CB_XS, KT);
    for (uint32_t k = 0; k < KT; k += BW) {
        tile_regs_acquire();
        for (uint32_t j = 0; j < BW; ++j) {
            copy_tile(CB_XS, k + j, j);  // bf16 -> DEST (fp32 or bf16 per the composite): exact
        }
        tile_regs_commit();
        cb_reserve_back(CB_RM, BW);
        tile_regs_wait();
        pack_untilize_dest<BW, BW>(CB_RM, 1, 0);  // 32 rows x BW tiles ROW_MAJOR in BW tile-sized pages
        tile_regs_release();
        cb_push_back(CB_RM, BW);
    }
    pack_untilize_uninit(CB_RM);
    cb_pop_front(CB_XS, KT);
}
