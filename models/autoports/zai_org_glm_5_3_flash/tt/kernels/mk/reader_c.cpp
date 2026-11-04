// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Megakernel reader core — compute, every matmul phase in order (the ds_mm compute per phase): matmul_block
// subblocks of SUBW tiles over the K blocks, packed onto fp32 partials with packer L1 accumulation; the partial CB is
// shared by the phases (sized for the largest NR) and REWOUND after every block so every block accumulates onto the
// same tiles; after the last block the partials are unpacked exactly (fp32) and packed to the output dtype.
// fp32 DST; math fidelity per phase = the DSLinear's (FID, a MathFidelity value: HiFi4 for the low-rank projection,
// HiFi2 for the others), a PhaseOpts knob.
// Compile-time args: NPH, CB_IN0, CB_PART, CB_OUT, then per phase p 7 values at 4 + 7p: NB, KB, NR, SUBW, CB_W, FID,
// OUT_CB (the phase's output CB: bf16 CB_OUT or an fp32 one).
// Runtime args: per phase p: 1 = this reader computes it (0: it sits the phase out, mk_readers.py RP).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#if defined(MK_SCALE_MASK)
#include "api/compute/sfpu_binary_bcast.h"
#endif
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/stack.hpp"

namespace {

constexpr uint32_t NPH = get_compile_time_arg_val(PH_CT0 + 0);
constexpr uint32_t CB_IN0 = get_compile_time_arg_val(PH_CT0 + 1);
constexpr uint32_t CB_PART = get_compile_time_arg_val(PH_CT0 + 2);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(PH_CT0 + 3);

// matmul_block_init / matmul_block (api/compute/matmul.h: same UNPACK / MATH calls, same BH dynamic throttling) with
// the phase's math fidelity F instead of the kernel-wide MATH_FIDELITY
template <MathFidelity F>
ALWI void mm_init(uint32_t in0, uint32_t in1, uint32_t ct, uint32_t rt, uint32_t kt) {
    UNPACK((llk_unpack_AB_matmul_init(in0, in1, 0, ct, rt, kt)));
    MATH((llk_math_matmul_init<F, MM_THROTTLE>(in0, in1, 0, ct, rt)));
    MATH((throttled_mop_status = 0));
}
template <MathFidelity F>
ALWI void mm_block(uint32_t in0, uint32_t in1, uint32_t i0, uint32_t i1, uint32_t ct, uint32_t rt, uint32_t kt) {
    UNPACK((llk_unpack_AB_matmul(in0, in1, i0, i1, ct, rt, kt)));
#ifdef TRISC_MATH
    if (*(throttle_ptr) % 2) {
        if (throttled_mop_status != 1) {
            llk_math_matmul_init<F, MM_THROTTLE_MAX>(in0, in1, 0, ct, rt);
            throttled_mop_status = 1;
        }
        llk_math_matmul<F, MM_THROTTLE_MAX>(0, ct, rt);
    } else {
        if (throttled_mop_status != 0) {
            llk_math_matmul_init<F, MM_THROTTLE>(in0, in1, 0, ct, rt);
            throttled_mop_status = 0;
        }
        llk_math_matmul<F, MM_THROTTLE>(0, ct, rt);
    }
#endif
}

template <uint32_t P>
FORCE_INLINE void mm_phase() {
    constexpr uint32_t B = 4 + 7 * P;
    constexpr uint32_t NB = get_compile_time_arg_val(PH_CT0 + B + 0);
    constexpr uint32_t KB = get_compile_time_arg_val(PH_CT0 + B + 1);
    constexpr uint32_t NR = get_compile_time_arg_val(PH_CT0 + B + 2);
    constexpr uint32_t SUBW = get_compile_time_arg_val(PH_CT0 + B + 3);
    constexpr uint32_t CB_W = get_compile_time_arg_val(PH_CT0 + B + 4);
    constexpr auto FID = static_cast<MathFidelity>(get_compile_time_arg_val(PH_CT0 + B + 5));
    static_assert(NR % SUBW == 0, "subblocks");
    if (get_arg_val<uint32_t>(ph_rt0 + P) == 0) {
#if defined(MK_SCALE_MASK)
        if constexpr (((MK_SCALE_LAST) >> P) & 1) {  // sitting the group's last phase out: still consume the 1/rms(h)
            cb_wait_front(MK_INVH_CB, 1);
            cb_pop_front(MK_INVH_CB, 1);
        }
#endif
        return;
    }

    reconfig_data_format<SrcOrder::Reverse>(CB_IN0, CB_W);
    pack_reconfig_data_format(CB_PART);
    mm_init<FID>(CB_IN0, CB_W, SUBW, 1, KB);
    mk::cb_rewind(CB_PART);
    for (uint32_t blk = 0; blk < NB; ++blk) {
        cb_wait_front(CB_IN0, KB);
        cb_wait_front(CB_W, KB * NR);
        for (uint32_t sb = 0; sb < NR / SUBW; ++sb) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < KB; ++k) {
                mm_block<FID>(CB_IN0, CB_W, k, k * NR + sb * SUBW, SUBW, 1, KB);
            }
            tile_regs_commit();
            cb_reserve_back(CB_PART, SUBW);
            tile_regs_wait();
            PACK((llk_pack_reconfig_l1_acc(blk > 0 ? 1 : 0)));
            for (uint32_t i = 0; i < SUBW; ++i) {
                pack_tile(i, CB_PART);
            }
            tile_regs_release();
            cb_push_back(CB_PART, SUBW);
        }
        cb_pop_front(CB_IN0, KB);
        cb_pop_front(CB_W, KB * NR);
        if (blk + 1 < NB) {
            cb_wait_front(CB_PART, NR);
            cb_pop_front(CB_PART, NR);
            mk::cb_rewind(CB_PART);  // the next block accumulates onto the same tiles
        }
    }
    PACK((llk_pack_reconfig_l1_acc(0)));
    cb_wait_front(CB_PART, NR);
    constexpr uint32_t OUT_CB = get_compile_time_arg_val(PH_CT0 + B + 6);
    cb_reserve_back(OUT_CB, NR);
    reconfig_data_format_srca(CB_PART);
    pack_reconfig_data_format(OUT_CB);
#if defined(MK_SCALE_MASK)
    constexpr bool SCALE = ((MK_SCALE_MASK) >> P) & 1;
    if constexpr (SCALE) {  // Stage 14: deferred 1/rms(h) of the input rows (exact fp32 SFPU, column broadcast)
        cb_wait_front(MK_INVH_CB, 1);
    }
#else
    constexpr bool SCALE = false;
#endif
    copy_tile_to_dst_init_short(CB_PART);
    for (uint32_t j = 0; j < NR; ++j) {
        tile_regs_acquire();
        copy_tile(CB_PART, j, 0);
#if defined(MK_SCALE_MASK)
        if constexpr (SCALE) {
            copy_tile_to_dst_init_short(MK_INVH_CB);
            copy_tile(MK_INVH_CB, 0, 1);
            sfpu_mul_bcast_col_init();
            sfpu_mul_bcast_col(0, 1);
            copy_tile_to_dst_init_short(CB_PART);
        }
#endif
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, OUT_CB);
        tile_regs_release();
    }
#if defined(MK_SCALE_MASK)
    if constexpr (((MK_SCALE_LAST) >> P) & 1) {
        cb_pop_front(MK_INVH_CB, 1);
    }
#endif
    cb_push_back(OUT_CB, NR);
    cb_pop_front(CB_PART, NR);
    mk::cb_rewind(CB_PART);
    mk::cb_rewind(OUT_CB);  // pack side back to the base for the next phase (the mover rewinds its read side)
}

template <uint32_t P>
FORCE_INLINE void run_c() {
#if defined(MK_SPLIT)
    if constexpr (P == MK_SPLIT) {
        MK_BETWEEN();
    }
#endif
    if constexpr (P < NPH) {
        mm_phase<P>();
        run_c<P + 1>();
    }
}

}  // namespace

void kernel_main() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(
        CB_IN0, get_compile_time_arg_val(PH_CT0 + 8), CB_PART);  // phase 0's CB_W
    for (uint32_t layer = 0; layer < MK_LAYERS; ++layer) {       // stack program (kernels/mk/stack.hpp)
        run_c<0>();
    }
}
