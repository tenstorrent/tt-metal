// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MLA megakernel reader core — compute, every reader phase in order (kernels/mla/mla_common.hpp for the phase model):
// per sub-matmul j, matmul_block subblocks of SUBW tiles over its K blocks, packed onto fp32 partials with packer L1
// accumulation (kernels/mk/reader_c.cpp's arithmetic: CB_PART rewound between blocks, fp32 DST, the phase's math
// fidelity), then the partials unpacked exactly (fp32) and packed to the output dtype. SRC_NORM phases first compute
// RMSNorm(x[0 : KX]) row scales (kernels/mla/rmsnorm.hpp) and produce every K block of in0 themselves (x * inv * w).
// Compile-time args: NPH, CB_IN0, CB_PART, CB_X, CB_NW, CB_INV, CB_GIN, OUT_CAP (output CB tiles), CB_QN (the
//   normalised in0 of SRC_NORM phases: a CB of its own, produced by the packer — the pack side keeps a local
//   tiles_received count, so one CB must not have a data-movement producer in one phase and the packer in another),
//   CB_OUT32 + OUT32_CAP (the fp32 output CB of fp32-output phases and its tiles), CB_RS (the row-scale tile: RS phases
//   multiply every output row t by its column-0 value — the q_b / I3 phases of the one-program half run on the raw q_a
//   with the RMSNorm weight folded into their weight rows, and 1 / rms(q_a) comes from the q-norm core), then per phase
//   p 13 values at 12 + 13p: SRC, NB, KB, NR, NH, KG, SUBW, CB_W, FID, OUT_CB, INV_N_BITS, EPS_BITS, RS.
// Runtime args: per phase p: 1 = this reader computes it (0: it sits the phase out).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/mla_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/rmsnorm.hpp"

namespace {

constexpr uint32_t NPH = get_compile_time_arg_val(PH_CT0 + 0);
constexpr uint32_t CB_IN0 = get_compile_time_arg_val(PH_CT0 + 1);
constexpr uint32_t CB_PART = get_compile_time_arg_val(PH_CT0 + 2);
constexpr uint32_t CB_X = get_compile_time_arg_val(PH_CT0 + 3);
constexpr uint32_t CB_NW = get_compile_time_arg_val(PH_CT0 + 4);
constexpr uint32_t CB_INV = get_compile_time_arg_val(PH_CT0 + 5);
constexpr uint32_t CB_GIN = get_compile_time_arg_val(PH_CT0 + 6);
constexpr uint32_t OUT_CAP = get_compile_time_arg_val(PH_CT0 + 7);
constexpr uint32_t CB_QN = get_compile_time_arg_val(PH_CT0 + 8);
constexpr uint32_t CB_OUT32 = get_compile_time_arg_val(PH_CT0 + 9);
constexpr uint32_t OUT32_CAP = get_compile_time_arg_val(PH_CT0 + 10);
constexpr uint32_t CB_RS = get_compile_time_arg_val(PH_CT0 + 11);
constexpr uint32_t PH0 = 12, PHN = 13;

// matmul_block_init / matmul_block with the phase's math fidelity F (kernels/mk/reader_c.cpp)
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
    constexpr uint32_t B = PH0 + PHN * P;
    constexpr uint32_t SRC = get_compile_time_arg_val(PH_CT0 + B + 0);
    constexpr uint32_t NB = get_compile_time_arg_val(PH_CT0 + B + 1);
    constexpr uint32_t KB = get_compile_time_arg_val(PH_CT0 + B + 2);
    constexpr uint32_t NR = get_compile_time_arg_val(PH_CT0 + B + 3);
    constexpr uint32_t NH = get_compile_time_arg_val(PH_CT0 + B + 4);
    constexpr uint32_t KG = get_compile_time_arg_val(PH_CT0 + B + 5);
    constexpr uint32_t SUBW = get_compile_time_arg_val(PH_CT0 + B + 6);
    constexpr uint32_t CB_W = get_compile_time_arg_val(PH_CT0 + B + 7);
    constexpr auto FID = static_cast<MathFidelity>(get_compile_time_arg_val(PH_CT0 + B + 8));
    constexpr uint32_t OUT_CB = get_compile_time_arg_val(PH_CT0 + B + 9);
    constexpr uint32_t INV_N_BITS = get_compile_time_arg_val(PH_CT0 + B + 10);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(PH_CT0 + B + 11);
    constexpr uint32_t RS = get_compile_time_arg_val(PH_CT0 + B + 12);
    constexpr uint32_t IN = SRC == mla::SRC_ROWS ? CB_GIN : (SRC == mla::SRC_NORM ? CB_QN : CB_IN0);
    constexpr uint32_t BPH = KG / KB;
    static_assert(NR % SUBW == 0 && KG % KB == 0 && NB == NH * BPH, "phase geometry");
    if (get_arg_val<uint32_t>(ph_rt0 + P) == 0) {
        return;
    }

    if constexpr (SRC == mla::SRC_NORM) {
        cb_wait_front(CB_X, KG);
        mlan::norm_inv<KG>(CB_X, CB_INV, INV_N_BITS, EPS_BITS);
        cb_wait_front(CB_INV, 1);
    }
    for (uint32_t j = 0; j < NH; ++j) {
        mk::cb_rewind(CB_PART);
        for (uint32_t b = 0; b < BPH; ++b) {
            if constexpr (SRC == mla::SRC_NORM) {
                // in0 block b: x[b KB + k] * inv * w  (bf16, packed without L1 accumulation)
                cb_wait_front(CB_NW, KB);
                cb_reserve_back(CB_QN, KB);
                PACK((llk_pack_reconfig_l1_acc(0)));
                for (uint32_t k = 0; k < KB; ++k) {
                    tile_regs_acquire();
                    mlan::norm_tile(CB_X, b * KB + k, CB_INV, CB_NW, k);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_reconfig_data_format(CB_QN);
                    pack_tile(0, CB_QN);
                    tile_regs_release();
                }
                cb_push_back(CB_QN, KB);
                cb_pop_front(CB_NW, KB);
            }
            reconfig_data_format<SrcOrder::Reverse>(IN, CB_W);
            pack_reconfig_data_format(CB_PART);
            mm_init<FID>(IN, CB_W, SUBW, 1, KB);
            cb_wait_front(IN, KB);
            cb_wait_front(CB_W, KB * NR);
            for (uint32_t sb = 0; sb < NR / SUBW; ++sb) {
                tile_regs_acquire();
                for (uint32_t k = 0; k < KB; ++k) {
                    mm_block<FID>(IN, CB_W, k, k * NR + sb * SUBW, SUBW, 1, KB);
                }
                tile_regs_commit();
                cb_reserve_back(CB_PART, SUBW);
                tile_regs_wait();
                PACK((llk_pack_reconfig_l1_acc(b > 0 ? 1 : 0)));
                for (uint32_t i = 0; i < SUBW; ++i) {
                    pack_tile(i, CB_PART);
                }
                tile_regs_release();
                cb_push_back(CB_PART, SUBW);
            }
            cb_pop_front(IN, KB);
            cb_pop_front(CB_W, KB * NR);
            if (b + 1 < BPH) {
                cb_wait_front(CB_PART, NR);
                cb_pop_front(CB_PART, NR);
                mk::cb_rewind(CB_PART);  // the next block accumulates onto the same tiles
            }
        }
        if constexpr (SRC == mla::SRC_NORM) {
            if (j + 1 == NH) {
                cb_pop_front(CB_X, KG);
                cb_pop_front(CB_INV, 1);
            }
        }
        PACK((llk_pack_reconfig_l1_acc(0)));
        cb_wait_front(CB_PART, NR);
        // the output CB empty (the mover wrote and popped the previous sub-matmul's tiles), then from its base
        cb_reserve_back(OUT_CB, OUT_CB == CB_OUT32 ? OUT32_CAP : OUT_CAP);
        mk::cb_rewind(OUT_CB);
        cb_reserve_back(OUT_CB, NR);
        if constexpr (RS == 1) {
            if (j == 0) {
                cb_wait_front(CB_RS, 1);
            }
            pack_reconfig_data_format(OUT_CB);
            for (uint32_t c = 0; c < NR; ++c) {  // output row t * rs[t] (fp32 DST, column broadcast)
                tile_regs_acquire();
                reconfig_data_format_srca(CB_PART);
                copy_tile_to_dst_init_short(CB_PART);
                copy_tile(CB_PART, c, 0);
                reconfig_data_format_srca(CB_RS);
                copy_tile_to_dst_init_short(CB_RS);
                copy_tile(CB_RS, 0, 1);
                sfpu_mul_bcast_col_init();
                sfpu_mul_bcast_col(0, 1);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, OUT_CB);
                tile_regs_release();
            }
        } else {
            reconfig_data_format_srca(CB_PART);
            pack_reconfig_data_format(OUT_CB);
            copy_tile_to_dst_init_short(CB_PART);
            for (uint32_t c = 0; c < NR; ++c) {
                tile_regs_acquire();
                copy_tile(CB_PART, c, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, OUT_CB);
                tile_regs_release();
            }
        }
        cb_push_back(OUT_CB, NR);
        cb_pop_front(CB_PART, NR);
        mk::cb_rewind(CB_PART);
    }
    if constexpr (RS == 1) {
        cb_pop_front(CB_RS, 1);
    }
    // phase end: every in0 page this RISC produced / consumed is popped -> back to the bases for the next phase
    mk::cb_rewind(IN);
    if constexpr (SRC == mla::SRC_NORM) {
        mk::cb_rewind(CB_X);
        mk::cb_rewind(CB_NW);
        mk::cb_rewind(CB_INV);
    }
}

template <uint32_t P>
FORCE_INLINE void run_c() {
    if constexpr (P < NPH) {
        mm_phase<P>();
        run_c<P + 1>();
    }
}

}  // namespace

void kernel_main() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_IN0, get_compile_time_arg_val(PH_CT0 + PH0 + 7), CB_PART);
    run_c<0>();
}
