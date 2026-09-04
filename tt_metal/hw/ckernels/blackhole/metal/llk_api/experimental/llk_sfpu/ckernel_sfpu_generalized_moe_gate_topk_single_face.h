// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpu/experimental/ckernel_sfpu_generalized_moe_gate_topk_single_face.h"
#include "llk_math_eltwise_sfpu_op.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void generalized_moe_gate_sum_top2() {
    _generalized_moe_gate_sum_top2<APPROXIMATION_MODE, is_fp32_dest_acc_en>();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void generalized_moe_gate_sort_top4_groups() {
    _generalized_moe_gate_sort_top4_groups<APPROXIMATION_MODE, is_fp32_dest_acc_en>();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void generalized_moe_gate_top8(uint32_t eps, uint32_t scale) {
    _generalized_moe_gate_top8<APPROXIMATION_MODE, is_fp32_dest_acc_en>(eps, scale);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, uint32_t read_base, uint32_t store_lo, uint32_t store_hi>
inline void generalized_moe_gate_merge4_top8() {
    _gmg_merge4_top8<is_fp32_dest_acc_en, read_base, store_lo, store_hi>();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, uint32_t store_lo, uint32_t store_hi, uint32_t idx_offset>
inline void generalized_moe_gate_merge16_to_run() {
    _gmg_merge16_to_run<APPROXIMATION_MODE, is_fp32_dest_acc_en, store_lo, store_hi, idx_offset>();
}

template <bool APPROXIMATION_MODE, uint32_t from_lo, uint32_t from_hi, uint32_t to_lo, uint32_t to_hi>
inline void generalized_moe_gate_copy_topk_run() {
    _gmg_copy_topk_run<from_lo, from_hi, to_lo, to_hi>();
}

template <bool APPROXIMATION_MODE, uint32_t field, uint32_t src_lo, uint32_t src_hi, uint32_t dst_lo, uint32_t dst_hi>
inline void generalized_moe_gate_place_field_from_interm() {
    _gmg_place_field_from_interm<field, src_lo, src_hi, dst_lo, dst_hi>();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, uint32_t topk = 8, bool output_softmax = false>
inline void generalized_moe_gate_finalize_ungrouped(uint32_t eps, uint32_t scale) {
    _generalized_moe_gate_finalize_ungrouped<APPROXIMATION_MODE, is_fp32_dest_acc_en, topk, output_softmax>(eps, scale);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void generalized_moe_gate_topk_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    _init_generalized_moe_gate_topk<APPROXIMATION_MODE, is_fp32_dest_acc_en>();
}

// ---------------------------------------------------------------------------------------------------
// Op structs for the generalized MoE gate SFPU phases (api/compute/experimental/generalized_moe_gate.h).
// Every phase shares one SFPU setup (generalized_moe_gate_topk_init), so each struct below mixes in
// GeneralizedMoeGateTopkInit and any of them can be used for init(); the compute API uses the
// dedicated init-only GeneralizedMoeGateTopk for readability. All phases run on dest tile 0 with
// VectorMode::RC_custom and take APPROXIMATION_MODE first, then their own compile-time row/field
// parameters, then DST_SYNC, DST_ACCUM (fed into the kernels' is_fp32_dest_acc_en).
// ---------------------------------------------------------------------------------------------------
// Shared init_kernel mixin; each op re-exports it with a using-declaration so it hides SfpuOpBase::init_kernel.
template <bool APPROXIMATION_MODE, bool DST_ACCUM>
struct GeneralizedMoeGateTopkInit {
    static void init_kernel() { generalized_moe_gate_topk_init<APPROXIMATION_MODE, DST_ACCUM>(); }
};

// GeneralizedMoeGateTopk<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM>::init()
//   Shared SFPU setup used by generalized_moe_gate_init / generalized_moe_gate / generalized_moe_gate_combine_init.
//   Init-only: it has no kernel, so calculate() is not available on it.
template <bool APPROXIMATION_MODE, DstSync DST_SYNC, bool DST_ACCUM>
struct GeneralizedMoeGateTopk
    : SfpuUnaryOp<GeneralizedMoeGateTopk<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM>, DST_SYNC, DST_ACCUM>,
      GeneralizedMoeGateTopkInit<APPROXIMATION_MODE, DST_ACCUM> {
    using GeneralizedMoeGateTopkInit<APPROXIMATION_MODE, DST_ACCUM>::init_kernel;
};

// GeneralizedMoeGateSumTop2<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM>::calculate(0, RC_custom)
//   generalized_moe_gate: per-group sum of the top-2 scores.
template <bool APPROXIMATION_MODE, DstSync DST_SYNC, bool DST_ACCUM>
struct GeneralizedMoeGateSumTop2
    : SfpuUnaryOp<GeneralizedMoeGateSumTop2<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM>, DST_SYNC, DST_ACCUM>,
      GeneralizedMoeGateTopkInit<APPROXIMATION_MODE, DST_ACCUM> {
    using GeneralizedMoeGateTopkInit<APPROXIMATION_MODE, DST_ACCUM>::init_kernel;
    static void kernel() { generalized_moe_gate_sum_top2<APPROXIMATION_MODE, DST_ACCUM>(); }
};

// GeneralizedMoeGateSortTop4Groups<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM>::calculate(0, RC_custom)
//   generalized_moe_gate (grouped path): select the top-4 groups.
template <bool APPROXIMATION_MODE, DstSync DST_SYNC, bool DST_ACCUM>
struct GeneralizedMoeGateSortTop4Groups
    : SfpuUnaryOp<GeneralizedMoeGateSortTop4Groups<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM>, DST_SYNC, DST_ACCUM>,
      GeneralizedMoeGateTopkInit<APPROXIMATION_MODE, DST_ACCUM> {
    using GeneralizedMoeGateTopkInit<APPROXIMATION_MODE, DST_ACCUM>::init_kernel;
    static void kernel() { generalized_moe_gate_sort_top4_groups<APPROXIMATION_MODE, DST_ACCUM>(); }
};

// GeneralizedMoeGateTop8<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM>::calculate(0, RC_custom, eps, scale)
//   generalized_moe_gate (grouped path): top-8 merge + normalize.
template <bool APPROXIMATION_MODE, DstSync DST_SYNC, bool DST_ACCUM>
struct GeneralizedMoeGateTop8
    : SfpuUnaryOp<GeneralizedMoeGateTop8<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM>, DST_SYNC, DST_ACCUM>,
      GeneralizedMoeGateTopkInit<APPROXIMATION_MODE, DST_ACCUM> {
    using GeneralizedMoeGateTopkInit<APPROXIMATION_MODE, DST_ACCUM>::init_kernel;
    static void kernel(uint32_t eps, uint32_t scale) {
        generalized_moe_gate_top8<APPROXIMATION_MODE, DST_ACCUM>(eps, scale);
    }
};

// GeneralizedMoeGateMerge4Top8<APPROXIMATION_MODE, READ_BASE, STORE_LO, STORE_HI, DST_SYNC, DST_ACCUM>::calculate(0,
// RC_custom)
//   generalized_moe_gate (ungrouped path): top-8 of four groups -> run at {STORE_LO, STORE_HI}.
template <
    bool APPROXIMATION_MODE,
    uint32_t READ_BASE,
    uint32_t STORE_LO,
    uint32_t STORE_HI,
    DstSync DST_SYNC,
    bool DST_ACCUM>
struct GeneralizedMoeGateMerge4Top8
    : SfpuUnaryOp<
          GeneralizedMoeGateMerge4Top8<APPROXIMATION_MODE, READ_BASE, STORE_LO, STORE_HI, DST_SYNC, DST_ACCUM>,
          DST_SYNC,
          DST_ACCUM>,
      GeneralizedMoeGateTopkInit<APPROXIMATION_MODE, DST_ACCUM> {
    using GeneralizedMoeGateTopkInit<APPROXIMATION_MODE, DST_ACCUM>::init_kernel;
    static void kernel() {
        generalized_moe_gate_merge4_top8<APPROXIMATION_MODE, DST_ACCUM, READ_BASE, STORE_LO, STORE_HI>();
    }
};

// GeneralizedMoeGateMerge16ToRun<APPROXIMATION_MODE, STORE_LO, STORE_HI, IDX_OFFSET, DST_SYNC, DST_ACCUM>::calculate(0,
// RC_custom)
//   generalized_moe_gate (produce_run): merge topA{0,2} + topB{4,6} into a re-mergeable run, idx += IDX_OFFSET.
template <
    bool APPROXIMATION_MODE,
    uint32_t STORE_LO,
    uint32_t STORE_HI,
    uint32_t IDX_OFFSET,
    DstSync DST_SYNC,
    bool DST_ACCUM>
struct GeneralizedMoeGateMerge16ToRun
    : SfpuUnaryOp<
          GeneralizedMoeGateMerge16ToRun<APPROXIMATION_MODE, STORE_LO, STORE_HI, IDX_OFFSET, DST_SYNC, DST_ACCUM>,
          DST_SYNC,
          DST_ACCUM>,
      GeneralizedMoeGateTopkInit<APPROXIMATION_MODE, DST_ACCUM> {
    using GeneralizedMoeGateTopkInit<APPROXIMATION_MODE, DST_ACCUM>::init_kernel;
    static void kernel() {
        generalized_moe_gate_merge16_to_run<APPROXIMATION_MODE, DST_ACCUM, STORE_LO, STORE_HI, IDX_OFFSET>();
    }
};

// GeneralizedMoeGateCopyTopkRun<APPROXIMATION_MODE, FROM_LO, FROM_HI, TO_LO, TO_HI, DST_SYNC, DST_ACCUM>::calculate(0,
// RC_custom)
//   generalized_moe_gate_relocate_run: move a run between column pairs.
template <
    bool APPROXIMATION_MODE,
    uint32_t FROM_LO,
    uint32_t FROM_HI,
    uint32_t TO_LO,
    uint32_t TO_HI,
    DstSync DST_SYNC,
    bool DST_ACCUM>
struct GeneralizedMoeGateCopyTopkRun
    : SfpuUnaryOp<
          GeneralizedMoeGateCopyTopkRun<APPROXIMATION_MODE, FROM_LO, FROM_HI, TO_LO, TO_HI, DST_SYNC, DST_ACCUM>,
          DST_SYNC,
          DST_ACCUM>,
      GeneralizedMoeGateTopkInit<APPROXIMATION_MODE, DST_ACCUM> {
    using GeneralizedMoeGateTopkInit<APPROXIMATION_MODE, DST_ACCUM>::init_kernel;
    static void kernel() { generalized_moe_gate_copy_topk_run<APPROXIMATION_MODE, FROM_LO, FROM_HI, TO_LO, TO_HI>(); }
};

// GeneralizedMoeGatePlaceFieldFromInterm<APPROXIMATION_MODE, FIELD, SRC_LO, SRC_HI, DST_LO, DST_HI, DST_SYNC,
// DST_ACCUM>
//   ::calculate(0, RC_custom)
//   generalized_moe_gate_place_field_from_interm: copy one field (bias/idx/score) of a run from the
//   interm region into its home region.
template <
    bool APPROXIMATION_MODE,
    uint32_t FIELD,
    uint32_t SRC_LO,
    uint32_t SRC_HI,
    uint32_t DST_LO,
    uint32_t DST_HI,
    DstSync DST_SYNC,
    bool DST_ACCUM>
struct GeneralizedMoeGatePlaceFieldFromInterm : SfpuUnaryOp<
                                                    GeneralizedMoeGatePlaceFieldFromInterm<
                                                        APPROXIMATION_MODE,
                                                        FIELD,
                                                        SRC_LO,
                                                        SRC_HI,
                                                        DST_LO,
                                                        DST_HI,
                                                        DST_SYNC,
                                                        DST_ACCUM>,
                                                    DST_SYNC,
                                                    DST_ACCUM>,
                                                GeneralizedMoeGateTopkInit<APPROXIMATION_MODE, DST_ACCUM> {
    using GeneralizedMoeGateTopkInit<APPROXIMATION_MODE, DST_ACCUM>::init_kernel;
    static void kernel() {
        generalized_moe_gate_place_field_from_interm<APPROXIMATION_MODE, FIELD, SRC_LO, SRC_HI, DST_LO, DST_HI>();
    }
};

// GeneralizedMoeGateFinalizeUngrouped<APPROXIMATION_MODE, TOPK, OUTPUT_SOFTMAX, DST_SYNC, DST_ACCUM>
//   ::calculate(0, RC_custom, eps, scale)
//   generalized_moe_gate (single block) / generalized_moe_gate_combine_finalize: bitonic sort of the 16
//   candidates -> top-TOPK + normalize (softmax over the kept when OUTPUT_SOFTMAX).
template <bool APPROXIMATION_MODE, uint32_t TOPK, bool OUTPUT_SOFTMAX, DstSync DST_SYNC, bool DST_ACCUM>
struct GeneralizedMoeGateFinalizeUngrouped
    : SfpuUnaryOp<
          GeneralizedMoeGateFinalizeUngrouped<APPROXIMATION_MODE, TOPK, OUTPUT_SOFTMAX, DST_SYNC, DST_ACCUM>,
          DST_SYNC,
          DST_ACCUM>,
      GeneralizedMoeGateTopkInit<APPROXIMATION_MODE, DST_ACCUM> {
    using GeneralizedMoeGateTopkInit<APPROXIMATION_MODE, DST_ACCUM>::init_kernel;
    static void kernel(uint32_t eps, uint32_t scale) {
        generalized_moe_gate_finalize_ungrouped<APPROXIMATION_MODE, DST_ACCUM, TOPK, OUTPUT_SOFTMAX>(eps, scale);
    }
};

}  // namespace sfpu
}  // namespace ckernel
