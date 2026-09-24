// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/experimental/ckernel_sfpu_generalized_moe_gate_topk_single_face.h"

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
inline void generalized_moe_gate_top8(std::uint32_t eps, std::uint32_t scale) {
    _generalized_moe_gate_top8<APPROXIMATION_MODE, is_fp32_dest_acc_en>(eps, scale);
}

// Keep the two-argument functor above callable through SFPU_UNARY_CALL unchanged.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void generalized_moe_gate_top8_scaled(std::uint32_t eps, std::uint32_t scale, std::uint32_t extra_scale) {
    _generalized_moe_gate_top8<APPROXIMATION_MODE, is_fp32_dest_acc_en, true>(eps, scale, extra_scale);
}

template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    std::uint32_t read_base,
    std::uint32_t store_lo,
    std::uint32_t store_hi>
inline void generalized_moe_gate_merge4_top8() {
    _gmg_merge4_top8<is_fp32_dest_acc_en, read_base, store_lo, store_hi>();
}

template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    std::uint32_t store_lo,
    std::uint32_t store_hi,
    std::uint32_t idx_offset>
inline void generalized_moe_gate_merge16_to_run() {
    _gmg_merge16_to_run<APPROXIMATION_MODE, is_fp32_dest_acc_en, store_lo, store_hi, idx_offset>();
}

template <
    bool APPROXIMATION_MODE,
    std::uint32_t from_lo,
    std::uint32_t from_hi,
    std::uint32_t to_lo,
    std::uint32_t to_hi>
inline void generalized_moe_gate_copy_topk_run() {
    _gmg_copy_topk_run<from_lo, from_hi, to_lo, to_hi>();
}

template <
    bool APPROXIMATION_MODE,
    std::uint32_t field,
    std::uint32_t src_lo,
    std::uint32_t src_hi,
    std::uint32_t dst_lo,
    std::uint32_t dst_hi>
inline void generalized_moe_gate_place_field_from_interm() {
    _gmg_place_field_from_interm<field, src_lo, src_hi, dst_lo, dst_hi>();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, std::uint32_t topk = 8, bool output_softmax = false>
inline void generalized_moe_gate_finalize_ungrouped(std::uint32_t eps, std::uint32_t scale) {
    _generalized_moe_gate_finalize_ungrouped<APPROXIMATION_MODE, is_fp32_dest_acc_en, topk, output_softmax>(eps, scale);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void generalized_moe_gate_topk_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    _init_generalized_moe_gate_topk<APPROXIMATION_MODE, is_fp32_dest_acc_en>();
}

// Op classes for the generalized MoE gate top-k stages. Each kernel walks the single-face gate layout
// in Dest itself. generalized_moe_gate_topk_init is the shared init of every stage.

// Op class that sums the top-2 scores of every expert group.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
struct GeneralizedMoeGateSumTop2 : SfpuUnaryOp<GeneralizedMoeGateSumTop2<APPROXIMATION_MODE, is_fp32_dest_acc_en>> {
    static constexpr bool walks_faces = false;

    static inline __attribute__((always_inline)) void calculate() {
        generalized_moe_gate_sum_top2<APPROXIMATION_MODE, is_fp32_dest_acc_en>();
    }

    static inline __attribute__((always_inline)) void init_op() {
        generalized_moe_gate_topk_init<APPROXIMATION_MODE, is_fp32_dest_acc_en>();
    }
};

// Op class that selects the top-4 expert groups.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
struct GeneralizedMoeGateSortTop4Groups
    : SfpuUnaryOp<GeneralizedMoeGateSortTop4Groups<APPROXIMATION_MODE, is_fp32_dest_acc_en>> {
    static constexpr bool walks_faces = false;

    static inline __attribute__((always_inline)) void calculate() {
        generalized_moe_gate_sort_top4_groups<APPROXIMATION_MODE, is_fp32_dest_acc_en>();
    }
};

// Op class that merges the selected groups into the normalized top-8.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
struct GeneralizedMoeGateTop8 : SfpuUnaryOp<GeneralizedMoeGateTop8<APPROXIMATION_MODE, is_fp32_dest_acc_en>> {
    static constexpr bool walks_faces = false;

    static inline __attribute__((always_inline)) void calculate(std::uint32_t eps, std::uint32_t scale) {
        generalized_moe_gate_top8<APPROXIMATION_MODE, is_fp32_dest_acc_en>(eps, scale);
    }
};

// Op class that merges the selected groups into the normalized top-8, folding in an extra scale.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
struct GeneralizedMoeGateTop8Scaled
    : SfpuUnaryOp<GeneralizedMoeGateTop8Scaled<APPROXIMATION_MODE, is_fp32_dest_acc_en>> {
    static constexpr bool walks_faces = false;

    static inline __attribute__((always_inline)) void calculate(
        std::uint32_t eps, std::uint32_t scale, std::uint32_t extra_scale) {
        generalized_moe_gate_top8_scaled<APPROXIMATION_MODE, is_fp32_dest_acc_en>(eps, scale, extra_scale);
    }
};

// Op class that merges four groups into a top-8 run.
template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    std::uint32_t read_base,
    std::uint32_t store_lo,
    std::uint32_t store_hi>
struct GeneralizedMoeGateMerge4Top8
    : SfpuUnaryOp<
          GeneralizedMoeGateMerge4Top8<APPROXIMATION_MODE, is_fp32_dest_acc_en, read_base, store_lo, store_hi>> {
    static constexpr bool walks_faces = false;

    static inline __attribute__((always_inline)) void calculate() {
        generalized_moe_gate_merge4_top8<APPROXIMATION_MODE, is_fp32_dest_acc_en, read_base, store_lo, store_hi>();
    }
};

// Op class that merges two top-8 runs into a re-mergeable top-8 run.
template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    std::uint32_t store_lo,
    std::uint32_t store_hi,
    std::uint32_t idx_offset>
struct GeneralizedMoeGateMerge16ToRun
    : SfpuUnaryOp<
          GeneralizedMoeGateMerge16ToRun<APPROXIMATION_MODE, is_fp32_dest_acc_en, store_lo, store_hi, idx_offset>> {
    static constexpr bool walks_faces = false;

    static inline __attribute__((always_inline)) void calculate() {
        generalized_moe_gate_merge16_to_run<APPROXIMATION_MODE, is_fp32_dest_acc_en, store_lo, store_hi, idx_offset>();
    }
};

// Op class that relocates a top-k run between column pairs.
template <
    bool APPROXIMATION_MODE,
    std::uint32_t from_lo,
    std::uint32_t from_hi,
    std::uint32_t to_lo,
    std::uint32_t to_hi>
struct GeneralizedMoeGateCopyTopkRun
    : SfpuUnaryOp<GeneralizedMoeGateCopyTopkRun<APPROXIMATION_MODE, from_lo, from_hi, to_lo, to_hi>> {
    static constexpr bool walks_faces = false;

    static inline __attribute__((always_inline)) void calculate() {
        generalized_moe_gate_copy_topk_run<APPROXIMATION_MODE, from_lo, from_hi, to_lo, to_hi>();
    }
};

// Op class that places one field of a run from the interm region into its home region.
template <
    bool APPROXIMATION_MODE,
    std::uint32_t field,
    std::uint32_t src_lo,
    std::uint32_t src_hi,
    std::uint32_t dst_lo,
    std::uint32_t dst_hi>
struct GeneralizedMoeGatePlaceFieldFromInterm
    : SfpuUnaryOp<GeneralizedMoeGatePlaceFieldFromInterm<APPROXIMATION_MODE, field, src_lo, src_hi, dst_lo, dst_hi>> {
    static constexpr bool walks_faces = false;

    static inline __attribute__((always_inline)) void calculate() {
        generalized_moe_gate_place_field_from_interm<APPROXIMATION_MODE, field, src_lo, src_hi, dst_lo, dst_hi>();
    }
};

// Op class that sorts two top-8 runs into the global top-k and normalizes it.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, std::uint32_t topk = 8, bool output_softmax = false>
struct GeneralizedMoeGateFinalizeUngrouped
    : SfpuUnaryOp<GeneralizedMoeGateFinalizeUngrouped<APPROXIMATION_MODE, is_fp32_dest_acc_en, topk, output_softmax>> {
    static constexpr bool walks_faces = false;

    static inline __attribute__((always_inline)) void calculate(std::uint32_t eps, std::uint32_t scale) {
        generalized_moe_gate_finalize_ungrouped<APPROXIMATION_MODE, is_fp32_dest_acc_en, topk, output_softmax>(
            eps, scale);
    }

    static inline __attribute__((always_inline)) void init_op() {
        generalized_moe_gate_topk_init<APPROXIMATION_MODE, is_fp32_dest_acc_en>();
    }
};

}  // namespace sfpu
}  // namespace ckernel
