// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
// Blackhole-only: the MoE-gate FPU/SFPU LLKs live only in the Blackhole llk_lib.
#if defined(TRISC_MATH) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_math_deepseek_moe_gate_eltwise_binary_api.h"  // fpu kernel which puts values into tile 0 of dest and values+bias into tile 2
#include "experimental/llk_sfpu/llk_math_generic_moe_gate_topk_api.h"
#endif

namespace ckernel {

#if defined(ARCH_BLACKHOLE)

ALWI void generic_moe_gate_init(uint32_t icb0, uint32_t icb1) {
    UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(icb0, icb1)));

    // Init copy add (FPU)
    MATH((llk_math_deepseek_moe_gate_eltwise_binary_init_with_operands<
          EltwiseBinaryType::ELWADD,
          DeepseekMoeGateEltwiseBinaryMode::COPY,
          MATH_FIDELITY>(icb0, icb1, false)));
    // Init topk (SFPU)
    MATH((sfpu::llk_math_sfpu_generic_moe_gate_topk_init()));
}

// generate_indices lets the SFPU fill DST[1] with the position identity [0..N-1] before sorting, which is
// what standalone GenericMoeGate needs since it uploads no index tensor. Set it false when the caller has
// pre-loaded DST[1] via copy_tile(input_indices, 0, 1) and wants that mapping to survive the sort -- the
// bit-15-flagged SRAM slots the hot/cold KimiMoeGate path relies on.
template <
    bool normalize = false,
    int num_selected_experts = 8,
    int num_total_experts = 256,
    bool zero_tail = false,
    bool full_sort = false,
    bool generate_indices = true,
    bool do_extra_scale = false>
ALWI void generic_moe_gate(uint32_t icb0, uint32_t icb1, uint32_t eps, uint32_t scale, uint32_t extra_scale = 0) {
    static_assert(num_selected_experts >= 1 && num_selected_experts <= 16);

    // Copy add (FPU)
    UNPACK((llk_unpack_AB(icb0, icb1, 0, 0)));
    MATH((llk_math_deepseek_moe_gate_eltwise_binary<EltwiseBinaryType::ELWADD, DST_ACCUM_MODE, MATH_FIDELITY>(
        icb0, icb1, 0, true)));

    // Topk SFPU
    MATH((sfpu::llk_math_sfpu_generic_moe_gate_topk<
          normalize,
          num_selected_experts,
          num_total_experts,
          zero_tail,
          full_sort,
          generate_indices,
          do_extra_scale>(eps, scale, extra_scale)));
}

#endif

}  // namespace ckernel
