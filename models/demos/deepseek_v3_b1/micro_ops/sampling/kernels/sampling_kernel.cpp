// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Sampling unified kernel (k=1 argmax fast path)
//
// Current scope:
// - Single-device, single-core
// - Input scores: bf16 [1, 160]
// - Input indices: uint32 [1, 160]
// - Output index: uint32 [1, 1]
//
// Tie-break: lowest index when scores are equal.

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/numeric/bfloat16.h"

struct Core {
    static constexpr bool is_active_core = get_named_compile_time_arg_val("sampling_is_active_core") == 1;
};

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    if constexpr (Core::is_active_core) {
        constexpr uint32_t num_values = get_named_compile_time_arg_val("sampling_num_values");

        const uint32_t scores_addr = get_common_arg_val<uint32_t>(0);
        const uint32_t indices_addr = get_common_arg_val<uint32_t>(1);
        const uint32_t output_addr = get_common_arg_val<uint32_t>(2);

        auto scores_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scores_addr);
        auto indices_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(indices_addr);
        auto output_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(output_addr);

        uint16_t best_score = NEG_INF_BFLOAT16;
        uint32_t best_index = 0xFFFFFFFF;

        for (uint32_t i = 0; i < num_values; ++i) {
            const uint16_t score = scores_ptr[i];
            const uint32_t index = indices_ptr[i];

            if (bfloat16_greater(score, best_score) || (score == best_score && index < best_index)) {
                best_score = score;
                best_index = index;
            }
        }

        output_ptr[0] = best_index;
    }

#elif defined(COMPILE_FOR_BRISC)
    // No-op for k=1 argmax fast path.

#elif defined(COMPILE_FOR_TRISC)
    // No-op for k=1 argmax fast path.
#endif
}
