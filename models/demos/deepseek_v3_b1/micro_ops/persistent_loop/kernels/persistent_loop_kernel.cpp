// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/persistent_loop.hpp"

void kernel_main() {
    constexpr uint32_t persistent_mode = get_named_compile_time_arg_val("persistent_mode");
    constexpr uint32_t termination_semaphore_addr = get_named_compile_time_arg_val("termination_semaphore_addr");
    constexpr uint32_t max_iterations = get_named_compile_time_arg_val("max_iterations");
    constexpr uint32_t iteration_count_addr = get_named_compile_time_arg_val("iteration_count_addr");

    deepseek_b1_ops::PersistentLoop<persistent_mode == 1> loop(termination_semaphore_addr, max_iterations);
    while (loop.next()) {
#if defined(COMPILE_FOR_BRISC)
        volatile tt_l1_ptr uint32_t* count_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iteration_count_addr);
        *count_ptr = loop.iteration();
#endif
    }
}
