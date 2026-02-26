// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Deepseek MoE Gate unified kernel
// Single kernel file, compiles for all RISC cores
//
// NCRISC: Sets up sharded CBs (input, bias, indices)
// BRISC: Waits for output CBs
// TRISC: Computes gate logic (sigmoid, bias add, sorting, normalization)

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/deepseek_moe_gate.hpp"

// Compile-time role flag for dead code elimination via if constexpr
struct Core {
    static constexpr bool is_active_core = get_named_compile_time_arg_val("moe_gate_is_active_core") == 1;
};

void kernel_main() {
// ============================================================================
// Define CTArgs per RISC
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    using MoeGateCTArgs = deepseek_b1_ops::DeepseekMoeGate::ReaderCTArgs;

    // Named compile-time args for sharded buffer setup
    constexpr uint32_t input_cb = get_named_compile_time_arg_val("moe_gate_input_cb");
    constexpr uint32_t bias_cb = get_named_compile_time_arg_val("moe_gate_bias_cb");
    constexpr uint32_t input_indices_cb = get_named_compile_time_arg_val("moe_gate_input_indices_cb");

    // Setup sharded persistent buffers (all tensor-backed)
    if constexpr (Core::is_active_core) {
        unified_kernels::setup_sharded_buffer(input_cb, 1);
        unified_kernels::setup_sharded_buffer(bias_cb, 1);
        unified_kernels::setup_sharded_buffer(input_indices_cb, 1);
    }

#elif defined(COMPILE_FOR_BRISC)
    using MoeGateCTArgs = deepseek_b1_ops::DeepseekMoeGate::WriterCTArgs<
        get_named_compile_time_arg_val("moe_gate_output_cb"),
        get_named_compile_time_arg_val("moe_gate_output_indices_cb")>;

#elif defined(COMPILE_FOR_TRISC)
    using MoeGateCTArgs = deepseek_b1_ops::DeepseekMoeGate::ComputeCTArgs<
        get_named_compile_time_arg_val("moe_gate_input_cb"),
        get_named_compile_time_arg_val("moe_gate_bias_cb"),
        get_named_compile_time_arg_val("moe_gate_input_indices_cb"),
        get_named_compile_time_arg_val("moe_gate_output_cb"),
        get_named_compile_time_arg_val("moe_gate_output_indices_cb"),
        get_named_compile_time_arg_val("moe_gate_eps"),
        get_named_compile_time_arg_val("moe_gate_scaling_factor"),
        get_named_compile_time_arg_val("moe_gate_enable_sigmoid")>;
    deepseek_compute_kernel_init();
#endif

    // ========================================================================
    // Deepseek MoE Gate operation
    // ========================================================================
    deepseek_b1_ops::DeepseekMoeGate::Op<MoeGateCTArgs, Core::is_active_core> moe_gate;
    moe_gate();
}
