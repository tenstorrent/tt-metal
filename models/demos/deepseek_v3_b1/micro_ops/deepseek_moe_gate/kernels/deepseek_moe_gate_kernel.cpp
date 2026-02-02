// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Deepseek MoE Gate unified kernel
// Single kernel file, compiles for all RISC cores
//
// NCRISC: Signals tensor-backed input CBs ready
// BRISC: Waits for output CBs
// TRISC: Computes gate logic (sigmoid, bias add, sorting, normalization)

#include "../../../unified_kernels/kernel_op_api.hpp"
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
    using MoeGateCTArgs = deepseek_b1_ops::DeepseekMoeGate::ReaderCTArgs<
        get_named_compile_time_arg_val("moe_gate_input_cb"),
        get_named_compile_time_arg_val("moe_gate_bias_cb"),
        get_named_compile_time_arg_val("moe_gate_input_indices_cb")>;

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
#endif

    // ========================================================================
    // Deepseek MoE Gate operation
    // ========================================================================
    deepseek_b1_ops::DeepseekMoeGate::Op<MoeGateCTArgs, Core::is_active_core> moe_gate;
    moe_gate();
}
