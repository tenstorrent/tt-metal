// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "reconfig_data_format.h"
#include "tt_metal/include/compute_kernel_api/common_globals.h"

namespace ckernel {

// Forward declarations to avoid circular dependency with pack.h
ALWI void pack_reconfig_data_format(uint32_t new_cb_id);
ALWI void pack_reconfig_data_format(uint32_t old_cb_id, uint32_t new_cb_id);

// Invalid buffer index sentinel value
constexpr uint32_t INVALID_CB_INDEX = 0xFFFFFFFF;

/**
 * These enum values are used with state_configure<Operation::TYPE>() to determine
 * which hardware registers need reconfiguration.
 */
namespace Operation {
enum Type : uint8_t {
    UNARY,      // Single source operand (srcA): square, abs, exp, sqrt, etc.
    BINARY,     // Two source operands (srcA, srcB): add, mul, sub, broadcast ops, etc.
    MATMUL,     // Matrix multiplication (srcB, srcA)
    REDUCE,     // Reduction operations (srcA, srcB_scaler)
    PACK,       // Pack/output operations (dst)
    TILIZE,     // Tilize operations (srcA)
    UNTILIZE,   // Untilize operations (srcA)
    COPY,       // Copy/move operations (srcA)
    TRANSPOSE,  // Transpose operations (srcA)
    CUSTOM      // Custom operations (manual configuration)
};
}

/**
 * Global state tracking variables
 *
 * These track the currently configured circular buffer indices:
 * - g_state_srca_cb: Source A register configuration
 * - g_state_srcb_cb: Source B register configuration
 * - g_state_pack_cb: Pack/destination register configuration
 *
 * Initialized to INVALID_CB_INDEX to ensure first configuration always executes.
 */
static uint32_t g_state_srca_cb = INVALID_CB_INDEX;
static uint32_t g_state_srcb_cb = INVALID_CB_INDEX;
static uint32_t g_state_pack_cb = INVALID_CB_INDEX;

// Helper reconfig functions
ALWI void reconfigure_single_operand(uint32_t cb, bool is_srcA) {
    if (is_srcA) {
        reconfig_data_format_srca(g_state_srca_cb, cb);
        g_state_srca_cb = cb;
        return;
    }
    reconfig_data_format_srcb(g_state_srcb_cb, cb);
    g_state_srcb_cb = cb;
}

ALWI void reconfigure_dual_operand(uint32_t cb_a, uint32_t cb_b) {
    bool srcAChanged = g_state_srca_cb != cb_a;
    bool srcBChanged = g_state_srcb_cb != cb_b;

    if (srcAChanged && srcBChanged) {
        reconfig_data_format(g_state_srca_cb, cb_a, g_state_srcb_cb, cb_b);
        g_state_srca_cb = cb_a;
        g_state_srcb_cb = cb_b;
        return;
    }
    if (srcAChanged) {
        reconfig_data_format_srca(g_state_srca_cb, cb_a);
        g_state_srca_cb = cb_a;
    }
    if (srcBChanged) {
        reconfig_data_format_srcb(g_state_srcb_cb, cb_b);
        g_state_srcb_cb = cb_b;
    }
}

ALWI void reconfigure_pack_operand(uint32_t cb_out) {
    if (g_state_pack_cb != cb_out) {
        pack_reconfig_data_format(g_state_pack_cb, cb_out);
        g_state_pack_cb = cb_out;
    }
}

ALWI void reconfig_all_operands(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
    if (g_state_srca_cb != cb_a && g_state_srcb_cb != cb_b && g_state_pack_cb != cb_out) {
        reconfig_data_format(g_state_srca_cb, cb_a, g_state_srcb_cb, cb_b);
        pack_reconfig_data_format(g_state_pack_cb, cb_out);
        g_state_srca_cb = cb_a;
        g_state_srcb_cb = cb_b;
        g_state_pack_cb = cb_out;
        return;
    }
    reconfigure_dual_operand(cb_a, cb_b);
    reconfigure_pack_operand(cb_out);
}

// --- UNARY Operation Handler ---
template <bool to_from_int8>
ALWI void state_impl_unary(uint32_t cb) {
    // UNARY-specific implementation: exp, sqrt, abs, etc.
}

// --- COPY Operation Handler ---
template <bool to_from_int8>
ALWI void state_impl_copy(uint32_t cb) {
    // COPY-specific implementation: copy_tile
}

// --- TRANSPOSE Operation Handler ---
template <bool to_from_int8>
ALWI void state_impl_transpose(uint32_t cb) {
    // TRANSPOSE-specific implementation: transpose_wh
}

// --- BINARY Operation Handler ---
template <bool to_from_int8>
ALWI void state_impl_binary(uint32_t cb_a, uint32_t cb_b) {
    // reconfigure_dual_operand(cb_a, cb_b);
}

// --- MATMUL Operation Handler ---
template <bool to_from_int8>
ALWI void state_impl_matmul(uint32_t cb_a, uint32_t cb_b) {
    // MATMUL-specific implementation: matrix multiplication
}

// --- REDUCE Operation Handler ---
template <bool to_from_int8>
ALWI void state_impl_reduce(uint32_t cb_input, uint32_t cb_scaler) {
    // REDUCE-specific implementation: reduction operations
}

// --- UNARY with output Operation Handler ---
template <bool to_from_int8>
ALWI void state_impl_unary_with_output(uint32_t cb_in, uint32_t cb_out) {
    // UNARY with output implementation: unary_op_init_common
}

// --- TRANSPOSE with output Operation Handler ---
template <bool to_from_int8>
ALWI void state_impl_transpose_with_output(uint32_t cb_in, uint32_t cb_out) {
    // TRANSPOSE with output implementation: transpose_wh_init
}

// --- TILIZE Operation Handler ---
template <bool to_from_int8>
ALWI void state_impl_tilize(uint32_t cb_in, uint32_t cb_out) {
    // TILIZE-specific implementation
}

// --- UNTILIZE Operation Handler ---
template <bool to_from_int8>
ALWI void state_impl_untilize(uint32_t cb_in, uint32_t cb_out) {
    // UNTILIZE-specific implementation
}

// --- BINARY with output Operation Handler ---
template <bool to_from_int8>
ALWI void state_impl_binary_with_output(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
    reconfig_all_operands(cb_a, cb_b, cb_out);
}

// --- MATMUL with output Operation Handler ---
template <bool to_from_int8>
ALWI void state_impl_matmul_with_output(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
    // MATMUL with output implementation: full mm_init
}

// --- PACK Operation Handler ---
template <bool to_from_int8>
ALWI void state_impl_pack(uint32_t cb) {
    reconfigure_pack_operand(cb);
}

// --- REDUCE with output Operation Handler ---
template <bool to_from_int8>
ALWI void state_impl_reduce_with_output(uint32_t cb_input, uint32_t cb_scaler, uint32_t cb_out) {
    bool input_changed = g_state_srca_cb != cb_input;
    bool scaler_changed = g_state_srcb_cb != cb_scaler;
    bool output_changed = g_state_pack_cb != cb_out;
    if (input_changed && scaler_changed && output_changed) {
        g_state_srca_cb = cb_input;
        g_state_srcb_cb = cb_scaler;
        g_state_pack_cb = cb_out;
        return;
    }
    if (input_changed) {
        reconfig_data_format_srca<to_from_int8>(cb_input);
        g_state_srca_cb = cb_input;
    }
    if (scaler_changed) {
        reconfig_data_format_srcb<to_from_int8>(cb_scaler);
        g_state_srcb_cb = cb_scaler;
    }
    if (output_changed) {
        pack_reconfig_data_format(g_state_pack_cb, cb_out);
        g_state_pack_cb = cb_out;
    }
}

/**
 * UNARY, COPY, TRANSPOSE operations
 * Dispatches to operation-specific implementation
 */
template <Operation::Type op_type, bool to_from_int8>
ALWI void state_configure(uint32_t cb) {
    static_assert(
        op_type == Operation::UNARY || op_type == Operation::COPY || op_type == Operation::TRANSPOSE ||
            op_type == Operation::PACK,
        "Invalid operation type for single-operand state_configure");

    if constexpr (op_type == Operation::UNARY) {
        reconfigure_single_operand(cb, true);
        // state_impl_unary<to_from_int8>(cb);
    } else if constexpr (op_type == Operation::COPY) {
        // state_impl_copy<to_from_int8>(cb);
        reconfigure_single_operand(cb, true);
    } else if constexpr (op_type == Operation::TRANSPOSE) {
        // state_impl_transpose<to_from_int8>(cb);
        reconfigure_single_operand(cb, true);
    } else if constexpr (op_type == Operation::PACK) {
        // state_impl_pack<to_from_int8>(cb);
        reconfigure_pack_operand(cb);
    }
}

/**
 * BINARY, MATMUL, REDUCE operations
 * Dispatches to operation-specific implementation
 */
template <Operation::Type op_type, bool to_from_int8 = false>
ALWI void state_configure(uint32_t cb_a, uint32_t cb_b) {
    static_assert(
        op_type == Operation::BINARY || op_type == Operation::MATMUL || op_type == Operation::REDUCE,
        "Invalid operation type for dual-operand state_configure");

    if constexpr (op_type == Operation::BINARY) {
        // state_impl_binary<to_from_int8>(cb_a, cb_b);
        reconfigure_dual_operand(cb_a, cb_b);
    } else if constexpr (op_type == Operation::MATMUL) {
        // state_impl_matmul<to_from_int8>(cb_a, cb_b);
    } else if constexpr (op_type == Operation::REDUCE) {
        // state_impl_redce<to_from_int8>(cb_a, cb_b);
        reconfigure_dual_operand(cb_a, cb_b);
    }
}

ALWI void set_g_states(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
    g_state_srca_cb = cb_a;
    g_state_srcb_cb = cb_b;
    g_state_pack_cb = cb_out;
}

ALWI void set_g_srca_srcb(uint32_t cb_a, uint32_t cb_b) {
    g_state_srca_cb = cb_a;
    g_state_srcb_cb = cb_b;
}

ALWI void set_g_srca(uint32_t cb_a) { g_state_srca_cb = cb_a; }
ALWI void set_g_srcb(uint32_t cb_b) { g_state_srcb_cb = cb_b; }
ALWI void set_g_pack(uint32_t cb_out) { g_state_pack_cb = cb_out; }

/**
 * UNARY, TRANSPOSE, TILIZE, UNTILIZE with output
 * Dispatches to operation-specific implementation
 */
template <Operation::Type op_type, bool to_from_int8 = false>
ALWI void state_configure_with_output(uint32_t cb_in, uint32_t cb_out) {
    static_assert(
        op_type == Operation::UNARY || op_type == Operation::TRANSPOSE || op_type == Operation::TILIZE ||
            op_type == Operation::UNTILIZE,
        "Invalid operation type for state_configure_with_output");

    if constexpr (op_type == Operation::UNARY) {
        // state_impl_unary_with_output<to_from_int8>(cb_in, cb_out);
    } else if constexpr (op_type == Operation::TRANSPOSE) {
        // state_impl_transpose_with_output<to_from_int8>(cb_in, cb_out);
    } else if constexpr (op_type == Operation::TILIZE) {
        // state_impl_tilize<to_from_int8>(cb_in, cb_out);
    } else if constexpr (op_type == Operation::UNTILIZE) {
        // state_impl_untilize<to_from_int8>(cb_in, cb_out);
    }
}

// -------------------------
// Full pipeline operations (3 CBs: input + input + output)
// -------------------------

/**
 * BINARY, MATMUL, REDUCE with output
 * Dispatches to operation-specific implementation
 */
template <Operation::Type op_type, bool to_from_int8 = false>
ALWI void state_configure(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
    static_assert(
        op_type == Operation::BINARY || op_type == Operation::MATMUL || op_type == Operation::REDUCE,
        "Invalid operation type for full pipeline state_configure");

    if constexpr (op_type == Operation::BINARY) {
        // state_impl_binary_with_output<to_from_int8>(cb_a, cb_b, cb_out);
        reconfig_all_operands(cb_a, cb_b, cb_out);
    } else if constexpr (op_type == Operation::MATMUL) {
        // state_impl_matmul_with_output<to_from_int8>(cb_a, cb_b, cb_out);
    } else if constexpr (op_type == Operation::REDUCE) {
        // state_impl_reduce_with_output<to_from_int8>(cb_a, cb_b, cb_out);
        reconfig_all_operands(cb_a, cb_b, cb_out);
    }
}

/**
 * @brief Reset all global state to invalid
 *
 * Call this at kernel boundaries or when external events may have
 * changed hardware state outside the API's knowledge.
 */
ALWI void reset_state_tracker() {
    g_state_srca_cb = INVALID_CB_INDEX;
    g_state_srcb_cb = INVALID_CB_INDEX;
    g_state_pack_cb = INVALID_CB_INDEX;
}

/**
 * @brief Get current srcA state (for debugging)
 */
ALWI uint32_t get_state_srca() { return g_state_srca_cb; }

/**
 * @brief Get current srcB state (for debugging)
 */
ALWI uint32_t get_state_srcb() { return g_state_srcb_cb; }

/**
 * @brief Get current pack state (for debugging)
 */
ALWI uint32_t get_state_pack() { return g_state_pack_cb; }

}  // namespace ckernel
