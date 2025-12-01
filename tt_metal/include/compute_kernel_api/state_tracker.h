// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "reconfig_data_format.h"
#include "tt_metal/include/compute_kernel_api/common_globals.h"

namespace ckernel {

/*
 * State Tracking Overview:
 *
 * The state_configure functions are organized into three overloads based on operand count:
 *
 * 1) Single operand: state_configure(cb)
 *    - Handles operations that only need one input buffer (UNARY, COPY, TRANSPOSE, PACK)
 *    - Reconfigures either srcA or pack registers depending on operation type
 *
 * 2) Dual operand: state_configure(cb_a, cb_b)
 *    - Handles operations with two buffers (BINARY, MATMUL, REDUCE inputs, or UNARY with output)
 *    - Reconfigures srcA and srcB registers, or srcA and pack for UNARY with output
 *
 * 3) Three operand: state_configure(cb_a, cb_b, cb_out)
 *    - Handles full pipeline operations (BINARY, MATMUL, REDUCE with explicit output buffer)
 *    - Reconfigures all three registers: srcA, srcB, and pack
 *
 * Each operation type uses compile-time dispatch (if constexpr) to call the appropriate
 * reconfiguration helper, minimizing runtime overhead while maintaining flexibility.
 */

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

/*
 *  *----------------------------------------------------------------------------------------------------*
 *  | Reconfiguration util functions, called from state_configure and used for specific reconfigurations |
 *  | based on global state                                                                              |
 *  *----------------------------------------------------------------------------------------------------*
 */

ALWI void reconfigure_single_operand(uint32_t cb, bool is_srcA = true) {
    if (is_srcA) {
        reconfig_data_format_srca(g_state_srca_cb, cb);
        g_state_srca_cb = cb;
        return;
    } else {
        reconfig_data_format_srcb(g_state_srcb_cb, cb);
        g_state_srcb_cb = cb;
    }
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

/*
 *  *-------------------------------*
 *  |  Operation-specific handlers  |
 *  *-------------------------------*
 */

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

/*
 *  *---------------------------------*
 *  |  State tracking main interface  |
 *  *---------------------------------*
 */

/**
 * UNARY, COPY, TRANSPOSE, TILIZE operations
 * Dispatches to operation-specific implementation
 */
template <Operation::Type op_type, bool to_from_int8 = false>
ALWI void state_configure(uint32_t cb) {
    static_assert(
        op_type == Operation::UNARY || op_type == Operation::COPY || op_type == Operation::TRANSPOSE ||
            op_type == Operation::PACK || op_type == Operation::TILIZE || op_type == Operation::UNTILIZE,
        "Invalid operation type for single-operand state_configure");

    if constexpr (op_type == Operation::UNARY) {
        reconfigure_single_operand(cb);
    } else if constexpr (op_type == Operation::COPY) {
        reconfigure_single_operand(cb);
    } else if constexpr (op_type == Operation::TRANSPOSE) {
        reconfigure_single_operand(cb);
    } else if constexpr (op_type == Operation::PACK) {
        reconfigure_pack_operand(cb);
    } else if constexpr (op_type == Operation::TILIZE) {
        reconfigure_single_operand(cb);
    } else if constexpr (op_type == Operation::UNTILIZE) {
        reconfigure_single_operand(cb);
    }
}

/**
 * BINARY, MATMUL, REDUCE, UNARY, TILIZE operations
 * Dispatches to operation-specific implementation
 */
template <Operation::Type op_type, bool to_from_int8 = false>
ALWI void state_configure(uint32_t cb_a, uint32_t cb_b) {
    static_assert(
        op_type == Operation::BINARY || op_type == Operation::MATMUL || op_type == Operation::REDUCE ||
            op_type == Operation::UNARY || op_type == Operation::TILIZE || op_type == Operation::UNTILIZE ||
            op_type == Operation::TRANSPOSE,
        "Invalid operation type for dual-operand state_configure");

    if constexpr (op_type == Operation::BINARY) {
        reconfigure_dual_operand(cb_a, cb_b);
    } else if constexpr (op_type == Operation::MATMUL) {
        reconfigure_dual_operand(cb_b, cb_a);  // SrcA and SrcB are swapped for matmul
    } else if constexpr (op_type == Operation::REDUCE) {
        reconfigure_dual_operand(cb_a, cb_b);
    } else if constexpr (op_type == Operation::UNARY) {
        reconfigure_single_operand(cb_a);
        reconfigure_pack_operand(cb_b);
    } else if constexpr (op_type == Operation::TILIZE) {
        reconfigure_single_operand(cb_a);
        reconfigure_pack_operand(cb_b);
    } else if constexpr (op_type == Operation::UNTILIZE) {
        reconfigure_single_operand(cb_a);
        reconfigure_pack_operand(cb_b);
    } else if constexpr (op_type == Operation::TRANSPOSE) {
        reconfigure_single_operand(cb_a);
        reconfigure_pack_operand(cb_b);
    }
}

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
        reconfig_all_operands(cb_a, cb_b, cb_out);
    } else if constexpr (op_type == Operation::MATMUL) {
        reconfig_all_operands(cb_b, cb_a, cb_out);  // SrcA and SrcB are swapped for matmul
    } else if constexpr (op_type == Operation::REDUCE) {
        reconfig_all_operands(cb_a, cb_b, cb_out);
    }
}

/*
 *  *------------------------------------*
 *  | State tracking getters and setters |
 *  *------------------------------------*
 */

/*
 * @brief Set current global states for srcA, srcB, and pack. Used in common intialization functions.
 */
ALWI void set_g_states(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
    g_state_srca_cb = cb_a;
    g_state_srcb_cb = cb_b;
    g_state_pack_cb = cb_out;
}

/**
 * @brief Set current srcA and srcB state. Used in common intialization functions.
 */
ALWI void set_g_srca_srcb(uint32_t cb_a, uint32_t cb_b) {
    g_state_srca_cb = cb_a;
    g_state_srcb_cb = cb_b;
}

/**
 * @brief Set current srcA state (for debugging)
 */
ALWI void set_g_srca(uint32_t cb_a) { g_state_srca_cb = cb_a; }

/**
 * @brief Set current srcB state (for debugging)
 */
ALWI void set_g_srcb(uint32_t cb_b) { g_state_srcb_cb = cb_b; }

/**
 * @brief Set current pack state (for debugging)
 */
ALWI void set_g_pack(uint32_t cb_out) { g_state_pack_cb = cb_out; }

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

}  // namespace ckernel
