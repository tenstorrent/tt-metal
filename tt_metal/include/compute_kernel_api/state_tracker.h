// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "reconfig_data_format.h"
#include "tt_metal/include/compute_kernel_api/common_globals.h"

#define RECONFIG_NOTHING_CHANGED 0x00
#define RECONFIG_CHANGED_SRCA 0x01
#define RECONFIG_CHANGED_SRCB 0x02
#define RECONFIG_CHANGED_PACK 0x04

#ifdef TT_METAL_STATE_TRACKER_TESTING_ENABLED
#define TEST_RECONFIG_CALLS(expected_calls) reconfigure_tester.testReconfigureCalls(expected_calls)
#define SET_CALLED_RECONFIG(call_bit) reconfigure_tester.reconfigCalls |= (call_bit)
#else
#define TEST_RECONFIG_CALLS(expected_calls) false
#define SET_CALLED_RECONFIG(call_bit)
#endif

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

struct StateTracker {
    // Circular buffer indices (12 bytes)
    uint32_t srca_cb = INVALID_CB_INDEX;
    uint32_t srcb_cb = INVALID_CB_INDEX;
    uint32_t pack_cb = INVALID_CB_INDEX;

    // Data formats and operation type (4 bytes total)
    DataFormat srca_format = DataFormat::Invalid;    // uint8_t
    DataFormat srcb_format = DataFormat::Invalid;    // uint8_t
    DataFormat pack_format = DataFormat::Invalid;    // uint8_t
    Operation::Type current_op = Operation::CUSTOM;  // uint8_t

    // Kernel execution flag (1 byte)
    bool kernel_executing = false;
};

// Global State Tracker instance
static StateTracker g_state_tracker;

// Testing interface
#ifdef TT_METAL_STATE_TRACKER_TESTING_ENABLED

// Testing-only interface to access internal state
struct StateTrackerTestInterface {
    /*
     * Lower 3 bits are used to check what reconfig calls were made
     * Deafault is 0 (no calls made)
     * Bit 0 - reconfig_data_format_srca called
     * Bit 1 - reconfig_data_format_srcb called
     * Bit 2 - pack_reconfig_data_format called
     */
    uint8_t reconfigCalls = 0;
    ALWI bool testReconfigureCalls(uint8_t expected_calls) {
        bool old_value = (reconfigCalls & 0x07) == (expected_calls & 0x07);
        reconfigCalls = 0;  // Reset after check
        return old_value;
    }
};

static StateTrackerTestInterface reconfigure_tester;

#endif

/*
 *  *----------------------------------------------------------------------------------------------------*
 *  | Reconfiguration util functions, called from state_configure and used for specific reconfigurations |
 *  | based on global state                                                                              |
 *  *----------------------------------------------------------------------------------------------------*
 */

template <bool to_from_int8 = false>
ALWI void reconfigure_single_operand(uint32_t cb, bool is_srcA = true) {
    if (is_srcA) {
        reconfig_data_format_srca<to_from_int8>(g_state_tracker.srca_cb, cb);
        g_state_tracker.srca_cb = cb;
        SET_CALLED_RECONFIG(RECONFIG_CHANGED_SRCA);
        return;
    } else {
        reconfig_data_format_srcb<to_from_int8>(g_state_tracker.srcb_cb, cb);
        g_state_tracker.srcb_cb = cb;
        SET_CALLED_RECONFIG(RECONFIG_CHANGED_SRCB);
    }
}

template <bool to_from_int8 = false>
ALWI void reconfigure_dual_operand(uint32_t cb_a, uint32_t cb_b) {
    bool srcAChanged = g_state_tracker.srca_cb != cb_a;
    bool srcBChanged = g_state_tracker.srcb_cb != cb_b;

    if (srcAChanged && srcBChanged) {
        reconfig_data_format<to_from_int8>(g_state_tracker.srca_cb, cb_a, g_state_tracker.srcb_cb, cb_b);
        g_state_tracker.srca_cb = cb_a;
        g_state_tracker.srcb_cb = cb_b;
        SET_CALLED_RECONFIG(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB);
        return;
    }
    if (srcAChanged) {
        reconfigure_single_operand<to_from_int8>(cb_a);
    }
    if (srcBChanged) {
        reconfigure_single_operand<to_from_int8>(cb_b, false);
    }
}

ALWI void reconfigure_pack_operand(uint32_t cb_out) {
    if (g_state_tracker.pack_cb != cb_out) {
        pack_reconfig_data_format(g_state_tracker.pack_cb, cb_out);
        g_state_tracker.pack_cb = cb_out;
        SET_CALLED_RECONFIG(RECONFIG_CHANGED_PACK);
    }
}

template <bool to_from_int8 = false>
ALWI void reconfig_all_operands(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
    if (g_state_tracker.srca_cb != cb_a && g_state_tracker.srcb_cb != cb_b && g_state_tracker.pack_cb != cb_out) {
        reconfig_data_format<to_from_int8>(g_state_tracker.srca_cb, cb_a, g_state_tracker.srcb_cb, cb_b);
        pack_reconfig_data_format(g_state_tracker.pack_cb, cb_out);
        g_state_tracker.srca_cb = cb_a;
        g_state_tracker.srcb_cb = cb_b;
        g_state_tracker.pack_cb = cb_out;
        SET_CALLED_RECONFIG(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB | RECONFIG_CHANGED_PACK);
        return;
    }
    reconfigure_dual_operand<to_from_int8>(cb_a, cb_b);
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
    reconfig_all_operands<to_from_int8>(cb_a, cb_b, cb_out);
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
        reconfigure_single_operand<to_from_int8>(cb);
    } else if constexpr (op_type == Operation::COPY) {
        reconfigure_single_operand<to_from_int8>(cb);
    } else if constexpr (op_type == Operation::TRANSPOSE) {
        reconfigure_single_operand<to_from_int8>(cb);
    } else if constexpr (op_type == Operation::PACK) {
        reconfigure_pack_operand(cb);
    } else if constexpr (op_type == Operation::TILIZE) {
        reconfigure_single_operand<to_from_int8>(cb);
    } else if constexpr (op_type == Operation::UNTILIZE) {
        reconfigure_single_operand<to_from_int8>(cb);
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
        reconfigure_dual_operand<to_from_int8>(cb_a, cb_b);
    } else if constexpr (op_type == Operation::MATMUL) {
        reconfigure_dual_operand<to_from_int8>(cb_b, cb_a);  // SrcA and SrcB are swapped for matmul
    } else if constexpr (op_type == Operation::REDUCE) {
        reconfigure_dual_operand<to_from_int8>(cb_a, cb_b);
    } else if constexpr (op_type == Operation::UNARY) {
        reconfigure_single_operand<to_from_int8>(cb_a);
        reconfigure_pack_operand(cb_b);
    } else if constexpr (op_type == Operation::TILIZE) {
        reconfigure_single_operand<to_from_int8>(cb_a);
        reconfigure_pack_operand(cb_b);
    } else if constexpr (op_type == Operation::UNTILIZE) {
        reconfigure_single_operand<to_from_int8>(cb_a);
        reconfigure_pack_operand(cb_b);
    } else if constexpr (op_type == Operation::TRANSPOSE) {
        reconfigure_single_operand<to_from_int8>(cb_a);
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
        reconfig_all_operands<to_from_int8>(cb_a, cb_b, cb_out);
    } else if constexpr (op_type == Operation::MATMUL) {
        reconfig_all_operands<to_from_int8>(cb_b, cb_a, cb_out);  // SrcA and SrcB are swapped for matmul
    } else if constexpr (op_type == Operation::REDUCE) {
        reconfig_all_operands<to_from_int8>(cb_a, cb_b, cb_out);
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
    g_state_tracker.srca_cb = cb_a;
    g_state_tracker.srcb_cb = cb_b;
    g_state_tracker.pack_cb = cb_out;
}

/**
 * @brief Set current srcA and srcB state. Used in common intialization functions.
 */
ALWI void set_g_srca_srcb(uint32_t cb_a, uint32_t cb_b) {
    g_state_tracker.srca_cb = cb_a;
    g_state_tracker.srcb_cb = cb_b;
}

/**
 * @brief Set current srcA state (for debugging)
 */
ALWI void set_g_srca(uint32_t cb_a) { g_state_tracker.srca_cb = cb_a; }

/**
 * @brief Set current srcB state (for debugging)
 */
ALWI void set_g_srcb(uint32_t cb_b) { g_state_tracker.srcb_cb = cb_b; }

/**
 * @brief Set current pack state (for debugging)
 */
ALWI void set_g_pack(uint32_t cb_out) { g_state_tracker.pack_cb = cb_out; }

/**
 * @brief Get current srcA state (for debugging)
 */
ALWI uint32_t get_state_srca() { return g_state_tracker.srca_cb; }

/**
 * @brief Get current srcB state (for debugging)
 */
ALWI uint32_t get_state_srcb() { return g_state_tracker.srcb_cb; }

/**
 * @brief Get current pack state (for debugging)
 */
ALWI uint32_t get_state_pack() { return g_state_tracker.pack_cb; }

/**
 * @brief Mark the beginning of kernel execution
 *
 * Call this at the start of your kernel to indicate that a kernel is now executing.
 */
ALWI void kernel_start() { g_state_tracker.kernel_executing = true; }

/**
 * @brief Mark the end of kernel execution
 *
 * Call this at the end of your kernel to indicate that the kernel has finished.
 */
ALWI void kernel_end() { g_state_tracker.kernel_executing = false; }

/**
 * @brief Check if a kernel is currently executing
 *
 * @return true if a kernel is executing, false otherwise
 */
ALWI bool is_kernel_executing() { return g_state_tracker.kernel_executing; }

/**
 * @brief Reset all global state to invalid
 *
 * Call this at kernel boundaries or when external events may have
 * changed hardware state outside the API's knowledge.
 */
ALWI void reset_state_tracker() {
    g_state_tracker.srca_cb = INVALID_CB_INDEX;
    g_state_tracker.srcb_cb = INVALID_CB_INDEX;
    g_state_tracker.pack_cb = INVALID_CB_INDEX;
    g_state_tracker.kernel_executing = false;
}

}  // namespace ckernel
