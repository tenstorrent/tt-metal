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
 *    - Handles operations that only need one input buffer
 *    - Reconfigures either srcA or pack registers depending on operation type
 *
 * 2) Dual operand: state_configure(cb_a, cb_b)
 *    - Handles operations with two buffers
 *    - Reconfigures srcA and srcB registers, or srcA and pack
 *
 * 3) Three operand: state_configure(cb_a, cb_b, cb_out)
 *    - Handles full pipeline operations
 *    - Reconfigures all three registers: srcA, srcB, and pack
 *
 * Each operation type uses compile-time dispatch (if constexpr) to call the appropriate
 * reconfiguration helper, minimizing runtime overhead while maintaining flexibility.
 */
// Forward declarations to avoid circular dependency with pack.h
ALWI void pack_reconfig_data_format(uint32_t new_cb_id);
ALWI void pack_reconfig_data_format(uint32_t old_cb_id, uint32_t new_cb_id);

/**
 * These enum values are used with state_configure<OperationType::TYPE>() to determine
 * which hardware registers need reconfiguration.
 */
enum class OperationType : uint8_t {
    UNARY,      // Single source operand (srcA): square, abs, exp, sqrt, etc.
    BINARY,     // Two source operands (srcA, srcB): add, mul, sub, broadcast ops, etc.
    MATMUL,     // Matrix multiplication (srcB, srcA)
    REDUCE,     // Reduction operations (srcA, srcB_scaler)
    PACK,       // Pack/output operations (dst)
    TILIZE,     // Tilize operations (srcA)
    UNTILIZE,   // Untilize operations (srcA)
    COPY,       // Copy/move operations (srcA)
    TRANSPOSE,  // Transpose operations (srcA)
};

// Invalid buffer index sentinel value
constexpr uint32_t INVALID_CB_INDEX = 0xFFFFFFFF;

struct StateTracker {
    uint32_t srca_cb = INVALID_CB_INDEX;
    uint32_t srcb_cb = INVALID_CB_INDEX;
    uint32_t pack_cb = INVALID_CB_INDEX;
};

// Global State Tracker instance
static StateTracker g_state_tracker;

// Testing interface
#ifdef TT_METAL_STATE_TRACKER_TESTING_ENABLED

// Testing-only interface to access internal state
struct StateTrackerTestInterface {
    /*
     * Lower 3 bits are used to check what reconfig calls were made
     * Default is 0 (no calls made)
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
        if (g_state_tracker.srca_cb == cb) {
            return;
        }
        reconfig_data_format_srca<to_from_int8>(g_state_tracker.srca_cb, cb);
        SET_CALLED_RECONFIG(RECONFIG_CHANGED_SRCA);
        return;
    } else {
        if (g_state_tracker.srcb_cb == cb) {
            return;
        }
        reconfig_data_format_srcb<to_from_int8>(g_state_tracker.srcb_cb, cb);
        SET_CALLED_RECONFIG(RECONFIG_CHANGED_SRCB);
    }
}

template <bool to_from_int8 = false>
ALWI void reconfigure_dual_operand(uint32_t cb_a, uint32_t cb_b) {
    bool srcAChanged = g_state_tracker.srca_cb != cb_a;
    bool srcBChanged = g_state_tracker.srcb_cb != cb_b;

    if (srcAChanged && srcBChanged) {
        reconfig_data_format<to_from_int8>(g_state_tracker.srca_cb, cb_a, g_state_tracker.srcb_cb, cb_b);
        SET_CALLED_RECONFIG(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB);
        return;
    }
    reconfigure_single_operand<to_from_int8>(cb_a);
    reconfigure_single_operand<to_from_int8>(cb_b, false);
}

ALWI void reconfigure_pack_operand(uint32_t cb_out) {
    if (g_state_tracker.pack_cb != cb_out) {
        pack_reconfig_data_format(g_state_tracker.pack_cb, cb_out);
        SET_CALLED_RECONFIG(RECONFIG_CHANGED_PACK);
    }
}

template <bool to_from_int8 = false>
ALWI void reconfig_all_operands(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
    if (g_state_tracker.srca_cb != cb_a && g_state_tracker.srcb_cb != cb_b && g_state_tracker.pack_cb != cb_out) {
        reconfig_data_format<to_from_int8>(g_state_tracker.srca_cb, cb_a, g_state_tracker.srcb_cb, cb_b);
        pack_reconfig_data_format(g_state_tracker.pack_cb, cb_out);
        SET_CALLED_RECONFIG(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB | RECONFIG_CHANGED_PACK);
        return;
    }
    reconfigure_dual_operand<to_from_int8>(cb_a, cb_b);
    reconfigure_pack_operand(cb_out);
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
template <OperationType op_type, bool to_from_int8 = false>
ALWI void state_configure(uint32_t cb) {
    static_assert(
        op_type == OperationType::UNARY || op_type == OperationType::COPY || op_type == OperationType::TRANSPOSE ||
            op_type == OperationType::PACK || op_type == OperationType::TILIZE || op_type == OperationType::UNTILIZE,
        "Invalid operation type for single-operand state_configure");

    if constexpr (op_type == OperationType::UNARY) {
        reconfigure_single_operand<to_from_int8>(cb);
    } else if constexpr (op_type == OperationType::COPY) {
        reconfigure_single_operand<to_from_int8>(cb);
    } else if constexpr (op_type == OperationType::TRANSPOSE) {
        reconfigure_single_operand<to_from_int8>(cb);
    } else if constexpr (op_type == OperationType::PACK) {
        reconfigure_pack_operand(cb);
    } else if constexpr (op_type == OperationType::TILIZE) {
        reconfigure_single_operand<to_from_int8>(cb);
    } else if constexpr (op_type == OperationType::UNTILIZE) {
        reconfigure_single_operand<to_from_int8>(cb);
    }
}

/**
 * BINARY, MATMUL, REDUCE, UNARY, TILIZE operations
 * Dispatches to operation-specific implementation
 */
template <OperationType op_type, bool to_from_int8 = false>
ALWI void state_configure(uint32_t cb_a, uint32_t cb_b) {
    static_assert(
        op_type == OperationType::BINARY || op_type == OperationType::MATMUL || op_type == OperationType::REDUCE ||
            op_type == OperationType::UNARY || op_type == OperationType::TILIZE || op_type == OperationType::UNTILIZE ||
            op_type == OperationType::TRANSPOSE,
        "Invalid operation type for dual-operand state_configure");

    if constexpr (op_type == OperationType::BINARY) {
        reconfigure_dual_operand<to_from_int8>(cb_a, cb_b);
    } else if constexpr (op_type == OperationType::MATMUL) {
        reconfigure_dual_operand<to_from_int8>(cb_b, cb_a);  // SrcA and SrcB are swapped for matmul
    } else if constexpr (op_type == OperationType::REDUCE) {
        reconfigure_dual_operand<to_from_int8>(cb_a, cb_b);
    } else if constexpr (op_type == OperationType::UNARY) {
        reconfigure_single_operand<to_from_int8>(cb_a);
        reconfigure_pack_operand(cb_b);
    } else if constexpr (op_type == OperationType::TILIZE) {
        reconfigure_single_operand<to_from_int8>(cb_a);
        reconfigure_pack_operand(cb_b);
    } else if constexpr (op_type == OperationType::UNTILIZE) {
        reconfigure_single_operand<to_from_int8>(cb_a);
        reconfigure_pack_operand(cb_b);
    } else if constexpr (op_type == OperationType::TRANSPOSE) {
        reconfigure_single_operand<to_from_int8>(cb_a);
        reconfigure_pack_operand(cb_b);
    }
}

/**
 * BINARY, MATMUL, REDUCE with output
 * Dispatches to operation-specific implementation
 */
template <OperationType op_type, bool to_from_int8 = false>
ALWI void state_configure(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
    static_assert(
        op_type == OperationType::BINARY || op_type == OperationType::MATMUL || op_type == OperationType::REDUCE,
        "Invalid operation type for full pipeline state_configure");

    if constexpr (op_type == OperationType::BINARY) {
        reconfig_all_operands<to_from_int8>(cb_a, cb_b, cb_out);
    } else if constexpr (op_type == OperationType::MATMUL) {
        reconfig_all_operands<to_from_int8>(cb_b, cb_a, cb_out);  // SrcA and SrcB are swapped for matmul
    } else if constexpr (op_type == OperationType::REDUCE) {
        reconfig_all_operands<to_from_int8>(cb_a, cb_b, cb_out);
    }
}

/*
 *  *------------------------------------*
 *  | State tracking getters and setters |
 *  *------------------------------------*
 */

/*
 * @brief Set current global states for srcA, srcB, and pack. Used in common initialization functions.
 */
ALWI void set_global_states(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
    g_state_tracker.srca_cb = cb_a;
    g_state_tracker.srcb_cb = cb_b;
    g_state_tracker.pack_cb = cb_out;
}

/**
 * @brief Set current srcA and srcB state. Used in common intialization functions.
 */
ALWI void set_global_srca_srcb(uint32_t cb_a, uint32_t cb_b) {
    g_state_tracker.srca_cb = cb_a;
    g_state_tracker.srcb_cb = cb_b;
}

/**
 * @brief Set current srcA state
 */
ALWI void set_global_srca(uint32_t cb_a) { g_state_tracker.srca_cb = cb_a; }

/**
 * @brief Set current srcB state
 */
ALWI void set_global_srcb(uint32_t cb_b) { g_state_tracker.srcb_cb = cb_b; }

/**
 * @brief Set current pack state
 */
ALWI void set_global_pack(uint32_t cb_out) { g_state_tracker.pack_cb = cb_out; }

/**
 * @brief Get current srcA state
 */
ALWI uint32_t get_state_srca() { return g_state_tracker.srca_cb; }

/**
 * @brief Get current srcB state
 */
ALWI uint32_t get_state_srcb() { return g_state_tracker.srcb_cb; }

/**
 * @brief Get current pack state
 */
ALWI uint32_t get_state_pack() { return g_state_tracker.pack_cb; }

/**
 * @brief Reset the state tracker to initial invalid state
 */
ALWI void reset_state_tracker() {
    g_state_tracker.srca_cb = INVALID_CB_INDEX;
    g_state_tracker.srcb_cb = INVALID_CB_INDEX;
    g_state_tracker.pack_cb = INVALID_CB_INDEX;
}

}  // namespace ckernel
