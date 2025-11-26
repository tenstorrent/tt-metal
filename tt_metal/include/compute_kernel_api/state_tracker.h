// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"

/**
 * @file state_tracker.h
 * @brief Tracks compute kernel operand state to minimize redundant hardware reconfigurations.
 *
 * The StateTracker singleton maintains current circular buffer assignments for srcA, srcB,
 * and pack operands. When an operation requests a reconfiguration, the tracker compares
 * against the current state and only triggers hardware updates when necessary.
 */

#define RECONFIG_NOTHING_CHANGED 0x00
#define RECONFIG_CHANGED_SRCA 0x01
#define RECONFIG_CHANGED_SRCB 0x02
#define RECONFIG_CHANGED_PACK 0x04

namespace ckernel {

/**
 * Operand identifiers for state_configure<Operand::*>() template parameter.
 * Values correspond to RECONFIG_CHANGED_* bit flags for SET_CALLED_RECONFIG tracking.
 */
enum class Operand : uint8_t {
    SRCA = 0x1,
    SRCB = 0x2,
    PACK = 0x4,
};

// Invalid buffer index sentinel value
constexpr uint32_t INVALID_CB_INDEX = 0xFFFFFFFFU;

/**
 * @brief Singleton class for tracking and managing compute kernel operand state.
 *
 * Maintains current circular buffer assignments for srcA, srcB, and pack operands,
 * and handles reconfiguration when operands change to minimize redundant hardware updates.
 */
class StateTracker {
public:
    // Singleton access
    static StateTracker& instance();

    // Deleted copy/move operations for singleton
    StateTracker(const StateTracker&) = delete;
    StateTracker& operator=(const StateTracker&) = delete;
    StateTracker(StateTracker&&) = delete;
    StateTracker& operator=(StateTracker&&) = delete;

    // Getters
    uint32_t srca() const;
    uint32_t srcb() const;
    uint32_t pack() const;

    // Setters (chainable)
    StateTracker& set_srca(uint32_t cb);
    StateTracker& set_srcb(uint32_t cb);
    StateTracker& set_pack(uint32_t cb);
    StateTracker& reset();

    // Reconfiguration methods
    template <Operand operand>
    void reconfigure_single_operand(uint32_t cb);

    template <Operand operand_a, Operand operand_b>
    void reconfigure_dual_operand(uint32_t cb_a, uint32_t cb_b);

    void reconfig_all_operands(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out);

private:
    StateTracker() = default;
    ~StateTracker() = default;

    uint32_t m_srca_cb = INVALID_CB_INDEX;
    uint32_t m_srcb_cb = INVALID_CB_INDEX;
    uint32_t m_pack_cb = INVALID_CB_INDEX;
};

#ifdef TT_METAL_STATE_TRACKER_TESTING_ENABLED
/**
 * @brief Singleton spy class for testing reconfiguration call patterns.
 *
 * Tracks which hardware reconfiguration functions are invoked during compute kernel
 * execution, enabling verification of optimization effectiveness in tests.
 */
class StateTrackerSpy {
public:
    // Singleton access
    static StateTrackerSpy& instance() {
        static StateTrackerSpy spy;
        return spy;
    }

    // Deleted copy/move operations for singleton
    StateTrackerSpy(const StateTrackerSpy&) = delete;
    StateTrackerSpy& operator=(const StateTrackerSpy&) = delete;
    StateTrackerSpy(StateTrackerSpy&&) = delete;
    StateTrackerSpy& operator=(StateTrackerSpy&&) = delete;

    /**
     * @brief Record that a reconfiguration function was called.
     * @param call_bit Bit flag indicating which reconfig function was invoked.
     */
    ALWI void record_reconfig_call(uint8_t call_bit) { m_reconfig_calls |= call_bit; }

    /**
     * @brief Verify the recorded reconfig calls match expected pattern.
     * @param expected_calls Expected bit pattern of reconfig calls.
     * @return true if calls match expectations, false otherwise.
     * @note Automatically resets call tracking after verification.
     */
    ALWI bool verify_calls(uint8_t expected_calls) {
        constexpr uint8_t ALL_OPERANDS_MASK = static_cast<uint8_t>(Operand::SRCA) |
                                              static_cast<uint8_t>(Operand::SRCB) | static_cast<uint8_t>(Operand::PACK);

        bool result = (m_reconfig_calls & ALL_OPERANDS_MASK) == (expected_calls & ALL_OPERANDS_MASK);
        m_reconfig_calls = 0;  // Reset after verification
        return result;
    }

    /**
     * @brief Reset call tracking to initial state.
     */
    ALWI void reset() { m_reconfig_calls = 0; }

private:
    StateTrackerSpy() = default;
    ~StateTrackerSpy() = default;

    /**
     * Bit flags tracking which reconfig functions were called:
     * - Bit 0 (0x01): reconfig_data_format_srca
     * - Bit 1 (0x02): reconfig_data_format_srcb
     * - Bit 2 (0x04): pack_reconfig_data_format
     */
    uint8_t m_reconfig_calls = 0;
};

// State tracker tests run on the packer TRISC only. For other TRISCs (math/unpack),
// TEST_RECONFIG_CALLS returns false to point out misuse of the state tracker.
#ifdef TRISC_PACK
#define TEST_RECONFIG_CALLS(expected_calls) StateTrackerSpy::instance().verify_calls(expected_calls)
#define SET_CALLED_RECONFIG(call_bit) StateTrackerSpy::instance().record_reconfig_call(call_bit)
#else
#define TEST_RECONFIG_CALLS(expected_calls) true
#define SET_CALLED_RECONFIG(call_bit)
#endif
#else
#define TEST_RECONFIG_CALLS(expected_calls) false
#define SET_CALLED_RECONFIG(call_bit)
#endif

// Forward declarations to avoid circular dependency with pack.h
ALWI void pack_reconfig_data_format(uint32_t new_cb_id);
ALWI void pack_reconfig_data_format(uint32_t old_cb_id, uint32_t new_cb_id);

// Forward declarations to avoid circular dependency with reconfig_data_format.h
template <bool to_from_int8>
ALWI void reconfig_data_format_srca(const uint32_t srca_old_operand, const uint32_t srca_new_operand);
template <bool to_from_int8>
ALWI void reconfig_data_format_srcb(const uint32_t srcb_old_operand, const uint32_t srcb_new_operand);

// ==================================================
// StateTracker Implementation
// ==================================================

ALWI StateTracker& StateTracker::instance() {
    static StateTracker s_instance;
    return s_instance;
}

ALWI uint32_t StateTracker::srca() const { return m_srca_cb; }
ALWI uint32_t StateTracker::srcb() const { return m_srcb_cb; }
ALWI uint32_t StateTracker::pack() const { return m_pack_cb; }

ALWI StateTracker& StateTracker::set_srca(uint32_t cb) {
    m_srca_cb = cb;
    return *this;
}

ALWI StateTracker& StateTracker::set_srcb(uint32_t cb) {
    m_srcb_cb = cb;
    return *this;
}

ALWI StateTracker& StateTracker::set_pack(uint32_t cb) {
    m_pack_cb = cb;
    return *this;
}

ALWI StateTracker& StateTracker::reset() {
    m_srca_cb = INVALID_CB_INDEX;
    m_srcb_cb = INVALID_CB_INDEX;
    m_pack_cb = INVALID_CB_INDEX;
    return *this;
}

template <Operand operand>
ALWI void StateTracker::reconfigure_single_operand(uint32_t cb) {
    if constexpr (operand == Operand::SRCA) {
        if (m_srca_cb == cb) {
            return;
        }
        reconfig_data_format_srca<false>(m_srca_cb, cb);
    } else if constexpr (operand == Operand::SRCB) {
        if (m_srcb_cb == cb) {
            return;
        }
        reconfig_data_format_srcb<false>(m_srcb_cb, cb);
    } else if constexpr (operand == Operand::PACK) {
        if (m_pack_cb == cb) {
            return;
        }
        pack_reconfig_data_format(m_pack_cb, cb);
    }
    SET_CALLED_RECONFIG(static_cast<uint8_t>(operand));
}

template <Operand operand_a, Operand operand_b>
ALWI void StateTracker::reconfigure_dual_operand(uint32_t cb_a, uint32_t cb_b) {
    reconfigure_single_operand<operand_a>(cb_a);
    reconfigure_single_operand<operand_b>(cb_b);
}

ALWI void StateTracker::reconfig_all_operands(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
    // Reconfigure source operands (srcA and srcB)
    reconfigure_dual_operand<Operand::SRCA, Operand::SRCB>(cb_a, cb_b);
    // Reconfigure pack operand
    reconfigure_single_operand<Operand::PACK>(cb_out);
}

/*
 *  *---------------------------------*
 *  |  State tracking main interface  |
 *  *---------------------------------*
 *
 * State Tracking Overview:
 *
 * The state_configure functions are organized into three overloads based on operand count:
 *   1) Single operand: state_configure(cb)
 *   2) Dual operand: state_configure(cb_a, cb_b)
 *   3) Three operand: state_configure(cb_a, cb_b, cb_out)
 */

/**
 * UNARY, COPY, TRANSPOSE, TILIZE operations
 */
template <Operand operand = Operand::SRCA>
ALWI void state_configure(uint32_t cb) {
    auto& tracker = StateTracker::instance();
    tracker.reconfigure_single_operand<operand>(cb);
}

/**
 * BINARY, MATMUL, REDUCE, UNARY, TILIZE operations
 */
template <Operand operand_a = Operand::SRCA, Operand operand_b = Operand::SRCB>
ALWI void state_configure(uint32_t cb_a, uint32_t cb_b) {
    auto& tracker = StateTracker::instance();
    tracker.reconfigure_dual_operand<operand_a, operand_b>(cb_a, cb_b);
}

/**
 * BINARY, MATMUL, REDUCE with output
 */
ALWI void state_configure(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
    auto& tracker = StateTracker::instance();
    tracker.reconfig_all_operands(cb_a, cb_b, cb_out);
}

}  // namespace ckernel
