// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"

#ifdef TT_METAL_COMPUTE_KERNEL_SENTINEL_TESTING_ENABLED
#define TEST_RECONFIG_CALLS(expected_calls) ComputeKernelSentinel::instance().verify_calls(expected_calls)
#define SET_CALLED_RECONFIG(call_bit) ComputeKernelSentinel::instance().record_reconfig_call(call_bit)
#else
#define TEST_RECONFIG_CALLS(expected_calls) false
#define SET_CALLED_RECONFIG(call_bit)
#endif

#ifdef TT_METAL_COMPUTE_KERNEL_SENTINEL_TESTING_ENABLED

namespace ckernel {

/**
 * @file testing_spy.h
 * @brief StateMonitorSpy class for tracking reconfiguration function calls during testing.
 *
 * The StateMonitorSpy maintains a record of which reconfiguration functions were called,
 * allowing tests to verify expected behavior. Only active on TRISC_PACK.
 */
class StateMonitorSpy {
public:
    StateMonitorSpy() = default;
    ~StateMonitorSpy() = default;

    // Deleted copy/move operations
    StateMonitorSpy(const StateMonitorSpy&) = delete;
    StateMonitorSpy& operator=(const StateMonitorSpy&) = delete;
    StateMonitorSpy(StateMonitorSpy&&) = delete;
    StateMonitorSpy& operator=(StateMonitorSpy&&) = delete;

    /**
     * @brief Record that a reconfiguration function was called.
     * @param call_bit Bit flag indicating which reconfig function was called (RECONFIG_CHANGED_*)
     */
    void record_reconfig_call(uint8_t call_bit);

    /**
     * @brief Verify that the recorded calls match expected calls, then reset.
     * @param expected_calls Bit flags indicating expected calls
     * @return true if calls match, false otherwise
     */
    bool verify_calls(uint8_t expected_calls);

    /**
     * @brief Reset call tracking.
     */
    void reset();

private:
    uint8_t m_reconfig_calls = RECONFIG_NOTHING_CHANGED;
};

// ==================================================
// StateMonitorSpy Implementation
// ==================================================

ALWI void StateMonitorSpy::record_reconfig_call(uint8_t call_bit) { m_reconfig_calls |= call_bit; }

ALWI bool StateMonitorSpy::verify_calls(uint8_t expected_calls) {
    bool matches = (m_reconfig_calls == expected_calls);
    reset();
    return matches;
}

ALWI void StateMonitorSpy::reset() { m_reconfig_calls = RECONFIG_NOTHING_CHANGED; }

}  // namespace ckernel

#endif  // TT_METAL_COMPUTE_KERNEL_SENTINEL_TESTING_ENABLED
