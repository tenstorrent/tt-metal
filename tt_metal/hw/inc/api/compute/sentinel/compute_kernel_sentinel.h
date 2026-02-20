// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"

#include "api/compute/sentinel/sentinel_core.h"
#include "api/compute/sentinel/testing_spy.h"

#ifdef TT_METAL_COMPUTE_KERNEL_SENTINEL_ENABLED
constexpr bool sentinel_enabled = true;
#else
constexpr bool sentinel_enabled = false;
#endif

namespace ckernel {

/**
 * @file compute_kernel_sentinel.h
 * @brief ComputeKernelSentinel singleton for state tracking and auto call injection.
 * The ComputeKernelSentinel provides a singleton interface that delegates to SentinelCore,
 * which manages states and injections internally.
 */
template <bool Enabled = sentinel_enabled>
class ComputeKernelSentinelImpl {
public:
    static ComputeKernelSentinelImpl& instance();

    // Deleted copy/move operations
    ComputeKernelSentinelImpl(const ComputeKernelSentinelImpl&) = delete;
    ComputeKernelSentinelImpl& operator=(const ComputeKernelSentinelImpl&) = delete;
    ComputeKernelSentinelImpl(ComputeKernelSentinelImpl&&) = delete;
    ComputeKernelSentinelImpl& operator=(ComputeKernelSentinelImpl&&) = delete;

    // State accessors
    uint32_t srca() const;
    uint32_t srcb() const;
    uint32_t pack() const;
    uint32_t last_call_line() const;
    void set_last_call_line(uint32_t line);
    void reset();

    // State setters (chainable)
    ComputeKernelSentinelImpl& set_srca(uint32_t cb);
    ComputeKernelSentinelImpl& set_srcb(uint32_t cb);
    ComputeKernelSentinelImpl& set_pack(uint32_t cb);

    // SentinelCore control
    void enable_injection();
    void disable_injection();
    bool is_injection_enabled() const;

    // Main reconfiguration interface
    template <Operand operand>
    void reconfigure_single_operand(uint32_t cb, uint32_t call_line);

    template <Operand operand_a, Operand operand_b>
    void reconfigure_dual_operand(uint32_t cb_a, uint32_t cb_b, uint32_t call_line);

    void reconfig_all_operands(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out, uint32_t call_line);

#ifdef TT_METAL_COMPUTE_KERNEL_SENTINEL_TESTING_ENABLED
    void record_reconfig_call(uint8_t call_bit);
    bool verify_calls(uint8_t expected_calls);
#endif

private:
    ComputeKernelSentinelImpl() = default;
    ~ComputeKernelSentinelImpl() = default;

#ifdef TT_METAL_COMPUTE_KERNEL_SENTINEL_ENABLED
    SentinelCore m_core;
#ifdef TT_METAL_COMPUTE_KERNEL_SENTINEL_TESTING_ENABLED
    StateMonitorSpy m_state_spy;
#endif
#endif
};

// Type alias for default template parameter
using ComputeKernelSentinel = ComputeKernelSentinelImpl<>;

// ==================================================
// Implementation for Enabled = true (real implementation)
// ==================================================

// Note: It has to be wrapped in TT_METAL_COMPUTE_KERNEL_SENTINEL_ENABLED to avoid compilation errors (Use of Sentinel
// Core) when sentinel is disabled.
#ifdef TT_METAL_COMPUTE_KERNEL_SENTINEL_ENABLED
template <>
ALWI ComputeKernelSentinelImpl<true>& ComputeKernelSentinelImpl<true>::instance() {
    static ComputeKernelSentinelImpl<true> inst;
#ifdef TT_METAL_COMPUTE_KERNEL_SENTINEL_TESTING_ENABLED
    static bool spy_initialized = false;
    if (!spy_initialized) {
        inst.m_core.set_spy(&inst.m_state_spy);
        spy_initialized = true;
    }
#endif
    return inst;
}

template <>
ALWI ComputeKernelSentinelImpl<true>& ComputeKernelSentinelImpl<true>::set_srca(uint32_t cb) {
    m_core.set_srca(cb);
    return *this;
}

template <>
ALWI ComputeKernelSentinelImpl<true>& ComputeKernelSentinelImpl<true>::set_srcb(uint32_t cb) {
    m_core.set_srcb(cb);
    return *this;
}

template <>
ALWI ComputeKernelSentinelImpl<true>& ComputeKernelSentinelImpl<true>::set_pack(uint32_t cb) {
    m_core.set_pack(cb);
    return *this;
}

template <>
ALWI uint32_t ComputeKernelSentinelImpl<true>::srca() const {
    return m_core.srca();
}

template <>
ALWI uint32_t ComputeKernelSentinelImpl<true>::srcb() const {
    return m_core.srcb();
}

template <>
ALWI uint32_t ComputeKernelSentinelImpl<true>::pack() const {
    return m_core.pack();
}

template <>
ALWI uint32_t ComputeKernelSentinelImpl<true>::last_call_line() const {
    return m_core.last_call_line();
}

template <>
ALWI void ComputeKernelSentinelImpl<true>::set_last_call_line(uint32_t line) {
    m_core.set_last_call_line(line);
}

template <>
ALWI void ComputeKernelSentinelImpl<true>::reset() {
    m_core.reset();
#ifdef TT_METAL_COMPUTE_KERNEL_SENTINEL_TESTING_ENABLED
    m_state_spy.reset();
#endif
}

template <>
ALWI void ComputeKernelSentinelImpl<true>::enable_injection() {
    m_core.enable();
}

template <>
ALWI void ComputeKernelSentinelImpl<true>::disable_injection() {
    m_core.disable();
}

template <>
ALWI bool ComputeKernelSentinelImpl<true>::is_injection_enabled() const {
    return m_core.is_enabled();
}

template <>
template <Operand operand>
ALWI void ComputeKernelSentinelImpl<true>::reconfigure_single_operand(uint32_t cb, uint32_t call_line) {
    m_core.set_last_call_line(call_line);
    m_core.inject_single_operand<operand>(cb);
}

template <>
template <Operand operand_a, Operand operand_b>
ALWI void ComputeKernelSentinelImpl<true>::reconfigure_dual_operand(uint32_t cb_a, uint32_t cb_b, uint32_t call_line) {
    m_core.set_last_call_line(call_line);
    m_core.inject_dual_operand<operand_a, operand_b>(cb_a, cb_b);
}

template <>
ALWI void ComputeKernelSentinelImpl<true>::reconfig_all_operands(
    uint32_t cb_a, uint32_t cb_b, uint32_t cb_out, uint32_t call_line) {
    m_core.set_last_call_line(call_line);
    m_core.inject_all_operands(cb_a, cb_b, cb_out);
}

#ifdef TT_METAL_COMPUTE_KERNEL_SENTINEL_TESTING_ENABLED
template <>
ALWI void ComputeKernelSentinelImpl<true>::record_reconfig_call(uint8_t call_bit) {
    m_state_spy.record_reconfig_call(call_bit);
}

template <>
ALWI bool ComputeKernelSentinelImpl<true>::verify_calls(uint8_t expected_calls) {
    return m_state_spy.verify_calls(expected_calls);
}
#endif
#endif  // TT_METAL_COMPUTE_KERNEL_SENTINEL_ENABLED

// ==================================================
// Implementation for Enabled = false
// ==================================================

template <>
ALWI ComputeKernelSentinelImpl<false>& ComputeKernelSentinelImpl<false>::instance() {
    static ComputeKernelSentinelImpl<false> inst;
    return inst;
}

template <>
ALWI ComputeKernelSentinelImpl<false>& ComputeKernelSentinelImpl<false>::set_srca(uint32_t cb) {
    (void)cb;
    return *this;
}

template <>
ALWI ComputeKernelSentinelImpl<false>& ComputeKernelSentinelImpl<false>::set_srcb(uint32_t cb) {
    (void)cb;
    return *this;
}

template <>
ALWI ComputeKernelSentinelImpl<false>& ComputeKernelSentinelImpl<false>::set_pack(uint32_t cb) {
    (void)cb;
    return *this;
}

template <>
ALWI uint32_t ComputeKernelSentinelImpl<false>::srca() const {
    return 0;
}

template <>
ALWI uint32_t ComputeKernelSentinelImpl<false>::srcb() const {
    return 0;
}

template <>
ALWI uint32_t ComputeKernelSentinelImpl<false>::pack() const {
    return 0;
}

template <>
ALWI uint32_t ComputeKernelSentinelImpl<false>::last_call_line() const {
    return 0;
}

template <>
ALWI void ComputeKernelSentinelImpl<false>::set_last_call_line(uint32_t line) {
    (void)line;
}

template <>
ALWI void ComputeKernelSentinelImpl<false>::reset() {}

template <>
ALWI void ComputeKernelSentinelImpl<false>::enable_injection() {}

template <>
ALWI void ComputeKernelSentinelImpl<false>::disable_injection() {}

template <>
ALWI bool ComputeKernelSentinelImpl<false>::is_injection_enabled() const {
    return false;
}

template <>
template <Operand operand>
ALWI void ComputeKernelSentinelImpl<false>::reconfigure_single_operand(uint32_t cb, uint32_t call_line) {
    (void)cb;
    (void)call_line;
}

template <>
template <Operand operand_a, Operand operand_b>
ALWI void ComputeKernelSentinelImpl<false>::reconfigure_dual_operand(uint32_t cb_a, uint32_t cb_b, uint32_t call_line) {
    (void)cb_a;
    (void)cb_b;
    (void)call_line;
}

template <>
ALWI void ComputeKernelSentinelImpl<false>::reconfig_all_operands(
    uint32_t cb_a, uint32_t cb_b, uint32_t cb_out, uint32_t call_line) {
    (void)cb_a;
    (void)cb_b;
    (void)cb_out;
    (void)call_line;
}

#ifdef TT_METAL_COMPUTE_KERNEL_SENTINEL_TESTING_ENABLED
template <>
ALWI void ComputeKernelSentinelImpl<false>::record_reconfig_call(uint8_t call_bit) {
    (void)call_bit;
}

template <>
ALWI bool ComputeKernelSentinelImpl<false>::verify_calls(uint8_t expected_calls) {
    (void)expected_calls;
    return false;
}
#endif

}  // namespace ckernel

// ==================================================
// Configuration Functions
// ==================================================

template <Operand operand = Operand::SRCA>
ALWI void state_configure(uint32_t cb, uint32_t call_line) {
    ComputeKernelSentinel::instance().reconfigure_single_operand<operand>(cb, call_line);
}

template <Operand operand_a = Operand::SRCA, Operand operand_b = Operand::SRCB>
ALWI void state_configure(uint32_t cb_a, uint32_t cb_b, uint32_t call_line) {
    ComputeKernelSentinel::instance().reconfigure_dual_operand<operand_a, operand_b>(cb_a, cb_b, call_line);
}

ALWI void state_configure(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out, uint32_t call_line) {
    ComputeKernelSentinel::instance().reconfig_all_operands(cb_a, cb_b, cb_out, call_line);
}
