// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"

// Constants and types needed by both SentinelCore and testing components
#define RECONFIG_NOTHING_CHANGED 0x00
#define RECONFIG_CHANGED_SRCA 0x01
#define RECONFIG_CHANGED_SRCB 0x02
#define RECONFIG_CHANGED_PACK 0x04

#ifdef TT_METAL_COMPUTE_KERNEL_SENTINEL_TESTING_ENABLED
#include "api/compute/sentinel/testing_spy.h"
#endif

namespace ckernel {

enum class Operand : uint8_t {
    SRCA = 0x1,
    SRCB = 0x2,
    PACK = 0x4,
};

// Invalid buffer index sentinel value
constexpr uint32_t INVALID_CB_INDEX = 0xFFFFFFFFU;
constexpr uint32_t INVALID_LINE_NUMBER = 0xFFFFFFFFU;

/**
 * @file sentinel_core.h
 * @brief SentinelCore class for managing state and invoking hardware reconfiguration functions.
 *
 * The SentinelCore manages state internally and conditionally calls hardware reconfiguration
 * functions based on its enabled state. When enabled, it calls actual hardware functions only
 * if state changed. When disabled, it still updates state but skips hardware calls.
 */
class SentinelCore {
public:
    SentinelCore() = default;
    ~SentinelCore() = default;

    // Deleted copy/move operations
    SentinelCore(const SentinelCore&) = delete;
    SentinelCore& operator=(const SentinelCore&) = delete;
    SentinelCore(SentinelCore&&) = delete;
    SentinelCore& operator=(SentinelCore&&) = delete;

    /**
     * @brief Enable hardware function injection.
     */
    void enable();

    /**
     * @brief Disable hardware function injection (state still updates).
     */
    void disable();

    /**
     * @brief Check if injection is enabled.
     * @return true if enabled, false otherwise
     */
    bool is_enabled() const;

    // State accessors
    uint32_t srca() const;
    uint32_t srcb() const;
    uint32_t pack() const;
    uint32_t last_call_line() const;
    void set_last_call_line(uint32_t line);
    void reset();

    // State setters (chainable)
    SentinelCore& set_srca(uint32_t cb);
    SentinelCore& set_srcb(uint32_t cb);
    SentinelCore& set_pack(uint32_t cb);

    /**
     * @brief Inject single operand reconfiguration.
     * Checks state internally and only calls hardware if state changed and enabled.
     * @tparam operand Operand to reconfigure (SRCA, SRCB, or PACK)
     * @param cb New circular buffer index
     * @param call_line Line number where reconfiguration was requested
     */
    template <Operand operand>
    void inject_single_operand(uint32_t cb);

    /**
     * @brief Inject dual operand reconfiguration.
     * Checks state internally and only calls hardware if state changed and enabled.
     * @tparam operand_a First operand to reconfigure
     * @tparam operand_b Second operand to reconfigure
     * @param cb_a New circular buffer index for operand_a
     * @param cb_b New circular buffer index for operand_b
     */
    template <Operand operand_a, Operand operand_b>
    void inject_dual_operand(uint32_t cb_a, uint32_t cb_b);

    /**
     * @brief Inject all operands reconfiguration.
     * Checks state internally and only calls hardware if state changed and enabled.
     * @param cb_a New circular buffer index for srcA
     * @param cb_b New circular buffer index for srcB
     * @param cb_out New circular buffer index for pack
     */
    void inject_all_operands(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out);

    /**
     * @brief Set the StateMonitorSpy reference (for testing).
     * @param spy Pointer to StateMonitorSpy instance
     */
#ifdef TT_METAL_COMPUTE_KERNEL_SENTINEL_TESTING_ENABLED
    void set_spy(StateMonitorSpy* spy);
#endif

private:
    uint32_t m_srca_cb = INVALID_CB_INDEX;
    uint32_t m_srcb_cb = INVALID_CB_INDEX;
    uint32_t m_pack_cb = INVALID_CB_INDEX;
    uint32_t m_last_call_line = INVALID_LINE_NUMBER;
#ifdef TT_METAL_COMPUTE_KERNEL_SENTINEL_DEFAULT_INJECTION
    bool m_enabled = true;
#else
    bool m_enabled = false;
#endif

#ifdef TT_METAL_COMPUTE_KERNEL_SENTINEL_TESTING_ENABLED
    StateMonitorSpy* m_spy = nullptr;
#endif
};

// ==================================================
// SentinelCore Implementation
// ==================================================

ALWI void SentinelCore::enable() { m_enabled = true; }

ALWI void SentinelCore::disable() { m_enabled = false; }

ALWI bool SentinelCore::is_enabled() const { return m_enabled; }

ALWI uint32_t SentinelCore::srca() const { return m_srca_cb; }

ALWI uint32_t SentinelCore::srcb() const { return m_srcb_cb; }

ALWI uint32_t SentinelCore::pack() const { return m_pack_cb; }

ALWI uint32_t SentinelCore::last_call_line() const { return m_last_call_line; }

ALWI void SentinelCore::set_last_call_line(uint32_t line) { m_last_call_line = line; }

ALWI void SentinelCore::reset() {
    m_srca_cb = INVALID_CB_INDEX;
    m_srcb_cb = INVALID_CB_INDEX;
    m_pack_cb = INVALID_CB_INDEX;
    m_last_call_line = INVALID_LINE_NUMBER;
}

ALWI SentinelCore& SentinelCore::set_srca(uint32_t cb) {
    m_srca_cb = cb;
    return *this;
}

ALWI SentinelCore& SentinelCore::set_srcb(uint32_t cb) {
    m_srcb_cb = cb;
    return *this;
}

ALWI SentinelCore& SentinelCore::set_pack(uint32_t cb) {
    m_pack_cb = cb;
    return *this;
}

#ifdef TT_METAL_COMPUTE_KERNEL_SENTINEL_TESTING_ENABLED
ALWI void SentinelCore::set_spy(StateMonitorSpy* spy) { m_spy = spy; }
#endif

template <Operand operand>
ALWI void SentinelCore::inject_single_operand(uint32_t cb) {
    if constexpr (operand == Operand::SRCA) {
        if (m_srca_cb == cb) {
            return;
        }
        if (m_enabled) {
            reconfig_data_format_srca<>(m_srca_cb, cb);
        }

        DPRINT << "reconfig_data_format_srca - ";
        m_srca_cb = cb;
    } else if constexpr (operand == Operand::SRCB) {
        if (m_srcb_cb == cb) {
            return;
        }
        if (m_enabled) {
            reconfig_data_format_srcb<>(m_srcb_cb, cb);
        }

        DPRINT << "reconfig_data_format_srcb - ";
        m_srcb_cb = cb;
    } else if constexpr (operand == Operand::PACK) {
        if (m_pack_cb == cb) {
            return;
        }
        if (m_enabled) {
            pack_reconfig_data_format(m_pack_cb, cb);
        }

        DPRINT << "pack_reconfig_data_format - ";
        m_pack_cb = cb;
    }
    if (m_enabled) {
        DPRINT << "happened on line - " << m_last_call_line << ENDL();
    } else {
        DPRINT << "should be called before line - " << m_last_call_line << ENDL();
    }

#ifdef TT_METAL_COMPUTE_KERNEL_SENTINEL_TESTING_ENABLED
    if (m_spy) {
        m_spy->record_reconfig_call(static_cast<uint8_t>(operand));
    }
#endif
}

template <Operand operand_a, Operand operand_b>
ALWI void SentinelCore::inject_dual_operand(uint32_t cb_a, uint32_t cb_b) {
    inject_single_operand<operand_a>(cb_a);
    inject_single_operand<operand_b>(cb_b);
}

ALWI void SentinelCore::inject_all_operands(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
    // Reconfigure source operands (srcA and srcB)
    inject_dual_operand<Operand::SRCA, Operand::SRCB>(cb_a, cb_b);
    // Reconfigure pack operand
    inject_single_operand<Operand::PACK>(cb_out);
}

}  // namespace ckernel
