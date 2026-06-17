// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstring>

#include "ckernel.h"
#include "sanitizer/settings.h"
#include "sanitizer/types.h"

#if !defined(FULL_KERNEL_NAME)
#define FULL_KERNEL_NAME "<unknown>"
#endif

#ifndef LLK_SAN_ENABLE

#elif defined(LLK_SAN_SETTING_ASSERT)

#include "llk_assert.h"

#elif defined(LLK_SAN_SETTING_PRINT)

#ifdef ENV_LLK_INFRA
#error "llk::san | fault   | LLK_SAN_SETTING_PRINT is not supported in LLK INFRA, only in metal"
#endif

#include "api/debug/device_print.h"

#else
#error "llk::san | fault   | terrible failure :D"
#endif

namespace llk::san
{

NOINLINE NOCLONE ct_string _trigger_name(const Trigger trigger)
{
    switch (trigger)
    {
        case Trigger::PEDANTIC:
            return CTSTR("pedantic");
        case Trigger::WARN:
            return CTSTR("warn    ");
        case Trigger::ERROR:
            return CTSTR("error   ");
        case Trigger::FAULT:
            return CTSTR("fault   ");
        case Trigger::INFO:
            return CTSTR("info    ");
        case Trigger::INTERNAL:
            return CTSTR("internal");
    }
    __builtin_unreachable();
}

NOINLINE NOCLONE void _print_full_kernel()
{
    const ct_string kernel = CTSTR(FULL_KERNEL_NAME);
    DEVICE_PRINT(
        "│\r"
        "│  ┌[ Current Kernel ]─\r"
        "│  └── {}\r",
        kernel);
}

template <typename T>
NOINLINE NOCLONE void _print_operand_expected(const State<T> expected)
{
    if (expected.is_known())
    {
        DEVICE_PRINT(
            "│\r"
            "│  ┌[ Last time state was modified ]─\r"
            "│  ├── New state value ── {}\r",
            expected.get_underlying());
    }
    else if (expected.is_unknown())
    {
        DEVICE_PRINT(
            "│\r"
            "│  ┌[ Last time state was modified ]─\r"
            "│  ├── New state value ── UNKNOWN (value never configured?)\r");
    }
    else
    {
        __builtin_unreachable();
    }
}

template <typename T>
NOINLINE NOCLONE void _print_operand_actual(const State<T> actual)
{
    if (actual.is_known())
    {
        DEVICE_PRINT(
            "│  ┌[ Failed operand state check ]─\r"
            "│  ├── Provided state value ── {}\r",
            actual.get_underlying());
    }
    else
    {
        __builtin_unreachable();
    }
}

NOINLINE NOCLONE void _print_compute_info(const UnwindContext context)
{
    // Expected stack layout
    // [0] write_unwind_context
    // [1] thread_context_push_impl
    // [2] thread_context_push
    // [3] FunctionZone::FunctionZone
    // [4] Compute API <- first frame the user cares about
    // [5] ...
    // Discard 4 sanitizer-internal frames, print from Compute API
    DEVICE_PRINT("{}\r", dp_top_callstack_t(context.pc, context.ra, 4));
}

template <Trigger level, typename T>
TT_ALWAYS_INLINE void operand_assert(const State<T> expected, const State<T> actual, ct_string message, const UnwindContext update, const UnwindContext current)
{
    if (!enabled_trigger(level) || expected.assert_cond(actual))
    {
        return;
    }

    DEVICE_PRINT(
        "┌─[ llk::san ]─[ {} ]───\r"
        "│  {}\r",
        _trigger_name(level),
        message);

    _print_full_kernel();
    _print_operand_expected(expected);
    _print_compute_info(update);
    _print_operand_actual(actual);
    _print_compute_info(current);

    DEVICE_PRINT("└─────────────────────────────\n");
}

template <Trigger level>
NOINLINE NOCLONE bool operation_assert(const Operation expected, const Operation actual, const UnwindContext update, const UnwindContext current)
{
    if (!enabled_trigger(level) || expected == actual)
    {
        // If check is enabled and passed, return true.
        return enabled_trigger(level) && expected == actual;
    }

    DEVICE_PRINT(
        "┌─[ llk::san ]─[ {} ]───\r"
        "│  Called execute or uninit for operation that is not initialized\r",
        _trigger_name(level));

    _print_full_kernel();

    DEVICE_PRINT(
        "│\r"
        "│  ┌[ Last init call ]─\r"
        "│  ├── Operation ── {}\r",
        expected);
    _print_compute_info(update);

    DEVICE_PRINT(
        "│  ┌[ Violating execute / uninit call ]─\r"
        "│  ├── Operation ── {}\r",
        actual);
    _print_compute_info(current);

    DEVICE_PRINT("└─────────────────────────────\n");

    return false;
}

template <Trigger level>
NOINLINE NOCLONE void operation_argument_assert(
    const void* lhs, const void* rhs, size_t size, size_t idx, const UnwindContext update, const UnwindContext current)
{
    if constexpr (!enabled_trigger(level))
    {
        return;
    }

    if (std::memcmp(lhs, rhs, size) == 0)
    {
        return;
    }

    DEVICE_PRINT(
        "┌─[ llk::san ]─[ {} ]───\r"
        "│  Argument {} of llk::san::operation_init and llk::san::operation_check is mismatched\r",
        _trigger_name(level),
        idx);

    _print_full_kernel();

    DEVICE_PRINT(
        "│\r"
        "│  ┌[ llk::san::operation_init called from ]─\r");
    _print_compute_info(update);

    DEVICE_PRINT("│  ┌[ llk::san::operation_check called from ]─\r");
    _print_compute_info(current);

    DEVICE_PRINT("└─────────────────────────────\n");
}

static inline ct_string fsm_state_name(const FsmState state)
{
    switch (state)
    {
        case FsmState::Initial:
            return CTSTR("INITIAL");
        case FsmState::Configured:
            return CTSTR("CONFIGURED");
        case FsmState::Initialized:
            return CTSTR("INITIALIZED");
        case FsmState::Executed:
            return CTSTR("EXECUTED");
        case FsmState::Uninitialized:
            return CTSTR("UNINITIALIZED");
        case FsmState::Reconfigured:
            return CTSTR("RECONFIGURED");
    }
    __builtin_unreachable();
}

NOINLINE NOCLONE void _print_fsm_transition(const FsmState current_state, const FsmState next_state, const ct_string allowed)
{
    DEVICE_PRINT(
        "│\r"
        "│  ┌[ State machine ]─\r"
        "│  ├── Current state ───── {}\r"
        "│  ├── Attempted transition ─ {}\r"
        "│  └── Allowed transitions  ─ [{}]\r",
        fsm_state_name(current_state),
        fsm_state_name(next_state),
        allowed);
}

template <Trigger level>
NOINLINE NOCLONE void fsm_assert(
    bool success,
    ct_string message,
    FsmState transition_from,
    FsmState transition_to,
    ct_string transition_allowed,
    const UnwindContext update,
    const UnwindContext current)
{
    if (!enabled_trigger(level) || success)
    {
        return;
    }

    DEVICE_PRINT(
        "┌─[ llk::san ]─[ {} ]───\r"
        "│  {}\r",
        _trigger_name(level),
        message);

    _print_full_kernel();
    _print_fsm_transition(transition_from, transition_to, transition_allowed);

    DEVICE_PRINT(
        "│\r"
        "│  ┌[ Last successful transition ]─\r");
    _print_compute_info(update);

    DEVICE_PRINT("│  ┌[ Violating transition ]─\r");
    _print_compute_info(current);

    DEVICE_PRINT("└─────────────────────────────\n");
}

} // namespace llk::san
