// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sanitizer/settings.h"
#include "sanitizer/types.h"

#if !defined(FULL_KERNEL_NAME)
#define FULL_KERNEL_NAME "<unknown>"
#endif

#ifdef LLK_SAN_ENABLE

#if !defined(ENABLE_LLK_ASSERT) && !defined(DEBUG_PRINT_ENABLED)
#error "llk::san | fault   | LLK_SAN_ENABLE is set but neither ENABLE_LLK_ASSERT nor DEBUG_PRINT_ENABLED is defined"
#endif

#if defined(LLK_SAN_MOCK)

// Host (lit) mocks of the device print/assert primitives.

#define NOINLINE
#define NOCLONE

struct ct_string
{
    const char* ptr;
};

#define CTSTR(literal) (ct_string {literal})

struct dp_top_callstack_t
{
    std::uintptr_t pc;
    std::uintptr_t ra;
    std::uintptr_t skip_frames;

    dp_top_callstack_t(std::uintptr_t pc, std::uintptr_t ra, std::uintptr_t skip_frames) : pc(pc), ra(ra), skip_frames(skip_frames)
    {
    }
};

#if defined(DEBUG_PRINT_ENABLED)

#define FMT_HEADER_ONLY
#include <fmt/format.h>

template <>
struct fmt::formatter<ct_string> : fmt::formatter<const char*>
{
    auto format(const ct_string s, format_context& ctx) const
    {
        return fmt::formatter<const char*>::format(s.ptr, ctx);
    }
};

template <>
struct fmt::formatter<dp_top_callstack_t>
{
    constexpr auto parse(format_parse_context& ctx)
    {
        return ctx.begin();
    }

    auto format(const dp_top_callstack_t& callstack, format_context& ctx) const
    {
        return fmt::format_to(ctx.out(), "pc=0x{:x} ra=0x{:x} skip={}", callstack.pc, callstack.ra, callstack.skip_frames);
    }
};

#define DEVICE_PRINT(...) fmt::print(__VA_ARGS__)

#endif // DEBUG_PRINT_ENABLED

// Mirrors common/llk_assert.h.
#if defined(ENABLE_LLK_ASSERT)

#include <cstdio>
#include <cstdlib>

#define LLK_ASSERT(condition, message)        \
    do                                        \
    {                                         \
        if (!(condition))                     \
        {                                     \
            std::fputs(message "\n", stderr); \
            std::abort();                     \
        }                                     \
    } while (false)

#else

#define LLK_ASSERT(condition, message) ((void)sizeof((condition)))

#endif // ENABLE_LLK_ASSERT

#else // device

#include "ckernel.h"
#include "llk_assert.h"

#if defined(DEBUG_PRINT_ENABLED)

#ifdef ENV_LLK_INFRA
#error "llk::san | fault   | DEBUG_PRINT_ENABLED is not supported in LLK INFRA, only in metal"
#endif

#include "api/debug/device_print.h"

#else

// Assert-only build: device_print.h is absent, so supply the string carrier the report code expects.
struct ct_string
{
    const char* ptr;
};

#define CTSTR(literal) (ct_string {literal})

struct dp_top_callstack_t
{
    std::uintptr_t pc;
    std::uintptr_t ra;
    std::uintptr_t skip_frames;

    dp_top_callstack_t(std::uintptr_t pc, std::uintptr_t ra, std::uintptr_t skip_frames) : pc(pc), ra(ra), skip_frames(skip_frames)
    {
    }
};

#endif // DEBUG_PRINT_ENABLED

#endif // LLK_SAN_MOCK

namespace llk::san
{

namespace detail
{

#if defined(LLK_SAN_MOCK)
// Bumped once per emitted report so host tests can count without capturing stdout.
inline unsigned mock_report_count = 0;
#endif

inline void report_tally()
{
#if defined(LLK_SAN_MOCK)
    ++mock_report_count;
#endif
}

// No print backend: the report reduces to LLK_ASSERT. Arguments are still evaluated so the
// signatures stay warning-free and identical across backends.
template <typename... As>
inline void device_print_noop(const As&...)
{
}

} // namespace detail

} // namespace llk::san

#if !defined(DEVICE_PRINT)
#define DEVICE_PRINT(...) llk::san::detail::device_print_noop(__VA_ARGS__)
#endif

namespace llk::san
{

namespace detail
{

NOINLINE NOCLONE inline ct_string trigger_name(const Trigger trigger)
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

NOINLINE NOCLONE inline ct_string operation_status_name(const OperationStatus status)
{
    switch (status)
    {
        case OperationStatus::Uninitialized:
            return CTSTR("UNINITIALIZED");
        case OperationStatus::Initialized:
            return CTSTR("INITIALIZED");
        case OperationStatus::Executed:
            return CTSTR("EXECUTED");
    }
    __builtin_unreachable();
}

NOINLINE NOCLONE inline void print_full_kernel()
{
    DEVICE_PRINT(
        "│\r"
        "│  ┌[ Current Kernel ]─\r"
        "│  └── {}\r",
        CTSTR(FULL_KERNEL_NAME));
}

NOINLINE NOCLONE inline void print_compute_info(const UnwindContext context)
{
    // sstanisic todo: skip the sanitizer-internal frames again once the context zones are ported.
    DEVICE_PRINT("{}\r", dp_top_callstack_t(context.pc, context.ra, 0));
}

// Report a tracked field whose provided value differs from the recorded one. Serves both state
// kinds: the configured Operand state and the Operation record init() wrote; the message names the
// hook and the kind.
template <Trigger L, typename S, typename F>
NOINLINE NOCLONE inline void field_assert(
    const S& expected, const StateVal<F>& provided, const ct_string message, const UnwindContext update, const UnwindContext current)
{
    if (!enabled_trigger(L) || expected.equal(provided))
    {
        return;
    }

    DEVICE_PRINT(
        "┌─[ llk::san ]─[ {} ]───\r"
        "│  {}\r",
        trigger_name(L),
        message);

    print_full_kernel();

    if (expected.template knows<F>())
    {
        DEVICE_PRINT(
            "│\r"
            "│  ┌[ Recorded state ]─\r"
            "│  ├── Value ── {}\r",
            expected.template get<F>());
    }
    else
    {
        DEVICE_PRINT(
            "│\r"
            "│  ┌[ Recorded state ]─\r"
            "│  ├── Value ── UNKNOWN (value never recorded?)\r");
    }

    print_compute_info(update);

    DEVICE_PRINT(
        "│\r"
        "│  ┌[ Failed state check ]─\r"
        "│  ├── Provided value ── {}\r",
        provided.value);

    print_compute_info(current);

    DEVICE_PRINT("└─────────────────────────────\n");

    report_tally();

    LLK_ASSERT(false, "llk::san | error   | state assertion, look at the sanitizer log");
}

// Report a hook that named an Operation its Exu holds no record for. The caller detects the
// condition (the record type is erased here); this only words it.
template <Trigger L>
NOINLINE NOCLONE inline void operation_seated_assert(
    const OperationStatus seated, const ct_string message, const UnwindContext update, const UnwindContext current)
{
    if (!enabled_trigger(L))
    {
        return;
    }

    DEVICE_PRINT(
        "┌─[ llk::san ]─[ {} ]───\r"
        "│  {}\r",
        trigger_name(L),
        message);

    print_full_kernel();

    DEVICE_PRINT(
        "│\r"
        "│  ┌[ Seated operation ]─\r"
        "│  ├── Status ── {}\r",
        operation_status_name(seated));

    print_compute_info(update);

    DEVICE_PRINT(
        "│\r"
        "│  ┌[ Failed hook ]─\r");

    print_compute_info(current);

    DEVICE_PRINT("└─────────────────────────────\n");

    report_tally();

    LLK_ASSERT(false, "llk::san | error   | operation assertion, look at the sanitizer log");
}

// Report operand state that moved between init() and a later hook (the snapshot is no longer a
// subset of the configured state).
template <Trigger L>
NOINLINE NOCLONE inline void drift_assert(const bool subset, const ct_string message, const UnwindContext update, const UnwindContext current)
{
    if (!enabled_trigger(L) || subset)
    {
        return;
    }

    DEVICE_PRINT(
        "┌─[ llk::san ]─[ {} ]───\r"
        "│  {}\r",
        trigger_name(L),
        message);

    print_full_kernel();

    DEVICE_PRINT(
        "│\r"
        "│  ┌[ Operation initialized here ]─\r");

    print_compute_info(update);

    DEVICE_PRINT(
        "│\r"
        "│  ┌[ Failed hook ]─\r");

    print_compute_info(current);

    DEVICE_PRINT("└─────────────────────────────\n");

    report_tally();

    LLK_ASSERT(false, "llk::san | error   | drift assertion, look at the sanitizer log");
}

} // namespace detail

} // namespace llk::san

// =======================================================================================
// Legacy reporting -- DISABLED
// =======================================================================================
// Written against the old state model (State<T>, OperationState, FsmState, the Operation enum).
// Kept as the reference for the FSM reporting until that port lands.
#if 0

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
static TT_ALWAYS_INLINE void operand_assert(
    const State<T> expected, const State<T> actual, ct_string message, const UnwindContext update, const UnwindContext current)
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

    LLK_ASSERT(false, "Operand assertion, look at Sanitizer log");
}

static TT_ALWAYS_INLINE ct_string _operation_status_name(const OperationStatus status)
{
    switch (status)
    {
        case OperationStatus::None:
            return CTSTR("NONE");
        case OperationStatus::Initialized:
            return CTSTR("INITIALIZED");
        case OperationStatus::Executed:
            return CTSTR("EXECUTED");
        case OperationStatus::Uninitialized:
            return CTSTR("UNINITIALIZED");
    }
    __builtin_unreachable();
}

template <Trigger level>
NOINLINE NOCLONE bool operation_assert(
    ct_string message, const OperationState& state, OperationStatus status, Operation operation, const UnwindContext update, const UnwindContext current)
{
    if (!enabled_trigger(level) || state.operation == operation)
    {
        // If the check is enabled and the operation matches, report success.
        return enabled_trigger(level) && state.operation == operation;
    }

    DEVICE_PRINT(
        "┌─[ llk::san ]─[ {} ]───\r"
        "│  {}\r",
        _trigger_name(level),
        message);

    _print_full_kernel();

    DEVICE_PRINT(
        "│\r"
        "│  ┌[ {} ]─\r"
        "│  ├── Operation X ── {}\r",
        _operation_status_name(state.status),
        state.operation);

    _print_compute_info(update);

    DEVICE_PRINT(
        "│\r"
        "│  ┌[ {} ]─\r"
        "│  ├── Operation Y ── {}\r",
        _operation_status_name(status),
        operation);

    _print_compute_info(current);

    DEVICE_PRINT("└─────────────────────────────\n");

    LLK_ASSERT(false, "Operation assertion, look at Sanitizer log");

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

    LLK_ASSERT(false, "Operation argument assertion, look at Sanitizer log");
}

static inline ct_string _fsm_state_name(const FsmStateType state)
{
    switch (state)
    {
        case FsmStateType::Initial:
            return CTSTR("INITIAL");
        case FsmStateType::Configured:
            return CTSTR("CONFIGURED");
        case FsmStateType::Initialized:
            return CTSTR("INITIALIZED");
        case FsmStateType::Executed:
            return CTSTR("EXECUTED");
        case FsmStateType::Uninitialized:
            return CTSTR("UNINITIALIZED");
        case FsmStateType::Reconfigured:
            return CTSTR("RECONFIGURED");
    }
    __builtin_unreachable();
}

NOINLINE NOCLONE void _print_fsm_transition(const FsmState current_state, const FsmState next_state, const ct_string allowed)
{
    // sstanisic todo: try to find a non-invasive way to also print the operation where relevant
    DEVICE_PRINT(
        "│\r"
        "│  ┌[ State machine ]─\r"
        "│  ├── Current state ───── {}\r"
        "│  ├── Attempted transition ─ {}\r"
        "│  └── Allowed transitions  ─ {{ {} }}\r",
        _fsm_state_name(current_state.type),
        _fsm_state_name(next_state.type),
        allowed);
}

template <Trigger level>
NOINLINE NOCLONE bool fsm_assert(
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
        return true;
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

    LLK_ASSERT(false, "FSM assertion, look at Sanitizer log");

    return false;
}

} // namespace llk::san

#endif // legacy reporting

#endif // LLK_SAN_ENABLE
