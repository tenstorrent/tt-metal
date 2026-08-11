// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sanitizer/settings.h"
#include "sanitizer/types.h"

#if LLK_SAN_SETTING_HOST_DEPS
#include "sanitizer/deps/host.h"
#else
#include "sanitizer/deps/device.h"
#endif

namespace llk::san
{

namespace detail
{

/**
 * @brief The trigger's tag for the report header.
 *
 * @param trigger
 */
inline string trigger_name(const Trigger trigger)
{
    switch (trigger)
    {
        case Trigger::PEDANTIC:
            return SAN_STRING("pedantic");
        case Trigger::WARN:
            return SAN_STRING("warn    ");
        case Trigger::ERROR:
            return SAN_STRING("error   ");
        case Trigger::FAULT:
            return SAN_STRING("fault   ");
        case Trigger::INFO:
            return SAN_STRING("info    ");
        case Trigger::INTERNAL:
            return SAN_STRING("internal");
    }

    SAN_ASSERT(false, "llk::san | fault   | trigger_name() was given a Trigger it does not name");
    return SAN_STRING("<unknown>");
}

/**
 * @brief Api class name, for templating the report.
 *
 * @param api
 */
inline string api_class_name(const ApiClass api)
{
    switch (api)
    {
        case ApiClass::Configure:
            return SAN_STRING("CONFIGURE");
        case ApiClass::Reconfigure:
            return SAN_STRING("RECONFIGURE");
        case ApiClass::Initialize:
            return SAN_STRING("INIT");
        case ApiClass::Execute:
            return SAN_STRING("EXECUTE");
        case ApiClass::Uninitialize:
            return SAN_STRING("UNINIT");
        case ApiClass::None:
            break; // Not a hook, so no report names it.
    }

    SAN_ASSERT(false, "llk::san | fault   | api_class_name() was given an ApiClass no hook names");
    return SAN_STRING("<unknown>");
}

/**
 * @brief The seated operation's status, as a report spells it.
 *
 * @param status
 */
inline string operation_status_name(const OperationStatus status)
{
    switch (status)
    {
        case OperationStatus::Uninitialized:
            return SAN_STRING("UNINITIALIZED");
        case OperationStatus::Initialized:
            return SAN_STRING("INITIALIZED");
        case OperationStatus::Executed:
            return SAN_STRING("EXECUTED");
    }

    SAN_ASSERT(false, "llk::san | fault   | operation_status_name() was given an OperationStatus it does not name");
    return SAN_STRING("<unknown>");
}

/**
 * @brief Print the kernel the report belongs to.
 *
 * FULL_KERNEL_NAME comes from the build; where the build does not supply it, it reads <unknown>.
 */
inline void print_full_kernel()
{
    SAN_PRINT(
        "│\r"
        "│  ┌[ Current Kernel ]─\r"
        "│  └── {}\r",
        SAN_STRING(FULL_KERNEL_NAME));
}

/**
 * @brief Unwinds as much of the call stack as is possible.
 *
 * @param context UnwindContext to unwind from.
 */
inline void print_compute_info(const UnwindContext context)
{
    // Expected stack layout
    // [0] detail::unwind_context_read
    // [1] FunctionZone::FunctionZone
    // [2] Compute API <- first frame the user cares about
    // [3] ...
    // Discard 2 sanitizer-internal frames, print from Compute API
    SAN_PRINT("{}\r", callstack(context.pc, context.ra, 2));
}

/**
 * @brief Print the field a mismatch was found on.
 *
 * @tparam F The field being named.
 */
template <typename F>
inline void print_field()
{
    SAN_PRINT(
        "│\r"
        "│  ┌[ Mismatched field ]─\r"
        "│  └── {}\r",
        type_name<F>());
}

/**
 * @brief Error message for a field mismatch, based on the field's group.
 *
 * @tparam F The field.
 */
template <typename F>
inline string field_mismatch_text()
{
    if constexpr (is_operation_v<field_group_t<F>>)
    {
        return SAN_STRING("was given an Operation value that differs from init()");
    }
    else
    {
        return SAN_STRING("was given an Operand value that differs from the configured state");
    }
}

/**
 * @brief Check a value a hook named against the recorded state, and report when the two disagree.
 *
 * @tparam L The trigger level for this report.
 * @tparam A The class of hook that triggered the check.
 * @tparam S The recorded state's type.
 * @tparam F The field being compared.
 * @param expected The recorded state the value is held against.
 * @param provided The value the hook named.
 * @param update Where the recorded value was written.
 * @param current Where the hook was called from.
 */
template <Trigger L, ApiClass A, typename S, typename F>
inline void field_assert(const S& expected, const StateVal<F>& provided, const UnwindContext update, const UnwindContext current)
{
    if (!enabled_trigger(L) || expected.equal(provided))
    {
        return;
    }

    SAN_PRINT(
        "┌─[ llk::san ]─[ {} ]───\r"
        "│  {} {}\r",
        trigger_name(L),
        api_class_name(A),
        field_mismatch_text<F>());

    print_full_kernel();
    print_field<F>();

    if (expected.template knows<F>())
    {
        SAN_PRINT(
            "│\r"
            "│  ┌[ Recorded state ]─\r"
            "│  ├── Value ── {}\r",
            expected.template get<F>());
    }
    else
    {
        SAN_PRINT(
            "│\r"
            "│  ┌[ Recorded state ]─\r"
            "│  ├── Value ── UNKNOWN (value never recorded?)\r");
    }

    print_compute_info(update);

    SAN_PRINT(
        "│\r"
        "│  ┌[ Failed state check ]─\r"
        "│  ├── Provided value ── {}\r",
        provided.value);

    print_compute_info(current);

    SAN_PRINT("└─────────────────────────────\n");

    SAN_ASSERT(false, "llk::san | error   | state assertion, look at the sanitizer log");
}

/**
 * @brief Check that the hook named the Operation its Exu holds a record for, and report when it did not.
 *
 * @tparam L The trigger level for this report.
 * @tparam A The class of hook that triggered the check.
 * @tparam E The Exu the hook names.
 * @tparam R The type of record the hook looked up.
 * @param exu The Exu's state.
 * @param record The record the hook looked up, null when its Operation is not the seated one.
 * @return Whether the Operation is seated
 */
template <Trigger L, ApiClass A, Exu E, typename R>
inline bool operation_seated_assert(const ExuState<E>& exu, const R* record)
{
    const bool seated = record != nullptr;

    if (seated || is_exu_silent(exu) || !enabled_trigger(L))
    {
        return seated;
    }

    SAN_PRINT(
        "┌─[ llk::san ]─[ {} ]───\r"
        "│  {} was called for an Operation the Exu is not initialized for\r",
        trigger_name(L),
        api_class_name(A));

    print_full_kernel();

    SAN_PRINT(
        "│\r"
        "│  ┌[ Seated operation ]─\r"
        "│  ├── Status ── {}\r",
        operation_status_name(exu.operation.status));

    print_compute_info(exu.context.operation);

    SAN_PRINT(
        "│\r"
        "│  ┌[ Failed hook ]─\r");

    print_compute_info(exu.context.current);

    SAN_PRINT("└─────────────────────────────\n");

    SAN_ASSERT(false, "llk::san | error   | operation assertion, look at the sanitizer log");

    return false;
}

/**
 * @brief Report Operand state that moved between init() and a later hook.
 *
 * The snapshot init() took is no longer a subset of the configured state, so the operation is
 * running against state it was not set up for.
 *
 * @tparam L The trigger level; nothing is reported when it is off.
 * @tparam A The hook raising the report, named in the wording.
 * @param subset True while the snapshot still holds, which reports nothing.
 * @param update Where the operation was initialized.
 * @param current Where the hook found the change.
 */
template <Trigger L, ApiClass A>
inline void snapshot_assert(const bool subset, const UnwindContext update, const UnwindContext current)
{
    if (!enabled_trigger(L) || subset)
    {
        return;
    }

    SAN_PRINT(
        "┌─[ llk::san ]─[ {} ]───\r"
        "│  {} found the Operand state changed since init()\r",
        trigger_name(L),
        api_class_name(A));

    print_full_kernel();

    SAN_PRINT(
        "│\r"
        "│  ┌[ Operation initialized here ]─\r");

    print_compute_info(update);

    SAN_PRINT(
        "│\r"
        "│  ┌[ Failed hook ]─\r");

    print_compute_info(current);

    SAN_PRINT("└─────────────────────────────\n");

    SAN_ASSERT(false, "llk::san | error   | drift assertion, look at the sanitizer log");
}

/**
 * @brief Report an invariant the sanitizer broke on itself rather than a defect in the kernel.
 *
 * Silent zones do not gate this: a silenced kernel is not a reason to hide a bug in the sanitizer.
 * LLK_SAN_SETTING_FAULT does, and it is off by default.
 *
 * @param holds True while the invariant holds, which reports nothing.
 * @param message What the sanitizer expected of itself.
 * @param current Where it broke.
 */
inline void fault_assert(const bool holds, const string message, const UnwindContext current)
{
    if (!enabled_trigger(Trigger::FAULT) || holds)
    {
        return;
    }

    SAN_PRINT(
        "┌─[ llk::san ]─[ {} ]───\r"
        "│  {}\r",
        trigger_name(Trigger::FAULT),
        message);

    print_full_kernel();

    SAN_PRINT(
        "│\r"
        "│  ┌[ Failed hook ]─\r");

    print_compute_info(current);

    SAN_PRINT("└─────────────────────────────\n");

    SAN_ASSERT(false, "llk::san | fault   | sanitizer invariant broken, look at the sanitizer log");
}

} // namespace detail

} // namespace llk::san
