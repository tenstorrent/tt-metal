// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <initializer_list>
#include <new>
#include <type_traits>
#include <variant>

#include "sanitizer/operation.h"
#include "sanitizer/output.h"
#include "sanitizer/types.h"

namespace llk::san
{

namespace detail
{

// -------
// Defects
// -------

enum class OperandDefect : std::uint32_t
{
    None,
    NotThreadSupported,
    NotTypeParams,
    NotOperandParams,
    NotNativeParams
};

enum class OperationDefect : std::uint32_t
{
    None,
    NotThreadSupported,
    NotTypeOperation,
    NotNativeOperation,
    NotListedOperation,
    NotTypeParams,
    NotOwnOperationParams,
    NotOwnOperandParams
};

template <typename Defect>
constexpr Defect first_defect(std::initializer_list<Defect> defects)
{
    for (const Defect defect : defects)
    {
        if (defect != Defect::None)
        {
            return defect;
        }
    }

    return Defect::None;
}

// ---------------------------------------------
// Operand family: configure() and reconfigure()
// ---------------------------------------------

template <Thread T, typename V>
constexpr OperandDefect operand_parameter_defect()
{
    if constexpr (is_state_discard_v<V>)
    {
        return OperandDefect::None;
    }
    else if constexpr (!is_state_value_v<V>)
    {
        return OperandDefect::NotTypeParams;
    }
    else if constexpr (!is_operand_v<val_group_t<V>>)
    {
        return OperandDefect::NotOperandParams;
    }
    else
    {
        return is_exu_native(operand_of<val_group_t<V>>::exu, T) ? OperandDefect::None : OperandDefect::NotNativeParams;
    }
}

template <Thread T, typename... Vs>
constexpr OperandDefect operand_defect()
{
    if constexpr (!is_thread_supported(T))
    {
        return OperandDefect::NotThreadSupported;
    }
    else
    {
        return first_defect<OperandDefect>({operand_parameter_defect<T, Vs>()...});
    }
}

// -------------------------------------------------
// Operation family: init(), execute() and uninit()
// -------------------------------------------------

template <typename Op, typename V>
constexpr OperationDefect operation_parameter_defect()
{
    if constexpr (is_state_discard_v<V>)
    {
        return OperationDefect::None;
    }
    else if constexpr (!is_state_value_v<V>)
    {
        return OperationDefect::NotTypeParams;
    }
    else if constexpr (is_operation_v<val_group_t<V>>)
    {
        return std::is_same_v<val_group_t<V>, Op> ? OperationDefect::None : OperationDefect::NotOwnOperationParams;
    }
    else
    {
        return operand_of<val_group_t<V>>::exu == operation_of<Op>::exu ? OperationDefect::None : OperationDefect::NotOwnOperandParams;
    }
}

template <typename Op, Thread T, typename... Vs>
constexpr OperationDefect operation_defect()
{
    if constexpr (!is_thread_supported(T))
    {
        return OperationDefect::NotThreadSupported;
    }
    else if constexpr (!is_operation_v<Op>)
    {
        return OperationDefect::NotTypeOperation;
    }
    else if constexpr (!is_exu_native(operation_of<Op>::exu, T))
    {
        return OperationDefect::NotNativeOperation;
    }
    else if constexpr (!OperationUnion<operation_of<Op>::exu>::List::template contains<Op>())
    {
        return OperationDefect::NotListedOperation;
    }
    else
    {
        return first_defect<OperationDefect>({operation_parameter_defect<Op, Vs>()...});
    }
}

// ------------
// State access
// ------------

template <Exu E>
static inline auto& exu_state(State& sanitizer)
{
    if constexpr (E == Exu::Unpack)
    {
        return sanitizer.unpack;
    }
    else if constexpr (E == Exu::Fpu)
    {
        return sanitizer.fpu;
    }
    else if constexpr (E == Exu::Sfpu)
    {
        return sanitizer.sfpu;
    }
    else if constexpr (E == Exu::Pack)
    {
        return sanitizer.pack;
    }
    else
    {
        static_assert(always_false<E>(), "llk::san | fault   | Unsupported Exu");
    }
}

// -----------------
// Operation helpers
// -----------------

/**
 * @brief Seat Op on its Exu and return the record for its own fields.
 *
 * The record is replaced, not merged: every operation field is reset, so a field the seating hook
 * does not state reads back as unknown rather than as the previous operation's value. Status and
 * hoistable are stamped from Op at the same time.
 *
 * @tparam Op The Operation being seated.
 */
template <typename Op>
static inline typename Op::Struct& operation_set(State& sanitizer)
{
    auto& operation = exu_state<operation_of<Op>::exu>(sanitizer).operation;

    operation.status    = OperationStatus::Initialized;
    operation.hoistable = operation_of<Op>::hoistable;
    operation.snapshot  = {};

    return operation.specific.template emplace<typename Op::Struct>();
}

/**
 * @brief The seated record for Op's own fields, or null.
 *
 * Null is what a hook naming an operation other than the seated one sees, and what it sees when the
 * Exu has no operation seated at all -- the two cases a report tells apart by OperationStatus.
 *
 * @tparam Op The Operation whose record is wanted.
 */
template <typename Op>
static inline typename Op::Struct* operation_get(State& sanitizer)
{
    return std::get_if<typename Op::Struct>(&exu_state<operation_of<Op>::exu>(sanitizer).operation.specific);
}

// ---------------
// Operand context
// ---------------

/**
 * @brief Record the current site as the one that last wrote the operand slot F belongs to.
 *
 * A mismatch reported later points at this site rather than at the hook that noticed it. The unpack
 * Exu keeps one slot per operand and the others one per Exu
 *
 * @tparam F The Operand field being written.
 * @tparam E The execution unit whose context is updated.
 */
template <typename F, Exu E>
static inline void operand_context_update(ExuContext<E>& context)
{
    if constexpr (E == Exu::Unpack)
    {
        using Unpack = Operand<Exu::Unpack>;

        if constexpr (count_of_v<F, Unpack::InputFormatA, Unpack::OutputFormatA, Unpack::FaceHeightA, Unpack::NumFacesA> == 1)
        {
            context.operand.unpack_a = context.current;
        }
        else if constexpr (count_of_v<F, Unpack::InputFormatB, Unpack::OutputFormatB, Unpack::FaceHeightB, Unpack::NumFacesB> == 1)
        {
            context.operand.unpack_b = context.current;
        }

        // DestWidth32 is shared by both operands and records no site (see #47440).
    }
    else if constexpr (E == Exu::Fpu)
    {
        context.operand.fpu = context.current;
    }
    else if constexpr (E == Exu::Sfpu)
    {
        context.operand.sfpu = context.current;
    }
    else if constexpr (E == Exu::Pack)
    {
        context.operand.pack = context.current;
    }
    else
    {
        static_assert(always_false<E>(), "llk::san | fault   | Unsupported Exu");
    }
}

/**
 * @brief The site that last wrote the operand slot F belongs to.
 *
 * Read twin of operand_context_update(), which records into the same slot: one per operand on the
 * unpack Exu, one per Exu elsewhere. A field belonging to no single slot reads back as UNKNOWN.
 *
 * @tparam F The Operand field whose slot is read.
 * @tparam E The execution unit whose context is read.
 */
template <typename F, Exu E>
static inline UnwindContext operand_context_get(const ExuContext<E>& context)
{
    if constexpr (E == Exu::Unpack)
    {
        using Unpack = Operand<Exu::Unpack>;

        if constexpr (count_of_v<F, Unpack::InputFormatA, Unpack::OutputFormatA, Unpack::FaceHeightA, Unpack::NumFacesA> == 1)
        {
            return context.operand.unpack_a;
        }
        else if constexpr (count_of_v<F, Unpack::InputFormatB, Unpack::OutputFormatB, Unpack::FaceHeightB, Unpack::NumFacesB> == 1)
        {
            return context.operand.unpack_b;
        }
        else
        {
            // DestWidth32 records no site (see #47440).
            return UnwindContext::UNKNOWN;
        }
    }
    else if constexpr (E == Exu::Fpu)
    {
        return context.operand.fpu;
    }
    else if constexpr (E == Exu::Sfpu)
    {
        return context.operand.sfpu;
    }
    else if constexpr (E == Exu::Pack)
    {
        return context.operand.pack;
    }
    else
    {
        static_assert(always_false<E>(), "llk::san | fault   | Unsupported Exu");
    }
}

// --------------
// Thread context
// --------------

/**
 * @brief The site of the enclosing function, as a pc/ra pair.
 */
[[gnu::always_inline]] static inline UnwindContext unwind_context_read()
{
    UnwindContext context;

#if defined(__riscv)
    asm volatile("auipc %[pc], 0\n" : [pc] "=r"(context.pc));
#endif

    context.ra = reinterpret_cast<std::uintptr_t>(__builtin_return_address(0));

    return context;
}

/**
 * @brief Push the current UnwindContext for the execution unit E.
 *
 * Nested calls are ignored, only the outermost context is recorded.
 *
 * @tparam E The execution unit whose context is entered.
 * @tparam T The thread the zone runs on.
 */
template <Exu E, Thread T>
static inline void exu_context_push(State& sanitizer, const UnwindContext& current)
{
    if constexpr (is_exu_native(E, T))
    {
        ExuContext<E>& context = exu_state<E>(sanitizer).context;

        if (context.context_depth++ == 0)
        {
            context.current = current;
        }
    }
}

/**
 * @brief Pop the current UnwindContext for the execution unit E.
 *
 * Nested calls are ignored, only the outermost context is recorded.
 *
 * @tparam E The execution unit whose context is left.
 * @tparam T The thread the zone runs on.
 */
template <Exu E, Thread T>
static inline void exu_context_pop(State& sanitizer)
{
    if constexpr (is_exu_native(E, T))
    {
        ExuContext<E>& context = exu_state<E>(sanitizer).context;

        if (--context.context_depth == 0)
        {
            context.current = UnwindContext::UNKNOWN;
        }
    }
}

/**
 * @brief Open a silenced zone on E.
 *
 * Nests, so a zone opened inside another stays silent until the outermost one closes.
 *
 * @tparam E The execution unit being silenced.
 * @tparam T The thread the zone runs on.
 */
template <Exu E, Thread T>
static inline void exu_silent_push(State& sanitizer)
{
    if constexpr (is_exu_native(E, T))
    {
        ++exu_state<E>(sanitizer).context.silent_depth;
    }
}

/**
 * @brief Close a silenced zone on E.
 *
 * @tparam E The execution unit being unsilenced.
 * @tparam T The thread the zone runs on.
 */
template <Exu E, Thread T>
static inline void exu_silent_pop(State& sanitizer)
{
    if constexpr (is_exu_native(E, T))
    {
        --exu_state<E>(sanitizer).context.silent_depth;
    }
}

// --------------------
// Field Update Wrapper
// --------------------

template <ApiClass A, typename T>
static inline void field_update(State&, StateDiscard<T>)
{
}

template <ApiClass A, typename F>
static inline void field_update(State& sanitizer, const StateVal<F>& value)
{
    constexpr bool configures  = (A == ApiClass::Configure || A == ApiClass::Reconfigure);
    constexpr bool initializes = (A == ApiClass::Initialize);

    using G = field_group_t<F>;

    if constexpr (configures && is_operand_v<G>)
    {
        constexpr Exu E  = operand_of<G>::exu;
        ExuState<E>& exu = exu_state<E>(sanitizer);

        exu.operand.update(value);
        operand_context_update<F>(exu.context);
    }
    else if constexpr (initializes && is_operand_v<G>)
    {
        constexpr Exu E  = operand_of<G>::exu;
        ExuState<E>& exu = exu_state<E>(sanitizer);

        exu.operation.snapshot.update(value);
    }
    else if constexpr (initializes && is_operation_v<G>)
    {
        auto& exu    = exu_state<operation_of<G>::exu>(sanitizer);
        auto* record = operation_get<G>(sanitizer);

        // init() seats the operation before folding over its arguments, so null is a sanitizer fault.
        fault_assert(record != nullptr, SAN_STRING("field_update() found no record for its own Operation"), exu.context.current);

        if (record != nullptr)
        {
            record->update(value);
        }
    }
    else
    {
        static_assert(always_false<F>(), "llk::san | fault   | only init() writes Operation state");
    }
}

// -------------------
// Field Check Wrapper
// -------------------

template <ApiClass A, typename T>
static inline void field_check(State&, StateDiscard<T>)
{
}

template <ApiClass A, typename F>
static inline void field_check(State& sanitizer, const StateVal<F>& value)
{
    using G = field_group_t<F>;

    if constexpr (is_operation_v<G>)
    {
        auto& exu          = exu_state<operation_of<G>::exu>(sanitizer);
        const auto* record = operation_get<G>(sanitizer);

        fault_assert(record != nullptr, SAN_STRING("field_check() found no record for its own Operation"), exu.context.current);

        if (record != nullptr && !is_exu_silent(exu))
        {
            field_assert<Trigger::ERROR, A>(*record, value, exu.context.operation, exu.context.current);
        }
    }
    else
    {
        auto& exu = exu_state<operand_of<G>::exu>(sanitizer);

        if (!is_exu_silent(exu))
        {
            field_assert<Trigger::ERROR, A>(exu.operand, value, operand_context_get<F>(exu.context), exu.context.current);
        }
    }
}

// ------------
// Entry points
// ------------

template <Exu E, Thread T>
static inline void exu_init(State& sanitizer)
{
    if constexpr (is_exu_native(E, T))
    {
        new (&exu_state<E>(sanitizer)) ExuState<E>();
    }
}

template <Thread T, typename... Vs>
static inline void configure(State& sanitizer, Vs&&... values)
{
    constexpr OperandDefect defect = operand_defect<T, Vs...>();

    static_assert(defect != OperandDefect::NotThreadSupported, "configure() is not supported on TRISC3.");
    static_assert(defect != OperandDefect::NotTypeParams, "configure() only accepts StateVal<StateField<Group, Type>> and StateDiscard<Type> arguments.");
    static_assert(
        defect != OperandDefect::NotOperandParams, "configure() only accepts Operand fields; Operation fields belong to init(), execute() and uninit().");
    static_assert(defect != OperandDefect::NotNativeParams, "configure() was given a field whose Exu this thread does not drive.");

    if constexpr (defect == OperandDefect::None)
    {
        (field_update<ApiClass::Configure>(sanitizer, values), ...);
    }
}

template <Thread T, typename... Vs>
static inline void reconfigure(State& sanitizer, Vs&&... values)
{
    constexpr OperandDefect defect = operand_defect<T, Vs...>();

    static_assert(defect != OperandDefect::NotThreadSupported, "reconfigure() is not supported on TRISC3.");
    static_assert(defect != OperandDefect::NotTypeParams, "reconfigure() only accepts StateVal<StateField<Group, Type>> and StateDiscard<Type> arguments.");
    static_assert(
        defect != OperandDefect::NotOperandParams, "reconfigure() only accepts Operand fields; Operation fields belong to init(), execute() and uninit().");
    static_assert(defect != OperandDefect::NotNativeParams, "reconfigure() was given a field whose Exu this thread does not drive.");

    if constexpr (defect == OperandDefect::None)
    {
        (field_update<ApiClass::Reconfigure>(sanitizer, values), ...);
    }
}

template <typename Op, Thread T, typename... Vs>
static inline void init(State& sanitizer, Vs&&... values)
{
    constexpr OperationDefect defect = operation_defect<Op, T, Vs...>();

    static_assert(defect != OperationDefect::NotThreadSupported, "init() is not supported on TRISC3.");
    static_assert(defect != OperationDefect::NotTypeOperation, "init() requires an Operation as its first template argument.");
    static_assert(defect != OperationDefect::NotNativeOperation, "init() was given an Operation whose Exu this thread does not drive.");
    static_assert(defect != OperationDefect::NotListedOperation, "init() was given an Operation that its Exu's OperationList does not name.");
    static_assert(defect != OperationDefect::NotTypeParams, "init() only accepts StateVal<StateField<Group, Type>> and StateDiscard<Type> arguments.");
    static_assert(defect != OperationDefect::NotOwnOperationParams, "init() only accepts its own Operation's fields.");
    static_assert(defect != OperationDefect::NotOwnOperandParams, "init() only accepts Operand fields of its own Operation's Exu.");

    if constexpr (defect == OperationDefect::None)
    {
        auto& exu = exu_state<operation_of<Op>::exu>(sanitizer);

        operation_set<Op>(sanitizer);

        (field_update<ApiClass::Initialize>(sanitizer, values), ...);

        exu.context.operation = exu.context.current;
    }
}

template <typename Op, Thread T, typename... Vs>
static inline void execute(State& sanitizer, Vs&&... values)
{
    constexpr OperationDefect defect = operation_defect<Op, T, Vs...>();

    static_assert(defect != OperationDefect::NotThreadSupported, "execute() is not supported on TRISC3.");
    static_assert(defect != OperationDefect::NotTypeOperation, "execute() requires an Operation as its first template argument.");
    static_assert(defect != OperationDefect::NotNativeOperation, "execute() was given an Operation whose Exu this thread does not drive.");
    static_assert(defect != OperationDefect::NotListedOperation, "execute() was given an Operation that its Exu's OperationList does not name.");
    static_assert(defect != OperationDefect::NotTypeParams, "execute() only accepts StateVal<StateField<Group, Type>> and StateDiscard<Type> arguments.");
    static_assert(defect != OperationDefect::NotOwnOperationParams, "execute() only accepts its own Operation's fields.");
    static_assert(defect != OperationDefect::NotOwnOperandParams, "execute() only accepts Operand fields of its own Operation's Exu.");

    if constexpr (defect == OperationDefect::None)
    {
        constexpr ApiClass hook = ApiClass::Execute;

        auto& exu = exu_state<operation_of<Op>::exu>(sanitizer);

        if (!operation_seated_assert<Trigger::ERROR, hook>(exu, operation_get<Op>(sanitizer)))
        {
            return;
        }

        (field_check<hook>(sanitizer, values), ...);

        if (!is_exu_silent(exu))
        {
            snapshot_assert<Trigger::ERROR, hook>(exu.operation.snapshot.subset_of(exu.operand), exu.context.operation, exu.context.current);
        }

        exu.operation.status = OperationStatus::Executed;
    }
}

template <typename Op, Thread T, typename... Vs>
static inline void uninit(State& sanitizer, Vs&&... values)
{
    constexpr OperationDefect defect = operation_defect<Op, T, Vs...>();

    static_assert(defect != OperationDefect::NotThreadSupported, "uninit() is not supported on TRISC3.");
    static_assert(defect != OperationDefect::NotTypeOperation, "uninit() requires an Operation as its first template argument.");
    static_assert(defect != OperationDefect::NotNativeOperation, "uninit() was given an Operation whose Exu this thread does not drive.");
    static_assert(defect != OperationDefect::NotListedOperation, "uninit() was given an Operation that its Exu's OperationList does not name.");
    static_assert(defect != OperationDefect::NotTypeParams, "uninit() only accepts StateVal<StateField<Group, Type>> and StateDiscard<Type> arguments.");
    static_assert(defect != OperationDefect::NotOwnOperationParams, "uninit() only accepts its own Operation's fields.");
    static_assert(defect != OperationDefect::NotOwnOperandParams, "uninit() only accepts Operand fields of its own Operation's Exu.");

    if constexpr (defect == OperationDefect::None)
    {
        constexpr ApiClass hook = ApiClass::Uninitialize;

        auto& exu = exu_state<operation_of<Op>::exu>(sanitizer);

        if (!operation_seated_assert<Trigger::ERROR, hook>(exu, operation_get<Op>(sanitizer)))
        {
            return;
        }

        (field_check<hook>(sanitizer, values), ...);

        if (!is_exu_silent(exu))
        {
            snapshot_assert<Trigger::ERROR, hook>(exu.operation.snapshot.subset_of(exu.operand), exu.context.operation, exu.context.current);
        }

        exu.operation.status = OperationStatus::Uninitialized;
    }
}

} // namespace detail

} // namespace llk::san
