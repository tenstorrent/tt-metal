// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bitset>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

// #include "llk_defs.h"

namespace llk::san
{

// --------------------
// This is the closest i've came to becoming a professional magician.
// I have a few more tricks up my sleeve, but they are not for the faint of heart.

namespace detail
{

template <typename...>
inline constexpr bool always_false_v = false;

template <typename T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename T, typename... Pack>
inline constexpr std::size_t count_of_v = (std::size_t {0} + ... + std::size_t {std::is_same_v<T, Pack>});

} // namespace detail

enum class Thread : std::uint32_t
{
    TRISC0 = 0,
    TRISC1 = 1,
    TRISC2 = 2,
    TRISC3 = 3
};

enum class Exu : std::uint32_t
{
    Unpack = 0,
    Fpu    = 1,
    Sfpu   = 2,
    Pack   = 3
};

enum class Hoistable : std::uint32_t
{
    No  = 0,
    Yes = 1
};

// --------------------
// Operand<Exu>
// --------------------

/**
 * @brief Type that represents Operand state for a specific execution unit.
 *
 * State is specified by specializing Operand for a specific Exu.
 *
 * @tparam E The execution unit type.
 */
template <Exu E>
struct Operand
{
    static constexpr Exu exu()
    {
        return E;
    }
};

namespace detail
{

/**
 * @brief Trait to check if G represents Operand state for an execution unit.
 *
 * Checks if G is a specialization of Operand<Exu>.
 * Extracts the Exu type from the Operand specialization.
 *
 * @tparam G The group type to check.
 */
template <typename G>
struct operand_of : std::false_type
{
};

template <Exu E>
struct operand_of<Operand<E>> : std::true_type
{
    static constexpr Exu exu = E;
};

template <typename T>
inline constexpr bool is_operand_v = operand_of<detail::remove_cvref_t<T>>::value;

} // namespace detail

// ---------------------------------
// Operation<Exu, Hoistable>
// ---------------------------------

/**
 * @brief Base class for Operation state.
 *
 * Operation state is specified by deriving from Operation.
 *
 * @tparam E Exu that the operation is native to.
 * @tparam H Indicates whether the operation is hoistable.
 */
template <Exu E, Hoistable H>
struct Operation
{
    static constexpr Exu exu()
    {
        return E;
    }

    static constexpr Hoistable hoistable()
    {
        return H;
    }
};

namespace detail
{

/**
 * @brief Check if G represents Operation state for an execution unit.
 *
 * Checks if G is derived from Operation<Exu, Hoistable>.
 *
 * @tparam G Group type to check.
 */
template <typename G, typename = void>
struct operation_of : std::false_type
{
};

template <typename G>
struct operation_of<G, std::void_t<decltype(static_cast<const Operation<G::exu(), G::hoistable()>*>(std::declval<G*>()))>> : std::true_type
{
    static constexpr Exu exu             = G::exu();
    static constexpr Hoistable hoistable = G::hoistable();
};

template <typename T>
inline constexpr bool is_operation_v = operation_of<detail::remove_cvref_t<T>>::value;

/**
 * @brief Check if G is an Operation of Exu e.
 *
 * Total: false for a group that is not an Operation at all, rather than ill-formed.
 */
template <typename G>
constexpr bool is_operation_of_exu(Exu e)
{
    if constexpr (is_operation_v<G>)
    {
        return operation_of<G>::exu == e;
    }
    else
    {
        return false;
    }
}

} // namespace detail

// -----------------
// OperationExtended
// -----------------

enum class OperationStatus : std::uint32_t
{
    Initialized,
    Executed,
    Uninitialized
};

/**
 * @brief Extended operation state (includes status and the operand state dependencies).
 */
template <typename G>
struct OperationExtended
{
    OperationStatus status;
    typename G::Struct state;
    typename Operand<G::exu()>::Struct snap;
};

// -------------
// OperationList
// -------------

/**
 * @brief Container for Operation state types
 *
 * Currently used to group the operations of a single EXU.
 * @tparam Ops The operation types in the list.
 */
template <typename... Ops>
struct OperationList
{
    using Variant = std::variant<std::monostate, OperationExtended<Ops>...>;

    template <typename Op>
    static constexpr bool contains = (std::is_same_v<Op, Ops> || ...);
};

namespace detail
{

/**
 * @brief Check if L is OperationList<Ops...> for some Ops.
 */
template <typename L>
inline constexpr bool is_operation_list_v = false;

template <typename... Ops>
inline constexpr bool is_operation_list_v<OperationList<Ops...>> = true;

} // namespace detail

// -------------
// ExuOperations
// -------------

/**
 * @brief Container for the operations of a specific execution unit.
 *
 * Specialize ExuOperations for an Exu to list the operations it supports.
 *
 * @tparam E The execution unit.
 */
template <Exu E>
struct ExuOperations
{
    using type = OperationList<>;
};

namespace detail
{

enum class ListDefect : std::uint32_t
{
    None,
    NonOperation,
    Duplicate,
    WrongExu
};

template <Exu E, typename... Ops>
constexpr ListDefect list_defect(OperationList<Ops...>)
{
    if constexpr (!(is_operation_v<Ops> && ...))
    {
        return ListDefect::NonOperation;
    }
    else if constexpr (!((count_of_v<Ops, Ops...> == 1) && ...))
    {
        return ListDefect::Duplicate;
    }
    else if constexpr (!(is_operation_of_exu<Ops>(E) && ...))
    {
        return ListDefect::WrongExu;
    }
    else
    {
        return ListDefect::None;
    }
}

} // namespace detail

// --------------
// OperationUnion
// --------------

/**
 * @brief Container for the operations of a specific execution unit.
 *
 * Validates the operation list for an execution unit and provides a Struct type for storage.
 *
 * @tparam E The execution unit.
 */
template <Exu E>
struct OperationUnion
{
    using Declared = typename ExuOperations<E>::type;

    static constexpr bool is_list = detail::is_operation_list_v<Declared>;

    static constexpr detail::ListDefect defect = detail::list_defect<E>(std::conditional_t<is_list, Declared, OperationList<>> {});

    static_assert(is_list, "ExuOperations<E>::type must be an OperationList<Ops...>.");
    static_assert(defect != detail::ListDefect::NonOperation, "ExuOperations may only list Operation<Exu, Hoistable> derivations.");
    static_assert(defect != detail::ListDefect::Duplicate, "ExuOperations must not list the same Operation twice.");
    static_assert(defect != detail::ListDefect::WrongExu, "ExuOperations may only list Operations of its own Exu.");

    using List   = std::conditional_t<is_list && defect == detail::ListDefect::None, Declared, OperationList<>>;
    using Struct = typename List::Variant;
};

// -----------------------
// StateField<Group, Type>
// -----------------------

/**
 * @brief Field of state group G of type T.
 *
 * Create a field by inheriting from StateField<Group, Type> inside the Group.
 *
 * @tparam G Group the field belongs to.
 * @tparam T Type of the field.
 */
template <typename G, typename T, typename Enable = void>
class StateField
{
public:
    // ----- invalid type escape hatch -----
    static_assert(detail::always_false_v<G>, "StateField only accepts Operand<Group> or Operation<Group, Hoistable> as Group.");
};

template <typename G, typename T>
class StateField<G, T, std::enable_if_t<detail::is_operand_v<G> || detail::is_operation_v<G>>>
{
public:
    using Group = G;
    using Type  = T;

    static constexpr std::size_t size()
    {
        return sizeof(Type);
    }

    static constexpr std::size_t align()
    {
        return alignof(Type);
    }
};

namespace detail
{

/**
 * @brief Type trait to check if F is a StateField<Group, Type>.
 *
 * Checks if F is derived from StateField<Group, Type> for some Group and Type.
 * Extracts the Group and Type.
 *
 * @tparam F Type to check.
 */
template <typename F, typename = void>
struct extract_field_base : std::false_type
{
    // ----- invalid type escape hatch -----
    using Group = void;
    using Type  = void;
};

template <typename Derived>
struct extract_field_base<
    Derived,
    std::void_t<decltype(static_cast<const StateField<typename Derived::Group, typename Derived::Type>*>(std::declval<Derived*>()))>> : std::true_type
{
    using Group = typename Derived::Group;
    using Type  = typename Derived::Type;
};

template <typename F>
using field_of_t = extract_field_base<detail::remove_cvref_t<F>>;

template <typename F>
inline constexpr bool is_state_field_v = field_of_t<F>::value;

template <typename F>
using field_group_t = typename field_of_t<F>::Group;

template <typename F>
using field_type_t = typename field_of_t<F>::Type;

} // namespace detail

// ------------
// StateDiscard
// ------------

/**
 * @brief Wrapper for discarded state values.
 *
 * Explicitly discard a value of a hook
 *
 * @tparam T Type of the value to discard.
 */
template <typename T>
class StateDiscard
{
public:
    using Type = T;

    constexpr StateDiscard([[maybe_unused]] T value)
    {
    }
};

namespace detail
{

template <typename T>
struct discard_of : std::false_type
{
};

template <typename T>
struct discard_of<StateDiscard<T>> : std::true_type
{
};

template <typename T>
inline constexpr bool is_state_discard_v = discard_of<remove_cvref_t<T>>::value;

} // namespace detail

// -------------------------
// StateVal<StateField<G,T>>
// -------------------------

/**
 * @brief Wrapper for state values.
 *
 * Encapsulates a value of a state field.
 *
 * @tparam T Type of the state field.
 */
template <typename T, typename Enable = void>
class StateVal
{
public:
    // ----- invalid type escape hatch -----
    constexpr StateVal() = default;

    template <typename U>
    constexpr StateVal(U&&)
    {
    }
};

template <typename T>
class StateVal<T, std::enable_if_t<detail::is_state_field_v<T>>>
{
public:
    using Type = detail::field_type_t<T>;

    Type value;

    constexpr StateVal(Type v) : value(v)
    {
    }
};

namespace detail
{

/**
 * @brief Type trait to check if F is a StateVal<StateField<Group, Type>>.
 *
 * Checks if V is derived from StateVal<StateField<Group, Type>> for some Group and Type.
 * Extracts the Group and Type.
 *
 * @tparam V Type to check.
 */
template <typename V>
struct value_of : std::false_type
{
    // ----- invalid type escape hatch -----
    // Mirrors the specialization's member names, for the reason extract_field_base spells out.
    using Group = void;
};

template <typename F, typename E>
struct value_of<StateVal<F, E>> : std::bool_constant<is_state_field_v<F>>
{
    using Group = field_group_t<F>;
};

template <typename T>
inline constexpr bool is_state_value_v = value_of<remove_cvref_t<T>>::value;

template <typename V>
using val_group_t = typename value_of<remove_cvref_t<V>>::Group;

} // namespace detail

// -------------------------------
// StateStruct<StateField<G,T>...>
// -------------------------------

// Indirection StateStruct -> StateStructImpl used to allow for SFINAE on the parameter pack of fields

namespace detail
{

template <typename... Fs>
struct FieldList;

template <typename G, typename List, typename Enable = void>
class StateStructImpl
{
    // ----- invalid type escape hatch -----
    static_assert(detail::always_false_v<List>, "StateStruct only accepts StateField<Group, Type> types.");
};

template <typename G, typename... Fs>
class StateStructImpl<G, FieldList<Fs...>, std::enable_if_t<(is_state_field_v<Fs> && ...)>>
{
    std::tuple<field_type_t<Fs>...> values;
    std::bitset<sizeof...(Fs)> known;

public:
    using Group = G;

    static_assert(
        (std::is_trivially_copyable_v<field_type_t<Fs>> && ...), "StateField type must be trivially copyable to allow for state tracking and comparison.");

    static_assert((std::is_same_v<field_group_t<Fs>, G> && ...), "StateFields must all belong to StateStruct's Group.");

    static_assert(((count_of_v<Fs, Fs...> == 1) && ...), "StateStruct must have unique fields.");

    template <typename F>
    static constexpr bool contains()
    {
        static_assert(is_state_field_v<F>, "StateStruct::contains() only accepts StateField<Group, Type>");
        return (std::is_same_v<Fs, F> || ...);
    }

    template <typename F>
    static constexpr std::size_t index()
    {
        static_assert(contains<F>(), "Field is not a member of this StateStruct.");
        constexpr bool matches[] = {std::is_same_v<Fs, F>...};
        for (std::size_t i = 0; i < sizeof...(Fs); ++i)
        {
            if (matches[i])
            {
                return i;
            }
        }
        __builtin_unreachable();
    }

    template <typename F>
    void update(const StateVal<F>& rhs)
    {
        static_assert(contains<F>(), "Field is not a member of this StateStruct.");
        if constexpr (contains<F>())
        {
            constexpr std::size_t idx = index<F>();
            std::get<idx>(values)     = rhs.value;
            known.set(idx);
        }
    }

    template <typename F>
    bool equal(const StateVal<F>& rhs) const
    {
        static_assert(contains<F>(), "Field is not a member of this StateStruct.");
        if constexpr (contains<F>())
        {
            constexpr std::size_t idx = index<F>();
            return known.test(idx) && std::get<idx>(values) == rhs.value;
        }
        else
        {
            return false;
        }
    }
};

} // namespace detail

template <typename G, typename... Fs>
using StateStruct = detail::StateStructImpl<G, detail::FieldList<Fs...>>;

namespace detail
{

// -------------
// Thread Checks
// -------------

/**
 *  @brief Check if an Exu is native to a Thread.
 */
constexpr bool is_exu_native(Exu e, Thread t)
{
    switch (t)
    {
        case Thread::TRISC0:
            return e == Exu::Unpack;
        case Thread::TRISC1:
            return e == Exu::Fpu || e == Exu::Sfpu; // the math thread drives both
        case Thread::TRISC2:
            return e == Exu::Pack;
        case Thread::TRISC3:
            return false;
    }

    __builtin_unreachable();
}

/**
 * @brief Check if a Thread is supported by sanitizer.
 *
 * Used to prevent hooks being called from TRISC3.
 */
constexpr bool is_thread_supported(Thread t)
{
    return is_exu_native(Exu::Unpack, t) || is_exu_native(Exu::Fpu, t) || is_exu_native(Exu::Sfpu, t) || is_exu_native(Exu::Pack, t);
}

/**
 * @brief Check if an Operand is native to a Thread.
 */
template <typename G>
constexpr bool is_operand_native(Thread t)
{
    if constexpr (is_operand_v<G>)
    {
        return is_exu_native(operand_of<G>::exu, t);
    }
    else
    {
        return false;
    }
}

/**
 * @brief Check if an Operation is native to a Thread.
 */
template <typename G>
constexpr bool is_operation_native(Thread t)
{
    if constexpr (is_operation_v<G>)
    {
        return is_exu_native(operation_of<G>::exu, t);
    }
    else
    {
        return false;
    }
}

/**
 * @brief Check if an Operand belongs to the same Exu as an Operation.
 */
template <typename Op, typename G>
constexpr bool is_operand_of_operation_exu()
{
    if constexpr (is_operation_v<Op> && is_operand_v<G>)
    {
        return operand_of<G>::exu == operation_of<Op>::exu;
    }
    else
    {
        return false;
    }
}

// Asking it is not a nicety. Without this layer an unregistered operation reaches std::variant and
// fails there instead -- five errors deep inside libstdc++, about __exactly_once and alternatives,
// naming neither the operation nor the list it is missing from.
template <typename G>
constexpr bool is_operation_listed()
{
    if constexpr (is_operation_v<G>)
    {
        return OperationUnion<operation_of<G>::exu>::List::template contains<G>;
    }
    else
    {
        return false;
    }
}

// ---------------------------------------------------------------------------------------
// The guards
// ---------------------------------------------------------------------------------------

// Layered so that at most one rule can be the one that broke, and therefore at most one
// static_assert at the entry point can speak. Layers run coarsest first, so a reader gets the most
// general true statement about their mistake rather than a consequence of it.
//
// That ordering is about correctness, not tidiness: unlayered, a call carrying a bare int is also
// told that its Operation fields belong elsewhere, because the kind layer is false only because the
// parameter layer is.
//
// Each family answers with its first defect rather than a flag per rule, the same shape
// detail::list_defect uses for a registration. The order then exists in exactly one place -- the
// `if constexpr` chain -- so inserting a rule is one branch and one enumerator, and no second
// spelling of the order can drift out of step with the first.
//
// The defect is the whole interface: an entry point asks for it once and writes one flat
// static_assert per rule, `defect != <that rule>`, with no bookkeeping of its own. Each of those
// reads "this rule holds, or an earlier one already broke", which falls out of there being a single
// answer -- if an earlier layer broke, this layer's enumerator is not it, so its assert is
// vacuously satisfied.
//
// Only the conditions live here. The wording stays at each entry point: a C++17 static_assert
// message must be a literal, and each entry point accepts something different -- what configure()
// rejects, init() requires.
//
// One note on totality. The questions below must still be total, and the region above explains why,
// but the chains are no longer what depends on it: `else if constexpr` genuinely discards a later
// fold, where the old `pN && (... && ...)` chain instantiated every fold regardless. Totality stays
// load-bearing elsewhere -- list_defect asks is_operation_of_exu about entries that may not be
// operations at all -- so nothing here licenses a question that hard-errors.

// Every entry point accepts a tracked value or an explicit discard, and nothing else.
template <typename V>
constexpr bool is_admissible_parameter()
{
    return is_state_value_v<V> || is_state_discard_v<V>;
}

// -----------------------------------
// Operand family: configure() and reconfigure()
// -----------------------------------

// A StateDiscard names no group, so the group layers pass it through explicitly.
template <typename V>
constexpr bool is_operand_field()
{
    return is_state_discard_v<V> || is_operand_v<val_group_t<V>>;
}

template <Thread T, typename V>
constexpr bool is_native_field()
{
    return is_state_discard_v<V> || is_operand_native<val_group_t<V>>(T);
}

enum class OperandDefect : std::uint32_t
{
    None,
    Unsupported,
    Params,
    Kind,
    Native
};

// Every argument is an Operand field of an Exu this thread drives.
template <Thread T, typename... Vs>
constexpr OperandDefect operand_defect()
{
    if constexpr (!is_thread_supported(T))
    {
        return OperandDefect::Unsupported;
    }
    else if constexpr (!(is_admissible_parameter<Vs>() && ...))
    {
        return OperandDefect::Params;
    }
    else if constexpr (!(is_operand_field<Vs>() && ...))
    {
        return OperandDefect::Kind;
    }
    else if constexpr (!(is_native_field<T, Vs>() && ...))
    {
        return OperandDefect::Native;
    }
    else
    {
        return OperandDefect::None;
    }
}

// ---------------------------------------------------
// Operation family: init(), execute() and uninit()
// ---------------------------------------------------

// Each argument is either one of the named operation's own fields, or an Operand field of that
// operation's own Exu. The latter is admitted so a hook can restate the operand values its LLK
// function was handed: init() snapshots them and execute()/uninit() compare against the snapshot,
// which is what catches an operand value drifting between the two halves of one operation.
template <typename Op, typename V>
constexpr bool is_own_field_or_operand()
{
    return is_state_discard_v<V> || std::is_same_v<val_group_t<V>, Op> || is_operand_of_operation_exu<Op, val_group_t<V>>();
}

enum class OperationDefect : std::uint32_t
{
    None,
    Unsupported,
    NotAnOperation,
    NotNative,
    NotListed,
    Params,
    Kind
};

// The operation is named as a template argument, so it is checked once here rather than inferred
// from the arguments and then proved unambiguous. That is what removes the old "one call describes
// one operation" layer entirely -- with the operation named, two operations' fields in one call is
// simply an argument that is not this operation's.
template <typename Op, Thread T, typename... Vs>
constexpr OperationDefect operation_defect()
{
    if constexpr (!is_thread_supported(T))
    {
        return OperationDefect::Unsupported;
    }
    else if constexpr (!is_operation_v<Op>)
    {
        return OperationDefect::NotAnOperation;
    }
    else if constexpr (!is_operation_native<Op>(T))
    {
        return OperationDefect::NotNative;
    }
    else if constexpr (!is_operation_listed<Op>())
    {
        return OperationDefect::NotListed;
    }
    else if constexpr (!(is_admissible_parameter<Vs>() && ...))
    {
        return OperationDefect::Params;
    }
    else if constexpr (!(is_own_field_or_operand<Op, Vs>() && ...))
    {
        return OperationDefect::Kind;
    }
    else
    {
        return OperationDefect::None;
    }
}

} // namespace detail

// --------
// ApiClass
// --------

enum class ApiClass : std::uint32_t
{
    None,
    Configure,
    Initialize,
    Execute,
    Uninitialize,
    Reconfigure
};

// -------------
// UnwindContext
// -------------

struct UnwindContext
{
    std::uintptr_t pc = UINTPTR_MAX;
    std::uintptr_t ra = UINTPTR_MAX;

    static const UnwindContext UNKNOWN;
};

inline const UnwindContext UnwindContext::UNKNOWN {UINTPTR_MAX, UINTPTR_MAX};

// -----------------
// OperandContext<G>
// -----------------

template <Exu E>
struct OperandContext;

template <>
struct OperandContext<Exu::Unpack>
{
    UnwindContext configure_a;
    UnwindContext configure_b;
};

template <>
struct OperandContext<Exu::Fpu>
{
    UnwindContext configure;
};

template <>
struct OperandContext<Exu::Sfpu>
{
    UnwindContext configure;
};

template <>
struct OperandContext<Exu::Pack>
{
    UnwindContext configure;
};

// -------------
// ExuContext<G>
// -------------

template <typename G>
struct ExuContext
{
    std::size_t context_depth = 0;
    std::size_t silent_depth  = 0;
    UnwindContext current;
    UnwindContext previous;
    UnwindContext operation;
    OperandContext<G::exu()> operand;
};

// -------------
// ExuState<Exu>
// -------------

template <Exu E>
struct ExuState
{
    ApiClass previous;
    typename OperationUnion<E>::Struct operation;
    typename Operand<E>::Struct operand;
};

} // namespace llk::san
