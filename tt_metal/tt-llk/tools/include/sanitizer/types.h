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

namespace detail
{

template <typename...>
constexpr bool always_false()
{
    return false;
}

template <auto...>
constexpr bool always_false()
{
    return false;
}

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

// ---------------
// OperationStatus
// ---------------

enum class OperationStatus : std::uint32_t
{
    Uninitialized,
    Initialized,
    Executed
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
    using Variant = std::variant<std::monostate, typename Ops::Struct...>;

    template <typename Op>
    static constexpr bool contains()
    {
        return (std::is_same_v<Op, Ops> || ...);
    }
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

template <Exu E, typename L>
struct list_defect;

template <Exu E, typename... Ops>
struct list_defect<E, OperationList<Ops...>>
{
    static constexpr ListDefect check()
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
};

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

    using Candidate = std::conditional_t<is_list, Declared, OperationList<>>;

    static constexpr detail::ListDefect defect = detail::list_defect<E, Candidate>::check();

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
    static_assert(detail::always_false<G>(), "StateField only accepts Operand<Group> or Operation<Group, Hoistable> as Group.");
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
    static_assert(detail::always_false<List>(), "StateStruct only accepts StateField<Group, Type> types.");
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
        __builtin_unreachable();
    }

    template <typename F>
    bool knows() const
    {
        static_assert(contains<F>(), "Field is not a member of this StateStruct.");
        if constexpr (contains<F>())
        {
            return known.test(index<F>());
        }
        __builtin_unreachable();
    }

    template <typename F>
    field_type_t<F> get() const
    {
        static_assert(contains<F>(), "Field is not a member of this StateStruct.");
        if constexpr (contains<F>())
        {
            constexpr std::size_t idx = index<F>();
            return std::get<idx>(values);
        }
        __builtin_unreachable();
    }

    bool subset_of(const StateStructImpl& other) const
    {
        return subset_of(other, std::index_sequence_for<Fs...> {});
    }

private:
    template <std::size_t... Is>
    bool subset_of(const StateStructImpl& other, std::index_sequence<Is...>) const
    {
        return ((!known.test(Is) || (other.known.test(Is) && std::get<Is>(values) == std::get<Is>(other.values))) && ...);
    }
};

} // namespace detail

template <typename G, typename... Fs>
using StateStruct = detail::StateStructImpl<G, detail::FieldList<Fs...>>;

namespace detail
{

// ------------
// Exu dispatch
// ------------

template <typename F>
constexpr void exu_dispatch(F&& f)
{
    f(std::integral_constant<Exu, Exu::Unpack> {});
    f(std::integral_constant<Exu, Exu::Fpu> {});
    f(std::integral_constant<Exu, Exu::Sfpu> {});
    f(std::integral_constant<Exu, Exu::Pack> {});
}

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
    UnwindContext unpack_a;
    UnwindContext unpack_b;
};

template <>
struct OperandContext<Exu::Fpu>
{
    UnwindContext fpu;
};

template <>
struct OperandContext<Exu::Sfpu>
{
    UnwindContext sfpu;
};

template <>
struct OperandContext<Exu::Pack>
{
    UnwindContext pack;
};

// -------------
// ExuContext<E>
// -------------

template <Exu E>
struct ExuContext
{
    std::size_t context_depth = 0;
    std::size_t silent_depth  = 0;
    UnwindContext current;
    UnwindContext operation;
    OperandContext<E> operand;
};

// --------------------
// OperationExtended<E>
// --------------------

template <Exu E>
struct OperationExtended
{
    OperationStatus status;
    Hoistable hoistable;
    typename Operand<E>::Struct snapshot;
    typename OperationUnion<E>::Struct specific;
};

// -------------
// ExuState<Exu>
// -------------

template <Exu E>
struct ExuState
{
    ApiClass previous;
    OperationExtended<E> operation;
    typename Operand<E>::Struct operand;
    ExuContext<E> context;
};

namespace detail
{

/**
 * @brief Whether E is inside a silenced zone.
 *
 * Only kernel-facing reports honour this.
 * Faults will not respect this, and will report regardless of the silent flag.
 *
 * @tparam E The execution unit being asked about.
 */
template <Exu E>
static inline bool is_exu_silent(const ExuState<E>& exu)
{
    return exu.context.silent_depth > 0;
}

} // namespace detail

} // namespace llk::san
