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

#include "ckernel.h"
#include "llk_defs.h"

namespace llk::san
{

template <typename T>
class State
{
private:
    T underlying;
    ValueState state_type;

public:
    template <typename U>
    friend class State;

    // CONSTRUCTION
    // Default to UNKNOWN state because hardware is not initialized
    State() noexcept(std::is_nothrow_default_constructible_v<T>) : underlying {}, state_type(ValueState::Unknown)
    {
    }

    State(const State&) noexcept(std::is_nothrow_copy_constructible_v<T>) = default;
    State(State&&) noexcept(std::is_nothrow_move_constructible_v<T>)      = default;

    // CONVERSION
    // - llk::san::IGNORE     -> State with state_type == Ignore
    // - llk::san::UNKNOWN    -> State with state_type == Unknown
    // - other                -> State with state_type == Known (storing the value)

    // Constructor for IGNORE
    constexpr State(const Ignore&) noexcept : underlying {}, state_type(ValueState::Ignore)
    {
    }

    // Constructor for UNKNOWN
    constexpr State(const Unknown&) noexcept : underlying {}, state_type(ValueState::Unknown)
    {
    }

    // Constructor for KNOWN value
    template <typename U, typename = std::enable_if_t<is_known_value_v<U>>>
    constexpr State(U&& value) noexcept(std::is_nothrow_constructible_v<T, U&&>) : underlying(std::forward<U>(value)), state_type(ValueState::Known)
    {
    }

    // ASSIGNMENT
    // if RHS of assignment is ValueState::Ignore, noop (stays old value)
    // otherwise take the state_type and underlying of RHS

    template <typename U>
    State& operator=(const State<U>& rhs) noexcept(std::is_nothrow_copy_assignable_v<T>)
    {
        if (rhs.state_type == ValueState::Ignore)
        {
            return *this;
        }

        state_type = rhs.state_type;
        underlying = rhs.underlying;

        return *this;
    }

    template <typename U>
    State& operator=(State<U>&& rhs) noexcept(std::is_nothrow_move_assignable_v<T>)
    {
        if (rhs.state_type == ValueState::Ignore)
        {
            return *this; // No-op
        }

        state_type = rhs.state_type;
        underlying = std::move(rhs.underlying);

        return *this;
    }

    State& operator=(const State& rhs) noexcept(std::is_nothrow_copy_assignable_v<T>)
    {
        return this->template operator= <T>(rhs);
    }

    State& operator=(State&& rhs) noexcept(std::is_nothrow_move_assignable_v<T>)
    {
        return this->template operator= <T>(std::move(rhs));
    }

    // RHS of assignment is:
    // - compatible with T
    // - Unknown
    // - Ignore
    template <typename U, typename = std::enable_if_t<!is_state_v<std::decay_t<U>>>>
    State& operator=(U&& rhs) noexcept(std::is_nothrow_constructible_v<T, U&&>)
    {
        *this = State<T>(std::forward<U>(rhs));
        return *this;
    }

    constexpr bool is_known() const noexcept
    {
        return state_type == ValueState::Known;
    }

    constexpr bool is_unknown() const noexcept
    {
        return state_type == ValueState::Unknown;
    }

    constexpr bool is_ignore() const noexcept
    {
        return state_type == ValueState::Ignore;
    }

    const T& get_underlying() const
    {
        LLK_ASSERT(is_known(), "llk::san | fault    | underlying value is not known");
        return underlying;
    }

    template <typename U>
    bool assert_cond(const State<U>& rhs) const noexcept
    {
        if (is_ignore() || rhs.is_ignore())
        {
            return true;
        }
        if (is_unknown() || rhs.is_unknown())
        {
            return false;
        }
        return get_underlying() == rhs.get_underlying();
    }

    template <typename U>
    bool panic_cond(const State<U>& rhs) const noexcept
    {
        if (is_ignore() || rhs.is_ignore())
        {
            return true;
        }
        if (is_unknown() || rhs.is_unknown())
        {
            return false;
        }
        return get_underlying() != rhs.get_underlying();
    }

    template <typename U>
    void update(const State<U>& rhs) noexcept(std::is_nothrow_copy_assignable_v<T>)
    {
        *this = rhs;
    }
};

enum class StateType {
    None,
    OperandUnpack,
    OperandFpu,
    OperandSfpu,
    OperandPack,
    Operation
};

template <StateType Category, typename Group, typename T>
class StateField {
public:
    using GroupTag = Group;   // enables the reverse field -> owning-struct lookup
    using ValueType = T;

    static constexpr StateType state_type() { return Category; }

    static constexpr std::size_t size() { return sizeof(T); }
    static constexpr std::size_t align() { return alignof(T); }

};


template <typename Id>
class StateVal {
public:
    using StateIdType = Id;
    using ValueType = typename StateIdType::ValueType;
    using GroupTag = typename StateIdType::GroupTag;
    using State = typename GroupTag::State;

    ValueType value;

    constexpr StateVal(ValueType v) : value(v) {}
};

template <typename... Fields>
class StateStruct {

    template <typename Field>
    using ValueType = typename Field::ValueType;

private:
    std::tuple<ValueType<Fields>...> values;
    std::bitset<sizeof...(Fields)> known;

    template <typename F>
    static constexpr std::size_t index()
    {
        constexpr bool matches[] = {std::is_same_v<Fields, F>...};
        for (std::size_t i = 0; i < sizeof...(Fields); ++i)
        {
            if (matches[i])
            {
                return i;
            }
        }
        return sizeof...(Fields);
    }

public:
    template <typename Field>
    bool update()
    {
        constexpr std::size_t idx = index<Field>();
        static_assert(idx < sizeof...(Fields), "field does not belong to this StateStruct");
        known.set(idx);
        return true;
    }

    // Access a field's value by the field itself, e.g. get<OperandUnpack::InputFormatA>().
    // Marks the field known/touched -- use the const overload for a read that shouldn't.
    template <typename F>
    auto& get()
    {
        constexpr std::size_t idx = index<F>();
        static_assert(idx < sizeof...(Fields), "field does not belong to this StateStruct");
        known.set(idx);
        return std::get<idx>(values);
    }

    template <typename F>
    const auto& get() const
    {
        constexpr std::size_t idx = index<F>();
        static_assert(idx < sizeof...(Fields), "field does not belong to this StateStruct");
        return std::get<idx>(values);
    }

    // Has this field been touched via get<F>() at least once?
    template <typename F>
    bool is_set() const
    {
        constexpr std::size_t idx = index<F>();
        static_assert(idx < sizeof...(Fields), "field does not belong to this StateStruct");
        return known.test(idx);
    }
};


// Reverse lookup: given a field, recover its owning group tag and StateStruct.
template <typename F>
using group_of_t = typename F::GroupTag;

template <typename F>
using state_of_t = typename group_of_t<F>::State;


class Operation
{
public:
    enum class Exu : std::uint32_t
    {
        Unpack = 0,
        Fpu    = 1,
        Sfpu   = 2,
        Pack   = 3
    };

    enum class Thread : std::uint32_t
    {
        Trisc0 = 0,
        Trisc1 = 1,
        Trisc2 = 2,
        Trisc3 = 3
    };

    enum class Hoistable : std::uint32_t
    {
        No  = 0,
        Yes = 1
    };

    enum class Dependency : std::uint32_t
    {   
        None = 0,
        
        // UNPACK EXU
        UnpackAInputFormat  = 1u << 0,
        UnpackAOutputFormat = 1u << 1,
        UnpackAFaceHeight   = 1u << 2,
        UnpackANumFaces     = 1u << 3,
        UnpackBInputFormat  = 1u << 4,
        UnpackBOutputFormat = 1u << 5,
        UnpackBFaceHeight   = 1u << 6,
        UnpackBNumFaces     = 1u << 7,
        UnpackDestWidth32   = 1u << 8,

        UnpackInputFormat   = UnpackAInputFormat | UnpackBInputFormat,
        UnpackOutputFormat  = UnpackAOutputFormat | UnpackBOutputFormat,
        UnpackFaceHeight    = UnpackAFaceHeight | UnpackBFaceHeight,
        UnpackNumFaces      = UnpackANumFaces | UnpackBNumFaces,
    
        // FPU EXU 
        FpuFormat           = 1u << 0,
        
        // SFPU EXU
        SfpuFormat          = 1u << 1,

        // PACK EXU
        PackInputFormat    = 1u << 0,
        PackOutputFormat   = 1u << 1,
        PackFaceHeight     = 1u << 2,
        PackTileWidth      = 1u << 3,
        PackNumFaces       = 1u << 4,
        PackPartialFace    = 1u << 5,
        PackNarrowTile     = 1u << 6,
        PackDestWidth32    = 1u << 7,

    };



private:
    // First word.
    std::uint32_t _thread       : 2;
    std::uint32_t _exu          : 2;
    std::uint32_t _hoistable    : 1;
    std::uint32_t
    std::uint32_t _id           : 27;

    // Operand Dependencies.

    std::


    // sstanisic todo: add bitset for operand dependencies for each operation

    // Operand Clobbers.

    // sstanisic todo: add bitset for operand clobbers for each operation

    
    constexpr Operation(std::uint32_t thread, std::uint32_t exu, std::uint32_t id) : _thread(thread), _exu(exu), _id(id)
    {
    }

    static constexpr Operation make(Exu exu, Thread thread, std::uint32_t id)
    {
        std::uint32_t _exu      = std::to_underlying(exu);
        std::uint32_t _thread   = std::to_underlying(thread);
        return Operation(_thread, _exu, id);
    }

public:
    constexpr Thread thread()
    {
        return static_cast<Thread>(_thread);
    }

    constexpr Exu exu()
    {
        return static_cast<Exu>(_exu);
    }

    constexpr 

    constexpr std::uint32_t id()
    {
        return _id;
    }

    // ----------------------
    // Operation Declarations
    // ----------------------

    static const Operation None;
    // UNPACK EXU (TRISC0)
    static const Operation UnpackA;
    static const Operation UnpackABMatmul;
    static const Operation UnpackUntilize;
    // FPU EXU (TRISC1)
    static const Operation FpuMatmul;
    static const Operation FpuEltwiseUnaryDatacopy;
    static const Operation FpuEltwiseBinaryAdd;
    static const Operation FpuEltwiseBinarySub;
    static const Operation FpuEltwiseBinaryMul;
    static const Operation FpuEltwiseBinaryAddDestReuse;
    static const Operation FpuEltwiseBinarySubDestReuse;
    static const Operation FpuEltwiseBinaryMulDestReuse;
    // PACK EXU (TRISC2)
    static const Operation Pack;
    static const Operation PackUntilize;
};

static_assert(sizeof(Operation) == sizeof(std::uint32_t), "Operation must be 32 bits");


enum class OperationStatus : std::uint32_t
{
    None,
    Initialized,
    Executed,
    Uninitialized
};

struct OperationState
{
    static constexpr size_t BUFFER_SIZE = 96;

    // aligned to max alignment so that content of buffer
    // is accessible through T* irrespective of the alignment of T
    alignas(alignof(max_align_t)) char buffer[BUFFER_SIZE] {};
    OperationStatus status = OperationStatus::None;
    Operation operation    = Operation::None;
};

enum class FsmStateType : std::uint32_t
{
    Initial,
    Configured,
    Initialized,
    Executed,
    Uninitialized,
    Reconfigured
};

struct FsmState
{
    FsmStateType type   = FsmStateType::Initial;
    Operation operation = Operation::None; // Metadata for INIT and EXECUTE and UNINIT
};

struct UnwindContext
{
    std::uintptr_t pc = UINTPTR_MAX;
    std::uintptr_t ra = UINTPTR_MAX;

    static const UnwindContext UNKNOWN;
};

inline const UnwindContext UnwindContext::UNKNOWN {UINTPTR_MAX, UINTPTR_MAX};

struct ThreadOutputContext
{
    UnwindContext operation;
    UnwindContext fsm;
    UnwindContext current;
    std::size_t context_depth = 0;
    std::size_t silent_depth  = 0;
};

struct UnpackOutputContext : ThreadOutputContext
{
    UnwindContext configure_a;
    UnwindContext configure_b;
};

struct MathOutputContext : ThreadOutputContext
{
    UnwindContext configure_fpu;
    UnwindContext configure_sfpu;
};

struct PackOutputContext : ThreadOutputContext
{
    UnwindContext configure_pack;
};

struct OutputContext
{
    UnpackOutputContext unpack;
    MathOutputContext math;
    PackOutputContext pack;
};

struct SanitizerState
{
    OutputContext context;
    OperandState operand;
    OperationState operation[MAX_THREADS];
    FsmState fsm[MAX_THREADS];
};

} // namespace llk::san
