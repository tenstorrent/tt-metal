// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "ckernel.h"
#include "llk_defs.h"

namespace llk::san
{

struct Ignore
{
    constexpr Ignore() = default;
};

struct Unknown
{
    constexpr Unknown() = default;
};

static constexpr Ignore IGNORE   = Ignore();
static constexpr Unknown UNKNOWN = Unknown();

enum class StateType : std::uint8_t
{
    Known,
    Unknown,
    Ignore
};

template <typename T>
class State;

template <typename T>
struct is_state : std::false_type
{
};

template <typename T>
struct is_state<State<T>> : std::true_type
{
};

template <typename T>
inline constexpr bool is_state_v = is_state<T>::value;

// True for value types that construct a KNOWN State, i.e. anything that is not a State<>,
// nor one of the IGNORE / UNKNOWN markers.
template <typename T>
inline constexpr bool is_known_value_v = !is_state_v<std::decay_t<T>> && !std::is_same_v<std::decay_t<T>, Ignore> && !std::is_same_v<std::decay_t<T>, Unknown>;

template <typename T>
class State
{
private:
    T underlying;
    StateType state_type;

public:
    template <typename U>
    friend class State;

    // CONSTRUCTION
    // Default to UNKNOWN state because hardware is not initialized
    State() noexcept(std::is_nothrow_default_constructible_v<T>) : underlying {}, state_type(StateType::Unknown)
    {
    }

    State(const State&) noexcept(std::is_nothrow_copy_constructible_v<T>) = default;
    State(State&&) noexcept(std::is_nothrow_move_constructible_v<T>)      = default;

    // CONVERSION
    // - llk::san::IGNORE     -> State with state_type == Ignore
    // - llk::san::UNKNOWN    -> State with state_type == Unknown
    // - other                -> State with state_type == Known (storing the value)

    // Constructor for IGNORE
    constexpr State(const Ignore&) noexcept : underlying {}, state_type(StateType::Ignore)
    {
    }

    // Constructor for UNKNOWN
    constexpr State(const Unknown&) noexcept : underlying {}, state_type(StateType::Unknown)
    {
    }

    // Constructor for KNOWN value
    template <typename U, typename = std::enable_if_t<is_known_value_v<U>>>
    constexpr State(U&& value) noexcept(std::is_nothrow_constructible_v<T, U&&>) : underlying(std::forward<U>(value)), state_type(StateType::Known)
    {
    }

    // ASSIGNMENT
    // if RHS of assignment is StateType::Ignore, noop (stays old value)
    // otherwise take the state_type and underlying of RHS

    template <typename U>
    State& operator=(const State<U>& rhs) noexcept(std::is_nothrow_copy_assignable_v<T>)
    {
        if (rhs.state_type == StateType::Ignore)
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
        if (rhs.state_type == StateType::Ignore)
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
        return state_type == StateType::Known;
    }

    constexpr bool is_unknown() const noexcept
    {
        return state_type == StateType::Unknown;
    }

    constexpr bool is_ignore() const noexcept
    {
        return state_type == StateType::Ignore;
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

// TODO: refactor below

enum class llk_san_cfg_t
{
    Addrmod,
    Mop,
    DvalidDisable,
    CH0Strides,
    CH1Strides,
    TileDesc,
    AdcXX,
    Transpose,
    L1Offset
};

enum class llk_san_operand_t
{
    SrcA,
    SrcB,
    Dst
};

// UNPACK operand state
struct UnpackSrcState
{
    State<std::uint32_t> input_format;
    State<std::uint32_t> output_format;
    State<std::uint32_t> face_height;
    State<std::uint32_t> num_faces;
};

struct UnpackOperandState
{
    UnpackSrcState src_a;
    UnpackSrcState src_b;
    State<bool> dest_width_32;
    bool is_configured = false;
};

// MATH operand state
struct MathSrcState
{
    State<std::uint32_t> input_format;
};

struct MathOperandState
{
    MathSrcState src_a;
    MathSrcState src_b;
    bool is_configured = false;
};

// PACK operand state
struct PackOperandState
{
    State<std::uint32_t> input_format;
    State<std::uint32_t> output_format;
    State<std::uint32_t> face_height;
    State<std::uint32_t> tile_width;
    State<std::uint32_t> num_faces;
    State<bool> partial_face;
    State<bool> narrow_tile;
    State<bool> dest_width_32;
    bool is_configured = false;
};

struct OperandState
{
    UnpackOperandState unpack;
    MathOperandState math;
    PackOperandState pack;
};

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

private:
    // First word.
    std::uint32_t _thread   : 2;
    std::uint32_t _exu      : 2;
    std::uint32_t _id       : 28;

    // Operand Dependencies.

    // sstanisic todo: add bitset for operand dependencies for each operation

    // Operand Clobbers.

    // sstanisic todo: add bitset for operand clobbers for each operation

    static_assert(sizeof(Operation) == sizeof(std::uint32_t), "Operation must be 32 bits");

    constexpr Operation(std::uint32_t thread, std::uint32_t exu, std::uint32_t id) : _thread(thread), _exu(exu), _id(id)
    {
    }

    static constexpr Operation make(Exu exu, Thread thread, std::uint32_t id)
    {
        std::uint32_t _exu      = ckernel::to_underlying(exu);
        std::uint32_t _thread   = ckernel::to_underlying(thread);
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

    constexpr std::uint32_t id()
    {
        return _id;
    }

    // ---------------------
    // Operation Definitions
    // ---------------------

    static constexpr Operation None = make(Exu::Unpack, Thread::Trisc0, 0);
    // UNPACK EXU (TRISC0)
    static constexpr Operation UnpackA        = make(Exu::Unpack, Thread::Trisc0, 1);
    static constexpr Operation UnpackABMatmul = make(Exu::Unpack, Thread::Trisc0, 2);
    static constexpr Operation UnpackUntilize = make(Exu::Unpack, Thread::Trisc0, 3);
    // MATH EXU (TRISC1)
    static constexpr Operation Matmul                       = make(Exu::Fpu, Thread::Trisc1, 1);
    static constexpr Operation FpuEltwiseUnaryDatacopy      = make(Exu::Fpu, Thread::Trisc1, 2);
    static constexpr Operation FpuEltwiseBinaryAdd          = make(Exu::Fpu, Thread::Trisc1, 3);
    static constexpr Operation FpuEltwiseBinarySub          = make(Exu::Fpu, Thread::Trisc1, 4);
    static constexpr Operation FpuEltwiseBinaryMul          = make(Exu::Fpu, Thread::Trisc1, 5);
    static constexpr Operation FpuEltwiseBinaryAddDestReuse = make(Exu::Fpu, Thread::Trisc1, 6);
    static constexpr Operation FpuEltwiseBinarySubDestReuse = make(Exu::Fpu, Thread::Trisc1, 7);
    static constexpr Operation FpuEltwiseBinaryMulDestReuse = make(Exu::Fpu, Thread::Trisc1, 8);
    // PACK EXU (TRISC2)
    static constexpr Operation Pack         = make(Exu::Pack, Thread::Trisc2, 1);
    static constexpr Operation PackUntilize = make(Exu::Pack, Thread::Trisc2, 2);
};

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
