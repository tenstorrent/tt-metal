// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

#include "llk_assert.h"
#include "sanitizer/output.h"
#include "sanitizer/types.h"

namespace llk::san
{

static inline auto& thread_context_get_impl(SanitizerState& sanitizer) noexcept
{
    if constexpr (COMPILE_FOR_TRISC == 0)
    {
        return sanitizer.context.unpack;
    }
    else if constexpr (COMPILE_FOR_TRISC == 1)
    {
        return sanitizer.context.math;
    }
    else if constexpr (COMPILE_FOR_TRISC == 2)
    {
        return sanitizer.context.pack;
    }
}

static inline void thread_init_impl(SanitizerState& sanitizer)
{
    if constexpr (COMPILE_FOR_TRISC == 0)
    {
        // Unpacker thread
        new (&sanitizer.operand.unpack) llk::san::UnpackOperandState();
        new (&sanitizer.context.unpack) llk::san::UnpackOutputContext();
    }
    else if constexpr (COMPILE_FOR_TRISC == 1)
    {
        // Math thread
        new (&sanitizer.operand.math) llk::san::MathOperandState();
        new (&sanitizer.context.math) llk::san::MathOutputContext();
    }
    else if constexpr (COMPILE_FOR_TRISC == 2)
    {
        // Packer thread
        new (&sanitizer.operand.pack) llk::san::PackOperandState();
        new (&sanitizer.context.pack) llk::san::PackOutputContext();
    }

    new (&sanitizer.operation[COMPILE_FOR_TRISC]) llk::san::OperationState();
    new (&sanitizer.fsm[COMPILE_FOR_TRISC]) llk::san::FsmState(llk::san::FsmState::INITIAL);
}

static inline void thread_silent_push_impl(ThreadOutputContext& context)
{
    context.silent_depth++;
}

static inline void thread_silent_pop_impl(ThreadOutputContext& context)
{
    context.silent_depth--;
}

static inline bool thread_silent_get_impl(const ThreadOutputContext& context)
{
    return context.silent_depth > 0;
}

static TT_ALWAYS_INLINE void write_unwind_context(UnwindContext& context)
{
    asm volatile(
        "auipc %[pc], 0\n"
        "mv %[ra], ra"
        : [pc] "=r"(context.pc), [ra] "=r"(context.ra) // Output operands
    );
}

static inline void thread_context_push_impl(ThreadOutputContext& context)
{
    // to handle nested compute api calls, we only push for the first function.
    if (context.context_depth++ == 0)
    {
        write_unwind_context(context.current);
    }
}

static inline void thread_context_pop_impl(ThreadOutputContext& context)
{
    // to handle nested compute api calls, we only pop for the first function.
    if (--context.context_depth == 0)
    {
        context.current = UnwindContext::UNKNOWN;
    }
}

// Goes in LLK_LIB in HWConfigure and HWReconfig
// State set + no hw config within kernel check
template <bool reconfig>
static inline void unpack_operand_configure_impl(
    UnpackOutputContext& context,
    UnpackOperandState& state,
    State<bool> dst_acc_en,
    State<std::uint32_t> src_fmt_A,
    State<std::uint32_t> src_fmt_B,
    State<std::uint32_t> dst_fmt_A,
    State<std::uint32_t> dst_fmt_B,
    State<std::uint32_t> face_height_A,
    State<std::uint32_t> face_height_B,
    State<std::uint32_t> num_faces_A,
    State<std::uint32_t> num_faces_B)
{
    UnpackSrcState& src_a = state.src_a;

    src_a.input_format  = src_fmt_A;
    src_a.output_format = dst_fmt_A;
    src_a.face_height   = face_height_A;
    src_a.num_faces     = num_faces_A;

    if (src_fmt_A.is_known() || dst_fmt_A.is_known() || face_height_A.is_known() || num_faces_A.is_known())
    {
        context.configure_a = context.current;
    }

    UnpackSrcState& src_b = state.src_b;

    src_b.input_format  = src_fmt_B;
    src_b.output_format = dst_fmt_B;
    src_b.face_height   = face_height_B;
    src_b.num_faces     = num_faces_B;

    if (src_fmt_B.is_known() || dst_fmt_B.is_known() || face_height_B.is_known() || num_faces_B.is_known())
    {
        context.configure_b = context.current;
    }

    // fixme: this is shared state, only set during the configure. current assumption is that it won't change.
    state.dest_width_32 = dst_acc_en;
    state.is_configured = true;
}

// State set + no hw config within kernel check
template <bool reconfig = false>
static inline void math_operand_configure_impl(
    MathOutputContext& context, MathOperandState& state, State<std::uint32_t> math_fmt_A, State<std::uint32_t> math_fmt_B)
{
    if (math_fmt_A.is_known())
    {
        context.configure_fpu = context.current;
    }
    state.src_a.input_format = math_fmt_A;

    if (math_fmt_B.is_known())
    {
        context.configure_sfpu = context.current;
    }
    state.src_b.input_format = math_fmt_B;

    state.is_configured      = true;
}

// State set + no hw config within kernel check
template <bool reconfig>
static inline void pack_operand_configure_impl(
    PackOutputContext& context,
    PackOperandState& state,
    State<bool> dest_acc_en,
    State<std::uint32_t> src_fmt,
    State<std::uint32_t> dst_fmt,
    State<std::uint32_t> face_height,
    State<std::uint32_t> tile_width,
    State<std::uint32_t> num_faces,
    State<bool> partial_face,
    State<bool> narrow_tile)
{
    if (src_fmt.is_known() || dst_fmt.is_known() || face_height.is_known() || tile_width.is_known() || num_faces.is_known() || partial_face.is_known() ||
        narrow_tile.is_known() || dest_acc_en.is_known())
    {
        context.configure_pack = context.current;
    }

    state.input_format  = src_fmt;
    state.output_format = dst_fmt;
    state.face_height   = face_height;
    state.tile_width    = tile_width;
    state.num_faces     = num_faces;
    state.partial_face  = partial_face;
    state.narrow_tile   = narrow_tile;
    state.dest_width_32 = dest_acc_en;
    state.is_configured = true;
}

// Goes in LLK_LIB in Init, Execute and Uninit
// No state set, just check that non x arguments match the stored ones
static inline void unpack_operand_check_impl(
    const UnpackOutputContext& context,
    UnpackOperandState& state,
    State<bool> dest_acc_en,
    State<std::uint32_t> src_fmt_A,
    State<std::uint32_t> src_fmt_B,
    State<std::uint32_t> dst_fmt_A,
    State<std::uint32_t> dst_fmt_B,
    State<std::uint32_t> face_height_A,
    State<std::uint32_t> face_height_B,
    State<std::uint32_t> num_faces_A,
    State<std::uint32_t> num_faces_B)
{
    if (!thread_silent_get_impl(context))
    {
        const auto pc = context.current.pc;

        LLK_SAN_PEDANTIC_PANIC(!state.is_configured, "{:#x} : executing init/execute/uninit before hwconfigure", pc);

        LLK_SAN_ERROR_ASSERT(state.dest_width_32.assert_cond(dest_acc_en), "{:#x} : dest_acc_en doesn't match state.dest_width_32", pc);

        LLK_SAN_ERROR_ASSERT(state.src_a.input_format.assert_cond(src_fmt_A), "{:#x} : src_fmt_A doesn't match state.src_a.input_format", pc);
        LLK_SAN_ERROR_ASSERT(state.src_b.input_format.assert_cond(src_fmt_B), "{:#x} : src_fmt_B doesn't match state.src_b.input_format", pc);
        LLK_SAN_ERROR_ASSERT(state.src_a.output_format.assert_cond(dst_fmt_A), "{:#x} : dst_fmt_A doesn't match state.src_a.output_format", pc);
        LLK_SAN_ERROR_ASSERT(state.src_b.output_format.assert_cond(dst_fmt_B), "{:#x} : dst_fmt_B doesn't match state.src_b.output_format", pc);
        LLK_SAN_ERROR_ASSERT(state.src_a.face_height.assert_cond(face_height_A), "{:#x} : face_height_A doesn't match state.src_a.face_height", pc);
        LLK_SAN_ERROR_ASSERT(state.src_b.face_height.assert_cond(face_height_B), "{:#x} : face_height_B doesn't match state.src_b.face_height", pc);
        LLK_SAN_ERROR_ASSERT(state.src_a.num_faces.assert_cond(num_faces_A), "{:#x} : num_faces_A doesn't match state.src_a.num_faces", pc);
        LLK_SAN_ERROR_ASSERT(state.src_b.num_faces.assert_cond(num_faces_B), "{:#x} : num_faces_B doesn't match state.src_b.num_faces", pc);
    }
}

// No state set, just check that non x arguments match the stored ones
static inline void math_operand_check_impl(
    const MathOutputContext& context, MathOperandState& state, State<std::uint32_t> math_fmt_A, State<std::uint32_t> math_fmt_B)
{
    if (!thread_silent_get_impl(context))
    {
        const auto pc = context.current.pc;
        LLK_SAN_PEDANTIC_PANIC(!state.is_configured, "{:#x} : executing init/execute/uninit before hwconfigure", pc);
        LLK_SAN_ERROR_ASSERT(state.src_a.input_format.assert_cond(math_fmt_A), "{:#x} : math_fmt_A doesn't match state.src_a.input_format", pc);
        LLK_SAN_ERROR_ASSERT(state.src_b.input_format.assert_cond(math_fmt_B), "{:#x} : math_fmt_B doesn't match state.src_b.input_format", pc);
    }
}

// No state set, just check that non x arguments match the stored ones
static inline void pack_operand_check_impl(
    const PackOutputContext& context,
    PackOperandState& state,
    State<bool> dest_acc_en,
    State<std::uint32_t> src_fmt,
    State<std::uint32_t> dst_fmt,
    [[maybe_unused]] State<std::uint32_t> face_height,
    State<std::uint32_t> tile_width,
    State<std::uint32_t> num_faces,
    State<bool> partial_face,
    State<bool> narrow_tile)
{
    if (!thread_silent_get_impl(context))
    {
        const auto pc = context.current.pc;
        LLK_SAN_PEDANTIC_PANIC(!state.is_configured, "{:#x} : executing init/execute/uninit before hwconfigure", pc);
        LLK_SAN_ERROR_ASSERT(state.dest_width_32.assert_cond(dest_acc_en), "{:#x} : dest_acc_en doesn't match state.dest_width_32", pc);
        LLK_SAN_ERROR_ASSERT(state.input_format.assert_cond(src_fmt), "{:#x} : src_fmt doesn't match state.input_format", pc);
        LLK_SAN_ERROR_ASSERT(state.output_format.assert_cond(dst_fmt), "{:#x} : dst_fmt doesn't match state.output_format", pc);
        // sstanisic fixme: LLK_SAN_ERROR_ASSERT(state.face_height.assert_cond(face_height), "{:#x} : face_height doesn't match state.face_height", pc);
        LLK_SAN_ERROR_ASSERT(state.tile_width.assert_cond(tile_width), "{:#x} : tile_width doesn't match state.tile_width", pc);
        LLK_SAN_ERROR_ASSERT(state.num_faces.assert_cond(num_faces), "{:#x} : num_faces doesn't match state.num_faces", pc);
        LLK_SAN_ERROR_ASSERT(state.partial_face.assert_cond(partial_face), "{:#x} : partial_face doesn't match state.partial_face", pc);
        LLK_SAN_ERROR_ASSERT(state.narrow_tile.assert_cond(narrow_tile), "{:#x} : narrow_tile doesn't match state.narrow_tile", pc);
    }
}

template <typename... Ts>
constexpr std::uint8_t _args_count()
{
    static_assert((sizeof...(Ts) <= std::numeric_limits<std::uint8_t>::max()), "llk::san | fault    | argument count can't fit in uint8_t");

    return static_cast<std::uint8_t>(sizeof...(Ts));
}

template <typename... Ts>
constexpr std::array<std::uint8_t, sizeof...(Ts)> _args_sizeof()
{
    static_assert(((sizeof(Ts) <= std::numeric_limits<std::uint8_t>::max()) && ...), "llk::san | fault    | sizeof can't fit in uint8_t");

    return {static_cast<std::uint8_t>(sizeof(Ts))...};
}

template <typename... Ts>
static inline constexpr std::array<std::uint8_t, sizeof...(Ts)> _args_alignof()
{
    static_assert(((alignof(Ts) <= std::numeric_limits<std::uint8_t>::max()) && ...), "llk::san | fault    | alignof can't fit in uint8_t");

    return {static_cast<std::uint8_t>(alignof(Ts))...};
}

template <size_t N>
constexpr std::array<size_t, N + 1> _args_offsetof(const std::array<std::uint8_t, N>& args_sizeof, const std::array<std::uint8_t, N>& args_alignof)
{
    if constexpr (N == 0)
    {
        return {0};
    }

    std::array<size_t, N + 1> args_offset = {};

    args_offset[0] = 0;
    for (size_t i = 1; i < N; i++)
    {
        size_t align   = args_alignof[i];
        size_t end     = args_offset[i - 1] + args_sizeof[i - 1];
        size_t padding = (align - end % align) % align;
        args_offset[i] = end + padding;
    }

    constexpr size_t max_align = alignof(max_align_t);
    size_t final_end           = args_offset[N - 1] + args_sizeof[N - 1];
    size_t final_padding       = (max_align - final_end % max_align) % max_align;
    args_offset[N]             = final_end + final_padding;

    return args_offset;
}

template <std::uint8_t N>
constexpr size_t _operation_entry_size(
    const std::array<std::uint8_t, N>& args_sizeof, const std::array<std::uint8_t, N>& args_alignof, const std::array<size_t, N + 1>& args_offsetof)
{
    if constexpr (N == 0)
    {
        return sizeof(N);
    }

    constexpr size_t max_align = alignof(max_align_t);
    constexpr size_t metadata  = sizeof(N) + N * sizeof(args_sizeof[0]) + N * sizeof(args_alignof[0]);
    constexpr size_t padding   = (max_align - metadata % max_align) % max_align;

    return metadata + padding + args_offsetof[N];
}

// Goes in LLK_LIB in Init
// Store operation type and push arguments to state stack
template <Operation op, typename... Ts>
static inline void operation_init_impl(OperationState& state, const Ts... args)
{
    state.operation     = op;
    state.expect_uninit = operation_must_uninit<op>;

    constexpr std::uint8_t args_count = _args_count<Ts...>();

    constexpr std::array<std::uint8_t, args_count> args_sizeof  = _args_sizeof<Ts...>();
    constexpr std::array<std::uint8_t, args_count> args_alignof = _args_alignof<Ts...>();
    constexpr std::array<size_t, args_count + 1> args_offsetof  = _args_offsetof(args_sizeof, args_alignof);

    constexpr size_t entry_size = _operation_entry_size<args_count>(args_sizeof, args_alignof, args_offsetof);

    static_assert(entry_size <= OperationState::BUFFER_SIZE, "llk::san | fault    | operation entry will overflow the buffer");

    // | ARG_COUNT | SIZEOF(args[0]) ... | ALIGNOF(args[1]) ... | args[0] PADDING ... |

    char* ptr = state.buffer;

    std::memcpy(ptr, &args_count, sizeof(args_count));
    ptr += sizeof(args_count);

    if constexpr (args_count > 0)
    {
        std::memcpy(ptr, args_sizeof.data(), args_count * sizeof(args_sizeof[0]));
        ptr += args_count * sizeof(args_sizeof[0]);

        std::memcpy(ptr, args_alignof.data(), args_count * sizeof(args_alignof[0]));
        ptr += args_count * sizeof(args_alignof[0]);

        constexpr size_t max_align = alignof(max_align_t);
        size_t padding             = (max_align - reinterpret_cast<uintptr_t>(ptr) % max_align) % max_align;
        ptr += padding;

        size_t i = 0;
        (std::memcpy(ptr + args_offsetof[i++], &args, sizeof(args)), ...);
    }
}

// Goes in LLK_LIB in Execute
// Check operation type and arguments against stored ones
template <Operation op, typename... Ts>
void operation_check_impl(const ThreadOutputContext& context, OperationState& state, const Ts... args)
{
    if (!thread_silent_get_impl(context))
    {
        const auto pc = context.current.pc;

        LLK_SAN_ERROR_PANIC(state.operation != op, "{:#x} : operation type doesn't match stored operation", pc);

        constexpr std::uint8_t args_count = _args_count<Ts...>();

        constexpr std::array<std::uint8_t, args_count> args_sizeof  = _args_sizeof<Ts...>();
        constexpr std::array<std::uint8_t, args_count> args_alignof = _args_alignof<Ts...>();
        constexpr std::array<size_t, args_count + 1> args_offsetof  = _args_offsetof(args_sizeof, args_alignof);

        constexpr size_t entry_size = _operation_entry_size<args_count>(args_sizeof, args_alignof, args_offsetof);

        static_assert(entry_size <= OperationState::BUFFER_SIZE, "llk::san | fault    | operation entry will overflow the buffer");

        // | ARG_COUNT | SIZEOF(args[0]) ... | ALIGNOF(args[1]) ... | args[0] PADDING ... |

        char* ptr = state.buffer;

        LLK_SAN_FAULT_ASSERT(std::memcmp(&args_count, ptr, sizeof(args_count)) == 0, "{:#x} : saved vs provided args_count mismatch", pc);
        ptr += sizeof(args_count);

        if constexpr (args_count > 0)
        {
            LLK_SAN_FAULT_ASSERT(
                std::memcmp(args_sizeof.data(), ptr, args_count * sizeof(args_sizeof[0])) == 0, "{:#x} : saved vs provided args_sizeof mismatch", pc);
            ptr += args_count * sizeof(args_sizeof[0]);

            LLK_SAN_FAULT_ASSERT(
                std::memcmp(args_alignof.data(), ptr, args_count * sizeof(args_alignof[0])) == 0, "{:#x} : saved vs provided args_alignof mismatch", pc);
            ptr += args_count * sizeof(args_alignof[0]);

            constexpr size_t max_align = alignof(max_align_t);
            size_t padding             = (max_align - reinterpret_cast<uintptr_t>(ptr) % max_align) % max_align;
            ptr += padding;

            [[maybe_unused]] size_t i = 0;
            ([&]([[maybe_unused]] const auto& arg)
             { LLK_SAN_ERROR_ASSERT(std::memcmp(ptr + args_offsetof[i++], &arg, sizeof(arg)) == 0, "{:#x} : saved vs provided args mismatch", pc); }(args),
             ...);
        }
    }
}

// Goes in LLK_LIB in Uninit
// Check operation type and clear must uninit flag
template <Operation op>
void operation_uninit_impl(const ThreadOutputContext& context, OperationState& state)
{
    if (!thread_silent_get_impl(context))
    {
        LLK_SAN_ERROR_PANIC(state.operation != op, "{:#x} : tried to uninit wrong operation type", context.current.pc);
    }

    state.expect_uninit = false;
}

static inline ct_string fsm_state_name(const FsmState state)
{
    switch (state)
    {
        case FsmState::INITIAL:
            return CTSTR("initial");
        case FsmState::CONFIGURED:
            return CTSTR("configured");
        case FsmState::INITIALIZED:
            return CTSTR("initialized");
        case FsmState::EXECUTED:
            return CTSTR("executed");
        case FsmState::UNINITIALIZED:
            return CTSTR("uninitialized");
        case FsmState::RECONFIGURED:
            return CTSTR("reconfigured");
    }
    __builtin_unreachable();
}

template <FsmState next>
void fsm_advance_impl(const ThreadOutputContext& context, FsmState& current, [[maybe_unused]] const OperationState& operation)
{
    ct_string next_name = fsm_state_name(next);

    if (!thread_silent_get_impl(context))
    {
        const auto pc = context.current.pc;

        LLK_SAN_ERROR_PANIC(
            current == FsmState::INITIAL && next != FsmState::CONFIGURED, "{:#x} : initial -> {}, expected first operation to be configure", pc, next_name);

        LLK_SAN_ERROR_PANIC(
            current == FsmState::CONFIGURED && next != FsmState::INITIALIZED, "{:#x} : configured -> {}, expected init after configure", pc, next_name);

        LLK_SAN_ERROR_PANIC(
            current == FsmState::INITIALIZED && next != FsmState::EXECUTED, "{:#x} : initialized -> {}, expected execute after init", pc, next_name);

        LLK_SAN_ERROR_PANIC(
            current == FsmState::EXECUTED && operation.expect_uninit && (next != FsmState::UNINITIALIZED && next != FsmState::EXECUTED),
            "{:#x} : executed -> {}, expected execute or uninit after execute",
            pc,
            next_name);

        // shouldn't be possible to trigger, sanity check
        LLK_SAN_FAULT_PANIC(
            current == FsmState::EXECUTED && !operation.expect_uninit && next == FsmState::UNINITIALIZED,
            "{:#x} : executed -> {}, unexpected uninit after execute that doesn't require uninit",
            pc,
            next_name);

        LLK_SAN_ERROR_PANIC(
            current == FsmState::EXECUTED && !operation.expect_uninit &&
                (next != FsmState::EXECUTED && next != FsmState::INITIALIZED && next != FsmState::RECONFIGURED),
            "{:#x} : executed -> {}, expected execute, init or reconfig operation after execute",
            pc,
            next_name);

        LLK_SAN_ERROR_PANIC(
            current == FsmState::UNINITIALIZED && (next != FsmState::INITIALIZED && next != FsmState::RECONFIGURED),
            "{:#x} : uninitialized -> {}, expected init or reconfigure after uninit",
            pc,
            next_name);
    }

    // valid transition -> commit
    current = next;
}

} // namespace llk::san
