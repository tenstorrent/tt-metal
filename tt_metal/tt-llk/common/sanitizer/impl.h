// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>

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
    new (&sanitizer.fsm[COMPILE_FOR_TRISC]) llk::san::FsmState(llk::san::FsmState::Initial);
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
    asm volatile("auipc %[pc], 0\n" : [pc] "=r"(context.pc));
    context.ra = (uintptr_t)__builtin_return_address(0);
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

    // sstanisic fixme: this is shared state, only set during the configure. current assumption is that it won't change. (see #47440)
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

    state.is_configured = true;
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
        const UnwindContext current = context.current;

        operand_assert<Trigger::ERROR>(
            state.dest_width_32, dest_acc_en, CTSTR("configured vs provided UNPACK DEST ACCUMULATION are mismatched"), context.configure_a, current);
        operand_assert<Trigger::ERROR>(
            state.src_a.input_format, src_fmt_A, CTSTR("configured vs provided UNPACK A L1 FORMAT are mismatched"), context.configure_a, current);
        operand_assert<Trigger::ERROR>(
            state.src_b.input_format, src_fmt_B, CTSTR("configured vs provided UNPACK B L1 FORMAT are mismatched"), context.configure_b, current);
        operand_assert<Trigger::ERROR>(
            state.src_a.output_format, dst_fmt_A, CTSTR("configured vs provided SRC A FORMAT are mismatched"), context.configure_a, current);
        operand_assert<Trigger::ERROR>(
            state.src_b.output_format, dst_fmt_B, CTSTR("configured vs provided SRC B FORMAT are mismatched"), context.configure_b, current);
        operand_assert<Trigger::ERROR>(
            state.src_a.face_height, face_height_A, CTSTR("configured vs provided UNPACK A L1 FACE HEIGHT are mismatched"), context.configure_a, current);
        operand_assert<Trigger::ERROR>(
            state.src_b.face_height, face_height_B, CTSTR("configured vs provided UNPACK B L1 FACE HEIGHT are mismatched"), context.configure_b, current);
        operand_assert<Trigger::ERROR>(
            state.src_a.num_faces, num_faces_A, CTSTR("configured vs provided UNPACK A L1 NUM FACES are mismatched"), context.configure_a, current);
        operand_assert<Trigger::ERROR>(
            state.src_b.num_faces, num_faces_B, CTSTR("configured vs provided UNPACK B L1 NUM FACES are mismatched"), context.configure_b, current);
    }
}

// No state set, just check that non x arguments match the stored ones
static inline void math_operand_check_impl(
    const MathOutputContext& context, MathOperandState& state, State<std::uint32_t> math_fmt_A, State<std::uint32_t> math_fmt_B)
{
    if (!thread_silent_get_impl(context))
    {
        const UnwindContext current = context.current;

        operand_assert<Trigger::ERROR>(
            state.src_a.input_format, math_fmt_A, CTSTR("configured vs provided FPU FORMAT are mismatched"), context.configure_fpu, current);
        operand_assert<Trigger::ERROR>(
            state.src_b.input_format, math_fmt_B, CTSTR("configured vs provided SFPU FORMAT are mismatched"), context.configure_sfpu, current);
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
        const UnwindContext current = context.current;

        operand_assert<Trigger::ERROR>(
            state.dest_width_32, dest_acc_en, CTSTR("configured vs provided PACK DEST ACCUMULATION are mismatched"), context.configure_pack, current);
        operand_assert<Trigger::ERROR>(
            state.input_format, src_fmt, CTSTR("configured vs provided PACK DEST FORMAT are mismatched"), context.configure_pack, current);
        operand_assert<Trigger::ERROR>(
            state.output_format, dst_fmt, CTSTR("configured vs provided PACK L1 FORMAT are mismatched"), context.configure_pack, current);
        // sstanisic fixme: face_height check (see #47440)
        operand_assert<Trigger::ERROR>(
            state.tile_width, tile_width, CTSTR("configured vs provided PACK L1 TILE WIDTH are mismatched"), context.configure_pack, current);
        operand_assert<Trigger::ERROR>(
            state.num_faces, num_faces, CTSTR("configured vs provided PACK L1 NUM FACES are mismatched"), context.configure_pack, current);
        operand_assert<Trigger::ERROR>(
            state.partial_face, partial_face, CTSTR("configured vs provided PACK L1 PARTIAL FACE are mismatched"), context.configure_pack, current);
        operand_assert<Trigger::ERROR>(
            state.narrow_tile, narrow_tile, CTSTR("configured vs provided PACK L1 NARROW TILE are mismatched"), context.configure_pack, current);
    }
}

template <typename... Ts>
constexpr std::uint8_t _args_count()
{
    static_assert((sizeof...(Ts) <= std::numeric_limits<std::uint8_t>::max()), "llk::san | fault   | argument count can't fit in uint8_t");

    return static_cast<std::uint8_t>(sizeof...(Ts));
}

template <typename... Ts>
constexpr std::array<std::uint8_t, sizeof...(Ts)> _args_sizeof()
{
    static_assert(((sizeof(Ts) <= std::numeric_limits<std::uint8_t>::max()) && ...), "llk::san | fault   | sizeof can't fit in uint8_t");

    return {static_cast<std::uint8_t>(sizeof(Ts))...};
}

template <typename... Ts>
static inline constexpr std::array<std::uint8_t, sizeof...(Ts)> _args_alignof()
{
    static_assert(((alignof(Ts) <= std::numeric_limits<std::uint8_t>::max()) && ...), "llk::san | fault   | alignof can't fit in uint8_t");

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
static inline void operation_init_impl(ThreadOutputContext& context, OperationState& state, const Ts... args)
{
    state.operation     = op;
    state.expect_uninit = operation_must_uninit<op>;
    context.operation   = context.current;

    constexpr std::uint8_t args_count = _args_count<Ts...>();

    constexpr std::array<std::uint8_t, args_count> args_sizeof  = _args_sizeof<Ts...>();
    constexpr std::array<std::uint8_t, args_count> args_alignof = _args_alignof<Ts...>();
    constexpr std::array<size_t, args_count + 1> args_offsetof  = _args_offsetof(args_sizeof, args_alignof);

    constexpr size_t entry_size = _operation_entry_size<args_count>(args_sizeof, args_alignof, args_offsetof);

    static_assert(entry_size <= OperationState::BUFFER_SIZE, "llk::san | fault   | operation entry will overflow the buffer");

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
static inline void operation_check_impl(const ThreadOutputContext& context, OperationState& state, const Ts... args)
{
    if (thread_silent_get_impl(context))
    {
        return;
    }

    const bool passed = operation_assert<Trigger::ERROR>(state.operation, op, context.operation, context.current);

    if (!passed)
    {
        return;
    }

    constexpr std::uint8_t args_count = _args_count<Ts...>();

    constexpr std::array<std::uint8_t, args_count> args_sizeof  = _args_sizeof<Ts...>();
    constexpr std::array<std::uint8_t, args_count> args_alignof = _args_alignof<Ts...>();
    constexpr std::array<size_t, args_count + 1> args_offsetof  = _args_offsetof(args_sizeof, args_alignof);

    constexpr size_t entry_size = _operation_entry_size<args_count>(args_sizeof, args_alignof, args_offsetof);

    static_assert(entry_size <= OperationState::BUFFER_SIZE, "llk::san | fault   | operation entry will overflow the buffer");

    // | ARG_COUNT | SIZEOF(args[0]) ... | ALIGNOF(args[1]) ... | args[0] PADDING ... |

    char* ptr = state.buffer;

    LLK_ASSERT(std::memcmp(&args_count, ptr, sizeof(args_count)) == 0, "llk::san | fault   | saved vs provided args_count mismatch");
    ptr += sizeof(args_count);

    if constexpr (args_count > 0)
    {
        LLK_ASSERT(
            std::memcmp(args_sizeof.data(), ptr, args_count * sizeof(args_sizeof[0])) == 0, "llk::san | fault   | saved vs provided args_sizeof mismatch");
        ptr += args_count * sizeof(args_sizeof[0]);

        LLK_ASSERT(
            std::memcmp(args_alignof.data(), ptr, args_count * sizeof(args_alignof[0])) == 0, "llk::san | fault   | saved vs provided args_alignof mismatch");
        ptr += args_count * sizeof(args_alignof[0]);

        constexpr size_t max_align = alignof(max_align_t);
        size_t padding             = (max_align - reinterpret_cast<uintptr_t>(ptr) % max_align) % max_align;
        ptr += padding;

        [[maybe_unused]] size_t i = 0;
        (
            [&]([[maybe_unused]] const auto& arg)
            {
                operation_argument_assert<Trigger::ERROR>(ptr + args_offsetof[i], &arg, sizeof(arg), i, context.operation, context.current);
                ++i;
            }(args),
            ...);
    }
}

// Goes in LLK_LIB in Uninit
// Check operation type and clear must uninit flag
template <Operation op>
static inline void operation_uninit_impl(const ThreadOutputContext& context, OperationState& state)
{
    if (!thread_silent_get_impl(context))
    {
        operation_assert<Trigger::ERROR>(state.operation, op, context.operation, context.current);
    }

    state.expect_uninit = false;
}

template <FsmState next>
static inline void fsm_advance_impl(ThreadOutputContext& context, FsmState& current, [[maybe_unused]] const OperationState& operation)
{
    if (!thread_silent_get_impl(context))
    {
        fsm_assert<Trigger::ERROR>(
            current != FsmState::Initial || next == FsmState::Configured,
            CTSTR("First transition must be INITIAL -> CONFIGURED"),
            current,
            next,
            CTSTR("CONFIGURED"),
            UnwindContext::UNKNOWN,
            context.current);

        fsm_assert<Trigger::ERROR>(
            current != FsmState::Configured || next == FsmState::Initialized,
            CTSTR("Expected CONFIGURED -> INITIALIZED"),
            current,
            next,
            CTSTR("INITIALIZED"),
            context.fsm,
            context.current);

        fsm_assert<Trigger::ERROR>(
            current != FsmState::Initialized || !operation.expect_uninit || next == FsmState::Executed,
            CTSTR("Operation UNINIT required, expected INITIALIZED -> EXECUTED"),
            current,
            next,
            CTSTR("EXECUTED"),
            context.fsm,
            context.current);

        // Reconfig after init (without an intervening execute) is tolerated for operations that don't require
        // uninit, but it is still likely indicative of a bug, hence WARN rather than ERROR.
        fsm_assert<Trigger::WARN>(
            current != FsmState::Initialized || operation.expect_uninit || next == FsmState::Executed || next == FsmState::Reconfigured,
            CTSTR("Operation UNINIT not required, expected INITIALIZED -> [EXECUTED, RECONFIGURED]"),
            current,
            next,
            CTSTR("EXECUTED, RECONFIGURED"),
            context.fsm,
            context.current);

        fsm_assert<Trigger::ERROR>(
            current != FsmState::Executed || !operation.expect_uninit || next == FsmState::Uninitialized || next == FsmState::Executed,
            CTSTR("Operation UNINIT required, expected EXECUTED -> [UNINITIALIZED, EXECUTED]"),
            current,
            next,
            CTSTR("UNINITIALIZED, EXECUTED"),
            context.fsm,
            context.current);

        fsm_assert<Trigger::ERROR>(
            current != FsmState::Executed || operation.expect_uninit || next == FsmState::Executed || next == FsmState::Initialized ||
                next == FsmState::Reconfigured,
            CTSTR("Operation UNINIT not required, expected EXECUTED -> [EXECUTED, INITIALIZED, RECONFIGURED]"),
            current,
            next,
            CTSTR("EXECUTED, INITIALIZED, RECONFIGURED"),
            context.fsm,
            context.current);

        fsm_assert<Trigger::ERROR>(
            current != FsmState::Uninitialized || next == FsmState::Initialized || next == FsmState::Reconfigured,
            CTSTR("Expected UNINITIALIZED -> [INITIALIZED, RECONFIGURED]"),
            current,
            next,
            CTSTR("INITIALIZED, RECONFIGURED"),
            context.fsm,
            context.current);

        fsm_assert<Trigger::ERROR>(
            current != FsmState::Reconfigured || next == FsmState::Initialized || next == FsmState::Reconfigured,
            CTSTR("Expected RECONFIGURED -> [INITIALIZED, RECONFIGURED]"),
            current,
            next,
            CTSTR("INITIALIZED, RECONFIGURED"),
            context.fsm,
            context.current);
    }

    // valid transition -> commit
    current     = next;
    context.fsm = context.current;
}

} // namespace llk::san
