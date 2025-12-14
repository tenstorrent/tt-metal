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

// Goes in ComputeAPI
// State set only
// sstanisic todo: implement support_backtrace_impl
// static inline void support_backtrace_impl(std::string function_name)
// {
//     LLK_SAN_ASSERT(false, "not implemented");
// }

// Goes in LLK_API
// State set only
// sstanisic todo: implement support_globals_impl
// static inline void support_globals_impl(bool dst_acc_mode, DstSync dst_sync, bool approx, std::int32_t math_fidelity)
// {
//     LLK_SAN_ASSERT(false, "not implemented");
// }

static inline void thread_init_impl(SanitizerState& sanitizer)
{
    if constexpr (COMPILE_FOR_TRISC == 0)
    {
        // Unpacker thread
        new (&sanitizer.operand.unpack) llk::san::UnpackOperandState();
    }
    else if constexpr (COMPILE_FOR_TRISC == 1)
    {
        // Math thread
        new (&sanitizer.operand.math) llk::san::MathOperandState();
    }
    else if constexpr (COMPILE_FOR_TRISC == 2)
    {
        // Packer thread
        new (&sanitizer.operand.pack) llk::san::PackOperandState();
    }

    new (&sanitizer.operation[COMPILE_FOR_TRISC]) llk::san::OperationState();
    new (&sanitizer.fsm[COMPILE_FOR_TRISC]) llk::san::FsmState(llk::san::FsmState::INITIAL);

    sanitizer.function_curr[COMPILE_FOR_TRISC]  = CTSTR("<unknown>");
    sanitizer.function_depth[COMPILE_FOR_TRISC] = 0;

    sanitizer.silent_depth[COMPILE_FOR_TRISC] = 0;
}

static inline void thread_silent_push_impl(SanitizerState& sanitizer)
{
    sanitizer.silent_depth[COMPILE_FOR_TRISC]++;
}

static inline void thread_silent_pop_impl(SanitizerState& sanitizer)
{
    sanitizer.silent_depth[COMPILE_FOR_TRISC]--;
}

static inline bool thread_silent_get_impl(const SanitizerState& sanitizer)
{
    return sanitizer.silent_depth[COMPILE_FOR_TRISC] > 0;
}

static inline void thread_function_push_impl(SanitizerState& sanitizer, ct_string function)
{
    // to handle nested compute api calls, we only push for the first function.
    if (sanitizer.function_depth[COMPILE_FOR_TRISC]++ == 0)
    {
        sanitizer.function_curr[COMPILE_FOR_TRISC] = function;
        return;
    }
}

static inline void thread_function_pop_impl(SanitizerState& sanitizer)
{
    // to handle nested compute api calls, we only pop for the first function.
    if (--sanitizer.function_depth[COMPILE_FOR_TRISC] == 0)
    {
        sanitizer.function_curr[COMPILE_FOR_TRISC] = CTSTR("<unknown>");
        return;
    }
}

// Goes in LLK_LIB in HWConfigure and HWReconfig
// State set + no hw config within kernel check
template <bool reconfig>
static inline void unpack_operand_configure_impl(
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

    UnpackSrcState& src_b = state.src_b;

    src_b.input_format  = src_fmt_B;
    src_b.output_format = dst_fmt_B;
    src_b.face_height   = face_height_B;
    src_b.num_faces     = num_faces_B;

    state.dest_width_32 = dst_acc_en;
    state.is_configured = true;
}

// State set + no hw config within kernel check
template <bool reconfig = false>
static inline void math_operand_configure_impl(MathOperandState& state, State<std::uint32_t> math_fmt_A, State<std::uint32_t> math_fmt_B)
{
    state.src_a.input_format = math_fmt_A;
    state.src_b.input_format = math_fmt_B;
    state.is_configured      = true;
}

// State set + no hw config within kernel check
template <bool reconfig>
static inline void pack_operand_configure_impl(
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
    ct_string function,
    bool silent,
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
    if (!silent)
    {
        LLK_SAN_PEDANTIC_PANIC(!state.is_configured, "{} : executing init/execute/uninit before hwconfigure", function);

        LLK_SAN_ERROR_ASSERT(state.dest_width_32.assert_cond(dest_acc_en), "{} : dest_acc_en doesn't match state.dest_width_32", function);

        LLK_SAN_ERROR_ASSERT(state.src_a.input_format.assert_cond(src_fmt_A), "{} : src_fmt_A doesn't match state.src_a.input_format", function);
        LLK_SAN_ERROR_ASSERT(state.src_b.input_format.assert_cond(src_fmt_B), "{} : src_fmt_B doesn't match state.src_b.input_format", function);
        LLK_SAN_ERROR_ASSERT(state.src_a.output_format.assert_cond(dst_fmt_A), "{} : dst_fmt_A doesn't match state.src_a.output_format", function);
        LLK_SAN_ERROR_ASSERT(state.src_b.output_format.assert_cond(dst_fmt_B), "{} : dst_fmt_B doesn't match state.src_b.output_format", function);
        LLK_SAN_ERROR_ASSERT(state.src_a.face_height.assert_cond(face_height_A), "{} : face_height_A doesn't match state.src_a.face_height", function);
        LLK_SAN_ERROR_ASSERT(state.src_b.face_height.assert_cond(face_height_B), "{} : face_height_B doesn't match state.src_b.face_height", function);
        LLK_SAN_ERROR_ASSERT(state.src_a.num_faces.assert_cond(num_faces_A), "{} : num_faces_A doesn't match state.src_a.num_faces", function);
        LLK_SAN_ERROR_ASSERT(state.src_b.num_faces.assert_cond(num_faces_B), "{} : num_faces_B doesn't match state.src_b.num_faces", function);
    }
}

// No state set, just check that non x arguments match the stored ones
static inline void math_operand_check_impl(
    ct_string function, bool silent, MathOperandState& state, State<std::uint32_t> math_fmt_A, State<std::uint32_t> math_fmt_B)
{
    if (!silent)
    {
        LLK_SAN_PEDANTIC_PANIC(!state.is_configured, "{} : executing init/execute/uninit before hwconfigure", function);
        LLK_SAN_ERROR_ASSERT(state.src_a.input_format.assert_cond(math_fmt_A), "{} : math_fmt_A doesn't match state.src_a.input_format", function);
        LLK_SAN_ERROR_ASSERT(state.src_b.input_format.assert_cond(math_fmt_B), "{} : math_fmt_B doesn't match state.src_b.input_format", function);
    }
}

// No state set, just check that non x arguments match the stored ones
static inline void pack_operand_check_impl(
    ct_string function,
    bool silent,
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
    if (!silent)
    {
        LLK_SAN_PEDANTIC_PANIC(!state.is_configured, "{} : executing init/execute/uninit before hwconfigure", function);
        LLK_SAN_ERROR_ASSERT(state.dest_width_32.assert_cond(dest_acc_en), "{} : dest_acc_en doesn't match state.dest_width_32", function);
        LLK_SAN_ERROR_ASSERT(state.input_format.assert_cond(src_fmt), "{} : src_fmt doesn't match state.input_format", function);
        LLK_SAN_ERROR_ASSERT(state.output_format.assert_cond(dst_fmt), "{} : dst_fmt doesn't match state.output_format", function);
        // sstanisic fixme: LLK_SAN_ERROR_ASSERT(state.face_height.assert_cond(face_height), "{} : face_height doesn't match state.face_height", function);
        LLK_SAN_ERROR_ASSERT(state.tile_width.assert_cond(tile_width), "{} : tile_width doesn't match state.tile_width", function);
        LLK_SAN_ERROR_ASSERT(state.num_faces.assert_cond(num_faces), "{} : num_faces doesn't match state.num_faces", function);
        LLK_SAN_ERROR_ASSERT(state.partial_face.assert_cond(partial_face), "{} : partial_face doesn't match state.partial_face", function);
        LLK_SAN_ERROR_ASSERT(state.narrow_tile.assert_cond(narrow_tile), "{} : narrow_tile doesn't match state.narrow_tile", function);
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
void operation_check_impl(ct_string function, bool silent, OperationState& state, const Ts... args)
{
    if (!silent)
    {
        LLK_SAN_ERROR_PANIC(state.operation != op, "{} : operation type doesn't match stored operation", function);

        constexpr std::uint8_t args_count = _args_count<Ts...>();

        constexpr std::array<std::uint8_t, args_count> args_sizeof  = _args_sizeof<Ts...>();
        constexpr std::array<std::uint8_t, args_count> args_alignof = _args_alignof<Ts...>();
        constexpr std::array<size_t, args_count + 1> args_offsetof  = _args_offsetof(args_sizeof, args_alignof);

        constexpr size_t entry_size = _operation_entry_size<args_count>(args_sizeof, args_alignof, args_offsetof);

        static_assert(entry_size <= OperationState::BUFFER_SIZE, "llk::san | fault    | operation entry will overflow the buffer");

        // | ARG_COUNT | SIZEOF(args[0]) ... | ALIGNOF(args[1]) ... | args[0] PADDING ... |

        char* ptr = state.buffer;

        LLK_SAN_FAULT_ASSERT(std::memcmp(&args_count, ptr, sizeof(args_count)) == 0, "{} : saved vs provided args_count mismatch", function);
        ptr += sizeof(args_count);

        if constexpr (args_count > 0)
        {
            LLK_SAN_FAULT_ASSERT(
                std::memcmp(args_sizeof.data(), ptr, args_count * sizeof(args_sizeof[0])) == 0, "{} : saved vs provided args_sizeof mismatch", function);
            ptr += args_count * sizeof(args_sizeof[0]);

            LLK_SAN_FAULT_ASSERT(
                std::memcmp(args_alignof.data(), ptr, args_count * sizeof(args_alignof[0])) == 0, "{} : saved vs provided args_alignof mismatch", function);
            ptr += args_count * sizeof(args_alignof[0]);

            constexpr size_t max_align = alignof(max_align_t);
            size_t padding             = (max_align - reinterpret_cast<uintptr_t>(ptr) % max_align) % max_align;
            ptr += padding;

            [[maybe_unused]] size_t i = 0;
            ([&]([[maybe_unused]] const auto& arg)
             { LLK_SAN_ERROR_ASSERT(std::memcmp(ptr + args_offsetof[i++], &arg, sizeof(arg)) == 0, "{} : saved vs provided args mismatch", function); }(args),
             ...);
        }
    }
}

// Goes in LLK_LIB in Uninit
// Check operation type and clear must uninit flag
template <Operation op>
void operation_uninit_impl(ct_string function, bool silent, OperationState& state)
{
    if (!silent)
    {
        LLK_SAN_ERROR_PANIC(state.operation != op, "{} : tried to uninit wrong operation type", function);
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
    UNREACHABLE();
}

template <FsmState next>
void fsm_advance_impl(ct_string function, bool silent, FsmState& current, [[maybe_unused]] const OperationState& operation)
{
    ct_string next_name = fsm_state_name(next);

    if (!silent)
    {
        LLK_SAN_ERROR_PANIC(
            current == FsmState::INITIAL && next != FsmState::CONFIGURED, "{} : initial -> {}, expected first operation to be configure", function, next_name);

        LLK_SAN_ERROR_PANIC(
            current == FsmState::CONFIGURED && next != FsmState::INITIALIZED, "{} : configured -> {}, expected init after configure", function, next_name);

        LLK_SAN_ERROR_PANIC(
            current == FsmState::INITIALIZED && next != FsmState::EXECUTED, "{} : initialized -> {}, expected execute after init", function, next_name);

        LLK_SAN_ERROR_PANIC(
            current == FsmState::EXECUTED && operation.expect_uninit && (next != FsmState::UNINITIALIZED && next != FsmState::EXECUTED),
            "{} : executed -> {}, expected execute or uninit after execute",
            function,
            next_name);

        // shouldn't be possible to trigger, sanity check
        LLK_SAN_FAULT_PANIC(
            current == FsmState::EXECUTED && !operation.expect_uninit && next == FsmState::UNINITIALIZED,
            "{} : executed -> {}, unexpected uninit after execute that doesn't require uninit",
            function,
            next_name);

        LLK_SAN_ERROR_PANIC(
            current == FsmState::EXECUTED && !operation.expect_uninit &&
                (next != FsmState::EXECUTED && next != FsmState::INITIALIZED && next != FsmState::RECONFIGURED),
            "{} : executed -> {}, expected execute, init or reconfig operation after execute",
            function,
            next_name);

        LLK_SAN_ERROR_PANIC(
            current == FsmState::UNINITIALIZED && (next != FsmState::INITIALIZED && next != FsmState::RECONFIGURED),
            "{} : uninitialized -> {}, expected init or reconfigure after uninit",
            function,
            next_name);
    }

    // valid transition -> commit
    current = next;
}

} // namespace llk::san
