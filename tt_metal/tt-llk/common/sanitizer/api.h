// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

#include "sanitizer/settings.h"
#include "sanitizer/types.h"

#if defined(LLK_SAN_ENABLE)

#include "sanitizer/extended.h"
#include "sanitizer/impl.h"

namespace llk::san
{

// per thread state
extern SanitizerState* const sanitizer;

// Goes in ComputeAPI
// State set only
// sstanisic todo: implement support_backtrace_impl
// static inline void support_backtrace(std::string function_name)
// {
//     support_backtrace_impl(function_name);
// }

// Goes in LLK_API
// State set only
// sstanisic todo: implement support_globals_impl
// static inline void support_globals(bool dst_acc_mode, DstSync dst_sync, bool approx, std::int32_t math_fidelity)
// {
//     support_globals_impl(dst_acc_mode, dst_sync, approx, math_fidelity);
// }

// State set only
// sstanisic todo: implement support_operand_impl
// template <llk_san_operand operand>
// static inline void llk_san_support_operand(
//     std::int32_t src_fmt,
//     std::int32_t dst_fmt,
//     std::int32_t num_faces,
//     std::int32_t partial_face,
//     std::int32_t face_r_dim,
//     std::int32_t narrow_tile,
//     std::int32_t tile_r_dim,
//     std::int32_t tile_c_dim,
//     std::int32_t tile_size,
//     std::int32_t page_size)
// {
//     support_operand_impl<operand>(
//         src_fmt, dst_fmt, num_faces, partial_face, face_r_dim, narrow_tile, tile_r_dim, tile_c_dim, tile_size, page_size);
// }

static inline void thread_init()
{
    thread_init_impl(*sanitizer);
}

static inline void thread_silent_push()
{
    thread_silent_push_impl(*sanitizer);
}

static inline void thread_silent_pop()
{
    thread_silent_pop_impl(*sanitizer);
}

static inline void thread_function_push(const ct_string function)
{
    thread_function_push_impl(*sanitizer, function);
}

static inline void thread_function_pop()
{
    thread_function_pop_impl(*sanitizer);
}

// Goes in LLK_LIB in HWConfigure and HWReconfig
// State set + no hw config within kernel check
template <bool reconfig = false>
static inline void unpack_operand_configure(
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
    if constexpr (!reconfig)
    {
        fsm_advance_impl<FsmState::CONFIGURED>(
            sanitizer->function_curr[COMPILE_FOR_TRISC],
            thread_silent_get_impl(*sanitizer),
            sanitizer->fsm[COMPILE_FOR_TRISC],
            sanitizer->operation[COMPILE_FOR_TRISC]);
    }
    else
    {
        fsm_advance_impl<FsmState::RECONFIGURED>(
            sanitizer->function_curr[COMPILE_FOR_TRISC],
            thread_silent_get_impl(*sanitizer),
            sanitizer->fsm[COMPILE_FOR_TRISC],
            sanitizer->operation[COMPILE_FOR_TRISC]);
    }

    unpack_operand_configure_impl<reconfig>(
        sanitizer->operand.unpack, dst_acc_en, src_fmt_A, src_fmt_B, dst_fmt_A, dst_fmt_B, face_height_A, face_height_B, num_faces_A, num_faces_B);
}

// State set + no hw config within kernel check
template <bool reconfig = false>
static inline void math_operand_configure(State<std::uint32_t> math_fmt_A, State<std::uint32_t> math_fmt_B)
{
    if constexpr (!reconfig)
    {
        fsm_advance_impl<FsmState::CONFIGURED>(
            sanitizer->function_curr[COMPILE_FOR_TRISC],
            thread_silent_get_impl(*sanitizer),
            sanitizer->fsm[COMPILE_FOR_TRISC],
            sanitizer->operation[COMPILE_FOR_TRISC]);
    }
    else
    {
        fsm_advance_impl<FsmState::RECONFIGURED>(
            sanitizer->function_curr[COMPILE_FOR_TRISC],
            thread_silent_get_impl(*sanitizer),
            sanitizer->fsm[COMPILE_FOR_TRISC],
            sanitizer->operation[COMPILE_FOR_TRISC]);
    }

    math_operand_configure_impl<reconfig>(sanitizer->operand.math, math_fmt_A, math_fmt_B);
}

// State set + no hw config within kernel check
template <bool reconfig = false>
static inline void pack_operand_configure(
    State<bool> dest_acc_en,
    State<std::uint32_t> src_fmt,
    State<std::uint32_t> dst_fmt,
    State<std::uint32_t> face_height,
    State<std::uint32_t> tile_width,
    State<std::uint32_t> num_faces,
    State<bool> partial_face,
    State<bool> narrow_tile)
{
    if constexpr (!reconfig)
    {
        fsm_advance_impl<FsmState::CONFIGURED>(
            sanitizer->function_curr[COMPILE_FOR_TRISC],
            thread_silent_get_impl(*sanitizer),
            sanitizer->fsm[COMPILE_FOR_TRISC],
            sanitizer->operation[COMPILE_FOR_TRISC]);
    }
    else
    {
        fsm_advance_impl<FsmState::RECONFIGURED>(
            sanitizer->function_curr[COMPILE_FOR_TRISC],
            thread_silent_get_impl(*sanitizer),
            sanitizer->fsm[COMPILE_FOR_TRISC],
            sanitizer->operation[COMPILE_FOR_TRISC]);
    }

    pack_operand_configure_impl<reconfig>(
        sanitizer->operand.pack, dest_acc_en, src_fmt, dst_fmt, face_height, tile_width, num_faces, partial_face, narrow_tile);
}

// Goes in LLK_LIB in Init, Execute and Uninit
// No state set, just check that non x arguments match the stored ones
static inline void unpack_operand_check(
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
    unpack_operand_check_impl(
        sanitizer->function_curr[COMPILE_FOR_TRISC],
        thread_silent_get_impl(*sanitizer),
        sanitizer->operand.unpack,
        dst_acc_en,
        src_fmt_A,
        src_fmt_B,
        dst_fmt_A,
        dst_fmt_B,
        face_height_A,
        face_height_B,
        num_faces_A,
        num_faces_B);
}

// No state set, just check that non x arguments match the stored ones
static inline void math_operand_check(State<std::uint32_t> math_fmt_A, State<std::uint32_t> math_fmt_B)
{
    math_operand_check_impl(sanitizer->function_curr[COMPILE_FOR_TRISC], thread_silent_get_impl(*sanitizer), sanitizer->operand.math, math_fmt_A, math_fmt_B);
}

// No state set, just check that non x arguments match the stored ones
static inline void pack_operand_check(
    State<bool> dest_acc_en,
    State<std::uint32_t> src_fmt,
    State<std::uint32_t> dst_fmt,
    State<std::uint32_t> face_height,
    State<std::uint32_t> tile_width,
    State<std::uint32_t> num_faces,
    State<bool> partial_face,
    State<bool> narrow_tile)
{
    pack_operand_check_impl(
        sanitizer->function_curr[COMPILE_FOR_TRISC],
        thread_silent_get_impl(*sanitizer),
        sanitizer->operand.pack,
        dest_acc_en,
        src_fmt,
        dst_fmt,
        face_height,
        tile_width,
        num_faces,
        partial_face,
        narrow_tile);
}

// Goes in LLK_LIB in Init
// Store operation type and save arguments
template <Operation op, typename... Ts>
static inline void operation_init(Ts... args)
{
    fsm_advance_impl<FsmState::INITIALIZED>(
        sanitizer->function_curr[COMPILE_FOR_TRISC],
        thread_silent_get_impl(*sanitizer),
        sanitizer->fsm[COMPILE_FOR_TRISC],
        sanitizer->operation[COMPILE_FOR_TRISC]);

    operation_init_impl<op, Ts...>(sanitizer->operation[COMPILE_FOR_TRISC], args...);
}

// Goes in LLK_LIB in Execute
// Check operation type and arguments against stored ones
template <Operation op, typename... Ts>
static inline void operation_check(Ts... args)
{
    const bool silent = thread_silent_get_impl(*sanitizer);
    fsm_advance_impl<FsmState::EXECUTED>(
        sanitizer->function_curr[COMPILE_FOR_TRISC], silent, sanitizer->fsm[COMPILE_FOR_TRISC], sanitizer->operation[COMPILE_FOR_TRISC]);
    operation_check_impl<op, Ts...>(sanitizer->function_curr[COMPILE_FOR_TRISC], silent, sanitizer->operation[COMPILE_FOR_TRISC], args...);
}

// Goes in LLK_LIB in Uninit
// Check operation type and clear must uninit flag
template <Operation op>
void operation_uninit()
{
    const bool silent = thread_silent_get_impl(*sanitizer);
    fsm_advance_impl<FsmState::UNINITIALIZED>(
        sanitizer->function_curr[COMPILE_FOR_TRISC], silent, sanitizer->fsm[COMPILE_FOR_TRISC], sanitizer->operation[COMPILE_FOR_TRISC]);
    operation_uninit_impl<op>(sanitizer->function_curr[COMPILE_FOR_TRISC], silent, sanitizer->operation[COMPILE_FOR_TRISC]);
}

class FunctionZone
{
public:
    FunctionZone(const ct_string function_name)
    {
        thread_function_push(function_name);
    }

    ~FunctionZone()
    {
        thread_function_pop();
    }
};

class SilentZone
{
public:
    SilentZone()
    {
        thread_silent_push();
    }

    ~SilentZone()
    {
        thread_silent_pop();
    }
};

} // namespace llk::san

#define LLK_SAN_FUNCTION(function)         \
    llk::san::FunctionZone _function_zone_ \
    {                                      \
        CTSTR(function)                    \
    }

#define LLK_SAN_SILENT_ZONE() [[maybe_unused]] llk::san::SilentZone _silent_zone_

#else

namespace llk::san
{

static inline void thread_init()
{
}

template <bool reconfig = false>
static inline void unpack_operand_configure(
    [[maybe_unused]] State<bool> dst_acc_en,
    [[maybe_unused]] State<std::uint32_t> src_fmt_A,
    [[maybe_unused]] State<std::uint32_t> src_fmt_B,
    [[maybe_unused]] State<std::uint32_t> dst_fmt_A,
    [[maybe_unused]] State<std::uint32_t> dst_fmt_B,
    [[maybe_unused]] State<std::uint32_t> face_height_A,
    [[maybe_unused]] State<std::uint32_t> face_height_B,
    [[maybe_unused]] State<std::uint32_t> num_faces_A,
    [[maybe_unused]] State<std::uint32_t> num_faces_B)
{
}

template <bool reconfig = false>
static inline void math_operand_configure([[maybe_unused]] State<std::uint32_t> math_fmt_A, [[maybe_unused]] State<std::uint32_t> math_fmt_B)
{
}

template <bool reconfig = false>
static inline void pack_operand_configure(
    [[maybe_unused]] State<bool> dest_acc_en,
    [[maybe_unused]] State<std::uint32_t> src_fmt,
    [[maybe_unused]] State<std::uint32_t> dst_fmt,
    [[maybe_unused]] State<std::uint32_t> face_height,
    [[maybe_unused]] State<std::uint32_t> tile_width,
    [[maybe_unused]] State<std::uint32_t> num_faces,
    [[maybe_unused]] State<bool> partial_face,
    [[maybe_unused]] State<bool> narrow_tile)
{
}

static inline void unpack_operand_check(
    [[maybe_unused]] State<bool> dst_acc_en,
    [[maybe_unused]] State<std::uint32_t> src_fmt_A,
    [[maybe_unused]] State<std::uint32_t> src_fmt_B,
    [[maybe_unused]] State<std::uint32_t> dst_fmt_A,
    [[maybe_unused]] State<std::uint32_t> dst_fmt_B,
    [[maybe_unused]] State<std::uint32_t> face_height_A,
    [[maybe_unused]] State<std::uint32_t> face_height_B,
    [[maybe_unused]] State<std::uint32_t> num_faces_A,
    [[maybe_unused]] State<std::uint32_t> num_faces_B)
{
}

static inline void math_operand_check([[maybe_unused]] State<std::uint32_t> math_fmt_A, [[maybe_unused]] State<std::uint32_t> math_fmt_B)
{
}

static inline void pack_operand_check(
    [[maybe_unused]] State<bool> dest_acc_en,
    [[maybe_unused]] State<std::uint32_t> src_fmt,
    [[maybe_unused]] State<std::uint32_t> dst_fmt,
    [[maybe_unused]] State<std::uint32_t> face_height,
    [[maybe_unused]] State<std::uint32_t> tile_width,
    [[maybe_unused]] State<std::uint32_t> num_faces,
    [[maybe_unused]] State<bool> partial_face,
    [[maybe_unused]] State<bool> narrow_tile)
{
}

template <Operation op, typename... Ts>
static inline void operation_init([[maybe_unused]] Ts... args)
{
}

template <Operation op, typename... Ts>
static inline void operation_check([[maybe_unused]] Ts... args)
{
}

template <Operation op>
void operation_uninit()
{
}

} // namespace llk::san

#define LLK_SAN_FUNCTION(function) \
    do                             \
    {                              \
    } while (false)

#define LLK_SAN_SILENT_ZONE() \
    do                        \
    {                         \
    } while (false)

#endif
