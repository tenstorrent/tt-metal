// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include "sanitizer/settings.h"
#include "sanitizer/types.h"

#if defined(LLK_SAN_ENABLE)

#include "sanitizer/impl.h"

namespace llk::san
{

// per thread state
extern SanitizerState* const sanitizer;

static inline void thread_init()
{
    thread_init_impl(*sanitizer);
}

static inline auto& thread_context_get()
{
    return thread_context_get_impl(*sanitizer);
}

static inline void thread_silent_push()
{
    thread_silent_push_impl(thread_context_get());
}

static inline void thread_silent_pop()
{
    thread_silent_pop_impl(thread_context_get());
}

static inline void thread_context_push()
{
    thread_context_push_impl(thread_context_get());
}

static inline void thread_context_pop()
{
    thread_context_pop_impl(thread_context_get());
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
        fsm_configure_impl(thread_context_get(), sanitizer->fsm[COMPILE_FOR_TRISC]);
    }
    else
    {
        fsm_reconfigure_impl(thread_context_get(), sanitizer->fsm[COMPILE_FOR_TRISC]);
    }

    unpack_operand_configure_impl<reconfig>(
        sanitizer->context.unpack,
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

// State set + no hw config within kernel check
template <bool reconfig = false>
static inline void math_operand_configure(State<std::uint32_t> math_fmt_A, State<std::uint32_t> math_fmt_B)
{
    if constexpr (!reconfig)
    {
        fsm_configure_impl(thread_context_get(), sanitizer->fsm[COMPILE_FOR_TRISC]);
    }
    else
    {
        fsm_reconfigure_impl(thread_context_get(), sanitizer->fsm[COMPILE_FOR_TRISC]);
    }

    math_operand_configure_impl<reconfig>(sanitizer->context.math, sanitizer->operand.math, math_fmt_A, math_fmt_B);
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
        fsm_configure_impl(thread_context_get(), sanitizer->fsm[COMPILE_FOR_TRISC]);
    }
    else
    {
        fsm_reconfigure_impl(thread_context_get(), sanitizer->fsm[COMPILE_FOR_TRISC]);
    }

    pack_operand_configure_impl<reconfig>(
        sanitizer->context.pack, sanitizer->operand.pack, dest_acc_en, src_fmt, dst_fmt, face_height, tile_width, num_faces, partial_face, narrow_tile);
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
        sanitizer->context.unpack,
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
    math_operand_check_impl(sanitizer->context.math, sanitizer->operand.math, math_fmt_A, math_fmt_B);
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
        sanitizer->context.pack, sanitizer->operand.pack, dest_acc_en, src_fmt, dst_fmt, face_height, tile_width, num_faces, partial_face, narrow_tile);
}

// Goes in LLK_LIB in Init
// Store operation type and save arguments
template <Operation op, typename... Ts>
static inline void operation_init(Ts&&... args)
{
    const bool fsm_success = fsm_init_impl(thread_context_get(), sanitizer->fsm[COMPILE_FOR_TRISC], op);

    if (!fsm_success)
    {
        thread_silent_push();
    }

    operation_init_impl(thread_context_get(), sanitizer->operation[COMPILE_FOR_TRISC], op, std::forward<Ts>(args)...);

    if (!fsm_success)
    {
        thread_silent_pop();
    }
}

// Goes in LLK_LIB in Execute
// Check operation type and arguments against stored ones
template <Operation op, typename... Ts>
static inline void operation_check(Ts&&... args)
{
    const bool fsm_success = fsm_execute_impl(thread_context_get(), sanitizer->fsm[COMPILE_FOR_TRISC], op);

    if (!fsm_success)
    {
        thread_silent_push();
    }

    operation_execute_impl(thread_context_get(), sanitizer->operation[COMPILE_FOR_TRISC], op, std::forward<Ts>(args)...);

    if (!fsm_success)
    {
        thread_silent_pop();
    }
}

// Goes in LLK_LIB in Uninit
// Check operation type and clear must uninit flag
template <Operation op>
void operation_uninit()
{
    const bool fsm_success = fsm_uninit_impl(thread_context_get(), sanitizer->fsm[COMPILE_FOR_TRISC], op);

    if (!fsm_success)
    {
        thread_silent_push();
    }

    operation_uninit_impl(thread_context_get(), sanitizer->operation[COMPILE_FOR_TRISC], op);

    if (!fsm_success)
    {
        thread_silent_pop();
    }
}

class FunctionZone
{
public:
    FunctionZone()
    {
        thread_context_push();
    }

    ~FunctionZone()
    {
        thread_context_pop();
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

#define LLK_SAN_FUNCTION() llk::san::FunctionZone _function_zone_

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
static inline void operation_init([[maybe_unused]] Ts&&... args)
{
}

template <Operation op, typename... Ts>
static inline void operation_check([[maybe_unused]] Ts&&... args)
{
}

template <Operation op>
void operation_uninit()
{
}

} // namespace llk::san

#define LLK_SAN_FUNCTION() \
    do                     \
    {                      \
    } while (false)

#define LLK_SAN_SILENT_ZONE() \
    do                        \
    {                         \
    } while (false)

#endif
