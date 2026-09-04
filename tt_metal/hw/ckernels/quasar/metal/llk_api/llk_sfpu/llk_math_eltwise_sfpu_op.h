// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_binary_sfpu.h"
#include "llk_math_eltwise_ternary_sfpu.h"

/*
 * SFPU dispatch layer, Quasar variant. See the wormhole_b0 header for the full description; the class
 * surface (SfpuOpBase, SfpuUnaryOp/SfpuBinaryOp/SfpuTernaryOp, SfpuUnaryFn/SfpuBinaryFn/SfpuTernaryFn) and
 * the init contract (shared _llk_math_eltwise_sfpu_init_(), then Derived::init_kernel(args...)) are
 * identical. Differences: dest bounds use the compile-time
 * trisc::get_dest_max_tiles<DST_SYNC, DST_ACCUM, TILE_SHAPE> (so DST_ACCUM is live here), vector modes are
 * validated, and TILE_SHAPE is forwarded to _llk_math_eltwise_*_sfpu_params_<TILE_SHAPE>.
 */

namespace ckernel {

template <DstSync DST_SYNC, bool DST_ACCUM, trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
struct SfpuOpBase {
    static constexpr DstSync dst_sync = DST_SYNC;
    static constexpr bool dst_accum = DST_ACCUM;
    static constexpr trisc::DstTileShape tile_shape = TILE_SHAPE;

    inline __attribute__((always_inline)) static void check_dst_index(
        [[maybe_unused]] std::uint32_t dst_index, [[maybe_unused]] const char* message) {
        LLK_ASSERT((dst_index < trisc::get_dest_max_tiles<DST_SYNC, DST_ACCUM, TILE_SHAPE>()), message);
    }

    // Unary ops additionally accept RC_custom; binary/ternary do not.
    template <bool ALLOW_RC_CUSTOM>
    inline __attribute__((always_inline)) static void check_vector_mode([[maybe_unused]] VectorMode vector_mode) {
        LLK_ASSERT(
            vector_mode == VectorMode::R || vector_mode == VectorMode::C || vector_mode == VectorMode::RC ||
                vector_mode == VectorMode::None || (ALLOW_RC_CUSTOM && vector_mode == VectorMode::RC_custom),
            "Quasar SFPU supports vector modes R, C, RC, None (and RC_custom for unary ops)");
    }

    // Default op-specific init: nothing beyond the shared SFPU init. A derived op that needs more
    // (ADDR_MOD_6, LUT / constant programming, replay buffers, ...) defines its own init_kernel, which
    // hides this one.
    inline __attribute__((always_inline)) static void init_kernel() {}
};

// ---------------------------------------------------------------------------------------------------
// Unary: one dest tile in place.
// ---------------------------------------------------------------------------------------------------
template <
    class Derived,
    DstSync DST_SYNC,
    bool DST_ACCUM,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
struct SfpuUnaryOp : SfpuOpBase<DST_SYNC, DST_ACCUM, TILE_SHAPE> {
    using Base = SfpuOpBase<DST_SYNC, DST_ACCUM, TILE_SHAPE>;

    template <class... A>
    inline __attribute__((always_inline)) static void calculate(
        std::uint32_t dst_index, VectorMode vector_mode, A&&... args) {
        Base::check_dst_index(dst_index, "dst_index exceeds max dest tiles");
        Base::template check_vector_mode<true>(vector_mode);
        _llk_math_eltwise_unary_sfpu_params_<TILE_SHAPE>(
            [](auto&&... x) __attribute__((always_inline)) { Derived::kernel(static_cast<decltype(x)&&>(x)...); },
            dst_index,
            vector_mode,
            std::forward<A>(args)...);
    }

    template <class... A>
    inline __attribute__((always_inline)) static void init(A&&... args) {
        _llk_math_eltwise_sfpu_init_();
        Derived::init_kernel(std::forward<A>(args)...);
    }
};

// ---------------------------------------------------------------------------------------------------
// Binary: in0, in1 -> out. The kernel receives the three tile indices ahead of its own arguments.
// ---------------------------------------------------------------------------------------------------
template <
    class Derived,
    DstSync DST_SYNC,
    bool DST_ACCUM,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
struct SfpuBinaryOp : SfpuOpBase<DST_SYNC, DST_ACCUM, TILE_SHAPE> {
    using Base = SfpuOpBase<DST_SYNC, DST_ACCUM, TILE_SHAPE>;

    template <class... A>
    inline __attribute__((always_inline)) static void calculate(
        std::uint32_t dst_index_in0,
        std::uint32_t dst_index_in1,
        std::uint32_t dst_index_out,
        VectorMode vector_mode,
        A&&... args) {
        Base::check_dst_index(dst_index_in0, "dst_index_in0 exceeds max dest tiles");
        Base::check_dst_index(dst_index_in1, "dst_index_in1 exceeds max dest tiles");
        Base::check_dst_index(dst_index_out, "dst_index_out exceeds max dest tiles");
        Base::template check_vector_mode<false>(vector_mode);
        _llk_math_eltwise_binary_sfpu_params_<TILE_SHAPE>(
            [](auto&&... x) __attribute__((always_inline)) { Derived::kernel(static_cast<decltype(x)&&>(x)...); },
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            vector_mode,
            std::forward<A>(args)...);
    }

    template <class... A>
    inline __attribute__((always_inline)) static void init(A&&... args) {
        _llk_math_eltwise_sfpu_init_();
        Derived::init_kernel(std::forward<A>(args)...);
    }
};

// ---------------------------------------------------------------------------------------------------
// Ternary: in0, in1, in2 -> out.
// ---------------------------------------------------------------------------------------------------
template <
    class Derived,
    DstSync DST_SYNC,
    bool DST_ACCUM,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
struct SfpuTernaryOp : SfpuOpBase<DST_SYNC, DST_ACCUM, TILE_SHAPE> {
    using Base = SfpuOpBase<DST_SYNC, DST_ACCUM, TILE_SHAPE>;

    template <class... A>
    inline __attribute__((always_inline)) static void calculate(
        std::uint32_t dst_index_in0,
        std::uint32_t dst_index_in1,
        std::uint32_t dst_index_in2,
        std::uint32_t dst_index_out,
        VectorMode vector_mode,
        A&&... args) {
        Base::check_dst_index(dst_index_in0, "dst_index_in0 exceeds max dest tiles");
        Base::check_dst_index(dst_index_in1, "dst_index_in1 exceeds max dest tiles");
        Base::check_dst_index(dst_index_in2, "dst_index_in2 exceeds max dest tiles");
        Base::check_dst_index(dst_index_out, "dst_index_out exceeds max dest tiles");
        Base::template check_vector_mode<false>(vector_mode);
        _llk_math_eltwise_ternary_sfpu_params_<TILE_SHAPE>(
            [](auto&&... x) __attribute__((always_inline)) { Derived::kernel(static_cast<decltype(x)&&>(x)...); },
            dst_index_in0,
            dst_index_in1,
            dst_index_in2,
            dst_index_out,
            vector_mode,
            std::forward<A>(args)...);
    }

    template <class... A>
    inline __attribute__((always_inline)) static void init(A&&... args) {
        _llk_math_eltwise_sfpu_init_();
        Derived::init_kernel(std::forward<A>(args)...);
    }
};

// ---------------------------------------------------------------------------------------------------
// Generic adapters: wrap an already fully specialised kernel (function pointer) as a derived op.
// For kernels without a dedicated op struct. INIT_KERNEL (optional) runs after the shared init when
// init() is called; with the default nullptr, init() takes no arguments and does only the shared init.
// ---------------------------------------------------------------------------------------------------
template <
    auto KERNEL,
    DstSync DST_SYNC,
    bool DST_ACCUM,
    auto INIT_KERNEL = nullptr,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
struct SfpuUnaryFn
    : SfpuUnaryOp<SfpuUnaryFn<KERNEL, DST_SYNC, DST_ACCUM, INIT_KERNEL, TILE_SHAPE>, DST_SYNC, DST_ACCUM, TILE_SHAPE> {
    template <class... A>
    inline __attribute__((always_inline)) static void init_kernel(A&&... args) {
        if constexpr (INIT_KERNEL != nullptr) {
            INIT_KERNEL(std::forward<A>(args)...);
        } else {
            static_assert(sizeof...(A) == 0, "this Sfpu*Fn adapter has no INIT_KERNEL; init() takes no arguments");
        }
    }

    template <class... A>
    inline __attribute__((always_inline)) static void kernel(A&&... args) {
        KERNEL(std::forward<A>(args)...);
    }
};

template <
    auto KERNEL,
    DstSync DST_SYNC,
    bool DST_ACCUM,
    auto INIT_KERNEL = nullptr,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
struct SfpuBinaryFn : SfpuBinaryOp<
                          SfpuBinaryFn<KERNEL, DST_SYNC, DST_ACCUM, INIT_KERNEL, TILE_SHAPE>,
                          DST_SYNC,
                          DST_ACCUM,
                          TILE_SHAPE> {
    template <class... A>
    inline __attribute__((always_inline)) static void init_kernel(A&&... args) {
        if constexpr (INIT_KERNEL != nullptr) {
            INIT_KERNEL(std::forward<A>(args)...);
        } else {
            static_assert(sizeof...(A) == 0, "this Sfpu*Fn adapter has no INIT_KERNEL; init() takes no arguments");
        }
    }

    template <class... A>
    inline __attribute__((always_inline)) static void kernel(A&&... args) {
        KERNEL(std::forward<A>(args)...);
    }
};

template <
    auto KERNEL,
    DstSync DST_SYNC,
    bool DST_ACCUM,
    auto INIT_KERNEL = nullptr,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
struct SfpuTernaryFn : SfpuTernaryOp<
                           SfpuTernaryFn<KERNEL, DST_SYNC, DST_ACCUM, INIT_KERNEL, TILE_SHAPE>,
                           DST_SYNC,
                           DST_ACCUM,
                           TILE_SHAPE> {
    template <class... A>
    inline __attribute__((always_inline)) static void init_kernel(A&&... args) {
        if constexpr (INIT_KERNEL != nullptr) {
            INIT_KERNEL(std::forward<A>(args)...);
        } else {
            static_assert(sizeof...(A) == 0, "this Sfpu*Fn adapter has no INIT_KERNEL; init() takes no arguments");
        }
    }

    template <class... A>
    inline __attribute__((always_inline)) static void kernel(A&&... args) {
        KERNEL(std::forward<A>(args)...);
    }
};

}  // namespace ckernel
