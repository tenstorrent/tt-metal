// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include "llk_assert.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "llk_math_eltwise_ternary_sfpu_params.h"

/*
 * SFPU dispatch layer: static-only CRTP classes that bracket an SFPU kernel with the dest-index /
 * vector-mode preconditions and the tt-llk start / apply-vector-mode / done sequence, and that run the
 * op's init.
 *
 *   SfpuOpBase<DST_SYNC, DST_ACCUM, TILE_SHAPE>          preconditions + default init hook, shared by every arity
 *   SfpuUnaryOp<Derived, ...>   : SfpuOpBase              calculate() / init() for one dest tile in place
 *   SfpuBinaryOp<Derived, ...>  : SfpuOpBase              calculate() / init() for in0, in1 -> out
 *   SfpuTernaryOp<Derived, ...> : SfpuOpBase              calculate() / init() for in0, in1, in2 -> out
 *
 * A concrete op is a struct template whose template parameters are *all* of the op's compile-time
 * configuration (approximation mode, data format, DST_SYNC, DST_ACCUM, iteration count, ...). It derives
 * from one of the three bases (CRTP) and supplies:
 *
 *   static void kernel(<dst indices for binary/ternary>, runtime_args...);   // per-face body
 *   static void init_kernel(runtime_args...);   // optional: op-specific init, shadows the base's no-op
 *
 * init() always runs the shared SFPU init first (_llk_math_eltwise_sfpu_init_: SFPU config register,
 * ADDR_MOD_7 = {0,0,0}, RWC counter reset) and then Derived::init_kernel(args...). An op that needs any
 * further hardware state -- most commonly ADDR_MOD_6 with a non-zero dest increment so its SFPSTOREs
 * auto-advance -- programs it inside its own init_kernel, e.g.
 *
 *   static void init_kernel() {
 *       addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
 *   }
 *
 * The base classes carry no per-op knowledge: there is no op enumeration and no central init table.
 * Calling init(args...) on an op whose init_kernel takes no arguments is a compile error.
 *
 * Op structs live next to their kernels in this directory's ckernel_sfpu_<op>.h, one per arch, so the
 * per-arch differences in kernel template lists stay out of the compute API headers. The struct's
 * template parameter list is the same on every arch; an arch that does not use a parameter ignores it.
 *
 * calculate() and init() are plain (non-template) static members taking only runtime arguments, so a
 * call site inside a templated compute-API function needs no `template` keyword:
 *
 *   sfpu::Exp<approx, clamp, DST_SYNC_MODE, is_fp32_dest_acc_en, scale_en, iterations>::calculate(idst, vm, scale);
 *   sfpu::Exp<approx, clamp, DST_SYNC_MODE, is_fp32_dest_acc_en>::init();
 *
 * Kernels that have no op struct (test harnesses, one-off kernels in downstream code) use the generic
 * SfpuUnaryFn / SfpuBinaryFn / SfpuTernaryFn adapters, which take the fully specialised kernel (and
 * optionally its init function) as function-pointer template arguments:
 *
 *   using MyOp = SfpuUnaryFn<sfpu::my_kernel<APPROX, 8>, DST_SYNC_MODE, DST_ACCUM_MODE, sfpu::my_init<APPROX>>;
 *   MyOp::init();
 *   MyOp::calculate(idst, VectorMode::RC, arg);
 *
 * DST_SYNC and DST_ACCUM are plain template parameters: this header never reads DST_SYNC_MODE or
 * DST_ACCUM_MODE. Compute API entry points default their own template parameter to DST_ACCUM_MODE and
 * pass it down, so the kernel-wide define is consulted in exactly one place.
 *
 * Everything is static and always_inline; there is no object, no vtable and no runtime cost.
 */

namespace ckernel {

template <DstSync DST_SYNC, bool DST_ACCUM, DstTileShape TILE_SHAPE = DstTileShape::Tile32x32>
struct SfpuOpBase {
    static constexpr DstSync dst_sync = DST_SYNC;
    static constexpr bool dst_accum = DST_ACCUM;
    static constexpr DstTileShape tile_shape = TILE_SHAPE;

    // WH/BH read the accumulation mode back from hardware so the bound stays correct after
    // enable/disable_fp32_dest_acc; DST_ACCUM is carried for the ops themselves and for Quasar.
    inline __attribute__((always_inline)) static void check_dst_index(
        [[maybe_unused]] std::uint32_t dst_index, [[maybe_unused]] const char* message) {
        LLK_ASSERT((dst_index < get_dest_max_tiles_rt<DST_SYNC, TILE_SHAPE>()), message);
    }

    inline __attribute__((always_inline)) static void check_vector_mode([[maybe_unused]] VectorMode vector_mode) {}

    // Default op-specific init: nothing beyond the shared SFPU init. A derived op that needs more
    // (ADDR_MOD_6, LUT / constant programming, replay buffers, ...) defines its own init_kernel, which
    // hides this one.
    inline __attribute__((always_inline)) static void init_kernel() {}
};

// ---------------------------------------------------------------------------------------------------
// Unary: one dest tile in place.
// ---------------------------------------------------------------------------------------------------
template <class Derived, DstSync DST_SYNC, bool DST_ACCUM, DstTileShape TILE_SHAPE = DstTileShape::Tile32x32>
struct SfpuUnaryOp : SfpuOpBase<DST_SYNC, DST_ACCUM, TILE_SHAPE> {
    using Base = SfpuOpBase<DST_SYNC, DST_ACCUM, TILE_SHAPE>;

    template <class... A>
    inline __attribute__((always_inline)) static void calculate(
        std::uint32_t dst_index, VectorMode vector_mode, A&&... args) {
        Base::check_dst_index(dst_index, "dst_index exceeds max dest tiles");
        Base::check_vector_mode(vector_mode);
        _llk_math_eltwise_unary_sfpu_params_(
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
template <class Derived, DstSync DST_SYNC, bool DST_ACCUM, DstTileShape TILE_SHAPE = DstTileShape::Tile32x32>
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
        Base::check_vector_mode(vector_mode);
        _llk_math_eltwise_binary_sfpu_params_(
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
template <class Derived, DstSync DST_SYNC, bool DST_ACCUM, DstTileShape TILE_SHAPE = DstTileShape::Tile32x32>
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
        Base::check_vector_mode(vector_mode);
        _llk_math_eltwise_ternary_sfpu_params_(
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
    DstTileShape TILE_SHAPE = DstTileShape::Tile32x32>
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
    DstTileShape TILE_SHAPE = DstTileShape::Tile32x32>
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
    DstTileShape TILE_SHAPE = DstTileShape::Tile32x32>
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
