// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Quasar SFPU op types: SfpuOp<OP, REG, SOURCES...> names one op and the route it runs on; its functions run it.
// The tile format comes from SOURCES at compile time. Call init() once, then the per-tile calls.

#pragma once

#ifndef ARCH_QUASAR
#error "api/compute/experimental/sfpu.h is Quasar-only"
#endif

#include <type_traits>

#include "api/compute/common_globals.h"
#include "api/compute/compute_kernel_api.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_ops.h"
#include "sfpu_reg.h"

// The ops, on the two threads that run them. Each op header brings both op types.
#if defined(TRISC_MATH) || defined(TRISC_ISOLATE_SFPU)
#include "ckernel_sfpu_add.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_square.h"
#endif
#ifdef TRISC_ISOLATE_SFPU
#include "llk_sfpu_srcs_api.h"
#endif

namespace ckernel {

using sfpu::DstIndex;
using sfpu::SfpuReg;

// The op tags are in ckernel_sfpu_ops.h; op::Exp<> defaults to the kernel's APPROX constant.
namespace op {

template <bool APPROXIMATION_MODE = APPROX>
struct Exp;

}  // namespace op

namespace detail {

#ifdef TRISC_MATH
// I32 Dest tiles: two's complement when copied from one DFB, sign-magnitude when an FPU op left them.
template <auto... DFB_IDS>
constexpr sfpu::IntEncoding sfpu_dest_int_encoding_of() {
    return sizeof...(DFB_IDS) == 1 ? sfpu::IntEncoding::TwosComplement : sfpu::IntEncoding::SignMagnitude;
}
#endif

#if defined(TRISC_MATH) || defined(TRISC_ISOLATE_SFPU)
// Layout of Dest tiles copied or unpacked from one DFB: its register format on this Dest width. Compile time.
template <auto DFB_ID>
constexpr sfpi::DataLayout sfpu_dest_layout_of() {
    return sfpu::dest_layout<static_cast<DataFormat>(unpack_dst_format[DFB_ID]), DST_ACCUM_MODE>();
}

// Layout of the Dest tiles an FPU op left, from the DFBs it read into SrcA and SrcB. Compile time.
template <auto SRCA_DFB_ID, auto SRCB_DFB_ID>
constexpr sfpi::DataLayout sfpu_dest_layout_of() {
    return sfpu::fpu_dest_layout<
        static_cast<DataFormat>(unpack_dst_format[SRCA_DFB_ID]),
        static_cast<DataFormat>(unpack_dst_format[SRCB_DFB_ID]),
        DST_ACCUM_MODE>();
}

// True when every one of these DFBs holds 32x32 tiles: the only Dest slot an op addresses. Compile time.
template <auto... DFB_IDS>
inline constexpr bool sfpu_dest_tiles_32x32 =
    ((unpack_tile_r_dim[DFB_IDS] == TILE_R_DIM && unpack_tile_c_dim[DFB_IDS] == TILE_C_DIM) && ...);

// One key per op type; its address names the op in sfpu_init_owner.
template <typename OP_TYPE>
inline constexpr char sfpu_op_key = 0;

// Debug check: init() of OP_TYPE was the last SFPU init on this thread. Every SFPU init clears sfpu_init_owner.
template <typename OP_TYPE>
ALWI void sfpu_op_check_init() {
    LLK_ASSERT(sfpu_init_owner == &sfpu_op_key<OP_TYPE>, "SfpuOp: call init() after any other SFPU init");
}
#endif

#ifdef TRISC_ISOLATE_SFPU
// The SrcS op type: every input from L1, or input 0 a Dest tile that came from DEST_DFB_IDS.
template <typename OP, sfpi::DataLayout LAYOUT, auto... DEST_DFB_IDS>
struct sfpu_srcs_op {
    using type = sfpu::srcs_dest_op_t<OP, LAYOUT, sfpu_dest_layout_of<DEST_DFB_IDS...>()>;
};
template <typename OP, sfpi::DataLayout LAYOUT>
struct sfpu_srcs_op<OP, LAYOUT> {
    using type = sfpu::srcs_op_t<OP, LAYOUT>;
};

// Debug check: the SrcS pipeline moves a whole 32x32 tile, so a smaller DFB entry would be overrun.
template <typename... DFBS>
ALWI void sfpu_srcs_check_tiles(const DFBS&... dfbs) {
    LLK_ASSERT(
        ((dfbs.get_tile_r_dim() == TILE_R_DIM && dfbs.get_tile_c_dim() == TILE_C_DIM) && ...),
        "SfpuOp: SrcS takes DFBs of 32x32 tiles");
}
#endif

}  // namespace detail

// clang-format off
/**
 * One SFPU op on one route, on 32x32 tiles. Name it once with a using-alias and call its functions; every function
 * expands to nothing on the threads that do not run the op.
 *
 * | Template Parameter | Description                                                  | Type                  | Valid Range                  | Required |
 * |--------------------|--------------------------------------------------------------|-----------------------|------------------------------|----------|
 * | OP                 | The op tag (namespace op)                                    | struct                | op::*                        | True     |
 * | REG                | Dest: MATH thread. SrcS: the SrcS pipeline, SFPU thread      | SfpuReg               | SfpuReg::Dest, SfpuReg::SrcS | True     |
 * | SOURCES            | Where the tiles came from; gives their format (see each REG) | uint32_t / DataFormat |                              | True     |
 */
// clang-format on
template <typename OP, SfpuReg REG, auto... SOURCES>
struct SfpuOp {
    static_assert(
        sizeof(OP) == 0,
        "SfpuOp<OP, SfpuReg::Dest, dfb ids...> or SfpuOp<OP, SfpuReg::SrcS, SrcS DataFormat[, Dest dfb ids...]>");
};

// clang-format off
/**
 * OP on Dest, on the MATH thread. Call init() before the first tile(), and again after any other SFPU init.
 * SOURCES are compile-time DFB ids (constexpr uint32_t id = dfb::in;):
 *
 * | SOURCES            | The Dest tiles                                                                |
 * |--------------------|-------------------------------------------------------------------------------|
 * | DFB_ID             | Were copied or unpacked to Dest from this DFB (copy_tile)                     |
 * | SRCA_DFB, SRCB_DFB | Were left by an FPU op that read these DFBs into SrcA and SrcB, in that order |
 */
// clang-format on
template <typename OP, auto... DFB_IDS>
struct SfpuOp<OP, SfpuReg::Dest, DFB_IDS...> {
    static_assert(
        sizeof...(DFB_IDS) == 1 || sizeof...(DFB_IDS) == 2,
        "Dest takes the DFB the tiles were copied from, or the SrcA and SrcB DFBs of the FPU op that left them");
    static_assert((std::is_integral_v<decltype(DFB_IDS)> && ...), "Dest sources are DFB ids");

    // Set up the SFPU for this op.
    ALWI static void init() {
        MATH((Op::init()));
        MATH((sfpu_init_owner = &detail::sfpu_op_key<Op>));
    }

    // clang-format off
    /**
     * Run a one-input op on one Dest tile, in place. DST must be acquired via *acquire_dst*.
     *
     * Return value: None
     *
     * | Argument    | Description                     | Type       | Valid Range                                    | Required |
     * |-------------|---------------------------------|------------|------------------------------------------------|----------|
     * | idst        | The index of the tile in DST    | DstIndex   | Must be less than the acquired size of DST REG | True     |
     * | vector_mode | The faces to compute            | VectorMode | R, C, RC, None, RC_custom                      | False    |
     */
    // clang-format on
    ALWI static void tile(DstIndex idst, VectorMode vector_mode = VectorMode::RC) {
#ifdef TRISC_MATH
        static_assert(inputs == 1, "this op takes two Dest tiles: tile(idst0, idst1, odst)");
#endif
        MATH((detail::sfpu_op_check_init<Op>()));
        MATH((Op::template run<DST_SYNC_MODE, DST_ACCUM_MODE>(idst, vector_mode)));
    }

    // clang-format off
    /**
     * Run a two-input op on two Dest tiles, writing a third (odst may equal idst0 or idst1). DST must be
     * acquired via *acquire_dst*.
     *
     * Return value: None
     *
     * | Argument    | Description                            | Type       | Valid Range                                    | Required |
     * |-------------|----------------------------------------|------------|------------------------------------------------|----------|
     * | idst0       | The index of the first operand in DST  | DstIndex   | Must be less than the acquired size of DST REG | True     |
     * | idst1       | The index of the second operand in DST | DstIndex   | Must be less than the acquired size of DST REG | True     |
     * | odst        | The index of the output tile in DST    | DstIndex   | Must be less than the acquired size of DST REG | True     |
     * | vector_mode | The faces to compute                   | VectorMode | R, C, RC, None, RC_custom                      | False    |
     */
    // clang-format on
    ALWI static void tile(DstIndex idst0, DstIndex idst1, DstIndex odst, VectorMode vector_mode = VectorMode::RC) {
#ifdef TRISC_MATH
        static_assert(inputs == 2, "this op takes one Dest tile: tile(idst)");
#endif
        MATH((detail::sfpu_op_check_init<Op>()));
        MATH((Op::template run<DST_SYNC_MODE, DST_ACCUM_MODE>(idst0, idst1, odst, vector_mode)));
    }

#ifdef TRISC_MATH

private:
    static_assert(detail::sfpu_dest_tiles_32x32<DFB_IDS...>, "SfpuOp takes 32x32 Dest tiles");
    static constexpr sfpi::DataLayout layout = detail::sfpu_dest_layout_of<DFB_IDS...>();
    static constexpr sfpu::IntEncoding int_encoding = detail::sfpu_dest_int_encoding_of<DFB_IDS...>();
    static_assert(sfpu::op_takes_layout<OP, layout>, "this op does not take tiles of this format");
    static constexpr int inputs = sfpu::math_arity<sfpu::math_for_t<OP, layout, int_encoding>>;
    using Op = sfpu::dest_op_t<OP, layout, SFPU_ITERATIONS, trisc::DstTileShape::Tile32x32, int_encoding>;
#endif
};

// clang-format off
/**
 * OP on the SrcS pipeline, on the SFPU thread (TRISC3). Call init() once. Per tile: unpack() and pack() name the L1
 * tiles, then tile() runs the op on them.
 *
 * | Template Parameter | Description                                                           | Type       | Valid Range                        | Required |
 * |--------------------|-----------------------------------------------------------------------|------------|------------------------------------|----------|
 * | SRCS_FMT           | Format in the SrcS register (unpack dst = pack src)                   | DataFormat | Float16, Float16_b, Float32, Int32 | True     |
 * | DEST_DFB_IDS       | Given when input 0 is a Dest tile: the DFBs it came from, as for Dest | uint32_t   |                                    | False    |
 */
// clang-format on
template <typename OP, DataFormat SRCS_FMT, auto... DEST_DFB_IDS>
struct SfpuOp<OP, SfpuReg::SrcS, SRCS_FMT, DEST_DFB_IDS...> {
    // Set up unpacker 2, packer 1 and the SFPU for this op.
    ALWI static void init() {
        SFPU_SRCS((Op::init()));
        SFPU_SRCS((sfpu_init_owner = &detail::sfpu_op_key<Op>));
    }

    // The tile unpacker 2 unpacks into SrcS: the front entry of in. For the one input that comes from L1.
    template <typename DFB>
    ALWI static void unpack([[maybe_unused]] const DFB& in) {
#ifdef TRISC_ISOLATE_SFPU
        static_assert(l1_inputs == 1, "this op takes two inputs from L1: unpack(in0, in1)");
#endif
        SFPU_SRCS((detail::sfpu_srcs_check_tiles(in)));
        SFPU_SRCS((Op::unpack(in.get_read_ptr(), l1_format(in))));
    }

    // The same for two inputs from L1. Both have the same L1 format.
    template <typename DFB>
    ALWI static void unpack([[maybe_unused]] const DFB& in0, [[maybe_unused]] const DFB& in1) {
#ifdef TRISC_ISOLATE_SFPU
        static_assert(l1_inputs == 2, "this op takes one input from L1: unpack(in)");
        LLK_ASSERT(l1_format(in0) == l1_format(in1), "SfpuOp: both inputs must have the same L1 format");
#endif
        SFPU_SRCS((detail::sfpu_srcs_check_tiles(in0, in1)));
        SFPU_SRCS((Op::unpack(in0.get_read_ptr(), in1.get_read_ptr(), l1_format(in0))));
    }

    // The tile packer 1 packs the result to: the back entry of out.
    template <typename DFB>
    ALWI static void pack([[maybe_unused]] const DFB& out) {
        SFPU_SRCS((detail::sfpu_srcs_check_tiles(out)));
        SFPU_SRCS((Op::pack(out.get_write_ptr(), static_cast<DataFormat>(pack_dst_format[out.get_id()]))));
    }

    // Run the op on the tiles unpack() and pack() named: unpack, compute and pack, slice by slice.
    ALWI static void tile() {
        static_assert(sizeof...(DEST_DFB_IDS) == 0, "input 0 of this op is a Dest tile: tile(idst)");
        SFPU_SRCS((detail::sfpu_op_check_init<Op>()));
        SFPU_SRCS((Op::run()));
    }

    // The same with Dest tile idst as input 0. The SFPU thread must hold the Dest loan: tile_regs_borrow.
    ALWI static void tile([[maybe_unused]] DstIndex idst) {
        static_assert(sizeof...(DEST_DFB_IDS) != 0, "name the DFBs the Dest tile came from after the SrcS format");
        SFPU_SRCS((detail::sfpu_op_check_init<Op>()));
        SFPU_SRCS((Op::template run<DST_SYNC_MODE, DST_ACCUM_MODE>(idst)));
    }

#ifdef TRISC_ISOLATE_SFPU

private:
    static_assert(detail::sfpu_dest_tiles_32x32<DEST_DFB_IDS...>, "SfpuOp takes 32x32 Dest tiles");
    static constexpr sfpi::DataLayout layout = sfpu::sfpu_layout<SRCS_FMT>();
    static_assert(sfpu::op_takes_layout<OP, layout>, "this op does not take tiles of this format");
    static constexpr int l1_inputs =
        sfpu::math_arity<sfpu::math_for_t<OP, layout>> - (sizeof...(DEST_DFB_IDS) != 0 ? 1 : 0);
    using Op = typename detail::sfpu_srcs_op<OP, layout, DEST_DFB_IDS...>::type;

    template <typename DFB>
    static DataFormat l1_format(const DFB& in) {
        return static_cast<DataFormat>(unpack_src_format[in.get_id()]);
    }
#endif
};

}  // namespace ckernel
