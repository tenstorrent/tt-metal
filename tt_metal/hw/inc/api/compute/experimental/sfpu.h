// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Quasar SFPU op types: SfpuOp<OP, REG, SOURCES...> names one op on one register file; its functions run it.
// The tile format comes from SOURCES at compile time. Call init() once, then tile() per tile.

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

// I32 Dest tiles: two's complement when copied from one DFB, sign-magnitude when an FPU op left them.
template <auto... DFB_IDS>
constexpr sfpu::IntEncoding sfpu_dest_int_encoding_of() {
    return sizeof...(DFB_IDS) == 1 ? sfpu::IntEncoding::TwosComplement : sfpu::IntEncoding::SignMagnitude;
}

#endif

#if defined(TRISC_MATH) || defined(TRISC_ISOLATE_SFPU)
// One key per op type; its address names the op in sfpu_init_owner.
template <typename OP_TYPE>
inline constexpr char sfpu_op_key = 0;

// Debug check: init() of OP_TYPE was the last SFPU init on this thread. Every SFPU init clears sfpu_init_owner.
template <typename OP_TYPE>
ALWI void sfpu_op_check_init() {
    LLK_ASSERT(sfpu_init_owner == &sfpu_op_key<OP_TYPE>, "SfpuOp: call init() after any other SFPU init");
}
#endif

}  // namespace detail

// clang-format off
/**
 * One SFPU op on one register file. Name it once with a using-alias and call its functions; every function
 * expands to nothing on the threads that do not run the op.
 *
 * | Template Parameter | Description                                                  | Type                  | Valid Range                  | Required |
 * |--------------------|--------------------------------------------------------------|-----------------------|------------------------------|----------|
 * | OP                 | The op tag (namespace op)                                    | struct                | op::*                        | True     |
 * | REG                | The register file the tiles are in                           | SfpuReg               | SfpuReg::Dest, SfpuReg::SrcS | True     |
 * | SOURCES            | Where the tiles came from; gives their format (see each REG) | uint32_t / DataFormat |                              | True     |
 */
// clang-format on
template <typename OP, SfpuReg REG, auto... SOURCES>
struct SfpuOp {
    static_assert(
        sizeof(OP) == 0,
        "SfpuOp<OP, SfpuReg::Dest, dfb ids...> or SfpuOp<OP, SfpuReg::SrcS, SrcS register DataFormat>");
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
        MATH((Op::run(idst, vector_mode)));
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
        MATH((Op::run(idst0, idst1, odst, vector_mode)));
    }

#ifdef TRISC_MATH

private:
    static constexpr sfpi::DataLayout layout = detail::sfpu_dest_layout_of<DFB_IDS...>();
    static constexpr sfpu::IntEncoding int_encoding = detail::sfpu_dest_int_encoding_of<DFB_IDS...>();
    static_assert(sfpu::op_takes_layout<OP, layout>, "this op does not take tiles of this format");
    static constexpr int inputs = sfpu::math_arity<sfpu::math_for_t<OP, layout, int_encoding>>;
    using Op = sfpu::dest_op_t<OP, layout, SFPU_ITERATIONS, trisc::DstTileShape::Tile32x32, int_encoding>;
#endif
};

// SrcS input in L1: an address (16-byte units) and a format, or a DataflowBuffer's front entry.
struct unpack_from {
    uint32_t addr_16B = 0;
    DataFormat format = DataFormat::Invalid;

    constexpr unpack_from(uint32_t l1_addr_16B, DataFormat l1_format) : addr_16B(l1_addr_16B), format(l1_format) {}

    template <typename DFB>
    explicit unpack_from([[maybe_unused]] const DFB& dfb) {
#ifdef TRISC_ISOLATE_SFPU
        addr_16B = dfb.get_read_ptr();
        format = static_cast<DataFormat>(unpack_src_format[dfb.get_id()]);
#endif
    }
};

// SrcS output in L1: an address (16-byte units) and a format, or a DataflowBuffer's back entry.
struct pack_to {
    uint32_t addr_16B = 0;
    DataFormat format = DataFormat::Invalid;

    constexpr pack_to(uint32_t l1_addr_16B, DataFormat l1_format) : addr_16B(l1_addr_16B), format(l1_format) {}

    template <typename DFB>
    explicit pack_to([[maybe_unused]] const DFB& dfb) {
#ifdef TRISC_ISOLATE_SFPU
        addr_16B = dfb.get_write_ptr();
        format = static_cast<DataFormat>(pack_dst_format[dfb.get_id()]);
#endif
    }
};

// clang-format off
/**
 * OP on SrcS, on the SFPU thread (TRISC3), over whole tile ranges: per tile, unpacker 2 moves it from L1 into
 * SrcS, the SFPU computes, packer 1 writes it back to L1.
 *
 * | Template Parameter | Description                                         | Type       | Valid Range                        | Required |
 * |--------------------|-----------------------------------------------------|------------|------------------------------------|----------|
 * | SRCS_FMT           | Format in the SrcS register (unpack dst = pack src) | DataFormat | Float16, Float16_b, Float32, Int32 | True     |
 */
// clang-format on
template <typename OP, DataFormat SRCS_FMT>
struct SfpuOp<OP, SfpuReg::SrcS, SRCS_FMT> {
    // Set up unpacker 2, packer 1 and the SFPU for this op: once before the first tile().
    ALWI static void init() {
        SFPU_SRCS((Op::configure()));
        SFPU_SRCS((sfpu_init_owner = &detail::sfpu_op_key<Op>));
    }

    // Run a one-input op on one tile: unpack it from in, compute, pack the result to out.
    ALWI static void tile(const unpack_from in, const pack_to out) { tiles(in, out, 1); }

    // The same on num_tiles consecutive tiles.
    ALWI static void tiles(const unpack_from in, const pack_to out, const uint32_t num_tiles) {
#ifdef TRISC_ISOLATE_SFPU
        static_assert(inputs == 1, "this op takes two inputs: tile(unpack_from, unpack_from, pack_to)");
#endif
        SFPU_SRCS((detail::sfpu_op_check_init<Op>()));
        SFPU_SRCS((Op::set_l1(in.addr_16B, in.format, out.addr_16B, out.format)));
        SFPU_SRCS((Op::run(num_tiles)));
    }

    // Run a two-input op on one tile. Both inputs have the same L1 format.
    ALWI static void tile(const unpack_from in0, const unpack_from in1, const pack_to out) { tiles(in0, in1, out, 1); }

    // The same on num_tiles consecutive tiles.
    ALWI static void tiles(const unpack_from in0, const unpack_from in1, const pack_to out, const uint32_t num_tiles) {
#ifdef TRISC_ISOLATE_SFPU
        static_assert(inputs == 2, "this op takes one input: tile(unpack_from, pack_to)");
        LLK_ASSERT(in0.format == in1.format, "SfpuOp: both inputs must have the same L1 format");
#endif
        SFPU_SRCS((detail::sfpu_op_check_init<Op>()));
        SFPU_SRCS((Op::set_l1(in0.addr_16B, in1.addr_16B, in0.format, out.addr_16B, out.format)));
        SFPU_SRCS((Op::run(num_tiles)));
    }

#ifdef TRISC_ISOLATE_SFPU

private:
    static constexpr sfpi::DataLayout layout = sfpu::sfpu_layout<SRCS_FMT>();
    static_assert(sfpu::op_takes_layout<OP, layout>, "this op does not take tiles of this format");
    static constexpr int inputs = sfpu::math_arity<sfpu::math_for_t<OP, layout>>;
    using Op = sfpu::srcs_op_t<OP, layout>;
#endif
};

}  // namespace ckernel
