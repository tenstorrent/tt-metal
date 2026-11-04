// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Quasar SFPU op types: one call per op; the register file and, on Dest, the tile format are template tags.
//
//   sfpu_init<op::Exp, SfpuReg::Dest, fmt>();                               // fmt: the tiles' DataFormat
//   sfpu_tile<op::Exp, SfpuReg::Dest, fmt>(idst);                           // in place, MATH thread
//   sfpu_init<op::Exp, SfpuReg::SrcS>(in_addr, in_fmt, out_addr, out_fmt, srcs_fmt);
//   sfpu_tiles<op::Exp, SfpuReg::SrcS>(srcs_fmt, n);                        // L1 -> SrcS -> op -> L1, TRISC3

#pragma once

#ifndef ARCH_QUASAR
#error "api/compute/experimental/sfpu.h is Quasar-only"
#endif

#include "api/compute/common_globals.h"
#include "api/compute/compute_kernel_api.h"
#include "sfpu_reg.h"

#ifdef TRISC_MATH
#include "ckernel_sfpu_add.h"
#include "ckernel_sfpu_binary.h"
#include "ckernel_sfpu_dest.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_square.h"
#endif
#ifdef TRISC_ISOLATE_SFPU
#include "ckernel_sfpu_add.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_square.h"
#include "llk_sfpu_srcs_api.h"
#endif

namespace ckernel {

using sfpu::SfpuReg;

// Op tags. Dest<L> and Srcs<L> name the op type for an SFPLOAD/SFPSTORE layout L; is_int says which
// formats the op takes. APPROX and DST_ACCUM_MODE are the kernel's constants.
namespace op {

/** @brief exp(x). APPROX picks the hardware LUT, otherwise fp32-accurate on a 32-bit Dest. */
struct Exp {
    static constexpr bool is_int = false;
#ifdef TRISC_MATH
    template <sfpi::DataLayout L>
    using Dest = sfpu::ExpDest<APPROX, DST_ACCUM_MODE, L>;
#endif
#ifdef TRISC_ISOLATE_SFPU
    template <sfpi::DataLayout L>
    using Srcs = sfpu::ExpSrcs<L, APPROX>;
#endif
};

/** @brief x * x. */
struct Square {
    static constexpr bool is_int = false;
#ifdef TRISC_MATH
    template <sfpi::DataLayout L>
    using Dest = sfpu::SquareDest<L>;
#endif
#ifdef TRISC_ISOLATE_SFPU
    template <sfpi::DataLayout L>
    using Srcs = sfpu::SquareSrcs<L>;
#endif
};

/** @brief in0 + in1, float. bf16 RNE on a 16-bit Dest stays with add_binary_tile<NearestEven>. */
struct Add {
    static constexpr bool is_int = false;
#ifdef TRISC_MATH
    template <sfpi::DataLayout L>
    using Dest = sfpu::AddDest<DST_ACCUM_MODE, DstRoundingMode::Default, L>;
#endif
#ifdef TRISC_ISOLATE_SFPU
    template <sfpi::DataLayout L>
    using Srcs = sfpu::AddFloatSrcs<L>;
#endif
};

/** @brief in0 + in1, Int32, sign-magnitude (what add_int_tile passes on Quasar). */
struct AddInt {
    static constexpr bool is_int = true;
#ifdef TRISC_MATH
    template <sfpi::DataLayout L>
    using Dest = sfpu::AddIntDest<true /*SIGN_MAGNITUDE_FORMAT*/>;  // always explicit I32
#endif
#ifdef TRISC_ISOLATE_SFPU
    template <sfpi::DataLayout L>
    using Srcs = sfpu::AddIntSrcs<L>;
#endif
};

}  // namespace op

namespace detail {

template <typename OP, DataFormat FMT>
constexpr void check_dest_format() {
    static_assert((FMT == DataFormat::Int32) == OP::is_int, "this op does not take tiles of this format");
}

}  // namespace detail

// clang-format off
/**
 * Initialize OP on Dest. Call once before the first sfpu_tile of that op, and again after any other SFPU
 * init, since ops share the programmable constant registers.
 *
 * Return value: None
 *
 * | Template Parameter | Description                                        | Type       | Valid Range                          | Required |
 * |--------------------|----------------------------------------------------|------------|--------------------------------------|----------|
 * | OP                 | The op tag (namespace op)                          | struct     | op::*                                | True     |
 * | REG                | The register file                                  | SfpuReg    | SfpuReg::Dest                        | True     |
 * | FMT                | Format of the tiles in Dest                        | DataFormat | Float16, Float16_b, Float32, Int32   | True     |
 */
// clang-format on
template <typename OP, SfpuReg REG, DataFormat FMT>
ALWI void sfpu_init() {
    static_assert(REG == SfpuReg::Dest, "SrcS init takes the L1 addresses and formats: see the SrcS overload");
    detail::check_dest_format<OP, FMT>();
    MATH((OP::template Dest<sfpu::dest_layout<FMT, DST_ACCUM_MODE>()>::init()));
}

// clang-format off
/**
 * Run the unary op OP on one Dest tile, in place. The DST register buffer must be in acquired state via
 * *acquire_dst*. This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument    | Description                     | Type       | Valid Range                                    | Required |
 * |-------------|---------------------------------|------------|------------------------------------------------|----------|
 * | idst        | The index of the tile in DST    | uint32_t   | Must be less than the acquired size of DST REG | True     |
 * | vector_mode | The faces to compute            | VectorMode | R, C, RC, None, RC_custom                      | False    |
 */
// clang-format on
template <typename OP, SfpuReg REG, DataFormat FMT>
ALWI void sfpu_tile(uint32_t idst, VectorMode vector_mode = VectorMode::RC) {
    static_assert(REG == SfpuReg::Dest, "SrcS ops run whole tile ranges: sfpu_tiles<OP, SrcS>(srcs_fmt, n)");
    detail::check_dest_format<OP, FMT>();
    MATH((OP::template Dest<sfpu::dest_layout<FMT, DST_ACCUM_MODE>()>::run(idst, vector_mode)));
}

// clang-format off
/**
 * Run the binary op OP on two Dest tiles, writing a third (odst may equal idst0 or idst1). The DST register
 * buffer must be in acquired state via *acquire_dst*. This call is blocking and is only available on the
 * compute engine.
 *
 * Return value: None
 *
 * | Argument    | Description                            | Type       | Valid Range                                    | Required |
 * |-------------|----------------------------------------|------------|------------------------------------------------|----------|
 * | idst0       | The index of the first operand in DST  | uint32_t   | Must be less than the acquired size of DST REG | True     |
 * | idst1       | The index of the second operand in DST | uint32_t   | Must be less than the acquired size of DST REG | True     |
 * | odst        | The index of the output tile in DST    | uint32_t   | Must be less than the acquired size of DST REG | True     |
 * | vector_mode | The faces to compute                   | VectorMode | R, C, RC, None, RC_custom                      | False    |
 */
// clang-format on
template <typename OP, SfpuReg REG, DataFormat FMT>
ALWI void sfpu_tile(uint32_t idst0, uint32_t idst1, uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
    static_assert(REG == SfpuReg::Dest, "SrcS ops run whole tile ranges: sfpu_tiles<OP, SrcS>(srcs_fmt, n)");
    detail::check_dest_format<OP, FMT>();
    MATH((OP::template Dest<sfpu::dest_layout<FMT, DST_ACCUM_MODE>()>::run(idst0, idst1, odst, vector_mode)));
}

// SrcS: the op runs on the SFPU thread over whole tile ranges. The SrcS register format is a runtime
// value; each call dispatches once to the op's layout (dispatch_sfpu_srcs_op), not per tile.

// clang-format off
/**
 * Initialize the unary op OP on SrcS: the unpacker-2 and packer-1 descriptors for one input and one output
 * ring in L1, and the op's own init. Call once before sfpu_tiles.
 *
 * Return value: None
 *
 * | Argument        | Description                                         | Type       | Valid Range                        | Required |
 * |-----------------|-----------------------------------------------------|------------|------------------------------------|----------|
 * | l1_in_addr_16B  | Input ring base in L1, 16-byte units                | uint32_t   |                                    | True     |
 * | l1_in_format    | Input format in L1                                  | DataFormat |                                    | True     |
 * | l1_out_addr_16B | Output ring base in L1, 16-byte units               | uint32_t   |                                    | True     |
 * | l1_out_format   | Output format in L1                                 | DataFormat |                                    | True     |
 * | srcs_format     | Format in the SrcS register (unpack dst = pack src) | DataFormat | Float16, Float16_b, Float32, Int32 | True     |
 */
// clang-format on
template <typename OP, SfpuReg REG>
ALWI void sfpu_init(
    uint32_t l1_in_addr_16B,
    DataFormat l1_in_format,
    uint32_t l1_out_addr_16B,
    DataFormat l1_out_format,
    DataFormat srcs_format) {
    static_assert(REG == SfpuReg::SrcS, "the Dest init takes no arguments: sfpu_init<OP, Dest, FMT>()");
    SFPU_SRCS((sfpu::dispatch_sfpu_srcs_op<OP::template Srcs>(srcs_format, srcs_format, [&](auto op) {
        decltype(op)::init(l1_in_addr_16B, l1_in_format, l1_out_addr_16B, l1_out_format);
    })));
}

// clang-format off
/**
 * Initialize the binary op OP on SrcS: two input rings (same L1 format) and one output ring.
 *
 * Return value: None
 *
 * | Argument         | Description                                         | Type       | Valid Range                        | Required |
 * |------------------|-----------------------------------------------------|------------|------------------------------------|----------|
 * | l1_in0_addr_16B  | First input ring base in L1, 16-byte units          | uint32_t   |                                    | True     |
 * | l1_in1_addr_16B  | Second input ring base in L1, 16-byte units         | uint32_t   |                                    | True     |
 * | l1_in_format     | Input format in L1, both inputs                     | DataFormat |                                    | True     |
 * | l1_out_addr_16B  | Output ring base in L1, 16-byte units               | uint32_t   |                                    | True     |
 * | l1_out_format    | Output format in L1                                 | DataFormat |                                    | True     |
 * | srcs_format      | Format in the SrcS register (unpack dst = pack src) | DataFormat | Float16, Float16_b, Float32, Int32 | True     |
 */
// clang-format on
template <typename OP, SfpuReg REG>
ALWI void sfpu_init(
    uint32_t l1_in0_addr_16B,
    uint32_t l1_in1_addr_16B,
    DataFormat l1_in_format,
    uint32_t l1_out_addr_16B,
    DataFormat l1_out_format,
    DataFormat srcs_format) {
    static_assert(REG == SfpuReg::SrcS, "the Dest init takes no arguments: sfpu_init<OP, Dest, FMT>()");
    SFPU_SRCS((sfpu::dispatch_sfpu_srcs_op<OP::template Srcs>(srcs_format, srcs_format, [&](auto op) {
        decltype(op)::init(l1_in0_addr_16B, l1_in1_addr_16B, l1_in_format, l1_out_addr_16B, l1_out_format);
    })));
}

// clang-format off
/**
 * Run OP on num_tiles consecutive tiles from the rings given to sfpu_init: per tile, unpack L1 -> SrcS,
 * the op over the tile's slices, pack SrcS -> L1. Only available on the SFPU thread; expands to nothing elsewhere.
 *
 * Return value: None
 *
 * | Argument    | Description                                        | Type       | Valid Range                        | Required |
 * |-------------|----------------------------------------------------|------------|------------------------------------|----------|
 * | srcs_format | Format in the SrcS register, as given to sfpu_init | DataFormat | Float16, Float16_b, Float32, Int32 | True     |
 * | num_tiles   | Tiles to process                                   | uint32_t   |                                    | True     |
 */
// clang-format on
template <typename OP, SfpuReg REG>
ALWI void sfpu_tiles(DataFormat srcs_format, uint32_t num_tiles) {
    static_assert(REG == SfpuReg::SrcS, "Dest ops run one tile at a time: sfpu_tile<OP, Dest, FMT>(idst)");
    SFPU_SRCS((sfpu::dispatch_sfpu_srcs_op<OP::template Srcs>(
        srcs_format, srcs_format, [&](auto op) { decltype(op)::run(num_tiles); })));
}

}  // namespace ckernel
