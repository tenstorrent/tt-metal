// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Quasar SFPU op types: one call per op. The register file is a tag; on Dest the DFB the tiles came from gives
// their format at compile time, on SrcS the register format is a tag.
//
//   sfpu_init<op::Exp<>, SfpuReg::Dest, dfb_id>();                                  // once, before the first tile
//   sfpu_tile<op::Exp<>, SfpuReg::Dest, dfb_id>(idst);                              // in place, MATH thread
//   sfpu_init<op::Exp<>, SfpuReg::SrcS, srcs_fmt>(in_addr, in_fmt, out_addr, out_fmt);
//   sfpu_tiles<op::Exp<>, SfpuReg::SrcS, srcs_fmt>(n);                              // L1 -> SrcS -> op -> L1, TRISC3

#pragma once

#ifndef ARCH_QUASAR
#error "api/compute/experimental/sfpu.h is Quasar-only"
#endif

#include "api/compute/common_globals.h"
#include "api/compute/compute_kernel_api.h"
#include "ckernel_defs.h"
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

// Op tags. Dest<L> and Srcs<L> name the op type for an SFPLOAD/SFPSTORE layout L; is_int says which formats
// the op takes. Variants are template parameters, defaulted from the kernel's constants.
namespace op {

/** @brief exp(x). APPROX_MODE picks the hardware LUT, otherwise fp32-accurate on a 32-bit Dest. */
template <bool APPROX_MODE = APPROX>
struct Exp {
    static constexpr bool is_int = false;
#ifdef TRISC_MATH
    template <sfpi::DataLayout L>
    using Dest = sfpu::ExpDest<APPROX_MODE, DST_ACCUM_MODE, L>;
#endif
#ifdef TRISC_ISOLATE_SFPU
    template <sfpi::DataLayout L>
    using Srcs = sfpu::ExpSrcs<L, APPROX_MODE>;
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

/** @brief in0 + in1, float. ROUNDING = NearestEven rounds the store to bf16 on a 16-bit Dest; SrcS ignores it. */
template <DstRoundingMode ROUNDING = DstRoundingMode::Default>
struct Add {
    static constexpr bool is_int = false;
#ifdef TRISC_MATH
    template <sfpi::DataLayout L>
    using Dest = sfpu::AddDest<DST_ACCUM_MODE, ROUNDING, L>;
#endif
#ifdef TRISC_ISOLATE_SFPU
    template <sfpi::DataLayout L>
    using Srcs = sfpu::AddSrcs<L>;
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
    using Srcs = sfpu::AddSrcs<L>;
#endif
};

}  // namespace op

namespace detail {

#ifdef TRISC_MATH
// Layout of the Dest tiles a DFB feeds: its unpack-destination format, F32 on a 32-bit Dest. Compile time.
template <typename OP, std::uint32_t DFB_ID>
constexpr sfpi::DataLayout dest_layout_of() {
    constexpr DataFormat fmt = static_cast<DataFormat>(unpack_dst_format[DFB_ID]);
    static_assert((fmt == DataFormat::Int32) == OP::is_int, "this op does not take tiles of this format");
    return sfpu::dest_layout<fmt, DST_ACCUM_MODE>();
}
#endif

#ifdef TRISC_ISOLATE_SFPU
template <typename OP, DataFormat SRCS_FMT>
constexpr sfpi::DataLayout srcs_layout_of() {
    static_assert((SRCS_FMT == DataFormat::Int32) == OP::is_int, "this op does not take tiles of this format");
    return sfpu::srcs_layout<SRCS_FMT>();
}
#endif

}  // namespace detail

// clang-format off
/**
 * Initialize OP on Dest. Call once before the first sfpu_tile of that op, and again after any other SFPU
 * init, since ops share the programmable constant registers.
 *
 * Return value: None
 *
 * | Template Parameter | Description                                        | Type     | Valid Range   | Required |
 * |--------------------|----------------------------------------------------|----------|---------------|----------|
 * | OP                 | The op tag (namespace op)                          | struct   | op::*         | True     |
 * | REG                | The register file                                  | SfpuReg  | SfpuReg::Dest | True     |
 * | DFB_ID             | The DFB whose tiles are in Dest; gives their format | uint32_t | a DFB id      | True     |
 */
// clang-format on
template <typename OP, SfpuReg REG, std::uint32_t DFB_ID>
ALWI void sfpu_init() {
    static_assert(REG == SfpuReg::Dest, "SrcS init takes the L1 addresses and formats: see the SrcS overload");
    MATH((OP::template Dest<detail::dest_layout_of<OP, DFB_ID>()>::init()));
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
template <typename OP, SfpuReg REG, std::uint32_t DFB_ID>
ALWI void sfpu_tile(uint32_t idst, VectorMode vector_mode = VectorMode::RC) {
    static_assert(REG == SfpuReg::Dest, "SrcS ops run whole tile ranges: sfpu_tiles<OP, SrcS, FMT>(n)");
    MATH((OP::template Dest<detail::dest_layout_of<OP, DFB_ID>()>::run(idst, vector_mode)));
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
template <typename OP, SfpuReg REG, std::uint32_t DFB_ID>
ALWI void sfpu_tile(uint32_t idst0, uint32_t idst1, uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
    static_assert(REG == SfpuReg::Dest, "SrcS ops run whole tile ranges: sfpu_tiles<OP, SrcS, FMT>(n)");
    MATH((OP::template Dest<detail::dest_layout_of<OP, DFB_ID>()>::run(idst0, idst1, odst, vector_mode)));
}

// SrcS: the op runs on the SFPU thread over whole tile ranges, in the register format SRCS_FMT.

// clang-format off
/**
 * Initialize the unary op OP on SrcS: the unpacker-2 and packer-1 descriptors for one input and one output
 * ring in L1, and the op's own init. Call once before sfpu_tiles.
 *
 * Return value: None
 *
 * | Argument        | Description                           | Type       | Valid Range | Required |
 * |-----------------|---------------------------------------|------------|-------------|----------|
 * | l1_in_addr_16B  | Input ring base in L1, 16-byte units  | uint32_t   |             | True     |
 * | l1_in_format    | Input format in L1                    | DataFormat |             | True     |
 * | l1_out_addr_16B | Output ring base in L1, 16-byte units | uint32_t   |             | True     |
 * | l1_out_format   | Output format in L1                   | DataFormat |             | True     |
 *
 * | Template Parameter | Description                                         | Type       | Valid Range                        | Required |
 * |--------------------|-----------------------------------------------------|------------|------------------------------------|----------|
 * | SRCS_FMT           | Format in the SrcS register (unpack dst = pack src) | DataFormat | Float16, Float16_b, Float32, Int32 | True     |
 */
// clang-format on
template <typename OP, SfpuReg REG, DataFormat SRCS_FMT>
ALWI void sfpu_init(
    uint32_t l1_in_addr_16B, DataFormat l1_in_format, uint32_t l1_out_addr_16B, DataFormat l1_out_format) {
    static_assert(REG == SfpuReg::SrcS, "the Dest init takes no arguments: sfpu_init<OP, Dest, DFB_ID>()");
    SFPU_SRCS((OP::template Srcs<detail::srcs_layout_of<OP, SRCS_FMT>()>::init(
        l1_in_addr_16B, l1_in_format, l1_out_addr_16B, l1_out_format)));
}

// clang-format off
/**
 * Initialize the binary op OP on SrcS: two input rings (same L1 format) and one output ring.
 *
 * Return value: None
 *
 * | Argument         | Description                                 | Type       | Valid Range | Required |
 * |------------------|---------------------------------------------|------------|-------------|----------|
 * | l1_in0_addr_16B  | First input ring base in L1, 16-byte units  | uint32_t   |             | True     |
 * | l1_in1_addr_16B  | Second input ring base in L1, 16-byte units | uint32_t   |             | True     |
 * | l1_in_format     | Input format in L1, both inputs             | DataFormat |             | True     |
 * | l1_out_addr_16B  | Output ring base in L1, 16-byte units       | uint32_t   |             | True     |
 * | l1_out_format    | Output format in L1                         | DataFormat |             | True     |
 */
// clang-format on
template <typename OP, SfpuReg REG, DataFormat SRCS_FMT>
ALWI void sfpu_init(
    uint32_t l1_in0_addr_16B,
    uint32_t l1_in1_addr_16B,
    DataFormat l1_in_format,
    uint32_t l1_out_addr_16B,
    DataFormat l1_out_format) {
    static_assert(REG == SfpuReg::SrcS, "the Dest init takes no arguments: sfpu_init<OP, Dest, DFB_ID>()");
    SFPU_SRCS((OP::template Srcs<detail::srcs_layout_of<OP, SRCS_FMT>()>::init(
        l1_in0_addr_16B, l1_in1_addr_16B, l1_in_format, l1_out_addr_16B, l1_out_format)));
}

// clang-format off
/**
 * Run OP on num_tiles consecutive tiles from the rings given to sfpu_init: per tile, unpack L1 -> SrcS, the op
 * over the tile's slices, pack SrcS -> L1. Only available on the SFPU thread; expands to nothing elsewhere.
 *
 * Return value: None
 *
 * | Argument  | Description      | Type     | Valid Range | Required |
 * |-----------|------------------|----------|-------------|----------|
 * | num_tiles | Tiles to process | uint32_t |             | True     |
 */
// clang-format on
template <typename OP, SfpuReg REG, DataFormat SRCS_FMT>
ALWI void sfpu_tiles(uint32_t num_tiles) {
    static_assert(REG == SfpuReg::SrcS, "Dest ops run one tile at a time: sfpu_tile<OP, Dest, DFB_ID>(idst)");
    SFPU_SRCS((OP::template Srcs<detail::srcs_layout_of<OP, SRCS_FMT>()>::run(num_tiles)));
}

}  // namespace ckernel
