// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API for Quasar SFPU op types: one call per op, the register file is a template tag.
//
//   sfpu_init<op::Exp, SfpuReg::Dest>();          // once, before the first tile
//   sfpu_tile<op::Exp, SfpuReg::Dest>(idst);      // exp on Dest tile idst, in place
//   sfpu_tile<op::Add, SfpuReg::Dest>(in0, in1, out);
//
//   sfpu_init<op::Exp, SfpuReg::SrcS>(in_addr, in_fmt, out_addr, out_fmt, srcs_fmt);   // L1 ring bases
//   sfpu_tiles<op::Exp, SfpuReg::SrcS>(srcs_fmt, n);     // n tiles: unpack to SrcS, exp, pack, per tile
//
// Choosing the register: data already in Dest because an FPU op (matmul, reduce, copy) just produced
// it takes the Dest form, in place, on the MATH thread. Data in L1 with no FPU op before the SFPU op
// takes the SrcS form, which does unpack, op and pack on the SFPU thread (TRISC3) and leaves the other
// three threads free. Packing a Dest result to L1 and running the SrcS form is a deliberate throughput
// trade (SFPU on TRISC3 overlapping the next FPU tile) that costs a pack and an unpack; the writer, or
// the op factory, picks it explicitly. Nothing infers it.
//
// The SrcS form takes L1 addresses and formats, not DataflowBuffers: TRISC3 has no DataflowBuffer
// interface yet (issue #63416). The tile loop therefore carries no DFB handshake; the producer and
// consumer of the two L1 regions synchronize with the SFPU thread by other means until then. An
// overload on DataflowBuffers is added on top of this one when the interface lands.
//
// The op tags (namespace op) name one Dest op type each. The kernel-wide constants APPROX and
// DST_ACCUM_MODE every compute kernel has select the op type's variant; the writer passes nothing.

#pragma once

#include "api/compute/common_globals.h"
#include "sfpu_reg.h"

#ifdef TRISC_MATH
#include "ckernel_sfpu_add.h"
#include "ckernel_sfpu_binary.h"
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

namespace op {

/// exp(x). APPROX picks the hardware LUT, otherwise fp32-accurate on a 32-bit Dest.
struct Exp {
#ifdef TRISC_MATH
    using Dest = sfpu::ExpDest<APPROX, DST_ACCUM_MODE>;
#endif
#ifdef TRISC_ISOLATE_SFPU
    template <sfpi::DataLayout L>
    using Srcs = sfpu::ExpSrcs<L, APPROX>;
#endif
};

/// x * x.
struct Square {
#ifdef TRISC_MATH
    using Dest = sfpu::SquareDest<>;
#endif
#ifdef TRISC_ISOLATE_SFPU
    template <sfpi::DataLayout L>
    using Srcs = sfpu::SquareSrcs<L>;
#endif
};

/// in0 + in1, float; bf16 RNE on a 16-bit Dest is a separate variant (add_binary_tile<NearestEven>).
struct Add {
#ifdef TRISC_MATH
    using Dest = sfpu::AddDest<DST_ACCUM_MODE>;
#endif
#ifdef TRISC_ISOLATE_SFPU
    template <sfpi::DataLayout L>
    using Srcs = sfpu::AddFloatSrcs<L>;  // float layouts only; Int32 is op::AddInt
#endif
};

/// in0 + in1, Int32, sign-magnitude: the same SIGN_MAGNITUDE_FORMAT=true that add_int_tile passes on Quasar.
struct AddInt {
#ifdef TRISC_MATH
    using Dest = sfpu::AddIntDest<true /*SIGN_MAGNITUDE_FORMAT*/>;
#endif
#ifdef TRISC_ISOLATE_SFPU
    template <sfpi::DataLayout L>
    using Srcs = sfpu::AddIntSrcs<L>;  // I32 only; a float format is rejected at the dispatch
#endif
};

}  // namespace op

/**
 * Initialize the SFPU for OP on REG. Call once before the first sfpu_tile of that op; call again after
 * any other SFPU init, since ops share the programmable constant registers.
 *
 * Return value: None
 *
 * | Template Parameter | Description                                 | Type    | Valid Range   | Required |
 * |--------------------|---------------------------------------------|---------|---------------|----------|
 * | OP                 | The op tag (namespace op)                   | struct  | op::*         | True     |
 * | REG                | The register file the op reads and writes   | SfpuReg | SfpuReg::Dest | True     |
 */
template <typename OP, SfpuReg REG>
ALWI void sfpu_init() {
    static_assert(REG == SfpuReg::Dest, "SrcS init takes the L1 addresses and formats: see the SrcS overload");
    MATH((OP::Dest::init()));
}

/**
 * Run the unary op OP on one Dest tile, in place. The DST register buffer must be in acquired state via
 * *acquire_dst*. This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument    | Description                                   | Type       | Valid Range | Required |
 * |-------------|-----------------------------------------------|------------|------------------------------------------------|----------|
 * | idst        | The index of the tile in DST register         | uint32_t   | Must be less than the acquired size of
 * DST REG | True     | | vector_mode | The faces to compute                          | VectorMode | R, C, RC | False |
 */
template <typename OP, SfpuReg REG>
ALWI void sfpu_tile(uint32_t idst, VectorMode vector_mode = VectorMode::RC) {
    static_assert(REG == SfpuReg::Dest, "SrcS ops run whole tile ranges: sfpu_tiles<OP, SrcS>(srcs_fmt, n)");
    MATH((OP::Dest::run(idst, vector_mode)));
}

/**
 * Run the binary op OP on two Dest tiles, writing a third (odst may equal idst0 or idst1). The DST register
 * buffer must be in acquired state via *acquire_dst*. This call is blocking and is only available on the
 * compute engine.
 *
 * Return value: None
 *
 * | Argument    | Description                                   | Type       | Valid Range | Required |
 * |-------------|-----------------------------------------------|------------|------------------------------------------------|----------|
 * | idst0       | The index of the first operand tile in DST    | uint32_t   | Must be less than the acquired size of
 * DST REG | True     | | idst1       | The index of the second operand tile in DST   | uint32_t   | Must be less than
 * the acquired size of DST REG | True     | | odst        | The index of the output tile in DST           | uint32_t |
 * Must be less than the acquired size of DST REG | True     | | vector_mode | The faces to compute | VectorMode | R, C,
 * RC                                       | False    |
 */
template <typename OP, SfpuReg REG>
ALWI void sfpu_tile(uint32_t idst0, uint32_t idst1, uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
    static_assert(REG == SfpuReg::Dest, "SrcS ops run whole tile ranges: sfpu_tiles<OP, SrcS>(srcs_fmt, n)");
    MATH((OP::Dest::run(idst0, idst1, odst, vector_mode)));
}

// SrcS: the op runs on the SFPU thread over whole tile ranges. The SrcS register format (unpack
// destination = pack source) is a runtime value, so each call dispatches once to the op's layout
// (ckernel_sfpu_srcs.h, dispatch_sfpu_srcs_op); the tile loop inside is not re-dispatched per tile.

/**
 * Initialize the unary op OP on SrcS: the unpacker-2 and packer-1 descriptors for one input and one output
 * ring in L1, and the op's own init. Call once before sfpu_tiles.
 *
 * Return value: None
 *
 * | Argument          | Description                                              | Type       | Valid Range | Required
 * |
 * |-------------------|----------------------------------------------------------|------------|--------------------------------------|----------|
 * | l1_in_addr_16B    | Input ring base in L1, 16-byte units                     | uint32_t   | | True     | |
 * l1_in_format      | Input format in L1                                       | DataFormat | | True     | |
 * l1_out_addr_16B   | Output ring base in L1, 16-byte units                    | uint32_t   | | True     | |
 * l1_out_format     | Output format in L1                                      | DataFormat | | True     | |
 * srcs_format       | Format in the SrcS register (unpack dst = pack src)      | DataFormat | Float16, Float16_b,
 * Float32, Int32   | True     |
 */
template <typename OP, SfpuReg REG>
ALWI void sfpu_init(
    uint32_t l1_in_addr_16B,
    DataFormat l1_in_format,
    uint32_t l1_out_addr_16B,
    DataFormat l1_out_format,
    DataFormat srcs_format) {
    static_assert(REG == SfpuReg::SrcS, "the Dest init takes no arguments: sfpu_init<OP, Dest>()");
    SFPU_SRCS((sfpu::dispatch_sfpu_srcs_op<OP::template Srcs>(srcs_format, srcs_format, [&](auto op) {
        decltype(op)::init(l1_in_addr_16B, l1_in_format, l1_out_addr_16B, l1_out_format);
    })));
}

/**
 * Initialize the binary op OP on SrcS: two input rings (same L1 format) and one output ring.
 */
template <typename OP, SfpuReg REG>
ALWI void sfpu_init(
    uint32_t l1_in0_addr_16B,
    uint32_t l1_in1_addr_16B,
    DataFormat l1_in_format,
    uint32_t l1_out_addr_16B,
    DataFormat l1_out_format,
    DataFormat srcs_format) {
    static_assert(REG == SfpuReg::SrcS, "the Dest init takes no arguments: sfpu_init<OP, Dest>()");
    SFPU_SRCS((sfpu::dispatch_sfpu_srcs_op<OP::template Srcs>(srcs_format, srcs_format, [&](auto op) {
        decltype(op)::init(l1_in0_addr_16B, l1_in1_addr_16B, l1_in_format, l1_out_addr_16B, l1_out_format);
    })));
}

/**
 * Run OP on num_tiles consecutive tiles from the rings given to sfpu_init: per tile, unpack L1 -> SrcS,
 * the op over the tile's slices, pack SrcS -> L1. Unary and binary ops alike. Only available on the
 * SFPU thread; expands to nothing elsewhere.
 *
 * Return value: None
 *
 * | Argument    | Description                                         | Type       | Valid Range | Required |
 * |-------------|-----------------------------------------------------|------------|------------------------------------|----------|
 * | srcs_format | Format in the SrcS register, as given to sfpu_init  | DataFormat | Float16, Float16_b, Float32, Int32
 * | True     | | num_tiles   | Tiles to process                                    | uint32_t   | | True     |
 */
template <typename OP, SfpuReg REG>
ALWI void sfpu_tiles(DataFormat srcs_format, uint32_t num_tiles) {
    static_assert(REG == SfpuReg::SrcS, "Dest ops run one tile at a time: sfpu_tile<OP, Dest>(idst)");
    SFPU_SRCS((sfpu::dispatch_sfpu_srcs_op<OP::template Srcs>(
        srcs_format, srcs_format, [&](auto op) { decltype(op)::run(num_tiles); })));
}

}  // namespace ckernel
