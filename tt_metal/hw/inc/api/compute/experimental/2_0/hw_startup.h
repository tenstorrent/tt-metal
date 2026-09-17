// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"                        // ALWI, UNPACK/MATH/PACK, DST_ACCUM_MODE, PackMode
#include "api/compute/src_order.h"                     // SrcOrder
#include "api/compute/experimental/2_0/llk_operand.h"  // LLKOperand
#include "experimental/2_0/llk_hw_configure.h"  // id-free llk_{unpack,math,pack}_hw_configure (+ pack_dest_init / math_pack_sync_init per TRISC)

#ifdef TRISC_PACK
#include "experimental/2_0/llk_pack_tile.h"  // id-free llk_pack_init<DESC>
#endif

// =====================================================================================================
// Id-free (2.0) compute_kernel_hw_startup: an OVERLOAD of the shipping compute_kernel_hw_startup
// (api/compute/compute_kernel_hw_startup.h, CB-id) -- same name, distinguished by taking LLKOperands
// instead of CB ids. Every data format and tile geometry comes from the operand descriptors (NTTPs);
// there is no runtime L1-format inference and no CB-id dependency on the compute-op surface. The
// operands' runtime l1_address is unused here (startup only programs formats/geometry).
//
// NOTE: kept OUT of the shipping header on purpose. That header is pulled into every legacy kernel via
// common.h -> kernel_args.h, but this overload references ckernel::experimental, which collides with
// metal's top-level ::experimental (kernel_args.h) under `using namespace ckernel`. 2.0 kernels (which
// never include kernel_args.h) include THIS header instead; legacy kernels are untouched.
//
// src_order selects how (in0, in1) map onto SrcA/SrcB (Reverse for matmul), exactly as the CB-id
// overload. The two source-register formats are reconciled to a common exponent-width family inside
// the LLK hw_configure. Blackhole only (2.0 scope).
//
// CAVEAT: hw_startup writes the shared per-engine UNPACK/MATH/PACK config registers and must run
// exactly ONCE, at the top of the kernel, before any op-specific init or tile traffic on that engine --
// reprogramming these registers while an op is in flight is a data race on engine state. Mirrors the
// CB-id overload's contract.
// =====================================================================================================

namespace ckernel {

// clang-format off
/**
 * Id-free (2.0) two-input hardware startup. Programs the UNPACK/MATH/PACK register formats + tile geometry
 * from the operand descriptors (NTTPs) -- no CB ids, no runtime L1-format inference. Call exactly once at the
 * top of the kernel, before any op-specific init. src_order selects how (in0, in1) map onto SrcA/SrcB
 * (SrcOrder::Reverse for matmul: in0 -> SrcB, in1 -> SrcA). The operands' runtime l1_address is unused here.
 * The two source-register formats are reconciled to a common exponent-width family; a mixed-width pairing
 * that is not Float32-rebiasable is a hard compile error.
 *
 * | Param Type | Name             | Description                                      | Type                   | Valid Range | Required |
 * |------------|------------------|--------------------------------------------------|------------------------|-------------|----------|
 * | Template   | src_order           | (in0,in1) -> SrcA/SrcB mapping (Reverse=matmul)  | SrcOrder               | N/A         | False    |
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode                         | bool                   |             | False    |
 * | Template   | FA/SA, FB/SB, FO/SO | in0 / in1 / out L1 format + geometry (deduced) | DataFormat/TensorShape | N/A         | True     |
 * | Function   | in0 / in1        | Input operands (natural order)                   | LLKOperand             | N/A         | True     |
 * | Function   | out              | Output operand                                   | LLKOperand             | N/A         | True     |
 */
// clang-format on
template <
    SrcOrder src_order = SrcOrder::Regular,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat FA,
    TensorShape SA,
    DataFormat FB,
    TensorShape SB,
    DataFormat FO,
    TensorShape SO>
ALWI void compute_kernel_hw_startup(
    experimental::LLKOperand<FA, SA> /*in0*/,
    experimental::LLKOperand<FB, SB> /*in1*/,
    experimental::LLKOperand<FO, SO> /*out*/) {
    static_assert(experimental::is_legal_tile_shape(SA), "compute_kernel_hw_startup: illegal in0 tile shape.");
    static_assert(experimental::is_legal_tile_shape(SB), "compute_kernel_hw_startup: illegal in1 tile shape.");
    static_assert(experimental::is_legal_tile_shape(SO), "compute_kernel_hw_startup: illegal out tile shape.");
    // Map the operands onto the physical source registers. For SrcOrder::Reverse (matmul), in0 -> SrcB and
    // in1 -> SrcA (resolved at compile time). Each descriptor is referenced only inside its own thread's
    // macro (UNPACK/MATH/PACK), so it is unused on the other threads -- fine under -Wno-unused-variable.
    constexpr bool reverse = (src_order == SrcOrder::Reverse);
    constexpr experimental::LLKMemDescriptor SRCA =
        reverse ? experimental::LLKOperand<FB, SB>::descriptor : experimental::LLKOperand<FA, SA>::descriptor;
    constexpr experimental::LLKMemDescriptor SRCB =
        reverse ? experimental::LLKOperand<FA, SA>::descriptor : experimental::LLKOperand<FB, SB>::descriptor;
    constexpr experimental::LLKMemDescriptor OUT = experimental::LLKOperand<FO, SO>::descriptor;

    UNPACK((llk_unpack_hw_configure<is_fp32_dest_acc_en, SRCA, SRCB>()));

    MATH((llk_math_pack_sync_init<is_fp32_dest_acc_en>()));
    MATH((llk_math_hw_configure<is_fp32_dest_acc_en, SRCA, SRCB>()));

    PACK((llk_pack_hw_configure<is_fp32_dest_acc_en, OUT>()));
    PACK((llk_pack_init<OUT, is_fp32_dest_acc_en, PackMode::Default>()));
    PACK((_llk_pack_dest_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>()));
}

// clang-format off
/**
 * Id-free (2.0) single-input hardware startup: convenience overload that configures both source registers from
 * the same operand. Equivalent to the two-input overload with in0 == in1. Used by single-operand ops
 * (copy/tilize/pack_untilize).
 *
 * | Param Type | Name       | Description                              | Type                   | Valid Range | Required |
 * |------------|------------|------------------------------------------|------------------------|-------------|----------|
 * | Template   | src_order           | (in,in) -> SrcA/SrcB mapping             | SrcOrder               | N/A         | False    |
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode                | bool                   |             | False    |
 * | Template   | F/S, FO/SO          | in / out L1 format + geometry (deduced)  | DataFormat/TensorShape | N/A         | True     |
 * | Function   | in         | The single input operand                 | LLKOperand             | N/A         | True     |
 * | Function   | out        | Output operand                           | LLKOperand             | N/A         | True     |
 */
// clang-format on
template <
    SrcOrder src_order = SrcOrder::Regular,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat F,
    TensorShape S,
    DataFormat FO,
    TensorShape SO>
ALWI void compute_kernel_hw_startup(experimental::LLKOperand<F, S> in, experimental::LLKOperand<FO, SO> out) {
    compute_kernel_hw_startup<src_order, is_fp32_dest_acc_en>(in, in, out);
}

}  // namespace ckernel
