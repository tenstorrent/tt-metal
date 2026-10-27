// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/experimental/custom_mm_reuse_dest_srcb.h"
#include "api/compute/experimental/2_0/llk_operand.h"
#include "sanitizer/api.h"

// LLKOperand overloads live in this opt-in header, like 2_0/hw_startup.h:
// pulling ckernel::experimental into legacy kernels can collide with the
// top-level experimental namespace used by kernel_args.h.
//
// These retain the CB API's synchronization, layout and LoFi contracts. L1
// operand addresses are in 16-byte words, using the same base convention as
// cb_read_address(). The caller owns input readiness and output CB capacity.
// Short init configures only the operation: the caller must first configure
// hardware formats/geometry and destination synchronization. It must quiesce
// prior operations before reconfiguration. Output packing remains CB-based.
// Activations already reside in DEST. isrc/idst and src_tile_stride are DEST
// row offsets; in1_tile_index/in1_k_stride count weight tiles. The destination
// must be zero before execution. Use the existing pack init/uninit pair for
// the result's 16-row tile layout.
// As in the CB API, kt_dim must be even and in [2, 256]: the unpacker's MOP
// consumes two K tiles per iteration. All kt_dim source tiles must be in DEST.

namespace ckernel {

template <bool load_replay = true, DataFormat F, TensorShape S>
ALWI void custom_mm_reuse_dest_srcb_block_init_short(
    experimental::LLKOperand<F, S> /*in1*/, const std::uint32_t nt_dim) {
    SAN_HOOK(unsupported());
    static_assert(
        S.face_r_dim == 16 && S.face_c_dim == 16 && S.num_faces_r_dim == 2 && S.num_faces_c_dim == 2,
        "custom_mm_reuse_dest_srcb: weight tile shape must be [32, 32]");
    UNPACK((_llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_init_(nt_dim, S.face_r_dim, S.total_num_faces())));
    MATH((llk_math_custom_mm_reuse_dest_srcb_init<load_replay>()));
}

template <uint32_t in0_tile_r_dim, DataFormat F, TensorShape S>
ALWI void custom_mm_reuse_dest_srcb_block(
    experimental::LLKOperand<F, S> in1,
    const std::uint32_t in1_tile_index,
    const std::uint32_t isrc,
    const std::uint32_t idst,
    const std::uint32_t kt_dim,
    const std::uint32_t nt_dim,
    const std::uint32_t in1_k_stride,
    const std::uint32_t src_tile_stride = CUSTOM_MM_DEST_TILE_ROWS) {
    SAN_HOOK(unsupported());
    constexpr std::uint32_t in1_tile_size = experimental::tile_stride_words(F, S);
    UNPACK((llk_unpack_A_sdpa_set_srcb_dummy_valid()));
    UNPACK((_llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_(
        in1.l1_address, in1_tile_index, in1_tile_size, kt_dim, nt_dim, in1_k_stride)));
    MATH((llk_math_custom_mm_reuse_dest_srcb<in0_tile_r_dim>(isrc, idst, kt_dim, nt_dim, src_tile_stride)));
}

}  // namespace ckernel
