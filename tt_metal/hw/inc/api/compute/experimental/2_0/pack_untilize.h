// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#include "api/compute/experimental/2_0/llk_mem_descriptor.h"

#ifdef TRISC_MATH
#include "experimental/2_0/llk_math_unary_datacopy.h"
#endif

#ifdef TRISC_UNPACK
#include "experimental/2_0/llk_unpack_A.h"
#endif

#ifdef TRISC_PACK
#include "experimental/2_0/llk_pack_untilize.h"
#include "experimental/2_0/llk_pack_tile.h"
#endif

namespace ckernel {
namespace experimental {

#ifdef ARCH_BLACKHOLE

// clang-format off
/**
 * Experimental id-free pack-untilize init. Takes an input and an output LLKOperand (data format + tile
 * geometry as NTTPs) instead of CB ids. Initializes all three threads (UNPACK/MATH/PACK) to move tilized
 * tiles CB -> DEST (datacopy) and pack them out row-major. Register formats are derived INSIDE the LLK.
 *
 * Mirrors the legacy pack_untilize_init (3-thread) + pack_untilize_dest_init PACK sequence, with one
 * substitution: the legacy PACK path calls llk_pack_reconfig_data_format(ocb) to program the packer format
 * registers from CB metadata; the id-free path calls the LLKOperand overload of that same op (formats from
 * OUT). llk_pack_untilize_init itself only programs addrmod/MOP/z-stride, so this format program is
 * required. BH DEST remap is (re)configured here (llk_math_reconfig_remap), matching the default legacy init.
 *
 * | Template | block_ct_dim/full_ct_dim | Block width / full input width in tiles | uint32_t | | False |
 * | Function | in                       | Input operand (tilized; format+geometry) | LLKOperand | | True |
 * | Function | out                      | Output operand (row-major destination)   | LLKOperand | | True |
 */
// clang-format on
template <
    std::uint32_t block_ct_dim = 8,
    std::uint32_t full_ct_dim = block_ct_dim,
    DataFormat InFormat,
    TensorShape InShape,
    DataFormat OutFormat,
    TensorShape OutShape>
ALWI void pack_untilize_init(LLKOperand<InFormat, InShape> /*in*/, LLKOperand<OutFormat, OutShape> /*out*/) {
    // UNPACK + MATH: configure the CB -> DEST datacopy (input format drives the SrcA/Dest register format).
    UNPACK((llk_unpack_A_init<
            LLKOperand<InFormat, InShape>::descriptor,
            DST_ACCUM_MODE,
            BroadcastType::NONE,
            false /*acc_to_dest*/,
            EltwiseBinaryReuseDestType::NONE,
            UnpackToDestEn>(0 /*transpose_of_faces*/, 0 /*within_face_16x16_transpose*/)));
    MATH((llk_math_eltwise_unary_datacopy_init<
          LLKOperand<InFormat, InShape>::descriptor,
          DataCopyType::A2D,
          DST_ACCUM_MODE,
          BroadcastType::NONE,
          false /*is_int_en*/,
          PackMode::Default>()));

    // PACK: (re)configure BH DEST remap, program the packer output formats, then the untilize MOP/strides
    // and the untilize dest-offset registers.
    MATH((llk_math_reconfig_remap(true /*remap_enable*/)));
    PACK((llk_pack_reconfig_data_format<LLKOperand<OutFormat, OutShape>::descriptor, DST_ACCUM_MODE>()));
    PACK((llk_pack_untilize_init<
          LLKOperand<OutFormat, OutShape>::descriptor,
          DST_ACCUM_MODE,
          block_ct_dim,
          full_ct_dim>()));
    PACK((llk_init_packer_dest_offset_registers<PackMode::Untilize, false /*diagonal*/>()));
}

// clang-format off
/**
 * Experimental id-free pack-untilize of one block. The op owns the row/column loops and self-syncs DEST.
 * Runtime "where": in.l1_address (unpack base; the op offsets by the per-tile input stride derived from
 * InShape) and out.l1_address (the block's first row-major output tile).
 *
 * ADDRESSING ASSUMPTION vs the legacy BH pack_untilize (fifo_page_size == one tile_size):
 *   Legacy pack_untilize advances the packer write address across tile-rows by the CB's ACTUAL
 *   fifo_page_size (full_ct_dim * fifo_page_size, read from the CB interface). This id-free op has no CB
 *   handle, so llk_pack_untilize derives that per-row stride from the OUTPUT descriptor and ASSUMES
 *   fifo_page_size == a single tile's size (SCALE_DATUM_SIZE >> 4). Exact for linear formats
 *   (Float32 / Float16 / int, incl. the tested bf16); for block formats (Bfp8/Bfp4 -- shared-exponent bytes)
 *   or padded/multi-tile pages the derived stride diverges and block_rt_dim > 1 output would be mis-placed.
 *   The input side has the same assumption for the per-tile unpack base (in.l1_address + c * in_stride).
 *   block_rt_dim == 1 AND block_ct_dim == 1 (the current test path) are unaffected (c and the row stride
 *   are never applied). General fix: read the CB fifo_page_size (as legacy) or add the exponent bytes.
 *
 * | Template | block_ct_dim/full_ct_dim | Block width / full input width in tiles | uint32_t | | False |
 * | Function | in            | Input operand (tilized; unpack base)          | LLKOperand | | True |
 * | Function | block_rt_dim  | Height of the block in tiles (rows to pack)   | uint32_t   | >= 1 | True |
 * | Function | out           | Output operand (first row-major tile address) | LLKOperand | | True |
 * | Function | block_c_index | Block column index (when full_ct_dim > block_ct_dim) | uint32_t | >= 0 | False |
 */
// clang-format on
template <
    std::uint32_t block_ct_dim = 8,
    std::uint32_t full_ct_dim = block_ct_dim,
    DataFormat InFormat,
    TensorShape InShape,
    DataFormat OutFormat,
    TensorShape OutShape>
ALWI void pack_untilize_block(
    LLKOperand<InFormat, InShape> in,
    std::uint32_t block_rt_dim,
    LLKOperand<OutFormat, OutShape> out,
    std::uint32_t block_c_index = 0) {
    // Per-tile input stride (16B words), folded to a compile-time constant. Assumes fifo_page_size == a
    // single tile's size (see the header note); exact for linear formats.
    constexpr std::uint32_t in_tile_stride =
        SCALE_DATUM_SIZE(static_cast<std::uint32_t>(InFormat), InShape.total_tensor_size()) >> 4;

    for (std::uint32_t r = 0; r < block_rt_dim; ++r) {
        MATH((llk_math_wait_for_dest_available()));
        for (std::uint32_t c = 0; c < block_ct_dim; ++c) {
            UNPACK((llk_unpack_A<
                    LLKOperand<InFormat, InShape>::descriptor,
                    DST_ACCUM_MODE,
                    BroadcastType::NONE,
                    false /*acc_to_dest*/,
                    EltwiseBinaryReuseDestType::NONE,
                    UnpackToDestEn>(in.l1_address + c * in_tile_stride)));
            MATH((llk_math_eltwise_unary_datacopy<
                  LLKOperand<InFormat, InShape>::descriptor,
                  DataCopyType::A2D,
                  DST_ACCUM_MODE,
                  BroadcastType::NONE,
                  UnpackToDestEn>(c)));
        }
        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
        PACK((llk_packer_wait_for_math_done()));
        PACK((llk_pack_untilize<LLKOperand<OutFormat, OutShape>::descriptor, DST_ACCUM_MODE, block_ct_dim, full_ct_dim>(
            1 /*block_rt_dim*/, out.l1_address, block_c_index)));
        PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
    }
}

// clang-format off
/**
 * Experimental id-free pack-untilize uninit. Restores the packer Z stride (via the output descriptor's
 * register format) and resets the packer to Default mode so a subsequent op can reprogram it. Mirrors the
 * legacy pack_untilize_uninit (BH path).
 */
// clang-format on
template <DataFormat OutFormat, TensorShape OutShape>
ALWI void pack_untilize_uninit(LLKOperand<OutFormat, OutShape> /*out*/) {
    PACK((llk_pack_untilize_uninit<LLKOperand<OutFormat, OutShape>::descriptor, DST_ACCUM_MODE>()));
    PACK((llk_init_packer_dest_offset_registers<PackMode::Default>()));
    PACK((llk_pack_reconfig_data_format<LLKOperand<OutFormat, OutShape>::descriptor, DST_ACCUM_MODE>()));
    PACK((llk_pack_init<LLKOperand<OutFormat, OutShape>::descriptor, DST_ACCUM_MODE, PackMode::Default>()));
}

#endif  // ARCH_BLACKHOLE

}  // namespace experimental
}  // namespace ckernel
