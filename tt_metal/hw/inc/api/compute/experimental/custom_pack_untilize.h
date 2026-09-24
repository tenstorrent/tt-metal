// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/pack_untilize.h"
#include "sanitizer/api.h"

namespace ckernel {
#if defined(ARCH_BLACKHOLE)

#ifdef TRISC_PACK
namespace pack_untilize_detail {
template <
    std::uint32_t block_ct_dim,
    std::uint32_t full_ct_dim,
    bool narrow_row,
    std::uint32_t row_num_datums,
    bool dense>
inline void configure_explicit_geometry(std::uint32_t ocb, std::uint32_t face_r_dim, std::uint32_t num_faces) {
    const std::uint32_t output_id = get_output_id(ocb);
    // Match the geometry-to-tile mapping used by JitBuildOptions: one/two
    // faces occupy one face row; four faces occupy two face rows.
    const std::uint32_t tile_r_dim = face_r_dim * (num_faces > 2 ? 2 : 1);
    const std::uint32_t tile_c_dim = num_faces == 1 ? FACE_C_DIM : TILE_C_DIM;
    const bool partial_face = tile_r_dim < TILE_R_DIM;
    const std::uint32_t tile_size = get_local_cb_interface(output_id).fifo_page_size;
    SAN_HOOK(configure(
        StateVal<Operand<Exu::Pack>::DestWidth32>(DST_ACCUM_MODE),
        StateVal<Operand<Exu::Pack>::InputFormat>(pack_src_format[output_id]),
        StateVal<Operand<Exu::Pack>::OutputFormat>(pack_dst_format[output_id]),
        StateVal<Operand<Exu::Pack>::FaceHeight>(face_r_dim),
        StateVal<Operand<Exu::Pack>::TileWidth>(tile_c_dim),
        StateVal<Operand<Exu::Pack>::NumFaces>(num_faces),
        StateVal<Operand<Exu::Pack>::PartialFace>(partial_face),
        StateDiscard<std::uint32_t>(tile_size)));
    _llk_pack_hw_configure_<DST_ACCUM_MODE, PackMode::Default>(
        pack_src_format[output_id],
        pack_dst_format[output_id],
        tile_size,
        face_r_dim,
        tile_c_dim,
        num_faces,
        partial_face,
        0);
    llk_pack_untilize_init_impl<block_ct_dim, full_ct_dim, narrow_row, row_num_datums, dense>(
        pack_src_format[output_id], pack_dst_format[output_id], face_r_dim, num_faces);
}

template <
    std::uint32_t block_ct_dim,
    std::uint32_t full_ct_dim,
    bool narrow_row,
    std::uint32_t row_num_datums,
    bool dense>
inline void pack_explicit_geometry(
    std::uint32_t ocb,
    std::uint32_t face_r_dim,
    std::uint32_t num_faces,
    std::uint32_t block_rt_dim,
    std::uint32_t block_c_index,
    std::uint32_t tile_dst_rt_offset) {
    const std::uint32_t output_id = get_output_id(ocb);
    llk_pack_untilize_impl<block_ct_dim, full_ct_dim, narrow_row, row_num_datums, 0, dense>(
        block_rt_dim,
        get_local_cb_interface(output_id).fifo_wr_ptr - 1,
        pack_src_format[output_id],
        pack_dst_format[output_id],
        full_ct_dim * get_local_cb_interface(output_id).fifo_page_size,
        face_r_dim,
        num_faces,
        block_c_index,
        tile_dst_rt_offset);
}
}  // namespace pack_untilize_detail
#endif

// Pack data already in DEST using explicit face geometry, independent of the
// output CB's logical tile shape. This is needed by tiny-tile SDPA reductions.
// Does not reconfigure math/remap or reset destination synchronization.
// face_r_dim must be 1..16; num_faces must be 1, 2, or 4. Block width must fit
// the caller's acquired DEST capacity and divide full_ct_dim.
// As with pack_untilize_dest, block-float output is unsupported.
template <
    std::uint32_t block_ct_dim = 8,
    std::uint32_t full_ct_dim = block_ct_dim,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM,
    bool dense = false>
ALWI void custom_pack_untilize_dest_init(
    std::uint32_t ocb,
    std::uint32_t face_r_dim = 16,
    std::uint32_t num_faces = 4,
    std::uint32_t call_line = __builtin_LINE()) {
    static_assert(block_ct_dim > 0 && full_ct_dim % block_ct_dim == 0, "untilize block width must divide full width");
    state_configure<Operand::PACK>(ocb, call_line);
    PACK((
        pack_untilize_detail::configure_explicit_geometry<block_ct_dim, full_ct_dim, narrow_row, row_num_datums, dense>(
            ocb, face_r_dim, num_faces)));
    PACK((llk_init_packer_dest_offset_registers<PackMode::Untilize, false>(ocb)));
}

// Pair with custom_pack_untilize_dest_init using the same template parameters
// and face geometry.
// The ordinary pack_untilize_dest reads num_faces from CB metadata again.
template <
    std::uint32_t block_ct_dim = 8,
    std::uint32_t full_ct_dim = block_ct_dim,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM,
    bool dense = false>
ALWI void custom_pack_untilize_dest(
    std::uint32_t ocb,
    std::uint32_t face_r_dim,
    std::uint32_t num_faces,
    std::uint32_t block_rt_dim = 1,
    std::uint32_t block_c_index = 0,
    std::uint32_t tile_dst_rt_offset = 0) {
    PACK((pack_untilize_detail::pack_explicit_geometry<block_ct_dim, full_ct_dim, narrow_row, row_num_datums, dense>(
        ocb, face_r_dim, num_faces, block_rt_dim, block_c_index, tile_dst_rt_offset)));
}

#endif  // ARCH_BLACKHOLE
}  // namespace ckernel
