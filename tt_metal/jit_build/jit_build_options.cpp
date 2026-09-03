// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_build_options.hpp"

#include <string>

#include <tt-metalium/constants.hpp>

#include "build.hpp"

namespace tt::tt_metal {

namespace {

bool is_supported_tile_shape(uint32_t tile_height, uint32_t tile_width) {
    if (tile_width != constants::FACE_WIDTH && tile_width != constants::TILE_WIDTH) {
        return false;
    }
    if (tile_width == constants::FACE_WIDTH) {
        return tile_height == 1 || tile_height == 2 || tile_height == 4 || tile_height == 8 ||
               tile_height == constants::FACE_HEIGHT || tile_height == constants::TILE_HEIGHT;
    }
    return tile_height == 1 || tile_height == 2 || tile_height == 4 || tile_height == 8 ||
           tile_height == constants::FACE_HEIGHT || tile_height == constants::TILE_HEIGHT;
}

std::optional<Tile> tile_from_unpack_face_geometry(const FaceGeometry& face_geometry) {
    const uint32_t tile_height =
        face_geometry.face_r_dim * (face_geometry.num_faces > 2 ? constants::TILE_HEIGHT / constants::FACE_HEIGHT : 1);
    const uint32_t tile_width = face_geometry.num_faces == 1 ? constants::FACE_WIDTH : constants::TILE_WIDTH;
    if (!is_supported_tile_shape(tile_height, tile_width)) {
        return std::nullopt;
    }
    return Tile({tile_height, tile_width});
}

}  // namespace

JitBuildOptions::JitBuildOptions(const JitBuildEnv& env) : build_env(env), hlk_desc(env.get_max_cbs()) {}

void JitBuildOptions::set_name(const std::string& n) {
    name = n;
    path = build_env.get_out_kernel_root_path() + n;
}

void JitBuildOptions::set_hlk_math_fidelity_all_cores(MathFidelity math_fidelity) {
    hlk_desc.set_hlk_math_fidelity(math_fidelity);
}

void JitBuildOptions::set_hlk_math_approx_mode_all_cores(bool approx_mode) {
    hlk_desc.set_hlk_math_approx_mode(approx_mode);
}

void JitBuildOptions::set_hlk_args_all_cores(void* args, size_t size) { hlk_desc.set_hlk_args(args, size); }

void JitBuildOptions::set_cb_dataformat_all_cores(CBIndex cb_id, DataFormat data_format) {
    set_hlk_operand_dataformat_all_cores((HlkOperand)cb_id, data_format);
}

void JitBuildOptions::set_cb_tile_dims_all_cores(
    CBIndex cb_id,
    uint32_t num_faces,
    uint32_t partial_face,
    uint32_t face_r_dim,
    uint32_t narrow_tile,
    uint32_t tile_r_dim,
    uint32_t tile_c_dim) {
    hlk_desc.set_buf_num_faces((int)cb_id, num_faces);
    hlk_desc.set_buf_partial_face((int)cb_id, partial_face);
    hlk_desc.set_buf_face_r_dim((int)cb_id, face_r_dim);
    hlk_desc.set_buf_narrow_tile((int)cb_id, narrow_tile);
    hlk_desc.set_buf_tile_r_dim((int)cb_id, tile_r_dim);
    hlk_desc.set_buf_tile_c_dim((int)cb_id, tile_c_dim);
}

void JitBuildOptions::set_cb_tile_size_all_cores(CBIndex cb_id, uint32_t tile_size) {
    hlk_desc.set_buf_tile_size((int)cb_id, tile_size);
}

void JitBuildOptions::set_cb_data_fmt_and_tile(CBIndex cb_id, DataFormat data_format, const std::optional<Tile>& tile) {
    set_cb_dataformat_all_cores(cb_id, data_format);
    if (tile.has_value()) {
        set_cb_tile_dims_all_cores(
            cb_id,
            tile->get_num_faces(),
            tile->get_partial_face(),
            tile->get_face_shape()[0],
            tile->get_narrow_tile(),
            tile->get_tile_shape()[0],
            tile->get_tile_shape()[1]);
        set_cb_tile_size_all_cores(cb_id, tile->get_tile_size(data_format));
    } else {
        Tile default_tile;
        set_cb_tile_size_all_cores(cb_id, default_tile.get_tile_size(data_format));
    }
}

void JitBuildOptions::set_cb_data_fmt_tile_and_face_geometry(
    CBIndex cb_id,
    DataFormat data_format,
    const std::optional<Tile>& tile,
    const std::optional<FaceGeometry>& unpack_face_geometry) {
    set_cb_dataformat_all_cores(cb_id, data_format);

    if (tile.has_value() || unpack_face_geometry.has_value()) {
        const Tile default_tile;
        const Tile& requested_tile = tile.value_or(default_tile);
        const std::optional<Tile> face_geometry_tile =
            unpack_face_geometry.has_value() ? tile_from_unpack_face_geometry(*unpack_face_geometry) : std::nullopt;
        const Tile& effective_tile = face_geometry_tile.value_or(requested_tile);
        const uint32_t num_faces =
            unpack_face_geometry.has_value() ? unpack_face_geometry->num_faces : requested_tile.get_num_faces();
        const uint32_t face_r_dim =
            unpack_face_geometry.has_value() ? unpack_face_geometry->face_r_dim : requested_tile.get_face_shape()[0];

        set_cb_tile_dims_all_cores(
            cb_id,
            num_faces,
            effective_tile.get_partial_face(),
            face_r_dim,
            effective_tile.get_narrow_tile(),
            effective_tile.get_tile_shape()[0],
            effective_tile.get_tile_shape()[1]);
        set_cb_tile_size_all_cores(cb_id, effective_tile.get_tile_size(data_format));
    } else {
        Tile default_tile;
        set_cb_tile_size_all_cores(cb_id, default_tile.get_tile_size(data_format));
    }
}

void JitBuildOptions::set_hlk_operand_dataformat_all_cores(HlkOperand op_id, DataFormat data_format) {
    hlk_desc.set_buf_dataformat((int)op_id, data_format);
}

}  // namespace tt::tt_metal
