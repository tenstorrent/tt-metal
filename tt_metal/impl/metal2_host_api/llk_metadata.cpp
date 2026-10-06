// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/metal2_host_api/llk_metadata.hpp"

#include <algorithm>
#include <cstdint>

#include <fmt/format.h>
#include <tt-metalium/constants.hpp>

#include "impl/data_format/hw_data_format.hpp"

namespace tt::tt_metal {

std::string serialize_llk_metadata(const LLKMetadata& metadata) {
    const Tile& tile = metadata.tile;
    const uint32_t face_r_dim = tile.get_face_shape()[0];
    const uint32_t num_faces = tile.get_num_faces();
    const uint32_t num_faces_c_dim = std::min(tile.get_width() / constants::FACE_WIDTH, num_faces);
    const uint32_t num_faces_r_dim = num_faces / num_faces_c_dim;
    return fmt::format(
        "::binding_details::LLKMetadata{{.format = {}u, .face_r_dim = {}u, .face_c_dim = {}u, "
        ".num_faces_r_dim = {}u, .num_faces_c_dim = {}u}}",
        host_data_format_to_hw(metadata.format),
        face_r_dim,
        constants::FACE_WIDTH,
        num_faces_r_dim,
        num_faces_c_dim);
}

}  // namespace tt::tt_metal
