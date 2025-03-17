// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_impl.hpp>

namespace tt::tt_fabric {

void append_fabric_connection_rt_args(
    chip_id_t src_phys_chip_id,
    chip_id_t dst_phys_chip_id,
    routing_plane_id_t routing_plane,
    tt::tt_metal::Program& worker_program,
    CoreRangeSet worker_cores,
    std::vector<uint32_t>& worker_args);

}  // namespace tt::tt_fabric
