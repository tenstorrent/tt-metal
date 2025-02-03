// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>

namespace tt::tt_metal::distributed::test::utils {

std::vector<std::shared_ptr<Program>> create_eltwise_bin_programs(
    std::shared_ptr<MeshDevice>& mesh_device,
    std::vector<std::shared_ptr<MeshBuffer>>& src0_bufs,
    std::vector<std::shared_ptr<MeshBuffer>>& src1_bufs,
    std::vector<std::shared_ptr<MeshBuffer>>& output_bufs);

}  // namespace tt::tt_metal::distributed::test::utils
