// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include <tt-metalium/data_types.hpp>
#include <tt-metalium/experimental/tensor/tensor_types.hpp>
#include <tt-metalium/experimental/tensor/topology/distributed_tensor_configs.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/shape.hpp>

#include "tt_metal/distributed/hd_socket_descriptor.hpp"

namespace tt::tt_metal::distributed {

struct D2HStreamServiceDescriptor {
    static constexpr uint32_t kVersion = 2;

    tt::tt_metal::Shape global_shape;
    DataType global_dtype = DataType::INVALID;

    MeshShape mesh_shape;
    MeshMapperConfig mapper_config;

    uint32_t socket_page_size = 0;
    uint32_t num_socket_pages = 0;
    uint32_t metadata_size_bytes = 0;

    MeshComposerConfig composer_config;

    std::vector<std::pair<MeshCoordinate, HDSocketDescriptor>> per_coord_entries;

    void write_to_file(const std::string& path) const;
    static D2HStreamServiceDescriptor wait_and_read(const std::string& path, uint32_t timeout_ms = 10000);
};

inline std::string descriptor_path_for_d2h_service(const std::string& service_id) {
    return fmt::format("/dev/shm/tt_d2h_stream_service_{}.bin", service_id);
}

}  // namespace tt::tt_metal::distributed
