// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/experimental/tensor/tensor_types.hpp>
#include <tt-metalium/experimental/tensor/topology/distributed_tensor_configs.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/shape.hpp>

#include "tt_metal/distributed/hd_socket_descriptor.hpp"

namespace tt::tt_metal::distributed {

// Serializable descriptor for cross-process H2DStreamService attachment.
struct H2DStreamServiceDescriptor {
    static constexpr uint32_t kVersion = 1;

    // Source-tensor spec snapshot. Limited to ROW_MAJOR / DRAM-interleaved today.
    tt::tt_metal::Shape global_shape;
    DataType global_dtype = DataType::INVALID;

    // Full mesh shape the service was built against; fed to create_mesh_mapper(mesh_shape, mapper_config).
    MeshShape mesh_shape;

    MeshMapperConfig mapper_config;

    // Chunk plan baked at construction; the connector validates it against its derived plan (mismatch is fatal).
    uint32_t socket_page_size = 0;
    uint32_t num_socket_pages = 0;

    // 0 disables the metadata path entirely on the connector.
    uint32_t metadata_size_bytes = 0;

    BufferType socket_buffer_type = BufferType::L1;
    H2DMode socket_mode = H2DMode::DEVICE_PULL;

    // One entry per participating mesh coord, passed directly to H2DSocket::connect_from_descriptor.
    std::vector<std::pair<MeshCoordinate, HDSocketDescriptor>> per_coord_entries;

    /**
     * @brief Serialize this descriptor to a flatbuffer file.
     * @param path Output path (typically /dev/shm/tt_h2d_stream_service_<id>.bin).
     */
    void write_to_file(const std::string& path) const;

    /**
     * @brief Wait for a service descriptor file to appear and deserialize it.
     * @param path Full path to the descriptor file.
     * @param timeout_ms Max wait time in milliseconds.
     * @return Populated descriptor. Throws if the version stamp does not match.
     */
    static H2DStreamServiceDescriptor wait_and_read(const std::string& path, uint32_t timeout_ms = 10000);
};

inline std::string descriptor_path_for_service(const std::string& service_id) {
    return fmt::format("/dev/shm/tt_h2d_stream_service_{}.bin", service_id);
}

}  // namespace tt::tt_metal::distributed
