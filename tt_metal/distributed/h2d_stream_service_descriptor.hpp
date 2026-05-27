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

/**
 * @brief Serializable descriptor for cross-process H2DStreamService attachment.
 *
 * The owner process constructs an H2DStreamService and exports a flatbuffer
 * descriptor that bundles:
 *   - the un-sharded tensor spec (so the connector can validate inputs and
 *     wrap them as borrowed tensors for the mapper),
 *   - the mesh shape and mapper config (so the connector can build a shape-only
 *     mapper without holding a MeshDevice),
 *   - the chunk plan + metadata size (so the connector can sanity-check its
 *     understanding of the wire format),
 *   - one inline HDSocketDescriptor per participating mesh coord (so the
 *     connector attaches every socket in-memory via
 *     H2DSocket::connect_from_descriptor, with no extra file reads).
 *
 * The single-file design eliminates the race between service-level and
 * socket-level descriptors becoming visible on disk.
 */
struct H2DStreamServiceDescriptor {
    static constexpr uint32_t kVersion = 1;

    // Source-tensor spec snapshot. Limited to the regime currently supported by
    // the streaming path (ROW_MAJOR / DRAM-interleaved); layout / memory layout
    // / buffer type are reconstructed by the connector using the same fixed
    // values. Extend this together with the streaming path's support.
    tt::tt_metal::Shape global_shape;
    DataType global_dtype = DataType::INVALID;

    // Full mesh shape the service was constructed against. Fed to the
    // shape-only `create_mesh_mapper(mesh_shape, mapper_config)`.
    MeshShape mesh_shape;

    // Mapper config. Reconstructed via `mapper_->config()` on the owner.
    MeshMapperConfig mapper_config;

    // Chunk plan baked at construction. The connector validates these against
    // its own derived plan at attach time; a mismatch is TT_FATAL.
    uint32_t socket_page_size = 0;
    uint32_t num_socket_pages = 0;

    // 0 disables the metadata path entirely on the connector.
    uint32_t metadata_size_bytes = 0;

    // Mirrored from `Config` so the connector reconstructs the same socket
    // contract.
    BufferType socket_buffer_type = BufferType::L1;
    H2DMode socket_mode = H2DMode::DEVICE_PULL;

    // One entry per participating mesh coord. The HDSocketDescriptor is the
    // exact struct the connector hands to H2DSocket::connect_from_descriptor.
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
