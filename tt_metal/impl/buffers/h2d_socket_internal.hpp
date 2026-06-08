// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>

namespace tt::tt_metal::distributed {
class H2DSocket;
class MeshDevice;
enum class H2DMode : uint8_t;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::experimental::detail {

// Friend access struct: lets this translation unit reach H2DSocket::try_write_impl
// without exposing it on the public H2DSocket surface.
struct H2DSocketTryWriteAccess {
    static bool try_write(distributed::H2DSocket& socket, void* data, uint32_t num_pages);
};

// Non-blocking H2DSocket write. Returns true if `num_pages` were successfully
// pushed; returns false immediately if the FIFO doesn't currently have room
// (no spinning). Caller is expected to retry, or attempt a different socket,
// without blocking.
inline bool try_write(distributed::H2DSocket& socket, void* data, uint32_t num_pages) {
    return H2DSocketTryWriteAccess::try_write(socket, data, num_pages);
}

// Friend access struct: lets this translation unit construct an H2DSocket that
// targets a DRAM-core recv_core, bypassing MeshBuffer (the L1 allocator is
// worker-only) and using caller-supplied DRISC-L1 offsets for the config and
// data buffers.
struct H2DSocketDramRecvAccess {
    static std::unique_ptr<distributed::H2DSocket> create(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device,
        const distributed::MeshCoreCoord& recv_core,
        uint32_t fifo_size,
        uint32_t config_l1_local_addr,
        uint32_t data_l1_local_addr,
        uint64_t dram_l1_noc_offset);
};

// Construct an H2DSocket for a DRAM-core recv. The caller owns the L1 layout
// and passes pre-computed local L1 offsets for the socket's config buffer
// (sizeof(receiver_socket_md) bytes) and data FIFO (fifo_size bytes plus PCIe
// alignment slack). Host writes go via cluster.write_core with
// dram_l1_noc_offset added to the local address. Always uses HOST_PUSH mode.
inline std::unique_ptr<distributed::H2DSocket> create_h2d_socket_for_dram_recv(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const distributed::MeshCoreCoord& recv_core,
    uint32_t fifo_size,
    uint32_t config_l1_local_addr,
    uint32_t data_l1_local_addr,
    uint64_t dram_l1_noc_offset) {
    return H2DSocketDramRecvAccess::create(
        mesh_device, recv_core, fifo_size, config_l1_local_addr, data_l1_local_addr, dram_l1_noc_offset);
}

}  // namespace tt::tt_metal::experimental::detail
