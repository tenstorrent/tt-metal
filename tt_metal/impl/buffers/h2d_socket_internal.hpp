// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

#include <tt_stl/span.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>

namespace tt::tt_metal::distributed {
class H2DSocket;
class MeshDevice;
enum class H2DMode : uint8_t;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::experimental::detail {

// Friend access struct: lets this translation unit reach H2DSocket's non-blocking writes
// (try_write_impl, try_stage_impl, notify_receiver_unfenced) without exposing them on the
// public H2DSocket surface.
struct H2DSocketTryWriteAccess {
    static bool try_write(distributed::H2DSocket& socket, void* data, uint32_t num_pages);
    static uint32_t try_stage(distributed::H2DSocket& socket, ttsl::Span<const uint8_t* const> pages);
    static void publish_staged(ttsl::Span<distributed::H2DSocket* const> sockets);
};

// Non-blocking H2DSocket write. Returns true if `num_pages` were successfully
// pushed; returns false immediately if the FIFO doesn't currently have room
// (no spinning). Caller is expected to retry, or attempt a different socket,
// without blocking.
inline bool try_write(distributed::H2DSocket& socket, void* data, uint32_t num_pages) {
    return H2DSocketTryWriteAccess::try_write(socket, data, num_pages);
}

// Non-blocking writes to many sockets at once. try_stage writes as many of `pages` (one pointer per
// page, each the socket's page size) into one socket as its FIFO has room for, in order, and returns how
// many (0 if none fit); it reads only the pages it writes. publish_staged makes everything staged on
// `sockets` visible to their receivers. A staged page reaches its receiver only through publish_staged.
// try_write fences each socket twice for each write; a batch of pages on many sockets shares
// publish_staged's two fences.
inline uint32_t try_stage(distributed::H2DSocket& socket, ttsl::Span<const uint8_t* const> pages) {
    return H2DSocketTryWriteAccess::try_stage(socket, pages);
}

inline void publish_staged(ttsl::Span<distributed::H2DSocket* const> sockets) {
    H2DSocketTryWriteAccess::publish_staged(sockets);
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
// alignment slack). Host writes land at dram_l1_noc_offset plus the local address, through a
// window anchored there on Blackhole silicon. Always uses HOST_PUSH mode.
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
