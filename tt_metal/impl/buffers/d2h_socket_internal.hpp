// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_metal::distributed {
class D2HSocket;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::experimental::detail {

// Friend access struct: lets internal callers reach D2HSocket::try_read_impl
// without exposing it on the public D2HSocket surface.
struct D2HSocketTryReadAccess {
    static bool try_read(distributed::D2HSocket& socket, void* data, uint32_t num_pages, bool notify_sender);
};

// Non-blocking D2HSocket read. Returns true if `num_pages` were successfully
// consumed; returns false immediately if that many pages are not yet available.
inline bool try_read(distributed::D2HSocket& socket, void* data, uint32_t num_pages, bool notify_sender = true) {
    return D2HSocketTryReadAccess::try_read(socket, data, num_pages, notify_sender);
}

// Friend access struct: lets the real-time profiler re-point a socket at its sender device's coordinate in a
// new mesh view. MeshDevice::reshape() replaces the view a socket was created under, and the socket resolves
// its sender device through the view (is_local / get_device) on every read and in its destructor.
struct D2HSocketSenderCoordAccess {
    static void rebind_sender_device_coord(
        distributed::D2HSocket& socket, const distributed::MeshCoordinate& device_coord);
};

inline void rebind_sender_device_coord(
    distributed::D2HSocket& socket, const distributed::MeshCoordinate& device_coord) {
    D2HSocketSenderCoordAccess::rebind_sender_device_coord(socket, device_coord);
}

}  // namespace tt::tt_metal::experimental::detail
