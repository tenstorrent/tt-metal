// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>

namespace tt::tt_metal::distributed {

class MeshDevice;

// A core, described so that another process on the same host can reach it over PCIe.
// Shared by the socket descriptors that other processes read (H2D, D2H, D2D).
struct PCIeCoreLocation {
    // Physical PCIe device number (/dev/tenstorrent/N), not the UMD logical chip id. The logical id
    // depends on TT_VISIBLE_DEVICES, so two processes can disagree on it. The physical number is the
    // same in every process on the host, and PCIeCoreWriter maps it back to a local chip.
    uint32_t pcie_device_id = 0;
    MeshCoreCoord core;      // mesh coord + logical core
    CoreCoord virtual_core;  // virtual (translated) NoC coords

    bool operator==(const PCIeCoreLocation&) const = default;

    // Looks up `core` on `mesh_device`. Fails if the chip has no PCIe device of its own.
    static PCIeCoreLocation resolve(MeshDevice& mesh_device, const MeshCoreCoord& core);
};

}  // namespace tt::tt_metal::distributed
