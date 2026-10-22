// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <tt-metalium/mesh_device.hpp>

namespace tt::tt_metal::distributed {

/**
 * Carries the host-bridged fabric links of a mesh: E2H (the router writes its packets and credit
 * updates as frames into a pinned host ring), H2H (MPI, the HostMeshSocket transport; in-process
 * loopback when the far router is on this host too) and H2E (the frames are replayed on the far
 * router's core: payloads by the PCIe DMA engine, credits by a stream register write). The fabric's
 * own credit system runs unchanged across the bridge; nothing here is fabric-aware beyond the frame
 * format in hostdevcommon/fabric_host_bridge.h.
 *
 * Which links are bridged comes from TT_METAL_FABRIC_HOST_BRIDGE (tt_metal/fabric/host_bridge_links.hpp),
 * the same setting the router builder read, so the routers and the bridge agree. Construct it on every
 * rank that owns a bridged router, after the mesh is open; the routers start sending once it has armed
 * them. The relay runs on the process-wide RelayLoop thread.
 */
class FabricHostBridge {
public:
    explicit FabricHostBridge(const std::shared_ptr<MeshDevice>& mesh_device, uint32_t ring_pages = 64);
    ~FabricHostBridge();
    FabricHostBridge(const FabricHostBridge&) = delete;
    FabricHostBridge& operator=(const FabricHostBridge&) = delete;

    /** Bridged router ends on this host. */
    size_t num_ends() const;
    /** First relay error, or empty. */
    std::string first_error() const;
    /** Per-link counters for logs. */
    std::string describe() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal::distributed
