// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace tt::tt_metal::distributed {

/**
 * One end of a host-bridged fabric link: the router on `eth_chan` of `device` was built with
 * TT_METAL_FABRIC_HOST_BRIDGE (tt_metal/fabric/host_bridge_links.hpp) and the far router's host runs a
 * FabricHostBridge for the same link. `link_id` must agree on both ranks: it selects the MPI tags.
 */
struct FabricHostBridgeLink {
    MeshCoordinate device{0, 0};
    uint32_t eth_chan = 0;
    multihost::Rank peer_rank{0};
    uint32_t link_id = 0;
};

/**
 * Carries host-bridged fabric links between two hosts: E2H (the router writes its packets and credit
 * updates as frames into a pinned host ring), H2H (MPI, the HostMeshSocket transport) and H2E (the
 * frames are replayed on the far router's core: payloads by the PCIe DMA engine, credits by a stream
 * register write). The fabric's own credit system runs unchanged across the bridge; nothing here is
 * fabric-aware beyond the frame format in hostdevcommon/fabric_host_bridge.h.
 *
 * Construct it on both ranks after the mesh is open; the routers start sending once it has armed them.
 * The relay runs on the process-wide RelayLoop thread.
 */
class FabricHostBridge {
public:
    FabricHostBridge(
        const std::shared_ptr<MeshDevice>& mesh_device,
        std::vector<FabricHostBridgeLink> links,
        uint32_t ring_pages = 64);
    ~FabricHostBridge();
    FabricHostBridge(const FabricHostBridge&) = delete;
    FabricHostBridge& operator=(const FabricHostBridge&) = delete;

    /** First relay error, or empty. */
    std::string first_error() const;
    /** Per-link counters for logs. */
    std::string describe() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal::distributed
