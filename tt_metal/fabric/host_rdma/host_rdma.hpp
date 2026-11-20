// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Host-initiated RDMA over TT-Fabric (single domain), through a router's host_rdma sender channel
// (TT_METAL_ENABLE_FABRIC_HOST_RDMA): HostRdmaService. The host posts write requests (device addresses: source on the
// chip, destination on another chip) to a service kernel on a Tensix core, which reads the data, forms the packets and
// injects them into the router. The host_rdma channel is a worker channel the router treats as any other; the fabric's
// worker connection APIs do not hand it out, so only the service kernel connects to it.
//
// Completions use the cross-domain "self domain" ack: the last packet of a write is a fused write + atomic increment of
// a completion counter on the destination chip, so the counter moves only after every byte of the write has landed.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <hostdevcommon/fabric_common.h>  // chan_id_t
#include "tt_metal/fabric/hw/inc/edm_fabric/host_rdma_service_abi.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-umd/types/cluster_descriptor_types.hpp>  // ChipId

namespace tt {
class Cluster;
}

namespace tt::tt_metal {
class MetalContext;
namespace distributed {
class CoreWriter;
class MeshDevice;
}  // namespace distributed
}  // namespace tt::tt_metal

namespace tt::tt_fabric::host_rdma {

// ---------------------------------------------------------------------------------------------------------------------
// Router channels
// ---------------------------------------------------------------------------------------------------------------------

// A router's host_rdma channel, as a worker connection to it takes it (addresses in the router's L1).
struct RouterChannel {
    FabricNodeId src{MeshId{0}, 0};
    chan_id_t eth_chan = 0;
    ChipId chip = 0;
    tt_metal::CoreCoord router_core;  // virtual
    uint32_t channel = 0;             // its sender channel index (VC0's last)
    uint32_t slots_base = 0;          // L1 address of slot 0
    uint32_t num_slots = 0;
    uint32_t slot_bytes = 0;                 // the slot stride
    uint32_t free_slots_stream = 0;          // its free-slots stream
    uint32_t connection_info_addr = 0;       // its EDMChannelWorkerLocationInfo
    uint32_t connection_semaphore_addr = 0;  // its connection semaphore
    uint32_t producer_cursor_addr = 0;       // its SenderChannelProducerCursor
};

// The host_rdma channel of the router on `src` at `eth_chan`, in `context`'s fabric.
RouterChannel find_router_channel(tt::tt_metal::MetalContext& context, FabricNodeId src, chan_id_t eth_chan);
// The Ethernet channel of the router on `src` that sends toward `dst` on link `link_idx` (2D: the first hop of the
// route; 1D: the direction toward it).
chan_id_t eth_chan_toward(
    tt::tt_metal::MetalContext& context, FabricNodeId src, FabricNodeId dst, uint32_t link_idx = 0);
// 1D fabrics: chips crossed from `src`, leaving through `eth_chan`, to `dst`.
uint32_t hops_between(tt::tt_metal::MetalContext& context, FabricNodeId src, chan_id_t eth_chan, FabricNodeId dst);

// ---------------------------------------------------------------------------------------------------------------------
// The request service
// ---------------------------------------------------------------------------------------------------------------------

// The self-domain ack of a write: once every packet of the write has landed, `increment` is added atomically to the
// 32-bit counter at `addr` on `virtual_core` of the destination chip.
struct Completion {
    tt_metal::CoreCoord virtual_core;
    uint64_t addr = 0;
    uint32_t increment = 1;
};

// A NoC endpoint: a core (virtual coordinates) and an address in it.
struct DeviceEndpoint {
    tt_metal::CoreCoord virtual_core;
    uint32_t addr = 0;
};

// Write `size` bytes from `src` on the service's chip to `dst_at` on chip `dst`. Addresses and the size are 16 B
// aligned.
struct WriteRequest {
    DeviceEndpoint src;
    FabricNodeId dst{MeshId{0}, 0};
    DeviceEndpoint dst_at;
    uint32_t size = 0;
    std::optional<Completion> completion;
};

// The host_rdma request service of one chip: a kernel on a Tensix core of the chip services the host's write requests.
// For each request it reads the data from the source on its chip, forms the packets' headers and injects the packets
// into the router toward the destination (that router's host_rdma channel), several packets in flight; the router
// credits the kernel. The host posts a request with two MMIO writes (the 64 B request into the channel's queue in the
// kernel's L1, then the queue's produced count) whatever its size, or a batch of requests with one block write and one
// produced count per router; it builds no packets.
//
// Uses L1 of its core from the unreserved base (its channel table and request queues, then a packet staging area), so
// nothing else may use that core. One per chip; not thread safe.
class HostRdmaService : public std::enable_shared_from_this<HostRdmaService> {
public:
    // A service on `logical_core` of the unit mesh `mesh_device`, its kernel started.
    static std::shared_ptr<HostRdmaService> create(
        std::shared_ptr<tt_metal::distributed::MeshDevice> mesh_device, tt_metal::CoreCoord logical_core);
    // Stops the kernel, which closes its router connections. The service must be drained first (drain()): the
    // destructor does not wait for requests.
    ~HostRdmaService() noexcept;

    HostRdmaService(const HostRdmaService&) = delete;
    HostRdmaService& operator=(const HostRdmaService&) = delete;

    // Posts `request` to the router toward its destination (opening that router's channel on first use). Blocks while
    // the channel's queue is full. Returns once posted: wait on the request's completion for its data to land.
    void write(const WriteRequest& request);
    // Posts `requests` as batches: per router channel, in order, one block write of as many requests as its queue takes
    // and one produced-count write for them, instead of two MMIO writes per request.
    void write_batch(std::span<const WriteRequest> requests);
    // Waits until the kernel has injected every posted request.
    void drain();

    FabricNodeId node() const { return node_; }
    std::string describe() const;

private:
    struct Channel {
        uint32_t entry = 0;
        RouterChannel router;
        uint32_t produced = 0;  // requests posted
        uint32_t consumed = 0;  // the kernel's count, as last read
    };

    HostRdmaService(std::shared_ptr<tt_metal::distributed::MeshDevice> mesh_device, tt_metal::CoreCoord logical_core);
    size_t channel_toward(FabricNodeId dst);  // the channel's index in channels_, opened on first use
    service_abi::Request encode(const WriteRequest& request, const Channel& ch) const;
    void post(Channel& ch, std::span<const service_abi::Request> requests);
    uint32_t channel_addr(uint32_t entry) const;
    uint32_t read_word(uint32_t addr) const;
    void write_word(uint32_t addr, uint32_t value);

    std::shared_ptr<tt_metal::distributed::MeshDevice> mesh_device_;
    tt::tt_metal::MetalContext& context_;                        // the mesh device's
    std::unique_ptr<tt_metal::distributed::CoreWriter> writer_;  // every host write to the service core
    const tt::Cluster& cluster_;
    ChipId chip_ = 0;
    FabricNodeId node_{MeshId{0}, 0};
    tt_metal::CoreCoord core_;  // virtual
    bool is_2d_ = false;
    uint32_t layout_addr_ = 0;
    std::vector<Channel> channels_;
};

}  // namespace tt::tt_fabric::host_rdma
