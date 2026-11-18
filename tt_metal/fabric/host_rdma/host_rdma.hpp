// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Host-initiated RDMA over TT-Fabric (single domain), through a router's host_rdma sender channel
// (TT_METAL_ENABLE_FABRIC_HOST_RDMA). Two ways in:
//   - HostRdmaService: the host posts write requests (device addresses: source on the chip, destination on another
//     chip) to a service kernel on a Tensix core, which reads the data, forms the packets and injects them. The router
//     credits the kernel.
//   - HostRdmaChannel: the host builds the packets itself and pushes them into the router's slots; the router credits
//     the host in host memory.
//
// The two host-facing paths are abstract, so their transport can change without touching the channel:
//   - DeviceWriter: how the host writes into device memory (packets, the channel's doorbell). Today: UMD PCIe DMA.
//   - HostMemoryTarget: memory the device writes to for the host (the channel's credits). Today: pinned host memory,
//     written over the NoC through the PCIe core.
//
// Completions use the cross-domain "self domain" ack: the last packet of a write is a fused write + atomic increment of
// a completion counter on the destination chip, so the counter moves only after every byte of the write has landed.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
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
class Hal;
namespace distributed {
class MeshDevice;
}
namespace experimental {
class PinnedMemory;
}
}  // namespace tt::tt_metal

namespace tt::tt_fabric {
class FabricContext;
class ControlPlane;
}  // namespace tt::tt_fabric

namespace tt::tt_fabric::host_rdma {

// ---------------------------------------------------------------------------------------------------------------------
// Host -> device: the write side
// ---------------------------------------------------------------------------------------------------------------------

// How the host writes into a core's memory. `write` may be unordered against later writes until `fence`; `write_reg`
// is ordered after every fenced write and is what tells the router a packet is there.
class DeviceWriter {
public:
    virtual ~DeviceWriter() = default;

    // Bulk write to `addr` on `virtual_core` of `chip`.
    virtual void write(
        ChipId chip, const tt_metal::CoreCoord& virtual_core, uint64_t addr, std::span<const std::byte> data) = 0;
    // Every earlier `write` to `virtual_core` has landed before any later write to it.
    virtual void fence(ChipId chip, const tt_metal::CoreCoord& virtual_core) = 0;
    // An ordered 4-byte write (a register or an L1 word).
    virtual void write_word(ChipId chip, const tt_metal::CoreCoord& virtual_core, uint64_t addr, uint32_t value) = 0;
    // A 4-byte read (setup only: off the data path).
    virtual uint32_t read_word(ChipId chip, const tt_metal::CoreCoord& virtual_core, uint64_t addr) = 0;
};

// Writes through UMD: bulk writes by PCIe DMA where UMD takes the shape, MMIO otherwise; fence by an L1 barrier on the
// core; words by MMIO register writes.
class UmdDmaDeviceWriter final : public DeviceWriter {
public:
    explicit UmdDmaDeviceWriter(const tt::Cluster& cluster);

    void write(
        ChipId chip, const tt_metal::CoreCoord& virtual_core, uint64_t addr, std::span<const std::byte> data) override;
    void fence(ChipId chip, const tt_metal::CoreCoord& virtual_core) override;
    void write_word(ChipId chip, const tt_metal::CoreCoord& virtual_core, uint64_t addr, uint32_t value) override;
    uint32_t read_word(ChipId chip, const tt_metal::CoreCoord& virtual_core, uint64_t addr) override;

private:
    const tt::Cluster& cluster_;
};

// ---------------------------------------------------------------------------------------------------------------------
// Device -> host: the write to host
// ---------------------------------------------------------------------------------------------------------------------

// Where a device writes a word for the host: the node in front of host memory and the address there, as the device's
// NoC takes them (HostRdmaChannelInfo carries them to the router).
struct DeviceVisibleAddress {
    uint32_t noc_xy = 0;
    uint64_t addr = 0;
};

// Words of host memory a device can write. The router of a host_rdma channel writes its credit count to one of them.
class HostMemoryTarget {
public:
    virtual ~HostMemoryTarget() = default;

    virtual size_t num_words() const = 0;
    // Where a core on `chip` writes word `index`.
    virtual DeviceVisibleAddress device_address(ChipId chip, size_t index) const = 0;
    // The word as the device last wrote it (or as the host last set it).
    virtual uint32_t read(size_t index) const = 0;
    virtual void write(size_t index, uint32_t value) = 0;
};

// Pinned host memory mapped to the NoC of one chip (the chip whose PCIe core the device writes through).
class PinnedHostMemoryTarget final : public HostMemoryTarget {
public:
    // `num_words` words of pinned memory, zeroed, mapped for the unit mesh `mesh_device`.
    PinnedHostMemoryTarget(tt_metal::distributed::MeshDevice& mesh_device, size_t num_words);
    ~PinnedHostMemoryTarget() override;

    size_t num_words() const override { return num_words_; }
    DeviceVisibleAddress device_address(ChipId chip, size_t index) const override;
    uint32_t read(size_t index) const override;
    void write(size_t index, uint32_t value) override;

private:
    size_t num_words_;
    size_t alloc_bytes_;
    uint32_t* words_ = nullptr;
    std::shared_ptr<tt_metal::experimental::PinnedMemory> pinned_;
};

// ---------------------------------------------------------------------------------------------------------------------
// Host -> router: how packets reach the channel's slots
// ---------------------------------------------------------------------------------------------------------------------

// A router's host_rdma slot ring, as the router laid it out.
struct RouterSlots {
    ChipId chip = 0;
    tt_metal::CoreCoord router_core;  // virtual
    uint32_t slots_base = 0;          // L1 address of slot 0
    uint32_t num_slots = 0;
    uint32_t slot_bytes = 0;
    uint32_t free_slots_addr = 0;  // the channel's free-slots register, its increment-on-write address
    uint32_t start = 0;            // the router's slot count when the channel opened: slot index = count % num_slots
};

// How packets the host builds get into the router's slots. The channel builds each packet in slot(i), the host memory
// that becomes router slot i, and submits runs of contiguous slots; a transport delivers a run's packets whole into
// the router's slots and only then takes the slots on the router's free-slots register.
class PacketTransport {
public:
    virtual ~PacketTransport() = default;

    // Host memory for slot `slot` (RouterSlots::slot_bytes of it). Free to write once the router has freed the slot.
    virtual std::byte* slot(uint32_t slot) = 0;
    // Delivers packets in slots first, first + 1, ... (no wrap), of packet_bytes[i] bytes each (header included).
    virtual void submit(uint32_t first, std::span<const uint32_t> packet_bytes) = 0;
    // The transport's state, for error messages.
    virtual std::string describe() const { return {}; }
};

// Builds a channel's transport for its router's slots.
using TransportFactory = std::function<std::unique_ptr<PacketTransport>(const RouterSlots&)>;

// The host pushes: one DeviceWriter write of the run into the router's slots, a fence, and a doorbell write to the
// free-slots register. Each run costs the writer's per-call overheads (with UMD: a DMA call and an L1 barrier).
TransportFactory host_push_transport(std::shared_ptr<DeviceWriter> writer);

// A router's host_rdma channel as the host finds it: its slots (with the router's slot count now as `start`) and its
// connection info in the router's L1.
struct RouterChannel {
    FabricNodeId src{MeshId{0}, 0};
    chan_id_t eth_chan = 0;
    uint32_t channel = 0;  // its sender channel index (VC0's last)
    uint32_t connection_info_addr = 0;
    RouterSlots slots;
};

// The host_rdma channel of the router on `src` at `eth_chan`.
RouterChannel find_router_channel(FabricNodeId src, chan_id_t eth_chan);
// Points the router's credits for the channel at `target` (a node and an address it writes the count to), and enables
// them; or withdraws them.
void set_router_credit_target(const RouterChannel& channel, const DeviceVisibleAddress& target);
void clear_router_credit_target(const RouterChannel& channel);
// 1D fabrics: chips crossed from `src`, leaving through `eth_chan`, to `dst`.
uint32_t hops_between(FabricNodeId src, chan_id_t eth_chan, FabricNodeId dst);

// ---------------------------------------------------------------------------------------------------------------------
// The channel
// ---------------------------------------------------------------------------------------------------------------------

// The self-domain ack of a write: once every packet of the write has landed, `increment` is added atomically to the
// 32-bit counter at `addr` on `virtual_core` of the destination chip.
struct Completion {
    tt_metal::CoreCoord virtual_core;
    uint64_t addr = 0;
    uint32_t increment = 1;
};

// One router's host_rdma channel, opened by the host. Writes are split into packets of at most max_payload_bytes();
// each waits for a free slot (the router's credit in host memory), is written into the slot, fenced, and then rung in
// on the channel's free-slots register. The router sends packets in order, so a write's packets land in order.
//
// Not thread safe; one opener per channel (the router keeps no per-producer state for it).
class HostRdmaChannel {
public:
    // The Ethernet channel of the router on `src` that sends toward `dst` on link `link_idx` (2D: the first hop of the
    // route; 1D: the direction toward it).
    static chan_id_t eth_chan_toward(FabricNodeId src, FabricNodeId dst, uint32_t link_idx = 0);

    // The host_rdma channel of the router on `src` that sends toward `dst` (2D: the first hop of the route; 1D: the
    // neighbour direction), on link `link_idx`. Its credits go to word `credit_word` of `credits`, which must be
    // writable from `src`'s chip.
    static std::unique_ptr<HostRdmaChannel> open_toward(
        FabricNodeId src,
        FabricNodeId dst,
        uint32_t link_idx,
        std::shared_ptr<DeviceWriter> writer,
        std::shared_ptr<HostMemoryTarget> credits,
        size_t credit_word);

    // The host_rdma channel of the router on `src` at Ethernet channel `eth_chan`. `writer` opens and closes the
    // channel (its connection info in the router's L1); packets go through the transport `transport` builds, the host
    // push through `writer` by default.
    HostRdmaChannel(
        FabricNodeId src,
        chan_id_t eth_chan,
        std::shared_ptr<DeviceWriter> writer,
        std::shared_ptr<HostMemoryTarget> credits,
        size_t credit_word,
        const TransportFactory& transport = {});
    // Drains the channel, then withdraws the credit target from the router.
    ~HostRdmaChannel();

    HostRdmaChannel(const HostRdmaChannel&) = delete;
    HostRdmaChannel& operator=(const HostRdmaChannel&) = delete;

    // Writes `data` to `dst_addr` on `dst_virtual_core` of chip `dst` (same mesh as the source; not the source chip).
    // With `completion`, the last packet also increments the completion counter.
    //
    // Packets go to the router in batches: they are staged in host memory and written as one contiguous transfer, one
    // fence and one doorbell, at max_batch_packets() packets, at the end of the slot ring, after a write with a
    // completion, or on flush()/drain(). So a write without a completion may stay staged until one of those; a write
    // with a completion returns with every packet with the router (not delivered: wait on the completion for that).
    void write(
        FabricNodeId dst,
        const tt_metal::CoreCoord& dst_virtual_core,
        uint64_t dst_addr,
        std::span<const std::byte> data,
        const std::optional<Completion>& completion = std::nullopt);

    // Sends the staged packets to the router.
    void flush();
    // Flushes, then waits until the router has freed every slot this channel filled (every packet has left the router).
    void drain();

    // The most packets sent to the router as one batch (default: half the channel's slots, so the router drains one
    // half while the host fills the other).
    void set_max_batch_packets(uint32_t packets);
    uint32_t max_batch_packets() const { return max_batch_; }
    uint32_t num_slots() const { return num_slots_; }

    size_t max_payload_bytes() const { return max_payload_bytes_; }
    FabricNodeId src() const { return src_; }
    chan_id_t eth_chan() const { return eth_chan_; }
    // The packets the host has rung in, and the slots the router has freed (its credit count).
    uint32_t packets_sent() const { return sent_; }
    uint32_t packets_credited() const;

private:
    // Stages the packet in header_scratch_ plus `payload` (chunk_bytes of it) in the next slot, flushing first where
    // the batch is full or the ring wraps.
    void stage_packet(size_t chunk_bytes, std::span<const std::byte> payload);
    // Waits (with a timeout) until at most `in_flight_limit` packets are uncredited.
    void wait_for_credits(uint32_t in_flight_limit);
    // A packet header for `dst`, of the run's header type, into header_scratch_ (its size: header_bytes_).
    void build_header(
        FabricNodeId dst,
        uint64_t write_noc_addr,
        uint32_t payload_bytes,
        const std::optional<std::pair<uint64_t, uint32_t>>& completion_noc_addr_and_inc);
    uint64_t noc_addr(const tt_metal::CoreCoord& virtual_core, uint64_t addr) const;
    uint32_t hops_to(FabricNodeId dst) const;  // 1D only

    const tt::tt_fabric::FabricContext& fabric_context_;
    const tt::tt_fabric::ControlPlane& control_plane_;
    const tt::tt_metal::Hal& hal_;
    RouterChannel router_;
    FabricNodeId src_;
    chan_id_t eth_chan_;
    ChipId chip_;
    tt_metal::CoreCoord router_core_;
    std::shared_ptr<DeviceWriter> writer_;
    std::shared_ptr<HostMemoryTarget> credits_;
    size_t credit_word_;

    // The channel, as the router laid it out.
    uint32_t slots_base_ = 0;
    uint32_t num_slots_ = 0;
    uint32_t slot_bytes_ = 0;
    uint32_t connection_info_addr_ = 0;
    uint32_t doorbell_addr_ = 0;  // the channel's free-slots register's increment-on-write address
    size_t header_bytes_ = 0;
    size_t max_payload_bytes_ = 0;

    // 2D: the source chip's route table, as its routers hold it.
    bool is_2d_ = false;
    std::vector<uint8_t> routing_table_;

    uint32_t sent_ = 0;    // packets rung in (absolute, from the router's count at open)
    uint32_t staged_ = 0;  // packets staged and not yet rung in, in slots from staged_first_slot_
    uint32_t staged_first_slot_ = 0;
    std::vector<uint32_t> staged_bytes_;  // the staged packets' sizes
    uint32_t max_batch_ = 1;
    uint32_t next_slot_ = 0;      // the slot the next packet goes into
    uint32_t known_credits_ = 0;  // the latest credit count read
    std::vector<std::byte> header_scratch_;
    std::unique_ptr<PacketTransport> transport_;  // last: it may run device work that reads the members above
};

// ---------------------------------------------------------------------------------------------------------------------
// The request service
// ---------------------------------------------------------------------------------------------------------------------

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
    // Drains, withdraws the routers' credits from the kernel and stops it.
    ~HostRdmaService();

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
    const tt::Cluster& cluster_;
    ChipId chip_ = 0;
    FabricNodeId node_{MeshId{0}, 0};
    tt_metal::CoreCoord core_;  // virtual
    bool is_2d_ = false;
    uint32_t layout_addr_ = 0;
    std::vector<Channel> channels_;
};

}  // namespace tt::tt_fabric::host_rdma
