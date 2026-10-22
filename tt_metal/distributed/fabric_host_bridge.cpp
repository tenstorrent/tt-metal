// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/fabric_host_bridge.hpp>

#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt_stl/assert.hpp>
#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt-umd/tt_device/tt_device.hpp>

#include "hostdevcommon/fabric_host_bridge.h"
#include "tt_metal/distributed/host_transport/host_transport.hpp"
#include "tt_metal/distributed/host_transport/socket_relay.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/host_bridge_links.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include "impl/context/metal_env_impl.hpp"
#include "distributed/mesh_device_impl.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <map>
#include <optional>
#include <stdexcept>
#include <thread>
#include <tuple>

namespace tt::tt_metal::distributed {

namespace {

namespace hb = tt::tt_fabric::host_bridge;
using host_transport::HostTransport;
using host_transport::RelayEndpoint;
using host_transport::RelayLoop;
using host_transport::RelaySender;
using host_transport::RingGeometry;
using host_transport::TransportParams;

// Own tag range: HostMeshSocket reserves from 0x7248 upwards.
constexpr int kTagBase = 0x6A00;
// Per link: stream 0/1 are the lower rank's tx/rx sockets, 2/3 the higher rank's.
constexpr int kStreamsPerLink = 4;

struct PinnedRing {
    std::shared_ptr<std::byte> mem;
    size_t bytes = 0;
    std::shared_ptr<tt::tt_metal::experimental::PinnedMemory> pinned;
    uint64_t iova = 0;
};

PinnedRing make_pinned_ring(MeshDevice& mesh, const MeshCoordinate& device, ChipId device_id, size_t bytes) {
    PinnedRing ring;
    void* raw = nullptr;
    TT_FATAL(posix_memalign(&raw, 4096, bytes) == 0, "FabricHostBridge: cannot allocate a {} B receive ring", bytes);
    std::memset(raw, 0, bytes);
    ring.mem = std::shared_ptr<std::byte>(static_cast<std::byte*>(raw), [](std::byte* p) { std::free(p); });
    ring.bytes = bytes;
    HostBuffer view(ttsl::Span<std::byte>(ring.mem.get(), bytes), MemoryPin(ring.mem));
    ring.pinned = tt::tt_metal::experimental::PinnedMemory::Create(
        mesh, MeshCoordinateRangeSet(MeshCoordinateRange(device, device)), view, /*map_to_noc=*/true);
    ring.iova = ring.pinned->get_device_addr(device_id);
    return ring;
}

// Replays the peer router's frames on the local router's core: PACKET through the PCIe DMA engine,
// REG_WRITE as a register write. One frame per ring page, in order; a page is credited back as soon as
// its frame has been applied.
class BridgeInjector final : public RelayEndpoint {
public:
    BridgeInjector(
        HostTransport& transport,
        RingGeometry geometry,
        const std::byte* ring,
        uint64_t ring_iova,
        uint64_t l1_offset,
        tt::umd::TTDevice* tt_device,
        tt::umd::CoreCoord translated_core,
        const tt::Cluster& cluster,
        tt_cxy_pair target,
        std::string name) :
        transport_(transport),
        geom_(geometry),
        ring_(ring),
        ring_iova_(ring_iova),
        l1_offset_(l1_offset),
        tt_device_(tt_device),
        core_(translated_core),
        cluster_(cluster),
        target_(target),
        name_(std::move(name)) {}

    bool poll() override {
        PollMark mark{polls_};
        transport_.poll();
        const uint64_t delivered = transport_.pages_delivered();
        bool progress = false;
        uint64_t applied = applied_.load(std::memory_order_relaxed);
        while (applied < delivered) {
            apply(applied);
            applied++;
            progress = true;
        }
        if (progress) {
            applied_.store(applied, std::memory_order_release);
            transport_.set_consumed(applied);
        }
        if (applied != credited_ && transport_.post_credit(applied)) {
            credited_ = applied;
            progress = true;
        }
        return progress;
    }

    uint64_t watermark() const override { return applied_.load(std::memory_order_acquire); }
    bool drained(uint64_t) const override { return true; }

    std::string describe() const override {
        return fmt::format(
            "{}: applied={} packets={} reg_writes={} | {}",
            name_,
            applied_.load(std::memory_order_acquire),
            packets_.load(std::memory_order_relaxed),
            reg_writes_.load(std::memory_order_relaxed),
            transport_.describe());
    }

private:
    void apply(uint64_t page) {
        const size_t offset = static_cast<size_t>(page % geom_.num_pages) * geom_.page_size;
        const auto* header = reinterpret_cast<const hb::FrameHeader*>(ring_ + offset);
        const uint32_t expected = static_cast<uint32_t>(page + 1);
        TT_FATAL(
            header->seq == expected,
            "{}: frame {} carries seq {} (kind {}), expected {}",
            name_,
            page,
            header->seq,
            header->kind,
            expected);
        switch (header->kind) {
            case hb::PACKET: {
                TT_FATAL(
                    header->value != 0 && header->value <= geom_.page_size - hb::kFrameHeaderBytes,
                    "{}: frame {} carries a {} B packet, page holds {}",
                    name_,
                    page,
                    header->value,
                    geom_.page_size - hb::kFrameHeaderBytes);
                const uint64_t dst = header->addr + l1_offset_;
                if (tt_device_ != nullptr) {
                    tt_device_->dma_write_zero_copy(
                        ring_iova_ + offset + hb::kFrameHeaderBytes, dst, header->value, core_);
                } else {
                    cluster_.write_core(ring_ + offset + hb::kFrameHeaderBytes, header->value, target_, dst);
                }
                packets_.fetch_add(1, std::memory_order_relaxed);
                break;
            }
            case hb::REG_WRITE: {
                const uint32_t value = header->value;
                cluster_.write_reg(&value, target_, header->addr);
                reg_writes_.fetch_add(1, std::memory_order_relaxed);
                break;
            }
            default: TT_THROW("{}: frame {} has unknown kind {}", name_, page, header->kind);
        }
    }

    HostTransport& transport_;
    RingGeometry geom_;
    const std::byte* ring_;
    uint64_t ring_iova_;
    uint64_t l1_offset_;
    tt::umd::TTDevice* tt_device_;
    tt::umd::CoreCoord core_;
    const tt::Cluster& cluster_;
    tt_cxy_pair target_;
    std::string name_;
    std::atomic<uint64_t> applied_{0};
    uint64_t credited_ = 0;
    std::atomic<uint64_t> packets_{0};
    std::atomic<uint64_t> reg_writes_{0};
};

}  // namespace

struct FabricHostBridge::Impl {
    struct Stream {
        std::unique_ptr<D2HSocket> d2h;  // outbound: the local router's ring
        PinnedRing ring;                 // inbound: the peer router's frames land here
        std::unique_ptr<HostTransport> transport;
        std::shared_ptr<RelayEndpoint> endpoint;
    };
    // One bridged router on this host. A link inside this host contributes two ends.
    struct End {
        tt::tt_fabric::FabricNodeId node{tt::tt_fabric::MeshId{0}, 0};
        tt::tt_fabric::chan_id_t eth_chan = 0;
        MeshCoordinate coord{0, 0};
        tt::tt_fabric::FabricNodeId peer_node{tt::tt_fabric::MeshId{0}, 0};
        tt::tt_fabric::chan_id_t peer_chan = 0;
        multihost::Rank peer_rank{0};
        // Both ends of a link derive the same id, which selects its MPI tags (or the loopback pair).
        uint32_t link_id = 0;
        // The lower end owns streams 0 and 1 (its tx and rx rings), the upper end streams 2 and 3.
        bool lower = false;
        ChipId device_id = 0;
        CoreCoord eth_noc0;
        tt_cxy_pair target{0, 0, 0};
        uint64_t status_addr = 0;
        std::array<Stream, kStreamsPerLink> streams;
    };

    std::shared_ptr<MeshDevice> mesh;
    std::vector<End> ends;
    std::vector<std::shared_ptr<RelayEndpoint>> endpoints;
};

namespace {

using LinkKey = std::tuple<uint32_t, uint32_t, uint32_t>;  // mesh, chip, chan

LinkKey link_key(tt::tt_fabric::FabricNodeId node, tt::tt_fabric::chan_id_t chan) {
    return {*node.mesh_id, static_cast<uint32_t>(node.chip_id), static_cast<uint32_t>(chan)};
}

// The MPI rank whose mesh device holds `node`.
multihost::Rank rank_of(
    const tt::tt_fabric::ControlPlane& control_plane, tt::tt_fabric::FabricNodeId node, multihost::Rank my_rank) {
    const auto host_rank = control_plane.get_mesh_graph().get_host_rank_for_chip(node.mesh_id, node.chip_id);
    const auto local_meshes = control_plane.get_local_mesh_id_bindings();
    if (std::find(local_meshes.begin(), local_meshes.end(), node.mesh_id) != local_meshes.end() &&
        host_rank == control_plane.get_local_host_rank_id_binding()) {
        return my_rank;
    }
    for (const auto& [rank, binding] : control_plane.get_global_logical_bindings()) {
        if (binding.first == node.mesh_id && host_rank.has_value() && binding.second == *host_rank) {
            return rank;
        }
    }
    TT_THROW("FabricHostBridge: no rank owns mesh {} chip {}", *node.mesh_id, node.chip_id);
}

}  // namespace

FabricHostBridge::FabricHostBridge(const std::shared_ptr<MeshDevice>& mesh_device, uint32_t ring_pages) :
    impl_(std::make_unique<Impl>()) {
    TT_FATAL(mesh_device != nullptr, "FabricHostBridge needs a mesh device");
    TT_FATAL(ring_pages >= 2, "FabricHostBridge: ring_pages must be at least 2");
    impl_->mesh = mesh_device;

    auto& env = mesh_device->impl().metal_env();
    auto& cluster = env.get_cluster();
    const auto& hal = env.get_hal();
    const auto& control_plane = env.get_control_plane();
    const auto& spec = env.get_rtoptions().get_fabric_host_bridge();
    const auto& router_cfg = control_plane.get_fabric_context().get_builder_context().get_fabric_router_config();
    TT_FATAL(
        !spec.empty() && router_cfg.host_bridge_block_address != 0,
        "FabricHostBridge: no router was built host-bridged; set TT_METAL_FABRIC_HOST_BRIDGE before opening the mesh");
    const uint32_t block = static_cast<uint32_t>(router_cfg.host_bridge_block_address);
    const uint32_t page = hb::frame_page_bytes(static_cast<uint32_t>(router_cfg.channel_buffer_size_bytes));
    const uint64_t l1_offset = hal.get_l1_noc_offset(HalProgrammableCoreType::ACTIVE_ETH);

    const auto context = multihost::DistributedContext::get_current_world();
    const multihost::Rank my_rank = context->rank();

    // Every bridged router on this host, with its peer across the cable.
    std::vector<Impl::End> ends;
    for (const auto& coord : MeshCoordinateRange(mesh_device->shape())) {
        IDevice* device = mesh_device->get_device(coord);
        const auto node = control_plane.get_fabric_node_id_from_physical_chip_id(device->id());
        for (const auto& [chan, direction] : control_plane.get_active_fabric_eth_channels(node)) {
            if (!tt::tt_fabric::host_bridge::link_is_bridged(spec, control_plane, node, chan)) {
                continue;
            }
            const auto peer = control_plane.try_get_connected_mesh_chip_chan_ids(node, chan);
            TT_FATAL(
                peer.has_value(),
                "FabricHostBridge: bridged router chip {} eth_chan {} has no peer across the cable",
                device->id(),
                chan);
            Impl::End end;
            end.node = node;
            end.eth_chan = chan;
            end.coord = coord;
            end.peer_node = peer->first;
            end.peer_chan = peer->second;
            end.peer_rank = rank_of(control_plane, peer->first, my_rank);
            end.lower = link_key(node, chan) < link_key(peer->first, peer->second);
            end.device_id = device->id();
            ends.push_back(std::move(end));
        }
    }

    // Link ids: per peer rank, the position of the link's canonical key in sorted order. The peer rank
    // enumerates the same links with the same keys, so it derives the same ids; the two ends of a link
    // inside this host share one id and pair up through the loopback transport. Both ranks must also
    // build the streams in the same order: each MPI transport exchanges a warm-up message when it is
    // constructed.
    std::sort(ends.begin(), ends.end(), [](const Impl::End& a, const Impl::End& b) {
        const LinkKey ka = std::min(link_key(a.node, a.eth_chan), link_key(a.peer_node, a.peer_chan));
        const LinkKey kb = std::min(link_key(b.node, b.eth_chan), link_key(b.peer_node, b.peer_chan));
        return std::tie(a.peer_rank, ka, a.lower) < std::tie(b.peer_rank, kb, b.lower);
    });
    {
        std::optional<multihost::Rank> group_rank;
        std::optional<LinkKey> last_key;
        uint32_t next_id = 0;
        for (auto& end : ends) {
            const LinkKey key = std::min(link_key(end.node, end.eth_chan), link_key(end.peer_node, end.peer_chan));
            if (group_rank != end.peer_rank) {
                group_rank = end.peer_rank;
                last_key.reset();
                next_id = 0;
            }
            if (last_key.has_value() && *last_key == key) {
                end.link_id = next_id - 1;  // the other end of a link inside this host
            } else {
                end.link_id = next_id++;
            }
            last_key = key;
        }
    }

    for (auto& end : ends) {
        const auto& soc = cluster.get_soc_desc(end.device_id);
        const auto eth_noc0 = soc.get_eth_core_for_channel(end.eth_chan, CoordSystem::NOC0);
        end.eth_noc0 = CoreCoord{eth_noc0.x, eth_noc0.y};
        const CoreCoord eth_virtual =
            cluster.get_virtual_coordinate_from_physical_coordinates(end.device_id, end.eth_noc0);
        end.target = tt_cxy_pair(end.device_id, eth_virtual);
        const tt::umd::CoreCoord translated = soc.get_coord_at(eth_virtual, CoordSystem::TRANSLATED);
        tt::umd::TTDevice* tt_device =
            cluster.is_mock_or_emulated() ? nullptr : cluster.get_driver()->get_tt_device(end.device_id);
        end.status_addr = block + l1_offset;

        const int tag_base =
            kTagBase + static_cast<int>(end.link_id) * kStreamsPerLink * host_transport::kTagsPerConnection;
        for (int k = 0; k < kStreamsPerLink; k++) {
            auto& stream = end.streams[k];
            const bool mine = end.lower ? (k < 2) : (k >= 2);
            const uint32_t socket_offset = (k % 2 == 0) ? hb::kTxSocketOffset : hb::kRxSocketOffset;
            const std::string name = fmt::format(
                "bridge link {} chip {} eth {} {}{}",
                end.link_id,
                end.device_id,
                end.eth_chan,
                mine ? "out" : "in",
                (k % 2 == 0) ? "/tx" : "/rx");

            TransportParams tp;
            tp.geometry = RingGeometry{.page_size = page, .num_pages = ring_pages};
            tp.is_sender = mine;
            tp.peer_rank = *end.peer_rank;
            tp.context = context;
            tp.tag_base = tag_base + k * host_transport::kTagsPerConnection;
            tp.max_batch_pages = 0;

            if (mine) {
                stream.d2h = std::make_unique<D2HSocket>(
                    mesh_device,
                    MeshCoreCoord(end.coord, end.eth_noc0),
                    page * ring_pages,
                    D2HSocket::ExternalConfigBuffer{
                        .address = block + socket_offset,
                        .sender_core_type = HalProgrammableCoreType::ACTIVE_ETH,
                    },
                    D2HSocket::ProcessScope::InProcess);
                stream.d2h->set_page_size(page);
                tp.ring = stream.d2h->host_fifo().data();
                stream.transport = host_transport::make_host_transport(tp);
                stream.endpoint = std::make_shared<RelaySender>(*stream.transport, *stream.d2h, tp.geometry, 0);
            } else {
                stream.ring = make_pinned_ring(*mesh_device, end.coord, end.device_id, tp.geometry.fifo_bytes());
                tp.ring = stream.ring.mem.get();
                stream.transport = host_transport::make_host_transport(tp);
                stream.endpoint = std::make_shared<BridgeInjector>(
                    *stream.transport,
                    tp.geometry,
                    stream.ring.mem.get(),
                    stream.ring.iova,
                    l1_offset,
                    tt_device,
                    translated,
                    cluster,
                    end.target,
                    name);
            }
            impl_->endpoints.push_back(stream.endpoint);
        }

        // Both socket configs are in L1: let the router open them. The builder zeroed the block before
        // the router launched, so nothing stale can be read as armed.
        const uint32_t armed = hb::kArmed;
        cluster.write_core(&armed, sizeof(armed), end.target, end.status_addr + offsetof(hb::Status, armed));
        log_info(
            tt::LogFabric,
            "FabricHostBridge: link {} mesh {} chip {} (device {}) eth_chan {} <-> mesh {} chip {} eth_chan {} on rank "
            "{}{} armed (page {} B, ring {} pages, block 0x{:x})",
            end.link_id,
            *end.node.mesh_id,
            end.node.chip_id,
            end.device_id,
            end.eth_chan,
            *end.peer_node.mesh_id,
            end.peer_node.chip_id,
            end.peer_chan,
            *end.peer_rank,
            end.peer_rank == my_rank ? " (this host)" : "",
            page,
            ring_pages,
            block);
        impl_->ends.push_back(std::move(end));
    }

    for (auto& endpoint : impl_->endpoints) {
        RelayLoop::instance().add(endpoint);
    }
}

FabricHostBridge::~FabricHostBridge() {
    if (!impl_) {
        return;
    }
    for (auto& endpoint : impl_->endpoints) {
        RelayLoop::instance().remove(endpoint);
    }
    impl_->endpoints.clear();
    // Transports before the rings and sockets they point into.
    for (auto& end : impl_->ends) {
        for (auto& stream : end.streams) {
            stream.endpoint.reset();
            stream.transport.reset();
        }
    }
    impl_->ends.clear();
}

size_t FabricHostBridge::num_ends() const { return impl_->ends.size(); }

std::string FabricHostBridge::first_error() const { return RelayLoop::instance().first_error(); }

std::string FabricHostBridge::describe() const {
    std::string out;
    auto& cluster = impl_->mesh->impl().metal_env().get_cluster();
    for (const auto& end : impl_->ends) {
        hb::Status status{};
        cluster.read_core(&status, sizeof(status), end.target, end.status_addr);
        out += fmt::format(
            "link {} chip {} eth_chan {}: router armed=0x{:x} packets={} reg_writes={} tx(opened={} "
            "unarmed={} ring_full={}) rx(opened={} unarmed={} ring_full={})\n",
            end.link_id,
            end.device_id,
            end.eth_chan,
            status.armed,
            status.packets,
            status.reg_writes,
            status.tx_opened,
            status.tx_unarmed,
            status.tx_ring_full,
            status.rx_opened,
            status.rx_unarmed,
            status.rx_ring_full);
        for (const uint32_t off : {hb::kTxSocketOffset, hb::kRxSocketOffset}) {
            uint32_t w[32] = {0};
            cluster.read_core(w, sizeof(w), end.target, end.status_addr + off);
            // sender_socket_md at word 0, bytes_acked at 8, frame header scratch at 16
            out += fmt::format(
                "  link {} {} socket: bytes_sent={} write_ptr={} acked={} | last frame kind={} addr=0x{:x} "
                "value=0x{:x} seq={}\n",
                end.link_id,
                off == hb::kTxSocketOffset ? "tx" : "rx",
                w[0],
                w[2],
                w[8],
                w[16],
                w[17],
                w[18],
                w[19]);
        }
    }
    for (const auto& endpoint : impl_->endpoints) {
        out += "  ";
        out += RelayLoop::instance().describe(*endpoint);
        out += '\n';
    }
    return out;
}

}  // namespace tt::tt_metal::distributed
