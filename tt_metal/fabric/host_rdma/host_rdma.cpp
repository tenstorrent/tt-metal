// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/host_rdma/host_rdma.hpp"

#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <map>
#include <mutex>
#include <type_traits>

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-umd/driver_atomics.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <hostdevcommon/fabric_common.h>

#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include "tt_metal/distributed/core_writer.hpp"
#include "tt_metal/distributed/noc_pinned_buffer.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/packet_header_types.hpp"
#include "tt_metal/fabric/builder/fabric_stream_assignment.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/host_rdma_service_abi.hpp"

namespace tt::tt_fabric::host_rdma {

namespace {

// The free-slots register takes a signed slot increment at this bit of its increment-on-write register
// (REMOTE_DEST_BUF_WORDS_FREE_INC in the architecture's noc_overlay_parameters.h; the device side is
// pack_value_for_inc_on_write_stream_reg_write).
constexpr uint32_t kFreeSlotsIncShift = 6;

// Bulk writes go through DMA only in the shapes UMD takes them (Cluster::dma_transfer_shape_supported).
constexpr uint32_t kMinDmaBytes = 32;

// Each word of a pinned host memory target sits alone in a 16 B slot: a NoC write into host memory must keep the
// PCIe write alignment (NOC_PCIE_WRITE_ALIGNMENT_BYTES), and its source word (the router's read counter) is 16 B
// aligned.
constexpr size_t kHostWordStrideBytes = 16;
constexpr size_t kHostWordStride = kHostWordStrideBytes / sizeof(uint32_t);

// How long the host waits for the router's credits before declaring the channel stuck.
constexpr std::chrono::seconds kCreditTimeout{10};

}  // namespace

// ---------------------------------------------------------------------------------------------------------------------
// UmdDmaDeviceWriter
// ---------------------------------------------------------------------------------------------------------------------

UmdDmaDeviceWriter::UmdDmaDeviceWriter(const tt::Cluster& cluster) : cluster_(cluster) {}

void UmdDmaDeviceWriter::write(
    ChipId chip, const tt_metal::CoreCoord& virtual_core, uint64_t addr, std::span<const std::byte> data) {
    const bool dma_shape = data.size() >= kMinDmaBytes && data.size() % 4 == 0 && addr % 4 == 0;
    if (dma_shape && !tt::tt_metal::MetalContext::instance().rtoptions().get_disable_dma_ops()) {
        const auto& soc_desc = cluster_.get_soc_desc(chip);
        const tt::umd::CoreCoord core = soc_desc.get_coord_at(virtual_core, CoordSystem::TRANSLATED);
        cluster_.get_driver()->dma_write_to_device(data.data(), data.size(), chip, core, addr);
    } else {
        cluster_.write_core(data.data(), data.size(), tt_cxy_pair(chip, virtual_core), addr);
    }
}

void UmdDmaDeviceWriter::fence(ChipId chip, const tt_metal::CoreCoord& virtual_core) {
    cluster_.l1_barrier(chip, {virtual_core});
}

void UmdDmaDeviceWriter::write_word(
    ChipId chip, const tt_metal::CoreCoord& virtual_core, uint64_t addr, uint32_t value) {
    cluster_.write_reg(&value, tt_cxy_pair(chip, virtual_core), addr);
}

uint32_t UmdDmaDeviceWriter::read_word(ChipId chip, const tt_metal::CoreCoord& virtual_core, uint64_t addr) {
    uint32_t value = 0;
    cluster_.read_reg(&value, tt_cxy_pair(chip, virtual_core), addr);
    return value;
}

// ---------------------------------------------------------------------------------------------------------------------
// PinnedHostMemoryTarget
// ---------------------------------------------------------------------------------------------------------------------

PinnedHostMemoryTarget::PinnedHostMemoryTarget(tt_metal::distributed::MeshDevice& mesh_device, size_t num_words) :
    num_words_(num_words) {
    TT_FATAL(num_words > 0, "A host memory target needs at least one word");
    TT_FATAL(mesh_device.get_devices().size() == 1, "A host memory target is mapped for a unit mesh (one chip)");
    chip_ = mesh_device.get_devices().front()->id();
    constexpr size_t page_bytes = 4096;
    const size_t alloc_bytes = ((num_words * kHostWordStrideBytes) + page_bytes - 1) / page_bytes * page_bytes;
    void* p = mmap(nullptr, alloc_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    TT_FATAL(p != MAP_FAILED, "Failed to allocate {} B of host memory for a host memory target", alloc_bytes);
    std::memset(p, 0, alloc_bytes);
    words_ =
        std::shared_ptr<uint32_t[]>(static_cast<uint32_t*>(p), [alloc_bytes](uint32_t* w) { munmap(w, alloc_bytes); });

    auto pinned = tt_metal::distributed::pin_host_memory_for_noc(
        mesh_device,
        tt_metal::distributed::MeshCoordinateRangeSet(tt_metal::distributed::MeshCoordinateRange(mesh_device.shape())),
        words_,
        alloc_bytes / sizeof(uint32_t),
        chip_,
        "host_rdma host memory target");
    pinned_ = std::move(pinned.pinned_memory);
    base_ = DeviceVisibleAddress{.noc_xy = pinned.pcie_xy_enc, .addr = pinned.addr};
}

DeviceVisibleAddress PinnedHostMemoryTarget::device_address(ChipId chip, size_t index) const {
    TT_FATAL(index < num_words_, "Word {} is past the target's {} words", index, num_words_);
    TT_FATAL(chip == chip_, "The host memory target is mapped for chip {}, not chip {}", chip_, chip);
    return DeviceVisibleAddress{.noc_xy = base_.noc_xy, .addr = base_.addr + index * kHostWordStrideBytes};
}

uint32_t PinnedHostMemoryTarget::read(size_t index) const {
    TT_FATAL(index < num_words_, "Word {} is past the target's {} words", index, num_words_);
    return std::atomic_ref<uint32_t>(words_[index * kHostWordStride]).load(std::memory_order_acquire);
}

void PinnedHostMemoryTarget::write(size_t index, uint32_t value) {
    TT_FATAL(index < num_words_, "Word {} is past the target's {} words", index, num_words_);
    std::atomic_ref<uint32_t>(words_[index * kHostWordStride]).store(value, std::memory_order_release);
}

// ---------------------------------------------------------------------------------------------------------------------
// Transports
// ---------------------------------------------------------------------------------------------------------------------

namespace {

// The host pushes each run into the router's slots through a DeviceWriter.
class HostPushTransport final : public PacketTransport {
public:
    HostPushTransport(std::shared_ptr<DeviceWriter> writer, const RouterSlots& slots) :
        writer_(std::move(writer)), slots_(slots), staging_(static_cast<size_t>(slots.num_slots) * slots.slot_bytes) {}

    std::byte* slot(uint32_t slot) override { return staging_.data() + static_cast<size_t>(slot) * slots_.slot_bytes; }

    void submit(uint32_t first, std::span<const uint32_t> packet_bytes) override {
        const uint32_t count = static_cast<uint32_t>(packet_bytes.size());
        const size_t last_bytes = packet_bytes.back();
        // One write for the run's slots (through the last packet's end), one fence so every packet is whole in its slot
        // before the router sees any, and one doorbell taking every slot of the run.
        const size_t bytes = static_cast<size_t>(count - 1) * slots_.slot_bytes + last_bytes;
        const size_t offset = static_cast<size_t>(first) * slots_.slot_bytes;
        writer_->write(
            slots_.chip,
            slots_.router_core,
            slots_.slots_base + offset,
            std::span<const std::byte>(staging_.data() + offset, bytes));
        writer_->fence(slots_.chip, slots_.router_core);
        writer_->write_word(
            slots_.chip,
            slots_.router_core,
            slots_.free_slots_addr,
            static_cast<uint32_t>(-static_cast<int32_t>(count)) << kFreeSlotsIncShift);
    }

private:
    std::shared_ptr<DeviceWriter> writer_;
    RouterSlots slots_;
    std::vector<std::byte> staging_;
};

}  // namespace

TransportFactory host_push_transport(std::shared_ptr<DeviceWriter> writer) {
    return [writer = std::move(writer)](const RouterSlots& slots) -> std::unique_ptr<PacketTransport> {
        return std::make_unique<HostPushTransport>(writer, slots);
    };
}

// ---------------------------------------------------------------------------------------------------------------------
// Router channels
// ---------------------------------------------------------------------------------------------------------------------

RouterChannel find_router_channel(FabricNodeId src, chan_id_t eth_chan) {
    auto& context = tt::tt_metal::MetalContext::instance();
    const auto& hal = context.hal();
    TT_FATAL(
        hal.get_arch() == tt::ARCH::BLACKHOLE,
        "Host-initiated RDMA is supported on Blackhole only (it encodes NoC addresses with coordinate virtualization)");
    const auto& control_plane = context.get_control_plane();
    const auto& builder_context = control_plane.get_fabric_context().get_builder_context();
    TT_FATAL(
        builder_context.requires_host_rdma(),
        "The fabric was built without host_rdma channels: set TT_METAL_ENABLE_FABRIC_HOST_RDMA=1 before fabric init");
    const auto& cluster = context.get_cluster();

    RouterChannel rc;
    rc.src = src;
    rc.eth_chan = eth_chan;
    rc.slots.chip = control_plane.get_physical_chip_id_from_fabric_node_id(src);
    rc.slots.router_core = cluster.get_virtual_eth_core_from_channel(rc.slots.chip, eth_chan);

    // The channel: the router's last VC0 sender, laid out by the router itself.
    const auto& layout = builder_context.get_router_channel_layout(src, eth_chan);
    TT_FATAL(!layout.senders[0].empty(), "Router {} channel {} has no VC0 senders", src, eth_chan);
    rc.channel = static_cast<uint32_t>(layout.senders[0].size() - 1);
    rc.slots.slots_base = layout.senders[0].back().base_address;
    rc.slots.num_slots = layout.senders[0].back().num_slots;
    const auto& edm_config = builder_context.get_fabric_router_config();
    rc.slots.slot_bytes = static_cast<uint32_t>(edm_config.channel_buffer_size_bytes);
    rc.connection_info_addr =
        static_cast<uint32_t>(edm_config.sender_channels_worker_conn_info_base_address[rc.channel]);
    const uint32_t free_slots_stream =
        builder_context.get_stream_assignment(src.mesh_id).id(StreamRole::SENDER_FREE_SLOTS, 0, rc.channel);
    rc.slots.free_slots_addr = hal.get_noc_overlay_start_addr() +
                               free_slots_stream * hal.get_noc_stream_reg_space_size() +
                               hal.get_noc_stream_remote_dest_buf_space_available_update_reg_index() * sizeof(uint32_t);
    TT_FATAL(
        rc.slots.num_slots > 0 && rc.slots.slot_bytes > 0,
        "Router {} channel {} has an empty host_rdma channel",
        src,
        eth_chan);
    TT_FATAL(
        rc.connection_info_addr != 0,
        "Router {} channel {}: the router config set aside no connection info for sender channel {}",
        src,
        eth_chan,
        rc.channel);

    // Take up where the router is: it has freed `edm_read_counter` slots of this channel since it started, and (with
    // the channel idle) its next slot is that count's.
    cluster.read_reg(
        &rc.slots.start,
        tt_cxy_pair(rc.slots.chip, rc.slots.router_core),
        rc.connection_info_addr + offsetof(HostRdmaChannelInfo, edm_read_counter));
    return rc;
}

void set_router_credit_target(const RouterChannel& rc, const DeviceVisibleAddress& target) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto core = tt_cxy_pair(rc.slots.chip, rc.slots.router_core);
    const auto write = [&](size_t offset, uint32_t value) {
        cluster.write_reg(&value, core, rc.connection_info_addr + offset);
    };
    // The target first, then (ordered after it) the enable.
    write(offsetof(HostRdmaChannelInfo, credit_addr_lo), static_cast<uint32_t>(target.addr & 0xFFFFFFFFull));
    write(offsetof(HostRdmaChannelInfo, credit_addr_hi), static_cast<uint32_t>(target.addr >> 32));
    write(offsetof(HostRdmaChannelInfo, credit_noc_xy), target.noc_xy);
    cluster.l1_barrier(rc.slots.chip, {rc.slots.router_core});
    write(offsetof(HostRdmaChannelInfo, credit_enabled), 1);
    cluster.l1_barrier(rc.slots.chip, {rc.slots.router_core});
}

void clear_router_credit_target(const RouterChannel& rc) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const uint32_t disabled = 0;
    cluster.write_reg(
        &disabled,
        tt_cxy_pair(rc.slots.chip, rc.slots.router_core),
        rc.connection_info_addr + offsetof(HostRdmaChannelInfo, credit_enabled));
    cluster.l1_barrier(rc.slots.chip, {rc.slots.router_core});
}

uint32_t hops_between(FabricNodeId src, chan_id_t eth_chan, FabricNodeId dst) {
    const auto route = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_route(src, dst, eth_chan);
    // The route lists each router it passes (two per chip crossed); count the chip changes.
    uint32_t hops = 0;
    FabricNodeId previous = src;
    for (const auto& [node, chan] : route) {
        if (node != previous) {
            hops++;
            previous = node;
        }
    }
    TT_FATAL(hops > 0, "No route from {} eth chan {} to {}", src, eth_chan, dst);
    return hops;
}

// ---------------------------------------------------------------------------------------------------------------------
// HostRdmaChannel
// ---------------------------------------------------------------------------------------------------------------------

chan_id_t HostRdmaChannel::eth_chan_toward(FabricNodeId src, FabricNodeId dst, uint32_t link_idx) {
    TT_FATAL(src != dst, "A host_rdma write goes to another chip (src and dst are both {})", src);
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const std::optional<RoutingDirection> direction = control_plane.get_forwarding_direction(src, dst);
    TT_FATAL(direction.has_value(), "No forwarding direction from {} to {}", src, dst);
    const auto eth_chans = control_plane.get_active_fabric_eth_channels_in_direction(src, *direction);
    TT_FATAL(
        link_idx < eth_chans.size(),
        "Link {} is out of range: {} has {} links toward {}",
        link_idx,
        src,
        eth_chans.size(),
        dst);
    return eth_chans[link_idx];
}

std::unique_ptr<HostRdmaChannel> HostRdmaChannel::open_toward(
    FabricNodeId src,
    FabricNodeId dst,
    uint32_t link_idx,
    std::shared_ptr<DeviceWriter> writer,
    std::shared_ptr<HostMemoryTarget> credits,
    size_t credit_word) {
    return std::make_unique<HostRdmaChannel>(
        src, eth_chan_toward(src, dst, link_idx), std::move(writer), std::move(credits), credit_word);
}

HostRdmaChannel::HostRdmaChannel(
    FabricNodeId src,
    chan_id_t eth_chan,
    std::shared_ptr<DeviceWriter> writer,
    std::shared_ptr<HostMemoryTarget> credits,
    size_t credit_word,
    const TransportFactory& transport) :
    fabric_context_(tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context()),
    control_plane_(tt::tt_metal::MetalContext::instance().get_control_plane()),
    hal_(tt::tt_metal::MetalContext::instance().hal()),
    src_(src),
    eth_chan_(eth_chan),
    writer_(std::move(writer)),
    credits_(std::move(credits)),
    credit_word_(credit_word) {
    TT_FATAL(
        writer_ != nullptr && credits_ != nullptr, "A host_rdma channel needs a device writer and a credit target");
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    router_ = find_router_channel(src_, eth_chan_);
    chip_ = router_.slots.chip;
    router_core_ = router_.slots.router_core;
    slots_base_ = router_.slots.slots_base;
    num_slots_ = router_.slots.num_slots;
    slot_bytes_ = router_.slots.slot_bytes;
    connection_info_addr_ = router_.connection_info_addr;
    doorbell_addr_ = router_.slots.free_slots_addr;

    header_bytes_ = fabric_context_.get_fabric_packet_header_size_bytes();
    max_payload_bytes_ =
        std::min<size_t>(fabric_context_.get_fabric_max_payload_size_bytes(), slot_bytes_ - header_bytes_);
    is_2d_ = fabric_context_.is_2D_routing_enabled();
    header_scratch_.resize(header_bytes_);
    max_batch_ = std::max<uint32_t>(1, num_slots_ / 2);

    if (is_2d_) {
        // The source chip's route table, as its routers read it: the same rows a worker on that chip widens.
        const auto table_addr = hal_.get_dev_addr(
            tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::ROUTING_TABLE);
        routing_table_.resize(sizeof(tt::tt_fabric::routing_l1_info_t));
        cluster.read_core(routing_table_.data(), routing_table_.size(), tt_cxy_pair(chip_, router_core_), table_addr);
    }

    const uint32_t start = router_.slots.start;
    sent_ = start;
    known_credits_ = start;
    next_slot_ = start % num_slots_;
    credits_->write(credit_word_, start);

    transport_ = transport ? transport(router_.slots) : host_push_transport(writer_)(router_.slots);

    // The router credits this channel in host memory.
    set_router_credit_target(router_, credits_->device_address(chip_, credit_word_));

    log_debug(
        tt::LogFabric,
        "host_rdma: opened {} eth chan {} (core {}): channel {} at 0x{:x}, {} slots of {} B, doorbell 0x{:x}, "
        "starting at {} credits",
        src_,
        eth_chan_,
        router_core_.str(),
        router_.channel,
        slots_base_,
        num_slots_,
        slot_bytes_,
        doorbell_addr_,
        start);
}

HostRdmaChannel::~HostRdmaChannel() {
    try {
        drain();
        clear_router_credit_target(router_);
    } catch (const std::exception& e) {
        log_warning(tt::LogFabric, "host_rdma: closing {} eth chan {} failed: {}", src_, eth_chan_, e.what());
    }
}

uint32_t HostRdmaChannel::packets_credited() const { return credits_->read(credit_word_); }

uint64_t HostRdmaChannel::noc_addr(const tt_metal::CoreCoord& virtual_core, uint64_t addr) const {
    // With coordinate virtualization, a router writes to a virtual core at NOC_XY_ADDR(x, y, addr) on either NoC.
    return (static_cast<uint64_t>(hal_.noc_xy_encoding(virtual_core.x, virtual_core.y))
            << hal_.get_noc_addr_local_bits()) |
           addr;
}

uint32_t HostRdmaChannel::hops_to(FabricNodeId dst) const { return hops_between(src_, eth_chan_, dst); }

void HostRdmaChannel::build_header(
    FabricNodeId dst,
    uint64_t write_noc_addr,
    uint32_t payload_bytes,
    const std::optional<std::pair<uint64_t, uint32_t>>& completion_noc_addr_and_inc) {
    PacketHeaderConfig config{.is_2d = is_2d_};
    if (is_2d_) {
        config.route_buffer_size = fabric_context_.get_2d_pkt_hdr_route_buffer_size();
    } else {
        config.extension_words = fabric_context_.get_1d_pkt_hdr_extension_words();
    }
    std::fill(header_scratch_.begin(), header_scratch_.end(), std::byte{0});
    visit_packet_header_type(config, [&]<typename Header>(std::type_identity<Header>) {
        TT_FATAL(
            sizeof(Header) == header_bytes_,
            "Header type of {} B, but the fabric's header is {} B",
            sizeof(Header),
            header_bytes_);
        // The header types are built in place on device; here, in zeroed storage of the header's size and alignment.
        alignas(Header) std::array<std::byte, sizeof(Header)> storage{};
        auto& header = *reinterpret_cast<Header*>(storage.data());

        if constexpr (std::is_same_v<decltype(header.routing_fields), LowLatencyMeshRoutingFields>) {
            // 2D: the destination's [Y | X] action maps from the source chip's route table (widen_2d_route_to_chip).
            TT_FATAL(dst.mesh_id == src_.mesh_id, "host_rdma writes stay in the source's mesh ({} -> {})", src_, dst);
            const auto* table = reinterpret_cast<const routing_l1_info_t*>(routing_table_.data());
            const uint32_t y_size = table->mesh_y_size;
            const uint32_t x_size = table->mesh_x_size;
            TT_FATAL(y_size + x_size <= sizeof(header.route_buffer), "The mesh's action map does not fit the header");
            const uint32_t dst_id = dst.chip_id;
            TT_FATAL(dst_id < y_size * x_size, "Chip {} is outside the {}x{} mesh", dst_id, y_size, x_size);
            const uint32_t dst_y = dst_id / x_size;
            const uint32_t dst_x = dst_id % x_size;
            const uint8_t* vectors = table->route_table_2d.action_vectors;
            const uint8_t* y_row = Routing2DCodec::y_row(vectors, y_size, dst_y);
            const uint8_t* x_row = Routing2DCodec::x_row(vectors, y_size, x_size, dst_x);
            for (uint32_t y = 0; y < y_size; ++y) {
                header.route_buffer[y] =
                    Routing2DCodec::widen(Fabric2DAxis::Y, Routing2DCodec::get_action_2bit(y_row, y));
            }
            for (uint32_t x = 0; x < x_size; ++x) {
                header.route_buffer[y_size + x] =
                    Routing2DCodec::widen(Fabric2DAxis::X, Routing2DCodec::get_action_2bit(x_row, x));
            }
            header.route_buffer[y_size + dst_x] |= Routing2DCodec::ACTION_LOCAL_DELIVER;
            header.dst_start_node_id = (static_cast<uint32_t>(*dst.mesh_id) << 16) | dst_id;
            header.mcast_params_64 = 0;
            header.routing_fields.value = 0;

            // The router this channel belongs to must be the route's first hop.
            const uint32_t src_y = table->my_mesh_coord_y;
            const uint32_t src_x = table->my_mesh_coord_x;
            const uint8_t first =
                header.route_buffer[src_y] != 0 ? header.route_buffer[src_y] : header.route_buffer[y_size + src_x];
            const auto my_direction = control_plane_.get_eth_chan_direction(src_, eth_chan_);
            TT_FATAL(
                (first & Routing2DCodec::ACTION_ETH_MASK) == Routing2DCodec::action_bit(my_direction),
                "The route from {} to {} leaves by action 0x{:x}, not through this channel's router (direction {})",
                src_,
                dst,
                first,
                static_cast<int>(my_direction));
        } else {
            header.to_chip_unicast(static_cast<uint8_t>(hops_to(dst)));
        }

        if (completion_noc_addr_and_inc.has_value()) {
            header.noc_send_type = NOC_FUSED_UNICAST_ATOMIC_INC;
            header.command_fields.unicast_seminc_fused = NocUnicastAtomicIncFusedCommandHeader(
                write_noc_addr,
                completion_noc_addr_and_inc->first,
                completion_noc_addr_and_inc->second,
                /*flush=*/true);
        } else {
            header.noc_send_type = NOC_UNICAST_WRITE;
            header.command_fields.unicast_write = NocUnicastCommandHeader{write_noc_addr};
        }
        header.payload_size_bytes = static_cast<uint16_t>(payload_bytes);
        std::memcpy(header_scratch_.data(), storage.data(), sizeof(Header));
        return 0;
    });
}

void HostRdmaChannel::wait_for_credits(uint32_t in_flight_limit) {
    // The router credits the absolute count of slots freed (wrapping), so `sent_ - credits` is what is in flight; the
    // staged packets' slots are not free either (they were free when staged, and the router has not seen them yet).
    if (sent_ - known_credits_ <= in_flight_limit) {
        return;
    }
    const auto deadline = std::chrono::steady_clock::now() + kCreditTimeout;
    while (true) {
        known_credits_ = credits_->read(credit_word_);
        if (sent_ - known_credits_ <= in_flight_limit) {
            return;
        }
        if (std::chrono::steady_clock::now() > deadline) {
            const auto info_word = [&](size_t offset) {
                return writer_->read_word(chip_, router_core_, connection_info_addr_ + offset);
            };
            const uint32_t free_slots_reg = writer_->read_word(
                chip_,
                router_core_,
                doorbell_addr_ - hal_.get_noc_stream_remote_dest_buf_space_available_update_reg_index() * 4 +
                    hal_.get_noc_stream_remote_dest_buf_space_available_reg_index() * 4);
            TT_THROW(
                "host_rdma: {} eth chan {} gave no credit for {} s: {} packets sent, {} credited in host memory; the "
                "router's read counter is {} and its free-slots register 0x{:x}; its credit target is {} "
                "(0x{:x}:0x{:08x}{:08x}). {}",
                src_,
                eth_chan_,
                std::chrono::duration_cast<std::chrono::seconds>(kCreditTimeout).count(),
                sent_,
                known_credits_,
                info_word(offsetof(HostRdmaChannelInfo, edm_read_counter)),
                free_slots_reg,
                info_word(offsetof(HostRdmaChannelInfo, credit_enabled)) != 0 ? "enabled" : "disabled",
                info_word(offsetof(HostRdmaChannelInfo, credit_noc_xy)),
                info_word(offsetof(HostRdmaChannelInfo, credit_addr_hi)),
                info_word(offsetof(HostRdmaChannelInfo, credit_addr_lo)),
                transport_ ? transport_->describe() : std::string{});
        }
    }
}

void HostRdmaChannel::stage_packet(size_t chunk_bytes, std::span<const std::byte> payload) {
    // A batch is contiguous slots: the next packet starts a new batch at the ring's end or at the batch limit.
    if (staged_ != 0 && (next_slot_ == 0 || staged_ == max_batch_)) {
        flush();
    }
    // Room for this packet: every packet rung in or staged, and this one, fit in the ring.
    wait_for_credits(num_slots_ - 1 - staged_);
    if (staged_ == 0) {
        staged_first_slot_ = next_slot_;
    }
    std::byte* slot = transport_->slot(next_slot_);
    // Header then payload, padded to the 4 B DMA granule.
    const size_t packet_bytes = (header_bytes_ + chunk_bytes + 3) & ~size_t{3};
    std::memcpy(slot, header_scratch_.data(), header_bytes_);
    std::memcpy(slot + header_bytes_, payload.data(), chunk_bytes);
    std::fill(slot + header_bytes_ + chunk_bytes, slot + packet_bytes, std::byte{0});
    staged_bytes_.push_back(static_cast<uint32_t>(packet_bytes));
    staged_++;
    next_slot_ = (next_slot_ + 1 == num_slots_) ? 0 : next_slot_ + 1;
}

void HostRdmaChannel::flush() {
    if (staged_ == 0) {
        return;
    }
    transport_->submit(staged_first_slot_, staged_bytes_);
    sent_ += staged_;
    staged_bytes_.clear();
    staged_ = 0;
}

void HostRdmaChannel::set_max_batch_packets(uint32_t packets) {
    TT_FATAL(
        packets >= 1 && packets <= num_slots_,
        "A batch is 1 to {} packets (the channel's slots), not {}",
        num_slots_,
        packets);
    flush();
    max_batch_ = packets;
}

void HostRdmaChannel::write(
    FabricNodeId dst,
    const tt_metal::CoreCoord& dst_virtual_core,
    uint64_t dst_addr,
    std::span<const std::byte> data,
    const std::optional<Completion>& completion) {
    TT_FATAL(!data.empty(), "A host_rdma write needs data");
    TT_FATAL(dst != src_, "A host_rdma write goes to another chip (the destination is the source, {})", src_);
    size_t offset = 0;
    while (offset < data.size()) {
        const size_t chunk = std::min(max_payload_bytes_, data.size() - offset);
        const bool last = offset + chunk == data.size();
        std::optional<std::pair<uint64_t, uint32_t>> completion_noc;
        if (last && completion.has_value()) {
            completion_noc = {noc_addr(completion->virtual_core, completion->addr), completion->increment};
        }
        build_header(dst, noc_addr(dst_virtual_core, dst_addr + offset), static_cast<uint32_t>(chunk), completion_noc);
        stage_packet(chunk, data.subspan(offset, chunk));
        offset += chunk;
    }
    // A write with a completion ends a transfer: send it now, so its completion is not held behind later writes.
    if (completion.has_value()) {
        flush();
    }
}

void HostRdmaChannel::drain() {
    flush();
    wait_for_credits(0);
}

// ---------------------------------------------------------------------------------------------------------------------
// HostRdmaService
// ---------------------------------------------------------------------------------------------------------------------

namespace sabi = tt::tt_fabric::host_rdma::service_abi;

std::shared_ptr<HostRdmaService> HostRdmaService::create(
    std::shared_ptr<tt_metal::distributed::MeshDevice> mesh_device, tt_metal::CoreCoord logical_core) {
    return std::shared_ptr<HostRdmaService>(new HostRdmaService(std::move(mesh_device), logical_core));
}

HostRdmaService::HostRdmaService(
    std::shared_ptr<tt_metal::distributed::MeshDevice> mesh_device, tt_metal::CoreCoord logical_core) :
    mesh_device_(std::move(mesh_device)), cluster_(tt::tt_metal::MetalContext::instance().get_cluster()) {
    auto& context = tt::tt_metal::MetalContext::instance();
    const auto& hal = context.hal();
    const auto& fabric_context = context.get_control_plane().get_fabric_context();
    TT_FATAL(mesh_device_ != nullptr, "The host_rdma service needs a mesh device");
    TT_FATAL(mesh_device_->get_devices().size() == 1, "The host_rdma service runs on a unit mesh (one chip)");
    chip_ = mesh_device_->get_devices().front()->id();
    node_ = context.get_control_plane().get_fabric_node_id_from_physical_chip_id(chip_);
    core_ = mesh_device_->worker_core_from_logical_core(logical_core);
    is_2d_ = fabric_context.is_2D_routing_enabled();

    // L1 of the service core from the unreserved base: the layout (header, channel table), then the staging slots,
    // one packet each.
    const uint32_t l1_base = static_cast<uint32_t>(hal.get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::DEFAULT_UNRESERVED));
    const uint32_t l1_end = l1_base + static_cast<uint32_t>(hal.get_dev_size(
                                          tt::tt_metal::HalProgrammableCoreType::TENSIX,
                                          tt::tt_metal::HalL1MemAddrType::DEFAULT_UNRESERVED));
    layout_addr_ = (l1_base + 63) & ~63u;
    const uint32_t staging_addr = (layout_addr_ + sizeof(sabi::Layout) + 63) & ~63u;
    const uint32_t staging_stride =
        (static_cast<uint32_t>(fabric_context.get_fabric_channel_buffer_size_bytes()) + 63) & ~63u;
    TT_FATAL(
        staging_addr + sabi::WINDOW * staging_stride <= l1_end,
        "The host_rdma service core's L1 cannot hold {} staging slots of {} B",
        sabi::WINDOW,
        staging_stride);

    auto layout = std::make_unique<sabi::Layout>();
    std::memset(static_cast<void*>(layout.get()), 0, sizeof(sabi::Layout));
    layout->header.staging_addr = staging_addr;
    layout->header.staging_stride = staging_stride;
    // Requests, counts and flags go through a window onto the core where one is free (#64622: strict MMIO writes
    // into Tensix L1 cost about a microsecond each). The layout goes first and is flushed before the kernel starts.
    writer_ = std::make_unique<tt_metal::distributed::CoreWriter>(
        cluster_, chip_, core_, staging_addr, /*l1_offset=*/0, "host_rdma service");
    writer_->write(layout.get(), sizeof(sabi::Layout), layout_addr_);
    tt_driver_atomics::sfence();
    cluster_.l1_barrier(chip_, {core_});

    tt_metal::Program program = tt_metal::CreateProgram();
    const auto kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/fabric/impl/kernels/host_rdma_service.cpp",
        logical_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});
    tt_metal::SetRuntimeArgs(program, kernel, logical_core, {layout_addr_});
    tt_metal::distributed::MeshWorkload workload;
    workload.add_program(tt_metal::distributed::MeshCoordinateRange(mesh_device_->shape()), std::move(program));
    tt_metal::distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload, /*blocking=*/false);
}

HostRdmaService::~HostRdmaService() {
    try {
        drain();
    } catch (const std::exception& e) {
        log_warning(tt::LogFabric, "host_rdma: draining the service on chip {} failed: {}", chip_, e.what());
    }
    try {
        for (auto& ch : channels_) {
            clear_router_credit_target(ch.router);
            write_word(channel_addr(ch.entry) + offsetof(sabi::Channel, active), 0);
        }
        write_word(layout_addr_ + offsetof(sabi::Header, terminate), 1);
        tt_metal::distributed::Finish(mesh_device_->mesh_command_queue());
    } catch (const std::exception& e) {
        log_warning(tt::LogFabric, "host_rdma: stopping the service on chip {} failed: {}", chip_, e.what());
    }
}

uint32_t HostRdmaService::channel_addr(uint32_t entry) const {
    return layout_addr_ + offsetof(sabi::Layout, channels) + entry * static_cast<uint32_t>(sizeof(sabi::Channel));
}

uint32_t HostRdmaService::read_word(uint32_t addr) const {
    uint32_t value = 0;
    cluster_.read_reg(&value, tt_cxy_pair(chip_, core_), addr);
    return value;
}

void HostRdmaService::write_word(uint32_t addr, uint32_t value) {
    // A word publishes what the host wrote before it (a produced count, an active flag): the earlier writes leave
    // the host first, and on the writer's one static VC they reach the core first.
    tt_driver_atomics::sfence();
    writer_->write(&value, sizeof(value), addr);
    tt_driver_atomics::sfence();
}

size_t HostRdmaService::channel_toward(FabricNodeId dst) {
    const chan_id_t eth_chan = HostRdmaChannel::eth_chan_toward(node_, dst);
    for (size_t i = 0; i < channels_.size(); i++) {
        if (channels_[i].router.eth_chan == eth_chan) {
            return i;
        }
    }
    TT_FATAL(
        channels_.size() < sabi::MAX_CHANNELS,
        "The host_rdma service on chip {} serves at most {} routers",
        chip_,
        sabi::MAX_CHANNELS);
    const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
    Channel ch;
    ch.entry = static_cast<uint32_t>(channels_.size());
    ch.router = find_router_channel(node_, eth_chan);
    const uint32_t max_payload = static_cast<uint32_t>(std::min<size_t>(
        fabric_context.get_fabric_max_payload_size_bytes(),
        ch.router.slots.slot_bytes - fabric_context.get_fabric_packet_header_size_bytes()));

    // The entry, inactive, then the router's credits pointed at it, then active.
    sabi::Channel entry{};
    std::memset(static_cast<void*>(&entry), 0, sizeof(entry));
    entry.credits = ch.router.slots.start;
    entry.start = ch.router.slots.start;
    entry.router_x = static_cast<uint32_t>(ch.router.slots.router_core.x);
    entry.router_y = static_cast<uint32_t>(ch.router.slots.router_core.y);
    entry.slots_base = ch.router.slots.slots_base;
    entry.free_slots_addr = ch.router.slots.free_slots_addr;
    entry.num_slots = ch.router.slots.num_slots;
    entry.slot_bytes = ch.router.slots.slot_bytes;
    entry.max_payload = max_payload & ~15u;
    cluster_.write_core(&entry, offsetof(sabi::Channel, queue), tt_cxy_pair(chip_, core_), channel_addr(ch.entry));
    cluster_.l1_barrier(chip_, {core_});
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    set_router_credit_target(
        ch.router,
        DeviceVisibleAddress{
            .noc_xy = hal.noc_xy_encoding(core_.x, core_.y),
            .addr = channel_addr(ch.entry) + offsetof(sabi::Channel, credits)});
    write_word(channel_addr(ch.entry) + offsetof(sabi::Channel, active), 1);
    log_debug(
        tt::LogFabric,
        "host_rdma: service on chip {} serves {} eth chan {} as entry {} ({} slots, {} B payload per packet)",
        chip_,
        node_,
        eth_chan,
        ch.entry,
        entry.num_slots,
        entry.max_payload);
    channels_.push_back(ch);
    return channels_.size() - 1;
}

sabi::Request HostRdmaService::encode(const WriteRequest& request, const Channel& ch) const {
    TT_FATAL(request.size > 0, "A host_rdma write needs data");
    TT_FATAL(request.dst != node_, "A host_rdma write goes to another chip (the destination is the source, {})", node_);
    TT_FATAL(
        request.src.addr % 16 == 0 && request.dst_at.addr % 16 == 0 && request.size % 16 == 0,
        "host_rdma write addresses and sizes are 16 B aligned (src 0x{:x}, dst 0x{:x}, {} B)",
        request.src.addr,
        request.dst_at.addr,
        request.size);
    sabi::Request r{};
    std::memset(static_cast<void*>(&r), 0, sizeof(r));
    r.src = {
        request.src.addr,
        static_cast<uint16_t>(request.src.virtual_core.x),
        static_cast<uint16_t>(request.src.virtual_core.y)};
    r.dst = {
        request.dst_at.addr,
        static_cast<uint16_t>(request.dst_at.virtual_core.x),
        static_cast<uint16_t>(request.dst_at.virtual_core.y)};
    r.size = request.size;
    r.dst_dev_id = static_cast<uint16_t>(request.dst.chip_id);
    r.dst_mesh_id = static_cast<uint16_t>(*request.dst.mesh_id);
    r.hops = is_2d_ ? 0 : hops_between(node_, ch.router.eth_chan, request.dst);
    if (request.completion.has_value()) {
        r.completion = {
            static_cast<uint32_t>(request.completion->addr),
            static_cast<uint16_t>(request.completion->virtual_core.x),
            static_cast<uint16_t>(request.completion->virtual_core.y)};
        r.completion_inc = request.completion->increment;
    }
    return r;
}

void HostRdmaService::post(Channel& ch, std::span<const sabi::Request> requests) {
    size_t done = 0;
    while (done < requests.size()) {
        // Room in the queue: fewer than QUEUE_DEPTH posted and not yet consumed (the kernel's count is read only
        // when the host's last reading leaves too little room).
        uint32_t room = sabi::QUEUE_DEPTH - (ch.produced - ch.consumed);
        if (room == 0) {
            const auto deadline = std::chrono::steady_clock::now() + kCreditTimeout;
            while ((room = sabi::QUEUE_DEPTH -
                           (ch.produced - (ch.consumed = read_word(
                                               channel_addr(ch.entry) + offsetof(sabi::Channel, consumed))))) == 0) {
                TT_FATAL(
                    std::chrono::steady_clock::now() < deadline,
                    "host_rdma: the service on chip {} took no request for {} s: {}",
                    chip_,
                    std::chrono::duration_cast<std::chrono::seconds>(kCreditTimeout).count(),
                    describe());
            }
        }
        // As many as fit, up to the queue's end: one block write, then (fenced after it) one produced count covering
        // them all.
        const uint32_t slot = ch.produced % sabi::QUEUE_DEPTH;
        const uint32_t count =
            static_cast<uint32_t>(std::min<size_t>({requests.size() - done, room, sabi::QUEUE_DEPTH - slot}));
        writer_->write(
            requests.data() + done,
            count * sizeof(sabi::Request),
            channel_addr(ch.entry) + offsetof(sabi::Channel, queue) +
                slot * static_cast<uint32_t>(sizeof(sabi::Request)));
        ch.produced += count;
        write_word(channel_addr(ch.entry) + offsetof(sabi::Channel, produced), ch.produced);
        done += count;
    }
}

void HostRdmaService::write(const WriteRequest& request) { write_batch(std::span<const WriteRequest>(&request, 1)); }

void HostRdmaService::write_batch(std::span<const WriteRequest> requests) {
    // Each request to the router toward its destination, in order per router; then each router's requests as blocks.
    std::vector<std::pair<size_t, std::vector<sabi::Request>>> by_channel;
    for (const auto& request : requests) {
        const size_t index = channel_toward(request.dst);
        auto it = std::find_if(by_channel.begin(), by_channel.end(), [&](const auto& e) { return e.first == index; });
        if (it == by_channel.end()) {
            by_channel.emplace_back(index, std::vector<sabi::Request>{});
            it = std::prev(by_channel.end());
        }
        it->second.push_back(encode(request, channels_[index]));
    }
    for (auto& [index, encoded] : by_channel) {
        post(channels_[index], encoded);
    }
}

void HostRdmaService::drain() {
    const auto deadline = std::chrono::steady_clock::now() + kCreditTimeout;
    for (auto& ch : channels_) {
        while ((ch.consumed = read_word(channel_addr(ch.entry) + offsetof(sabi::Channel, consumed))) != ch.produced) {
            TT_FATAL(
                std::chrono::steady_clock::now() < deadline,
                "host_rdma: the service on chip {} did not drain in {} s: {}",
                chip_,
                std::chrono::duration_cast<std::chrono::seconds>(kCreditTimeout).count(),
                describe());
        }
    }
}

std::string HostRdmaService::describe() const {
    std::string out = fmt::format(
        "service on chip {} core {}: heartbeat {}",
        chip_,
        core_.str(),
        read_word(layout_addr_ + offsetof(sabi::Header, heartbeat)));
    for (const auto& ch : channels_) {
        sabi::Channel entry{};
        cluster_.read_core(&entry, offsetof(sabi::Channel, queue), tt_cxy_pair(chip_, core_), channel_addr(ch.entry));
        out += fmt::format(
            "; eth chan {}: produced {} (host {}), consumed {}, router credits {} (start {}, {} slots)",
            ch.router.eth_chan,
            entry.produced,
            ch.produced,
            entry.consumed,
            entry.credits,
            entry.start,
            entry.num_slots);
    }
    return out;
}

}  // namespace tt::tt_fabric::host_rdma
