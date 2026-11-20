// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/host_rdma/host_rdma.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-umd/driver_atomics.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <hostdevcommon/fabric_common.h>

#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include "tt_metal/distributed/core_writer.hpp"
#include "tt_metal/distributed/mesh_device_impl.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "tt_metal/fabric/builder/fabric_stream_assignment.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/host_rdma_service_abi.hpp"

namespace tt::tt_fabric::host_rdma {

namespace {

// How long the host waits for the service kernel before declaring it stuck.
constexpr std::chrono::seconds kServiceTimeout{10};

}  // namespace

// ---------------------------------------------------------------------------------------------------------------------
// Router channels
// ---------------------------------------------------------------------------------------------------------------------

RouterChannel find_router_channel(tt::tt_metal::MetalContext& context, FabricNodeId src, chan_id_t eth_chan) {
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
    rc.chip = control_plane.get_physical_chip_id_from_fabric_node_id(src);
    rc.router_core = cluster.get_virtual_eth_core_from_channel(rc.chip, eth_chan);

    // The channel: the router's last VC0 sender, laid out by the router itself.
    const auto& layout = builder_context.get_router_channel_layout(src, eth_chan);
    TT_FATAL(!layout.senders[0].empty(), "Router {} channel {} has no VC0 senders", src, eth_chan);
    rc.channel = static_cast<uint32_t>(layout.senders[0].size() - 1);
    // A worker channel of its own: never the one the fabric's worker connection APIs hand out.
    TT_FATAL(
        rc.channel != get_worker_connected_sender_channel(),
        "Router {} channel {}: the host_rdma channel must not be the fabric's worker channel ({})",
        src,
        eth_chan,
        rc.channel);
    rc.slots_base = layout.senders[0].back().base_address;
    rc.num_slots = layout.senders[0].back().num_slots;
    const auto& edm_config = builder_context.get_fabric_router_config();
    rc.slot_bytes = static_cast<uint32_t>(edm_config.channel_buffer_size_bytes);
    rc.connection_info_addr =
        static_cast<uint32_t>(edm_config.sender_channels_worker_conn_info_base_address[rc.channel]);
    rc.connection_semaphore_addr =
        static_cast<uint32_t>(edm_config.sender_channels_connection_semaphore_address[rc.channel]);
    rc.producer_cursor_addr =
        static_cast<uint32_t>(edm_config.sender_channels_buffer_index_semaphore_address[rc.channel]);
    rc.free_slots_stream =
        builder_context.get_stream_assignment(src.mesh_id).id(StreamRole::SENDER_FREE_SLOTS, 0, rc.channel);
    TT_FATAL(
        rc.num_slots > 0 && rc.slot_bytes > 0, "Router {} channel {} has an empty host_rdma channel", src, eth_chan);
    TT_FATAL(
        rc.connection_info_addr != 0 && rc.connection_semaphore_addr != 0,
        "Router {} channel {}: the router config set aside no connection words for sender channel {}",
        src,
        eth_chan,
        rc.channel);
    return rc;
}

chan_id_t eth_chan_toward(tt::tt_metal::MetalContext& context, FabricNodeId src, FabricNodeId dst, uint32_t link_idx) {
    TT_FATAL(src != dst, "A host_rdma write goes to another chip (src and dst are both {})", src);
    const auto& control_plane = context.get_control_plane();
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

uint32_t hops_between(tt::tt_metal::MetalContext& context, FabricNodeId src, chan_id_t eth_chan, FabricNodeId dst) {
    const auto route = context.get_control_plane().get_fabric_route(src, dst, eth_chan);
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
// HostRdmaService
// ---------------------------------------------------------------------------------------------------------------------

namespace sabi = tt::tt_fabric::host_rdma::service_abi;

std::shared_ptr<HostRdmaService> HostRdmaService::create(
    std::shared_ptr<tt_metal::distributed::MeshDevice> mesh_device, tt_metal::CoreCoord logical_core) {
    TT_FATAL(mesh_device != nullptr, "The host_rdma service needs a mesh device");
    return std::shared_ptr<HostRdmaService>(new HostRdmaService(std::move(mesh_device), logical_core));
}

HostRdmaService::HostRdmaService(
    std::shared_ptr<tt_metal::distributed::MeshDevice> mesh_device, tt_metal::CoreCoord logical_core) :
    mesh_device_(std::move(mesh_device)),
    context_(mesh_device_->impl().metal_context()),
    cluster_(context_.get_cluster()) {
    const auto& hal = context_.hal();
    const auto& fabric_context = context_.get_control_plane().get_fabric_context();
    TT_FATAL(mesh_device_->get_devices().size() == 1, "The host_rdma service runs on a unit mesh (one chip)");
    chip_ = mesh_device_->get_devices().front()->id();
    node_ = context_.get_control_plane().get_fabric_node_id_from_physical_chip_id(chip_);
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

HostRdmaService::~HostRdmaService() noexcept {
    // No drain: the owner drains before closing (draining could wait on the device indefinitely; #66677). The kernel
    // stops as soon as it sees `terminate`, closing its connections.
    try {
        write_word(layout_addr_ + offsetof(sabi::Header, terminate), 1);
        tt_metal::distributed::Finish(mesh_device_->mesh_command_queue());
    } catch (const std::exception& e) {
        log_warning(tt::LogFabric, "host_rdma: stopping the service on chip {} failed: {}", chip_, e.what());
    } catch (...) {
        log_warning(
            tt::LogFabric, "host_rdma: stopping the service on chip {} failed with an unknown exception", chip_);
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
    const chan_id_t eth_chan = eth_chan_toward(context_, node_, dst);
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
    const auto& fabric_context = context_.get_control_plane().get_fabric_context();
    Channel ch;
    ch.entry = static_cast<uint32_t>(channels_.size());
    ch.router = find_router_channel(context_, node_, eth_chan);
    // The kernel's connection object keeps the slot count in 8 bits and the slot stride in 16.
    TT_FATAL(
        ch.router.num_slots <= std::numeric_limits<uint8_t>::max() &&
            ch.router.slot_bytes <= std::numeric_limits<uint16_t>::max(),
        "Router {} eth chan {}: {} slots of {} B do not fit a fabric connection",
        node_,
        eth_chan,
        ch.router.num_slots,
        ch.router.slot_bytes);
    const uint32_t max_payload = static_cast<uint32_t>(std::min<size_t>(
        fabric_context.get_fabric_max_payload_size_bytes(),
        ch.router.slot_bytes - fabric_context.get_fabric_packet_header_size_bytes()));

    // The entry, inactive, then active (the kernel then opens the connection).
    sabi::Channel entry{};
    std::memset(static_cast<void*>(&entry), 0, sizeof(entry));
    entry.max_payload = max_payload & ~15u;
    entry.router_x = static_cast<uint32_t>(ch.router.router_core.x);
    entry.router_y = static_cast<uint32_t>(ch.router.router_core.y);
    entry.slots_base = ch.router.slots_base;
    entry.num_slots = ch.router.num_slots;
    entry.slot_bytes = ch.router.slot_bytes;
    entry.free_slots_stream = ch.router.free_slots_stream;
    entry.connection_semaphore_addr = ch.router.connection_semaphore_addr;
    entry.connection_info_addr = ch.router.connection_info_addr;
    entry.producer_cursor_addr = ch.router.producer_cursor_addr;
    cluster_.write_core(&entry, offsetof(sabi::Channel, queue), tt_cxy_pair(chip_, core_), channel_addr(ch.entry));
    cluster_.l1_barrier(chip_, {core_});
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
    r.hops = is_2d_ ? 0 : hops_between(context_, node_, ch.router.eth_chan, request.dst);
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
            const auto deadline = std::chrono::steady_clock::now() + kServiceTimeout;
            while ((room = sabi::QUEUE_DEPTH -
                           (ch.produced - (ch.consumed = read_word(
                                               channel_addr(ch.entry) + offsetof(sabi::Channel, consumed))))) == 0) {
                TT_FATAL(
                    std::chrono::steady_clock::now() < deadline,
                    "host_rdma: the service on chip {} took no request for {} s: {}",
                    chip_,
                    std::chrono::duration_cast<std::chrono::seconds>(kServiceTimeout).count(),
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
    const auto deadline = std::chrono::steady_clock::now() + kServiceTimeout;
    for (auto& ch : channels_) {
        while ((ch.consumed = read_word(channel_addr(ch.entry) + offsetof(sabi::Channel, consumed))) != ch.produced) {
            TT_FATAL(
                std::chrono::steady_clock::now() < deadline,
                "host_rdma: the service on chip {} did not drain in {} s: {}",
                chip_,
                std::chrono::duration_cast<std::chrono::seconds>(kServiceTimeout).count(),
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
            "; eth chan {}: produced {} (host {}), consumed {}, router credits {} ({} slots)",
            ch.router.eth_chan,
            entry.produced,
            ch.produced,
            entry.consumed,
            entry.credits,
            entry.num_slots);
    }
    return out;
}

}  // namespace tt::tt_fabric::host_rdma
