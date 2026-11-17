// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/inter_domain.hpp>

#include <thread>

#include <enchantum/enchantum.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt_stl/assert.hpp>

#include "erisc_datamover_builder.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_interface.hpp"
#include "fabric_builder_context.hpp"
#include "fabric_context.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_fabric::experimental {

namespace {

using Clock = std::chrono::steady_clock;

const FabricContext& fabric_context() {
    return tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
}

const FabricEriscDatamoverConfig& router_config() {
    return fabric_context().get_builder_context().get_fabric_router_config();
}

// the router's eth core, as the cluster addresses it
tt_cxy_pair router_core(const InterDomainPort& port) {
    const auto& control_plane = fabric_context().get_control_plane();
    const ChipId chip = control_plane.get_physical_chip_id_from_fabric_node_id(port.node);
    return tt_cxy_pair(chip, fabric_context().get_cluster().get_virtual_eth_core_from_channel(chip, port.eth_chan));
}

uint64_t ctrl_addr() {
    const auto addr = router_config().inter_domain_ctrl_address;
    TT_FATAL(addr != 0, "inter-domain fabric is off (TT_METAL_FABRIC_INTER_DOMAIN=1)");
    return addr;
}

InterDomainRouterControl read_control(const InterDomainPort& port) {
    InterDomainRouterControl c{};
    fabric_context().get_cluster().read_core(&c, sizeof(c), router_core(port), ctrl_addr());
    return c;
}

// writes, then reads back so the write has landed (a chip-wide barrier would poke routers mid-traffic)
void write_and_flush(const void* data, uint32_t size, tt_cxy_pair core, uint64_t addr) {
    const auto& cluster = fabric_context().get_cluster();
    cluster.write_core(data, size, core, addr);
    uint32_t readback = 0;
    cluster.read_core(&readback, sizeof(readback), core, addr);
}

void write_word(const InterDomainPort& port, size_t offset, uint32_t value) {
    write_and_flush(&value, sizeof(value), router_core(port), ctrl_addr() + offset);
}

void send_command(const InterDomainPort& port, InterDomainCommand command) {
    write_word(port, offsetof(InterDomainRouterControl, command), static_cast<uint32_t>(command));
}

// polls the control block until `done` or the deadline
template <typename Done>
std::optional<InterDomainRouterControl> wait_for(const InterDomainPort& port, Clock::time_point deadline, Done done) {
    while (true) {
        const auto c = read_control(port);
        if (done(c)) {
            return c;
        }
        if (Clock::now() > deadline) {
            return std::nullopt;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

const RouterChannelLayout& layout_of(const InterDomainPort& port) {
    const auto* layout = fabric_context().get_builder_context().find_router_channel_layout(port.node, port.eth_chan);
    TT_FATAL(layout != nullptr, "inter-domain: no router on {} eth channel {}", port.node, port.eth_chan);
    return *layout;
}

}  // namespace

std::vector<InterDomainPort> get_inter_domain_ports() {
    return fabric_context().get_builder_context().get_inter_domain_ports();
}

uint64_t get_inter_domain_port_asic(const InterDomainPort& port) {
    return fabric_context().get_cluster().get_unique_chip_ids().at(router_core(port).chip);
}

InterDomainMetadata get_inter_domain_metadata(const InterDomainPort& port, uint32_t domain_id) {
    const auto& vc2_rx = layout_of(port).receivers[2].at(0);
    return InterDomainMetadata{
        .domain_id = domain_id,
        .vc2_receiver_channel_addr = vc2_rx.base_address,
        .vc2_receiver_channel_slots = vc2_rx.num_slots,
        .max_packet_size_bytes = static_cast<uint32_t>(router_config().channel_buffer_size_bytes)};
}

InterDomainState get_inter_domain_state(const InterDomainPort& port) { return read_control(port).state; }

void connect_inter_domain(
    const InterDomainPort& port,
    uint32_t domain_id,
    const InterDomainMetadata& peer,
    std::chrono::milliseconds timeout) {
    TT_FATAL(
        port.has_router,
        "inter-domain: {} eth channel {} has no router ({}), can't connect over it",
        port.node,
        port.eth_chan,
        port.link_trained ? "Z isn't supported yet" : "the link never trained");
    const auto before = read_control(port);
    TT_FATAL(
        before.state == InterDomainState::DRAIN && before.command == InterDomainCommand::NONE,
        "inter-domain: {} eth channel {} is {}, not DRAIN",
        port.node,
        port.eth_chan,
        enchantum::to_string(before.state));

    write_word(port, offsetof(InterDomainRouterControl, local_domain_id), domain_id);
    write_word(port, offsetof(InterDomainRouterControl, peer_domain_id), peer.domain_id);
    write_word(port, offsetof(InterDomainRouterControl, peer_vc2_receiver_addr), peer.vc2_receiver_channel_addr);
    write_word(port, offsetof(InterDomainRouterControl, peer_vc2_receiver_slots), peer.vc2_receiver_channel_slots);
    write_word(port, offsetof(InterDomainRouterControl, peer_max_packet_size_bytes), peer.max_packet_size_bytes);
    send_command(port, InterDomainCommand::CONNECT);

    auto settled = [](const InterDomainRouterControl& c) { return c.command == InterDomainCommand::NONE; };
    auto c = wait_for(port, Clock::now() + timeout, settled);
    if (!c) {
        send_command(port, InterDomainCommand::ABORT);
        (void)wait_for(port, Clock::now() + std::chrono::seconds(5), settled);
        TT_THROW("inter-domain: {} eth channel {}: no peer within {} ms", port.node, port.eth_chan, timeout.count());
    }
    TT_FATAL(
        c->state == InterDomainState::SERVICE_TRAFFIC,
        "inter-domain: {} eth channel {}: connect to domain {} failed (peer metadata doesn't match this router)",
        port.node,
        port.eth_chan,
        peer.domain_id);
    log_info(
        tt::LogFabric,
        "inter-domain: {} eth channel {} connected, domain {} -> {}",
        port.node,
        port.eth_chan,
        domain_id,
        peer.domain_id);
}

void disconnect_inter_domain(const InterDomainPort& port, bool force, std::chrono::milliseconds timeout) {
    auto drained = [](const InterDomainRouterControl& c) {
        return c.command == InterDomainCommand::NONE && c.state == InterDomainState::DRAIN;
    };
    send_command(port, force ? InterDomainCommand::FORCE_DISCONNECT : InterDomainCommand::DISCONNECT);
    if (!wait_for(port, Clock::now() + timeout, drained)) {
        log_warning(
            tt::LogFabric,
            "inter-domain: {} eth channel {}: no disconnect from the peer, forcing",
            port.node,
            port.eth_chan);
        send_command(port, InterDomainCommand::FORCE_DISCONNECT);
        TT_FATAL(
            wait_for(port, Clock::now() + std::chrono::seconds(5), drained),
            "inter-domain: {} eth channel {}: router did not go back to DRAIN",
            port.node,
            port.eth_chan);
    }
}

void append_inter_domain_connection_rt_args(
    const InterDomainPort& port,
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreCoord& worker_core,
    std::vector<uint32_t>& rt_args) {
    // the router's worker channel (VC0's channel 0), listed in the worker's fabric connection table like any router's
    const auto& config = router_config();
    const auto& channel = layout_of(port).senders[0].at(0);
    const auto router = router_core(port);
    fabric_connection_info_t info{};
    info.edm_buffer_base_addr = channel.base_address;
    info.edm_connection_handshake_addr = config.sender_channels_connection_semaphore_address[0];
    info.edm_worker_location_info_addr = config.sender_channels_worker_conn_info_base_address[0];
    info.buffer_index_semaphore_id = config.sender_channels_buffer_index_semaphore_address[0];
    info.buffer_size_bytes = config.channel_buffer_size_bytes;
    info.edm_direction = fabric_context().get_control_plane().routing_direction_to_eth_direction(port.direction);
    info.edm_noc_x = router.x;
    info.edm_noc_y = router.y;
    info.num_buffers_per_channel = channel.num_slots;
    info.worker_free_slots_stream_id = connection_interface::sender_channel_0_free_slots_stream_id;

    const auto& cluster = fabric_context().get_cluster();
    const auto worker =
        cluster.get_virtual_coordinate_from_logical_coordinates(router.chip, worker_core, CoreType::WORKER);
    const uint64_t table = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::TENSIX_FABRIC_CONNECTIONS);
    write_and_flush(
        &info,
        sizeof(info),
        tt_cxy_pair(router.chip, worker),
        table + offsetof(tensix_fabric_connections_l1_info_t, read_only) + port.eth_chan * sizeof(info));

    const uint32_t teardown = tt::tt_metal::CreateSemaphore(program, {worker_core}, 0);
    const uint32_t buffer_index = tt::tt_metal::CreateSemaphore(program, {worker_core}, 0);
    append_worker_to_fabric_edm_sender_rt_args(port.eth_chan, teardown, buffer_index, rt_args);
}

}  // namespace tt::tt_fabric::experimental
