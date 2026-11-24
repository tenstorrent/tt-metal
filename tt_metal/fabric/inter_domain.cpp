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
#include "fabric_builder_context.hpp"
#include "fabric_context.hpp"
#include "impl/context/metal_context.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_fabric::experimental {

namespace {

using Clock = std::chrono::steady_clock;

tt::tt_metal::MetalContext& metal(const InterDomainPort& port) {
    return tt::tt_metal::MetalContext::instance(tt::tt_metal::ContextId{port.context});
}

const FabricContext& fabric_context(const InterDomainPort& port) {
    return metal(port).get_control_plane().get_fabric_context();
}

const FabricEriscDatamoverConfig& router_config(const InterDomainPort& port) {
    return fabric_context(port).get_builder_context().get_fabric_router_config();
}

tt_cxy_pair router_core(const InterDomainPort& port) {
    const auto& control_plane = fabric_context(port).get_control_plane();
    const ChipId chip = control_plane.get_physical_chip_id_from_fabric_node_id(port.node);
    return tt_cxy_pair(chip, fabric_context(port).get_cluster().get_virtual_eth_core_from_channel(chip, port.eth_chan));
}

uint64_t ctrl_addr(const InterDomainPort& port) {
    const auto addr = router_config(port).inter_domain_ctrl_address;
    TT_FATAL(addr != 0, "inter-domain fabric is off (TT_METAL_FABRIC_INTER_DOMAIN=1)");
    return addr;
}

InterDomainRouterControl read_control(const InterDomainPort& port) {
    InterDomainRouterControl c{};
    fabric_context(port).get_cluster().read_core(&c, sizeof(c), router_core(port), ctrl_addr(port));
    return c;
}

void write_and_flush(const tt::Cluster& cluster, const void* data, uint32_t size, tt_cxy_pair core, uint64_t addr) {
    cluster.write_core(data, size, core, addr);
    cluster.l1_barrier(core.chip, {tt::tt_metal::CoreCoord(core.x, core.y)});
}

void write_word(const InterDomainPort& port, size_t offset, uint32_t value) {
    write_and_flush(fabric_context(port).get_cluster(), &value, sizeof(value), router_core(port), ctrl_addr(port) + offset);
}

void send_command(const InterDomainPort& port, InterDomainCommand command) {
    write_word(port, offsetof(InterDomainRouterControl, command), static_cast<uint32_t>(command));
}

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
    const auto* layout = fabric_context(port).get_builder_context().find_router_channel_layout(port.node, port.eth_chan);
    TT_FATAL(layout != nullptr, "inter-domain: no router on {} eth channel {}", port.node, port.eth_chan);
    return *layout;
}

uint32_t layout_fingerprint(const InterDomainPort& port) {
    const auto& config = router_config(port);
    uint32_t h = 2166136261u;
    for (const size_t v :
         {config.handshake_addr,
          config.inter_domain_ctrl_address,
          sizeof(InterDomainRouterControl),
          offsetof(InterDomainRouterControl, peer_marker),
          offsetof(InterDomainRouterControl, marker)}) {
        h = (h ^ static_cast<uint32_t>(v)) * 16777619u;
    }
    return h;
}

}  // namespace

std::vector<InterDomainPort> get_inter_domain_ports(const tt::tt_metal::distributed::MeshDevice& mesh_device) {
    const auto context = mesh_device.impl().get_context_id();
    auto ports = tt::tt_metal::MetalContext::instance(context)
                     .get_control_plane()
                     .get_fabric_context()
                     .get_builder_context()
                     .get_inter_domain_ports();
    for (auto& port : ports) {
        port.context = context.get();
    }
    return ports;
}

uint64_t get_inter_domain_port_asic(const InterDomainPort& port) {
    return *fabric_context(port).get_control_plane().get_asic_id_from_fabric_node_id(port.node);
}

InterDomainMetadata get_inter_domain_metadata(const InterDomainPort& port, uint32_t domain_id) {
    const auto& vc2_rx = layout_of(port).receivers[2].at(0);
    return InterDomainMetadata{
        .domain_id = domain_id,
        .vc2_receiver_channel_addr = vc2_rx.base_address,
        .vc2_receiver_channel_slots = vc2_rx.num_slots,
        .max_packet_size_bytes = static_cast<uint32_t>(router_config(port).channel_buffer_size_bytes),
        .layout_fingerprint = layout_fingerprint(port)};
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
    TT_FATAL(
        peer.layout_fingerprint == layout_fingerprint(port),
        "inter-domain: {} eth channel {}: domain {}'s router L1 layout differs from this one's",
        port.node,
        port.eth_chan,
        peer.domain_id);
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
    const auto& control_plane = fabric_context(port).get_control_plane();
    auto info = control_plane.get_router_connection_info(port.node, port.eth_chan, 0);
    info.edm_direction = control_plane.routing_direction_to_eth_direction(port.direction);
    const auto router = router_core(port);

    const auto& cluster = fabric_context(port).get_cluster();
    const auto worker =
        cluster.get_virtual_coordinate_from_logical_coordinates(router.chip, worker_core, CoreType::WORKER);
    const uint64_t table = metal(port).hal().get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::TENSIX_FABRIC_CONNECTIONS);
    write_and_flush(
        cluster,
        &info,
        sizeof(info),
        tt_cxy_pair(router.chip, worker),
        table + offsetof(tensix_fabric_connections_l1_info_t, read_only) + port.eth_chan * sizeof(info));

    const uint32_t teardown = tt::tt_metal::CreateSemaphore(program, {worker_core}, 0);
    const uint32_t buffer_index = tt::tt_metal::CreateSemaphore(program, {worker_core}, 0);
    append_worker_to_fabric_edm_sender_rt_args(port.eth_chan, teardown, buffer_index, rt_args);
}

}  // namespace tt::tt_fabric::experimental
