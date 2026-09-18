// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/sockets/host_mesh_socket.hpp>

#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "tt_metal/distributed/host_transport/rdma_link.hpp"
#include "tt_metal/distributed/host_transport/socket_relay.hpp"
#include "tt_metal/distributed/mesh_socket_utils.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>

namespace tt::tt_metal::distributed {

using host_transport::RdmaChannel;
using host_transport::RdmaContext;
using host_transport::RdmaEndpoint;
using host_transport::RdmaRegion;
using host_transport::RelayLoop;
using host_transport::RelayReceiver;
using host_transport::RelaySender;
using host_transport::RingGeometry;

namespace {

// Handshake tag base; distinct from MeshSocket's descriptor exchange. Each socket
// instance takes the next tag so two sockets between the same rank pair (a
// forward and a reverse channel, say) cannot mix up each other's endpoints. Both
// ranks must construct their sockets in the same order, which is the same
// assumption MeshSocket's exchange-tag counter makes.
constexpr uint32_t kEndpointExchangeTagBase = 0x7248;

multihost::Tag next_exchange_tag() {
    static std::atomic<uint32_t> counter{0};
    return multihost::Tag{static_cast<int>(kEndpointExchangeTagBase + counter.fetch_add(1))};
}

// One RDMA context per process: the device and protection domain are shared by
// every channel, while completion queues stay per-channel.
RdmaContext& shared_context(const HostMeshSocket::TransportConfig& transport) {
    static std::unique_ptr<RdmaContext> context;
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    if (context == nullptr) {
        context = std::make_unique<RdmaContext>(RdmaContext::Config{
            .device_name = transport.rdma_device,
            .gid_index = transport.gid_index,
        });
    }
    return *context;
}

std::span<std::byte> as_bytes(RdmaEndpoint& endpoint) {
    return {reinterpret_cast<std::byte*>(&endpoint), sizeof(RdmaEndpoint)};
}

}  // namespace

struct HostMeshSocket::Impl {
    // One relayed connection, plus the per-core socket that terminates it.
    // Declaration order sets teardown order (members die in reverse), and here it
    // is load-bearing: stop polling, then destroy the queue pair so no work
    // request can still reference a memory region, then deregister the regions,
    // and only then free the memory they covered.
    struct Connection {
        std::unique_ptr<D2HSocket> d2h;           // sender endpoints only; owns its FIFO
        std::unique_ptr<H2DSocket> h2d;           // receiver endpoints only; owns its FIFO
        std::unique_ptr<uint32_t> credit_slot;    // sender: peer writes credit here
        std::unique_ptr<uint32_t> doorbell_slot;  // receiver: peer writes its total here
        std::unique_ptr<RdmaRegion> fifo_region;
        std::unique_ptr<RdmaRegion> credit_region;
        std::unique_ptr<RdmaRegion> doorbell_region;
        std::unique_ptr<RdmaChannel> channel;
        std::shared_ptr<host_transport::RelayEndpoint> relay;
    };

    SocketConfig config;
    TransportConfig transport;
    SocketEndpoint endpoint = SocketEndpoint::SENDER;
    MeshDevice* mesh_device = nullptr;
    std::shared_ptr<MeshBuffer> config_buffer;
    DeviceAddr config_buffer_address = 0;
    RingGeometry geometry;
    std::vector<Connection> connections;
    std::vector<MeshCoreCoord> active_cores;
    bool participates = false;

    bool poll() {
        bool progress = false;
        for (auto& connection : connections) {
            progress |= connection.relay->poll();
        }
        return progress;
    }
};

namespace {

// The local endpoint cores, in connection order, deduplicated.
std::vector<MeshCoreCoord> local_cores(const SocketConfig& config, SocketEndpoint endpoint) {
    std::vector<MeshCoreCoord> cores;
    for (const auto& connection : config.socket_connection_config) {
        const auto& core = endpoint == SocketEndpoint::SENDER ? connection.sender_core : connection.receiver_core;
        if (std::find(cores.begin(), cores.end(), core) == cores.end()) {
            cores.push_back(core);
        }
    }
    return cores;
}

}  // namespace

HostMeshSocket::HostMeshSocket(
    const std::shared_ptr<MeshDevice>& device, const SocketConfig& config, const TransportConfig& transport) :
    impl_(std::make_unique<Impl>()) {
    impl_->config = config;
    impl_->transport = transport;
    impl_->mesh_device = device.get();

    const uint32_t page_size = transport.page_size;
    const uint32_t fifo_size = config.socket_mem_config.fifo_size;
    TT_FATAL(page_size != 0, "HostMeshSocket requires TransportConfig::page_size");
    TT_FATAL(fifo_size != 0, "HostMeshSocket requires a non-zero fifo_size");
    TT_FATAL(
        fifo_size % page_size == 0,
        "fifo_size ({}) must be a whole number of pages of {} B: a partial tail page would have to be charged to the "
        "flow-control counters on both sides, and the page size need not be a power of two",
        fifo_size,
        page_size);
    TT_FATAL(!config.socket_connection_config.empty(), "HostMeshSocket requires at least one connection");
    impl_->geometry = RingGeometry{.page_size = page_size, .num_pages = fifo_size / page_size};

    const auto context =
        config.distributed_context ? config.distributed_context : multihost::DistributedContext::get_current_world();
    const auto rank = context->rank();
    TT_FATAL(
        config.sender_rank != config.receiver_rank,
        "HostMeshSocket connects two ranks over the host interconnect; sender_rank and receiver_rank are both {}",
        *config.sender_rank);

    if (rank == config.sender_rank) {
        impl_->endpoint = SocketEndpoint::SENDER;
    } else if (rank == config.receiver_rank) {
        impl_->endpoint = SocketEndpoint::RECEIVER;
    } else {
        log_warning(
            tt::LogDistributed,
            "Rank {} is neither the sender ({}) nor the receiver ({}) of this HostMeshSocket; it allocates nothing",
            *rank,
            *config.sender_rank,
            *config.receiver_rank);
        return;
    }
    impl_->participates = true;
    impl_->active_cores = local_cores(config, impl_->endpoint);

    // One height-sharded config buffer over the local endpoint cores, exactly as
    // MeshSocket allocates it: one page per core, so every core finds its own
    // metadata at the same L1 address and a multi-core kernel needs one address.
    impl_->config_buffer = create_socket_config_buffer(device, config, impl_->endpoint);
    impl_->config_buffer_address = impl_->config_buffer->address();

    auto& rdma = shared_context(transport);
    const bool is_sender = impl_->endpoint == SocketEndpoint::SENDER;
    const auto exchange_tag = next_exchange_tag();

    impl_->connections.resize(config.socket_connection_config.size());
    for (size_t i = 0; i < config.socket_connection_config.size(); i++) {
        const auto& wire = config.socket_connection_config[i];
        auto& connection = impl_->connections[i];

        connection.channel = std::make_unique<RdmaChannel>(rdma);
        RdmaEndpoint local = connection.channel->local_endpoint();

        if (is_sender) {
            connection.d2h = std::make_unique<D2HSocket>(
                device,
                wire.sender_core,
                fifo_size,
                D2HSocket::ExternalConfigBuffer{
                    .address = static_cast<uint32_t>(impl_->config_buffer_address),
                },
                D2HSocket::ProcessScope::InProcess);
            connection.d2h->set_page_size(page_size);

            // The pinned D2H ring is the NIC's source; the credit slot is where
            // the peer reports what the far device has consumed.
            auto fifo = connection.d2h->host_fifo();
            connection.fifo_region = std::make_unique<RdmaRegion>(rdma, fifo.data(), fifo.size());
            connection.credit_slot = std::make_unique<uint32_t>(0);
            connection.credit_region =
                std::make_unique<RdmaRegion>(rdma, connection.credit_slot.get(), sizeof(uint32_t));
            local.credit = connection.credit_region->descriptor();
        } else {
            connection.h2d = std::make_unique<H2DSocket>(
                device,
                wire.receiver_core,
                config.socket_mem_config.socket_storage_type,
                fifo_size,
                H2DMode::DEVICE_PULL,
                H2DSocket::ExternalConfigBuffer{
                    .address = static_cast<uint32_t>(impl_->config_buffer_address),
                });
            connection.h2d->set_page_size(page_size);

            // The pinned H2D ring is the NIC's destination.
            auto fifo = connection.h2d->host_fifo();
            connection.fifo_region = std::make_unique<RdmaRegion>(rdma, fifo.data(), fifo.size());
            local.fifo = connection.fifo_region->descriptor();

            // Where the sender publishes how many pages it has forwarded.
            connection.doorbell_slot = std::make_unique<uint32_t>(0);
            connection.doorbell_region =
                std::make_unique<RdmaRegion>(rdma, connection.doorbell_slot.get(), sizeof(uint32_t));
            local.doorbell = connection.doorbell_region->descriptor();
        }

        // Exchange in opposite order on the two sides so neither blocks on a send
        // the peer is not yet reading, mirroring MeshSocket::connect_with_peer.
        RdmaEndpoint remote{};
        if (is_sender) {
            context->send(as_bytes(local), config.receiver_rank, exchange_tag);
            context->recv(as_bytes(remote), config.receiver_rank, exchange_tag);
        } else {
            context->recv(as_bytes(remote), config.sender_rank, exchange_tag);
            context->send(as_bytes(local), config.sender_rank, exchange_tag);
        }

        if (is_sender) {
            TT_FATAL(
                remote.fifo.len == fifo_size,
                "peer FIFO is {} bytes, local is {}: both rings must hold the same number of pages",
                remote.fifo.len,
                fifo_size);
            TT_FATAL(remote.doorbell.len >= sizeof(uint32_t), "peer advertised no doorbell slot");
        }
        connection.channel->connect(remote);

        if (is_sender) {
            connection.relay = std::make_shared<RelaySender>(
                *connection.channel,
                *connection.d2h,
                *connection.fifo_region,
                *connection.credit_region,
                impl_->geometry,
                transport.max_batch_pages);
        } else {
            connection.relay = std::make_shared<RelayReceiver>(
                *connection.channel, *connection.h2d, *connection.doorbell_region, impl_->geometry);
        }
    }

    // Both sides are wired before either kernel can run.
    context->barrier();

    if (transport.own_relay_thread) {
        for (auto& connection : impl_->connections) {
            RelayLoop::instance().add(connection.relay);
        }
    }
}

HostMeshSocket::~HostMeshSocket() {
    if (impl_ == nullptr) {
        return;
    }
    if (impl_->transport.own_relay_thread) {
        for (auto& connection : impl_->connections) {
            if (connection.relay) {
                RelayLoop::instance().remove(connection.relay);
            }
        }
    }
}

HostMeshSocket::HostMeshSocket(HostMeshSocket&&) noexcept = default;
HostMeshSocket& HostMeshSocket::operator=(HostMeshSocket&&) noexcept = default;

DeviceAddr HostMeshSocket::get_config_buffer_address() const { return impl_->config_buffer_address; }

std::shared_ptr<MeshBuffer> HostMeshSocket::get_config_buffer() const { return impl_->config_buffer; }

std::shared_ptr<MeshBuffer> HostMeshSocket::get_data_buffer() const {
    TT_THROW(
        "HostMeshSocket has no device-side data buffer: the receiver's FIFO lives in pinned host memory and the "
        "kernel pulls from it into a destination of the caller's choosing. Allocate that landing buffer directly.");
}

const SocketConfig& HostMeshSocket::get_config() const { return impl_->config; }

SocketEndpoint HostMeshSocket::get_socket_endpoint_type() const { return impl_->endpoint; }

std::vector<MeshCoreCoord> HostMeshSocket::get_active_cores() const { return impl_->active_cores; }

MeshDevice* HostMeshSocket::get_mesh_device() const { return impl_->mesh_device; }

bool HostMeshSocket::poll() {
    if (!impl_->participates) {
        return false;
    }
    // A relay endpoint is single-threaded by construction: it owns a queue pair
    // whose completion queue must be drained by one consumer. Polling here while
    // the relay thread polls the same endpoint double-consumes completions and
    // corrupts the doorbell and credit accounting.
    TT_FATAL(
        !impl_->transport.own_relay_thread,
        "HostMeshSocket::poll() is only for a socket built with own_relay_thread = false; the relay thread is "
        "already polling this one");
    return impl_->poll();
}

void HostMeshSocket::set_latency_sampling(bool enabled) {
    for (auto& connection : impl_->connections) {
        if (auto* sender = dynamic_cast<RelaySender*>(connection.relay.get())) {
            sender->set_credit_latency_sampling(enabled);
        }
    }
}

std::vector<uint64_t> HostMeshSocket::take_latency_samples_ns() {
    std::vector<uint64_t> all;
    for (auto& connection : impl_->connections) {
        if (auto* sender = dynamic_cast<RelaySender*>(connection.relay.get())) {
            auto samples = sender->take_credit_latencies_ns();
            all.insert(all.end(), samples.begin(), samples.end());
        }
    }
    return all;
}

uint64_t HostMeshSocket::pages_transferred() const {
    uint64_t total = 0;
    for (const auto& connection : impl_->connections) {
        if (auto* sender = dynamic_cast<const RelaySender*>(connection.relay.get())) {
            total += sender->pages_forwarded();
        } else if (auto* receiver = dynamic_cast<const RelayReceiver*>(connection.relay.get())) {
            total += receiver->pages_arrived();
        }
    }
    return total;
}

void HostMeshSocket::barrier(std::optional<uint32_t> timeout_ms) {
    if (!impl_->participates) {
        return;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms.value_or(30000));
    const bool drive_here = !impl_->transport.own_relay_thread;

    // Snapshot what is outstanding now and wait for exactly that to drain. The
    // peer pipelines ahead, so anything that arrives after this point is not
    // ours to wait for -- waiting for the endpoint to fall idle would never
    // finish on a socket that is still being fed.
    std::vector<uint64_t> targets;
    targets.reserve(impl_->connections.size());
    for (const auto& connection : impl_->connections) {
        targets.push_back(connection.relay->watermark());
    }

    while (true) {
        if (drive_here) {
            impl_->poll();
        }
        bool done = true;
        for (size_t i = 0; i < impl_->connections.size(); i++) {
            const auto& relay = impl_->connections[i].relay;
            // An endpoint that threw was stopped by the relay rather than taking
            // the process down, so surface it here instead of spinning out.
            TT_FATAL(!relay->failed(), "HostMeshSocket relay failed: {}", relay->error());
            done = done && relay->drained(targets[i]);
        }
        if (done) {
            return;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            std::string state;
            for (size_t i = 0; i < impl_->connections.size(); i++) {
                state += fmt::format("\n  want {} drained: {}", targets[i], impl_->connections[i].relay->describe());
            }
            TT_THROW(
                "HostMeshSocket::barrier timed out as {}:{}",
                impl_->endpoint == SocketEndpoint::SENDER ? "sender" : "receiver",
                state);
        }
        std::this_thread::yield();
    }
}

}  // namespace tt::tt_metal::distributed
