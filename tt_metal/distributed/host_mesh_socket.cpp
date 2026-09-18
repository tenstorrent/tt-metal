// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/sockets/host_mesh_socket.hpp>

#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "tt_metal/distributed/host_transport/host_transport.hpp"
#include "tt_metal/distributed/host_transport/socket_relay.hpp"
#include "tt_metal/distributed/mesh_socket_utils.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>

namespace tt::tt_metal::distributed {

using host_transport::RelayLoop;
using host_transport::RelayReceiver;
using host_transport::RelaySender;
using host_transport::RingGeometry;

namespace {

// One tag per socket instance, so two sockets between the same rank pair cannot
// mix up each other's endpoints. Assumes both ranks construct in the same order,
// as MeshSocket's exchange-tag counter also does.
constexpr uint32_t kEndpointExchangeTagBase = 0x7248;

// Reserves a contiguous run, since each connection takes kTagsPerConnection of
// them. Both ends allocate in the same construction order, so both get the same
// base without exchanging it.
multihost::Tag reserve_exchange_tags(size_t connections) {
    static std::atomic<uint32_t> next{0};
    const uint32_t width = static_cast<uint32_t>(connections) * host_transport::kTagsPerConnection;
    return multihost::Tag{static_cast<int>(kEndpointExchangeTagBase + next.fetch_add(width))};
}

}  // namespace

struct HostMeshSocket::Impl {
    // Declaration order sets teardown order and is load-bearing: stop polling,
    // tear the transport down, then free the rings it was reading from.
    struct Connection {
        std::unique_ptr<D2HSocket> d2h;  // sender endpoints only; owns its ring
        std::unique_ptr<H2DSocket> h2d;  // receiver endpoints only; owns its ring
        std::unique_ptr<host_transport::HostTransport> transport;
        std::shared_ptr<host_transport::RelayEndpoint> relay;
    };

    SocketConfig config;
    TransportConfig transport;
    SocketEndpoint endpoint = SocketEndpoint::SENDER;
    MeshDevice* mesh_device = nullptr;
    std::shared_ptr<MeshBuffer> config_buffer;
    std::shared_ptr<MeshBuffer> data_buffer;  // receiver endpoints only
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

    // One page per core, so every core finds its metadata at the same L1 address
    // and a multi-core kernel needs only one.
    impl_->config_buffer = create_socket_config_buffer(device, config, impl_->endpoint);
    impl_->config_buffer_address = impl_->config_buffer->address();

    // Same allocation D2D makes, so a receiver kernel has a socket-owned landing
    // buffer to pull into and get_data_buffer() means the same thing on both.
    if (impl_->endpoint == SocketEndpoint::RECEIVER) {
        impl_->data_buffer = create_socket_data_buffer(device, config);
    }

    const bool is_sender = impl_->endpoint == SocketEndpoint::SENDER;
    const auto exchange_tag = reserve_exchange_tags(config.socket_connection_config.size());

    impl_->connections.resize(config.socket_connection_config.size());
    for (size_t i = 0; i < config.socket_connection_config.size(); i++) {
        const auto& wire = config.socket_connection_config[i];
        auto& connection = impl_->connections[i];

        host_transport::TransportParams tp;
        tp.geometry = impl_->geometry;
        tp.is_sender = is_sender;
        tp.peer_rank = is_sender ? *config.receiver_rank : *config.sender_rank;
        tp.context = context;
        tp.tag_base = *exchange_tag + static_cast<int>(i) * host_transport::kTagsPerConnection;
        tp.max_batch_pages = transport.max_batch_pages;

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
            tp.ring = connection.d2h->host_fifo().data();
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
            tp.ring = connection.h2d->host_fifo().data();
        }

        connection.transport = host_transport::make_host_transport(tp);

        if (is_sender) {
            connection.relay = std::make_shared<RelaySender>(
                *connection.transport, *connection.d2h, impl_->geometry, transport.max_batch_pages);
        } else {
            connection.relay = std::make_shared<RelayReceiver>(*connection.transport, *connection.h2d, impl_->geometry);
        }
    }

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
    // Unlike D2D this is not the FIFO -- that is the pinned host ring the NIC
    // reaches -- but it is the same size and shape, so a receiver kernel pulls
    // into it exactly where a D2D kernel would find its pages.
    TT_FATAL(impl_->data_buffer, "Cannot access the data buffer for a sender socket.");
    return impl_->data_buffer;
}

const SocketConfig& HostMeshSocket::get_config() const { return impl_->config; }

SocketEndpoint HostMeshSocket::get_socket_endpoint_type() const { return impl_->endpoint; }

std::vector<MeshCoreCoord> HostMeshSocket::get_active_cores() const { return impl_->active_cores; }

MeshDevice* HostMeshSocket::get_mesh_device() const { return impl_->mesh_device; }

bool HostMeshSocket::poll() {
    if (!impl_->participates) {
        return false;
    }
    // An endpoint's completion queue must be drained by one consumer. Polling
    // here while the relay thread polls it double-consumes completions and
    // corrupts the arrival and credit accounting.
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

    // Wait only for what is outstanding now. The peer pipelines ahead, so waiting
    // for full idle would never finish on a socket still being fed.
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
            // Surface a stopped endpoint instead of spinning to the deadline.
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
