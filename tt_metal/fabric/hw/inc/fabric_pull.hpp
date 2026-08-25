// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)

#include <cstdint>
#include <array>
#include <limits>
#include <type_traits>

#include "api/dataflow/fabric_dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/debug/assert.h"
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

namespace fabric_api = tt::tt_fabric::linear::experimental;

// Routes. A multicast cannot fork -- the router walks a positional per-hop
// stream with no branch offsets -- so covering every peer takes one route, and
// one packet header, per direction.

// fabric_max_routes and FabricPullRequestSet live in fabric_edm_types.hpp: the
// host needs both, to size the request scratchpad and the sender's runtime args.

#if defined(FABRIC_2D)
using FabricRange = tt::tt_fabric::mesh::experimental::MeshMcastRange;
inline FabricRange make_fabric_range(uint8_t e, uint8_t w, uint8_t n, uint8_t s) { return FabricRange{e, w, n, s}; }
inline constexpr bool kFabricIs2D = true;
#else
// 1D carries a single hop count; exactly one slot is ever nonzero.
using FabricRange = uint8_t;
inline FabricRange make_fabric_range(uint8_t e, uint8_t w, uint8_t n, uint8_t s) {
    return static_cast<uint8_t>(e + w + n + s);
}
inline constexpr bool kFabricIs2D = false;
#endif

// Three route shapes. Which one a topology needs follows from how it delivers:
// a line walks hops, a mesh walks a rectangle, and a star walks nothing.
struct LineRoute {
    FabricRange range;
    // eth_chan_directions. Selects the outgoing link: with no connection to
    // pick, this is how the worker names the port.
    uint8_t direction;
};

struct RectRoute {
    FabricRange range;
    // Anchor the E/W/N/S extents are measured from -- becomes
    // packet_header->dst_start_node_id, i.e. the chip this route's first hop
    // lands on, not any final destination.
    uint16_t dst_dev_id;
    uint16_t dst_mesh_id;
    uint8_t direction;
};

struct PeerRoute {
    // Bit i selects fabric node i, not target queue i -- the DE translates. No
    // range or direction: nothing forwards, so every peer is one hop.
    uint32_t peer_mask;
};

// How many peers a mask names. This is the source-read count where nothing
// forwards, since each named peer pulls the page from our L1 itself. Called
// once per set-state, not per send.
constexpr uint8_t peers_count(uint32_t peer_mask) { return static_cast<uint8_t>(__builtin_popcount(peer_mask)); }

template <tt::tt_fabric::Topology topology>
using McastRoute = std::conditional_t<
    !tt::tt_fabric::is_forwarding_topology(topology),
    PeerRoute,
    std::conditional_t<tt::tt_fabric::is_2D_topology(topology), RectRoute, LineRoute>>;

template <tt::tt_fabric::Topology topology>
struct FabricMcastRouteArgs {
    static_assert(
        tt::tt_fabric::is_2D_topology(topology) == kFabricIs2D,
        "topology template argument must agree with the FABRIC_2D build define");

    std::array<McastRoute<topology>, tt::tt_fabric::fabric_max_routes<topology>> routes = {};
    uint8_t num_routes = 0;

    // Chip multicast excludes the sender in every direction. An all-gather
    // wants its own chunk in its own replica; a broadcast whose source keeps
    // nothing does not. Served by a plain local NoC write, generating no SWQ.
    bool include_self = false;
};

// A pointer to the caller's request set, sized for this topology. The set lives
// in the worker's scratchpad, so the worker owns the storage and its lifetime.
template <tt::tt_fabric::Topology topology>
using FabricRequestRef =
    volatile tt::tt_fabric::FabricPullRequestSet<PACKET_HEADER_TYPE, tt::tt_fabric::fabric_max_routes<topology>>*;

// A header-only request makes the local DE copy one payload page out of worker
// L1 and forward the normal header+payload packet. Nothing is opened: link
// parameters come from the connection table at
// MEM_TENSIX_FABRIC_CONNECTIONS_BASE, indexed by the route's direction.
class Fabric {
public:
    Fabric() = default;
    Fabric(const Fabric&) = delete;
    Fabric& operator=(const Fabric&) = delete;
    Fabric(Fabric&&) = delete;
    Fabric& operator=(Fabric&&) = delete;

    // ---- multicast write ----

    // Fills one slot per route with its own range and anchor. The slot count is
    // M: the router issues one packet per direction and the chain
    // store-and-forwards it, so the sender's L1 is read once per direction
    // whatever the hop counts.
    template <tt::tt_fabric::Topology topology>
    void set_async_write_multicast_state(
        FabricRequestRef<topology> request, const FabricMcastRouteArgs<topology>& route) {
        ASSERT(route.num_routes > 0 && route.num_routes <= tt::tt_fabric::fabric_max_routes<topology>);

        request->used = route.num_routes;
        request->source_read_completions = source_read_completions(route);
        request->include_self = route.include_self ? 1u : 0u;
        for (uint8_t r = 0; r < route.num_routes; ++r) {
            auto* packet_header = &request->routes[r];
            set_mcast_range<topology>(packet_header, route.routes[r]);
            packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_WRITE;
            request->direction[r] = route.routes[r].direction;
        }
    }

    // size_bytes is the caller's: a partially filled entry sends only what it
    // holds. with_state patches PayloadSize on every call, so a short send
    // costs nothing extra.
    template <uint32_t MaxRoutes>
    void async_write_multicast_with_state(
        volatile tt::tt_fabric::FabricPullRequestSet<PACKET_HEADER_TYPE, MaxRoutes>* request,
        FabricDataflowBuffer& payload,
        uint64_t remote_noc_address,
        uint32_t size_bytes) {
        ASSERT(request->used > 0);

        // One transaction covering every route: the DFB read pointer advances
        // once, so the whole multicast costs one entry, not one per direction.
        auto transaction = prepare_transaction(payload, request->source_read_completions);

        // Chip multicast excludes the source chip, so the sender's own replica
        // needs an explicit local write. Same size as the packet, or a short
        // send would leave a garbage tail. Not counted in M: it generates no
        // SWQ, so no transaction-counter decrement is owed for it.
        if (request->include_self) {
            noc_async_write<NOC_MAX_BURST_SIZE + 1, true, false>(
                transaction.source_l1_address, remote_noc_address, size_bytes, noc_id_, NOC_UNICAST_WRITE_VC + 1);
        }

        for (uint8_t r = 0; r < request->used; ++r) {
            auto& sender = sender_for(request->direction[r]);
            validate_payload_fits(sender, payload, size_bytes);
            fabric_api::fabric_pull_multicast_noc_unicast_write_with_state<
                UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
                &sender,
                &request->routes[r],
                transaction.source_l1_address,
                transaction.transaction_id,
                tt::tt_fabric::NocUnicastCommandHeader{remote_noc_address},
                size_bytes);
        }
        // Flushes the header publishes and the include_self local write alike:
        // it waits per-NoC, not per-VC, so the local copy has finished reading
        // the entry before commit_transaction() advances past it.
        finish_header_publication();

        payload.commit_transaction();
        payload.try_complete_front_transaction();
    }

    // ---- multicast atomic inc ----

    // Header-only, so it claims no transaction id. Publishing it behind the
    // data requests is what orders it: the DE drains its queue in order.
    template <tt::tt_fabric::Topology topology>
    void set_atomic_inc_multicast_state(
        FabricRequestRef<topology> request, const FabricMcastRouteArgs<topology>& route) {
        ASSERT(route.num_routes > 0 && route.num_routes <= tt::tt_fabric::fabric_max_routes<topology>);

        request->used = route.num_routes;
        request->source_read_completions = 0;  // header-only: no payload is pulled
        request->include_self = route.include_self ? 1u : 0u;
        for (uint8_t r = 0; r < route.num_routes; ++r) {
            auto* packet_header = &request->routes[r];
            set_mcast_range<topology>(packet_header, route.routes[r]);
            // Header-only: size_bytes stays 0, which is what tells the
            // router there is nothing to fetch.
            packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_ATOMIC_INC;
            request->direction[r] = route.routes[r].direction;
        }
    }

    // `flush` makes each destination drain its preceding NoC writes before the
    // increment lands. Exposed, not assumed: it costs the receiver that drain.
    template <uint32_t MaxRoutes>
    void atomic_inc_multicast_with_state(
        volatile tt::tt_fabric::FabricPullRequestSet<PACKET_HEADER_TYPE, MaxRoutes>* request,
        uint64_t remote_noc_address,
        uint32_t value,
        bool flush) {
        ASSERT(request->used > 0);

        // Chip multicast excludes the source, so our own semaphore is bumped
        // here. The payload path's include_self is a local copy of the entry;
        // the analogue for a header-only atomic is a local increment.
        if (request->include_self) {
            noc_semaphore_inc(remote_noc_address, value, noc_id_);
        }

        for (uint8_t r = 0; r < request->used; ++r) {
            auto& sender = sender_for(request->direction[r]);
            fabric_api::fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::All>(
                &sender,
                &request->routes[r],
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{remote_noc_address, value, flush});
        }
        finish_header_publication();
    }

private:
    struct PreparedTransaction {
        uint32_t source_l1_address;
        uint32_t transaction_id;
    };

    // Reads this send will owe. A chain amortises -- one read per route serves
    // every peer behind it -- but without one, every named peer reads our L1.
    template <tt::tt_fabric::Topology topology>
    static uint8_t source_read_completions(const FabricMcastRouteArgs<topology>& route) {
        if constexpr (tt::tt_fabric::is_forwarding_topology(topology)) {
            return route.num_routes;
        } else {
            return peers_count(route.routes[0].peer_mask);
        }
    }

    static constexpr uint8_t kMaxDirections = 4;  // eth_chan_directions E/W/N/S
    static constexpr uint8_t kNoChannel = 0xff;

    template <tt::tt_fabric::Topology topology>
    static void set_mcast_range(volatile PACKET_HEADER_TYPE* packet_header, const McastRoute<topology>& route) {
        if constexpr (tt::tt_fabric::is_2D_topology(topology)) {
            fabric_set_mcast_route(
                packet_header,
                route.dst_dev_id,
                route.dst_mesh_id,
                route.range.e,
                route.range.w,
                route.range.n,
                route.range.s);
        } else if constexpr (tt::tt_fabric::is_forwarding_topology(topology)) {
            // Both directions start at distance 1, so start_distance is always
            // 1 and each route carries its own terminator.
            packet_header->to_chip_multicast(
                tt::tt_fabric::MulticastRoutingCommandHeader{/*start_distance=*/1, route.range});
        } else {
            // Nothing forwards, so there are no hops to encode and the routing
            // dword is free to carry the mask itself. The DE turns each set bit
            // into one SWQ on that peer's target queue.
            packet_header->to_chip_peer_multicast(route.peer_mask);
        }
    }

    // The link in `direction`, built once and cached. Nothing is opened, so
    // there is no handshake state to keep and no close.
    tt::tt_fabric::WorkerToFabricEdmSender& sender_for(uint8_t direction) {
        ASSERT(direction < kMaxDirections);
        if (!sender_valid_[direction]) {
            build_sender(direction);
            sender_valid_[direction] = true;
        }
        return senders_[direction];
    }

    void build_sender(uint8_t direction) {
        auto* connection_info =
            reinterpret_cast<tt_l1_ptr tensix_fabric_connections_l1_info_t*>(MEM_TENSIX_FABRIC_CONNECTIONS_BASE);

        // Indexed by eth channel, so find the valid one facing this direction.
        // Production takes the channel from a runtime arg; with no connection to
        // choose, the route names the direction and the channel follows.
        uint8_t eth_channel = kNoChannel;
        for (uint8_t ch = 0; ch < tensix_fabric_connections_l1_info_t::MAX_FABRIC_ENDPOINTS; ++ch) {
            if (((connection_info->valid_connections_mask >> ch) & 0x1u) == 0) {
                continue;
            }
            if (connection_info->read_only[ch].edm_direction == direction) {
                eth_channel = ch;
                break;
            }
        }
        ASSERT(eth_channel != kNoChannel);

        const auto* conn = &connection_info->read_only[eth_channel];
        auto* aligned_conn = &connection_info->read_write[eth_channel];

        // The handshake addresses are only touched by open/close, which the
        // pull path never calls, so they reuse the flow-control word rather
        // than worker semaphores the caller would otherwise have to allocate.
        auto* flow_control_sem = reinterpret_cast<volatile uint32_t*>(
            reinterpret_cast<uintptr_t>(&aligned_conn->worker_flow_control_semaphore));

        senders_[direction].init<ProgrammableCoreType::TENSIX>(
            /*connected_to_persistent_fabric=*/true,
            conn->edm_noc_x,
            conn->edm_noc_y,
            conn->edm_buffer_base_addr,
            conn->num_buffers_per_channel,
            conn->edm_connection_handshake_addr,
            conn->edm_worker_location_info_addr,
            conn->buffer_size_bytes,
            conn->buffer_index_semaphore_id,
            flow_control_sem,
            /*worker_teardown_addr=*/flow_control_sem,
            /*local_buffer_index_addr=*/conn->buffer_index_semaphore_id,
            static_cast<uint32_t>(conn->worker_free_slots_stream_id),
            StreamId{std::numeric_limits<uint32_t>::max()});
    }

    PreparedTransaction prepare_transaction(FabricDataflowBuffer& payload, uint32_t source_read_completion_count) {
        payload.wait_for_transaction_id();
        payload.wait_for_next_issue();

        const uint32_t source_l1_address = payload.get_next_issue_read_ptr();
        // One terminal source-read completion per route: each route generates
        // exactly one SWQ, whatever its hop count. That identity holds only
        // while a route is one packet -- scatter and multi-packet chunks would
        // break it, which is why this is a count and not the route count.
        const uint32_t transaction_id = payload.prepare_transaction(source_read_completion_count);
        return {
            .source_l1_address = source_l1_address,
            .transaction_id = transaction_id,
        };
    }

    template <typename Sender>
    static void validate_payload_fits(const Sender& sender, const FabricDataflowBuffer& payload, uint32_t size_bytes) {
        // A short send is fine; sending more than the entry holds is not.
        ASSERT(size_bytes > 0 && size_bytes <= payload.get_entry_size());
        ASSERT(size_bytes + sizeof(PACKET_HEADER_TYPE) <= sender.buffer_size_bytes);
    }

    void finish_header_publication() const {
        // Every stateful call reuses its request slots. Wait until the NoC has
        // copied them into the DE channel before modifying them again.
        noc_async_writes_flushed(noc_id_);
    }

    uint8_t noc_id_ = tt::tt_fabric::get_fabric_worker_noc();

    tt::tt_fabric::WorkerToFabricEdmSender senders_[kMaxDirections] = {};
    bool sender_valid_[kMaxDirections] = {};
};

#endif  // ARCH_QUASAR && !COMPILE_FOR_TRISC
