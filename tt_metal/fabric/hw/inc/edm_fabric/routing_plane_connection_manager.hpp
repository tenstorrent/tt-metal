// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

namespace tt::tt_fabric {

// Determine maximum number of routing-plane connections if not provided by the build.
#ifndef TT_FABRIC_MAX_ROUTING_PLANE_CONNECTIONS
// The manager holds at most one logical connection per output direction. When any local Blackhole
// mesh uses express routing, all local workers compile for the five-direction superset; a worker on
// a non-express mesh simply leaves the Z slot unused. Parallel lanes to one neighbor are routing-plane
// realizations, not additional logical connections.
#if defined(FABRIC_EXPRESS_ENABLED) && defined(ARCH_BLACKHOLE)
#define TT_FABRIC_MAX_ROUTING_PLANE_CONNECTIONS 5
#else
#define TT_FABRIC_MAX_ROUTING_PLANE_CONNECTIONS 4
// TODO: 3D, dragonfly and custom etc.
#endif
#endif

// Logical connection manager across routing planes. Capacity is fixed by TT_FABRIC_MAX_ROUTING_PLANE_CONNECTIONS.
// This is V2 of FabricConnectionManager.
class RoutingPlaneConnectionManager final {
public:
    static constexpr std::size_t MaxConnections = TT_FABRIC_MAX_ROUTING_PLANE_CONNECTIONS;
    using Sender = tt::tt_fabric::WorkerToFabricEdmSender;

    enum BuildFromArgsMode : uint8_t {
        BUILD_ONLY,
        BUILD_AND_OPEN_CONNECTION,
        BUILD_AND_OPEN_CONNECTION_START_ONLY,
    };

#if defined(FABRIC_2D)
    uint16_t my_mesh_id;
    uint16_t my_chip_id;
#endif

    struct ConnectionSlot {
        Sender sender;
        uint8_t tag;
#ifdef FABRIC_2D
        uint16_t dst_dev_id;
        uint16_t dst_mesh_id;
#endif
    };

    RoutingPlaneConnectionManager() : num_active_(0) {}

    template <
        BuildFromArgsMode build_mode = BuildFromArgsMode::BUILD_ONLY,
        uint8_t WORKER_HANDSHAKE_NOC = get_fabric_worker_noc(),
        bool sem_args_are_l1_addresses = false>
    static RoutingPlaneConnectionManager build_from_args(std::size_t& arg_idx, uint32_t num_connections_to_build) {
        RoutingPlaneConnectionManager mgr;
        ASSERT(num_connections_to_build <= MaxConnections);

        for (uint32_t i = 0; i < num_connections_to_build; ++i) {
            auto& conn = mgr.slots_[i];
            conn.tag = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
            conn.sender = tt::tt_fabric::WorkerToFabricEdmSender::
                build_from_args<ProgrammableCoreType::TENSIX, sem_args_are_l1_addresses>(arg_idx);
            start_open<build_mode, WORKER_HANDSHAKE_NOC>(conn.sender);
        }
        mgr.num_active_ = num_connections_to_build;
        mgr.finish_open<build_mode, WORKER_HANDSHAKE_NOC>();

#if defined(FABRIC_2D)
        mgr.my_chip_id = get_arg_val<uint32_t>(arg_idx++);
        mgr.my_mesh_id = get_arg_val<uint32_t>(arg_idx++);
        for (uint32_t i = 0; i < num_connections_to_build; i++) {
            auto& conn = mgr.slots_[i];
            conn.dst_dev_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_idx++));
            conn.dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_idx++));
        }
#endif

        return mgr;
    }

    // Builds a manager from a Metal 2.0 fabric connection binding: one slot for each target, in target order. A slot's
    // tag is the direction of its router; on a 2D fabric, its destination is the binding's destination for that
    // target.
    // precondition: the binding has at most MaxConnections targets, and they resolve to different directions.
    template <
        BuildFromArgsMode build_mode = BuildFromArgsMode::BUILD_ONLY,
        uint8_t WORKER_HANDSHAKE_NOC = get_fabric_worker_noc()>
    static RoutingPlaneConnectionManager build_from_binding(FabricConnectionBindingToken token) {
        ASSERT(token.num_targets() <= MaxConnections);
        RoutingPlaneConnectionManager mgr;
        for (uint32_t i = 0; i < token.num_targets(); ++i) {
            const uint32_t target_word = fabric_connection_binding::target_rta_word(token, i);
            auto& conn = mgr.slots_[mgr.num_active_++];
            conn.tag = get_router_direction(get_arg_val<uint32_t>(target_word));
            std::size_t arg_idx = target_word;
            conn.sender =
                tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
            start_open<build_mode, WORKER_HANDSHAKE_NOC>(conn.sender);
#if defined(FABRIC_2D)
            conn.dst_dev_id = static_cast<uint16_t>(get_arg_val<uint32_t>(target_word + 3));
            conn.dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(target_word + 4));
#endif
        }
        mgr.finish_open<build_mode, WORKER_HANDSHAKE_NOC>();

#if defined(FABRIC_2D)
        const auto* routing_table = reinterpret_cast<tt_l1_ptr routing_l1_info_t*>(ROUTING_TABLE_BASE);
        mgr.my_chip_id = routing_table->my_device_id;
        mgr.my_mesh_id = routing_table->my_mesh_id;
#endif
        return mgr;
    }

    inline uint32_t get_tag(uint32_t index) const {
        ASSERT(index < num_active_);
        return slots_[index].tag;
    }

    inline ConnectionSlot& get(uint32_t index) {
        ASSERT(index < num_active_);
        return slots_[index];
    }
    inline const ConnectionSlot& get(uint32_t index) const {
        ASSERT(index < num_active_);
        return slots_[index];
    }

    template <typename Fn>
    inline void for_each(Fn&& fn) {
        for (uint32_t i = 0; i < num_active_; ++i) {
            fn(slots_[i].sender, i, slots_[i].tag);
        }
    }

    template <bool SEND_CREDIT_ADDR = false, uint8_t WORKER_HANDSHAKE_NOC = get_fabric_worker_noc()>
    inline void open_start() {
        for_each([&](Sender& s, uint32_t, uint32_t) { s.open_start<SEND_CREDIT_ADDR, false, WORKER_HANDSHAKE_NOC>(); });
    }

    template <uint8_t WORKER_HANDSHAKE_NOC = get_fabric_worker_noc()>
    inline void open_finish() {
        for_each([&](Sender& s, uint32_t, uint32_t) { s.open_finish<false, WORKER_HANDSHAKE_NOC>(); });
    }

    template <bool SEND_CREDIT_ADDR = false, uint8_t WORKER_HANDSHAKE_NOC = get_fabric_worker_noc()>
    inline void open() {
        open_start<SEND_CREDIT_ADDR, WORKER_HANDSHAKE_NOC>();
        open_finish<WORKER_HANDSHAKE_NOC>();
    }

    template <uint8_t WORKER_HANDSHAKE_NOC = get_fabric_worker_noc()>
    inline void close_start() {
        for_each([&](Sender& s, uint32_t, uint32_t) { s.close_start<false, WORKER_HANDSHAKE_NOC>(); });
    }

    template <uint8_t WORKER_HANDSHAKE_NOC = get_fabric_worker_noc()>
    inline void close_finish() {
        for_each([&](Sender& s, uint32_t, uint32_t) { s.close_finish<false, WORKER_HANDSHAKE_NOC>(); });
    }

    template <uint8_t WORKER_HANDSHAKE_NOC = get_fabric_worker_noc()>
    inline void close() {
        close_start<WORKER_HANDSHAKE_NOC>();
        close_finish<WORKER_HANDSHAKE_NOC>();
    }

    inline uint32_t active_count() const { return num_active_; }

private:
    // Sends the open request of a just-built connection if `build_mode` opens connections.
    template <BuildFromArgsMode build_mode, uint8_t WORKER_HANDSHAKE_NOC>
    static void start_open(Sender& sender) {
        if constexpr (
            build_mode == BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION ||
            build_mode == BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY) {
            sender.template open_start<false, false, WORKER_HANDSHAKE_NOC>();
        }
    }

    // Waits for the open requests of the active connections if `build_mode` waits for them.
    template <BuildFromArgsMode build_mode, uint8_t WORKER_HANDSHAKE_NOC>
    void finish_open() {
        if constexpr (build_mode == BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION) {
            open_finish<WORKER_HANDSHAKE_NOC>();
        }
    }

    std::array<ConnectionSlot, MaxConnections> slots_{};
    uint32_t num_active_;
};

}  // namespace tt::tt_fabric
