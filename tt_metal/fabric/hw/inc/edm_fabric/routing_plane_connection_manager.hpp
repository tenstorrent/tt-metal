// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include "dataflow_api.h"
#include "edm_fabric_worker_adapters.hpp"

namespace tt::tt_fabric {

// Determine maximum number of routing-plane connections if not provided by the build.
#ifndef TT_FABRIC_MAX_ROUTING_PLANE_CONNECTIONS
#if defined(FABRIC_2D)
#define TT_FABRIC_MAX_ROUTING_PLANE_CONNECTIONS 4
#else  // 1D
#define TT_FABRIC_MAX_ROUTING_PLANE_CONNECTIONS 2
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

    // These field for FABRIC_2D are used by fabric_set_unicast_route
#if defined(FABRIC_2D)
    uint32_t ew_dim;
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

    template <BuildFromArgsMode build_mode = BuildFromArgsMode::BUILD_ONLY>
    static RoutingPlaneConnectionManager build_from_args(std::size_t& arg_idx, uint32_t num_connections_to_build) {
        constexpr bool connect = build_mode == BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION ||
                                 build_mode == BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY;
        constexpr bool wait_for_connection_open_finish = build_mode == BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION;

        RoutingPlaneConnectionManager mgr;
        ASSERT(num_connections_to_build <= MaxConnections);

        for (uint32_t i = 0; i < num_connections_to_build; ++i) {
            auto& conn = mgr.slots_[i];
            conn.tag = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
            conn.sender =
                tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
            if constexpr (connect) {
                conn.sender.open_start();
            }
        }

        mgr.num_active_ = num_connections_to_build;

        if constexpr (connect && wait_for_connection_open_finish) {
            for (uint32_t i = 0; i < mgr.num_active_; ++i) {
                mgr.slots_[i].sender.open_finish();
            }
        }

#if defined(FABRIC_2D)
        mgr.ew_dim = get_arg_val<uint32_t>(arg_idx++);
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

    template <typename Fn>
    inline void for_each_with_tag(uint32_t tag, Fn&& fn) {
        for (uint32_t i = 0; i < num_active_; ++i) {
            if (slots_[i].tag == tag) {
                fn(slots_[i].sender, i, slots_[i].tag);
            }
        }
    }

    template <bool SEND_CREDIT_ADDR = false>
    inline void open_start() {
        for_each([&](Sender& s, uint32_t, uint32_t) { s.open_start<SEND_CREDIT_ADDR>(); });
    }

    inline void open_finish() {
        for_each([&](Sender& s, uint32_t, uint32_t) { s.open_finish(); });
    }

    template <bool SEND_CREDIT_ADDR = false>
    inline void open() {
        open_start<SEND_CREDIT_ADDR>();
        open_finish();
    }

    inline void close_start() {
        for_each([&](Sender& s, uint32_t, uint32_t) { s.close_start(); });
    }

    inline void close_finish() {
        for_each([&](Sender& s, uint32_t, uint32_t) { s.close_finish(); });
    }

    inline void close() {
        close_start();
        close_finish();
    }

    inline uint32_t active_count() const { return num_active_; }

private:
    std::array<ConnectionSlot, MaxConnections> slots_{};
    uint32_t num_active_;
};

}  // namespace tt::tt_fabric
