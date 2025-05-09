// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "edm_fabric_worker_adapters.hpp"

class FabricConnectionManager final {
public:
    // return if there is/should be a connection - doesn't return whether or not the connection
    // is actually live
    inline bool is_logically_connected() const { return has_forward_connection() || has_backward_connection(); }

    // make the connection live
    inline void open() {
        DPRINT << "calling fabric connection open\n";
        if (has_forward_connection()) {
            forward_fabric_sender.open_start();
        }
        if (has_backward_connection()) {
            backward_fabric_sender.open_start();
        }
        if (has_forward_connection()) {
            forward_fabric_sender.open_finish();
        }
        if (has_backward_connection()) {
            backward_fabric_sender.open_finish();
        }
        DPRINT << "\tFinished fabric connection open\n";
    }
    inline bool has_forward_connection() const { return connection_flags & FORWARD_CONNECTION_FLAG_MASK; }
    inline bool has_backward_connection() const { return connection_flags & BACKWARD_CONNECTION_FLAG_MASK; }

    // Advanced usage API:
    // Expose a separate close_start() and close_finish() to allow the user to opt-in to a 2 step close
    // where the close_start() is sends the close request to the fabric and the close_finish() waits for
    // the ack from fabric.
    inline void close_start() {
        DPRINT << "calling fabric connection close_start\n";
        if (has_forward_connection()) {
            forward_fabric_sender.close_start();
        }
        if (has_backward_connection()) {
            backward_fabric_sender.close_start();
        }
        DPRINT << "\tFinished fabric connection close_start\n";
    }
    inline void close_finish() {
        DPRINT << "calling fabric connection close_finish\n";
        if (has_forward_connection()) {
            forward_fabric_sender.close_finish();
        }
        if (has_backward_connection()) {
            backward_fabric_sender.close_finish();
        }
        DPRINT << "\tFinished fabric connection close_finish\n";
    }
    inline void close() {
        DPRINT << "calling fabric connection close\n";
        close_start();
        close_finish();
        DPRINT << "\tFinished fabric connection close\n";
    }

    enum BuildFromArgsMode : uint8_t { BUILD_ONLY, BUILD_AND_OPEN_CONNECTION, BUILD_AND_OPEN_CONNECTION_START_ONLY };

    // Advanced usage API: build_mode
    // Allow the user to opt-in to a 3 build modes:
    //
    // BUILD_ONLY: just build the connection manager but don't open a connection
    //
    // BUILD_AND_OPEN_CONNECTION: build the connection manager and open a connection, wait for connection to be fully
    //         open and established before returning
    //
    // BUILD_AND_OPEN_CONNECTION_START_ONLY: build the connection manager and send the connection open request to
    //         fabric but don't wait for the connection readback to complete before returning.
    //         !!! IMPORTANT !!!
    //         User must call open_finish() manually, later, if they use this mode.
    template <BuildFromArgsMode build_mode = BuildFromArgsMode::BUILD_ONLY>
    static FabricConnectionManager build_from_args(std::size_t& arg_idx) {
        constexpr bool connect = build_mode == BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION ||
                                 build_mode == BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY;
        constexpr bool wait_for_connection_open_finish = build_mode == BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION;
        FabricConnectionManager connection_manager;
        auto fwd_connected = get_arg_val<uint32_t>(arg_idx++);
        // ASSERT(fwd_connected | bwd_connected <= 1);
        connection_manager.connection_flags = (fwd_connected != 0) << FORWARD_CONNECTION_FLAG_OFFSET;
        if (connection_manager.has_forward_connection()) {
            connection_manager.forward_fabric_sender =
                tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
            if constexpr (connect) {
                connection_manager.forward_fabric_sender.open_start();
            }
        }
        auto bwd_connected = get_arg_val<uint32_t>(arg_idx++);
        connection_manager.connection_flags |= static_cast<uint8_t>(bwd_connected != 0)
                                               << BACKWARD_CONNECTION_FLAG_OFFSET;
        if (connection_manager.has_backward_connection()) {
            connection_manager.backward_fabric_sender =
                tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
            if constexpr (connect) {
                connection_manager.backward_fabric_sender.open_start();
            }
        }

        if constexpr (connect && wait_for_connection_open_finish) {
            if (connection_manager.has_forward_connection()) {
                connection_manager.forward_fabric_sender.open_finish();
            }
            if (connection_manager.has_backward_connection()) {
                connection_manager.backward_fabric_sender.open_finish();
            }
        }
        return connection_manager;
    }

    inline void open_finish() {
        if (has_forward_connection()) {
            forward_fabric_sender.open_finish();
        }
        if (has_backward_connection()) {
            backward_fabric_sender.open_finish();
        }
    }

    tt::tt_fabric::WorkerToFabricEdmSender& get_forward_connection() {
        WAYPOINT("FWDC");
        ASSERT(has_forward_connection());
        while (!has_forward_connection());
        WAYPOINT("fwdc");
        return forward_fabric_sender;
    }
    tt::tt_fabric::WorkerToFabricEdmSender& get_backward_connection() {
        WAYPOINT("BWDC");
        ASSERT(has_backward_connection());
        while (!has_backward_connection());
        WAYPOINT("bwdc");
        return backward_fabric_sender;
    }

private:
    static constexpr uint8_t FORWARD_CONNECTION_FLAG_MASK = 0x01;
    static constexpr uint8_t BACKWARD_CONNECTION_FLAG_MASK = 0x02;
    static constexpr uint8_t FORWARD_CONNECTION_FLAG_OFFSET = 0x0;
    static constexpr uint8_t BACKWARD_CONNECTION_FLAG_OFFSET = 0x1;
    tt::tt_fabric::WorkerToFabricEdmSender forward_fabric_sender;
    tt::tt_fabric::WorkerToFabricEdmSender backward_fabric_sender;
    uint8_t connection_flags;
};
