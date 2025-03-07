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
        if (has_forward_connection()) {
            forward_fabric_sender.open();
        }
        if (has_backward_connection()) {
            backward_fabric_sender.open();
        }
    }
    inline bool has_forward_connection() const { return connection_flags & FORWARD_CONNECTION_FLAG_MASK; }
    inline bool has_backward_connection() const { return connection_flags & BACKWARD_CONNECTION_FLAG_MASK; }
    inline void close() {
        if (has_forward_connection()) {
            forward_fabric_sender.close();
        }
        if (has_backward_connection()) {
            backward_fabric_sender.close();
        }
    }

    static FabricConnectionManager build_from_args(std::size_t& arg_idx) {
        FabricConnectionManager connection_manager;
        connection_manager.connection_flags = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++) != 0)
                                              << FORWARD_CONNECTION_FLAG_OFFSET;
        if (connection_manager.has_forward_connection()) {
            connection_manager.forward_fabric_sender =
                tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
        }
        connection_manager.connection_flags |= static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++) != 0)
                                               << BACKWARD_CONNECTION_FLAG_OFFSET;
        if (connection_manager.has_backward_connection()) {
            connection_manager.backward_fabric_sender =
                tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
        }
        return connection_manager;
    }

    tt::tt_fabric::WorkerToFabricEdmSender& get_forward_connection() {
        ASSERT(has_forward_connection());
        return forward_fabric_sender;
    }
    tt::tt_fabric::WorkerToFabricEdmSender& get_backward_connection() {
        ASSERT(has_backward_connection());
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
