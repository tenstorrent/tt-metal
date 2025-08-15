// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include "dataflow_api.h"
#include "edm_fabric_worker_adapters.hpp"

namespace tt::tt_fabric {

// A scalable connection manager that can host an arbitrary number of
// WorkerToFabricEdmSender connections up to MaxConnections.
//
// Runtime layout expected by build_from_args():
//   - for i in [0, MaxConnections):
//       - uint32_t tag                         // arbitrary user tag/label (e.g., direction); opaque to the manager
//       - <WorkerToFabricEdmSender build args>
//
// Notes:
// - Tags are optional metadata provided by the host to enable selection/routing policies in kernels. If unused, host
// should pass 0.
template <std::size_t MaxConnections>
class FabricConnectionManagerV2 final {
public:
    using Sender = tt::tt_fabric::WorkerToFabricEdmSender;

    enum BuildFromArgsMode : uint8_t {
        BUILD_ONLY,
        BUILD_AND_OPEN_CONNECTION,
        BUILD_AND_OPEN_CONNECTION_START_ONLY,
    };

    struct ConnectionSlot {
        Sender sender;
        uint32_t tag;
    };

    template <BuildFromArgsMode build_mode = BuildFromArgsMode::BUILD_ONLY>
    static FabricConnectionManagerV2 build_from_args(std::size_t& arg_idx) {
        constexpr bool connect = build_mode == BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION ||
                                 build_mode == BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY;
        constexpr bool wait_for_connection_open_finish = build_mode == BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION;

        FabricConnectionManagerV2 mgr;

        for (uint32_t i = 0; i < MaxConnections; ++i) {
            mgr.slots_[i].tag = get_arg_val<uint32_t>(arg_idx++);
            mgr.slots_[i].sender =
                tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
            if constexpr (connect) {
                mgr.slots_[i].sender.open_start();
            }
        }

        if constexpr (connect && wait_for_connection_open_finish) {
            for (uint32_t i = 0; i < MaxConnections; ++i) {
                mgr.slots_[i].sender.open_finish();
            }
        }
        return mgr;
    }

    inline uint32_t get_tag(uint32_t index) const {
        ASSERT(index < MaxConnections);
        return slots_[index].tag;
    }

    inline Sender& get(uint32_t index) {
        ASSERT(index < MaxConnections);
        return slots_[index].sender;
    }
    inline const Sender& get(uint32_t index) const {
        ASSERT(index < MaxConnections);
        return slots_[index].sender;
    }

    template <typename Fn>
    inline void for_each(Fn&& fn) {
        for (uint32_t i = 0; i < MaxConnections; ++i) {
            fn(slots_[i].sender, i, slots_[i].tag);
        }
    }

    template <typename Fn>
    inline void for_each_with_tag(uint32_t tag, Fn&& fn) {
        for (uint32_t i = 0; i < MaxConnections; ++i) {
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

private:
    std::array<ConnectionSlot, MaxConnections> slots_;
};

}  // namespace tt::tt_fabric
