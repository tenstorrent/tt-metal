// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

//
// Controller: the central rendezvous for the fabric-manager coordinator service.
//
// It implements exactly one collective -- exchange() -- shared by every transport
// (in-process and TCP) and every op (barrier/all_gather/broadcast). It is fully
// thread-safe: each participant is served by its own caller thread (an in-process
// agent thread, or a per-connection TCP handler thread), and exchange() blocks
// that thread until all `count` participants of the (scope, epoch) collective have
// arrived, at which point every caller is released with the gathered contributions.
//
// The controller is intentionally a dumb relay: it performs NO domain merging. The
// agents merge locally via SystemCoordinator::reduce()'s default (apply_merge), so
// no fabric/tt_metal domain code needs to link into the controller. Central merge
// stays an optional future optimization, never a requirement.
//

#include <condition_variable>
#include <cstdint>
#include <map>
#include <mutex>

#include "tools/scaleout/fabric_manager/coordination/transport.hpp"

namespace tt::scaleout_tools::fabric_manager {

class Controller {
public:
    Controller() = default;

    // Thread-safe rendezvous. Blocks until `count` distinct participant indices
    // have called with the same (scope, epoch); returns all payloads in ascending
    // index order. Safe to call concurrently from many threads.
    [[nodiscard]] std::vector<Bytes> exchange(
        const ScopeKey& scope, uint64_t epoch, int index, int count, const Bytes& payload);

private:
    struct Key {
        std::optional<uint32_t> mesh_id;
        uint64_t epoch;
        bool operator<(const Key& o) const { return std::tie(mesh_id, epoch) < std::tie(o.mesh_id, o.epoch); }
    };

    struct Phase {
        int expected = 0;
        int arrived = 0;
        int retrieved = 0;
        bool released = false;
        std::map<int, Bytes> contributions;  // participant index -> payload (ordered)
        std::vector<Bytes> result;           // assembled once, read by all
    };

    std::mutex mu_;
    std::condition_variable cv_;
    std::map<Key, Phase> phases_;
};

}  // namespace tt::scaleout_tools::fabric_manager
