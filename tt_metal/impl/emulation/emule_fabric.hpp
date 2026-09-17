// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Fabric routing subsystem (multi-chip NOC routing / delivery / teleport). See
// docs/fabric-ccl-emulation.md. Internals live in emule_fabric.cpp; exposed here are only the
// pieces the engine's per-program ProgramRoutes cache snapshots/restores: ConnRoute + the route
// globals below. (Extern seam fabric<->engine; a later pass could replace with snapshot/restore
// accessors — cf. the g_core_map_* seam in emule_device_map.hpp.)
#include <atomic>
#include <cstdint>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

namespace tt::tt_metal::emule {

// Fabric connection routes recorded host-side by append_fabric_connection_rt_args: for 1D the dst
// chip is bound to the connection, not the header, so the host records each connection's direction
// + immediate neighbor here. See tt-emule docs/fabric-ccl-emulation.md.
struct ConnRoute {
    uint32_t dir;       // RoutingDirection (N/E/S/W)
    uint32_t neighbor;  // immediate neighbor physical chip
};

extern std::mutex g_conn_route_mu;
extern std::unordered_map<uint32_t, std::vector<ConnRoute>> g_conn_route;
extern std::unordered_map<uint64_t, std::vector<ConnRoute>> g_worker_conns;
extern std::unordered_map<uint64_t, uint32_t> g_mux_dir;
extern std::unordered_map<uint32_t, std::set<uint32_t>> g_ring_adj;
extern std::unordered_map<uint64_t, uint32_t> g_worker_dir;
extern std::atomic<bool> g_conn_route_dirty;

}  // namespace tt::tt_metal::emule
