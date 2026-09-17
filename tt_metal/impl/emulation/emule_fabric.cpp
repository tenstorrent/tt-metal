// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "emule_fabric.hpp"
#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>
#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include "impl/context/metal_context.hpp"
#include "umd/device/chip/sw_emule_chip.hpp"
#include "tt_emule/chip_store.hpp"
#include "tt_emule/device.hpp"
#include "tt_emule/rank_state.hpp"
#include "jit_hw/internal/emule_thread_ctx.h"
#include "emule_device_map.hpp"
#include "emule_multi_rank_runtime.hpp"
#include "emule_fiber_scheduler.hpp"
#include "emule_noc_bridge.hpp"

namespace tt::tt_metal::emule {
// ---------------------------------------------------------------------------
// Fabric route table: resolve a send's FINAL destination chip by a static walk of the
// control-plane mesh graph (no multi-hop router sim). 1D dst = (src, dir, distance);
// 2D dst = explicit FabricNodeId. See tt-emule docs/fabric-ccl-emulation.md.
// ---------------------------------------------------------------------------
static std::mutex g_fabric_route_mutex;
// (src_chip << 3 | dir) -> ordered chips at distance 1,2,... in that direction (cached; topology is static).
static std::unordered_map<uint32_t, std::vector<uint32_t>> g_fabric_walk_cache;

// Immediate same-mesh neighbor chip of `chip` in `dir`, or -1 if none. The neighbor may be owned by
// a peer rank, in which case this is its synthetic emule id rather than a local ChipId.
static int __emule_fabric_dir_neighbor(
    tt::tt_fabric::ControlPlane& cp, uint32_t chip, tt::tt_fabric::RoutingDirection dir) {
    try {
        tt::tt_fabric::FabricNodeId node(tt::tt_fabric::MeshId{0}, 0);
        if (!tt::tt_metal::emule::multi_rank::node_for_global_chip(cp, chip, node)) {
            return -1;
        }
        auto neighbors = cp.get_chip_neighbors(node, dir);
        for (auto& [mesh, chips] : neighbors) {
            if (!chips.empty()) {
                return tt::tt_metal::emule::multi_rank::global_chip_for_node(
                    cp, tt::tt_fabric::FabricNodeId(mesh, static_cast<std::uint32_t>(chips.front())));
            }
        }
    } catch (...) {
        // control plane unavailable / chip not in graph — caller falls back to the neighbor table
    }
    return -1;
}

// Chip-count backstop for a single-direction walk (loudbox = 8).
static constexpr int kMaxFabricWalkHops = 64;

// Ordered chips reachable from `src` at distance 1,2,... in `dir` (line or ring), cached.
static const std::vector<uint32_t>& __emule_fabric_walk(uint32_t src, tt::tt_fabric::RoutingDirection dir) {
    std::lock_guard<std::mutex> lock(g_fabric_route_mutex);
    uint32_t key = (src << 3) | static_cast<uint32_t>(dir);
    auto it = g_fabric_walk_cache.find(key);
    if (it != g_fabric_walk_cache.end()) {
        return it->second;
    }
    std::vector<uint32_t> walk;
    walk.reserve(kMaxFabricWalkHops);
    auto& cp = MetalContext::instance().get_control_plane();
    uint32_t cur = src;
    for (int hop = 0; hop < kMaxFabricWalkHops; ++hop) {
        int nxt = __emule_fabric_dir_neighbor(cp, cur, dir);
        if (nxt < 0 || static_cast<uint32_t>(nxt) == src) {
            break;  // line end, or ring wrapped back to the source
        }
        walk.push_back(static_cast<uint32_t>(nxt));
        cur = static_cast<uint32_t>(nxt);
    }
    return g_fabric_walk_cache.emplace(key, std::move(walk)).first->second;
}

// ===========================================================================
// Fabric teleport hooks (multi-chip CCL): decode the real-layout packet header,
// resolve the destination chip, and apply the terminal NOC command directly into
// that chip's L1. Delivery is synchronous; the peer-chip consumer observes it via
// its semaphore wait. See tt-emule docs/fabric-ccl-emulation.md.
// ---------------------------------------------------------------------------

// Cross-rank delivery state and rank coordination live in emule_multi_rank_runtime. This tally
// remains per-fiber here because the terminal delivery operation decides when stores are complete.
static thread_local uint32_t t_peer_writes = 0;

// Resolve (noc_addr) -> host pointer on an arbitrary chip, mirroring __emule_resolve_noc_addr but
// against the destination chip's cached core map (already built by that chip's launch).
extern "C" uint8_t* __emule_fabric_resolve_remote(uint32_t dst_chip, uint64_t noc_addr) {
    emule_require_self(__func__);
    std::lock_guard<std::mutex> lock(g_core_map_mutex);
    static const bool rdbg = std::getenv("EMULE_FABRIC_DEBUG") != nullptr;
    auto it = g_core_map_cache.find(dst_chip);
    if (it == g_core_map_cache.end() || !it->second) {
        // No local core map. Under multi-rank that is the NORMAL case for a chip a peer rank owns, so
        // try the peer's shared segment before treating it as a drop.
        if (auto* local = get_sw_emulated_chip(static_cast<tt::ChipId>(__emule_self->chip_id))) {
            if (uint8_t* peer = tt::tt_metal::emule::multi_rank::resolve_peer_l1(dst_chip, noc_addr, *local)) {
                ++t_peer_writes;
                return peer;
            }
        }
        // Returning nullptr DROPS the delivery: the peer's semaphore never moves and the only symptom
        // is a quiescent deadlock elsewhere. Report it once per chip even without EMULE_FABRIC_DEBUG.
        static std::set<uint32_t> warned;
        if (rdbg || warned.insert(dst_chip).second) {
            std::fprintf(
                stderr,
                "[EMULE_FABRIC] WARNING: dropping a fabric delivery — no core map for dst_chip=%u "
                "(cache has %zu chips). The addressed peer will never observe this write.\n",
                dst_chip,
                g_core_map_cache.size());
        }
        return nullptr;
    }
    auto& m = *it->second;
    uint32_t noc_x = (noc_addr >> NOC_LOCAL_BITS) & NOC_NODE_MASK;
    uint32_t noc_y = (noc_addr >> (NOC_LOCAL_BITS + NOC_NODE_ID_BITS)) & NOC_NODE_MASK;
    uint64_t local_addr = noc_addr & NOC_LOCAL_MASK;

    auto find_core = [&](uint32_t x, uint32_t y) { return m.find((uint64_t(x) << 32) | y); };

    // (noc_x,noc_y) are the SOURCE chip's coords (get_noc_addr packs the caller core's); resolve against
    // the destination chip's map. Cross-chip src->logical->dst translation applies ONLY to WORKER cores
    // (harvesting can shift them per chip); DRAM/ETH coords are chip-invariant, so translating them would
    // alias distinct banks. Try verbatim first; translate only if verbatim is a WORKER or missed.
    // See tt-emule docs/fabric-ccl-emulation.md.
    auto cit = find_core(noc_x, noc_y);
    const uint32_t src_chip = __emule_self->chip_id;
    const bool verbatim_is_worker = (cit != m.end() && cit->second->role() == tt_emule::CoreRole::WORKER);
    if (src_chip != dst_chip && (verbatim_is_worker || cit == m.end())) {
        auto* src_obj = get_sw_emulated_chip(src_chip);
        auto* dst_obj = get_sw_emulated_chip(dst_chip);
        if (src_obj != nullptr && dst_obj != nullptr) {
            try {
                auto logical = src_obj->get_soc_descriptor().translate_coord_to(
                    tt_xy_pair(noc_x, noc_y), CoordSystem::TRANSLATED, CoordSystem::LOGICAL);
                auto dst_xy = dst_obj->get_soc_descriptor().translate_coord_to(
                    tt_xy_pair(logical.x, logical.y), CoordSystem::LOGICAL, CoordSystem::TRANSLATED);
                auto t = find_core(static_cast<uint32_t>(dst_xy.x), static_cast<uint32_t>(dst_xy.y));
                if (t != m.end() && t->second->role() == tt_emule::CoreRole::WORKER) {
                    cit = t;  // harvesting-correct worker on the dest chip
                }
            } catch (...) {
                // translation unavailable — keep the verbatim result
            }
        }
    }
    if (cit == m.end()) {
        if (rdbg) {
            fprintf(
                stderr,
                "[EMULE_FABRIC]   resolve_remote: dst_chip=%u has map (%zu cores) but core (%u,%u) NOT FOUND\n",
                dst_chip,
                m.size(),
                noc_x,
                noc_y);
        }
        return nullptr;
    }
    // Bounded by the target's own size, as in __emule_resolve_noc_addr: an offset it cannot
    // hold is a miss, not something to mask into range.
    if (local_addr >= cit->second->l1_size()) {
        return nullptr;
    }
    return cit->second->l1_ptr(local_addr);
}

// Destination chip for a fabric send from src_chip: the single ethernet-connected neighbor of a
// directly-connected 2-chip system, from the cluster descriptor. See tt-emule docs/fabric-ccl-emulation.md.
extern "C" uint32_t __emule_fabric_neighbor(uint32_t src_chip) {
    auto ids = MetalContext::instance().get_cluster().get_ethernet_connected_device_ids(src_chip);
    if (!ids.empty()) {
        return *ids.begin();
    }
    return src_chip;
}

// emule route metadata, keyed by packet-header L1-alias address: the fabric_set_*_route shims record the
// kernel's semantic dst (2D FabricNodeId, 1D hop distance, or line-multicast extent) here; the teleport
// resolves it to physical chip(s). KIND constants KEEP IN SYNC with the shim. See tt-emule
// docs/fabric-ccl-emulation.md.
namespace emule_route_kind {
constexpr uint32_t UNSET = 0, UNICAST_1D = 1, UNICAST_2D = 2, MCAST_1D = 3, MCAST_2D = 4;
}
struct EmuleRoute {
    uint32_t kind = 0, a = 0, b = 0, c = 0, d = 0, e = 0, f = 0;
    uint32_t dir_index = 0;             // 1D: which of the worker's connections (fwd=0/bwd=1), set at send time
    uint32_t eth_channel = 0xFFFFFFFF;  // VC0 connection identity, set at send time
    // Mux-path direction hint (preferred over the range-match heuristic), set at send time:
    uint32_t mux_x = 0xFFFF, mux_y = 0xFFFF;  // worker's mux NOC (TRANSLATED) coords (fabric MUX path)
};
static std::mutex g_route_meta_mu;
// Keyed by the header's FULL host pointer (bridge_l1 + offset). Post-offset-migration a packet
// header's L1 offset is both chip- AND core-agnostic — it is 0-based within each core's L1, so the
// same offset recurs on every core of every chip. A (chip, offset) key would collide across cores
// of one chip (one core's route overwriting another's → wrong-chip delivery → PCC fail / a fiber
// waiting on an atomic-inc that lands elsewhere → quiescent deadlock). The baseline's key was the
// header's host pointer, which is inherently unique per (chip, core, offset) because every core's
// L1 is a distinct mapping; reconstruct that full (untruncated) pointer here — untruncated so it
// stays unique above 4 GB, and host-side-only (a runner map key, never a kernel/L1 value). Both set
// (shim offset) and read (teleport) run on the same fiber → same bridge_l1 → same key.
static std::unordered_map<uint64_t, EmuleRoute> g_route_meta;
static inline uint64_t emule_route_key(uint32_t hdr_off) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(__emule_self->bridge_l1 + hdr_off));
}

extern "C" void __emule_fabric_set_route(
    uint32_t hdr, uint32_t kind, uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e, uint32_t f) {
    emule_require_self(__func__);  // keys through __emule_self->bridge_l1 via emule_route_key
    std::lock_guard<std::mutex> lk(g_route_meta_mu);
    auto& r = g_route_meta[emule_route_key(hdr)];
    r.kind = kind;
    r.a = a;
    r.b = b;
    r.c = c;
    r.d = d;
    r.e = e;
    r.f = f;  // dir_index set separately at send
}

// Record a 1D send's per-connection direction signals: the fwd/bwd conn_index (direct path) and the
// worker's mux NOC coords (MUX path); 0xFFFF means unset. See tt-emule docs/fabric-ccl-emulation.md.
extern "C" void __emule_fabric_set_route_dir(
    uint32_t hdr, uint32_t conn_index, uint32_t eth_channel, uint32_t mux_x, uint32_t mux_y) {
    emule_require_self(__func__);  // keys through __emule_self->bridge_l1 via emule_route_key
    std::lock_guard<std::mutex> lk(g_route_meta_mu);
    auto& r = g_route_meta[emule_route_key(hdr)];
    r.dir_index = conn_index;
    r.eth_channel = eth_channel;
    r.mux_x = mux_x;
    r.mux_y = mux_y;
}

// Carry a stamped route from a source header address to a destination when the header bytes are copied
// (a worker staging a packet header into a forwarder relay slot). On silicon the routing fields ride inside
// the header, so a byte copy carries them for free; emule keeps them in the address-keyed side-table, so the
// copy must replicate the entry. No-op when src carries no route. src_key/dst_key are FULL host-pointer
// values (bridge_l1 + offset) in the same key space as the set/read sides — NOT bridge_l1-relative uint32
// offsets (see the ABI note in the body for why the widening was removed). See tt-emule
// docs/fabric-ccl-emulation.md.
extern "C" void __emule_fabric_route_follow(uint64_t src_key, uint64_t dst_key) {
    // src_key/dst_key are FULL host-pointer values (bridge_l1 + offset) — the same key space as
    // emule_route_key and the set + teleport-read sides. They MUST be full 64-bit pointers, not
    // bridge_l1-relative uint32 offsets: a forwarder's relay slot lives on a DIFFERENT core whose L1
    // mmap can be >4 GB from the writer's bridge_l1, so (dst - writer_bridge_l1) overflows uint32 and
    // mis-keys the copy → the forwarder's teleport lookup misses → kind UNSET → wrong-neighbor route
    // → many-to-one converge deadlock (reduce_to_one column dimension). See tt-emule
    // docs/fabric-ccl-emulation.md.
    if (src_key == dst_key) {
        return;
    }
    std::lock_guard<std::mutex> lk(g_route_meta_mu);
    auto it = g_route_meta.find(src_key);
    if (it == g_route_meta.end()) {
        return;  // src carries no route — not a packet-header copy; nothing to follow.
    }
    const EmuleRoute r = it->second;  // copy before the insert below can rehash/invalidate `it`.
    g_route_meta[dst_key] = r;
}

std::mutex g_conn_route_mu;
// Keyed by SRC CHIP only (line direction is a per-chip property; the connection-owner core can differ from
// the sender, so a per-core key would miss). Deduped by direction, kept sorted by RoutingDirection so the
// contents are a function of the op alone and not of which host thread recorded first. Order here is NOT a
// direction index — that is g_worker_conns' job. See tt-emule docs/fabric-ccl-emulation.md.
std::unordered_map<uint32_t, std::vector<ConnRoute>> g_conn_route;
// Same records, keyed (src<<32 | wx<<16 | wy) and kept in HOST RECORD ORDER, which
// append_fabric_connection_rt_args documents as the kernel's own open order (fwd, then bwd). A sender's
// dir_index is an index into ITS OWN open sequence, so it may only be resolved against this per-worker
// vector: the src-keyed table above holds the union over every worker on the chip, and on a chip with more
// than two active directions (the 4-directional path — all_to_all_combine / all_to_all_dispatch on a 4x8
// galaxy) index 1 of that union is not the same direction the worker opened second, which teleports whole
// payloads to the wrong chip. Per-worker keying also removes the cross-thread append race, since one
// worker's connections are recorded by one thread in order.
std::unordered_map<uint64_t, std::vector<ConnRoute>> g_worker_conns;
// Physical ring adjacency. Multi-rank seeding fills edges whose far endpoint is owned by another rank.
std::unordered_map<uint32_t, std::set<uint32_t>> g_ring_adj;
// Per-op reset flag: cleared at each new op's first connection-record so a later op's different line
// orientation can't corrupt the src-keyed, direction-deduped table. See tt-emule docs/fabric-ccl-emulation.md.
std::atomic<bool> g_conn_route_dirty{true};
// Per-worker resolved line direction, keyed (src<<32 | wx<<16 | wy): on a path with neither a worker-owned
// connection sequence nor MUX coordinates, infer it from a multicast's range and reuse it for unicasts.
// Reset per op.
// See tt-emule docs/fabric-ccl-emulation.md.
std::unordered_map<uint64_t, uint32_t> g_worker_dir;
// Per-mux-core line direction, keyed (src<<32 | logical_x<<16 | logical_y): the mux→EDM append records the
// mux's forwarding_direction; the teleport recovers it from the worker's carried mux NOC coords. Reset per
// op. See tt-emule docs/fabric-ccl-emulation.md.
std::unordered_map<uint64_t, uint32_t> g_mux_dir;
static inline uint64_t __emule_worker_key(uint32_t src, uint32_t wx, uint32_t wy) {
    return (static_cast<uint64_t>(src) << 32) | (static_cast<uint64_t>(wx & 0xFFFF) << 16) | (wy & 0xFFFF);
}

// Seed g_ring_adj with the WHOLE physical ring, including edges that leave this rank.
// ------------------------------------------------------------------------------------
// Accumulating adjacency from __emule_fabric_record_conn alone makes it rank-local: that hook fires
// only for connections this rank opens, so a ring walk dead-ends at the rank boundary and a CCL
// delivers to a TRUNCATED target set — silently, with no drop and no unresolved route, which is what
// a partially-correct all-gather looks like. The descriptor has what is missing: a link whose far
// side is not visible is recorded against the peer's UNIQUE id, which the gchip registry maps to a
// chip id. See tt-emule docs/multi-rank-emulation.md §3.
static bool g_ring_adj_seeded = false;
static void __emule_seed_global_ring_adj() {  // pre: g_conn_route_mu held
    if (g_ring_adj_seeded || !tt_emule::chip_store_job_is_multi_rank()) {
        return;
    }
    g_ring_adj_seeded = true;
    try {
        auto& cluster = MetalContext::instance().get_cluster();
        auto& cp = MetalContext::instance().get_control_plane();
        const auto* desc = cluster.get_cluster_desc();
        if (desc == nullptr) {
            return;
        }
        for (auto chip : desc->get_all_chips()) {
            const auto local = static_cast<uint32_t>(chip);
            for (auto nb : cluster.get_ethernet_connected_device_ids(chip)) {  // both sides visible
                g_ring_adj[local].insert(static_cast<uint32_t>(nb));
                g_ring_adj[static_cast<uint32_t>(nb)].insert(local);
            }
        }
        // Edges that leave this rank: keyed by the far side's unique id, so they survive the
        // visible-device slicing that hides the chip itself.
        for (const auto& [chip, by_chan] : desc->get_ethernet_connections_to_remote_devices()) {
            const auto local = static_cast<uint32_t>(chip);
            for (const auto& [chan, remote] : by_chan) {
                (void)chan;
                const uint64_t peer_uid = std::get<0>(remote);
                if (const auto peer = tt::tt_metal::emule::multi_rank::global_chip_for_asic(cp, peer_uid)) {
                    g_ring_adj[local].insert(*peer);
                    g_ring_adj[*peer].insert(local);
                }
            }
        }
    } catch (...) {
        // No cluster/control plane yet — the per-connection accumulation still applies.
    }
}

extern "C" void __emule_fabric_record_conn(uint32_t src, uint32_t wx, uint32_t wy, uint32_t dir, uint32_t dst) {
    // The caller (fabric.cpp) passes the connection's FINAL destination chip. On silicon a fabric
    // connection is per-hop, so for CCL that destination IS the adjacent chip and the two agree —
    // but a MeshSocket opens one connection straight to a peer that may be several hops away along
    // a line (1D requires only same row/column, not adjacency). Recording that distant chip as a
    // ring neighbor inserts a phantom edge into the persistent g_ring_adj, whose degree then
    // exceeds 2 and makes walk_ring TT_FATAL with "ambiguous ring continuation". Resolve the true
    // immediate neighbor from (src, dir) instead; when the destination really is adjacent this is
    // identity, so CCL topology is unchanged.
    uint32_t neighbor = dst;
    {
        auto& cp = MetalContext::instance().get_control_plane();
        const int nb = __emule_fabric_dir_neighbor(cp, src, static_cast<tt::tt_fabric::RoutingDirection>(dir));
        if (nb >= 0) {
            neighbor = static_cast<uint32_t>(nb);
        }
    }
    std::lock_guard<std::mutex> lk(g_conn_route_mu);
    if (g_conn_route_dirty.exchange(false)) {
        g_conn_route.clear();
        g_worker_conns.clear();
        g_worker_dir.clear();
        g_mux_dir.clear();
    }
    // Record the connection-owner core's (the mux core, on the MUX path) direction, keyed by its LOGICAL
    // coords — before the per-direction dedup below, which is for the src-keyed g_conn_route only.
    g_mux_dir[__emule_worker_key(src, wx, wy)] = dir;
    // Accumulate the undirected ring edge (persistent; unaffected by the per-op reset above).
    __emule_seed_global_ring_adj();  // multi-rank only; adds the edges this rank never opens
    g_ring_adj[src].insert(neighbor);
    g_ring_adj[neighbor].insert(src);
    // Per-worker, in open order — this is what a sender's dir_index indexes.
    auto& wv = g_worker_conns[__emule_worker_key(src, wx, wy)];
    if (std::none_of(wv.begin(), wv.end(), [dir](const ConnRoute& c) { return c.dir == dir; })) {
        wv.push_back(ConnRoute{dir, neighbor});
    }
    auto& v = g_conn_route[src];
    // Sorted by direction so the union's contents don't depend on host record order.
    auto at = v.begin();
    for (; at != v.end(); ++at) {
        if (at->dir == dir) {
            return;  // already recorded this direction for src
        }
        if (at->dir > dir) {
            break;
        }
    }
    v.insert(at, ConnRoute{dir, neighbor});
}

// Ordered ring members at distance 1,2,... from `src` in `start_dir`: first hop from g_conn_route[src], then
// follow the program's undirected adjacency g_ring_adj (unvisited non-prev neighbor), tracing the turning
// Hamiltonian cycle the compass walk can't. Stops at a dead end / cycle close; empty if src/start_dir was
// never recorded. TT_FATALs if a chip has >1 continuation (cross-axis edges from another op's collective,
// not disambiguable from the undirected union) rather than misroute. See docs/fabric-ccl-emulation.md.
static std::vector<uint32_t> __emule_fabric_walk_ring(uint32_t src, uint32_t start_dir) {
    std::vector<uint32_t> walk;
    std::lock_guard<std::mutex> lk(g_conn_route_mu);
    auto sit = g_conn_route.find(src);
    if (sit == g_conn_route.end()) {
        return walk;
    }
    int first = -1;
    for (const auto& c : sit->second) {
        if (c.dir == start_dir) {
            first = static_cast<int>(c.neighbor);
            break;
        }
    }
    if (first < 0) {
        return walk;  // start direction not recorded — caller falls back
    }
    walk.push_back(static_cast<uint32_t>(first));
    // Traverse the program's UNDIRECTED ring adjacency: g_conn_route only fixes the FIRST hop's direction;
    // connectivity comes from g_ring_adj so a chip that opened one connection this op doesn't dead-end.
    std::set<uint32_t> visited{src, static_cast<uint32_t>(first)};
    uint32_t prev = src, cur = static_cast<uint32_t>(first);
    for (int hop = 0; hop < 64; ++hop) {  // 64 = chip-count backstop
        auto ait = g_ring_adj.find(cur);
        if (ait == g_ring_adj.end()) {
            break;
        }
        // Continuation = the unvisited neighbor other than `prev` (exactly one on a valid 1D ring; see header).
        int next = -1;
        int n_cont = 0;
        for (uint32_t nb : ait->second) {
            if (nb != prev && visited.find(nb) == visited.end()) {
                if (++n_cont == 1) {
                    next = static_cast<int>(nb);
                }
            }
        }
        if (n_cont > 1) {
            // More than one continuation. This is NOT necessarily corrupt state: on a board whose chips
            // each have three ethernet neighbors (the 8-chip 2x4 with wraparound — every chip is degree 3)
            // the undirected union legitimately spans both axes, so a ring walk cannot pick the successor
            // on adjacency alone. Prefer the neighbor that continues in the walk's own routing direction;
            // that is the control plane's authoritative answer and keeps a straight line straight.
            int dir_next = -1;
            try {
                auto& cp = MetalContext::instance().get_control_plane();
                dir_next =
                    __emule_fabric_dir_neighbor(cp, cur, static_cast<tt::tt_fabric::RoutingDirection>(start_dir));
            } catch (...) {
                dir_next = -1;
            }
            const bool usable = dir_next >= 0 && static_cast<uint32_t>(dir_next) != prev &&
                                visited.find(static_cast<uint32_t>(dir_next)) == visited.end() &&
                                ait->second.count(static_cast<uint32_t>(dir_next)) > 0;
            if (!usable) {
                // Genuinely undecidable here. Abandon the ring walk rather than misroute — the caller
                // falls back to the direction-consistent compass walk (see the callers of this function),
                // which is the right answer on a multi-axis board anyway.
                return {};
            }
            next = dir_next;
        }
        if (next < 0 || static_cast<uint32_t>(next) == src) {
            break;  // dead end, or the cycle closed back at the source
        }
        walk.push_back(static_cast<uint32_t>(next));
        visited.insert(static_cast<uint32_t>(next));
        prev = cur;
        cur = static_cast<uint32_t>(next);
    }
    return walk;
}

// An unresolved route falls back to an arbitrary ethernet neighbor — a VALID but WRONG chip, so the
// write lands, nothing is dropped, and the only symptom is a deadlock or bad data somewhere else.
// Say so. EMULE_FABRIC_STRICT promotes it to a throw. docs/fabric-ccl-emulation.md.
static void __emule_fabric_route_unresolved(uint32_t src_chip, const char* why, uint32_t detail) {
    // Strict by DEFAULT: the loudbox gate and every socket suite resolve every route, so an
    // unresolved one is a bug rather than a mode we rely on. EMULE_FABRIC_STRICT=0 downgrades it
    // to a one-shot warning for bisecting.
    static const bool strict = [] {
        const char* v = std::getenv("EMULE_FABRIC_STRICT");
        return v == nullptr || (v[0] != '0' && v[0] != '\0');
    }();
    static std::mutex mu;
    static std::set<uint64_t> seen;
    const uint32_t fallback = __emule_fabric_neighbor(src_chip);
    if (strict) {
        TT_THROW(
            "emule fabric: unresolved route from chip {} ({}, detail={}). Would deliver to neighbor chip {} "
            "instead of the addressed peer.",
            src_chip,
            why,
            detail,
            fallback);
    }
    bool first = false;
    {
        std::lock_guard<std::mutex> g(mu);
        first = seen.insert((static_cast<uint64_t>(src_chip) << 32) | detail).second;
    }
    if (first) {
        std::fprintf(
            stderr,
            "[EMULE_FABRIC] WARNING: unresolved route from chip %u (%s, detail=%u) — delivering to "
            "neighbor chip %u, which is probably NOT the addressed peer. Set EMULE_FABRIC_STRICT=1 "
            "to make this fatal.\n",
            src_chip,
            why,
            detail,
            fallback);
    }
}

// Resolve the FINAL destination chip(s) for a send: one chip for unicast, the line members for a multicast.
// Gated by EMULE_FABRIC8 (off → legacy single neighbor). See tt-emule docs/fabric-ccl-emulation.md.
static std::vector<uint32_t> __emule_fabric_resolve_targets(const uint8_t* h, uint32_t src_chip) {
    static const bool fabric8 = std::getenv("EMULE_FABRIC8") != nullptr;
    if (!fabric8) {
        return {__emule_fabric_neighbor(src_chip)};
    }
    EmuleRoute r;
    {
        std::lock_guard<std::mutex> lk(g_route_meta_mu);
        // Round-trip h -> offset -> bridge_l1+offset via emule_route_key so the read key is derived
        // by the same helper as the set-side key (one source of truth for the key formula).
        auto it = g_route_meta.find(emule_route_key(static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(h) - reinterpret_cast<uintptr_t>(__emule_self->bridge_l1))));
        if (it == g_route_meta.end()) {
            __emule_fabric_route_unresolved(src_chip, "no route stamped for this packet header", 0);
            return {__emule_fabric_neighbor(src_chip)};
        }
        r = it->second;
    }
    static const bool rdbg = std::getenv("EMULE_FABRIC_DEBUG") != nullptr;
    if (rdbg) {
        std::lock_guard<std::mutex> lk(g_conn_route_mu);
        auto cit = g_conn_route.find(src_chip);
        fprintf(
            stderr,
            "[EMULE_FABRIC]   resolve src=%u kind=%u a=%u b=%u ewns=%u/%u/%u/%u dir_idx=%u conns=%zu mux=(%u,%u)\n",
            src_chip,
            r.kind,
            r.a,
            r.b,
            r.c,
            r.d,
            r.e,
            r.f,
            r.dir_index,
            cit == g_conn_route.end() ? (size_t)0 : cit->second.size(),
            r.mux_x,
            r.mux_y);
    }
    auto& cp = MetalContext::instance().get_control_plane();
    if (r.kind == emule_route_kind::UNICAST_2D) {  // a=dst_dev, b=dst_mesh
        // Through the registry: a 2D unicast names its destination by mesh, and under multi-rank
        // that mesh routinely belongs to a peer rank, which has no local physical chip id.
        const int g = tt::tt_metal::emule::multi_rank::global_chip_for_node(
            cp, tt::tt_fabric::FabricNodeId(tt::tt_fabric::MeshId{r.b}, r.a));
        if (g >= 0) {
            return {static_cast<uint32_t>(g)};
        }
    } else if (r.kind == emule_route_kind::MCAST_2D) {
        // Mesh multicast first routes to (a,b). The hop counts are interpreted from that start node.
        using RD = tt::tt_fabric::RoutingDirection;
        try {
            const uint32_t start_chip = static_cast<uint32_t>(cp.get_physical_chip_id_from_fabric_node_id(
                tt::tt_fabric::FabricNodeId(tt::tt_fabric::MeshId{r.b}, r.a)));
            std::vector<uint32_t> tgts;
            auto append_unique = [&](uint32_t chip) {
                if (std::find(tgts.begin(), tgts.end(), chip) == tgts.end()) {
                    tgts.push_back(chip);
                }
            };
            auto append_branch = [&](uint32_t root, RD dir, uint32_t hops) {
                const auto& walk = __emule_fabric_walk(root, dir);
                for (uint32_t k = 0; k < hops && k < walk.size(); ++k) {
                    append_unique(walk[k]);
                }
            };

            const uint32_t spine_hops = r.e != 0 ? r.e : r.f;
            if (spine_hops != 0) {
                const RD spine_dir = r.e != 0 ? RD::N : RD::S;
                uint32_t root = start_chip;
                for (uint32_t spine_index = 0; spine_index < spine_hops; ++spine_index) {
                    append_unique(root);
                    append_branch(root, RD::E, r.c);
                    append_branch(root, RD::W, r.d);
                    const auto& spine_walk = __emule_fabric_walk(root, spine_dir);
                    if (spine_index + 1 < spine_hops) {
                        if (spine_walk.empty()) {
                            break;
                        }
                        root = spine_walk[0];
                    }
                }
            } else {
                append_unique(start_chip);
                const RD line_dir = r.c != 0 ? RD::E : RD::W;
                const uint32_t line_hops = r.c != 0 ? r.c : r.d;
                if (line_hops > 1) {
                    append_branch(start_chip, line_dir, line_hops - 1);
                }
            }
            if (!tgts.empty()) {
                return tgts;
            }
        } catch (...) {
        }
    } else if (r.kind == emule_route_kind::MCAST_1D || r.kind == emule_route_kind::UNICAST_1D) {
        // 1D MUX path carries no direction tag: infer the worker's direction from a multicast's range and
        // cache it per (src, worker_core) for that worker's unicasts. See tt-emule docs/fabric-ccl-emulation.md.
        emule_require_self(__func__);
        TT_FATAL(__emule_self->core != nullptr, "{}: fiber has no core context", __func__);
        const uint32_t wx = __emule_self->core->logical_x;
        const uint32_t wy = __emule_self->core->logical_y;
        const uint64_t wkey = __emule_worker_key(src_chip, wx, wy);
        std::vector<ConnRoute> conns;
        // This worker's OWN open sequence, which dir_index indexes; empty on the MUX path, where the
        // recorded core is the mux, not the worker (direction comes from g_mux_dir there).
        std::vector<ConnRoute> wconns;
        {
            std::lock_guard<std::mutex> lk(g_conn_route_mu);
            auto it = g_conn_route.find(src_chip);
            if (it != g_conn_route.end()) {
                conns = it->second;
            }
            auto wit = g_worker_conns.find(wkey);
            if (wit != g_worker_conns.end()) {
                wconns = wit->second;
            }
        }
        // Resolve an open-sequence index against this worker's own sequence; the src-keyed union is only a
        // fallback for senders whose connections were recorded under another core (MUX).
        const std::vector<ConnRoute>& idx_conns = wconns.empty() ? conns : wconns;
        int dir = -1;
        // (1) VC0 channel identity: silicon indexes its connection table with this channel. Derive the same
        // direction from the control plane rather than duplicating the binding in fabric.cpp.
        if (r.eth_channel != 0xFFFFFFFF) {
            auto& cp = MetalContext::instance().get_control_plane();
            const auto src_node = cp.get_fabric_node_id_from_physical_chip_id(static_cast<ChipId>(src_chip));
            dir = static_cast<int>(
                cp.eth_direction_to_routing_direction(cp.get_eth_chan_direction(src_node, r.eth_channel)));
        }
        // (2) Mux-core direction: translate the worker's mux NOC coords to the mux's LOGICAL core and look up
        // the direction the mux→EDM append recorded. Resolves ring, where the range-match below cannot.
        if (dir < 0 && r.mux_x != 0xFFFF) {
            auto* src_obj = get_sw_emulated_chip(src_chip);
            if (src_obj != nullptr) {
                try {
                    auto lg = src_obj->get_soc_descriptor().translate_coord_to(
                        tt_xy_pair(r.mux_x, r.mux_y), CoordSystem::TRANSLATED, CoordSystem::LOGICAL);
                    std::lock_guard<std::mutex> lk(g_conn_route_mu);
                    auto mit = g_mux_dir.find(
                        __emule_worker_key(src_chip, static_cast<uint32_t>(lg.x), static_cast<uint32_t>(lg.y)));
                    if (mit != g_mux_dir.end()) {
                        dir = static_cast<int>(mit->second);
                    }
                } catch (...) {
                }
            }
        }
        // (3) Fallback — range-match heuristic (and its cached g_worker_dir / conn-index), used only when the
        // mux signal above is absent. See tt-emule docs/fabric-ccl-emulation.md.
        if (dir < 0 && r.kind == emule_route_kind::MCAST_1D) {
            const uint32_t range = r.b ? r.b : 1;
            // Pick the direction whose walk_ring reach equals the multicast range (measured along the actual
            // ring, which walk_ring follows and the compass walk can't). Disambiguates the two directions of a
            // bidirectional reduce_scatter on an open line; on a closed ring both reach N-1, so a tie falls
            // through to the recorded direction index rather than guessing conns[0]. See docs/fabric-ccl-emulation.md.
            int matched = -1;
            int n_match = 0;
            for (const auto& cr : conns) {
                if (__emule_fabric_walk_ring(src_chip, cr.dir).size() == range) {
                    if (++n_match == 1) {
                        matched = static_cast<int>(cr.dir);
                    }
                }
            }
            if (n_match == 1) {
                dir = matched;  // unique range-match — the disambiguated direction
            } else if (!idx_conns.empty()) {
                // no match, or an ambiguous closed-ring tie — use the actually-recorded send index
                dir = static_cast<int>(idx_conns[r.dir_index < idx_conns.size() ? r.dir_index : 0].dir);
            }
            if (dir >= 0) {
                std::lock_guard<std::mutex> lk(g_conn_route_mu);
                g_worker_dir[wkey] = static_cast<uint32_t>(dir);
            }
        } else if (dir < 0) {  // UNICAST_1D
            std::lock_guard<std::mutex> lk(g_conn_route_mu);
            // A direct sender's connection index refers to this worker's open sequence. Prefer that exact
            // mapping over the one-direction cache: a bidirectional collective can send equal hop counts on
            // both connections, so one cached direction cannot represent both slots.
            if (!wconns.empty()) {
                dir = static_cast<int>(wconns[r.dir_index < wconns.size() ? r.dir_index : 0].dir);
            } else if (auto wit = g_worker_dir.find(wkey); wit != g_worker_dir.end()) {
                dir = static_cast<int>(wit->second);
            } else if (!idx_conns.empty()) {
                dir = static_cast<int>(idx_conns[r.dir_index < idx_conns.size() ? r.dir_index : 0].dir);
            }
        }
        if (dir >= 0) {
            // Follow the real 1D ring via the recorded per-chip neighbors; fall back to the compass walk if
            // the chain is incomplete (walk[0] equals the compass neighbor). See tt-emule docs/fabric-ccl-emulation.md.
            std::vector<uint32_t> walk = __emule_fabric_walk_ring(src_chip, static_cast<uint32_t>(dir));
            if (walk.empty()) {
                walk = __emule_fabric_walk(src_chip, static_cast<tt::tt_fabric::RoutingDirection>(dir));
            }
            std::vector<uint32_t> tgts;
            if (r.kind == emule_route_kind::MCAST_1D) {
                const uint32_t start = r.a ? r.a : 1, range = r.b ? r.b : 1;
                tgts.reserve(std::min<size_t>(range, walk.size()));
                for (uint32_t hop = start; hop < start + range && hop - 1 < walk.size(); ++hop) {
                    tgts.push_back(walk[hop - 1]);
                }
            } else {
                const uint32_t dist = r.a ? r.a : 1;
                if (dist - 1 < walk.size()) {
                    tgts.push_back(walk[dist - 1]);
                }
            }
            if (!tgts.empty()) {
                return tgts;
            }
        }
    }
    // Fallthrough: the route WAS stamped but no target could be derived from it — the dangerous case,
    // since the kernel addressed a specific peer and we are about to pick a different one.
    __emule_fabric_route_unresolved(src_chip, "route stamped but no target resolved", r.kind);
    return {__emule_fabric_neighbor(src_chip)};
}

// Apply the terminal NOC command of a fabric send to ONE destination chip's L1 (the per-target delivery,
// looped over by the teleport for multicast).
static void __emule_fabric_deliver_ops(
    uint32_t dst_chip, const uint8_t* h, const void* payload, uint32_t size, uint8_t noc_send_type, bool dbg);

// Publishes any cross-rank writes this packet made, AFTER the terminal op's stores and their release
// fence. A peer decides global quiescence from these counters, so one that moved before its data
// would let the peer conclude "nothing new" and stay parked. See tt-emule docs/multi-rank-emulation.md.
static void __emule_fabric_deliver(
    uint32_t dst_chip, const uint8_t* h, const void* payload, uint32_t size, uint8_t noc_send_type, bool dbg) {
    t_peer_writes = 0;
    __emule_fabric_deliver_ops(dst_chip, h, payload, size, noc_send_type, dbg);
    if (t_peer_writes != 0) {
        std::atomic_thread_fence(std::memory_order_release);
        tt::tt_metal::emule::multi_rank::note_deliveries(t_peer_writes);
        t_peer_writes = 0;
    }
}

static void __emule_fabric_deliver_ops(
    uint32_t dst_chip, const uint8_t* h, const void* payload, uint32_t size, uint8_t noc_send_type, bool dbg) {
    const uint64_t noc_address = *reinterpret_cast<const uint64_t*>(h + 0);
    switch (noc_send_type) {
        case 0: {  // NOC_UNICAST_WRITE
            uint8_t* d = __emule_fabric_resolve_remote(dst_chip, noc_address);
            if (d != nullptr && payload != nullptr && size > 0) {
                std::memcpy(d, payload, size);
                std::atomic_thread_fence(std::memory_order_release);
                __emule_fiber_wake(d);
            }
            break;
        }
        case 1: {  // NOC_UNICAST_INLINE_WRITE: {noc_address; value@8}
            uint32_t value = *reinterpret_cast<const uint32_t*>(h + 8);
            uint8_t* d = __emule_fabric_resolve_remote(dst_chip, noc_address);
            if (d != nullptr) {
                reinterpret_cast<std::atomic<uint32_t>*>(d)->store(value, std::memory_order_release);
                __emule_fiber_wake(d);
            }
            break;
        }
        case 2: {  // NOC_UNICAST_ATOMIC_INC: {noc_address; val@8}
            uint32_t val = *reinterpret_cast<const uint32_t*>(h + 8);
            uint8_t* d = __emule_fabric_resolve_remote(dst_chip, noc_address);
            if (d != nullptr) {
                uint32_t old = reinterpret_cast<std::atomic<uint32_t>*>(d)->fetch_add(val, std::memory_order_release);
                if (dbg) {
                    fprintf(
                        stderr,
                        "[EMULE_FABRIC]   atomic_inc chip=%u dst=%p %u->%u (val=%u)\n",
                        dst_chip,
                        (void*)d,
                        old,
                        old + val,
                        val);
                }
                __emule_fiber_wake(d);
            }
            break;
        }
        case 3: {  // NOC_FUSED_UNICAST_ATOMIC_INC: {noc_address; semaphore_noc_address@8; val@16}
            uint64_t sem_addr = *reinterpret_cast<const uint64_t*>(h + 8);
            uint32_t val = *reinterpret_cast<const uint32_t*>(h + 16);
            uint8_t* d = __emule_fabric_resolve_remote(dst_chip, noc_address);
            const bool has_payload = payload != nullptr && size > 0;
            if (d != nullptr && has_payload) {
                std::memcpy(d, payload, size);
                std::atomic_thread_fence(std::memory_order_release);
                __emule_fiber_wake(d);
            }
            // The increment is this op's readiness signal. Raising it when the payload half did not
            // land hands the consumer stale bytes it believes are fresh, so refuse and say so; the
            // consumer then waits, which is diagnosable, instead of reading the wrong data.
            if (has_payload && d == nullptr) {
                std::fprintf(
                    stderr,
                    "[EMULE_FABRIC] WARNING: fused write+inc on chip %u could not resolve its data "
                    "address; withholding the semaphore increment rather than signalling stale bytes.\n",
                    static_cast<unsigned>(dst_chip));
                break;
            }
            uint8_t* s = __emule_fabric_resolve_remote(dst_chip, sem_addr);
            if (s != nullptr) {
                reinterpret_cast<std::atomic<uint32_t>*>(s)->fetch_add(val, std::memory_order_release);
                __emule_fiber_wake(s);
            }
            break;
        }
        case 4: {  // NOC_UNICAST_SCATTER_WRITE: noc_address[4]@0, chunk_size[3]@32, chunk_count@38, chunk_encoding@39
            // Also carries the fused scatter-write + atomic-inc: chunk_encoding holds a 2-bit code per chunk
            // (silicon NocScatterWriteChunkEncoding: 0 = NOP, 1 = unicast write, 2/3 = semaphore increment).
            // On silicon a scatter write is NOT left at 0 — to_noc_unicast_scatter_write fills every chunk with
            // encoding 1, and a fused packet marks its trailing chunk as a seminc (2/3) with the writes at 1.
            // Encoding 0 is CHUNK_ENCODING_NOP on the wire, so the write branch below handling enc 0 the same
            // as enc 1 is an emulator compatibility fallback only: emule's own NocUnicastScatterCommandHeader
            // defaults chunk_encoding to 0 for a plain scatter write. For a seminc chunk, fetch_add the value
            // stored in that chunk's size slot instead of copying payload (the seminc chunk carries no bytes).
            const uint64_t* na = reinterpret_cast<const uint64_t*>(h + 0);
            const uint16_t* cs = reinterpret_cast<const uint16_t*>(h + 32);
            uint8_t chunk_count = *(h + 38);
            uint8_t chunk_encoding = *(h + 39);
            uint32_t off = 0;
            // A seminc chunk is the readiness signal for the write chunks beside it, so a dropped
            // write must suppress it: resolve every write first, and if any is unreachable the
            // consumer is left waiting (diagnosable) rather than reading bytes that never arrived.
            bool writes_all_resolved = true;
            for (uint8_t i = 0; i < chunk_count; ++i) {
                const uint8_t enc = (chunk_encoding >> (i * 2)) & 0x3;
                if (enc == 2 || enc == 3) {
                    continue;
                }
                if (__emule_fabric_resolve_remote(dst_chip, na[i]) == nullptr) {
                    writes_all_resolved = false;
                }
            }
            if (!writes_all_resolved) {
                std::fprintf(
                    stderr,
                    "[EMULE_FABRIC] WARNING: scatter write+inc on chip %u could not resolve every write "
                    "chunk; withholding its semaphore increment rather than signalling stale bytes.\n",
                    static_cast<unsigned>(dst_chip));
            }
            for (uint8_t i = 0; i < chunk_count; ++i) {
                const uint8_t enc = (chunk_encoding >> (i * 2)) & 0x3;
                uint8_t* d = __emule_fabric_resolve_remote(dst_chip, na[i]);
                if (enc == 2 /*SEMINC_NO_FLUSH*/ || enc == 3 /*SEMINC_FLUSH*/) {
                    if (!writes_all_resolved) {
                        continue;
                    }
                    uint32_t val = cs[i];  // seminc value packed into this chunk's size slot
                    if (d != nullptr) {
                        reinterpret_cast<std::atomic<uint32_t>*>(d)->fetch_add(val, std::memory_order_release);
                        if (dbg) {
                            fprintf(
                                stderr,
                                "[EMULE_FABRIC]   scatter_seminc chip=%u dst=%p val=%u\n",
                                dst_chip,
                                (void*)d,
                                val);
                        }
                        __emule_fiber_wake(d);
                    }
                    continue;  // no payload advance for a seminc chunk
                }
                // Write chunk. The last write chunk's size is implicit (remaining payload).
                uint32_t csz = (i + 1 < chunk_count) ? cs[i] : (size - off);
                if (payload != nullptr && d != nullptr && csz > 0) {
                    std::memcpy(d, static_cast<const uint8_t*>(payload) + off, csz);
                    __emule_fiber_wake(d);
                }
                off += csz;
            }
            std::atomic_thread_fence(std::memory_order_release);
            break;
        }
        default: {
            // emule expresses multicast through the target list above, so 5-8 are expected to arrive
            // already fanned out. Reaching here means the op itself is unimplemented, and there is no
            // safe continuation: the destination word never changes, so a consumer waiting on it
            // blocks until a watchdog fires, far from the cause. Fail the dispatch instead — the
            // fiber's fault names the type and the run stops here rather than hanging elsewhere.
            TT_THROW(
                "EMULE fabric: send type {} has no delivery op, so this write would be dropped and its "
                "consumer would wait forever. Implement the terminal op for this type, or route it "
                "through the target list as a fanned-out unicast.",
                static_cast<unsigned>(noc_send_type));
        }
    }
}

// Top-level teleport: decode the real-layout packet header (NocCommandFields @0, payload_size @40,
// noc_send_type @42), resolve the destination chip(s), and apply the terminal NOC command. payload may be
// null for header-only commands (e.g. a bare atomic-inc). See tt-emule docs/fabric-ccl-emulation.md.
extern "C" void __emule_fabric_teleport(const void* packet_header, const void* payload, uint32_t payload_size) {
    emule_require_self(__func__);
    const uint8_t* h = static_cast<const uint8_t*>(packet_header);
    if (h == nullptr) {
        return;
    }
    const uint16_t hdr_payload_size = *reinterpret_cast<const uint16_t*>(h + 40);
    const uint8_t noc_send_type = *(h + 42);
    const uint32_t size = payload_size ? payload_size : hdr_payload_size;
    const uint32_t src_chip = __emule_self->chip_id;
    const std::vector<uint32_t> targets = __emule_fabric_resolve_targets(h, src_chip);
    static const bool dbg = std::getenv("EMULE_FABRIC_DEBUG") != nullptr;
    if (dbg) {
        const uint64_t noc_address = *reinterpret_cast<const uint64_t*>(h + 0);
        std::string ts;
        for (auto t : targets) {
            ts += " " + std::to_string(t);
        }
        fprintf(
            stderr,
            "[EMULE_FABRIC] teleport src=%u targets=[%s ] (neighbor=%u) send_type=%u noc_addr=0x%llx "
            "payload_size=%u\n",
            src_chip,
            ts.c_str(),
            __emule_fabric_neighbor(src_chip),
            noc_send_type,
            (unsigned long long)noc_address,
            size);
        // Route-table self-consistency: dump this src chip's per-direction distance walk (once per src).
        static std::mutex dump_mu;
        static std::unordered_map<uint32_t, bool> dumped;
        std::lock_guard<std::mutex> lk(dump_mu);
        if (!dumped[src_chip]) {
            dumped[src_chip] = true;
            const char* dn[4] = {"N", "E", "S", "W"};
            for (int d = 0; d < 4; ++d) {
                const auto& w = __emule_fabric_walk(src_chip, static_cast<tt::tt_fabric::RoutingDirection>(d));
                std::string s;
                for (auto c : w) {
                    s += " " + std::to_string(c);
                }
                fprintf(stderr, "[EMULE_FABRIC]   route src=%u dir=%s walk=[%s ]\n", src_chip, dn[d], s.c_str());
            }
        }
    }
    // EMULE_FABRIC_XFER=<file>: one compact line per payload-carrying send — the SOURCE L1
    // offset (bridge_l1-relative, so it is directly comparable to a kernel's get_write_ptr),
    // the sending core, each destination chip + destination L1 offset, and the payload's first
    // word. Enough to reconstruct a whole CCL's wiring offline and diff it against the ring the
    // op intends, which is how a misrouting bug gets localized without a device.
    //
    // Deliberately much cheaper than EMULE_FABRIC_DEBUG: that one is too verbose to leave on
    // without perturbing the very race under study.
    //
    // Caveat: the fopen/fprintf/fclose widens the window between reading the payload word and
    // the delivery memcpy, so under an active race the logged first word can disagree with the
    // bytes actually delivered. Trust the addresses and targets from this trace; get delivered
    // contents from a tensor dump.
    if (payload != nullptr && size > 0) {
        static const char* xfer_path = std::getenv("EMULE_FABRIC_XFER");
        if (xfer_path != nullptr) {
            const uint64_t noc_address = *reinterpret_cast<const uint64_t*>(h + 0);
            const uint64_t src_off =
                static_cast<uint64_t>(reinterpret_cast<const uint8_t*>(payload) - __emule_self->bridge_l1);
            uint32_t w0 = 0;
            std::memcpy(&w0, payload, sizeof(uint32_t));
            std::string ts;
            for (auto t : targets) {
                ts += " " + std::to_string(t);
            }
            static std::mutex xfer_mu;
            std::lock_guard<std::mutex> lk(xfer_mu);
            FILE* fp = std::fopen(xfer_path, "a");
            if (fp != nullptr) {
                std::fprintf(
                    fp,
                    "[XFER] src_chip=%u core=(%u,%u) proc=%u src_off=0x%llx size=%u "
                    "dst_noc=0x%llx w0=0x%08x targets=[%s ]\n",
                    src_chip,
                    (unsigned)__emule_self->core->logical_x,
                    (unsigned)__emule_self->core->logical_y,
                    (unsigned)__emule_self->processor_id,
                    (unsigned long long)src_off,
                    size,
                    (unsigned long long)noc_address,
                    w0,
                    ts.c_str());
                std::fclose(fp);
            }
        }
    }
    // One target for unicast; the line members for a multicast. Replay the terminal NOC op to each.
    for (uint32_t dst_chip : targets) {
        __emule_fabric_deliver(dst_chip, h, payload, size, noc_send_type, dbg);
    }
}
}  // namespace tt::tt_metal::emule
