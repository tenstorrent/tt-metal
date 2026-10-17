// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/experimental/virtual_cables.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <future>
#include <map>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include <mpi.h>
#include <sys/resource.h>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/hd_socket_descriptor.hpp>
#include "impl/context/metal_env_impl.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_metal/distributed/mesh_device_impl.hpp"
#include "tt_metal/fabric/bridged_cables.hpp"

#include "hostdevcommon/erisc_bridge_layout.h"
#include "tt_metal/distributed/erisc_bridge_doorbell.hpp"
#include "tt_metal/distributed/erisc_bridge_placement.hpp"
#include "tt_metal/distributed/erisc_bridge_region.hpp"
#include "tt_metal/distributed/erisc_credit_relay.hpp"
#include "tt_metal/distributed/erisc_e2h_leg.hpp"
#include "tt_metal/distributed/erisc_h2e_leg.hpp"
#include "tt_metal/distributed/erisc_h2h_socket.hpp"
#include "tt_metal/distributed/host_rdma_window.hpp"

namespace tt::tt_metal::experimental {

namespace eb = tt::tt_fabric::erisc_bridge;
using tt::tt_fabric::BridgeArena;
using tt::tt_fabric::BridgeCreditWord;
using tt::tt_fabric::FabricNodeId;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshCoordinateRange;
using tt::tt_metal::distributed::MeshDevice;

// Named, not anonymous: Impl has external linkage and holds these (see erisc_h2h_detail).
namespace virtual_cables_detail {

using Key = std::pair<FabricNodeId, tt::tt_fabric::chan_id_t>;

// Where far-receiver credits for one sender channel go: a stream register or an L1 running-total counter.
struct SendCreditTarget {
    bool use_streams = false;
    std::uint32_t completed_stream = 0;
    std::uint32_t acked_stream = 0;
    std::uint32_t completions_addr = 0;
    std::uint32_t acks_addr = 0;
    std::uint32_t injected_completions = 0;
    std::uint32_t injected_acks = 0;
};

struct End {
    FabricNodeId node{tt::tt_fabric::MeshId{0}, 0};
    tt::tt_fabric::chan_id_t chan = 0;
    FabricNodeId peer_node{tt::tt_fabric::MeshId{0}, 0};
    tt::tt_fabric::chan_id_t peer_chan = 0;
    std::uint32_t peer_rank = 0;
    IDevice* dev = nullptr;
    CoreCoord eth{0, 0};
    CoreCoord eth_noc0{0, 0};
    MeshCoordinate coord{0, 0};
    tt::tt_fabric::experimental::BridgedRouterLayout layout;  // this end's router, as built
    std::vector<std::uint32_t> far_vc;                        // by FAR sender channel: its VC
    std::uint32_t arena_base = 0;                             // this end's arenas are [arena_base, arena_base + arenas)
    std::uint32_t arenas = 0;                                 // 0: unbridged (fabric has no router on this channel)
    std::unique_ptr<E2HLeg> e2h;
    std::vector<std::unique_ptr<H2ELeg>> h2e;  // by VC; null where this router has no receiver for it
    std::vector<SendCreditTarget> send;        // by sender channel of THIS router
    std::array<std::uint32_t, eb::kBridgeChannels> relayed_completions{}, relayed_acks{};  // by FAR sender channel
    bool armed = false;
    // Counters: frames sent / consumed far away; frames injected here / taken by the router; credit totals.
    std::uint64_t forwarded = 0, retired = 0, injected = 0, landed = 0, credits_relayed = 0, credits_injected = 0;
};

std::int64_t env_int(const char* name, std::int64_t fallback) {
    const char* v = std::getenv(name);
    return v != nullptr ? std::strtoll(v, nullptr, 0) : fallback;
}

constexpr std::size_t kRecordWords = 1 + eb::kBridgeChannels;  // per bridged router, exchanged once at create
constexpr std::uint32_t kMaxPerPoll = 64;  // frames per end per pass, so one busy end cannot starve the rest

}  // namespace virtual_cables_detail

using virtual_cables_detail::End;
using virtual_cables_detail::Key;

struct VirtualCableBridge::Impl {
    tt::tt_metal::MetalEnvImpl* env = nullptr;
    std::int64_t stall_seconds = 60;  // <= 0 disables
    std::int64_t teardown_seconds = 60;
    int rank = 0;
    std::uint32_t capacity = 0, page = 0, ring = 0;

    std::unique_ptr<eb::EriscBridgeRegion> region;
    std::unique_ptr<eb::BridgeArenaAlias> alias;
    std::vector<std::uint8_t> empty_region;  // a rank with no ends still joins the collective window
    std::unique_ptr<EriscH2HSocket> h2h;
    std::vector<End> ends;
    std::vector<std::uint32_t> arena_end;  // by local arena: its end

    mutable std::mutex mu;
    std::thread thread;
    std::atomic<bool> running{false};
    std::string err;
    std::uint64_t last_pending = 0;
    std::chrono::steady_clock::time_point last_progress = std::chrono::steady_clock::now();

    void fail(const std::string& why) {
        if (err.empty()) {
            err = why;
            log_error(tt::LogFabric, "Virtual cable bridge (rank {}): {}", rank, why);
        }
    }

    bool set_armed(End& e, bool on) {
        const std::uint32_t v = on ? eb::kBridgeHostArmed : 0u;
        const auto addr = e.layout.block_addr + tt::tt_fabric::kStatusOffset + offsetof(eb::BridgeStatus, host_armed);
        const bool ok = eb::write_bytes_to_l1(*env, e.dev, e.eth, addr, &v, sizeof(v));
        e.armed = ok && on;
        return ok;
    }

    std::string describe_locked() const {
        std::string s =
            fmt::format("virtual cable bridge, rank {}: {} end(s), ring {} x {} B", rank, ends.size(), ring, page);
        for (const auto& e : ends) {
            s += fmt::format(
                "\n  {} chan {} <-> {} chan {} (rank {}), {} arena(s): forwarded {}, retired {}, injected {}, "
                "landed {}, credits relayed {}, injected {}",
                e.node,
                e.chan,
                e.peer_node,
                e.peer_chan,
                e.peer_rank,
                e.arenas,
                e.forwarded,
                e.retired,
                e.injected,
                e.landed,
                e.credits_relayed,
                e.credits_injected);
        }
        if (h2h != nullptr) {
            s += "\n  " + h2h->describe();
        }
        return err.empty() ? s : s + "\n  error: " + err;
    }

    std::uint32_t poll_once();
};

VirtualCableBridge::VirtualCableBridge() : impl_(std::make_unique<Impl>()) {}

VirtualCableBridge::~VirtualCableBridge() {
    auto& im = *impl_;
    if (im.running.exchange(false) && im.thread.joinable()) {
        im.thread.join();
    }
    for (auto& e : im.ends) {
        if (e.armed) {
            (void)im.set_armed(e, false);  // an armed router keeps writing into sockets about to go away
        }
    }
    // H2H frees its window collectively; if a peer rank died that never returns, so end the job instead of hanging.
    auto freed = std::async(std::launch::async, [&im] { im.h2h.reset(); });
    if (freed.wait_for(std::chrono::seconds(im.teardown_seconds)) != std::future_status::ready) {
        log_fatal(tt::LogFabric, "Virtual cable bridge (rank {}): teardown stuck; a peer rank is gone", im.rank);
        MPI_Abort(MPI_COMM_WORLD, 76);
    }
    im.ends.clear();
    if (im.region) {
        im.region->release();  // unpin before the overlays (alias, then region) go: the pin names those pages
    }
}

std::unique_ptr<VirtualCableBridge> VirtualCableBridge::create(
    const std::shared_ptr<MeshDevice>& mesh, std::string& err) {
    namespace d = virtual_cables_detail;
    err.clear();
    auto& env = mesh->impl().metal_env();
    const auto& cp = env.get_control_plane();
    const auto& all_ends = cp.bridged_cables().ends;
    if (all_ends.empty()) {
        return nullptr;  // global, so every rank returns here together
    }

    std::unique_ptr<VirtualCableBridge> bridge(new VirtualCableBridge());
    auto& im = *bridge->impl_;
    im.env = &env;
    im.stall_seconds = d::env_int("TT_METAL_E2H_BRIDGE_STALL_SECONDS", 60);
    im.teardown_seconds = std::max<std::int64_t>(d::env_int("TT_METAL_E2H_BRIDGE_TEARDOWN_SECONDS", 60), 1);
    im.ring = static_cast<std::uint32_t>(
        d::env_int("TT_METAL_E2H_BRIDGE_RING_PAGES", tt::tt_fabric::kBridgeDefaultRingPages));
    int world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &im.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    const auto my_rank = static_cast<std::uint32_t>(im.rank);

    // Every rank's ends, in the same order everywhere, so a route can name a peer's arena without asking.
    std::map<std::uint32_t, std::vector<Key>> by_rank;
    for (const auto& [key, end] : all_ends) {
        by_rank[end.rank].push_back(key);
    }
    std::string local_err;

    // ---- this rank's ends ----
    for (const auto& key : by_rank[my_rank]) {
        const auto& info = all_ends.at(key);
        End e{.node = key.first, .chan = key.second, .peer_node = info.peer_node, .peer_chan = info.peer_chan};
        e.peer_rank = info.peer_rank;
        for (const auto& coord : MeshCoordinateRange(mesh->shape())) {
            if (mesh->is_local(coord) && mesh->get_fabric_node_id(coord) == e.node) {
                e.coord = coord;
                e.dev = mesh->get_device(coord);
                break;
            }
        }
        if (e.dev == nullptr) {
            // The chip is not open, yet its cable's other end may hold traffic for it: refuse rather than hang.
            local_err = fmt::format(
                "{} chan {} is bridged but its chip is not in the opened mesh; open the whole mesh", e.node, e.chan);
        } else {
            const auto& soc = env.get_cluster().get_soc_desc(e.dev->id());
            const auto lg = soc.get_eth_core_for_channel(e.chan, tt::CoordSystem::LOGICAL);
            const auto n0 = soc.get_eth_core_for_channel(e.chan, tt::CoordSystem::NOC0);
            e.eth = CoreCoord(lg.x, lg.y);
            e.eth_noc0 = CoreCoord(n0.x, n0.y);
        }
        if (auto layout = e.dev != nullptr ? cp.bridged_cables().layout(e.node, key.second) : std::nullopt) {
            e.layout = std::move(*layout);  // none: fabric has no router on this channel, so the cable stays unbridged
            if (e.layout.sender_vc.size() > eb::kBridgeChannels) {  // block room is checked at router build
                local_err = fmt::format("{} chan {}: more sender channels than the bridge block holds", e.node, e.chan);
            } else if (im.capacity != 0 && im.capacity != e.layout.slot_bytes) {
                local_err = "bridged routers disagree on the channel slot size";
            }
            im.capacity = e.layout.slot_bytes;
        }
        im.ends.push_back(std::move(e));
    }

    // ---- to every rank, per end: [sender channels + 1 (0: no router), their VCs], then the slot size ----
    std::map<Key, std::size_t> index_of;
    std::uint64_t ends_hash = 0xcbf29ce484222325ull;  // every rank must index the same ends, or the table misaligns
    for (const auto& entry : all_ends) {
        index_of.emplace(entry.first, index_of.size());
        ends_hash = (ends_hash ^ ((std::uint64_t{*entry.first.first.mesh_id} << 40) |
                                  (std::uint64_t{entry.first.first.chip_id} << 8) | entry.first.second)) *
                    0x100000001b3ull;
    }
    if (std::string agree_err; !RdmaWindow::agree_value(ends_hash, agree_err)) {
        err = "ranks disagree on the bridged cable ends: " + agree_err;
        return nullptr;  // the verdict is the same on every rank, so all return here together
    }
    std::vector<std::uint32_t> table(index_of.size() * d::kRecordWords + 1, 0);
    auto record = [&](const Key& key) { return &table[index_of.at(key) * d::kRecordWords]; };
    for (const auto& e : im.ends) {
        if (e.layout.block_addr != 0) {
            auto* r = record({e.node, e.chan});
            const auto n = std::min<std::size_t>(e.layout.sender_vc.size(), eb::kBridgeChannels);
            r[0] = static_cast<std::uint32_t>(n) + 1;
            std::copy_n(e.layout.sender_vc.begin(), n, r + 1);
        }
    }
    table.back() = im.capacity;  // a rank with no ends still needs the agreed slot size
    MPI_Allreduce(MPI_IN_PLACE, table.data(), static_cast<int>(table.size()), MPI_UINT32_T, MPI_MAX, MPI_COMM_WORLD);
    // Arenas per end: the larger end's sender-channel count; 0 if either end has no router.
    auto arenas_for = [&](const Key& end, const Key& peer) -> std::uint32_t {
        const std::uint32_t a = record(end)[0], b = record(peer)[0];
        return a == 0 || b == 0 ? 0 : std::max(a, b) - 1;
    };
    std::map<Key, std::uint32_t> base_of;             // every end's first arena on its rank
    std::map<std::uint32_t, std::uint32_t> total_of;  // every rank's arena count
    for (const auto& [rank, keys] : by_rank) {
        for (const auto& key : keys) {
            const auto& info = all_ends.at(key);
            base_of[key] = total_of[rank];
            total_of[rank] += arenas_for(key, Key{info.peer_node, info.peer_chan});
        }
    }
    const std::uint32_t arenas = total_of[my_rank];
    for (std::uint32_t k = 0; k < im.ends.size(); ++k) {
        auto& e = im.ends[k];
        const Key key{e.node, e.chan};
        const Key peer{e.peer_node, e.peer_chan};
        e.arena_base = base_of[key];
        e.arenas = arenas_for(key, peer);
        if (e.arenas != 0) {
            e.far_vc.assign(record(peer) + 1, record(peer) + record(peer)[0]);
        }
        im.arena_end.insert(im.arena_end.end(), e.arenas, k);
    }
    im.page = tt::tt_fabric::bridge_socket_page_bytes(im.capacity);

    // ---- one region, laid out flat: exactly this rank's arenas ----
    auto reserve = [&] {  // one binding per arena; the region only counts them (routing lives in H2H's routes)
        const eb::EriscBridgeRegion::Geometry geom{.packet_capacity = im.capacity, .ring_pages = im.ring};
        im.region = std::make_unique<eb::EriscBridgeRegion>();
        (void)im.region->reserve(std::vector<eb::LinkBinding>(arenas), geom, local_err);  // sets local_err on failure
    };
    // An unset ring halves until the region fits ulimit -l; every rank then takes the smallest (H2H agrees on it).
    rlimit lim{};
    const std::uint64_t memlock =
        ::getrlimit(RLIMIT_MEMLOCK, &lim) == 0 && lim.rlim_cur != RLIM_INFINITY ? lim.rlim_cur : UINT64_MAX;
    if (local_err.empty() && arenas != 0) {
        reserve();
        while (std::getenv("TT_METAL_E2H_BRIDGE_RING_PAGES") == nullptr && local_err.empty() && im.ring > 2 &&
               im.region->region_bytes() > memlock) {
            im.ring /= 2;
            reserve();
        }
    }
    const std::uint32_t mine = im.ring;
    MPI_Allreduce(MPI_IN_PLACE, &im.ring, 1, MPI_UINT32_T, MPI_MIN, MPI_COMM_WORLD);
    if (im.ring != mine && local_err.empty() && arenas != 0) {
        reserve();
    }

    // ---- per end: disarm, clear the last process's socket cursors, open sockets over its Tx arenas ----
    std::vector<std::vector<std::unique_ptr<distributed::D2HSocket>>> socks(im.ends.size());
    std::vector<eb::BridgeArenaAlias::Slot> slots;
    for (std::size_t k = 0; k < im.ends.size() && local_err.empty(); ++k) {
        auto& e = im.ends[k];
        if (e.arenas == 0) {
            continue;
        }
        const std::array<std::uint32_t, 16> zeros{};
        bool ok = im.set_armed(e, false);
        try {
            for (std::uint32_t ch = 0; ch < e.layout.sender_vc.size() && ok; ++ch) {
                const std::uint32_t addr = e.layout.block_addr + tt::tt_fabric::bridge_socket_config_offset(ch);
                if (!(ok = eb::write_bytes_to_l1(env, e.dev, e.eth, addr, zeros.data(), sizeof(zeros)))) {
                    break;
                }
                distributed::D2HSocket::ExternalConfigBuffer ecb{addr, HalProgrammableCoreType::ACTIVE_ETH};
                auto& s = socks[k].emplace_back(std::make_unique<distributed::D2HSocket>(
                    mesh, distributed::MeshCoreCoord(e.coord, e.eth_noc0), im.page * im.ring, ecb));
                s->set_page_size(im.page);
                const auto sd = s->populate_descriptor();
                slots.push_back(
                    {e.arena_base + ch, BridgeArena::Tx, sd.shm_name, sd.shm_size, sd.data_offset, sd.fifo_size});
            }
        } catch (const std::exception& ex) {
            local_err = fmt::format("D2HSocket on {} chan {}: {}", e.node, e.chan, ex.what());
        }
        if (!ok) {
            local_err = fmt::format("could not disarm {} chan {} or clear its socket configs", e.node, e.chan);
        }
    }
    if (local_err.empty() && arenas != 0) {
        im.alias = eb::BridgeArenaAlias::map(*im.region, slots, local_err);  // both set local_err on failure
        const auto first = std::find_if(im.ends.begin(), im.ends.end(), [](const End& e) { return e.arenas != 0; });
        if (im.alias != nullptr) {
            (void)im.region->provision(mesh, first->coord, local_err);
        }
    }

    // ---- legs and credit targets ----
    for (std::size_t k = 0; k < im.ends.size() && local_err.empty(); ++k) {
        auto& e = im.ends[k];
        if (e.arenas == 0) {
            continue;
        }
        E2HLeg::Config c;
        c.arenas = arenas;
        c.chans_per_link = e.arenas;
        c.packet_capacity = im.capacity;
        c.ring_pages = im.ring;
        c.arena_base = e.arena_base;
        c.alias_region_base = im.region->base();
        if ((e.e2h = E2HLeg::create(std::move(socks[k]), c, local_err)) == nullptr) {
            break;
        }
        e.h2e.resize(e.layout.receiver_channel.size());
        for (std::uint32_t vc = 0; vc < e.layout.receiver_channel.size() && local_err.empty(); ++vc) {
            if (e.layout.receiver_channel[vc] == tt::tt_fabric::experimental::BridgedRouterLayout::kNoReceiver) {
                continue;
            }
            H2ELeg::Config h;
            h.page_bytes = im.page;
            h.eth_chan = e.chan;
            h.mesh_row = e.coord[0];
            h.mesh_col = e.coord[1];
            h.receiver_vc = vc;
            h.receiver_channel = e.layout.receiver_channel[vc];
            h.rx_base = e.layout.receiver_base[vc];
            h.rx_slots = e.layout.receiver_slots[vc];
            h.slot_bytes = e.layout.slot_bytes;
            h.in_far_slot_order = true;  // several far sender channels share this ring; keep the cable's order
            h.alias_region_base = im.region->base();
            e.h2e[vc] = H2ELeg::create(mesh, h, local_err);
        }
        const auto mesh_id = *e.node.mesh_id;
        e.send.resize(e.layout.sender_vc.size());
        for (std::uint32_t ch = 0; ch < e.send.size() && local_err.empty(); ++ch) {
            auto& t = e.send[ch];
            const auto vc = e.layout.sender_vc[ch];
            const auto idx = e.layout.sender_index_in_vc[ch];
            t.use_streams = eb::sender_credits_use_streams(cp, mesh_id, vc, idx);
            if (t.use_streams) {
                t.completed_stream = eb::sender_pkts_completed_stream(cp, mesh_id, vc, idx);
                t.acked_stream = eb::sender_pkts_acked_stream(cp, mesh_id, vc, idx);
                if (!eb::stream_id_valid_for_eth(t.completed_stream) || !eb::stream_id_valid_for_eth(t.acked_stream)) {
                    local_err = "sender credit stream id outside the ERISC's 32";
                }
            } else if (e.layout.completion_counters_addr == 0 || e.layout.ack_counters_addr == 0) {
                local_err = "sender credit counters are not reserved in this fabric config";
            } else {  // one uint32 per sender channel
                t.completions_addr = e.layout.completion_counters_addr + ch * 4u;
                t.acks_addr = e.layout.ack_counters_addr + ch * 4u;
            }
        }
    }

    // ---- H2H: collective. A failed rank joins with a null region, so every rank gets the same verdict. ----
    EriscH2HSocket::Config hc;
    hc.routed = true;
    hc.arenas = arenas;
    hc.ring_pages = im.ring;
    hc.host_rank = my_rank;
    hc.host_count = static_cast<std::uint32_t>(world);
    hc.max_put_bytes = 32u * 1024u;  // keep coalesced puts inside a typical eager limit
    for (const auto& e : im.ends) {
        const Key peer{e.peer_node, e.peer_chan};
        for (std::uint32_t ch = 0; ch < e.arenas; ++ch) {
            hc.routes.push_back(EriscH2HSocket::Config::Route{e.peer_rank, base_of[peer] + ch, total_of[e.peer_rank]});
        }
    }
    if (local_err.empty() && arenas == 0) {
        im.empty_region.assign(4096, 0);
        hc.region_base = im.empty_region.data();
        hc.region_bytes = im.empty_region.size();
    } else if (local_err.empty()) {  // no error, so the region was reserved
        hc.region_base = im.region->base();
        hc.region_bytes = im.region->region_bytes();
    }
    hc.page_bytes = tt::tt_fabric::bridge_socket_page_bytes(table.back());
    std::string herr;
    im.h2h = EriscH2HSocket::create(hc, herr);
    if (im.h2h == nullptr) {
        err = !local_err.empty() ? local_err : herr;
        return nullptr;
    }

    // ---- arm, then agree, so no rank moves frames for a peer that is not ready ----
    bool armed = true;
    for (auto& e : im.ends) {
        if (e.arenas != 0) {
            armed = im.set_armed(e, true) && armed;
        }
    }
    std::string agree_err;
    if (!RdmaWindow::agree(armed, agree_err)) {
        err = armed ? "another rank could not arm its bridged routers" : "could not arm the bridged routers";
        return nullptr;
    }
    (void)im.h2h->barrier();

    log_info(
        tt::LogFabric,
        "Virtual cable bridge: rank {} serves {} cable end(s), {} arena(s), ring {} x {} B, {:.1f} MiB pinned.",
        im.rank,
        std::count_if(im.ends.begin(), im.ends.end(), [](const End& e) { return e.arenas != 0; }),
        arenas,
        im.ring,
        im.page,
        im.region != nullptr ? static_cast<double>(im.region->region_bytes()) / (1 << 20) : 0.0);

    im.running = true;
    im.thread = std::thread([&im] {
        while (im.running.load(std::memory_order_relaxed)) {
            std::uint32_t moved = 0;
            std::string state;
            {
                std::lock_guard<std::mutex> lock(im.mu);
                moved = im.poll_once();
                if (!im.err.empty()) {
                    state = im.describe_locked();
                }
            }
            if (!state.empty()) {
                // A stopped bridge hangs fabric on every rank, so end the job with the state.
                log_fatal(tt::LogFabric, "Virtual cable bridge failed; aborting.\n{}", state);
                MPI_Abort(MPI_COMM_WORLD, 75);
            }
            if (moved == 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(20));
            }
        }
    });
    return bridge;
}

// One pass over every end: E2H -> H2H -> H2E, then credits both ways, then the stall check. Returns frames moved.
std::uint32_t VirtualCableBridge::Impl::poll_once() {
    if (!err.empty()) {
        return 0;
    }
    std::uint32_t progress = 0;

    for (auto& e : ends) {
        if (e.e2h == nullptr) {
            continue;
        }
        std::uint32_t taken = 0;
        progress += e.e2h->poll([&](const BridgeSendTask& t) {
            return taken < virtual_cables_detail::kMaxPerPoll && h2h->submit(t) && ++taken != 0;
        });
        e.forwarded += taken;
    }

    // Retire what peers consumed; deliver what arrived into the receiver channel of the VC its far sender is on.
    progress += h2h->poll(
        [&](std::uint32_t a, std::uint32_t pages) {
            if (a < arena_end.size()) {
                auto& e = ends[arena_end[a]];
                e.e2h->retire(a, pages);
                e.retired += pages;
            }
        },
        [&](const BridgeDeliverTask& task) {
            if (task.arena >= arena_end.size()) {
                fail(fmt::format("frame arrived in arena {}, which this rank does not hold", task.arena));
                return false;
            }
            auto& e = ends[arena_end[task.arena]];
            const std::uint32_t ch = task.arena - e.arena_base;  // the far router's sender channel
            const auto vc = ch < e.far_vc.size() ? e.far_vc[ch] : ~0u;
            auto* leg = vc < e.h2e.size() ? e.h2e[vc].get() : nullptr;
            if (leg == nullptr) {
                fail(fmt::format("{} chan {}: no receiver channel for far sender channel {}", e.node, e.chan, ch));
                return false;
            }
            if (!leg->publish(task)) {
                return false;  // router full, or an earlier slot not here yet: re-offered next pass
            }
            h2h->consumed(task.arena, 1);
            ++e.injected;
            return true;
        });
    (void)h2h->publish_credits();

    // Receive side: what this end's receivers credited each FAR sender channel goes to that channel.
    bool published = false;
    for (auto& e : ends) {
        eb::CreditSnapshot snap;
        if (e.arenas == 0 ||
            !eb::sample_recv_credits(*env, e.dev, e.eth, e.layout.recv_credits_addr, eb::kBridgeChannels, snap)) {
            continue;
        }
        for (std::uint32_t ch = 0; ch < e.far_vc.size(); ++ch) {
            auto relay = [&](std::uint32_t total, std::uint32_t& last, BridgeCreditWord word) {
                if (total != last) {
                    e.credits_relayed += total - last;
                    last = total;
                    published = h2h->publish_word(e.arena_base + ch, word, total) || published;
                }
            };
            relay(snap.completions[ch], e.relayed_completions[ch], BridgeCreditWord::EdmCompletions);
            relay(snap.acks[ch], e.relayed_acks[ch], BridgeCreditWord::EdmAcks);
        }
    }
    if (published) {
        (void)h2h->flush_words();
    }

    // Send side: what the far end published for each of THIS router's sender channels goes into that channel.
    for (auto& e : ends) {
        for (std::uint32_t ch = 0; ch < e.send.size(); ++ch) {
            auto& t = e.send[ch];
            const auto comp =
                static_cast<std::uint32_t>(h2h->peer_word(e.arena_base + ch, BridgeCreditWord::EdmCompletions));
            const auto ack = static_cast<std::uint32_t>(h2h->peer_word(e.arena_base + ch, BridgeCreditWord::EdmAcks));
            if (comp == t.injected_completions && ack == t.injected_acks) {
                continue;
            }
            const std::uint32_t old_c = t.injected_completions, old_a = t.injected_acks;
            std::string werr;
            bool ok = false;
            if (t.use_streams) {  // updates the injected totals itself
                ok = eb::inject_sender_credits_stream(
                    *env,
                    e.dev,
                    e.eth,
                    env->get_hal(),
                    t.completed_stream,
                    t.acked_stream,
                    comp,
                    ack,
                    t.injected_completions,
                    t.injected_acks,
                    werr);
            } else if ((ok = eb::inject_sender_credits(
                            *env, e.dev, e.eth, t.completions_addr, t.acks_addr, comp, ack, werr))) {
                t.injected_completions = comp;
                t.injected_acks = ack;
            }
            if (!ok) {
                fail(fmt::format("{} chan {} sender channel {}: {}", e.node, e.chan, ch, werr));
                return progress;
            }
            e.credits_injected +=
                std::uint32_t(t.injected_completions - old_c) + std::uint32_t(t.injected_acks - old_a);
        }
    }

    // Occupancy refresh (the only path that frees H2E credit), then frames in flight for the stall check.
    std::uint64_t pending = 0;
    for (auto& e : ends) {
        e.landed = 0;
        for (auto& leg : e.h2e) {
            if (leg != nullptr) {
                e.landed += leg->drained(0);
                if (!leg->first_error().empty()) {
                    fail(fmt::format("{} chan {}: {}", e.node, e.chan, leg->first_error()));
                }
            }
        }
        pending += (e.forwarded - std::min(e.forwarded, e.retired)) + (e.injected - std::min(e.injected, e.landed));
    }
    if (h2h->failed()) {
        fail(h2h->first_error());
    }
    // Stall: frames in flight and nothing moved for too long. A healthy cable drains in microseconds.
    const auto now = std::chrono::steady_clock::now();
    if (progress != 0 || pending != last_pending || pending == 0) {
        last_progress = now;
        last_pending = pending;
    } else if (stall_seconds > 0 && now - last_progress > std::chrono::seconds(stall_seconds)) {
        fail(fmt::format(
            "stalled: {} frame(s) in flight for {} s (TT_METAL_E2H_BRIDGE_STALL_SECONDS)", pending, stall_seconds));
    }
    return progress;
}

std::string VirtualCableBridge::first_error() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->err;
}

std::string VirtualCableBridge::describe() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->describe_locked();
}

}  // namespace tt::tt_metal::experimental
