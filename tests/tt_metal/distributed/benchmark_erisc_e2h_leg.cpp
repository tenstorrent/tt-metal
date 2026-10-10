// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// E2H leg benchmark: T6 -> bridged ERISC -> host, on the eth channel TT_BRIDGE_E2H_FORCE_CHAN names.

#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/hd_socket_descriptor.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_impl.hpp"
#include "tt_metal/distributed/mesh_device_impl.hpp"

#include "hostdevcommon/erisc_bridge_layout.h"
#include "tests/tt_metal/distributed/erisc_bridge_bench_common.hpp"
#include "tests/tt_metal/distributed/erisc_bridge_bench_mesh.hpp"
#include "tests/tt_metal/distributed/erisc_bridge_bench_producer.hpp"
#include "tt_metal/distributed/erisc_bridge_placement.hpp"
#include "tt_metal/distributed/erisc_bridge_region.hpp"
#include "tt_metal/distributed/erisc_e2h_leg.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;
using namespace tt::tt_metal::experimental;
namespace eb = tt::tt_fabric::erisc_bridge;
namespace bb = tt::tt_fabric::erisc_bridge::bench;

namespace {

std::uint32_t g_chan = 0;  // the bridged eth channel, from TT_BRIDGE_E2H_FORCE_CHAN
bool g_stalled = false;    // a stalled producer never finishes, so close() would wait forever

std::uint32_t env_u32(const char* n, std::uint32_t d) {
    const char* v = std::getenv(n);
    return v != nullptr ? static_cast<std::uint32_t>(std::strtoul(v, nullptr, 0)) : d;
}

// Built in the one order that works: socket, then alias, then pin, then leg.
struct Bound {
    ~Bound() {  // unpin before the alias unmaps the pages the pin names
        leg.reset();
        region.release();
    }
    eb::EriscBridgeRegion region;
    std::unique_ptr<eb::BridgeArenaAlias> alias;
    std::unique_ptr<E2HLeg> leg;
    std::uint32_t ceiling = 0;
    std::uint32_t chan_slot = 0;
    std::uint32_t producer_link = 0;  // g_chan's index among the channels forwarding to dst_node
    tt::tt_fabric::FabricNodeId src_node{tt::tt_fabric::MeshId{0}, 0};
    tt::tt_fabric::FabricNodeId dst_node{tt::tt_fabric::MeshId{0}, 0};
    CoreCoord eth_logical{0, 0};
    std::uint32_t status_addr = 0;
    std::string err;
};

// The router opens its sockets only while armed, so a config is never read half-written.
bool arm_router(IDevice* dev, const Bound& b, bool armed) {
    std::vector<std::uint32_t> v{armed ? eb::kBridgeHostArmed : 0u};  // non-const: the API takes a mutable ref
    return tt::tt_metal::detail::WriteToDeviceL1(
        dev, b.eth_logical, b.status_addr + offsetof(eb::BridgeStatus, host_armed), v, tt::CoreType::ETH);
}

bool bind_leg(const std::shared_ptr<MeshDevice>& mesh, const E2HLeg::Config& cfg, std::uint32_t nchan, Bound& out) {
    auto& env = mesh->impl().metal_env();
    const auto& cp = env.get_control_plane();
    auto* dev = mesh->get_device(MeshCoordinate(0, 0));
    const auto& soc = env.get_cluster().get_soc_desc(dev->id());

    const eb::EriscBridgePlacement place(eb::router_config(cp));
    const auto p = place.place();
    if (!p.ok) {
        out.err = "placement refused: " + p.why;
        return false;
    }
    out.ceiling = place.credit_ceiling_frames();
    out.chan_slot = place.channel_slot_bytes();
    out.status_addr = p.status_addr;

    // The producer aims at the chip across the bridged cable, so its frames cross that router.
    out.src_node = cp.get_fabric_node_id_from_physical_chip_id(static_cast<ChipId>(dev->id()));
    const auto chan = static_cast<tt::tt_fabric::chan_id_t>(g_chan);
    const auto peer = cp.try_get_connected_mesh_chip_chan_ids(out.src_node, chan);
    if (!peer.has_value()) {
        out.err = "TT_BRIDGE_E2H_FORCE_CHAN=" + std::to_string(g_chan) + " has no connected peer";
        return false;
    }
    out.dst_node = peer->first;
    // The list append_fabric_connection_rt_args indexes, or the producer connects to another router.
    const auto dir = cp.get_forwarding_direction(out.src_node, out.dst_node);
    const auto fwd = dir ? cp.get_active_fabric_eth_channels_in_direction(out.src_node, *dir)
                         : std::vector<tt::tt_fabric::chan_id_t>{};
    const auto it = std::find(fwd.begin(), fwd.end(), chan);
    if (it == fwd.end()) {
        out.err = "eth_chan " + std::to_string(g_chan) + " does not forward to its own peer";
        return false;
    }
    out.producer_link = static_cast<std::uint32_t>(std::distance(fwd.begin(), it));
    // LOGICAL for L1 writes, NOC0 for the socket; mixing them throws in the coordinate manager.
    const auto eth_noc0 = soc.get_eth_core_for_channel(g_chan, tt::CoordSystem::NOC0);
    const auto eth_logical = soc.get_eth_core_for_channel(g_chan, tt::CoordSystem::LOGICAL);
    out.eth_logical = CoreCoord(eth_logical.x, eth_logical.y);
    // Disarm before touching a socket config: the router must not open one the host is still writing.
    if (!arm_router(dev, out, false)) {
        out.err = "could not disarm the bridged router";
        return false;
    }

    // Reserve before any overlay: MAP_FIXED after the pin silently replaces pinned pages.
    std::vector<eb::LinkBinding> links(1);
    links[0].local_chan = static_cast<std::uint8_t>(g_chan);
    eb::EriscBridgeRegion::Geometry geom;
    geom.packet_capacity = out.chan_slot;  // the router's capacity fixes the slot stride, not packet_size
    geom.ring_pages = cfg.ring_pages;
    geom.chans_per_link = nchan;
    if (out.region.reserve(links, geom, out.err) == nullptr) {
        return false;
    }

    const std::uint32_t page = tt::tt_fabric::bridge_socket_page_bytes(out.chan_slot);
    std::vector<std::unique_ptr<D2HSocket>> socks;
    std::vector<eb::BridgeArenaAlias::Slot> slots;
    for (std::uint32_t i = 0; i < nchan; ++i) {
        const std::uint32_t cfg_addr = p.socket_config_addr(i);
        // A previous run's cursor survives in L1 and the socket constructor does not reset it.
        if (!bb::zero_socket_config(dev, out.eth_logical, cfg_addr)) {
            out.err = "could not zero the socket config for channel " + std::to_string(i);
            return false;
        }
        try {
            D2HSocket::ExternalConfigBuffer ecb{cfg_addr, HalProgrammableCoreType::ACTIVE_ETH};
            const MeshCoreCoord core(MeshCoordinate(0, 0), CoreCoord(eth_noc0.x, eth_noc0.y));
            socks.push_back(std::make_unique<D2HSocket>(mesh, core, page * cfg.ring_pages, ecb));
            socks.back()->set_page_size(page);
        } catch (const std::exception& e) {
            out.err = "D2HSocket[" + std::to_string(i) + "]: " + e.what();
            return false;
        }
        // Alias the socket's named shm onto its Tx arena; the hugepage path has no name to alias.
        const auto desc = socks.back()->populate_descriptor();
        if (desc.shm_name.empty()) {
            out.err = "socket " + std::to_string(i) + " is not CrossProcess-backed, so it cannot be aliased";
            return false;
        }
        slots.push_back(
            {tt::tt_fabric::bridge_arena_index(cfg.link_idx, i, nchan),
             tt::tt_fabric::BridgeArena::Tx,
             desc.shm_name,
             desc.shm_size,
             desc.data_offset,
             desc.fifo_size});
    }
    out.alias = eb::BridgeArenaAlias::map(out.region, slots, out.err);
    if (!out.alias || !out.region.provision(mesh, /*chip=*/0, out.err)) {
        return false;
    }
    E2HLeg::Config bound = cfg;
    bound.alias_region_base = out.region.base();
    bound.packet_capacity = out.chan_slot;
    bound.arenas = out.region.arena_count();
    bound.chans_per_link = nchan;
    out.leg = E2HLeg::create(std::move(socks), bound, out.err);
    return out.leg != nullptr && arm_router(dev, out, true);
}

std::string router_counters(IDevice* dev, const Bound& b) {
    eb::BridgeStatus s{};
    std::string ignored;
    if (!bb::bridge_router_armed(dev, b.eth_logical, b.status_addr, s, ignored)) {
        return "";
    }
    return " [router frames=" + std::to_string(s.frames) + " declined=" + std::to_string(s.declined) +
           " blocked_rx=" + std::to_string(s.blocked_rx) + " blocked_nodata=" + std::to_string(s.blocked_nodata) +
           " free_slots=" + std::to_string(s.free_slots) + "]";
}

struct E2HLegFixture : public benchmark::Fixture {
    void SetUp(benchmark::State& state) override {
        payload_bytes_ = static_cast<std::uint32_t>(state.range(bb::kPacketSize));
        verify_ = state.range(bb::kVerify) != 0;
        batch_ = std::max<std::uint32_t>(1, static_cast<std::uint32_t>(state.range(bb::kLegSpecific0)));
        // TT_BRIDGE_E2H_MIB overrides total_amt for soak runs too large to register as a case.
        const auto mib = env_u32("TT_BRIDGE_E2H_MIB", static_cast<std::uint32_t>(state.range(bb::kTotalAmt)));
        const std::uint64_t amt = mib * bb::kMiB;
        measured_ = static_cast<std::uint32_t>(amt / payload_bytes_);
        warmup_ = static_cast<std::uint32_t>(static_cast<std::uint64_t>(measured_) * state.range(bb::kWarmupPct) / 100);
        total_ = measured_ + warmup_;
    }

    std::uint32_t payload_bytes_ = 0;
    std::uint32_t measured_ = 0;
    std::uint32_t warmup_ = 0;
    std::uint32_t total_ = 0;
    bool verify_ = false;
    std::uint32_t batch_ = 1;
};

}  // namespace

BENCHMARK_DEFINE_F(E2HLegFixture, Bridge)(benchmark::State& state) {
    auto mesh = bb::the_mesh();
    if (g_stalled) {  // its producer still holds the worker core, so this case would queue behind it
        state.SkipWithError("skipped: an earlier case in this process stalled");
        return;
    }
    if (bb::sweep_requested(state)) {
        bb::sweep_not_implemented(state);  // one forced channel: nothing to sweep
        return;
    }
    E2HLeg::Config cfg;
    cfg.ring_pages = env_u32("TT_BRIDGE_RING_PAGES", tt::tt_fabric::kBridgeDefaultRingPages);
    const std::uint32_t nchan = std::max<std::uint32_t>(1, env_u32("TT_BRIDGE_E2H_CHANNELS", 1));
    Bound b;
    if (!bind_leg(mesh, cfg, nchan, b)) {
        state.SkipWithError(b.err.c_str());
        return;
    }
    // Payload plus fabric header must fit the channel slot, or the T6's send is rejected upstream.
    if (payload_bytes_ + 64 > b.chan_slot) {
        state.SkipWithError(("packet_size " + std::to_string(payload_bytes_) + " + header exceeds the " +
                             std::to_string(b.chan_slot) + " B channel slot")
                                .c_str());
        return;
    }
    auto& leg = *b.leg;
    const std::uint32_t want = total_;

    // A stock router forwards over the cable and drains nothing, so check the bridge is on first.
    auto* dev = mesh->get_device(MeshCoordinate(0, 0));
    eb::BridgeStatus st0{};
    std::string why;
    if (!bb::bridge_router_armed(dev, b.eth_logical, b.status_addr, st0, why)) {
        state.SkipWithError(why.c_str());
        return;
    }
    // Before the producer: every release_cyc must follow the anchor, or the modular difference wraps.
    const auto anchor = eb::measure_clock_anchor(dev, b.eth_logical);
    const auto prod = bb::launch_t6_producer(mesh, dev, b.src_node, b.dst_node, b.producer_link, payload_bytes_, want);
    if (!prod.ok) {
        state.SkipWithError(prod.err.c_str());
        return;
    }

    std::uint64_t drained = 0;
    std::vector<std::uint32_t> took_in(nchan, 0);
    // Reserved up front: regrowing these mid-run stalls the poll for milliseconds on long runs.
    std::vector<std::uint32_t> issue_cyc;
    std::vector<std::int64_t> rel_cyc;  // release cycles since the anchor, unwrapped
    std::vector<double> arr_us;         // host arrival, paired with rel_cyc
    issue_cyc.reserve(want);
    rel_cyc.reserve(want);
    arr_us.reserve(want);
    // release_cyc is 32 bits and wraps every ~3.2 s: unwrap it by summing per-frame deltas from the anchor.
    std::uint32_t prev_cyc = anchor.cyc;
    std::int64_t rel_since_anchor = 0;
    std::uint64_t verify_fail = 0;
    // Payload offset in words: the fabric header size is a build detail, so find it from the first frame.
    std::uint32_t pay_off_w = 0;
    bool pay_off_known = false;
    std::chrono::steady_clock::time_point t_first{}, t_last{};

    const auto kStall = std::chrono::seconds(2);  // resets on progress, so a stuck run reports fast
    std::string stall_why;
    for (auto _ : state) {
        auto give_up = std::chrono::steady_clock::now() + kStall;
        while (drained < want) {
            std::uint32_t took = 0;
            std::fill(took_in.begin(), took_in.end(), 0u);
            const std::uint32_t n = leg.poll([&](const BridgeSendTask& t) -> bool {
                if (took >= batch_) {
                    return false;
                }
                issue_cyc.push_back(static_cast<std::uint32_t>(t.elapsed & 0xFFFFFFFFull));
                if (verify_) {  // read the producer's tag in place in the aliased arena
                    const auto* w = reinterpret_cast<const std::uint32_t*>(b.region.base() + t.page_offset);
                    const std::uint32_t words = t.length / 4;
                    const std::uint32_t k0 = bb::kPayloadFirstTagWord;
                    // Two consecutive tag words mark the payload start; the header cannot imitate them.
                    for (std::uint32_t o = 0; !pay_off_known && o + k0 + 1 < words; ++o) {
                        if (w[o + k0] == bb::payload_tag_word(k0) && w[o + k0 + 1] == bb::payload_tag_word(k0 + 1)) {
                            pay_off_w = o;
                            pay_off_known = true;
                        }
                    }
                    bool ok = pay_off_known;
                    for (std::uint32_t k = k0; ok && pay_off_w + k < words; ++k) {
                        ok = w[pay_off_w + k] == bb::payload_tag_word(k);
                    }
                    verify_fail += ok ? 0 : 1;
                }
                rel_since_anchor += static_cast<std::int32_t>(t.release_cyc - prev_cyc);  // signed: channels interleave
                prev_cyc = t.release_cyc;
                if (drained >= warmup_) {
                    rel_cyc.push_back(rel_since_anchor);
                    arr_us.push_back(
                        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch())
                            .count());
                }
                ++took_in[t.arena < nchan ? t.arena : 0];
                ++took;
                return true;
            });
            if (n == 0) {
                if (std::chrono::steady_clock::now() > give_up) {
                    std::vector<std::uint32_t> ps;  // producer {stage, sent}: 0 never ran, 1 in open(), 2 sending
                    tt::tt_metal::detail::ReadFromDeviceL1(dev, CoreCoord(0, 0), bb::status_l1(dev), 8, ps);
                    stall_why = "stalled at " + std::to_string(drained) + " of " + std::to_string(want) + " frames" +
                                router_counters(dev, b) + " [producer stage=" + std::to_string(ps.at(0)) +
                                " sent=" + std::to_string(ps.at(1)) + "]";
                    // Router's worker-channel credit: all free means the producer's sends never reached it.
                    const auto& hal = mesh->impl().metal_env().get_hal();
                    const auto reg = eb::reg_read(dev, b.eth_logical, tt::CoreType::ETH, eb::available_addr(hal, 0));
                    stall_why += " [router ch0 free_slots_reg=" + std::to_string(reg) +
                                 " producer_link=" + std::to_string(b.producer_link) + "]";
                    break;
                }
                continue;
            }
            const auto now = std::chrono::steady_clock::now();
            if (drained <= warmup_ && drained + n > warmup_) {
                t_first = now;
            }
            if (drained + n > warmup_) {
                t_last = now;
            }
            drained += n;
            give_up = now + kStall;
            // Retire after reading, per arena: crediting early lets the ERISC overwrite unread pages.
            for (std::uint32_t a = 0; a < nchan; ++a) {
                leg.retire(a, took_in[a]);
            }
        }
    }

    state.counters["drained"] = static_cast<double>(drained);
    state.counters["credit_ceiling"] = static_cast<double>(b.ceiling);
    state.counters["packet_size"] = static_cast<double>(payload_bytes_);
    state.counters["warmup_frames"] = static_cast<double>(warmup_);
    if (drained > warmup_ + 1) {
        const double us = std::chrono::duration<double, std::micro>(t_last - t_first).count();
        if (us > 0.0) {
            const double mb = static_cast<double>(drained - warmup_ - 1) * payload_bytes_ / 1e6;
            bb::emit_throughput(state, mb / us * 1e6, drained - warmup_, /*sustained=*/drained > b.ceiling * 2);
        }
    }
    bb::emit_device_cycles(state, issue_cyc, {});
    // A second anchor gives the run's true average rate; one anchor's rate error grows ~linearly with time.
    const auto end = eb::measure_clock_anchor(dev, b.eth_logical);
    if (anchor.ok) {
        double rate = anchor.cyc_per_us;
        if (end.ok && end.host_us > anchor.host_us) {
            const double dt = end.host_us - anchor.host_us;
            const auto d32 = static_cast<std::uint32_t>(end.cyc - anchor.cyc);
            const double laps = std::round((dt * rate - d32) / 4294967296.0);
            rate = (laps * 4294967296.0 + d32) / dt;
        }
        for (size_t i = 0; i < arr_us.size(); ++i) {
            arr_us[i] -= anchor.host_us + static_cast<double>(rel_cyc[i]) / rate;  // now release -> arrival
        }
        state.counters["aiclk_mhz"] = rate;
        bb::emit_latency_us(state, std::move(arr_us), anchor.uncertainty_us + end.uncertainty_us);
    }
    eb::BridgeStatus st1{};
    std::string ignored;
    if (bb::bridge_router_armed(dev, b.eth_logical, b.status_addr, st1, ignored)) {
        state.counters["router_opened"] = static_cast<double>(st1.opened);
        state.counters["router_frames"] = static_cast<double>(st1.frames);
        state.counters["router_declined"] = static_cast<double>(st1.declined);
        state.counters["router_blocked_rx"] = static_cast<double>(st1.blocked_rx);
        state.counters["router_blocked_nodata"] = static_cast<double>(st1.blocked_nodata);
        state.counters["router_free_slots"] = static_cast<double>(st1.free_slots);
    }
    state.counters["payload_off_w"] = pay_off_known ? static_cast<double>(pay_off_w) : -1.0;
    state.counters["want"] = static_cast<double>(want);
    state.counters["unarmed"] = static_cast<double>(leg.unarmed());
    state.counters["verified"] = verify_ ? 1.0 : 0.0;
    bb::emit_health(state, verify_fail, /*credit_stalls=*/leg.unarmed(), leg.out_of_order());
    if (!stall_why.empty()) {  // a partial run must not read as a pass; its producer never finishes
        state.SkipWithError(stall_why.c_str());
        g_stalled = true;
    } else {
        Finish(mesh->mesh_command_queue());
    }
}

BENCHMARK_REGISTER_F(E2HLegFixture, Bridge)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->ArgNames(bb::arg_names_with({"batch"}))
    ->ArgsProduct({
        {1, 4, 16},          // total_amt: MiB of MEASURED payload
        {0, 10},             // warmup_pct: ADDED on top
        {1024, 2048, 4096},  // packet_size: BYTES
        {0, 1},              // verify
        {0},                 // pace: nothing releases the go flag yet
        {0},                 // sweep: one forced channel
        {1, 8, 32},          // batch: frames per poll()
    });

int main(int argc, char** argv) {
    // Before MeshDevice::create: routers compile during fabric init, so the bridge must be on by then.
    const char* chan = std::getenv("TT_BRIDGE_E2H_FORCE_CHAN");
    char* end = nullptr;
    const unsigned long v = chan != nullptr ? std::strtoul(chan, &end, 0) : 0;
    if (chan == nullptr || end == chan || *end != '\0') {
        std::fprintf(stderr, "set TT_BRIDGE_E2H_FORCE_CHAN to the eth channel to bridge, e.g. 10\n");
        return 2;
    }
    g_chan = static_cast<std::uint32_t>(v);
    auto& rt = tt::tt_metal::MetalContext::instance().rtoptions();
    rt.set_e2h_bridge_enable(true);
    rt.set_e2h_bridge_force_chan(g_chan);

    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D);
    bb::the_mesh() = MeshDevice::create(MeshDeviceConfig(std::nullopt));
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    if (g_stalled) {  // skip close(): it waits on the stuck producer. Reset the chips before the next run.
        std::fprintf(stderr, "E2H: producer stalled; exiting without closing the mesh (run tt-smi -r)\n");
        std::fflush(nullptr);
        std::_Exit(3);
    }
    bb::the_mesh()->close();
    bb::the_mesh().reset();
    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    return 0;
}
