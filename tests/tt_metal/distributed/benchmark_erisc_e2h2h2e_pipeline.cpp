// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// E2H -> H2H -> H2E on one LoudBox: two ranks, one mesh each, rank 0's end of the cable between them
// bridged. Rank 0's T6 sends across it; rank 1 injects what lands into the far router, to its T6.

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
#include <optional>
#include <string>
#include <vector>

#include <mpi.h>

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
#include "tt_metal/distributed/erisc_e2h2h2e_pipeline.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;
using namespace tt::tt_metal::experimental;
namespace eb = tt::tt_fabric::erisc_bridge;
namespace bb = tt::tt_fabric::erisc_bridge::bench;

namespace {

constexpr std::uint32_t kUnset = 0xFFFFFFFFu;
int g_rank = 0;
int g_size = 0;
std::uint32_t g_chan = kUnset;   // rank 0's bridged eth channel, from TT_BRIDGE_E2H_FORCE_CHAN
std::uint64_t g_far_frames = 0;  // frames rank 1 injected in this process: the far router's cursor
bool g_stalled = false;          // a stalled case leaves its producer holding the worker core

std::uint32_t env_u32(const char* n, std::uint32_t d) {
    const char* v = std::getenv(n);
    return v != nullptr ? static_cast<std::uint32_t>(std::strtoul(v, nullptr, 0)) : d;
}

// Rank 1 runs in lockstep but reports nothing; a null reporter would select the console one.
struct SilentReporter : benchmark::BenchmarkReporter {
    bool ReportContext(const Context&) override { return true; }
    void ReportRuns(const std::vector<Run>&) override {}
};

// Two tag words in a row mark the payload start, which the header cannot imitate; the rest must follow.
bool tags_ok(const std::uint32_t* w, std::uint32_t words) {
    const std::uint32_t k0 = bb::kPayloadFirstTagWord;
    for (std::uint32_t o = 0; o + k0 + 1 < words; ++o) {
        if (w[o + k0] == bb::payload_tag_word(k0) && w[o + k0 + 1] == bb::payload_tag_word(k0 + 1)) {
            for (std::uint32_t k = k0 + 2; o + k < words; ++k) {
                if (w[o + k] != bb::payload_tag_word(k)) {
                    return false;
                }
            }
            return true;
        }
    }
    return false;
}

bool arm(IDevice* dev, CoreCoord eth, std::uint32_t status_addr, bool on) {
    std::vector<std::uint32_t> v{on ? eb::kBridgeHostArmed : 0u};
    return tt::tt_metal::detail::WriteToDeviceL1(
        dev, eth, status_addr + offsetof(eb::BridgeStatus, host_armed), v, tt::CoreType::ETH);
}

// Disarm, socket, alias, pin, leg, arm: the router opens its socket only once the host arms it.
std::unique_ptr<E2HLeg> bind_e2h(
    const std::shared_ptr<MeshDevice>& mesh,
    IDevice* dev,
    CoreCoord eth_log,
    CoreCoord eth_noc0,
    const eb::BridgePlacement& p,
    std::uint32_t cap,
    std::uint32_t ring,
    eb::EriscBridgeRegion& region,
    std::unique_ptr<eb::BridgeArenaAlias>& alias,
    std::string& err) {
    if (!arm(dev, eth_log, p.status_addr, false) || !bb::zero_socket_config(dev, eth_log, p.socket_config_addr(0))) {
        err = "could not disarm the bridged router or zero its socket config";
        return nullptr;
    }
    const std::uint32_t page = tt::tt_fabric::bridge_socket_page_bytes(cap);
    std::vector<std::unique_ptr<D2HSocket>> socks;
    try {
        D2HSocket::ExternalConfigBuffer ecb{p.socket_config_addr(0), HalProgrammableCoreType::ACTIVE_ETH};
        socks.push_back(
            std::make_unique<D2HSocket>(mesh, MeshCoreCoord(MeshCoordinate(0, 0), eth_noc0), page * ring, ecb));
        socks.back()->set_page_size(page);
    } catch (const std::exception& e) {
        err = std::string("D2HSocket: ") + e.what();
        return nullptr;
    }
    const auto d = socks.back()->populate_descriptor();
    alias = eb::BridgeArenaAlias::map(
        region,
        {{tt::tt_fabric::bridge_arena_index(0, 0, 1),
          tt::tt_fabric::BridgeArena::Tx,
          d.shm_name,
          d.shm_size,
          d.data_offset,
          d.fifo_size}},
        err);
    if (!alias || !region.provision(mesh, /*chip=*/0, err)) {
        return nullptr;
    }
    E2HLeg::Config c;
    c.packet_capacity = cap;
    c.ring_pages = ring;
    c.arenas = region.arena_count();
    c.alias_region_base = region.base();
    auto leg = E2HLeg::create(std::move(socks), c, err);
    if (leg && !arm(dev, eth_log, p.status_addr, true)) {
        err = "could not arm the bridged router";
        return nullptr;
    }
    return leg;
}

struct PipeFixture : public benchmark::Fixture {
    void SetUp(benchmark::State& state) override {
        payload_ = static_cast<std::uint32_t>(state.range(bb::kPacketSize));
        verify_ = state.range(bb::kVerify) != 0;
        batch_ = std::max<std::uint32_t>(1, static_cast<std::uint32_t>(state.range(bb::kLegSpecific0)));
        // TT_BRIDGE_PIPE_MIB overrides total_amt so a sweep can use sizes the grid does not register.
        const auto mib = env_u32("TT_BRIDGE_PIPE_MIB", static_cast<std::uint32_t>(state.range(bb::kTotalAmt)));
        const std::uint64_t measured = static_cast<std::uint64_t>(mib) * bb::kMiB / payload_;
        warmup_ = measured * static_cast<std::uint64_t>(state.range(bb::kWarmupPct)) / 100;
        frames_ = static_cast<std::uint32_t>(measured + warmup_);
    }
    std::uint32_t payload_ = 0;
    std::uint64_t warmup_ = 0;
    std::uint32_t frames_ = 0;
    std::uint32_t batch_ = 1;
    bool verify_ = false;
};

}  // namespace

BENCHMARK_DEFINE_F(PipeFixture, Bridge)(benchmark::State& state) {
    if (g_stalled) {
        state.SkipWithError("skipped: an earlier case in this process stalled");
        return;
    }
    auto mesh = bb::the_mesh();
    auto& env = mesh->impl().metal_env();
    const auto& cp = env.get_control_plane();
    const std::uint32_t ring = env_u32("TT_BRIDGE_RING_PAGES", tt::tt_fabric::kBridgeDefaultRingPages);
    const eb::EriscBridgePlacement place(eb::router_config(cp));
    const auto p = place.place();
    const std::uint32_t cap = place.channel_slot_bytes();  // the router's slot fixes the stride, not packet_size
    std::string err = !p.ok ? "placement refused: " + p.why : "";
    if (err.empty() && payload_ + 64 > cap) {
        err = "packet_size " + std::to_string(payload_) + " + header exceeds the " + std::to_string(cap) + " B slot";
    }

    // Rank 0 names the cable's far end, which rank 1 needs to find the router H2E injects into.
    std::uint32_t far[4] = {0, 0, 0, 0};  // {ok, mesh, chip, chan}
    IDevice* dev = nullptr;
    tt::tt_fabric::FabricNodeId node{tt::tt_fabric::MeshId{0}, 0};
    if (g_rank == 0 && err.empty()) {
        dev = mesh->get_device(MeshCoordinate(0, 0));
        node = cp.get_fabric_node_id_from_physical_chip_id(static_cast<ChipId>(dev->id()));
        const auto facing = cp.get_intermesh_facing_eth_chans(node);
        const auto peer = cp.try_get_connected_mesh_chip_chan_ids(node, static_cast<tt::tt_fabric::chan_id_t>(g_chan));
        if (std::find(facing.begin(), facing.end(), g_chan) == facing.end() || !peer) {
            err = "set TT_BRIDGE_E2H_FORCE_CHAN to an inter-mesh channel on chip 0; candidates:";
            for (const auto c : facing) {
                err += " " + std::to_string(c);
            }
        } else {
            far[0] = 1;
            far[1] = *peer->first.mesh_id;
            far[2] = peer->first.chip_id;
            far[3] = peer->second;
        }
    }
    MPI_Bcast(far, 4, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    const tt::tt_fabric::FabricNodeId far_node{tt::tt_fabric::MeshId{far[1]}, far[2]};
    if (g_rank == 1 && err.empty() && far[0] == 0) {
        err = "rank 0 could not resolve the bridged link";
    }

    eb::EriscBridgeRegion region;
    std::vector<eb::LinkBinding> links(1);
    links[0].local_chan = static_cast<std::uint8_t>(g_rank == 0 ? g_chan : far[3]);
    eb::EriscBridgeRegion::Geometry geom;
    geom.packet_capacity = cap;
    geom.ring_pages = ring;
    geom.chans_per_link = 1;
    if (err.empty() && region.reserve(links, geom, err) == nullptr && err.empty()) {
        err = "could not reserve the bridge region";
    }

    std::unique_ptr<E2HLeg> e2h;
    std::unique_ptr<eb::BridgeArenaAlias> alias;
    CoreCoord eth_log{0, 0};
    if (g_rank == 0 && err.empty()) {
        const auto& soc = env.get_cluster().get_soc_desc(dev->id());
        const auto noc0 = soc.get_eth_core_for_channel(g_chan, tt::CoordSystem::NOC0);
        const auto lg = soc.get_eth_core_for_channel(g_chan, tt::CoordSystem::LOGICAL);
        eth_log = CoreCoord(lg.x, lg.y);
        e2h = bind_e2h(mesh, dev, eth_log, CoreCoord(noc0.x, noc0.y), p, cap, ring, region, alias, err);
    }
    std::unique_ptr<H2ELeg> h2e;
    if (g_rank == 1 && err.empty()) {
        for (const auto& coord : MeshCoordinateRange(mesh->shape())) {
            auto* d = mesh->get_device(coord);
            if (d != nullptr && cp.get_fabric_node_id_from_physical_chip_id(static_cast<ChipId>(d->id())) == far_node) {
                dev = d;
                H2ELeg::Config c;
                c.page_bytes = tt::tt_fabric::bridge_socket_page_bytes(cap);
                c.eth_chan = far[3];
                c.mesh_row = coord[0];
                c.mesh_col = coord[1];
                c.frames_before = g_far_frames;
                c.alias_region_base = region.base();  // frames off the Rx arena carry their own header
                h2e = H2ELeg::create(mesh, c, err);
            }
        }
        if (dev == nullptr) {
            err = "the bridged link's far chip is not in rank 1's mesh";
        }
    }

    // Collective: a rank that failed locally still joins, with a null region, so neither waits forever.
    EriscH2HSocket::Config hc;
    hc.page_bytes = tt::tt_fabric::bridge_socket_page_bytes(cap);
    hc.ring_pages = ring;
    hc.arenas = 1;
    hc.peer_rank = 1 - g_rank;
    hc.host_rank = static_cast<std::uint32_t>(g_rank);
    hc.host_count = static_cast<std::uint32_t>(g_size);
    hc.region_base = err.empty() ? region.base() : nullptr;
    hc.region_bytes = region.region_bytes();
    hc.max_queued_frames = ring;  // submit() only queues, so a deeper queue reuses a Tx slot still owed a put
    hc.collect_timing = true;
    hc.timing_samples = frames_;
    std::string herr;
    auto h2h = EriscH2HSocket::create(hc, herr);
    if (!h2h) {
        state.SkipWithError((err.empty() ? herr : err).c_str());
        return;
    }
    E2H2H2EPipeline pipe(std::move(h2h), std::move(e2h), std::move(h2e), batch_);
    (void)pipe.h2h().barrier();

    // Rank 0's producer sends to the far chip, so its frames cross the bridged router.
    eb::BridgeStatus st{};
    const auto fwd = g_rank == 0 && err.empty() ? cp.get_forwarding_eth_chans_to_chip(node, far_node)
                                                : std::vector<tt::tt_fabric::chan_id_t>{};
    const auto it = std::find(fwd.begin(), fwd.end(), static_cast<tt::tt_fabric::chan_id_t>(g_chan));
    if (g_rank == 0 && err.empty() && bb::bridge_router_armed(dev, eth_log, p.status_addr, st, err) &&
        it == fwd.end()) {
        err = "eth_chan " + std::to_string(g_chan) + " does not forward to the far chip";
    }
    int bad = err.empty() ? 0 : 1;
    MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (bad != 0) {
        state.SkipWithError(err.empty() ? "the other rank failed setup" : err.c_str());
        return;
    }
    // Before the producer: every release_cyc must follow the anchor, or the modular difference wraps.
    const auto a0 = g_rank == 0 ? eb::measure_clock_anchor(dev, eth_log) : eb::ClockAnchor{};
    const auto prod = g_rank == 0 ? bb::launch_t6_producer(
                                        mesh,
                                        dev,
                                        node,
                                        far_node,
                                        static_cast<std::uint32_t>(std::distance(fwd.begin(), it)),
                                        payload_,
                                        frames_)
                                  : bb::Producer{};
    if (g_rank == 0 && !prod.ok) {
        state.SkipWithError(prod.err.c_str());
        return;
    }

    // Each rank watches its own last hop: rank 0 once every frame is credited (so every E2H page is retired),
    // rank 1 once every frame landed in the far router.
    std::uint64_t verify_fail = 0;
    std::vector<std::uint32_t> rel;  // rank 1: each frame's release stamp from rank 0's ERISC, in order
    std::vector<double> land_us;     // rank 1: when the far router took that frame
    rel.reserve(g_rank == 1 ? frames_ : 0);
    land_us.reserve(g_rank == 1 ? frames_ : 0);
    const auto inspect = [&](const BridgeDeliverTask& d) {
        rel.push_back(d.release_cyc);
        if (verify_ && !tags_ok(reinterpret_cast<const std::uint32_t*>(region.base() + d.page_offset), d.length / 4)) {
            ++verify_fail;
        }
    };
    const auto done = [&] { return g_rank == 0 ? pipe.h2h().credit_total(0) : pipe.counters().landed; };
    auto t_meas = std::chrono::steady_clock::now();
    std::uint64_t m_meas = 0;
    const auto kStall = std::chrono::seconds(3);  // resets on progress, so a stuck hop reports fast
    for (auto _ : state) {
        auto give_up = std::chrono::steady_clock::now() + kStall;
        for (std::uint64_t last = 0; done() < frames_ && std::chrono::steady_clock::now() < give_up;) {
            pipe.poll(inspect);
            const auto now = std::chrono::steady_clock::now();
            const double now_us = std::chrono::duration<double, std::micro>(now.time_since_epoch()).count();
            while (land_us.size() < pipe.counters().landed) {  // the router takes frames in order
                land_us.push_back(now_us);
            }
            if (done() != last) {
                if (last < warmup_ && done() >= warmup_) {
                    t_meas = now;
                    m_meas = done();
                }
                last = done();
                give_up = now + kStall;
            }
        }
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_meas).count();
    g_far_frames += pipe.counters().injected;
    if (g_rank == 1 && verify_) {  // the final hop: the last payload in the far T6's L1
        std::vector<std::uint32_t> got(payload_ / 4);
        const bool ok =
            eb::read_bytes_from_l1(dev, CoreCoord(0, 0), bb::src_l1(dev), got.data(), payload_, tt::CoreType::WORKER);
        verify_fail += ok && tags_ok(got.data(), static_cast<std::uint32_t>(got.size())) ? 0 : 1;
    }

    // Rank 1 times the end-to-end rate; the counts are summed so rank 0's report names every hop.
    double rate = g_rank == 1 && secs > 0.0 ? static_cast<double>(done() - m_meas) * payload_ / secs / 1e6 : 0.0;
    MPI_Allreduce(MPI_IN_PLACE, &rate, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const auto& c = pipe.counters();
    std::uint64_t h[6] = {
        verify_fail,
        pipe.h2h().pass_stats().bad_order + c.out_of_order,
        done() < frames_ ? 1u : 0u,
        c.forwarded,
        c.injected,
        c.landed};
    MPI_Allreduce(MPI_IN_PLACE, h, 6, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    bb::emit_throughput(state, rate, h[5], h[5] > ring);
    state.counters["want"] = frames_;
    state.counters["forwarded"] = static_cast<double>(h[3]);
    state.counters["injected"] = static_cast<double>(h[4]);
    state.counters["landed"] = static_cast<double>(h[5]);
    state.counters["verified"] = verify_ ? 1.0 : 0.0;
    bb::emit_health(state, h[0], 0, h[1]);
    // Release on rank 0's ERISC -> taken by rank 1's router: one host clock, rank 0's anchors convert cycles.
    double anc[4] = {0, 0, 0, 0};  // {cyc, host_us, cyc_per_us, uncertainty_us}
    if (const auto a1 = g_rank == 0 ? eb::measure_clock_anchor(dev, eth_log) : eb::ClockAnchor{}; a0.ok) {
        double r = a0.cyc_per_us;
        if (a1.ok && a1.host_us > a0.host_us) {  // the run's average rate; one anchor's error grows with time
            const double dt = a1.host_us - a0.host_us;
            const auto d32 = static_cast<std::uint32_t>(a1.cyc - a0.cyc);
            r = (std::round((dt * r - d32) / 4294967296.0) * 4294967296.0 + d32) / dt;
        }
        anc[0] = a0.cyc, anc[1] = a0.host_us, anc[2] = r, anc[3] = a0.uncertainty_us + a1.uncertainty_us;
    }
    MPI_Bcast(anc, 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double lat[5] = {0, 0, 0, 0, 0};  // {min, p50, p99, max, samples}, computed on rank 1
    if (g_rank == 1 && anc[2] > 0.0) {
        std::vector<double> us;
        std::int64_t since = 0;  // release cycles since the anchor, unwrapped: the stamp is 32 bits
        auto prev = static_cast<std::uint32_t>(anc[0]);
        for (std::size_t i = 0; i < std::min(rel.size(), land_us.size()); ++i) {
            since += static_cast<std::int32_t>(rel[i] - prev);
            prev = rel[i];
            if (i >= warmup_) {
                us.push_back(land_us[i] - anc[1] - static_cast<double>(since) / anc[2]);
            }
        }
        std::sort(us.begin(), us.end());
        if (!us.empty()) {
            lat[0] = us.front(), lat[1] = bb::pct_of(us, 0.50), lat[2] = bb::pct_of(us, 0.99), lat[3] = us.back();
            lat[4] = static_cast<double>(us.size());
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, lat, 5, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    state.counters["e2e_us_min"] = lat[0];
    state.counters["e2e_us_p50"] = lat[1];
    state.counters["e2e_us_p99"] = lat[2];
    state.counters["e2e_us_max"] = lat[3];
    state.counters["e2e_samples"] = lat[4];
    state.counters["e2e_us_uncertainty"] = anc[3];
    {  // H2H put -> credit, on the sending rank: one hop, not the pipeline
        std::vector<double> us;
        for (const auto v : pipe.h2h().put_to_credit_ns()) {
            us.push_back(static_cast<double>(v) / 1000.0);
        }
        bb::emit_rtt_us(state, std::move(us));
    }
    if (!pipe.first_error().empty()) {
        state.SkipWithError(pipe.first_error().c_str());
    } else if (h[2] != 0) {
        g_stalled = true;
        const std::string why = "stalled: forwarded " + std::to_string(h[3]) + ", injected " + std::to_string(h[4]) +
                                ", landed " + std::to_string(h[5]) + " of " + std::to_string(frames_);
        state.SkipWithError(why.c_str());
    }
}

BENCHMARK_REGISTER_F(PipeFixture, Bridge)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->ArgNames(bb::arg_names_with({"batch"}))
    ->ArgsProduct({
        {1, 4, 16},                 // total_amt: MiB of measured payload
        {0, 10},                    // warmup_pct
        {1024, 2048, 4096, 14336},  // packet_size; 14336 needs TT_BRIDGE_PIPE_MAX_PAYLOAD >= 14400
        {0, 1},                     // verify
        {0},                        // pace: not used here
        {0},                        // sweep: one bridged channel per process
        {8, 32},                    // batch: frames per poll()
    });

int main(int argc, char** argv) {
    // Before MeshDevice::create: routers compile during fabric init. Only mesh 0 chip 0's channel is bridged.
    g_chan = env_u32("TT_BRIDGE_E2H_FORCE_CHAN", kUnset);
    if (g_chan != kUnset) {
        auto& rt = tt::tt_metal::MetalContext::instance().rtoptions();
        rt.set_e2h_bridge_enable(true);
        rt.set_e2h_bridge_force_chan(g_chan);
    }
    // Two meshes need FABRIC_2D. TT_BRIDGE_PIPE_MAX_PAYLOAD widens the router slot past its 4352 B default.
    tt::tt_fabric::FabricRouterConfig rc;
    if (const auto max = env_u32("TT_BRIDGE_PIPE_MAX_PAYLOAD", 0); max != 0) {
        rc.max_packet_payload_size_bytes = max;
    }
    tt::tt_fabric::SetFabricConfig(
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE,
        std::nullopt,
        tt::tt_fabric::FabricTensixConfig::DISABLED,
        tt::tt_fabric::FabricUDMMode::DISABLED,
        tt::tt_fabric::FabricManagerMode::DEFAULT,
        rc);
    // tt-metal owns MPI: MeshDevice::create initialises it, so main must not.
    bb::the_mesh() = MeshDevice::create(MeshDeviceConfig(std::nullopt));
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    int ret = 0;
    if (g_size != 2) {
        std::fprintf(stderr, "the pipeline benchmark needs exactly 2 ranks, got %d\n", g_size);
        ret = 2;
    } else {
        benchmark::Initialize(&argc, argv);
        if (g_rank == 0) {
            benchmark::RunSpecifiedBenchmarks();
        } else {
            SilentReporter silent;
            benchmark::RunSpecifiedBenchmarks(&silent);
        }
        benchmark::Shutdown();
    }
    bb::the_mesh()->close();
    bb::the_mesh().reset();
    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    return ret;  // no MPI_Finalize: tt-metal registered its own
}
