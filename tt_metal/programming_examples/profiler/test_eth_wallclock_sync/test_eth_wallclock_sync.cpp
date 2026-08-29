// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Measure the wall-clock offset between two devices over one ethernet link.
//
// A thin driver around eth_sync::measure_link() -- the same call the profiler makes at start() -- so this
// test exercises the production path rather than a parallel copy of it.

#include <thread>
#include <chrono>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/system_mesh.hpp>

#include "impl/context/metal_context.hpp"
#include "tools/profiler/sync/eth_wallclock_sync_host.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::eth_sync;

int main(int argc, char** argv) {
    LinkSyncConfig cfg;
    bool all_links = false;
    // --repeat N --interval-s S: measure the SAME link N times, S seconds apart, and print the offset
    // trajectory. This exists to settle one question: a rate fitted over a 51 ms window has a standard
    // error near 0.01 ppm, yet comparing an init fit against a close measurement 30 s later implies
    // several ppm. Either the clocks really excurse mid-session, or a short-window fit does not
    // extrapolate. A trajectory distinguishes them -- smooth movement means real drift, scatter between
    // consecutive measurements means the measurement (or the fit) is the problem.
    uint32_t repeat = 0;
    uint32_t interval_s = 5;
    // --clockwatch N: read the two chips' wall clocks straight from the HOST, N times, with no eth kernels
    // anywhere. This isolates the question the trajectory raised. The offset there took discrete forward
    // steps of 10-116 us while RTT stayed pinned at 856-858 cycles, which rules out link asymmetry (that
    // would inflate RTT) and points at the clocks themselves -- but the eth sync path is still in the loop.
    // Reading both clocks over MMIO removes it: host bracketing gives ~1-2 us of precision here, an order
    // of magnitude finer than the steps. Steps still present => the device clocks really do step relative
    // to each other. Steps gone => the eth measurement path is manufacturing them.
    uint32_t clockwatch = 0;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
            cfg.n_samples = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--gap-us") == 0 && i + 1 < argc) {
            cfg.gap_us = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--all-links") == 0) {
            all_links = true;
        } else if (std::strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
            repeat = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--interval-s") == 0 && i + 1 < argc) {
            interval_s = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--clockwatch") == 0 && i + 1 < argc) {
            clockwatch = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        }
    }

    const auto shape = distributed::SystemMesh::instance().shape();
    printf("[ethsync] system mesh %ux%u\n", (unsigned)shape[0], (unsigned)shape[1]);
    auto mesh_device = distributed::MeshDevice::create(distributed::MeshDeviceConfig(shape));
    auto devices = mesh_device->get_devices();

    uint64_t first_mid = 0;
    auto& cluster = MetalContext::instance().get_cluster();
    const double ghz = cluster.get_device_aiclk(devices.front()->id()) / 1000.0;
    const double cyc_to_us = 1.0 / (ghz * 1000.0);

    // One representative link per ordered device pair. With --all-links, every pair found; otherwise the
    // first. Measuring each pair once is what a spanning tree needs, and measuring BOTH directions of a
    // pair is the cheapest self-check there is: the two offsets must be equal and opposite.
    struct Edge { IDevice* a; CoreCoord ac; IDevice* b; CoreCoord bc; };
    std::vector<Edge> edges;
    for (IDevice* d : devices) {
        for (const CoreCoord& ec : d->get_active_ethernet_cores(true)) {
            auto [peer_id, peer_core] = d->get_connected_ethernet_core(ec);
            IDevice* peer = nullptr;
            for (IDevice* p : devices) {
                if (p->id() == (int)peer_id && p != d) { peer = p; break; }
            }
            if (peer == nullptr) { continue; }
            bool seen = false;
            for (const auto& e : edges) {
                if ((e.a == d && e.b == peer) || (e.a == peer && e.b == d)) { seen = true; break; }
            }
            if (seen) { continue; }
            edges.push_back(Edge{d, ec, peer, peer_core});
            if (!all_links) { break; }
        }
        if (!all_links && !edges.empty()) { break; }
    }
    printf("[ethsync] %zu link(s) to measure, %u samples %u us apart\n", edges.size(), cfg.n_samples, cfg.gap_us);

    if (clockwatch != 0 && !edges.empty()) {
        const auto& e = edges.front();
        // RISCV_DEBUG_REG_WALL_CLOCK_L latches H, so read L then H.
        constexpr uint64_t kWallL = 0xFFB121F0ULL;
        constexpr uint64_t kWallH = 0xFFB121F8ULL;
        const CoreCoord av = e.a->virtual_core_from_logical_core(e.ac, CoreType::ETH);
        const CoreCoord bv = e.b->virtual_core_from_logical_core(e.bc, CoreType::ETH);
        const tt_cxy_pair ta(e.a->id(), av), tb(e.b->id(), bv);
        auto read_clk = [&](const tt_cxy_pair& t) {
            uint32_t lo = 0, hi = 0;
            cluster.read_reg(&lo, t, kWallL);
            cluster.read_reg(&hi, t, kWallH);
            return ((uint64_t)hi << 32) | lo;
        };
        printf("\n[ethsync] CLOCKWATCH: dev %d eth(%zu,%zu) vs dev %d eth(%zu,%zu), %u reads, host-bracketed\n",
               e.a->id(), e.ac.x, e.ac.y, e.b->id(), e.bc.x, e.bc.y, clockwatch);
        // BRACKET WIDTH is the trust metric. offset = B - midpoint(A0,A1) assumes B landed halfway between
        // the two A reads; a host scheduling hiccup between any two of them breaks that and biases the
        // offset by roughly the hiccup. So take many triples and keep the one with the NARROWEST bracket --
        // the same idea as the eth sync's min-RTT filter, for the same reason. A jump that survives
        // min-bracket selection is a clock step; one that only appears in wide brackets was host jitter.
        constexpr uint32_t kTriples = 32;
        printf("%4s %11s %20s %13s %11s %12s %11s\n", "i", "elapsed_s", "offset_cyc", "delta_cyc",
               "delta_us", "bracket_cyc", "worst_brk");
        uint64_t first_a = 0;
        long long prev_off = 0;
        for (uint32_t i = 0; i < clockwatch; i++) {
            if (i != 0) { std::this_thread::sleep_for(std::chrono::milliseconds(500)); }
            // A, B, A again: the two A reads bracket B, so B is compared against A INTERPOLATED to B's
            // instant. That cancels the MMIO round trip to first order instead of assuming it is symmetric.
            uint64_t best_brk = ~0ULL, worst_brk = 0, best_mid = 0;
            long long off = 0;
            for (uint32_t k = 0; k < kTriples; k++) {
                const uint64_t a0 = read_clk(ta);
                const uint64_t b0 = read_clk(tb);
                const uint64_t a1 = read_clk(ta);
                const uint64_t brk = a1 - a0;
                if (brk > worst_brk) { worst_brk = brk; }
                if (brk < best_brk) {
                    best_brk = brk;
                    best_mid = a0 + brk / 2;
                    off = (long long)b0 - (long long)best_mid;
                }
            }
            if (i == 0) { first_a = best_mid; prev_off = off; }
            const double elapsed = (double)(best_mid - first_a) / (ghz * 1e9);
            printf("%4u %11.2f %20lld %13lld %11.3f %12llu %11llu\n", i, elapsed, off, off - prev_off,
                   (double)(off - prev_off) * cyc_to_us, (unsigned long long)best_brk,
                   (unsigned long long)worst_brk);
            prev_off = off;
        }
        mesh_device->close();
        return 0;
    }

    if (repeat != 0 && !edges.empty()) {
        const auto& e = edges.front();
        printf("\n[ethsync] TRAJECTORY: dev %d -> dev %d, %u measurements %u s apart\n",
               e.a->id(), e.b->id(), repeat, interval_s);
        // aiclk of BOTH chips alongside each measurement. The per-interval deviations are large POSITIVE
        // steps (10-116 us) on top of a clean ppm-level ramp, and a brief clock excursion on one chip is
        // the natural source: a ~1 ms dip at 1.35 -> 1.2 GHz shifts accumulated phase by ~100 us. If aiclk
        // is pinned at 1.35 GHz across every sample, that explanation is dead and the steps are something
        // else -- which is worth knowing either way.
        printf("%4s %11s %18s %10s %10s %8s %10s %10s\n", "i", "elapsed_s", "offset_cyc", "short_ppm",
               "cum_ppm", "resid", "aiclk_snd", "aiclk_rcv");
        struct Pt { double t_s; long long off; double rate; };
        std::vector<Pt> pts;
        for (uint32_t i = 0; i < repeat; i++) {
            if (i != 0) {
                std::this_thread::sleep_for(std::chrono::seconds(interval_s));
            }
            auto r = measure_link(e.a, e.ac, e.b, e.bc, cfg);
            if (!r.solution.valid) {
                printf("%4u  NO SOLUTION (%s / %s)\n", i, status_name(r.sender_status), status_name(r.receiver_status));
                continue;
            }
            const auto& s2 = r.solution;
            const double t_s = pts.empty() ? 0.0
                                           : (double)(long long)(s2.mid_ref - (uint64_t)pts.front().t_s) * 0.0;
            (void)t_s;
            // Elapsed measured on the SENDER's own clock, the same axis the offsets are quoted on.
            const double elapsed =
                pts.empty() ? 0.0 : (double)(long long)(s2.mid_ref - first_mid) / (ghz * 1e9);
            if (pts.empty()) {
                first_mid = s2.mid_ref;
            }
            // Cumulative rate implied by the offset change since the FIRST measurement: this is the number
            // the init-fit extrapolation is really betting on.
            const double cum_ppm =
                (elapsed > 0.0)
                    ? ((double)(s2.offset - pts.front().off) / (elapsed * ghz * 1e9)) * 1e6
                    : 0.0;
            const auto clk_s = cluster.get_device_aiclk(e.a->id());
            const auto clk_r = cluster.get_device_aiclk(e.b->id());
            printf("%4u %11.2f %18lld %10.2f %10.2f %8.1f %10u %10u\n", i, elapsed, (long long)s2.offset,
                   (s2.rate - 1.0) * 1e6, cum_ppm, s2.residual_rms, (unsigned)clk_s, (unsigned)clk_r);
            pts.push_back(Pt{elapsed, s2.offset, s2.rate});
        }
        if (pts.size() >= 2) {
            const double span = pts.back().t_s - pts.front().t_s;
            const double total_ppm =
                ((double)(pts.back().off - pts.front().off) / (span * ghz * 1e9)) * 1e6;
            double mean_short = 0.0;
            for (const auto& p : pts) { mean_short += (p.rate - 1.0) * 1e6; }
            mean_short /= (double)pts.size();
            printf("\n[ethsync] over %.1f s: long-baseline %+.2f ppm vs mean short-window %+.2f ppm "
                   "(difference %+.2f ppm)\n", span, total_ppm, mean_short, total_ppm - mean_short);
            printf("[ethsync] a difference near zero means short-window fits DO extrapolate; a large one "
                   "means they do not, and the session drift is a fit artefact rather than clock motion\n");
        }
        mesh_device->close();
        return 0;
    }

    int rc = 0;
    for (const auto& e : edges) {
        auto r = measure_link(e.a, e.ac, e.b, e.bc, cfg);
        printf("\n[ethsync] === dev %d -> dev %d (eth (%zu,%zu) -> (%zu,%zu)) ===\n",
            e.a->id(), e.b->id(), e.ac.x, e.ac.y, e.bc.x, e.bc.y);
        printf("[ethsync] status: sender %s | receiver %s | samples %zu/%zu\n",
            status_name(r.sender_status), status_name(r.receiver_status), r.sender_samples, r.receiver_samples);
        if (!r.solution.valid) {
            printf("[ethsync] NO SOLUTION\n");
            rc = 1;
            continue;
        }
        const auto& s = r.solution;
        printf("[ethsync] trips %zu usable / %zu kept | rtt min %llu cyc (%.3f us) med %llu cyc\n",
            s.n_total, s.n_kept, (unsigned long long)s.rtt_min, s.rtt_min * cyc_to_us,
            (unsigned long long)s.rtt_med);
        printf("[ethsync] OFFSET %lld cyc (%.3f us) | RATE %.9f (%.2f ppm)\n",
            (long long)s.offset, s.offset * cyc_to_us, s.rate, (s.rate - 1.0) * 1e6);
        printf("[ethsync] spread %lld cyc (%.3f us) | residual RMS %.1f cyc (%.4f us)\n",
            (long long)s.offset_spread, s.offset_spread * cyc_to_us, s.residual_rms, s.residual_rms * cyc_to_us);
    }

    mesh_device->close();
    return rc;
}
