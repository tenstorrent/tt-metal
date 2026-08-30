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
#include <cmath>
#include <map>
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
    // --whichclock N: the pair offset says the two clocks stepped relative to each other but CANNOT say
    // which one moved. Bringing in the host clock as a third reference does: each device is compared
    // against host time separately, so a step that appears in exactly one column is that chip's clock.
    // If a step appears in BOTH columns simultaneously and equally, the host reference moved instead.
    uint32_t whichclock = 0;
    uint32_t period_ms = 200;
    // --same-chip: point --whichclock at an ETH tile and a TENSIX WORKER tile on the SAME chip, instead of
    // one tile on each of two chips. This asks whether a step is chip-wide or tile-local, which separates
    // the two surviving explanations: a PLL relock under DVFS governance stops every tile on the chip at
    // once (steps appear in BOTH columns, same sample, same size), while per-tile clock gating moves only
    // the tile it gates (steps appear in one column). Both would look identical in a two-chip comparison.
    bool same_chip = false;
    // --catchstep N: sample ONE tile's wall clock in a tight loop against host time, no sleeping, and print
    // the samples around any interval where the device advanced far less than the host did. This separates
    // the two readings of "the clock pauses": counting alone cannot tell a hard STOP (counter frozen for
    // ~30 us) from a brief FREQUENCY COLLAPSE (running at a fraction of 1.35 GHz for ~60 us) -- both lose
    // the same number of cycles. An MMIO read costs ~1.5 us here, so the loop samples every ~3 us, enough
    // to land ~10 samples inside a 30 us event and see whether the counter FLATLINES or merely slows.
    uint32_t catchstep = 0;
    // --dvfs N: find FREQUENCY PLATEAUS rather than pauses. If the chip drops 1.35 -> 1.0 GHz, a counter
    // read as if it were still 1.35 advances at ratio 1.0/1.35 = 0.741 of host time -- a plateau, not a
    // stop. --catchstep looked for ratio < 0.5 and therefore could not see this at all; it is the reason
    // that mode found only one event. Here a RUN of consecutive intervals sharing a depressed ratio is the
    // signature, and its mean ratio x 1.35 GHz recovers the frequency the chip was actually running at.
    // Host preemption cannot fake it: that is a single interval, paired with a long one, never a plateau.
    uint32_t dvfs = 0;
    // --aiclkwatch N: poll the ARC's own aiclk telemetry for every device, N times, and report the
    // distribution. The clock-derived tests infer frequency from how fast a counter advances; this asks the
    // firmware directly. If aiclk never leaves its nominal value here, then either DVFS is not happening in
    // this state, or the wall-clock counter does not track aiclk -- and those need separating before any
    // clock-derived frequency claim means anything.
    uint32_t aiclkwatch = 0;
    // --probe-chip N: sample chip N's tile instead of the local one. The point is to reach a chip this
    // process has NOT opened. Opening a device boosts it to 1350, so a watch on an opened chip can never
    // see the 800 MHz state -- the act of observing creates the condition being tested. UMD maps every
    // local chip into the cluster regardless of which one metal opened, so an unopened chip is still
    // addressable over NoC while it sits at 800 MHz. Its counter should then advance at 800/1350 = 0.593
    // of host time, which is unmistakable. Uses the reference chip's eth virtual coords: the chips are
    // identical parts in this mesh, so the tile sits at the same place on each.
    int probe_chip = -1;
    // --unit-mesh: open ONLY device 0. Without this the test opens the whole system mesh, which boosts
    // every chip to 1350 and makes --probe-chip pointless -- it would read another already-boosted chip.
    // The pair (--unit-mesh --probe-chip N) is the actual experiment: hold one chip open so the cluster is
    // mapped, and watch a DIFFERENT chip that nothing has opened.
    bool unit_mesh = false;
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
        } else if (std::strcmp(argv[i], "--whichclock") == 0 && i + 1 < argc) {
            whichclock = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--period-ms") == 0 && i + 1 < argc) {
            period_ms = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--same-chip") == 0) {
            same_chip = true;
        } else if (std::strcmp(argv[i], "--catchstep") == 0 && i + 1 < argc) {
            catchstep = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--dvfs") == 0 && i + 1 < argc) {
            dvfs = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--aiclkwatch") == 0 && i + 1 < argc) {
            aiclkwatch = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--probe-chip") == 0 && i + 1 < argc) {
            probe_chip = (int)std::strtol(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--unit-mesh") == 0) {
            unit_mesh = true;
        }
    }

    const auto shape = distributed::SystemMesh::instance().shape();
    printf("[ethsync] system mesh %ux%u\n", (unsigned)shape[0], (unsigned)shape[1]);
    auto mesh_device = unit_mesh ? distributed::MeshDevice::create_unit_mesh(0)
                                 : distributed::MeshDevice::create(distributed::MeshDeviceConfig(shape));
    if (unit_mesh) {
        printf("[ethsync] --unit-mesh: only device 0 opened; every other chip left in whatever clock state "
               "it was already in\n");
    }
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

    if (aiclkwatch != 0) {
        printf("\n[ethsync] AICLK WATCH: %u polls across %zu device(s)\n", aiclkwatch, devices.size());
        std::map<int, std::map<uint32_t, uint32_t>> hist;  // device -> aiclk MHz -> count
        std::map<int, uint32_t> lo, hi;
        for (uint32_t i = 0; i < aiclkwatch; i++) {
            for (IDevice* d : devices) {
                const uint32_t v = (uint32_t)cluster.get_device_aiclk(d->id());
                hist[d->id()][v]++;
                if (lo.find(d->id()) == lo.end() || v < lo[d->id()]) { lo[d->id()] = v; }
                if (hi.find(d->id()) == hi.end() || v > hi[d->id()]) { hi[d->id()] = v; }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        for (const auto& [dev, h] : hist) {
            printf("  device %d: min %u MHz, max %u MHz, %zu distinct value(s)\n", dev, lo[dev], hi[dev],
                   h.size());
            for (const auto& [mhz, n] : h) {
                printf("      %5u MHz  x%-7u (%.1f%%)\n", mhz, n, 100.0 * n / (double)aiclkwatch);
            }
        }
        mesh_device->close();
        return 0;
    }

    if (dvfs != 0 && edges.empty() && probe_chip >= 0 && !devices.empty()) {
        // Unit-mesh probe: no peer, so take any active eth core on the opened device and address the SAME
        // tile position on the target chip (identical parts in this mesh).
        IDevice* d0 = devices.front();
        const auto eth_cores = d0->get_active_ethernet_cores(true);
        if (eth_cores.empty()) {
            printf("[ethsync] no active eth core to take coordinates from\n");
            mesh_device->close();
            return 1;
        }
        edges.push_back(Edge{d0, *eth_cores.begin(), d0, *eth_cores.begin()});
    }
    if (dvfs != 0 && !edges.empty()) {
        const auto& e = edges.front();
        constexpr uint64_t kWallL = 0xFFB121F0ULL;
        constexpr uint64_t kWallH = 0xFFB121F8ULL;
        const CoreCoord av = e.a->virtual_core_from_logical_core(e.ac, CoreType::ETH);
        const int target_chip = (probe_chip >= 0) ? probe_chip : e.a->id();
        const tt_cxy_pair ta(target_chip, av);
        struct Smp { double h; uint64_t d; };
        std::vector<Smp> smp;
        smp.reserve(dvfs);
        printf("\n[ethsync] DVFS WATCH: chip %d eth(%zu,%zu)%s, %u samples, nominal %.4f GHz\n",
               target_chip, e.ac.x, e.ac.y,
               (probe_chip >= 0 && probe_chip != e.a->id()) ? " [NOT opened by this process]" : "", dvfs,
               ghz);
        // Whole-run effective frequency, so a chip parked at a low state shows up even with no transition
        // to find: a steady 0.593 is the 800 MHz state, a steady 1.000 is 1350.
        printf("[ethsync] (a steady ratio of 0.593 across the whole run = parked at 800 MHz)\n");
        for (uint32_t i = 0; i < dvfs; i++) {
            uint32_t lo = 0, hi = 0;
            const double h0 = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::steady_clock::now().time_since_epoch()).count();
            cluster.read_reg(&lo, ta, kWallL);
            cluster.read_reg(&hi, ta, kWallH);
            smp.push_back(Smp{h0, ((uint64_t)hi << 32) | lo});
        }
        // Per-interval ratio, then group consecutive depressed intervals into plateaus.
        std::vector<double> ratio(smp.size(), 1.0);
        for (size_t i = 1; i < smp.size(); i++) {
            const double dh = smp[i].h - smp[i - 1].h;
            const double dd = (double)(smp[i].d - smp[i - 1].d) / ghz;
            ratio[i] = (dh > 0.0 && dd > -1000.0 && dd < 1e6) ? dd / dh : 1.0;
        }
        printf("%9s %11s %10s %12s %12s %10s\n", "start_s", "duration_us", "samples", "mean_ratio",
               "implied_GHz", "lost_us");
        uint32_t plateaus = 0;
        double total_lost = 0.0;
        for (size_t i = 1; i < smp.size(); i++) {
            if (ratio[i] >= 0.92 || ratio[i] <= 0.05) {
                continue;  // healthy, or the single-interval spike of a preemption pair
            }
            size_t j = i;
            double sum = 0.0;
            while (j < smp.size() && ratio[j] < 0.92 && ratio[j] > 0.05) {
                sum += ratio[j];
                j++;
            }
            const size_t n = j - i;
            // A plateau is >= 2 consecutive depressed intervals. One alone is a preemption artefact.
            if (n >= 2) {
                const double mean_ratio = sum / (double)n;
                const double dur_us = (smp[j - 1].h - smp[i - 1].h) / 1000.0;
                const double lost_us = dur_us * (1.0 - mean_ratio);
                plateaus++;
                total_lost += lost_us;
                if (plateaus <= 25) {
                    printf("%9.3f %11.2f %10zu %12.3f %12.4f %10.2f\n",
                           (smp[i - 1].h - smp[0].h) / 1e9, dur_us, n, mean_ratio, mean_ratio * ghz,
                           lost_us);
                }
            }
            i = j;
        }
        {
            // Overall ratio, robust to the odd corrupted read.
            std::vector<double> rr;
            rr.reserve(ratio.size());
            for (size_t i = 1; i < ratio.size(); i++) {
                if (ratio[i] > 0.05 && ratio[i] < 3.0) { rr.push_back(ratio[i]); }
            }
            std::sort(rr.begin(), rr.end());
            const double med = rr.empty() ? 0.0 : rr[rr.size() / 2];
            printf("[ethsync] MEDIAN ratio %.4f => %.4f GHz effective\n", med, med * ghz);
        }
        const double span_s = (smp.back().h - smp.front().h) / 1e9;
        printf("\n[ethsync] %u plateau(s) over %.2f s (%.2f/s), total lost %.1f us\n", plateaus, span_s,
               plateaus / span_s, total_lost);
        printf("[ethsync] mean_ratio x nominal = the frequency the chip actually ran at; 0.741 would be "
               "1.0 GHz against a 1.35 GHz nominal\n");
        mesh_device->close();
        return 0;
    }

    if (catchstep != 0 && !edges.empty()) {
        const auto& e = edges.front();
        constexpr uint64_t kWallL = 0xFFB121F0ULL;
        constexpr uint64_t kWallH = 0xFFB121F8ULL;
        const CoreCoord av = e.a->virtual_core_from_logical_core(e.ac, CoreType::ETH);
        const tt_cxy_pair ta(e.a->id(), av);
        struct Smp { double h; uint64_t d; };
        std::vector<Smp> smp;
        smp.reserve(catchstep);
        printf("\n[ethsync] CATCHSTEP: tight-loop sampling dev %d eth(%zu,%zu), %u samples\n",
               e.a->id(), e.ac.x, e.ac.y, catchstep);
        for (uint32_t i = 0; i < catchstep; i++) {
            uint32_t lo = 0, hi = 0;
            const double h0 = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::steady_clock::now().time_since_epoch()).count();
            cluster.read_reg(&lo, ta, kWallL);
            cluster.read_reg(&hi, ta, kWallH);
            smp.push_back(Smp{h0, ((uint64_t)hi << 32) | lo});
        }
        // Per-interval: how many device ns elapsed vs how many host ns. Ratio ~1 is healthy.
        //
        // Two DIFFERENT populations land in this filter and must not be conflated:
        //
        //  ARTEFACT  the host thread is preempted between taking h0 and doing the MMIO read, so THIS
        //            interval reads long on the device side and the NEXT reads short. They pair and cancel,
        //            and the running deficit returns to where it was. Nothing happened to the clock.
        //
        //  REAL      the chip's clock stops. The device simply stops advancing for the duration, so the
        //            deficit is PERMANENT -- there is no compensating interval afterwards, and every later
        //            sample carries the loss.
        //
        // The discriminator is the preceding interval (ratio > 1.5 means the pair) plus whether the deficit
        // is repaid within the next few samples. Counting "intervals where the device fell behind" without
        // separating these is what makes a host hiccup look like a hardware pause.
        double sum_h = 0, sum_d = 0;
        uint32_t events = 0, paired = 0, persistent = 0, bad_samples = 0;
        double persistent_us = 0.0;
        for (size_t i = 1; i < smp.size(); i++) {
            const double dh = smp[i].h - smp[i - 1].h;
            const double dd = (double)(smp[i].d - smp[i - 1].d) / ghz;
            // One corrupted MMIO read poisons a running total for the rest of the run (observed: a device
            // total of 1.4e13 ms against a host total of 1.5 s). The classification below is windowed and
            // survives it; these totals are not, so implausible intervals are excluded from them and
            // counted instead.
            if (dd < -1000.0 || dd > 1e6) {
                bad_samples++;
            } else {
                sum_h += dh;
                sum_d += dd;
            }
            if (dh > 200.0 && dd < 0.5 * dh) {
                events++;
                // Preceding interval: did the device run LONG just before (the pair), or normally (real)?
                double prev_ratio = 1.0;
                if (i >= 2) {
                    const double ph = smp[i - 1].h - smp[i - 2].h;
                    const double pd = (double)(smp[i - 1].d - smp[i - 2].d) / ghz;
                    prev_ratio = ph > 0 ? pd / ph : 1.0;
                }
                // Repaid within the next 8 samples? Sum host vs device across the window and see whether
                // the deficit incurred here comes back.
                double wh = 0, wd = 0;
                for (size_t k = i; k < i + 8 && k < smp.size(); k++) {
                    wh += smp[k].h - smp[k - 1].h;
                    wd += (double)(smp[k].d - smp[k - 1].d) / ghz;
                }
                const double deficit_us = (wh - wd) / 1000.0;
                if (prev_ratio > 1.5) {
                    paired++;
                } else if (deficit_us > 1.0) {
                    persistent++;
                    persistent_us += deficit_us;
                }
                if (events <= 6) {
                    printf("\n  EVENT at sample %zu: host %.3f us elapsed, device only %.3f us (ratio %.3f)\n",
                           i, dh / 1000.0, dd / 1000.0, dd / dh);
                    // Neighbours, so a flatline (ratio ~0 for several) is distinguishable from a slowdown
                    // (ratio well between 0 and 1 across the event).
                    for (size_t j = (i >= 4 ? i - 4 : 0); j < i + 5 && j < smp.size(); j++) {
                        if (j == 0) { continue; }
                        const double h2 = smp[j].h - smp[j - 1].h;
                        const double d2 = (double)(smp[j].d - smp[j - 1].d) / ghz;
                        printf("      %s j=%zu host %8.3f us  dev %8.3f us  ratio %6.3f\n",
                               j == i ? "->" : "  ", j, h2 / 1000.0, d2 / 1000.0, h2 > 0 ? d2 / h2 : 0.0);
                    }
                }
            }
        }
        printf("\n[ethsync] %u interval(s) where the device advanced < 50%% of host, across %zu samples\n",
               events, smp.size());
        printf("[ethsync]   PAIRED (host preemption, self-cancelling): %u\n", paired);
        printf("[ethsync]   PERSISTENT (deficit not repaid in 8 samples): %u, totalling %.1f us\n",
               persistent, persistent_us);
        printf("[ethsync] %u sample(s) excluded from the totals as corrupted reads\n", bad_samples);
        printf("[ethsync] mean cadence %.3f us | total host %.1f ms vs device %.1f ms (deficit %.1f us)\n",
               sum_h / (double)(smp.size() - 1) / 1000.0, sum_h / 1e6, sum_d / 1e6, (sum_h - sum_d) / 1000.0);
        printf("[ethsync] ratio ~0 across consecutive samples = counter STOPPED; ratio steady between 0 and "
               "1 = frequency COLLAPSED but still counting\n");
        mesh_device->close();
        return 0;
    }

    if (whichclock != 0 && !edges.empty()) {
        const auto& e = edges.front();
        constexpr uint64_t kWallL = 0xFFB121F0ULL;
        constexpr uint64_t kWallH = 0xFFB121F8ULL;
        const CoreCoord av = e.a->virtual_core_from_logical_core(e.ac, CoreType::ETH);
        const CoreCoord bv = same_chip
                                 ? e.a->virtual_core_from_logical_core(CoreCoord{0, 0}, CoreType::WORKER)
                                 : e.b->virtual_core_from_logical_core(e.bc, CoreType::ETH);
        const tt_cxy_pair ta(e.a->id(), av);
        const tt_cxy_pair tb(same_chip ? e.a->id() : e.b->id(), bv);
        auto read_clk = [&](const tt_cxy_pair& t) {
            uint32_t lo = 0, hi = 0;
            cluster.read_reg(&lo, t, kWallL);
            cluster.read_reg(&hi, t, kWallH);
            return ((uint64_t)hi << 32) | lo;
        };
        auto host_ns = [] {
            return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now().time_since_epoch())
                .count();
        };
        if (same_chip) {
            printf("\n[ethsync] WHICHCLOCK same-chip: dev %d ETH(%zu,%zu) vs dev %d WORKER(0,0) vs HOST, "
                   "%u samples %u ms apart\n",
                   e.a->id(), e.ac.x, e.ac.y, e.a->id(), whichclock, period_ms);
            printf("[ethsync] steps in BOTH columns together => chip-wide (PLL/DVFS); one column only => "
                   "tile-local gating\n");
        } else {
            printf("\n[ethsync] WHICHCLOCK: dev %d vs dev %d vs HOST, %u samples %u ms apart\n",
                   e.a->id(), e.b->id(), whichclock, period_ms);
        }
        printf("%5s %10s %14s %14s %14s %10s\n", "i", "elapsed_s", "dA_vs_host", "dB_vs_host", "d(B-A)",
               "brk_ns");
        printf("%5s %10s %14s %14s %14s %10s\n", "", "", "us", "us", "us", "");
        constexpr uint32_t kTriples = 8;
        double ra_prev = 0, rb_prev = 0, first_host = 0;
        bool have_prev = false;
        uint32_t steps_a = 0, steps_b = 0, steps_both = 0;
        for (uint32_t i = 0; i < whichclock; i++) {
            if (i != 0) { std::this_thread::sleep_for(std::chrono::milliseconds(period_ms)); }
            double best_span = 1e18, ra = 0, rb = 0, hmid = 0;
            for (uint32_t k = 0; k < kTriples; k++) {
                const double h0 = host_ns();
                const uint64_t a = read_clk(ta);
                const double h1 = host_ns();
                const uint64_t b = read_clk(tb);
                const double h2 = host_ns();
                const double span = h2 - h0;
                if (span < best_span) {
                    best_span = span;
                    // Each device compared against the host instant its own read was centred on.
                    ra = (double)a / ghz - (h0 + h1) * 0.5;
                    rb = (double)b / ghz - (h1 + h2) * 0.5;
                    hmid = (h0 + h2) * 0.5;
                }
            }
            if (!have_prev) {
                ra_prev = ra; rb_prev = rb; first_host = hmid; have_prev = true;
            }
            const double dA = (ra - ra_prev) / 1000.0;   // us
            const double dB = (rb - rb_prev) / 1000.0;
            printf("%5u %10.2f %14.3f %14.3f %14.3f %10.0f\n", i, (hmid - first_host) / 1e9, dA, dB,
                   dB - dA, best_span);
            // A step is anything well beyond the ~1-2 us MMIO noise floor on this box.
            const bool sa = std::fabs(dA) > 4.0, sb = std::fabs(dB) > 4.0;
            if (sa && sb) { steps_both++; } else if (sa) { steps_a++; } else if (sb) { steps_b++; }
            ra_prev = ra; rb_prev = rb;
        }
        printf("\n[ethsync] steps >4 us: dev %d only = %u | dev %d only = %u | BOTH together = %u\n",
               e.a->id(), steps_a, e.b->id(), steps_b, steps_both);
        printf("[ethsync] one-sided steps mean that chip's own clock moved; simultaneous ones would mean "
               "the HOST reference moved instead\n");
        mesh_device->close();
        return 0;
    }

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
