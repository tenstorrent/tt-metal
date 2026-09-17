// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Acceptance test for the streaming profiler's 1588 link layer (tools/profiler/sync/eth_ptp_link.hpp on
// hw/inc/internal/ethernet/eth_ptp.hpp) on real links, with no profiler pipeline: the two product link ends run as
// resident kernels on every ethernet link between local chips and record their stamp averages into an L1 table the
// host reads back; the kernels are the product's own resident ends. Per link: rounds issued, complete at both ends
// and inside the path band; each end's stamp drops;
// one way, turnaround and round trip inside the stamps; the offset-and-rate fit with its residual; the stamps' own
// noise. Per chip pair with several links: the fits' disagreement, which is the difference of the links' path
// asymmetries, the one term the sync cannot see from a single link.
//
//   test_eth_ptp_link [--seconds S] [--pace-ms M] [--links-per-pair N]

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "hostdev/streaming_profiler_common.h"
#include "llrt/tt_cluster.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr double kPathBandNs = 2.0;  // as the sync's solver: rounds whose one-way strays further are not fitted
constexpr size_t kNoiseHalfWindow = 12;
constexpr double kWindowNs = 250e6;  // the sync's solve window: a line holds across it, the clocks wander beyond

struct Link {
    uint32_t chip_a = 0, chip_b = 0;
    CoreCoord eth_a, eth_b, virt_a, virt_b;
    std::unique_ptr<Program> ps, pr;
};
struct Diag {
    uint32_t rounds, timer, hold_lo, hold_hi, wall_lo, wall_hi, ref_lo, ref_hi, hold_max, drop[5];
};
static_assert(sizeof(Diag) == 14 * sizeof(uint32_t));
struct Row {
    uint32_t round, role, lo, hi;
};
struct RoundNs {
    double t0, t1, t1b, t2;
    double mid_a() const { return 0.5 * (t0 + t2); }
    double mid_b() const { return 0.5 * (t1 + t1b); }
    double rtt() const { return t2 - t0; }
    double turn() const { return t1b - t1; }
    double path() const { return 0.5 * (rtt() - turn()); }
};
struct Fit {
    bool ok = false;
    double x0 = 0, offset = 0, rate = 0, resid_rms = 0;
    size_t kept = 0;
    double at(double x) const { return offset + rate * (x - x0); }
};

double pct(std::vector<double> v, double q) {
    if (v.empty()) {
        return NAN;
    }
    const size_t k = std::min(v.size() - 1, static_cast<size_t>(q * static_cast<double>(v.size())));
    std::nth_element(v.begin(), v.begin() + static_cast<long>(k), v.end());
    return v[k];
}

// Three passes of least squares, each dropping points beyond three sigma of the previous line.
Fit fit_line(const std::vector<double>& x, const std::vector<double>& y) {
    Fit f;
    if (x.size() < 4) {
        return f;
    }
    f.x0 = *std::min_element(x.begin(), x.end());
    std::vector<char> keep(x.size(), 1);
    for (int pass = 0; pass < 3; pass++) {
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        size_t n = 0;
        for (size_t i = 0; i < x.size(); i++) {
            if (!keep[i]) {
                continue;
            }
            const double xi = x[i] - f.x0;
            sx += xi, sy += y[i], sxx += xi * xi, sxy += xi * y[i], n++;
        }
        if (n < 4) {
            return f;
        }
        const double nn = static_cast<double>(n), den = nn * sxx - sx * sx;
        f.rate = den > 0 ? (nn * sxy - sx * sy) / den : 0.0;
        f.offset = (sy - f.rate * sx) / nn;
        double ss = 0;
        for (size_t i = 0; i < x.size(); i++) {
            if (keep[i]) {
                const double r = y[i] - f.at(x[i]);
                ss += r * r;
            }
        }
        f.resid_rms = std::sqrt(ss / nn);
        f.kept = n;
        for (size_t i = 0; i < x.size(); i++) {
            keep[i] = keep[i] && std::abs(y[i] - f.at(x[i])) <= 3.0 * f.resid_rms + 1e-9;
        }
    }
    f.ok = true;
    return f;
}

// Each round against a line through its neighbours: a stamp glitch shows here, the link's slow rate wander does not.
double neighbour_noise_rms(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() < 8) {
        return NAN;
    }
    double ss = 0;
    for (size_t i = 0; i < x.size(); i++) {
        const size_t lo = i > kNoiseHalfWindow ? i - kNoiseHalfWindow : 0;
        const size_t hi = std::min(x.size(), i + kNoiseHalfWindow + 1);
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        size_t m = 0;
        for (size_t j = lo; j < hi; j++) {
            if (j == i) {
                continue;
            }
            const double xj = x[j] - x[i];
            sx += xj, sy += y[j], sxx += xj * xj, sxy += xj * y[j], m++;
        }
        const double mm = static_cast<double>(m), den = mm * sxx - sx * sx;
        const double b = den > 0 ? (mm * sxy - sx * sy) / den : 0.0;
        const double a = (sy - b * sx) / mm;
        ss += (y[i] - a) * (y[i] - a);
    }
    return std::sqrt(ss / static_cast<double>(x.size()));
}

}  // namespace

int main(int argc, char** argv) {
    double seconds = 3.0;
    uint32_t pace_ms = 10, links_per_pair = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--seconds")) {
            seconds = std::strtod(argv[i + 1], nullptr);
        } else if (!std::strcmp(argv[i], "--pace-ms")) {
            pace_ms = static_cast<uint32_t>(std::strtoul(argv[i + 1], nullptr, 10));
        } else if (!std::strcmp(argv[i], "--links-per-pair")) {
            links_per_pair = static_cast<uint32_t>(std::strtoul(argv[i + 1], nullptr, 10));
        }
    }
    auto mesh_device = distributed::MeshDevice::create(
        distributed::MeshDeviceConfig(distributed::SystemMesh::instance().shape()),
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        /*num_command_queues=*/1);
    auto& cluster = MetalContext::instance().get_cluster();
    const auto& hal = MetalContext::instance().hal();
    std::map<uint32_t, IDevice*> devices;
    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        if (mesh_device->is_local(coord)) {
            IDevice* d = mesh_device->get_device(coord);
            devices[static_cast<uint32_t>(d->id())] = d;
        }
    }
    const uint32_t unres = hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint32_t unres_size = hal.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    // The link's L1 at the top of the region as in the product; the handshake at its base as in the product's
    // resident kernels; the table between them.
    const uint32_t link_l1 = unres + unres_size - kernel_profiler::kLinkSyncL1Bytes;
    const uint32_t ctl = link_l1 + kernel_profiler::kLinkSyncCtlOffset;
    const uint32_t table = unres + 4096;
    const uint32_t rows = std::min<uint32_t>(4096, (link_l1 - table - 4) / 16);

    std::vector<Link> links;
    for (auto ia = devices.begin(); ia != devices.end(); ++ia) {
        for (auto ib = std::next(ia); ib != devices.end(); ++ib) {
            const auto by_peer = cluster.get_ethernet_cores_grouped_by_connected_chips(static_cast<ChipId>(ia->first));
            const auto it = by_peer.find(static_cast<ChipId>(ib->first));
            if (it == by_peer.end()) {
                continue;
            }
            uint32_t taken = 0;
            for (const CoreCoord& eth_a : it->second) {
                if (links_per_pair != 0 && taken >= links_per_pair) {
                    break;
                }
                const CoreCoord eth_b = std::get<1>(
                    cluster.get_connected_ethernet_core(std::make_tuple(static_cast<ChipId>(ia->first), eth_a)));
                Link L;
                L.chip_a = ia->first;
                L.chip_b = ib->first;
                L.eth_a = eth_a;
                L.eth_b = eth_b;
                L.virt_a = cluster.get_virtual_coordinate_from_logical_coordinates(L.chip_a, eth_a, CoreType::ETH);
                L.virt_b = cluster.get_virtual_coordinate_from_logical_coordinates(L.chip_b, eth_b, CoreType::ETH);
                links.push_back(std::move(L));
                taken++;
            }
        }
    }
    if (links.empty()) {
        std::printf("[eth_ptp_link] no ethernet links between the local chips\n");
        mesh_device->close();
        return 2;
    }
    std::printf(
        "[eth_ptp_link] %zu links on %zu chips, %u ms rounds for %.1f s, table of %u rows at %#x, link L1 %#x\n",
        links.size(),
        devices.size(),
        pace_ms,
        seconds,
        rows,
        table,
        link_l1);

    const std::map<std::string, std::string> defines = {
        {"ETH_PTP_LINK_TABLE", std::to_string(table)}, {"ETH_PTP_LINK_TABLE_ROWS", std::to_string(rows)}};
    const uint32_t pace_ticks = pace_ms * 50'000u;
    const uint32_t zero[2] = {0, 0};
    for (Link& L : links) {
        for (const auto& [chip, virt] : {std::pair{L.chip_a, L.virt_a}, std::pair{L.chip_b, L.virt_b}}) {
            cluster.write_core(zero, sizeof(zero), tt_cxy_pair(chip, virt), ctl);
            cluster.write_core(zero, sizeof(uint32_t), tt_cxy_pair(chip, virt), table);
        }
        L.pr = std::make_unique<Program>(CreateProgram());
        L.ps = std::make_unique<Program>(CreateProgram());
        const auto kid_r = CreateKernel(
            *L.pr,
            "tt_metal/tools/profiler/sync/eth_ptp_link_receiver.cpp",
            L.eth_b,
            EthernetConfig{.noc = NOC::RISCV_0_default, .defines = defines});
        const auto kid_s = CreateKernel(
            *L.ps,
            "tt_metal/tools/profiler/sync/eth_ptp_link_sender.cpp",
            L.eth_a,
            EthernetConfig{.noc = NOC::RISCV_0_default, .defines = defines});
        SetRuntimeArgs(*L.pr, kid_r, L.eth_b, {link_l1});
        SetRuntimeArgs(*L.ps, kid_s, L.eth_a, {link_l1, pace_ticks});
        IDevice* dev_a = devices.at(L.chip_a);
        IDevice* dev_b = devices.at(L.chip_b);
        detail::CompileProgram(dev_b, *L.pr, /*force_slow_dispatch=*/true);
        detail::CompileProgram(dev_a, *L.ps, /*force_slow_dispatch=*/true);
        detail::WriteRuntimeArgsToDevice(dev_b, *L.pr, /*force_slow_dispatch=*/true);
        detail::WriteRuntimeArgsToDevice(dev_a, *L.ps, /*force_slow_dispatch=*/true);
        detail::LaunchProgram(dev_b, *L.pr, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);
        detail::LaunchProgram(dev_a, *L.ps, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);
    }
    for (const Link& L : links) {
        cluster.write_core(&kernel_profiler::kLinkSyncCtlRun, sizeof(uint32_t), tt_cxy_pair(L.chip_a, L.virt_a), ctl);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int64_t>(seconds * 1000.0)));

    const auto stop_end = [&](uint32_t chip, const CoreCoord& virt) {
        cluster.write_core(&kernel_profiler::kLinkSyncCtlStop, sizeof(uint32_t), tt_cxy_pair(chip, virt), ctl);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        for (;;) {
            uint32_t done = 0;
            cluster.read_core(&done, sizeof(done), tt_cxy_pair(chip, virt), ctl + 4);
            if (done != 0) {
                return true;
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    };
    const auto read_table = [&](uint32_t chip, const CoreCoord& virt) {
        uint32_t n = 0;
        cluster.read_core(&n, sizeof(n), tt_cxy_pair(chip, virt), table);
        std::vector<Row> out(std::min(n, rows));
        if (!out.empty()) {
            cluster.read_core(
                out.data(), static_cast<uint32_t>(out.size() * sizeof(Row)), tt_cxy_pair(chip, virt), table + 4);
        }
        return out;
    };
    const auto read_diag = [&](uint32_t chip, const CoreCoord& virt) {
        Diag d{};
        cluster.read_core(&d, sizeof(d), tt_cxy_pair(chip, virt), ctl + 8);
        return d;
    };

    bool all_ok = true;
    struct LinkFit {
        uint32_t a, b;
        CoreCoord eth_a, eth_b;
        Fit f;  // clock_b - clock_a at the sender's time, ns
    };
    std::vector<LinkFit> fits;
    for (Link& L : links) {
        // Sender first: its round in flight completes off the live receiver; then the receiver sees no more frames.
        const bool stopped_a = stop_end(L.chip_a, L.virt_a);
        const bool stopped_b = stop_end(L.chip_b, L.virt_b);
        const Diag da = read_diag(L.chip_a, L.virt_a), db = read_diag(L.chip_b, L.virt_b);
        std::map<uint32_t, std::array<std::optional<double>, 4>> rounds;
        for (const auto& [chip, virt] : {std::pair{L.chip_a, L.virt_a}, std::pair{L.chip_b, L.virt_b}}) {
            for (const Row& r : read_table(chip, virt)) {
                if (r.role < 4) {
                    rounds[r.round][r.role] = static_cast<double>((uint64_t{r.hi} << 32) | r.lo) / 4.0;  // ns
                }
            }
        }
        std::vector<RoundNs> complete;
        for (const auto& [id, s] : rounds) {
            if (s[0] && s[1] && s[2] && s[3]) {
                complete.push_back(RoundNs{*s[0], *s[1], *s[2], *s[3]});
            }
        }
        std::vector<double> path, turn, rtt;
        for (const RoundNs& r : complete) {
            path.push_back(r.path()), turn.push_back(r.turn()), rtt.push_back(r.rtt());
        }
        const double path_med = pct(path, 0.5);
        std::vector<double> x, y;
        for (const RoundNs& r : complete) {
            if (std::abs(r.path() - path_med) <= kPathBandNs) {
                x.push_back(r.mid_a());
                y.push_back(r.mid_b() - r.mid_a());
            }
        }
        const Fit fit = fit_line(x, y);
        const double noise = neighbour_noise_rms(x, y);
        std::vector<double> window_resid;
        for (size_t b = 0; b < x.size();) {
            size_t e = b;
            while (e < x.size() && x[e] - x[b] < kWindowNs) {
                e++;
            }
            if (e - b >= 8) {
                const Fit w = fit_line({x.begin() + b, x.begin() + e}, {y.begin() + b, y.begin() + e});
                if (w.ok) {
                    window_resid.push_back(w.resid_rms);
                }
            }
            b = e;
        }
        const double window_med = pct(window_resid, 0.5), window_max = pct(window_resid, 1.0);
        const uint32_t issued = da.rounds;
        const double complete_frac = issued ? static_cast<double>(complete.size()) / issued : 0.0;
        uint32_t drops = 0;
        for (const Diag* d : {&da, &db}) {
            drops += d->drop[0];
        }
        const bool ok = stopped_a && stopped_b && da.timer == 1 && db.timer == 1 && complete_frac >= 0.95 && fit.ok &&
                        noise <= 1.0 && window_med <= 1.5 && drops <= std::max<uint32_t>(2, issued / 50);
        all_ok = all_ok && ok;
        std::printf(
            "[eth_ptp_link] chip %u eth(%zu,%zu) -> chip %u eth(%zu,%zu): %s\n"
            "  rounds: %u issued, %zu complete at both ends, %zu inside the path band, %zu fitted; timer words "
            "%u/%u%s\n"
            "  drops: sender rounds %u, hand-off past the frames %u, unstamped bursts %u, ingress count off %u, waits "
            "given up %u; receiver %u/%u/%u/%u/%u\n"
            "  inside the stamps: one way %.1f ns (p10 %.1f, p90 %.1f), turnaround %.1f ns (p10 %.1f, p90 %.1f), "
            "round trip %.1f ns (p10 %.1f, p90 %.1f)\n"
            "  fit over the run: offset %.1f ns, rate %+.3f ppm, residual rms %.2f ns (the two clocks' wander); in 250 "
            "ms "
            "windows: residual rms median %.2f ns, worst %.2f ns; stamp noise against the neighbouring rounds %.2f ns "
            "rms\n",
            L.chip_a,
            L.eth_a.x,
            L.eth_a.y,
            L.chip_b,
            L.eth_b.x,
            L.eth_b.y,
            ok ? "PASS" : "FAIL",
            issued,
            complete.size(),
            x.size(),
            fit.kept,
            da.timer,
            db.timer,
            stopped_a && stopped_b ? "" : " (an end did not confirm its stop)",
            da.drop[0],
            da.drop[1],
            da.drop[2],
            da.drop[3],
            da.drop[4],
            db.drop[0],
            db.drop[1],
            db.drop[2],
            db.drop[3],
            db.drop[4],
            pct(path, 0.5),
            pct(path, 0.1),
            pct(path, 0.9),
            pct(turn, 0.5),
            pct(turn, 0.1),
            pct(turn, 0.9),
            pct(rtt, 0.5),
            pct(rtt, 0.1),
            pct(rtt, 0.9),
            fit.offset,
            fit.rate * 1e6,
            fit.resid_rms,
            window_med,
            window_max,
            noise);
        if (fit.ok) {
            fits.push_back({L.chip_a, L.chip_b, L.eth_a, L.eth_b, fit});
        }
    }
    // Around any loop of links the true clock offsets sum to zero, so a loop's closure is the sum of its links' path
    // asymmetries. Two links of one chip pair give their asymmetry difference outright. Over one link per pair, a
    // spanning tree gives every chip a time at one instant and each remaining pair closes one multi-chip loop, which
    // samples one port per chip. Were the ports' TX-RX latencies independent draws of spread sigma, a k-link closure
    // would be sigma * sqrt(k / 2) rms with sigma the pair differences' rms; a chip-wide common part shows as loops
    // beyond that, and a shared sign across pairs as structure by port.
    if (!fits.empty()) {
        std::map<std::pair<uint32_t, uint32_t>, size_t> first;  // pair -> the link the tree may use
        for (size_t i = 0; i < fits.size(); i++) {
            first.emplace(std::pair{fits[i].a, fits[i].b}, i);
        }
        std::map<uint32_t, double> t;
        std::map<uint32_t, std::pair<uint32_t, uint32_t>> tree;  // chip -> (parent, depth)
        std::vector<char> in_tree(fits.size(), 0);
        t[fits.front().a] = fits.front().f.x0;
        tree[fits.front().a] = {fits.front().a, 0};
        for (bool grew = true; grew;) {
            grew = false;
            for (const auto& [pair, i] : first) {
                const LinkFit& L = fits[i];
                const bool ha = t.count(L.a) != 0, hb = t.count(L.b) != 0;
                if (in_tree[i] || ha == hb) {
                    continue;
                }
                if (ha) {
                    t[L.b] = t[L.a] + L.f.at(t[L.a]);
                    tree[L.b] = {L.a, tree[L.a].second + 1};
                } else {
                    // The fit runs on the sender's time; one correction of a ppm-rate line is exact enough.
                    const double ta = t[L.b] - L.f.at(t[L.b]);
                    t[L.a] = t[L.b] - L.f.at(ta);
                    tree[L.a] = {L.b, tree[L.b].second + 1};
                }
                in_tree[i] = 1;
                grew = true;
            }
        }
        const auto tree_hops = [&](uint32_t a, uint32_t b) {
            uint32_t hops = 0;
            while (a != b) {
                if (tree[a].second >= tree[b].second) {
                    a = tree[a].first;
                } else {
                    b = tree[b].first;
                }
                hops++;
            }
            return hops;
        };
        std::vector<double> pairs, loops, predicted;
        for (size_t i = 0; i < fits.size(); i++) {
            const LinkFit& L = fits[i];
            if (in_tree[i] || t.count(L.a) == 0 || t.count(L.b) == 0) {
                continue;
            }
            const size_t rep = first.at({L.a, L.b});
            if (rep != i) {
                const double d = L.f.at(t[L.a]) - fits[rep].f.at(t[L.a]);
                pairs.push_back(d);
                std::printf(
                    "[eth_ptp_link] parallel links chips %u-%u: eth(%zu,%zu)->(%zu,%zu) minus eth(%zu,%zu)->(%zu,%zu) "
                    "%+.2f ns asymmetry difference\n",
                    L.a,
                    L.b,
                    L.eth_a.x,
                    L.eth_a.y,
                    L.eth_b.x,
                    L.eth_b.y,
                    fits[rep].eth_a.x,
                    fits[rep].eth_a.y,
                    fits[rep].eth_b.x,
                    fits[rep].eth_b.y,
                    d);
                continue;
            }
            const double closure = t[L.a] + L.f.at(t[L.a]) - t[L.b];
            const uint32_t k = tree_hops(L.a, L.b) + 1;
            loops.push_back(closure);
            predicted.push_back(std::sqrt(0.5 * k));
            std::printf(
                "[eth_ptp_link] loop closure chips %u-%u over %u links (this link minus the tree): %+.2f ns\n",
                L.a,
                L.b,
                k,
                closure);
        }
        const auto rms = [](const std::vector<double>& v) {
            double ss = 0;
            for (double x : v) {
                ss += x * x;
            }
            return v.empty() ? NAN : std::sqrt(ss / static_cast<double>(v.size()));
        };
        const double sigma = rms(pairs);
        double pred = 0;
        for (double p : predicted) {
            pred += sigma * sigma * p * p;
        }
        pred = predicted.empty() ? NAN : std::sqrt(pred / static_cast<double>(predicted.size()));
        std::printf(
            "[eth_ptp_link] %zu parallel pairs: per-port TX-RX spread sigma %.2f ns rms, so one link's asymmetry is "
            "%.2f ns rms; %zu multi-chip loops close to %.2f ns rms against %.2f ns predicted from independent draws\n",
            pairs.size(),
            sigma,
            sigma / std::sqrt(2.0),
            loops.size(),
            rms(loops),
            pred);
    }
    std::printf("[eth_ptp_link] %s\n", all_ok ? "PASS" : "FAIL");
    mesh_device->close();
    return all_ok ? 0 : 1;
}
