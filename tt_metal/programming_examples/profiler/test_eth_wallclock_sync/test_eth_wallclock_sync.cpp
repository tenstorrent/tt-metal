// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Measure the wall-clock offset between two devices over one ethernet link.
//
// A thin driver around eth_sync::measure_link() -- the same call the profiler makes at start() -- so this
// test exercises the production path rather than a parallel copy of it.

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
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
            cfg.n_samples = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--gap-us") == 0 && i + 1 < argc) {
            cfg.gap_us = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--all-links") == 0) {
            all_links = true;
        }
    }

    const auto shape = distributed::SystemMesh::instance().shape();
    printf("[ethsync] system mesh %ux%u\n", (unsigned)shape[0], (unsigned)shape[1]);
    auto mesh_device = distributed::MeshDevice::create(distributed::MeshDeviceConfig(shape));
    auto devices = mesh_device->get_devices();

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
