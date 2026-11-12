// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Test app: confirms the standalone telemetry firmware is still running after the loader has exited.
//
// It never opens X280Runtime. open() treats a tile that is out of reset without a live resident
// idle as needing a chip reset, so it cannot be used here. Instead this reads the heartbeat words
// straight through the Metal transport, which is read-only and touches nothing else.
//
// usage: x280_telemetry_v2_heartbeat_check [--samples N] [--interval-ms N]

#include "x280_metal_transport.hpp"

#include "x280_address_map.h"

#include <tt-metalium/mesh_device.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

namespace {

struct TileProbe {
    int chip_id;
    tt::umd::CoreCoord core;
    bool magic_ok = false;
    bool counter_moved = false;
    uint64_t first_counter = 0;
    uint64_t last_counter = 0;
};

}  // namespace

int main(int argc, char* argv[]) {
    int samples = 5;
    int interval_ms = 200;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
            samples = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--interval-ms") == 0 && i + 1 < argc) {
            interval_ms = std::atoi(argv[++i]);
        } else {
            std::cout << "usage: " << argv[0] << " [--samples N] [--interval-ms N]\n";
            return 0;
        }
    }
    if (samples < 2 || interval_ms < 1) {
        std::cout << "--samples must be at least 2 and --interval-ms at least 1.\n";
        return 1;
    }

    const uint64_t expected_magic = static_cast<uint64_t>(X280_STANDALONE_HEARTBEAT_MAGIC);

    auto mesh =
        tt::tt_metal::distributed::MeshDevice::create(tt::tt_metal::distributed::MeshDeviceConfig(std::nullopt));
    auto transport = x280::X280MetalTransport::bind(*mesh);

    // Every enabled L2CPU core on every chip we can reach over PCIe. We don't know which tile the
    // loader started, so check them all and report the ones that carry our magic.
    std::vector<TileProbe> probes;
    for (int chip_id : x280::sorted_usable_chip_ids(*mesh, *transport)) {
        if (!transport->is_mmio_local(chip_id)) {
            std::cout << "Chip " << chip_id << " is not PCIe-local, skipping.\n";
            continue;
        }
        for (const auto& core : transport->l2cpu_cores(chip_id)) {
            probes.push_back(TileProbe{.chip_id = chip_id, .core = core});
        }
    }

    // Sample the counter several times. The firmware ticks it about once a millisecond.
    for (int s = 0; s < samples; ++s) {
        for (auto& p : probes) {
            const uint64_t magic = transport->read_u64(p.chip_id, p.core, X280_STANDALONE_HEARTBEAT_ADDR);
            if (magic != expected_magic) {
                continue;
            }
            const uint64_t counter = transport->read_u64(p.chip_id, p.core, X280_STANDALONE_HEARTBEAT_COUNTER_ADDR);
            if (!p.magic_ok) {
                p.magic_ok = true;
                p.first_counter = counter;
            } else if (counter != p.last_counter) {
                p.counter_moved = true;
            }
            p.last_counter = counter;
        }
        if (s + 1 < samples) {
            std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
        }
    }

    int running = 0;
    int stuck = 0;
    for (const auto& p : probes) {
        if (!p.magic_ok) {
            continue;  // Not ours: no standalone firmware on this tile.
        }
        std::cout << "Chip " << p.chip_id << " L2CPU (" << p.core.x << "," << p.core.y << "): counter "
                  << p.first_counter << " -> " << p.last_counter;
        if (p.counter_moved) {
            std::cout << "  RUNNING (+" << (p.last_counter - p.first_counter) << ")\n";
            ++running;
        } else {
            std::cout << "  STUCK (magic present, counter not moving)\n";
            ++stuck;
        }
    }

    if (running == 0 && stuck == 0) {
        std::cout << "No tile shows the standalone heartbeat magic. Run the loader with --start first.\n";
        return 1;
    }
    return stuck == 0 ? 0 : 2;
}
