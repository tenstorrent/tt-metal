// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "x280/device/x280_runtime.hpp"

#include <tt-metalium/mesh_device.hpp>
#include <common/telemetry_table.h>

#include <iostream>

inline {
    void get_telemetry_table() {
        // RequirePrepared never primes; an unprimed chip is reported instead.
        auto opened =
            X280Runtime::open(mesh, RuntimeOpenConfig{.priming_policy = LimEccPrimingPolicy::RequirePrepared});
        if (!opened.ready()) {
            std::fprintf(stderr, "open: %s: %s\n", to_string(opened.outcome), opened.diagnostic.c_str());
            if (opened.requires_preparation()) {
                std::fprintf(stderr, "run with --prime first, then tt-smi -r 0\n");
            }
            return 1;
        }
        auto& runtime = *opened.runtime;

        const auto ids = runtime.tile_ids();
        if (ids.empty()) {
            std::fprintf(stderr, "no enabled X280 tiles\n");
            return 1;
        }
        auto& tile = runtime.tile(ids.front());

        const auto state = tile.inspect();
        if (!state.observation_valid) {
            std::fprintf(stderr, "inspect: %s\n", state.invalid_reason.c_str());
            return 1;
        }
        if (!state.resident_idle_live) {
            MultiTileBootRequest boot;
            boot.tiles = {ids.front()};
            boot.idle_image_path = std::string(X280_IMAGE_DIR) + "/firmware-lim-idle.bin";
            boot.allow_reset_release = true;
            const auto booted = runtime.boot_tiles(boot);
            if (!booted.ready()) {
                std::fprintf(stderr, "boot idle: %s\n", booted.diagnostic.c_str());
                for (const auto& report : booted.tiles) {
                    std::fprintf(stderr, "  %s\n", report.result.diagnostic.c_str());
                }
                return 1;
            }
            std::printf("booted resident idle\n");
        }
    }

    int main(int argc, char* argv[]) {}
