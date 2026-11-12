// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "x280/device/x280_runtime.hpp"

#include <tt-metalium/mesh_device.hpp>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

using namespace x280;

namespace {

void show_chip_state(const ChipOpenResult& chip_result) {
    std::cout << "Chip: " << chip_result.chip_id;
    switch (chip_result.outcome) {
        case RuntimeOpenOutcome::Ready: std::cout << " is ready, you can load your active firmware.\n"; break;

        case RuntimeOpenOutcome::PreparationRequired: std::cout << " must be primed.\n"; break;

        case RuntimeOpenOutcome::ResetRequired:
            std::cout << " reset via tt-smi -r " << chip_result.chip_id << "\n";
            break;

        case RuntimeOpenOutcome::Failed:
            std::cout << " is in a failed state: " << to_string(chip_result.outcome) << ", " << chip_result.diagnostic
                      << "\n";
            break;

        default: std::cout << " is in an unknown state.\n"; break;
    }
}

int prime_chips(const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh) {
    // AutoPrime (the default) primes an unprimed chip and reports ResetRequired.
    auto result = X280Runtime::open(mesh);
    std::cout << "There are " << result.chips.size() << " chips in this system.\n";
    if (result.ready()) {
        std::cout << "All chips are primed.\n";
        return 0;
    }

    for (const auto& curr_outcome : result.chips) {
        show_chip_state(curr_outcome);
    }
    return 1;
}

// Boots the standalone firmware on one tile per chip. Releasing reset is one-way: the firmware
// owns the tile until the chip is reset with tt-smi -r.
int start_firmware(const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh, const std::string& image) {
    // RequirePrepared never primes; an unprimed chip is reported instead.
    auto result = X280Runtime::open(mesh, RuntimeOpenConfig{.priming_policy = LimEccPrimingPolicy::RequirePrepared});
    if (!result.ready()) {
        std::cout << "Not all chips are ready.\n";
        for (const auto& curr_outcome : result.chips) {
            show_chip_state(curr_outcome);
        }
        return 1;
    }
    auto& runtime = *result.runtime;

    auto tile_ids = runtime.tile_ids();

    // We only start one tile per chip. tile_ids() is sorted by chip, so unique() keeps the first of each.
    auto last = std::unique(std::begin(tile_ids), std::end(tile_ids), [](const auto& t1, const auto& t2) {
        return t1.chip_id == t2.chip_id;
    });
    tile_ids.erase(last, tile_ids.end());

    if (tile_ids.empty()) {
        std::cout << "No enabled X280 tiles.\n";
        return 1;
    }

    for (const auto& curr_tile_id : tile_ids) {
        std::cout << "Starting firmware on chip " << curr_tile_id.chip_id << "\n";

        MultiTileBootRequest boot;
        boot.tiles = {curr_tile_id};
        boot.image_kind = BootImageKind::Standalone;
        boot.idle_image_path = image;
        boot.allow_reset_release = true;
        auto booted = runtime.boot_tiles(boot);
        std::cout << booted.diagnostic << "\n";
        for (const auto& report : booted.tiles) {
            std::cout << "\t" << report.result.diagnostic << "\n";
        }
        if (!booted.ready()) {
            return 1;
        }

        // ready() also covers "already running", where nothing was released and our image never loaded.
        for (const auto& report : booted.tiles) {
            if (report.result.outcome != BootOutcome::Booted) {
                std::cout << "Our firmware was not started on chip " << curr_tile_id.chip_id
                          << ". If idle firmware is live from an earlier run, reset the chip (tt-smi -r "
                          << curr_tile_id.chip_id << ") and run --start again.\n";
                return 1;
            }
        }
    }

    std::cout << "Firmware is running. It keeps running until the chip is reset (tt-smi -r).\n";
    return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
    bool do_prime = false;
    bool do_start = false;
    std::string image = std::string(X280_IMAGE_DIR) + "/firmware-telemetry-v2.bin";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--prime") == 0) {
            do_prime = true;
        } else if (std::strcmp(argv[i], "--start") == 0) {
            do_start = true;
        } else {
            image = argv[i];
        }
    }

    if (do_prime && do_start) {
        std::cout << "Use either --prime or --start, not both.\n";
        return 1;
    }

    if (!do_prime && !do_start) {
        std::cout << "usage: " << argv[0] << " --prime | --start [image.bin]\n";
        return 0;
    }

    auto mesh =
        tt::tt_metal::distributed::MeshDevice::create(tt::tt_metal::distributed::MeshDeviceConfig(std::nullopt));

    if (do_prime) {
        std::cout << "Priming system to get it ready for use.\n";
        return prime_chips(mesh);
    }

    std::cout << "Starting standalone firmware: " << image << "\n";
    return start_firmware(mesh, image);
}
