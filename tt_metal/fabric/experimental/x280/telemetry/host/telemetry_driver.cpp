// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Driver for the telemetry image on the first X280 tile of chip 0.
//
//   x280_telemetry_driver --prime     prime LIM ECC and exit (then: tt-smi -r 0)
//   x280_telemetry_driver [image.bin] boot resident idle if needed, run the image, return to idle
//
// Never resets the chip itself; when a reset is needed it says so and exits.

#include "x280/device/x280_runtime.hpp"

#include <tt-metalium/mesh_device.hpp>

#include <chrono>
#include <cstdio>
#include <string>

namespace {

using namespace x280;
using namespace std::chrono_literals;

constexpr uint64_t kRunValue = 1;

int prime(const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh) {
    // AutoPrime (the default) primes an unprimed chip and reports ResetRequired.
    auto result = X280Runtime::open(mesh);

    if (result.ready()) {
        std::cout << "All chips are primed.\n";
        return 0;
    }

    for (const auto& curr_outcome : result.chips) {
        std::cout << "Chip: " << curr_outcome.chip_id;
        switch (curr_outcome.outcome) {
            case RuntimeOpenOutcome::Ready: std::cout << " is ready, you can load your active firmware.\n"; break;

            case RuntimeOpenOutcome::PreparationRequired: std::cout << " must be primed.\n"; break;

            case RuntimeOpenOutcome::ResetRequired:
                std::cout << " reset via tt-smi -r " << curr_outcome.chip_id << "\n";
                break;

            case RuntimeOpenOutcome::Failed:
                std::cout << " is in a faileedd state.\n";
                std::cout << " prime failed: " << to_string(curr_outcome.outcome) << ", " << curr_outcome.diagnostic
                          << "\n";
                break;

            default: std::cout << " is in an unknown state.\n"; break;
        }
    }

    return 1;
}

int run(const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh, const std::string& image) {
    auto result = X280Runtime::open(mesh, RuntimeOpenConfig{.priming_policy = LimEccPrimingPolicy::RequirePrepared});
    if (!opened.ready()) {:

//    auto opened = X280Runtime::open(mesh, RuntimeOpenConfig{.priming_policy = LimEccPrimingPolicy::RequirePrepared});
//    if (!opened.ready()) {
//        std::fprintf(stderr, "open: %s: %s\n", to_string(opened.outcome), opened.diagnostic.c_str());
//        if (opened.requires_preparation()) {
//            std::fprintf(stderr, "run with --prime first, then tt-smi -r 0\n");
//        }
//        return 1;
//    }
//    auto& runtime = *opened.runtime;
//
//    const auto ids = runtime.tile_ids();
//    if (ids.empty()) {
//        std::fprintf(stderr, "no enabled X280 tiles\n");
//        return 1;
//    }
//    auto& tile = runtime.tile(ids.front());
//
//    const auto state = tile.inspect();
//    if (!state.observation_valid) {
//        std::fprintf(stderr, "inspect: %s\n", state.invalid_reason.c_str());
//        return 1;
//    }
//    if (!state.resident_idle_live) {
//        MultiTileBootRequest boot;
//        boot.tiles = {ids.front()};
//        boot.idle_image_path = std::string(X280_IMAGE_DIR) + "/firmware-lim-idle.bin";
//        boot.allow_reset_release = true;
//        const auto booted = runtime.boot_tiles(boot);
//        if (!booted.ready()) {
//            std::fprintf(stderr, "boot idle: %s\n", booted.diagnostic.c_str());
//            for (const auto& report : booted.tiles) {
//                std::fprintf(stderr, "  %s\n", report.result.diagnostic.c_str());
//            }
//            return 1;
//        }
//        std::printf("booted resident idle\n");
//    }
//
//    ActiveRunRequest request;
//    request.image_path = image;
//    request.run_value = kRunValue;
//    const auto enqueued = runtime.enqueue(tile, request);
//    if (!enqueued.enqueued()) {
//        std::fprintf(stderr, "enqueue: %s: %s\n", to_string(enqueued.outcome), enqueued.error.c_str());
//        if (enqueued.session) {
//            runtime.cleanup(enqueued.session, request.cleanup_timeout);
//        }
//        return 1;
//    }
//    std::printf("running %s\n", image.c_str());
//
//    // Fires up all the harts and waits for them to start running.
//    const auto waited = runtime.session(enqueued.session).wait_until_running(3000ms);
//    int rc = 0;
//    if (waited.succeeded()) {
//        std::printf("firmware running\n");
//    } else {
//        std::fprintf(stderr, "wait: %s\n", describe(waited).c_str());
//        rc = 1;
//    }
    //    const auto waited = runtime.session(enqueued.session).wait_until_result_ready(3000ms);
    //    int rc = 0;
    //    if (waited.succeeded()) {
    //        for (const auto& hart : waited.hartstates) {
    //            std::printf("hart %u result 0x%llx\n", hart.hart_id, static_cast<unsigned long
    //            long>(hart.state.result));
    //        }
    //    } else {
    //        std::fprintf(stderr, "wait: %s\n", describe(waited).c_str());
    //        rc = 1;
    //    }

    const auto cleaned = runtime.cleanup(enqueued.session, request.cleanup_timeout);
    }
    if (!cleaned.cleaned_up()) {
        std::fprintf(stderr, "cleanup: %s\n", describe(cleaned).c_str());
        rc = 1;
    }
    return rc;
}

}  // namespace

int main(int argc, char** argv) {
    bool do_prime = false;
    std::string image = std::string(X280_IMAGE_DIR) + "/firmware-telemetry.bin";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--prime") {
            do_prime = true;
        } else if (arg == "-h" || arg == "--help") {
            std::printf("usage: %s [--prime] [image.bin]\n", argv[0]);
            return 0;
        } else {
            image = arg;
        }
    }

    auto mesh =
        tt::tt_metal::distributed::MeshDevice::create(tt::tt_metal::distributed::MeshDeviceConfig(std::nullopt));
    // auto mesh = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
    return do_prime ? prime(mesh) : run(mesh, image);
}
