// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "x280/device/x280_runtime.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

namespace x280::test {

/**
 * The firmware image a device test runs: the path in @p environment when that variable is set
 * and non-empty, otherwise @p file in the build's image directory (X280_IMAGE_DIR, which
 * x280_images fills).
 * The returned path is not checked; a missing file fails where the test opens it.
 */
inline std::string test_image(const char* environment, const char* file) {
    if (const char* path = std::getenv(environment); path != nullptr && *path != '\0') {
        return path;
    }
    return std::string(X280_IMAGE_DIR) + "/" + file;
}

/**
 * The skip message for an open that returned no runtime: its diagnostic, plus the operator steps
 * when the open reports preparation or a reset. Device tests open with RequirePrepared and never
 * prime; priming is the opt-in x280_prime_test.
 */
inline std::string unprepared(const RuntimeOpenResult& opened) {
    std::string why = opened.diagnostic;
    if (opened.requires_reset()) {
        why += "; reset the device (tt-smi -r <device>)";
    }
    if (opened.requires_preparation() || opened.requires_reset()) {
        why +=
            "; an unprimed device is primed once by X280_ALLOW_LIM_ECC_PRIME=1 x280_prime_test, then "
            "tt-smi -r <device>";
    }
    return why;
}

/**
 * Brings @p id into resident idle unless it already runs it, through X280Runtime::boot_tiles with
 * reset release allowed and the image test_image("X280_IDLE_FW_IMAGE", "firmware-lim-idle.bin").
 * A cold boot is one-way for the chip-reset lifetime and leaves PLL4 at kSafePllMhz; no other
 * tile is booted.
 *
 * Records against the running test: a skip when the tile needs an ASIC reset first, a fatal
 * failure for any other refusal. From SetUp either one stops the test body; elsewhere the caller
 * checks IsSkipped() and HasFatalFailure().
 */
inline void boot_resident_idle(X280Runtime& runtime, const TileId& id) {
    const auto state = runtime.tile(id).inspect();
    ASSERT_TRUE(state.observation_valid) << state.invalid_reason;
    if (state.resident_idle_live) {
        return;
    }

    MultiTileBootRequest request;
    request.tiles = {id};
    request.idle_image_path = test_image("X280_IDLE_FW_IMAGE", "firmware-lim-idle.bin");
    request.allow_reset_release = true;
    const auto booted = runtime.boot_tiles(request);
    if (booted.ready()) {
        return;
    }
    if (!booted.tiles.empty() && reset_required(booted.tiles.front().result.outcome)) {
        GTEST_SKIP() << "resident idle cannot boot until an ASIC reset (tt-smi -r <device>): "
                     << booted.tiles.front().result.diagnostic;
    }
    FAIL() << "resident idle boot failed: " << booted.diagnostic
           << (booted.tiles.empty() ? std::string{} : "; " + booted.tiles.front().result.diagnostic);
}

}  // namespace x280::test
