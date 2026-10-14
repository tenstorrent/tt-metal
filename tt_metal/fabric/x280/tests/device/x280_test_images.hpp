// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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

}  // namespace x280::test
