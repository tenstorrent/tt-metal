// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// No-op compute for intentional_hang. Produces no output tiles.

#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    // intentionally empty — never pushes to cb_out
}
