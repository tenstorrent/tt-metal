// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Negative: the init is guarded by an if constexpr that resolves true; the
// discarded branch must not be treated as a path without the init.
#include "api/compute/common.h"
#include "llk_math_binary_api.h"

void kernel_main() {
    if constexpr (true) {
        llk_math_eltwise_binary_init<EltwiseBinaryType::ELWADD, BroadcastType::NONE, MathFidelity::HiFi4>(0, 1, 0);
    }
    llk_math_eltwise_binary<EltwiseBinaryType::ELWADD, BroadcastType::NONE, false, MathFidelity::HiFi4>(
        0, 1, 0, true);
}
