// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Negative: the runtime values differ, but neither is a compile-time constant,
// so the checker must report unknown rather than guess a mismatch.
#include "api/compute/common.h"
#include "llk_math_binary_api.h"

void kernel_main(uint32_t a, uint32_t b) {
    llk_math_eltwise_binary_init<EltwiseBinaryType::ELWADD, BroadcastType::NONE, MathFidelity::HiFi4>(a, b, 0);
    llk_math_eltwise_binary<EltwiseBinaryType::ELWADD, BroadcastType::NONE, false, MathFidelity::HiFi4>(
        a + 1, b, 0, true);
}
