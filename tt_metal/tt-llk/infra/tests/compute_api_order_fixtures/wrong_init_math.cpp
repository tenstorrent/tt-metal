// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Positive: an init for a different operation sits between the binary init and
// the binary operation.
#include "api/compute/common.h"
#include "llk_math_binary_api.h"
#include "llk_math_matmul_api.h"

void kernel_main() {
    llk_math_eltwise_binary_init<EltwiseBinaryType::ELWADD, BroadcastType::NONE, MathFidelity::HiFi4>(0, 1, 0);
    llk_math_matmul_init<MathFidelity::HiFi4>(0, 1);
    llk_math_eltwise_binary<EltwiseBinaryType::ELWADD, BroadcastType::NONE, false, MathFidelity::HiFi4>(
        0, 1, 0, true);
}
