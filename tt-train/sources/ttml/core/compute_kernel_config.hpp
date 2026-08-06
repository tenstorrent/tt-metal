// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>

namespace ttml::core {

// Highest math fidelity recommended with fp32 dest accumulation on this arch.
// Wormhole HW bug #38306: HiFi4 with fp32 dest acc corrupts occasional outputs by a power of
// two. HiFi3 is the recommended workaround (#39562) and cuts the observed rate by ~550x, but
// does not eliminate it — the RTL fault has no fix in WH silicon. Blackhole is unaffected.
[[nodiscard]] tt::tt_metal::MathFidelity max_fidelity_with_fp32_acc();

class ComputeKernelConfig {
public:
    static ttnn::WormholeComputeKernelConfig precise();
    static ttnn::WormholeComputeKernelConfig softmax();
    static ttnn::WormholeComputeKernelConfig matmul();
    static ttnn::WormholeComputeKernelConfig fast();
};

}  // namespace ttml::core
