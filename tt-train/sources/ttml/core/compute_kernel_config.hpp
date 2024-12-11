// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>

namespace ttml::core {

class ComputeKernelConfig {
public:
    static ttnn::WormholeComputeKernelConfig precise();
    static ttnn::WormholeComputeKernelConfig softmax();
    static ttnn::WormholeComputeKernelConfig matmul();
    static ttnn::WormholeComputeKernelConfig fast();
};

}  // namespace ttml::core
