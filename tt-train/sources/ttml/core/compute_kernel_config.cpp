// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_config.hpp"

#include <tt-metalium/hal.hpp>

namespace ttml::core {

tt::tt_metal::MathFidelity max_fidelity_with_fp32_acc() {
    return tt::tt_metal::hal::get_arch() == tt::ARCH::WORMHOLE_B0 ? tt::tt_metal::MathFidelity::HiFi3
                                                                  : tt::tt_metal::MathFidelity::HiFi4;
}

ttnn::WormholeComputeKernelConfig ComputeKernelConfig::precise() {
    ttnn::WormholeComputeKernelConfig config;
    config.fp32_dest_acc_en = true;
    config.math_approx_mode = false;
    config.math_fidelity = max_fidelity_with_fp32_acc();
    config.packer_l1_acc = true;
    return config;
}

ttnn::WormholeComputeKernelConfig ComputeKernelConfig::softmax() {
    ttnn::WormholeComputeKernelConfig config;
    config.fp32_dest_acc_en = false;
    config.math_approx_mode = false;
    config.math_fidelity = tt::tt_metal::MathFidelity::HiFi4;
    config.packer_l1_acc = true;
    return config;
}

ttnn::WormholeComputeKernelConfig ComputeKernelConfig::matmul() {
    ttnn::WormholeComputeKernelConfig config;
    config.fp32_dest_acc_en = true;
    config.math_approx_mode = false;
    config.math_fidelity = max_fidelity_with_fp32_acc();
    config.packer_l1_acc = true;
    return config;
}

ttnn::WormholeComputeKernelConfig ComputeKernelConfig::fast() {
    ttnn::WormholeComputeKernelConfig config;
    config.fp32_dest_acc_en = false;
    config.math_approx_mode = true;
    config.math_fidelity = tt::tt_metal::MathFidelity::LoFi;
    config.packer_l1_acc = false;
    return config;
}

}  // namespace ttml::core
