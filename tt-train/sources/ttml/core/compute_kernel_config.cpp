// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_config.hpp"

namespace ttml::core {

ttnn::WormholeComputeKernelConfig ComputeKernelConfig::precise() {
    ttnn::WormholeComputeKernelConfig config;
    config.fp32_dest_acc_en = true;
    config.math_approx_mode = false;
    config.math_fidelity = MathFidelity::HiFi4;
    config.packer_l1_acc = true;
    return config;
}

ttnn::WormholeComputeKernelConfig ComputeKernelConfig::softmax() {
    ttnn::WormholeComputeKernelConfig config;
    config.fp32_dest_acc_en = false;
    config.math_approx_mode = false;
    config.math_fidelity = MathFidelity::HiFi4;
    config.packer_l1_acc = true;
    return config;
}

ttnn::WormholeComputeKernelConfig ComputeKernelConfig::matmul() {
    ttnn::WormholeComputeKernelConfig config;
    config.fp32_dest_acc_en = true;
    config.math_approx_mode = false;
    config.math_fidelity = MathFidelity::HiFi4;
    config.packer_l1_acc = true;
    return config;
}

ttnn::WormholeComputeKernelConfig ComputeKernelConfig::fast() {
    ttnn::WormholeComputeKernelConfig config;
    config.fp32_dest_acc_en = false;
    config.math_approx_mode = true;
    config.math_fidelity = MathFidelity::LoFi;
    config.packer_l1_acc = false;
    return config;
}

}  // namespace ttml::core
