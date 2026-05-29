// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "polynorm_fw_device_operation_types.hpp"
#include "polynorm_fw_program_factory.hpp"

namespace ttml::metal::ops::polynorm3_fw::device {

// Device-level PolyNorm3 forward operation entry points used by TTNN operation launcher.
struct PolyNorm3ForwardDeviceOperation {
    using operation_attributes_t = ttml::metal::ops::polynorm3_fw::device::PolyNorm3FWAttributes;
    using tensor_args_t = ttml::metal::ops::polynorm3_fw::device::PolyNorm3FWTensorArgs;
    using spec_return_value_t = ttml::metal::ops::polynorm3_fw::device::PolyNorm3FWSpecReturn;
    using tensor_return_value_t = ttml::metal::ops::polynorm3_fw::device::PolyNorm3FWTensorReturn;
    using program_factory_t = std::variant<PolyNorm3ForwardProgramFactory>;

    // Validate tensor/device/layout constraints before kernel launch.
    static void validate_on_program_cache_miss(const PolyNorm3FWAttributes&, const PolyNorm3FWTensorArgs&);
    static PolyNorm3FWSpecReturn compute_output_specs(const PolyNorm3FWAttributes&, const PolyNorm3FWTensorArgs&);
    static PolyNorm3FWTensorReturn create_output_tensors(const PolyNorm3FWAttributes&, const PolyNorm3FWTensorArgs&);
    static ttsl::hash::hash_t compute_program_hash(const PolyNorm3FWAttributes&, const PolyNorm3FWTensorArgs&);
};

}  // namespace ttml::metal::ops::polynorm3_fw::device

namespace ttnn::prim {

ttml::metal::ops::polynorm3_fw::device::PolyNorm3ForwardDeviceOperation::tensor_return_value_t ttml_polynorm3_fw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight,
    const ttnn::Tensor& bias,
    float epsilon = 1e-5F,
    const std::optional<ttnn::Tensor>& preallocated_output = std::nullopt);

}  // namespace ttnn::prim
