// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "log_softmax_device_operation_types.hpp"
#include "log_softmax_program_factory.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::log_softmax::device {

struct LogSoftmaxDeviceOperation {
    using operation_attributes_t = ttml::metal::ops::log_softmax::device::operation_attributes_t;
    using tensor_args_t = ttml::metal::ops::log_softmax::device::tensor_args_t;
    using spec_return_value_t = ttml::metal::ops::log_softmax::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::log_softmax::device::tensor_return_value_t;
    using program_factory_t = std::variant<LogSoftmaxProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::log_softmax::device

namespace ttnn::prim {

// Row state: three FP32 column-vector tiles per input tile row (maximum, selected maximum index, log1p(tail)).
ttml::metal::ops::log_softmax::device::LogSoftmaxDeviceOperation::tensor_return_value_t ttml_log_softmax_fw(
    const ttnn::Tensor& input_tensor);

ttnn::Tensor ttml_log_softmax_bw(
    const ttnn::Tensor& input_tensor, const ttnn::Tensor& upstream_tensor, const ttnn::Tensor& state_tensor);

}  // namespace ttnn::prim
