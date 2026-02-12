// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "metal/ttnn_all_includes.hpp"
#include "softmax_backward_device_operation_types.hpp"
#include "softmax_backward_program_factory.hpp"

namespace ttml::metal::ops::softmax_backward::device {

struct SoftmaxBackwardDeviceOperation {
    using operation_attributes_t = SoftmaxBackwardParams;
    using tensor_args_t = SoftmaxBackwardInputs;
    using spec_return_value_t = ttml::metal::ops::softmax_backward::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::softmax_backward::device::tensor_return_value_t;
    using program_factory_t = std::variant<SoftmaxBackwardFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
};

}  // namespace ttml::metal::ops::softmax_backward::device

namespace ttnn::prim {

ttnn::Tensor ttml_softmax_backward(const ttnn::Tensor& softmax_output, const ttnn::Tensor& upstream_grad, uint32_t dim);

}  // namespace ttnn::prim
