// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "linear.hpp"

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/distributed/comm_ops.hpp"
#include "ops/linear_op.hpp"

namespace ttml::modules::distributed {

RowParallelLinear::RowParallelLinear(
    uint32_t in_features, uint32_t out_features, bool has_bias, bool input_is_parallel) :
    m_input_is_parallel(input_is_parallel) {
    initialize_tensors(in_features, out_features, has_bias);

    create_name("row_parallel_linear");
    register_tensor(m_weight, "weight");
    if (m_bias != nullptr) {
        register_tensor(m_bias, "bias");
    }
}

autograd::TensorPtr RowParallelLinear::operator()(autograd::TensorPtr tensor) {
    if (!m_input_is_parallel) {
        tensor = ops::distributed::scatter(tensor, tensor->rank() - 1U);
    }

    tensor = ops::linear_op(tensor, m_weight, m_bias);
    tensor = ops::distributed::all_reduce(tensor, tensor->rank() - 1U);
    return tensor;
}

void RowParallelLinear::initialize_tensors(uint32_t in_features, uint32_t out_features, bool has_bias) {
    auto* device = &autograd::ctx().get_device();
    auto num_devices = static_cast<uint32_t>(device->num_devices());
    if (out_features % num_devices != 0) {
        throw std::runtime_error(fmt::format(
            "Output features must be divisible by the number of devices. Output features = {}, devices = {}",
            out_features,
            num_devices));
    }

    auto weight_shape = core::create_shape({1, 1, out_features / num_devices, in_features});
    throw std::runtime_error("Not completely implemented yet");
}

ColumnParallelLinear::ColumnParallelLinear(
    uint32_t in_features, uint32_t out_features, bool has_bias, bool gather_output) :
    m_gather_output(gather_output) {
    initialize_tensors(in_features, out_features, has_bias);

    create_name("column_parallel_linear");
    register_tensor(m_weight, "weight");
    if (m_bias != nullptr) {
        register_tensor(m_bias, "bias");
    }
}

autograd::TensorPtr ColumnParallelLinear::operator()(autograd::TensorPtr tensor) {
    tensor = ops::linear_op(tensor, m_weight, m_bias);
    if (m_gather_output) {
        tensor = ops::distributed::all_gather(tensor, tensor->rank() - 1U);
    }
    return tensor;
}

void ColumnParallelLinear::initialize_tensors(uint32_t in_features, uint32_t out_features, bool has_bias) {
    throw std::runtime_error("Not implemented yet");
}

}  // namespace ttml::modules::distributed
