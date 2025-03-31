// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "linear.hpp"

#include <cmath>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "init/cpu_initializers.hpp"
#include "init/tensor_initializers.hpp"
#include "ops/binary_ops.hpp"
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

autograd::TensorPtr RowParallelLinear::operator()(const autograd::TensorPtr& tensor) {
    auto x = tensor;
    if (!m_input_is_parallel) {
        x = ops::distributed::scatter(x, tensor->get_rank() - 1U);
    }
    // do not pass bias
    x = ops::linear_op(x, m_weight, /* bias */ nullptr);
    x = ops::distributed::all_reduce(x);
    if (m_bias != nullptr) {
        x = ops::add(x, m_bias);
    }
    return x;
}

void RowParallelLinear::initialize_tensors(uint32_t in_features, uint32_t out_features, bool has_bias) {
    auto* device = &autograd::ctx().get_device();
    auto num_devices = static_cast<uint32_t>(device->num_devices());
    if (in_features % num_devices != 0) {
        throw std::runtime_error(fmt::format(
            "Input features must be divisible by the number of devices. Input features = {}, devices = {}",
            in_features,
            num_devices));
    }

    auto weight_shape = core::create_shape({1, 1, out_features, in_features});

    uint32_t rank = 4U;
    auto mesh_shape = device->shape();
    const float init_k = std::sqrt(1.F / static_cast<float>(in_features));

    ttml::core::XTensorToMeshVariant<float> shard_composer =
        ttml::core::ShardXTensorToMesh<float>(mesh_shape, rank - 1U);
    auto weight = init::uniform_init(weight_shape, init::UniformRange{-init_k, init_k});
    m_weight = autograd::create_tensor(
        ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(weight, device, shard_composer));

    if (has_bias) {
        auto bias_shape = core::create_shape({1, 1, 1, out_features});
        m_bias = ttml::autograd::create_tensor();
        init::uniform_init(m_bias, bias_shape, init::UniformRange{-init_k, init_k});
    }
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

autograd::TensorPtr ColumnParallelLinear::operator()(const autograd::TensorPtr& tensor) {
    auto x = tensor;
    x = ops::distributed::broadcast(x);
    x = ops::linear_op(x, m_weight, m_bias);
    if (m_gather_output) {
        x = ops::distributed::all_gather(x, tensor->get_rank() - 1U);
    }
    return x;
}

void ColumnParallelLinear::initialize_tensors(uint32_t in_features, uint32_t out_features, bool has_bias) {
    auto* device = &autograd::ctx().get_device();
    auto num_devices = static_cast<uint32_t>(device->num_devices());
    if (out_features % num_devices != 0) {
        throw std::runtime_error(fmt::format(
            "Output features must be divisible by the number of devices. Output features = {}, devices = {}",
            out_features,
            num_devices));
    }

    auto weight_shape = core::create_shape({1, 1, out_features, in_features});

    uint32_t rank = 4U;
    auto mesh_shape = device->shape();
    const float init_k = std::sqrt(1.F / static_cast<float>(in_features));

    ttml::core::XTensorToMeshVariant<float> shard_composer =
        ttml::core::ShardXTensorToMesh<float>(mesh_shape, rank - 2U);
    auto weight = init::uniform_init(weight_shape, init::UniformRange{-init_k, init_k});
    m_weight = autograd::create_tensor(
        ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(weight, device, shard_composer));

    if (has_bias) {
        auto bias_shape = core::create_shape({1, 1, 1, out_features});
        auto bias = init::uniform_init(bias_shape, init::UniformRange{-init_k, init_k});
        ttml::core::XTensorToMeshVariant<float> shard_composer =
            ttml::core::ShardXTensorToMesh<float>(mesh_shape, rank - 1U);
        m_bias = autograd::create_tensor(
            ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(bias, device, shard_composer));
    }
}

}  // namespace ttml::modules::distributed
