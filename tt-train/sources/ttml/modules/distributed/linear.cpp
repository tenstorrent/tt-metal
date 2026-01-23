// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
    uint32_t in_features,
    uint32_t out_features,
    bool has_bias,
    bool input_is_parallel,
    std::optional<uint32_t> shard_dim) :
    m_input_is_parallel(input_is_parallel), /* input is parallel across TP axis */
    m_shard_dim(shard_dim) /* shard dimension in the device mesh */ {
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
        x = ops::distributed::scatter(x, tensor->get_rank() - 1U, m_shard_dim);
    }
    // do not pass bias
    x = ops::linear_op(x, m_weight, /* bias */ nullptr);

    /*
        All reduce with noop backward to avoid double all reduce in backward pass. This happens due to broadcast (no op in forward pass)
        does all reduce in backward pass. See similar implementation in fairscale for more details.
        https://github.com/facebookresearch/fairscale/blob/main/fairscale/nn/model_parallel/mappings.py#L102
    */
    x = ops::distributed::all_reduce(x, /* noop_backward */ m_input_is_parallel, m_shard_dim);
    if (m_bias != nullptr) {
        x = ops::add(x, m_bias);
    }
    return x;
}

void RowParallelLinear::initialize_tensors(uint32_t in_features, uint32_t out_features, bool has_bias) {
    auto* device = &autograd::ctx().get_device();
    auto mesh_shape = device->shape();

    // Determine TP size based on mesh shape and shard_dim
    uint32_t tp_size = static_cast<uint32_t>(device->num_devices());
    if (m_shard_dim.has_value() && mesh_shape.dims() == 2) {
        // For 2D mesh with explicit shard_dim: TP size is mesh dimension specified by shard_dim
        tp_size = mesh_shape[m_shard_dim.value()];
    }

    if (in_features % tp_size != 0) {
        throw std::runtime_error(fmt::format(
            "Input features must be divisible by the TP size. Input features = {}, TP size = {}",
            in_features,
            tp_size));
    }

    const auto weight_shape = ttnn::Shape({1, 1, out_features, in_features});
    uint32_t rank = 4U;
    const float init_k = std::sqrt(1.F / static_cast<float>(in_features));

    auto weight = init::uniform_init(weight_shape, init::UniformRange{-init_k, init_k});

    const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, rank - 1U, m_shard_dim);
    m_weight = autograd::create_tensor(
        ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(weight, device, ttnn::Layout::TILE, mapper.get()));

    if (has_bias) {
        auto bias_shape = ttnn::Shape({1, 1, 1, out_features});
        m_bias = ttml::autograd::create_tensor();
        init::uniform_init(m_bias, bias_shape, init::UniformRange{-init_k, init_k});
    }
}

ColumnParallelLinear::ColumnParallelLinear(
    uint32_t in_features,
    uint32_t out_features,
    bool has_bias,
    bool gather_output,
    std::optional<uint32_t> shard_dim) :
    m_gather_output(gather_output),
    m_shard_dim(shard_dim) {
    initialize_tensors(in_features, out_features, has_bias);

    create_name("column_parallel_linear");
    register_tensor(m_weight, "weight");
    if (m_bias != nullptr) {
        register_tensor(m_bias, "bias");
    }
}

autograd::TensorPtr ColumnParallelLinear::operator()(const autograd::TensorPtr& tensor) {
    auto x = tensor;
    // Broadcast data along TP dimension to ensure all TP devices in each DP group have the same data
    x = ops::distributed::broadcast(x, m_shard_dim);
    x = ops::linear_op(x, m_weight, m_bias);
    if (m_gather_output) {
        // All-gather output along TP dimension to gather sharded outputs within each DP group
        x = ops::distributed::all_gather(x, tensor->get_rank() - 1U, m_shard_dim);
    }
    return x;
}

void ColumnParallelLinear::initialize_tensors(uint32_t in_features, uint32_t out_features, bool has_bias) {
    auto* device = &autograd::ctx().get_device();
    auto mesh_shape = device->shape();

    // Determine TP size based on mesh shape and shard_dim
    uint32_t tp_size = 1U;
    if (m_shard_dim.has_value() && mesh_shape.dims() == 2) {
        // For 2D mesh with explicit shard_dim: TP size is mesh dimension specified by shard_dim
        tp_size = mesh_shape[m_shard_dim.value()];
    } else {
        // 1D mesh: use all devices
        tp_size = static_cast<uint32_t>(device->num_devices());
    }

    if (out_features % tp_size != 0) {
        throw std::runtime_error(fmt::format(
            "Output features must be divisible by the TP size. Output features = {}, TP size = {}",
            out_features,
            tp_size));
    }

    auto weight_shape = ttnn::Shape({1, 1, out_features, in_features});
    uint32_t rank = 4U;
    const float init_k = std::sqrt(1.F / static_cast<float>(in_features));

    auto weight = init::uniform_init(weight_shape, init::UniformRange{-init_k, init_k});

    const auto weight_mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, rank - 2U, m_shard_dim);
    m_weight = autograd::create_tensor(
        ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(weight, device, ttnn::Layout::TILE, weight_mapper.get()));

    if (has_bias) {
        const auto bias_shape = ttnn::Shape({1, 1, 1, out_features});
        const auto bias = init::uniform_init(bias_shape, init::UniformRange{-init_k, init_k});
        const auto bias_mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, rank - 1U, m_shard_dim);
        m_bias = autograd::create_tensor(
            ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(bias, device, ttnn::Layout::TILE, bias_mapper.get()));
    }
}

}  // namespace ttml::modules::distributed
