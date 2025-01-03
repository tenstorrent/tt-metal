// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "optimizer_base.hpp"

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"

namespace ttml::optimizers {

ttnn::Tensor synchronize_tensor(const ttnn::Tensor& tensor) {
    auto* device = &autograd::ctx().get_device();
    auto devices = device->get_devices().size();

    assert(devices >= 1U);

    // no need to synchronize if there is only one device
    if (devices == 1U) {
        return tensor;
    }

    // debug prints, remove before merge
    {
        auto mesh_shape = device->shape();
        ttml::core::MeshToXTensorVariant<float> identity_composer = ttml::core::VectorMeshToXTensor<float>(mesh_shape);
        auto xtensors_back = ttml::core::to_xtensor(tensor, identity_composer);

        for (auto& xtensor : xtensors_back) {
            fmt::print("xtensor: ");
            for (auto& s : xtensor.shape()) {
                fmt::print("{} ", s);
            }
            fmt::print("\n");
        }
    }

    // all_reduce Mean is not supported, use sum and divide by #devices
    auto result = ttnn::experimental::all_reduce(
        tensor, ttnn::operations::reduction::ReduceType::Sum, 1, std::nullopt, ttnn::ccl::Topology::Ring);
    result = ttnn::multiply(result, 1.0F / devices);
    return result;
}

OptimizerBase::OptimizerBase(serialization::NamedParameters&& parameters) : m_parameters(std::move(parameters)) {
}

void OptimizerBase::print_stats() const {
    fmt::print("\n\nOptimization parameters values and gradients:\n");
    for (const auto& [name, tensor] : m_parameters) {
        core::print_tensor_stats(tensor->get_value(), fmt::format("{}/value", name));
        if (tensor->is_grad_initialized()) {
            core::print_tensor_stats(tensor->get_grad(), fmt::format("{}/gradient", name));
        }
    }
    fmt::print("=================================================\n");
}

}  // namespace ttml::optimizers
