// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rand_op.hpp"

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/uniform/uniform.hpp"

namespace ttml::ops {

void rand_(const autograd::TensorPtr& tensor, float a, float b, std::optional<uint32_t> seed) {
    auto& gen = autograd::ctx().get_generator();
    uint32_t effective_seed = detail::avoid_zero_seed(seed.value_or(gen()));

    auto t = tensor->get_value(autograd::PreferredPrecision::FULL);
    ttnn::uniform(t, a, b, effective_seed);
    tensor->set_value(t);
}

autograd::TensorPtr rand(
    const ttnn::Shape& shape,
    float a,
    float b,
    std::optional<uint32_t> seed,
    tt::tt_metal::DataType dtype,
    tt::tt_metal::Layout layout) {
    auto* device = &autograd::ctx().get_device();
    auto t = ttnn::empty(shape, dtype, layout, device, ttnn::MemoryConfig{});
    auto tensor = ttml::autograd::create_tensor(t);
    rand_(tensor, a, b, seed);
    return tensor;
}

}  // namespace ttml::ops
