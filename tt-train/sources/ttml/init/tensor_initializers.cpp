// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor_initializers.hpp"

#include <ttnn/operations/data_movement/copy/copy.hpp>
#include <ttnn/tensor/tensor_utils.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "cpu_initializers.hpp"
namespace ttml::init {

void uniform_init(ttml::autograd::TensorPtr& t, const ttnn::Shape& shape, UniformRange range) {
    auto* device = &autograd::ctx().get_device();
    assert(device);
    size_t volume = shape.volume();
    std::vector<float> vec(volume);
    uniform_init(vec, range);

    t->set_value(ttml::core::from_vector(vec, shape, device));
}

void normal_init(ttml::autograd::TensorPtr& t, const ttnn::Shape& shape, NormalParams params) {
    auto* device = &autograd::ctx().get_device();
    assert(device);
    size_t volume = shape.volume();
    std::vector<float> vec(volume);
    normal_init(vec, params);
    t->set_value(ttml::core::from_vector(vec, shape, device));
}

void constant_init(ttml::autograd::TensorPtr& t, const ttnn::Shape& shape, float value) {
    auto* device = &autograd::ctx().get_device();
    t->set_value(core::full(shape, value, device));
}

void xavier_uniform_init(ttml::autograd::TensorPtr& t, const ttnn::Shape& shape, FanParams params) {
    auto* device = &autograd::ctx().get_device();
    assert(device);
    size_t volume = shape.volume();
    std::vector<float> vec(volume);
    xavier_uniform_init(vec, params);

    t->set_value(ttml::core::from_vector(vec, shape, device));
}

void xavier_normal_init(ttml::autograd::TensorPtr& t, const ttnn::Shape& shape, FanParams params) {
    auto* device = &autograd::ctx().get_device();
    assert(device);
    size_t volume = shape.volume();
    std::vector<float> vec(volume);
    xavier_normal_init(vec, params);

    t->set_value(ttml::core::from_vector(vec, shape, device));
}

void kaiming_uniform_init(ttml::autograd::TensorPtr& t, const ttnn::Shape& shape, int fan_in) {
    auto* device = &autograd::ctx().get_device();
    assert(device);
    size_t volume = shape.volume();
    std::vector<float> vec(volume);
    kaiming_uniform_init(vec, fan_in);

    t->set_value(ttml::core::from_vector(vec, shape, device));
}

void kaiming_normal_init(ttml::autograd::TensorPtr& t, const ttnn::Shape& shape, int fan_out) {
    auto* device = &autograd::ctx().get_device();
    assert(device);
    size_t volume = shape.volume();
    std::vector<float> vec(volume);
    kaiming_normal_init(vec, fan_out);

    t->set_value(ttml::core::from_vector(vec, shape, device));
}
}  // namespace ttml::init
