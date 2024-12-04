// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "init/cpu_initializers.hpp"

namespace ttml::init {

void uniform_init(ttml::autograd::TensorPtr& t, const ttnn::Shape& shape, UniformRange range);

void normal_init(ttml::autograd::TensorPtr& t, const ttnn::Shape& shape, NormalParams params);

void constant_init(ttml::autograd::TensorPtr& t, const ttnn::Shape& shape, float value);

void xavier_uniform_init(ttml::autograd::TensorPtr& t, const ttnn::Shape& shape, FanParams params);

void xavier_normal_init(ttml::autograd::TensorPtr& t, const ttnn::Shape& shape, FanParams params);

void kaiming_uniform_init(ttml::autograd::TensorPtr& t, const ttnn::Shape& shape, int fan_in);

void kaiming_normal_init(ttml::autograd::TensorPtr& t, const ttnn::Shape& shape, int fan_out);

}  // namespace ttml::init
