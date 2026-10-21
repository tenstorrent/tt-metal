// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>

#include "autograd/tensor.hpp"

namespace ttml::ops {

// use_per_device_seed draws a different mask on every device. Devices that hold the same values must drop the same
// elements, so by default it is read off the tensor's layout: per device only when no two devices hold the same values.
autograd::TensorPtr dropout(
    const autograd::TensorPtr& tensor, float probability, std::optional<bool> use_per_device_seed = std::nullopt);

}  // namespace ttml::ops
