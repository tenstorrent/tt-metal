// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "serialization/serializable.hpp"

namespace ttml::core::distributed {

ttnn::Tensor synchronize_tensor(const ttnn::Tensor& tensor);
void synchronize_parameters(const serialization::NamedParameters& parameters);

}  // namespace ttml::core::distributed
