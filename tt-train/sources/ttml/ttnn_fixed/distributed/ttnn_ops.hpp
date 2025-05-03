// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <core/ttnn_all_includes.hpp>

namespace ttml::ttnn_fixed::distributed {

tt::tt_metal::Tensor all_reduce(const tt::tt_metal::Tensor& tensor);
tt::tt_metal::Tensor scatter(const tt::tt_metal::Tensor& tensor, int dim);

}  // namespace ttml::ttnn_fixed::distributed
