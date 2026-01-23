// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <core/ttnn_all_includes.hpp>

namespace ttml::ttnn_fixed::distributed {

tt::tt_metal::Tensor all_gather(const tt::tt_metal::Tensor& tensor, int dim, std::optional<uint32_t> cluster_axis = std::nullopt);
tt::tt_metal::Tensor all_reduce(const tt::tt_metal::Tensor& tensor, std::optional<uint32_t> cluster_axis = std::nullopt);
tt::tt_metal::Tensor reduce_scatter(const tt::tt_metal::Tensor& tensor, int dim, std::optional<uint32_t> cluster_axis = std::nullopt);

}  // namespace ttml::ttnn_fixed::distributed
