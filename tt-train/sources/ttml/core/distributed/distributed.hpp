// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "serialization/serializable.hpp"

namespace ttml::core::distributed {

using Rank = tt::tt_metal::distributed::multihost::Rank;
using Tag = tt::tt_metal::distributed::multihost::Tag;

ttnn::Tensor synchronize_tensor(const ttnn::Tensor& tensor);
void synchronize_parameters(const serialization::NamedParameters& parameters);

void send_tensor(const autograd::DistributedContext& ctx, const ttnn::Tensor& tensor, Rank dest, Tag tag = Tag{0});
void recv_tensor(const autograd::DistributedContext& ctx, ttnn::Tensor& tensor, Rank source, Tag tag = Tag{0});
void broadcast_tensor(const autograd::DistributedContext& ctx, ttnn::Tensor& tensor, Rank root);

}  // namespace ttml::core::distributed
