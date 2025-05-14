// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>

#include "serialization/serializable.hpp"

namespace ttml::core::distributed {

using Rank = tt::tt_metal::distributed::multihost::Rank;
using Tag = tt::tt_metal::distributed::multihost::Tag;

ttnn::Tensor synchronize_tensor(const ttnn::Tensor& tensor);
void synchronize_parameters(const serialization::NamedParameters& parameters);

void send_tensor(const ttnn::Tensor& tensor, Rank dest, Tag tag = Tag{0});
void recv_tensor(ttnn::Tensor& tensor, Rank source, Tag tag = Tag{0});
void broadcast_tensor(ttnn::Tensor& tensor, Rank root);
void broadcast_tensor_to_group(ttnn::Tensor& tensor, Rank root, std::span<Rank> client_ranks);

// by default reduction is sum
// this ops expects that client ranks will call send_tensor
// and root rank will call reduce_tensor where it will use recv_tensor in the implementation
void reduce_tensor(ttnn::Tensor& tensor, std::span<Rank> client_ranks);

}  // namespace ttml::core::distributed
