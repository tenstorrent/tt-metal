// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "serialization/serializable.hpp"

namespace ttml::core::distributed {

ttnn::Tensor synchronize_tensor(const ttnn::Tensor& tensor);
void synchronize_parameters(const serialization::NamedParameters& parameters);

void send_tensor(const ttnn::Tensor& tensor, int dest, int tag = -1);
void recv_tensor(ttnn::Tensor& tensor, int source, int tag = -1);
void broadcast_tensor(ttnn::Tensor& tensor, int root);
void broadcast_tensor_to_group(ttnn::Tensor& tensor, int root, std::span<int> client_ranks);

// by default reduction is sum
// this ops expects that client ranks will call send_tensor
// and root rank will call will call reduce_tensor where it will use recv_tensor in the implementation
void reduce_tensor(ttnn::Tensor& tensor, std::span<int> client_ranks);

}  // namespace ttml::core::distributed
