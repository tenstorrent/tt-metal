// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/auto_context.hpp"

namespace ttml::ops::distributed {

autograd::TensorPtr intermesh_send(const autograd::TensorPtr& tensor, ttml::core::distributed::Rank rank);
autograd::TensorPtr intermesh_recv(const autograd::TensorPtr& tensor, ttml::core::distributed::Rank rank);

}  // namespace ttml::ops::distributed
