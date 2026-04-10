// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_logit.hpp"

#include "device/gather_logit_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor gather_logit(const ttnn::Tensor& logit, const ttnn::Tensor& target, uint32_t first_v, uint32_t last_v) {
    return ttnn::prim::ttml_gather_logit(logit, target, first_v, last_v);
}

}  // namespace ttml::metal
