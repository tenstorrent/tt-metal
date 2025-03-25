// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/rmsnorm_fw/rmsnorm_fw.hpp"

namespace ttml::metal {

constexpr auto rmsnorm_fw = ttnn::register_operation_with_auto_launch_op<
    "ttml::metal::rmsnorm_fw",
    ttml::metal::ops::rmsnorm_fw::RMSNormForwardOperation>();

}  // namespace ttml::metal
