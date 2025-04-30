// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ops/cross_entropy_fw/cross_entropy_fw.hpp"
#include "ops/rmsnorm_fw/rmsnorm_fw.hpp"

namespace ttml::metal {

constexpr auto rmsnorm_fw =
    ttnn::register_operation<"ttml::metal::rmsnorm_fw", ttml::metal::ops::rmsnorm_fw::RMSNormForwardOperation>();

constexpr auto cross_entropy_fw = ttnn::register_operation<
    "ttml::metal::cross_entropy_fw",
    ttml::metal::ops::cross_entropy_fw::CrossEntropyForwardOperation>();

}  // namespace ttml::metal
