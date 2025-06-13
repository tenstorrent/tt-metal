// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ops/cross_entropy_bw/cross_entropy_bw.hpp"
#include "ops/cross_entropy_fw/cross_entropy_fw.hpp"
#include "ops/profiler_no_op/profiler_no_op.hpp"
#include "ops/rmsnorm_fw/rmsnorm_fw.hpp"

namespace ttml::metal {

constexpr auto rmsnorm_fw =
    ttnn::register_operation<"ttml::metal::rmsnorm_fw", ttml::metal::ops::rmsnorm_fw::RMSNormForwardOperation>();

constexpr auto cross_entropy_fw = ttnn::register_operation<
    "ttml::metal::cross_entropy_fw",
    ttml::metal::ops::cross_entropy_fw::CrossEntropyForwardOperation>();

constexpr auto cross_entropy_bw = ttnn::register_operation<
    "ttml::metal::cross_entropy_bw",
    ttml::metal::ops::cross_entropy_bw::CrossEntropyBackwardOperation>();

constexpr auto profiler_no_op =
    ttnn::register_operation<"ttml::metal::profiler_no_op", ttml::metal::ops::profiler_no_op::ProfilerNoopOperation>();

}  // namespace ttml::metal
