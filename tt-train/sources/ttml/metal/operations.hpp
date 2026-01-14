// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ops/cross_entropy_bw/cross_entropy_bw.hpp"
#include "ops/cross_entropy_fw/cross_entropy_fw.hpp"
#include "ops/layernorm_bw/layernorm_bw.hpp"
#include "ops/layernorm_fw/layernorm_fw.hpp"
#include "ops/profiler_no_op/profiler_no_op.hpp"
#include "ops/rmsnorm_bw/rmsnorm_bw.hpp"
#include "ops/rmsnorm_fw/rmsnorm_fw.hpp"
#include "ops/sdpa_bw/sdpa_bw.hpp"
#include "ops/sdpa_fw/sdpa_fw.hpp"
#include "ops/silu_bw/silu_bw.hpp"
#include "ops/softmax/softmax.hpp"
#include "ops/swiglu_fw/swiglu_fw.hpp"
#include "optimizers/sgd_fused/sgd_fused.hpp"

namespace ttml::metal {

constexpr auto rmsnorm_fw =
    ttnn::register_operation<"ttml::metal::rmsnorm_fw", ttml::metal::ops::rmsnorm_fw::RMSNormForwardOperation>();

constexpr auto rmsnorm_bw =
    ttnn::register_operation<"ttml::metal::rmsnorm_bw", ttml::metal::ops::rmsnorm_bw::RMSNormBackwardOperation>();

constexpr auto layernorm_bw =
    ttnn::register_operation<"ttml::metal::layernorm_bw", ttml::metal::ops::layernorm_bw::LayerNormBackwardOperation>();

constexpr auto layernorm_fw =
    ttnn::register_operation<"ttml::metal::layernorm_fw", ttml::metal::ops::layernorm_fw::LayerNormForwardOperation>();

constexpr auto cross_entropy_fw = ttnn::register_operation<
    "ttml::metal::cross_entropy_fw",
    ttml::metal::ops::cross_entropy_fw::CrossEntropyForwardOperation>();

constexpr auto cross_entropy_bw = ttnn::register_operation<
    "ttml::metal::cross_entropy_bw",
    ttml::metal::ops::cross_entropy_bw::CrossEntropyBackwardOperation>();

constexpr auto softmax =
    ttnn::register_operation<"ttml::metal::softmax", ttml::metal::ops::softmax::SoftmaxOperation>();

constexpr auto profiler_no_op =
    ttnn::register_operation<"ttml::metal::profiler_no_op", ttml::metal::ops::profiler_no_op::ProfilerNoopOperation>();

constexpr auto silu_bw =
    ttnn::register_operation<"ttml::metal::silu_bw", ttml::metal::ops::silu_bw::SiLUBackwardOperation>();

constexpr auto sdpa_fw =
    ttnn::register_operation<"ttml::metal::sdpa_fw", ttml::metal::ops::sdpa_fw::SDPAForwardOperation>();

constexpr auto sdpa_bw =
    ttnn::register_operation<"ttml::metal::sdpa_bw", ttml::metal::ops::sdpa_bw::SDPABackwardOperation>();

constexpr auto swiglu_fw =
    ttnn::register_operation<"ttml::metal::swiglu_fw", ttml::metal::ops::swiglu_fw::SwiGLUForwardOperation>();

constexpr auto sgd_fused =
    ttnn::register_operation<"ttml::metal::sgd_fused", ttml::metal::optimizers::sgd_fused::SGDFusedOptimizer>();

}  // namespace ttml::metal
