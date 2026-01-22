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

// rmsnorm_fw is now a free function declared in ops/rmsnorm_fw/rmsnorm_fw.hpp

// rmsnorm_bw is now a free function declared in ops/rmsnorm_bw/rmsnorm_bw.hpp

// layernorm_bw is now a free function declared in ops/layernorm_bw/layernorm_bw.hpp

// layernorm_fw is now a free function declared in ops/layernorm_fw/layernorm_fw.hpp

// cross_entropy_fw is now a free function declared in ops/cross_entropy_fw/cross_entropy_fw.hpp

// cross_entropy_bw is now a free function declared in ops/cross_entropy_bw/cross_entropy_bw.hpp

// softmax is now a free function declared in ops/softmax/softmax.hpp

// profiler_no_op is now a free function declared in ops/profiler_no_op/profiler_no_op.hpp

// silu_bw is now a free function declared in ops/silu_bw/silu_bw.hpp

// sdpa_fw is now a free function declared in ops/sdpa_fw/sdpa_fw.hpp

// sdpa_bw is now a free function declared in ops/sdpa_bw/sdpa_bw.hpp

// swiglu_fw is now a free function declared in ops/swiglu_fw/swiglu_fw.hpp

// sgd_fused is now a free function declared in optimizers/sgd_fused/sgd_fused.hpp

}  // namespace ttml::metal
