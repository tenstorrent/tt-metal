// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

<<<<<<< HEAD
#pragma once

#include "ops/cross_entropy_fw/cross_entropy_fw.hpp"
=======
#include "ops/cross_entropy_bw/cross_entropy_bw.hpp"
    >>>>>>> 7d35f54f99 (kernels implementation added)
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

}  // namespace ttml::metal
