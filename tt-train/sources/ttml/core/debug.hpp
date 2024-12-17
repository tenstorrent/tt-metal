// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ttml::core::debug {

struct Debug {
    static constexpr bool enable_backward_performance_measurement() {
        return false;
    }

    static constexpr bool enable_print_tensor_stats() {
        return false;
    };
};

}  // namespace ttml::core::debug
