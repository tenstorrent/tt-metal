// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>

namespace ttml::core {

template <typename OpenFunction, typename CloseFunction>
class Scoped {
    CloseFunction close_func_;

public:
    Scoped(OpenFunction&& open_func, CloseFunction&& close_func) : close_func_(std::move(close_func)) {
        open_func();
    }

    Scoped(const Scoped&) = delete;
    Scoped& operator=(const Scoped&) = delete;
    Scoped(Scoped&& other) = delete;
    Scoped& operator=(Scoped&&) = delete;

    ~Scoped() {
        close_func_();
    }
};

}  // namespace ttml::core
