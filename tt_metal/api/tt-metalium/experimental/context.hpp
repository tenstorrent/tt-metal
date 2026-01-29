// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/dispatch_core_common.hpp>

namespace tt::tt_metal::experimental {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-private-field"

class Context {
private:
    int num_cqs_;
    int l1_small_size_;
    int trace_region_size_;
    int worker_l1_size_;
    DispatchCoreConfig dispatch_core_config_;

    class ContextImpl;
    std::unique_ptr<ContextImpl> impl_;

public:
    explicit Context(
        int num_cqs,
        int l1_small_size,
        int trace_region_size,
        int worker_l1_size,
        DispatchCoreConfig dispatch_core_config);

    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) noexcept;
    Context& operator=(Context&&) noexcept;

    int get_num_cqs() const;
    int get_l1_small_size() const;
    int get_trace_region_size() const;
    int get_worker_l1_size() const;
    DispatchCoreConfig get_dispatch_core_config() const;
};

#pragma clang diagnostic pop

}  // namespace tt::tt_metal::experimental
