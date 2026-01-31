// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <mutex>
#include <tt-metalium/experimental/context.hpp>
#include <tt-metalium/experimental/runtime.hpp>
#include <tt-metalium/dispatch_core_common.hpp>

namespace tt::tt_metal::experimental {

class Context::ContextImpl {
public:
    // Future implementation details go here
};

Context::Context(
    int num_cqs,
    int l1_small_size,
    int trace_region_size,
    int worker_l1_size,
    DispatchCoreConfig dispatch_core_config) :
    num_cqs_(num_cqs),
    l1_small_size_(l1_small_size),
    trace_region_size_(trace_region_size),
    worker_l1_size_(worker_l1_size),
    dispatch_core_config_(dispatch_core_config),
    impl_(std::make_unique<ContextImpl>()) {}

Context::~Context() = default;

Context::Context(Context&&) noexcept = default;

Context& Context::operator=(Context&&) noexcept = default;

}  // namespace tt::tt_metal::experimental
