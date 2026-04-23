// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <string_view>

namespace tt::tt_metal {

// Thread-local allocation context stack.
// Guards push a context string (e.g. op name + compile args) before dispatching;
// the allocator records whatever context is on top of the stack at allocation time.
//
// Suppression markers can be checked against the entire active stack so
// outer scopes (for example corruptible_allocation_scope) can suppress
// allocations from nested operation contexts as well.
void push_allocation_context(std::string ctx);
void pop_allocation_context();
const std::string& current_allocation_context();
bool allocation_context_contains(std::string_view ctx);

// RAII guard that pushes/pops a context string on the thread-local allocation context stack.
// While this guard is alive, any tracked allocation records the context for later reporting.
class AllocationContextGuard {
public:
    explicit AllocationContextGuard(std::string ctx) { push_allocation_context(std::move(ctx)); }
    ~AllocationContextGuard() { pop_allocation_context(); }
    AllocationContextGuard(const AllocationContextGuard&) = delete;
    AllocationContextGuard& operator=(const AllocationContextGuard&) = delete;
};

}  // namespace tt::tt_metal
