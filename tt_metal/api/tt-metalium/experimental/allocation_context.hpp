// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace tt::tt_metal {

// Experimental thread-local allocation context stack.
// Guards push a context string (e.g. op name + compile args) before dispatching;
// the allocator records whatever context is on top of the stack at allocation time.
//
// Process-start configuration, cached during RunTimeOptions initialization.
bool trace_allocation_tracking_enabled();
bool trace_allocation_diagnostics_enabled();
bool trace_allocation_skip_program_cache_enabled();

// Allocation contexts with this prefix identify program-cache-owned buffers, which
// TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE excludes from unsafe-allocation accounting.
inline constexpr std::string_view kProgramCacheAllocationContextPrefix = "program_cache:";

void push_allocation_context(std::string_view ctx);
void pop_allocation_context();
const std::string& current_allocation_context();
bool allocation_context_contains(std::string_view ctx);

// RAII guard that pushes/pops a context string on the thread-local allocation context stack.
// While this guard is alive, any tracked allocation records the context for later reporting.
class AllocationContextGuard {
public:
    explicit AllocationContextGuard(std::string_view ctx) : active_(trace_allocation_tracking_enabled()) {
        if (active_) {
            push_allocation_context(ctx);
        }
    }
    ~AllocationContextGuard() {
        if (active_) {
            pop_allocation_context();
        }
    }
    AllocationContextGuard(const AllocationContextGuard&) = delete;
    AllocationContextGuard& operator=(const AllocationContextGuard&) = delete;

private:
    bool active_;
};

inline std::optional<AllocationContextGuard> make_allocation_context_guard(std::string_view ctx) {
    if (!trace_allocation_tracking_enabled()) {
        return std::nullopt;
    }
    return std::optional<AllocationContextGuard>{std::in_place, ctx};
}

}  // namespace tt::tt_metal
