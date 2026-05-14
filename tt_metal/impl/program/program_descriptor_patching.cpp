// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Implementation of the fast cache-hit patching helpers declared in
// tt-metalium/experimental/program_descriptor_patching.hpp.
//
// This is a temporary shim. Metal 2.0 solves the same problem at the framework
// level with native Tensor/Buffer bindings. Remove once op factories migrate.

#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/assert.hpp>

#include <algorithm>
#include <cstdint>
#include <span>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace tt::tt_metal {

ResolvedBindings resolve_bindings(
    Program& program, const ProgramDescriptor& desc, std::span<Buffer* const> tensor_buffers) {
    ResolvedBindings result;

    // If the same Buffer* appears in tensor_buffers more than once (e.g. matmul(X, X),
    // or an output that aliases an input), every binding for that buffer would map to
    // the first occurrence via std::find below.  At cache hit, all of those bindings
    // would be patched with current_buffers[first_slot].address(), so the second
    // tensor's address would never get written.  The result is silent miscompute when
    // a future call uses distinct tensors at the same shape/dtype.
    //
    // We can't disambiguate which binding corresponds to which slot from Buffer* alone,
    // so we fall back to the slow path (rebuild the descriptor) when this happens.
    {
        std::unordered_set<Buffer*> seen;
        seen.reserve(tensor_buffers.size());
        for (Buffer* buf : tensor_buffers) {
            if (buf && !seen.insert(buf).second) {
                return ResolvedBindings{};
            }
        }
    }

    // Map each Buffer* to its index in tensor_buffers.  Every binding Buffer* must be
    // present; TT_FATAL fires if a factory used a non-tensor buffer in emplace_runtime_args.
    auto find_idx = [&](Buffer* buf, std::string_view context) -> uint32_t {
        auto it = std::find(tensor_buffers.begin(), tensor_buffers.end(), buf);
        TT_FATAL(
            it != tensor_buffers.end(),
            "Buffer* in {} not found in tensor_args/tensor_return_value enumeration. "
            "All buffer bindings must come directly from input/output tensors.",
            context);
        return static_cast<uint32_t>(it - tensor_buffers.begin());
    };

    for (uint32_t k = 0; k < static_cast<uint32_t>(desc.kernels.size()); ++k) {
        for (const auto& b : desc.kernels[k].buffer_bindings) {
            auto& data = GetRuntimeArgs(program, k, b.core);

            // Validate that the registered position actually contains this buffer's address.
            // Fires when buffer_bindings.push_back() was called with the wrong arg_idx, or
            // when the runtime arg was written independently with a different value.
            TT_FATAL(
                b.arg_idx < data.size() && data[b.arg_idx] == b.buffer->address(),
                "BufferBinding for kernel {} at core ({},{}) arg[{}]: stored value {:#x} "
                "does not match buffer->address() {:#x}. Wrong arg_idx in buffer_bindings, "
                "or arg was written without using emplace_runtime_args().",
                k,
                b.core.x,
                b.core.y,
                b.arg_idx,
                b.arg_idx < data.size() ? data[b.arg_idx] : 0u,
                b.buffer->address());

            result.rt_args.push_back({k, b.core, b.arg_idx, find_idx(b.buffer, "buffer_bindings"), false});
        }

        for (const auto& b : desc.kernels[k].common_buffer_bindings) {
            auto& data = GetCommonRuntimeArgs(program, k);

            TT_FATAL(
                b.arg_idx < data.size() && data[b.arg_idx] == b.buffer->address(),
                "CommonBufferBinding for kernel {} arg[{}]: stored value {:#x} "
                "does not match buffer->address() {:#x}. Wrong arg_idx in common_buffer_bindings, "
                "or arg was written without using emplace_common_runtime_args().",
                k,
                b.arg_idx,
                b.arg_idx < data.size() ? data[b.arg_idx] : 0u,
                b.buffer->address());

            result.rt_args.push_back({k, {}, b.arg_idx, find_idx(b.buffer, "common_buffer_bindings"), true});
        }
    }

    // Only resolve CB bindings when the factory actually opted into the fast path
    // by declaring at least one runtime-arg buffer binding, whether via
    // emplace_runtime_args() or emplace_common_runtime_args()
    // (i.e. when !result.rt_args.empty()).
    // Without this guard, sharded operations that use the old API (passing
    // buffer->address() as uint32_t) would have non-empty resolved_bindings
    // due to CB entries alone, causing the adapter to take the fast path and
    // skip the full runtime-arg rebuild on cache hits.
    if (!result.rt_args.empty()) {
        auto program_cbs = program.circular_buffers();
        for (uint32_t ci = 0; ci < static_cast<uint32_t>(desc.cbs.size()); ++ci) {
            if (desc.cbs[ci].buffer) {
                result.cbs.push_back(
                    {program_cbs[ci]->id(), find_idx(desc.cbs[ci].buffer, "cbs"), desc.cbs[ci].address_offset});
            }
        }
    }

    // Sort rt_args by (is_common, kernel_idx, core, arg_idx) so that bindings sharing
    // the same (kernel, core) RuntimeArgsData are contiguous.  apply_resolved_bindings
    // amortises the GetRuntimeArgs lookup across each group instead of doing one
    // lookup per binding — significant win on multi-core ops with several Buffer*
    // slots per core.
    std::sort(result.rt_args.begin(), result.rt_args.end(), [](const auto& a, const auto& b) {
        if (a.is_common != b.is_common) {
            return !a.is_common;  // per-core first, then common
        }
        if (a.kernel_idx != b.kernel_idx) {
            return a.kernel_idx < b.kernel_idx;
        }
        if (!a.is_common) {
            if (a.core.x != b.core.x) {
                return a.core.x < b.core.x;
            }
            if (a.core.y != b.core.y) {
                return a.core.y < b.core.y;
            }
        }
        return a.arg_idx < b.arg_idx;
    });

    return result;
}

void apply_resolved_bindings(
    Program& program, const ResolvedBindings& bindings, std::span<Buffer* const> current_buffers) {
    // bindings.rt_args is sorted by (is_common, kernel_idx, core, arg_idx) at resolve
    // time, so consecutive entries share the same RuntimeArgsData reference whenever
    // they target the same (is_common, kernel_idx, core).  Cache the live reference
    // across that run instead of re-deriving it via GetRuntimeArgs per binding.
    //
    // The reference is re-derived on every apply call (not stored across calls) so
    // first-enqueue retargeting of rt_args_data to the command-sequence buffer is
    // observed correctly.
    RuntimeArgsData* current_data = nullptr;
    uint32_t prev_kernel_idx = 0;
    CoreCoord prev_core{};
    bool prev_is_common = false;
    bool first = true;

    for (const auto& b : bindings.rt_args) {
        const bool group_changed = first || b.is_common != prev_is_common || b.kernel_idx != prev_kernel_idx ||
                                   (!b.is_common && b.core != prev_core);
        if (group_changed) {
            current_data = b.is_common ? &GetCommonRuntimeArgs(program, b.kernel_idx)
                                       : &GetRuntimeArgs(program, b.kernel_idx, b.core);
            prev_is_common = b.is_common;
            prev_kernel_idx = b.kernel_idx;
            prev_core = b.core;
            first = false;
        }
        (*current_data)[b.arg_idx] = current_buffers[b.tensor_buffer_idx]->address();
    }
    for (const auto& cb : bindings.cbs) {
        UpdateDynamicCircularBufferAddress(
            program, cb.cb_id, *current_buffers[cb.tensor_buffer_idx], cb.address_offset);
    }
}

}  // namespace tt::tt_metal
