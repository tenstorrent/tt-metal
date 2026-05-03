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
#include <string_view>
#include <unordered_set>
#include <vector>

namespace tt::tt_metal {

// Pack (kernel_idx, core.x, core.y, arg_idx) into a single uint64_t for O(1) set lookup.
static uint64_t rt_binding_key(uint32_t kernel_idx, CoreCoord core, uint32_t arg_idx) {
    return ((uint64_t)(kernel_idx & 0xffu) << 48) | ((uint64_t)(core.x & 0xffu) << 40) |
           ((uint64_t)(core.y & 0xffu) << 32) | (uint64_t)arg_idx;
}

ResolvedBindings resolve_bindings(
    Program& program, const ProgramDescriptor& desc, const std::vector<Buffer*>& tensor_buffers) {
    ResolvedBindings result;

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

    // Track every registered (kernel, core, arg_idx) position and every buffer address.
    // Used below to detect unregistered positions that hold a buffer address — i.e.,
    // the push_back(buf->address()) mistake.
    std::unordered_set<uint64_t> registered_positions;
    std::unordered_set<uint32_t> registered_addresses;

    // Sentinel used to key common (non-per-core) arg positions in registered_positions.
    // Real device cores have small coordinates; 0xFF,0xFF is never a valid logical core.
    static constexpr CoreCoord kCommonArgSentinel{0xFF, 0xFF};

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
            registered_positions.insert(rt_binding_key(k, b.core, b.arg_idx));
            registered_addresses.insert(b.buffer->address());
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
            registered_positions.insert(rt_binding_key(k, kCommonArgSentinel, b.arg_idx));
            registered_addresses.insert(b.buffer->address());
        }
    }

    // Scan every per-core and common runtime arg.  If an arg's value matches a registered
    // buffer's address but no binding was declared at that position, the factory called
    // push_back(buf->address()) instead of push_back(buf).  On cache hits the fast path
    // would skip that arg and leave it stale.
    if (!registered_addresses.empty()) {
        for (uint32_t k = 0; k < static_cast<uint32_t>(desc.kernels.size()); ++k) {
            for (const auto& [core, args] : desc.kernels[k].runtime_args) {
                for (uint32_t i = 0; i < static_cast<uint32_t>(args.size()); ++i) {
                    if (registered_addresses.contains(args[i]) &&
                        !registered_positions.contains(rt_binding_key(k, core, i))) {
                        TT_FATAL(
                            false,
                            "Kernel {} core ({},{}) arg[{}]={:#x} matches a registered buffer "
                            "address but no BufferBinding was declared at this position. "
                            "Use push_back(buffer) instead of push_back(buffer->address()) so "
                            "the address is patched on cache hits.",
                            k,
                            core.x,
                            core.y,
                            i,
                            args[i]);
                    }
                }
            }
            const auto& common = desc.kernels[k].common_runtime_args;
            for (uint32_t i = 0; i < static_cast<uint32_t>(common.size()); ++i) {
                if (registered_addresses.contains(common[i]) &&
                    !registered_positions.contains(rt_binding_key(k, kCommonArgSentinel, i))) {
                    TT_FATAL(
                        false,
                        "Kernel {} common arg[{}]={:#x} matches a registered buffer address "
                        "but no CommonBufferBinding was declared at this position. "
                        "Use emplace_common_runtime_args() with Buffer* instead of uint32_t.",
                        k,
                        i,
                        common[i]);
                }
            }
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

    return result;
}

void apply_resolved_bindings(
    Program& program, const ResolvedBindings& bindings, const std::vector<Buffer*>& current_buffers) {
    for (const auto& b : bindings.rt_args) {
        // Re-derive the RuntimeArgsData reference on every call via GetRuntimeArgs /
        // GetCommonRuntimeArgs rather than storing a cross-call raw pointer.  This is
        // safe across first-enqueue (when the dispatch path retargets rt_args_data to
        // the command-sequence buffer) because we look up the live struct each time.
        RuntimeArgsData& data =
            b.is_common ? GetCommonRuntimeArgs(program, b.kernel_idx) : GetRuntimeArgs(program, b.kernel_idx, b.core);
        data[b.arg_idx] = current_buffers[b.tensor_buffer_idx]->address();
    }
    for (const auto& cb : bindings.cbs) {
        UpdateDynamicCircularBufferAddress(
            program, cb.cb_id, *current_buffers[cb.tensor_buffer_idx], cb.address_offset);
    }
}

}  // namespace tt::tt_metal
