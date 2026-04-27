// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * Fast cache-hit patching for descriptor-factory programs.
 *
 * This is a temporary shim. Metal 2.0 solves the same problem at the framework level with
 * native Tensor/Buffer bindings. This shim should be removed once op factories migrate to
 * the Metal 2.0 API.
 *
 * Usage:
 *   // In create_descriptor() — cache miss:
 *   KernelDescriptor::RTArgList args;
 *   args.push_back(in_buffer);    // Buffer* — binding auto-registered
 *   args.push_back(num_tiles);    // uint32_t
 *   kernel_desc.emplace_runtime_args(core, std::move(args));
 *
 *   // In the adapter (cache miss): store resolved bindings
 *   auto tensor_buffers = collect_tensor_buffers(tensor_args, tensor_return_value);
 *   auto resolved = tt::tt_metal::resolve_bindings(program, desc, tensor_buffers);
 *
 *   // In the adapter (cache hit): patch only the changed addresses
 *   auto current = collect_tensor_buffers(tensor_args, tensor_return_value);
 *   tt::tt_metal::apply_resolved_bindings(program, resolved, current);
 */

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include <cstdint>
#include <vector>

namespace tt::tt_metal {

class Program;
struct ProgramDescriptor;

// ---------------------------------------------------------------------------
// Resolved binding types — index-based, no raw Buffer* or RuntimeArgsData*
// ---------------------------------------------------------------------------

// Maps one buffer-typed runtime arg to its stable position and tensor slot.
// kernel_idx / core / arg_idx identify where to write; tensor_buffer_idx is
// the index into the collect_tensor_buffers() result from the current call.
// is_common == true means the arg is in common (non-per-core) runtime args.
struct ResolvedRtArgBinding {
    uint32_t kernel_idx = 0;
    CoreCoord core{};  // unused when is_common == true
    uint32_t arg_idx = 0;
    uint32_t tensor_buffer_idx = 0;
    bool is_common = false;
};

// Maps one dynamic CB to its CBHandle and tensor slot.
// cb_id is CBHandle (uintptr_t — 64-bit, not an index).
struct ResolvedCbBinding {
    uintptr_t cb_id = 0;
    uint32_t tensor_buffer_idx = 0;
    uint32_t address_offset = 0;
};

// All resolved bindings for one cached program. Non-empty only when the factory
// declared at least one buffer arg via KernelDescriptor::emplace_runtime_args().
struct ResolvedBindings {
    std::vector<ResolvedRtArgBinding> rt_args;
    std::vector<ResolvedCbBinding> cbs;
    bool empty() const noexcept { return rt_args.empty() && cbs.empty(); }
};

// ---------------------------------------------------------------------------
// resolve / apply
// ---------------------------------------------------------------------------

// Walk desc.kernels[k].buffer_bindings and desc.cbs[i].buffer, resolve each
// to a stable (kernel_idx, core, arg_idx) index tuple, and return the result.
//
// tensor_buffers is an ordered enumeration of all Buffer* reachable from the
// current call's tensor_args and tensor_return_value (built by the adapter via
// collect_tensor_buffers). Every binding Buffer* must appear in tensor_buffers;
// TT_FATAL fires if a factory used a non-tensor buffer in emplace_runtime_args.
//
// Call immediately after Program{desc} on cache miss; store in shared_variables.
ResolvedBindings resolve_bindings(
    Program& program, const ProgramDescriptor& desc, const std::vector<Buffer*>& tensor_buffers);

// Apply resolved bindings to the cached program on a cache hit.
// current_buffers must be the output of collect_tensor_buffers() for the
// current call's tensors — same enumeration order as at resolve time.
//
// Uses GetRuntimeArgs / GetCommonRuntimeArgs to obtain the live RuntimeArgsData
// reference on each call, so the write targets the correct storage (pre- or
// post-first-enqueue) without any cross-call pointer state.
void apply_resolved_bindings(
    Program& program, const ResolvedBindings& bindings, const std::vector<Buffer*>& current_buffers);

}  // namespace tt::tt_metal
