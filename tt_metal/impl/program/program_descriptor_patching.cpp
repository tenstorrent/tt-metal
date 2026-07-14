// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Implementation of the fast cache-hit patching helpers declared in
// tt-metalium/experimental/program_descriptor_patching.hpp.
//
// ⚠️ TEMPORARY SHIM — DO NOT BUILD ON THIS, DO NOT ADD CALLERS. ⚠️
//
// These helpers (resolve_bindings / apply_resolved_bindings / apply_dynamic_runtime_args)
// are a stop-gap that lets descriptor-based op factories re-patch Buffer addresses and the
// handful of hash-excluded scalar runtime args on a program-cache hit — standing in for the
// legacy override_runtime_arguments() path until Metal 2.0 lands.
//
// Metal 2.0 solves this at the framework level with native Tensor/Buffer bindings, at which
// point THIS ENTIRE FILE AND ITS HEADER ARE DELETED. Do not extend the API, do not depend on
// it outside the mesh-device-operation adapter. New code almost certainly wants the Metal 2.0
// binding instead.
//
// CODE REVIEWERS: if anyone builds on top of this, reject immediately — unless it's Diego.

#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <tt_stl/assert.hpp>

#include <algorithm>
#include <cstdint>
#include <span>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace tt::tt_metal {

ResolvedBindings resolve_bindings(
    Program& program,
    const ProgramDescriptor& desc,
    std::span<Buffer* const> tensor_buffers,
    size_t num_input_buffers,
    bool allow_inplace_output_tensor_alias) {
    ResolvedBindings result;

    // If the same Buffer* appears more than once, every binding for that buffer maps to the
    // first occurrence via std::find below; at cache hit all of those bindings get patched with
    // current_buffers[first_slot].address().  Whether that's safe depends on WHY it aliases:
    //
    //   - Duplicate WITHIN the inputs (e.g. matmul(X, X)): the two slots are distinct logical
    //     tensors that merely coincide on this call.  A future same-shape call with distinct
    //     tensors would silently miscompute (the second tensor's address is never written).
    //     We can't disambiguate from Buffer* alone, so bail to the slow path.
    //
    //   - An OUTPUT (or workload) buffer that aliases an INPUT: an in-place op writing back into
    //     its input.  Both slots are the SAME buffer by construction, on every dispatch, so
    //     mapping them to the one shared address is always correct.  Keep the fast path.
    //
    // tensor_buffers is ordered inputs-first; the first num_input_buffers entries are inputs.
    {
        // Buffers in the output/workload region.  When the op opts in via
        // allow_inplace_output_tensor_alias, an input-region entry that also appears here is the op's
        // own output carried inside tensor_args (e.g. an in-place op's optional output_tensor aliasing
        // an input) — a same-buffer-by-construction alias, NOT the ambiguous matmul(X, X) case — so it
        // must not bail.  This is UNSAFE unless the op re-applies every cache-hit-varying runtime arg
        // itself: otherwise the shared cached program is reused for a differently-shaped/-allocated
        // in-place call with stale args (SDXL in-place silu, MorehAdamW → PCC garbage; see #48928 /
        // #49573).  So it is OPT-IN: default false ⇒ such a duplicate bails to the safe slow-path
        // rebuild.  binary_ng opts in (#48928: in-place residual add; its get_dynamic re-derives all
        // per-core args).  When false, output_region stays empty and every in-place duplicate bails.
        std::unordered_set<Buffer*> output_region;
        if (allow_inplace_output_tensor_alias) {
            for (size_t i = num_input_buffers; i < tensor_buffers.size(); ++i) {
                if (tensor_buffers[i]) {
                    output_region.insert(tensor_buffers[i]);
                }
            }
        }
        std::unordered_set<Buffer*> input_buffers;   // buffers seen in the input region
        std::unordered_set<Buffer*> output_buffers;  // buffers seen in the output/workload region
        std::unordered_set<Buffer*> inplace_alias_used;  // output buffers already granted one input-region skip
        for (size_t i = 0; i < tensor_buffers.size(); ++i) {
            Buffer* buf = tensor_buffers[i];
            if (!buf) {
                continue;
            }
            const bool is_input = i < num_input_buffers;
            // An output/workload buffer that aliases an input is the safe in-place case — skip it.
            if (!is_input && input_buffers.contains(buf)) {
                continue;
            }
            // An output buffer carried in the input region (the op's output_tensor) is an in-place
            // alias, not a distinct input — skip it ONCE (opt-in only; output_region is empty
            // otherwise).  A second input-region occurrence means the buffer genuinely repeats across
            // input positions (e.g. op(X, X, out=X)); fall through so it still bails, else a later
            // same-shape op(X, Y, out=X) hit would patch input-b to X.
            if (is_input && output_region.contains(buf) && inplace_alias_used.insert(buf).second) {
                continue;
            }
            // Otherwise a repeat is ambiguous (matmul(X, X), or a repeated output) — bail to slow path.
            auto& seen = is_input ? input_buffers : output_buffers;
            if (!seen.insert(buf).second) {
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

    // Resolve every `.buffer = ...` CB binding whose buffer comes from
    // tensor_args / tensor_return_value, so the cache-hit fast path can patch
    // it.  CB buffers that come from elsewhere (e.g. a GlobalCircularBuffer
    // referenced by `operation_attributes`, or any other workload-scoped
    // resource the factory injects directly into a CBDescriptor) are SKIPPED
    // here rather than fatal:
    //
    //   - Such buffers have stable addresses across dispatches by design —
    //     the caller owns the resource and keeps it alive for the cache
    //     entry's lifetime.  Cache-hit patching would be a no-op.
    //   - Fatalling would make `emplace_runtime_args(buffer)` and
    //     `cbs[i].buffer = buffer` semantically asymmetric for the same buffer
    //     and break legitimate factories (e.g. `dram_prefetcher`'s reader CB
    //     pegged to a GlobalCircularBuffer's backing buffer).
    //
    // Runtime-arg buffer bindings stay strict (find_idx) above — they must
    // map to a tensor_args/return slot because raw rt-args ARE the only
    // mechanism by which input/output addresses can change between dispatches.
    //
    // The CB-resolution gate (whether to use the result on cache hit) lives
    // in DescriptorMeshWorkloadAdapter::apply_descriptor and is variant-aware:
    //   - ProgramDescriptor variant: fast-path only when rt-arg bindings are present;
    //     otherwise rebuild the descriptor.
    //   - WorkloadDescriptor variant: always fast-path; no rebuild fallback.
    {
        auto program_cbs = program.circular_buffers();
        for (uint32_t ci = 0; ci < static_cast<uint32_t>(desc.cbs.size()); ++ci) {
            const auto& cb_desc = desc.cbs[ci];
            TT_FATAL(
                !(cb_desc.buffer && cb_desc.tensor),
                "CBDescriptor cannot specify both buffer and tensor as the globally-allocated backing storage");

            Buffer* cb_buffer = cb_desc.buffer;
            if (!cb_buffer && cb_desc.tensor) {
                cb_buffer = cb_desc.tensor->mesh_buffer().get_reference_buffer();
            }
            if (cb_buffer) {
                auto it = std::find(tensor_buffers.begin(), tensor_buffers.end(), cb_buffer);
                if (it != tensor_buffers.end()) {
                    result.cbs.push_back(
                        {program_cbs[ci]->id(), static_cast<uint32_t>(it - tensor_buffers.begin()), cb_desc.address_offset});
                }
                // else: stable, non-tensor buffer; pegged at create time, no patching needed.
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

void apply_dynamic_runtime_args(Program& program, std::span<const DynamicRuntimeArg> dynamic_args) {
    // dynamic_args are not sorted (unlike resolved rt_args): the per-op get_dynamic_runtime_args()
    // typically emits only a handful of slots, so re-deriving the RuntimeArgsData reference per
    // entry is cheap and avoids imposing an ordering requirement on op authors.  The reference is
    // taken fresh on each call so first-enqueue retargeting of rt_args_data is observed correctly.
    for (const auto& d : dynamic_args) {
        auto& data =
            d.is_common ? GetCommonRuntimeArgs(program, d.kernel_idx) : GetRuntimeArgs(program, d.kernel_idx, d.core);
        TT_FATAL(
            d.arg_idx < data.size(),
            "DynamicRuntimeArg for kernel {} {}arg[{}] is out of range (runtime args size {}). "
            "The (kernel_idx, core, arg_idx) declared by get_dynamic_runtime_args() must match the "
            "slot populated in create_descriptor().",
            d.kernel_idx,
            d.is_common ? "common " : "",
            d.arg_idx,
            data.size());
        data[d.arg_idx] = d.value;
    }
}

}  // namespace tt::tt_metal
