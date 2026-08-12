// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * Fast cache-hit patching for descriptor-factory programs.
 *
 * ⚠️ TEMPORARY SHIM — DO NOT BUILD ON THIS, DO NOT ADD CALLERS. ⚠️
 *
 * A stop-gap that lets descriptor-based op factories re-patch Buffer addresses and the handful
 * of hash-excluded scalar runtime args on a program-cache hit — standing in for the legacy
 * override_runtime_arguments() path until Metal 2.0 lands. Metal 2.0 solves this at the
 * framework level with native Tensor/Buffer bindings, at which point THIS ENTIRE FILE AND ITS
 * IMPLEMENTATION ARE DELETED. Do not extend the API, do not depend on it outside the
 * mesh-device-operation adapter. New code almost certainly wants the Metal 2.0 binding instead.
 *
 * CODE REVIEWERS: if anyone builds on top of this, reject immediately — unless it's Diego.
 *
 * Usage:
 *   // In create_descriptor() — cache miss:
 *   KernelDescriptor::RTArgList args;
 *   args.push_back(in_buffer);    // Buffer* — binding auto-registered
 *   args.push_back(num_tiles);    // uint32_t
 *   kernel_desc.emplace_runtime_args(core, std::move(args));
 *
 *   // In the adapter (cache miss): store resolved bindings.  NOTE: this is the LEGACY address-inference
 *   // path, being migrated out.  An op that needs correct in-place / mixed-aliasing behavior should
 *   // instead define override_runtime_arguments() (re-derives all per-dispatch state itself, correct by
 *   // construction) — the adapter then bypasses resolve_bindings entirely for that op.
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
#include <limits>
#include <span>
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

// All resolved bindings for one cached program. Non-empty when the factory
// declared at least one buffer arg via KernelDescriptor::emplace_runtime_args(),
// (or none, when the descriptor declares no buffer bindings at all)
// (declarative MeshDescriptor path).
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
// The resolver is policy-free: it always resolves every binding the descriptor
// declares, both runtime-arg bindings and CB `.buffer` bindings.  Deciding
// whether the resulting `ResolvedBindings` is safe to fast-path on a given
// cache hit is the caller's job (see DescriptorMeshWorkloadAdapter::apply_descriptor),
// because the safety check depends on workload variant — specifically whether
// a slow-path rebuild is available to refresh raw (non-binding) runtime args.
//
// tensor_buffers is ordered inputs-first (see collect_tensor_buffers): the first
// num_input_buffers entries come from tensor_args, the rest from the output(s) and any
// workload buffers.  This split lets the resolver distinguish two kinds of aliasing:
//   - the SAME buffer appearing twice WITHIN the inputs (e.g. matmul(X, X)) is ambiguous —
//     a future call with distinct same-shape tensors would miscompute — so we bail to the
//     slow path.
//   - an OUTPUT buffer (from the output/workload region) that aliases an INPUT buffer (an
//     in-place op writing back into its input) is safe: every binding for that buffer resolves
//     to the one shared address, correct on every dispatch — so we keep the fast path.
// num_input_buffers defaults to SIZE_MAX, which treats every entry as an input (the original
// conservative behavior: bail on any duplicate).
//
// allow_inplace_output_tensor_alias (default false): when an op carries its own output INSIDE
// tensor_args (an optional output_tensor in the INPUT region) and it aliases an input, the output
// buffer appears in the input region.  That looks like the ambiguous matmul(X, X) duplicate, so by
// default we BAIL to the slow-path rebuild (correct for every op).  An op may set this true ONLY if
// it re-applies EVERY cache-hit-varying runtime arg itself (via get_dynamic_runtime_args + Buffer*
// bindings) so reusing the cached program for a differently-shaped/-allocated in-place call is
// correct.  binary_ng qualifies (its get_dynamic re-derives all per-core args).  unary/ternary/
// moreh_* do NOT — their get_dynamic is partial, so they must keep bailing (see #49573, #48928,
// SDXL in-place silu / MorehAdamW).  The op opts in via the adapter, keyed on a static trait.
//
// Call immediately after Program{desc} on cache miss; store in shared_variables.
ResolvedBindings resolve_bindings(
    Program& program,
    const ProgramDescriptor& desc,
    std::span<Buffer* const> tensor_buffers,
    size_t num_input_buffers = std::numeric_limits<size_t>::max(),
    bool allow_inplace_output_tensor_alias = false);

// Apply resolved bindings to the cached program on a cache hit.
// current_buffers must be the output of collect_tensor_buffers() for the
// current call's tensors — same enumeration order as at resolve time.
//
// Uses GetRuntimeArgs / GetCommonRuntimeArgs to obtain the live RuntimeArgsData
// reference on each call, so the write targets the correct storage (pre- or
// post-first-enqueue) without any cross-call pointer state.
void apply_resolved_bindings(
    Program& program, const ResolvedBindings& bindings, std::span<Buffer* const> current_buffers);

// ---------------------------------------------------------------------------
// Dynamic (non-Buffer) runtime args
// ---------------------------------------------------------------------------

// Declares one runtime-arg slot whose value is DYNAMIC: it is intentionally excluded from the
// program-cache hash (so two calls that differ only in this value still cache-hit) and therefore
// MUST be re-applied to the cached program on every dispatch.  This is the non-Buffer analog of
// BufferBinding: BufferBinding re-patches a buffer address on a cache hit; DynamicRuntimeArg
// re-patches an arbitrary scalar that a custom compute_program_hash deliberately omitted (e.g. an
// RNG seed, an [from,to) range, a semaphore L1 address).  The owning device operation produces the
// current values for each dispatch (see DeviceOperation::get_dynamic_runtime_args); this struct is
// just the destination (kernel/core/arg) plus the value to write.
struct DynamicRuntimeArg {
    uint32_t kernel_idx = 0;  // index into ProgramDescriptor::kernels
    CoreCoord core{};         // ignored when is_common == true
    uint32_t arg_idx = 0;     // position within that kernel/core's runtime args
    uint32_t value = 0;       // current value, derived from the live operation_attributes
    bool is_common = false;   // true => common (non-per-core) runtime args
};

// Write each DynamicRuntimeArg's value into the cached program's live runtime args.  Call on every
// cache hit (after apply_resolved_bindings) for ops that declare dynamic non-Buffer runtime args.
// Uses GetRuntimeArgs / GetCommonRuntimeArgs for the same pre/post-first-enqueue correctness as
// apply_resolved_bindings.
void apply_dynamic_runtime_args(Program& program, std::span<const DynamicRuntimeArg> dynamic_args);

// ---------------------------------------------------------------------------
// Fast-path parity check (debug / CI regression net)
// ---------------------------------------------------------------------------

// Universal, op-agnostic correctness invariant for the cache-hit fast path.  After the fast path
// has patched the cached program (apply_resolved_bindings + apply_dynamic_runtime_args), the caller
// rebuilds the descriptor for the CURRENT tensors into a scratch Program and passes both here.  This
// asserts the fast path reproduced EXACTLY what a full rebuild would: every per-core and common
// runtime arg (enumerated from `desc`) and every circular-buffer base address must match.
//
// A mismatch means resolved_bindings + get_dynamic_runtime_args is INCOMPLETE for this op — a stale
// runtime arg or (for sharded ops) a stale CB address survives on a differently-aliased or
// differently-allocated cache hit (the SDXL in-place silu / MorehAdamW failure mode).  It fires a
// TT_FATAL naming the op, kernel, core, arg, and both values, turning silent PCC garbage into a loud,
// pinpointed failure.  The rebuild is the oracle, so no per-op assertions have to be hand-maintained.
//
// Intended to be called only under -DTT_DESCRIPTOR_PATCHING_PARITY_CHECK (CI/debug); it is a no-cost
// helper otherwise since the caller guards the call site.  The check makes the in-place fast path a
// verified framework invariant rather than a per-op trust assertion.
void assert_fastpath_parity(
    const Program& fast, const Program& rebuilt, const ProgramDescriptor& desc, std::string_view op_name);

}  // namespace tt::tt_metal
