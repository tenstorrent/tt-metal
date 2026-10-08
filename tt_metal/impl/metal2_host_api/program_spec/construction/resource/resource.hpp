// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/metal2_host_api/llk_metadata.hpp"
#include "impl/metal2_host_api/program_spec/collection/collect_metadata.hpp"
#include "impl/metal2_host_api/program_spec/construction/processor_assignment/processor_assignment.hpp"
#include "impl/metal2_host_api/semaphore_scope.hpp"
#include "impl/program/program_impl.hpp"
#include "llrt/hal.hpp"

namespace tt::tt_metal::experimental {

// Break down of program & kernel construction by resource type.
// For each resource, there are always two sides:
// - Register with the Program
// - Register with a Kernel

// ----------------------------------------------------------------------------
// Tensor parameters (tensor.cpp)
// ----------------------------------------------------------------------------

// Per-TensorParameter resolved layout.
//
// cta_payload: positional CTA words appended to the kernel's compile-time args for
// each binding of this TensorParameter. Mirrors what TensorAccessorArgs::append_to
// would produce on the legacy path, but is built from the TensorSpec + MeshDevice
// since no Buffer exists at spec-build time.
//
// extra_crta_words: additional CRTA words (beyond the always-present base address
// slot) that this binding occupies, used by the device-side accessor to read
// runtime-resolved fields. Non-zero when the TensorParameter opts into a dynamic
// field that lives in CRTAs: either sharded + dynamic_tensor_shape (which puts
// `rank` shape words in CRTAs), or interleaved row-major + dynamic_tensor_shape (one
// page-size word). The two are mutually exclusive per binding -- see runtime_field_is_page_size.
struct ResolvedTensorParameter {
    std::vector<uint32_t> cta_payload;

    // How many CRTA words (beyond the base address) does this binding consume?
    // This is only used if TensorParameter relaxations have been requested.
    uint32_t extra_crta_words = 0;

    // What info the runtime field CRTA words actually contain depends on the relaxation.
    // Currently, there are only two mutually exclusive possibilities (though more may be added):
    //  1. The interleaved row-major page-size (one CRTA only)
    //  2. The sharded dynamic_tensor_shape shape (one CRTA per tensor dim)
    // For now, since there are only two mutually exclusive possibilities, it's sufficient to
    // distinguish them with a boolean.
    bool runtime_field_is_page_size = false;

    // Compile-time LLK metadata derived from the TensorParameter's spec, baked onto the binding token.
    // Always present: a tensor has a dtype and a tile.
    LLKMetadata llk_metadata;
};

// Register tensor parameters with the Program
std::unordered_map<TensorParamName, ResolvedTensorParameter> RegisterTensorParameters(
    const ProgramSpec& spec, const distributed::MeshDevice& mesh_device, detail::ProgramImpl& program_impl);

// Register tensor parameters with a Kernel
// Per-kernel resolved tensor binding data:
//  - All the kernel's TensorBindingHandle (type is defined in kernel.hpp)
//  - The positional CTAs to append to the kernel's (unnamed) CTAs
//  - The full CRTA buffer layout (named CRTAs + binding section + vararg-section start),
//    precomputed here so consumers (headergen, runtime) don't have to re-derive section
//    boundaries by walking handles. See KernelCrtaLayout in jit_build_settings.hpp.
struct TensorBindingsForKernel {
    std::vector<TensorBindingHandle> handles;
    // Binding-only CTA payload; appended after the user CTA-vararg positional prefix.
    std::vector<uint32_t> cta_words;
    KernelCrtaLayout crta_layout;
};

TensorBindingsForKernel ResolveTensorBindingsForKernel(
    const KernelSpec& kernel,
    const std::unordered_map<TensorParamName, ResolvedTensorParameter>& resolved_tensor_parameters,
    size_t base_named_crta_count,
    uint32_t base_cta_offset);

// ----------------------------------------------------------------------------
// Dataflow buffers (dfb.cpp)
// ----------------------------------------------------------------------------

// DFB name -> program-wide DFB ID map (host-side identity: aliasing, borrowed bindings)
using DFBNameToIdMap = std::unordered_map<DFBSpecName, uint32_t>;
// DFB name -> device slot map. The slot is what a kernel sees (the dfb::<name> accessor value) and
// what indexes the per-core config table, so it is what device-facing lowering must use.
using DFBNameToSlotMap = std::unordered_map<DFBSpecName, uint32_t>;

// Per-DFB handles produced by RegisterDataflowBuffers (relay_pipe_id is filled by RecordRelayPipeIds).
struct DataflowBufferHandles {
    DFBNameToIdMap id;
    DFBNameToSlotMap slot;
    std::unordered_map<DFBSpecName, bool> is_relay;
    std::unordered_map<DFBSpecName, uint8_t> relay_pipe_id;
};

// Register DFBs with the Program
DataflowBufferHandles RegisterDataflowBuffers(
    const ProgramSpec& spec,
    const CollectedSpecData& collected,
    const KernelRiscMaskMap& kernel_to_risc_mask,
    detail::ProgramImpl& program_impl);

void WireDFBAliases(const ProgramSpec& spec, const DFBNameToIdMap& dfb_name_to_id, detail::ProgramImpl& program_impl);

// Register DFBs with a Kernel
tt::tt_metal::DataflowBufferBindingHandleMap MakeDataflowBufferBindingHandles(
    const KernelSpec& kernel_spec,
    const DFBNameToSlotMap& dfb_name_to_slot,
    const std::unordered_map<DFBSpecName, bool>& dfb_name_to_is_relay,
    const std::unordered_map<DFBSpecName, uint8_t>& dfb_name_to_prefetcher_pipe_id,
    const std::unordered_map<DFBSpecName, const DataflowBufferSpec*>& dfb_by_name);

// ----------------------------------------------------------------------------
// PrefetcherPipes (prefetcher_pipe.cpp)
// ----------------------------------------------------------------------------

using PrefetcherPipeHandlesByKernel =
    std::unordered_map<const KernelSpec*, std::vector<tt::tt_metal::PrefetcherPipeBindingHandle>>;

// Register PrefetcherPipes with the Program (each kernel's PrefetcherPipe binding handles come out of the slot
// reservation)
PrefetcherPipeHandlesByKernel ReservePrefetcherPipeSlots(
    distributed::MeshDevice& mesh_device,
    const ProgramSpec& spec,
    const CollectedSpecData& collected,
    detail::ProgramImpl& program_impl,
    const DFBNameToIdMap& dfb_name_to_id);

std::unordered_map<DFBSpecName, uint8_t> RecordRelayPipeIds(
    const DFBNameToIdMap& dfb_name_to_id, const detail::ProgramImpl& program_impl);

// ----------------------------------------------------------------------------
// Semaphores (semaphore.cpp)
// ----------------------------------------------------------------------------

using SemaphoreNameToIdMap = std::unordered_map<SemaphoreSpecName, uint32_t>;

struct SemaphoreHandles {
    SemaphoreNameToIdMap id;
    sem_solver::SemaphoreNameToScopeMap scope;
};

// Register semaphores with the Program
SemaphoreHandles RegisterSemaphores(
    const ProgramSpec& spec,
    const CollectedSpecData& collected,
    MetalEnvImpl& metal_env,
    detail::ProgramImpl& program_impl);

// Register semaphores with a Kernel
tt::tt_metal::SemaphoreBindingHandleMap MakeSemaphoreBindingHandles(
    const KernelSpec& kernel_spec,
    const sem_solver::SemaphoreBinderCensus& semaphore_binders,
    const SemaphoreNameToIdMap& semaphore_name_to_id,
    const sem_solver::SemaphoreNameToScopeMap& semaphore_name_to_scope);

// ----------------------------------------------------------------------------
// Scratchpads (scratchpad.cpp)
// ----------------------------------------------------------------------------

// Scratchpad does not need to be registered with the program.

// Register scratchpads with a Kernel
// Per-kernel resolved scratchpad bindings:
//  - one CRTA word per binding (the scratchpad's allocated L1 base address), in declaration order
//  - the scratchpad section sits immediately after the TensorBinding section and before varargs, so
//    each binding's absolute CRTA word index (and thus addr_crta_word) is fixed at codegen time
//    (varargs are open-ended / runtime-counted, so a section placed after them would not be).
// The allocated_address is left 0 here; allocate_scratchpads fills it once L1 is allocated.
struct ScratchpadBindingsForKernel {
    std::vector<ScratchpadBindingHandle> handles;
    uint32_t section_words = 0;  // == number of scratchpad bindings
};

ScratchpadBindingsForKernel ResolveScratchpadBindingsForKernel(
    const KernelSpec& kernel,
    const std::unordered_map<ScratchpadSpecName, const ScratchpadSpec*>& scratchpad_by_name,
    size_t scratchpad_base_crta_word);

// ----------------------------------------------------------------------------
// Drivers (resource.cpp)
// ----------------------------------------------------------------------------

// Everything kernel lowering reads that resource registration creates.
// Used by kernel lowering.
struct ProgramResources {
    std::unordered_map<TensorParamName, ResolvedTensorParameter> tensor_parameters;
    DataflowBufferHandles dfbs;
    PrefetcherPipeHandlesByKernel prefetcher_pipes;
    SemaphoreHandles semaphores;
};

// Register every resource with the Program (resource.cpp fixes the order).
ProgramResources RegisterResources(
    distributed::MeshDevice& mesh_device,
    const ProgramSpec& spec,
    const CollectedSpecData& collected,
    const KernelRiscMaskMap& risc_masks,
    MetalEnvImpl& metal_env,
    detail::ProgramImpl& program_impl);

// A kernel's binding handles to the registered resources. (Tensor and scratchpad bindings also place
// words in the kernel's argument buffers, so kernel lowering resolves them with the argument layout.)
struct KernelResourceBindings {
    tt::tt_metal::DataflowBufferBindingHandleMap dfbs;
    tt::tt_metal::SemaphoreBindingHandleMap semaphores;
    std::vector<tt::tt_metal::PrefetcherPipeBindingHandle> prefetcher_pipes;
};

KernelResourceBindings ResolveKernelResourceBindings(
    const KernelSpec& kernel_spec, const CollectedSpecData& collected, const ProgramResources& resources);

}  // namespace tt::tt_metal::experimental
