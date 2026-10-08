// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/metal2_host_api/program_spec/construction/resource/resource.hpp"

#include <utility>

namespace tt::tt_metal::experimental {

ProgramResources RegisterResources(
    distributed::MeshDevice& mesh_device,
    const ProgramSpec& spec,
    const CollectedSpecData& collected,
    const KernelRiscMaskMap& risc_masks,
    MetalEnvImpl& metal_env,
    detail::ProgramImpl& program_impl) {
    ProgramResources resources;
    resources.tensor_parameters = RegisterTensorParameters(spec, mesh_device, program_impl);
    resources.dfbs = RegisterDataflowBuffers(spec, collected, risc_masks, program_impl);

    // Reserve PrefetcherPipe slots (one per kernel accessor group) from the spec geometry, register
    // relay DFBs against them, and record each parameter's placement for SetProgramRunArgs. Must
    // precede kernel creation: the slot id is baked into the kernel's `pipe::<accessor>` token and
    // a relay DFB's `dfb::` token.
    resources.prefetcher_pipes =
        ReservePrefetcherPipeSlots(mesh_device, spec, collected, program_impl, resources.dfbs.id);
    resources.dfbs.relay_pipe_id = RecordRelayPipeIds(resources.dfbs.id, program_impl);

    WireDFBAliases(spec, resources.dfbs.id, program_impl);
    resources.semaphores = RegisterSemaphores(spec, collected, metal_env, program_impl);
    return resources;
}

KernelResourceBindings ResolveKernelResourceBindings(
    const KernelSpec& kernel_spec, const CollectedSpecData& collected, const ProgramResources& resources) {
    KernelResourceBindings bindings{
        .dfbs = MakeDataflowBufferBindingHandles(
            kernel_spec,
            resources.dfbs.slot,
            resources.dfbs.is_relay,
            resources.dfbs.relay_pipe_id,
            collected.dfb_by_name),
        .semaphores = MakeSemaphoreBindingHandles(
            kernel_spec, collected.semaphore_binders, resources.semaphores.id, resources.semaphores.scope),
        .prefetcher_pipes = {}};
    if (auto pipe_it = resources.prefetcher_pipes.find(&kernel_spec); pipe_it != resources.prefetcher_pipes.end()) {
        bindings.prefetcher_pipes = pipe_it->second;
    }
    return bindings;
}

}  // namespace tt::tt_metal::experimental
