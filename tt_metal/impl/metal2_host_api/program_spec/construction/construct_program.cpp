// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/metal2_host_api/program_spec/construction/construct_program.hpp"

#include <memory>
#include <utility>

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/metal2_host_api/program_spec/construction/kernel_lowering.hpp"
#include "impl/metal2_host_api/program_spec/construction/processor_assignment/processor_assignment.hpp"
#include "impl/metal2_host_api/program_spec/construction/resource/resource.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "distributed/mesh_device_impl.hpp"

namespace tt::tt_metal::experimental {

Program BuildProgram(
    distributed::MeshDevice& mesh_device,
    const ProgramSpec& spec,
    const CollectedSpecData& collected,
    [[maybe_unused]] MetalContext& metal_ctx) {
    // Arch and the Emule check come from the mesh device's env, not the default context.
    MetalEnvImpl& metal_env = mesh_device.impl().metal_env();
    const Hal& hal = metal_env.get_hal();
    auto program_impl = std::make_shared<detail::ProgramImpl>(mesh_device.impl().get_context_id());
    program_impl->mark_created_from_spec();  // mark as Metal 2.0 ProgramSpec-created (for legality checks)

    // Step 1: Processor assignment
    const KernelRiscMaskMap risc_masks = SolveKernelRiscMasks(spec, collected, hal);

    // Step 2: Register resources with the Program
    const ProgramResources resources =
        RegisterResources(mesh_device, spec, collected, risc_masks, metal_env, *program_impl);

    // Step 3: Construct Kernels and add them to the Program
    AddKernels(spec, collected, risc_masks, resources, hal, *program_impl);

    return Program(std::move(program_impl));
}

}  // namespace tt::tt_metal::experimental
