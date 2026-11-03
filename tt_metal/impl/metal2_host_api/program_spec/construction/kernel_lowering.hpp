// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/metal2_host_api/program_spec/collection/collect_metadata.hpp"
#include "impl/metal2_host_api/program_spec/construction/processor_assignment/processor_assignment.hpp"
#include "impl/metal2_host_api/program_spec/construction/resource/resource.hpp"
#include "impl/program/program_impl.hpp"
#include "llrt/hal.hpp"

namespace tt::tt_metal::experimental {

// Step 3: lower every KernelSpec into a Kernel (source, argument layout, hardware config, binding
// handles) and add it to the Program, registering its name and runtime-arg schema.
void AddKernels(
    const ProgramSpec& spec,
    const CollectedSpecData& collected,
    const KernelRiscMaskMap& risc_masks,
    const ProgramResources& resources,
    const Hal& hal,
    detail::ProgramImpl& program_impl);

}  // namespace tt::tt_metal::experimental
