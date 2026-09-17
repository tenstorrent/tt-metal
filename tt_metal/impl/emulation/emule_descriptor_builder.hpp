// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// The emule marshaller: the ONLY code that reads private tt-metal types
// (ProgramImpl / Kernel / CircularBufferImpl / Semaphore / DataflowBufferImpl /
// metal_SocDescriptor / HAL) and flattens them into the public POD in
// emule_program_descriptor.hpp. Everything downstream consumes the POD, never a
// private type. See docs/state-tiers.md and the Emule Extraction Map.

#include "emule_program_descriptor.hpp"

namespace tt::tt_metal {
class IDevice;
class Program;
}  // namespace tt::tt_metal

namespace tt_emule {

// Device-scoped SoC geometry / bank maps / HAL addrs. The fabric routing identity (mesh/chip id
// + ROUTING_TABLE addr) is per program-context, so the program is passed to read it.
SocView build_soc_view(tt::tt_metal::IDevice* device, tt::tt_metal::Program& program);

// Per-program: config, kernels, kernel groups, per-core CB/DFB/semaphore setup.
EmuleProgramDescriptor build_emule_descriptor(tt::tt_metal::Program& program, tt::tt_metal::IDevice* device);

}  // namespace tt_emule
