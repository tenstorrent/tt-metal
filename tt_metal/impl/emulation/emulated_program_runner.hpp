// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal {
class IDevice;
class Program;
}  // namespace tt::tt_metal

namespace tt::tt_metal::emule {

/// Execute all kernels in a Program on the emulated device: resolve + JIT-compile each kernel
/// (cached), then run one cooperatively-scheduled fiber per (core, RISC) to completion. Kernel
/// memory I/O reaches the SWEmuleChip backing store through extern "C" bridge hooks.
/// See tt-emule docs/metal-integration.md and docs/fiber-engine.md.
void execute_program_emulated(IDevice* device, Program& program);

/// Multi-device (mesh) register/run split: begin_mesh_dispatch() puts the runner in "defer" mode
/// so each execute_program_emulated registers its fibers without running; run_mesh_dispatch() then
/// drives one run_until_idle so all devices' fibers run concurrently — required for inter-chip CCL,
/// where chips co-run. Single-device (begin_mesh_dispatch not called) runs synchronously.
void begin_mesh_dispatch();
void run_mesh_dispatch();

}  // namespace tt::tt_metal::emule
