// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// The emule engine: the per-core launch loop (launch_cores), program resolution/dispatch
// (prepare_program / dispatch_to_device), and the run-state + mesh-dispatch machinery.
// emulated_program_runner.cpp is now a thin forwarder over these entry points (it owns only
// the 6 public symbols of emulated_program_runner.hpp). This holds all the machinery those
// symbols drive, consuming the extracted modules (cb_dfb_setup, program_model, jit,
// kernel_defines, fabric, device_map, noc_bridge, metal2_emit, descriptor_builder).

#include <vector>

namespace tt::tt_metal {
class IDevice;
class Program;
namespace emule::engine {

// Impl behind emulated_program_runner.hpp's 6 public symbols.
void execute_program_emulated(IDevice* device, Program& program);
void begin_mesh_dispatch();
void run_mesh_dispatch();
void pump_device();
void drain_device(const std::vector<int>& device_ids);
// MeshDispatchLock ctor/dtor logic (the public RAII type stays in the runner).
void mesh_lock_acquire();
void mesh_lock_release();

}  // namespace emule::engine
}  // namespace tt::tt_metal
