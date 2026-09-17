// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Thin orchestrator. The 6 public symbols of emulated_program_runner.hpp forward to the emule
// engine (emule_engine.{hpp,cpp}), which owns the launch loop, program resolution/dispatch, and
// the run-state / mesh-dispatch machinery. The rest of the emulator is the emule_*.{hpp,cpp}
// concern modules (device_map, noc_bridge, jit, program_model, kernel_defines, cb_dfb_setup,
// metal2_emit, tile_geometry, fabric, diagnostics) + the tt_emule host types.

#include "emulated_program_runner.hpp"
#include "emule_engine.hpp"

#include <tt-metalium/device.hpp>
#include <tt-metalium/program.hpp>

namespace tt::tt_metal::emule {

void execute_program_emulated(IDevice* device, Program& program) { engine::execute_program_emulated(device, program); }

void begin_mesh_dispatch() { engine::begin_mesh_dispatch(); }

void run_mesh_dispatch() { engine::run_mesh_dispatch(); }

MeshDispatchLock::MeshDispatchLock() { engine::mesh_lock_acquire(); }
MeshDispatchLock::~MeshDispatchLock() { engine::mesh_lock_release(); }

void pump_device() { engine::pump_device(); }

void drain_device(const std::vector<int>& device_ids) { engine::drain_device(device_ids); }

}  // namespace tt::tt_metal::emule
