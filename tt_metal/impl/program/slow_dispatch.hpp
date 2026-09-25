// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal {
class IDevice;
class Program;
}  // namespace tt::tt_metal

// Slow dispatch building blocks for running a program on a single device, bypassing the command queue.
// `force_slow_dispatch` allows mixing with an active fast dispatch session (advanced feature, used internally to launch
// dispatch firmware, fabric, and profiler programs).
namespace tt::tt_metal::slow_dispatch {

// Allocates circular / dataflow buffers and writes kernel binaries and configs to the device.
void ConfigureDeviceWithProgram(IDevice& device, Program& program, bool force_slow_dispatch);

// Writes the runtime args saved in the program to the device.
void WriteRuntimeArgsToDevice(IDevice& device, Program& program, bool force_slow_dispatch);

// Compiles and configures `program`, writes its runtime args, and sends the go signal. Does not wait for completion,
// use WaitProgramDone for that.
void LaunchProgram(IDevice& device, Program& program, bool force_slow_dispatch);

// Waits until all cores used by the program are idle.
void WaitProgramDone(IDevice& device, const Program& program);

}  // namespace tt::tt_metal::slow_dispatch
