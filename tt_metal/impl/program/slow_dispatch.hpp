// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core_coord.hpp"
#include "llrt/hal.hpp"

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
void LaunchProgramAsync(IDevice& device, Program& program, bool force_slow_dispatch);

// Waits until all cores used by the program, i.e. the cores LaunchProgramAsync sends the go signal to, are idle.
void WaitProgramDone(IDevice& device, const Program& program);

// Writes a kernel group's launch message, and its go message when send_go is set, to a core, counting them as
// program config in host transfer stats.
void WriteLaunchMsgToCore(
    IDevice& device,
    const CoreCoord& virtual_core,
    dev_msgs::launch_msg_t::View launch_msg,
    dev_msgs::go_msg_t::ConstView go_msg,
    bool send_go);

}  // namespace tt::tt_metal::slow_dispatch
