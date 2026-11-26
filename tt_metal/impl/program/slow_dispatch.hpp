// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-umd/types/cluster_descriptor_types.hpp>  // ChipId

#include <tt-metalium/vocabulary/core_coord.hpp>
#include "llrt/hal.hpp"

namespace tt::tt_metal {
class Hal;
class IDevice;
class Program;
namespace experimental {
class KernelConfigBufferImpl;
}  // namespace experimental
}  // namespace tt::tt_metal

// Slow dispatch building blocks for running a program on a single device, bypassing the command queue.
// `force_slow_dispatch` allows mixing with an active fast dispatch session (advanced feature, used internally to launch
// dispatch firmware, fabric, and profiler programs).
//
// `kernel_config_buffer`, when given, holds the program's Tensix kernel binaries in place of the HAL KERNEL_CONFIG
// window, each core's at its own address (experimental::EnqueueMeshWorkload with a KernelConfigBuffer). The rest of
// the kernel-config block (runtime args, semaphores, CB and DFB configs) stays in the window.
namespace tt::tt_metal::slow_dispatch {

// The address `logical_core`'s kernel text offsets count from on `device_id`: the HAL KERNEL_CONFIG address of the
// core type, or, for a Tensix core with a `kernel_config_buffer`, the address that puts the program's text section at
// the start of the core's shard.
uint32_t GetKernelTextBase(
    const Hal& hal,
    const Program& program,
    uint32_t programmable_core_type_index,
    ChipId device_id,
    const CoreCoord& logical_core,
    const experimental::KernelConfigBufferImpl* kernel_config_buffer);

// Points the kernel text offsets of `launch_msg`, a copy of `logical_core`'s kernel group's, at the core's shard of
// `kernel_config_buffer`. Does nothing without a buffer or for a non-Tensix core.
void SetKernelTextOffsets(
    dev_msgs::launch_msg_t& launch_msg,
    const Hal& hal,
    const Program& program,
    uint32_t programmable_core_type_index,
    ChipId device_id,
    const CoreCoord& logical_core,
    const experimental::KernelConfigBufferImpl* kernel_config_buffer);

// Compiles `program` for `device` and lays out its kernel-config blocks, once. Checks no size limit: launch checks the
// blocks against the KERNEL_CONFIG window, and the kernel binaries against a kernel config buffer when one is given.
void PrepareProgram(IDevice& device, Program& program);

// Allocates circular / dataflow buffers and writes kernel binaries and configs to the device.
void ConfigureDeviceWithProgram(
    IDevice& device,
    Program& program,
    bool force_slow_dispatch,
    const experimental::KernelConfigBufferImpl* kernel_config_buffer = nullptr);

// Writes the runtime args saved in the program to the device.
void WriteRuntimeArgsToDevice(
    IDevice& device,
    Program& program,
    bool force_slow_dispatch,
    const experimental::KernelConfigBufferImpl* kernel_config_buffer = nullptr);

// Compiles and configures `program`, writes its runtime args, and sends the go signal. Does not wait for completion,
// use WaitProgramDone for that.
void LaunchProgramAsync(
    IDevice& device,
    Program& program,
    bool force_slow_dispatch,
    const experimental::KernelConfigBufferImpl* kernel_config_buffer = nullptr);

// experimental::ConfigureProgramWithoutLaunch, with an optional kernel config buffer.
void ConfigureProgramWithoutLaunch(
    IDevice* device, Program& program, const experimental::KernelConfigBufferImpl* kernel_config_buffer);

// experimental::DispatchCompiledProgramToDevice, with an optional kernel config buffer.
void DispatchCompiledProgram(
    IDevice* device, Program& program, const experimental::KernelConfigBufferImpl* kernel_config_buffer);

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
