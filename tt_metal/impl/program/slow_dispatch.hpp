// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-umd/types/cluster_descriptor_types.hpp>  // ChipId

#include <tt-metalium/core_coord.hpp>
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
// `kernel_config_buffer`, when given, holds the program's Tensix kernel-config blocks in place of the HAL KERNEL_CONFIG
// window, each core's at its own address (experimental::EnqueueMeshWorkload with a KernelConfigBuffer).
namespace tt::tt_metal::slow_dispatch {

// Where `logical_core`'s kernel-config block starts on `device_id`: its shard of `kernel_config_buffer` for a Tensix
// core when one is given, otherwise the HAL KERNEL_CONFIG address of the core type.
uint32_t GetKernelConfigBase(
    const Hal& hal,
    uint32_t programmable_core_type_index,
    ChipId device_id,
    const CoreCoord& logical_core,
    const experimental::KernelConfigBufferImpl* kernel_config_buffer);

// Compiles `program` for `device` and lays out its kernel-config blocks, once. Checks no size limit: launch checks the
// blocks against the KERNEL_CONFIG window or a kernel config buffer.
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
