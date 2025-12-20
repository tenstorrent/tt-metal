// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal {
class IDevice;
class Program;

namespace detail {

// Launches all kernels on cores specified with kernels in the program.
// All kernels on a given Tensix core must be launched.
void LaunchProgram(
    IDevice* device, Program& program, bool wait_until_cores_done = true, bool force_slow_dispatch = false);
void LaunchProgram(
    IDevice* device,
    const std::shared_ptr<Program>& program,
    bool wait_until_cores_done = true,
    bool force_slow_dispatch = false);
void WaitProgramDone(IDevice* device, Program& program, bool read_device_profiler_results = true);

/**
 *  Compiles all kernels within the program, and generates binaries that are written to
 * `<tt-metal-cache directory>/<build_key>/kernels/<kernel name>/<kernel hash>`
 *
 *  The build key component accounts for device architecture as binaries are not compatible across architectures.
 *  To speed up compilation there is a kernel compilation cache that skips over generating binaries for the previously
 * compiled kernels. Kernel uniqueness is determined by the kernel hash which is computed based on compile time args,
 * defines, and kernel type specific attributes such as NOC for data movement kernels and math fidelity for compute
 * kernels.
 *  On cache hits the kernel is not recompiled if the output binary directory exists, otherwise the kernel is compiled.
 *  This cache is static and is enabled for the duration of the running process.
 *  Across runs, previously compiled kernels are recompiled if the source code or dependencies have changed.
 *
 *  Return value: void
 *
 * | Argument                  | Description                                                      | Type      | Valid
 * Range                                        | Required |
 * |---------------------------|------------------------------------------------------------------|-----------|----------------------------------------------------|----------|
 * | device                    | Which device the program is compiled for                         | IDevice*  | Must be
 * initialized via tt_metal::InitializeDevice | Yes      | | program                   | The program to compile |
 * Program & |                                                    | Yes      | | force_slow_dispatch        | Set when
 * a user wants to compile a program with Slow Dispatch Force Enabled (advanced feature, currently used internally to
 * launch Fast Dispatch Firmware and in the Device Performance Profiler)           | bool      | | No |
 */
void CompileProgram(IDevice* device, Program& program, bool force_slow_dispatch = false);

/**
 * Writes runtime args that are saved in the program to device
 *
 * Return value: void
 *
 * | Argument            | Description                                                            | Type | Valid Range
 * | Required |
 * |---------------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | device              | The device to whcih runtime args will be written                       | IDevice* | | Yes |
 * | program             | The program holding the runtime args                                   | const Program & | |
 * Yes      |
 */
void WriteRuntimeArgsToDevice(IDevice* device, Program& program, bool force_slow_dispatch = false);

// Configures a given device with a given program.
// - Loads all kernel binaries into L1s of assigned Tensix cores
// - Configures circular buffers (inits regs with buffer data)
// - Takes the device out of reset
bool ConfigureDeviceWithProgram(IDevice* device, Program& program, bool force_slow_dispatch = false);

}  // namespace detail
}  // namespace tt::tt_metal
