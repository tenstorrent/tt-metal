/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <variant>
#include <vector>
#include "common/core_coord.h"
#include "tt_metal/impl/program.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include <optional>
/** @file */

/** \mainpage tt-metal Internal C++ Documentation
 *
 * Welcome. Please navigate using the Files menu. All APIs are documented
 * under the files listed in the Files menu.
 *
 * If you want to contribute to the documentation and are looking for a good
 * resource for generating Markdown tables, refer to
 * https://www.tablesgenerator.com/markdown_tables.
 * */

namespace tt {

namespace tt_metal {

class Program;
class Host;
class Device;
class CommandQueue;
class CircularBuffer;

// ==================================================
//                  HOST API: host and device
// ==================================================

/**
 * Instantiates a device object.
 *
 * Return value: Device *
 *
 * | Argument       | Description                                                      | Data type       | Valid range                                         | required |
 * |----------------|------------------------------------------------------------------|-----------------|-----------------------------------------------------|----------|
 * | device_id      | ID of device to target                                           | chip_id_t (int) | 0 to (Device::detect_num_available_devices - 1)     | Yes      |
 * */
Device *CreateDevice(chip_id_t device_id, const std::vector<uint32_t>& l1_bank_remap = {});

/**
 * Resets device and closes device
 *
 * Return value: bool
 *
 * | Argument | Description                | Type     | Valid Range | Required |
 * |----------|----------------------------|----------|-------------|----------|
 * | device   | Pointer to a device object | Device * |             | True     |
 */
bool CloseDevice(Device *device);

// ==================================================
//                  HOST API: program & kernels
// ==================================================

/**
 * Creates a data movement kernel with no compile time arguments and adds it to the program.
 *
 * Return value: Kernel ID (uintptr_t)
 *
 * | Argument     | Description                                                                                                                          | Type                                                     | Valid Range | Required |
 * |--------------|--------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------|-------------|----------|
 * | program      | The program to which this kernel will be added to                                                                                    | Program &                                                |             | Yes      |
 * | file_name    | Path to kernel src                                                                                                                   | const std::string &                                      |             | Yes      |
 * | core_spec    | Either a single logical core, a range of logical cores or a set of logical core ranges that indicate which cores kernel is placed on | const std::variant<CoreCoord, CoreRange, CoreRangeSet> & |             | Yes      |
 * | config       | Config for data movement kernels                                                                                                     | const std::optional<DataMovementConfig> &                |             | No       |
 */
KernelID CreateDataMovementKernel(Program &program, const std::string &file_name, const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec, const std::optional<DataMovementConfig> &config = {});

/**
 * Creates a compute kernel object, and adds it to the program.
 *
 * Return value: Kernel ID (uintptr_t)
 *
 * | Argument     | Description                                                                                                                          | Type                                                     | Valid Range | Required |
 * |--------------|--------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------|-------------|----------|
 * | program      | The program to which this kernel will be added to                                                                                    | Program &                                                |             | Yes      |
 * | file_name    | Path to kernel src                                                                                                                   | const std::string &                                      |             | Yes      |
 * | core_spec    | Either a single logical core, a range of logical cores or a set of logical core ranges that indicate which cores kernel is placed on | const std::variant<CoreCoord, CoreRange, CoreRangeSet> & |             | Yes      |
 * | config       | Config for compute kernels                                                                                                           | const std::optional<ComputeConfig> &                     |             | No       |
 */
KernelID CreateComputeKernel(Program &program, const std::string &file_name, const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec, const std::optional<ComputeConfig> &config = {});

// ==================================================
//                  HOST API: buffers
// ==================================================
/**
 * Creates a Circular Buffer (CB) in L1 memory of all cores within core ranges (inclusive) and adds it to the program. There can be a total of NUM_CIRCULAR_BUFFERS (32) circular buffers per core.
 * Circular buffers hold data and have an associated config which indicates usage of the address space.
 * If the config is specified for multiple buffer indices, the circular buffer address space is shared and each buffer index can potentially have a unique view of the shared space.
 *
 * Return value: Circular Buffer ID (uintptr_t)
 *
 * | Argument  | Description                                                                                                                                       | Type                                                     | Valid Range | Required |
 * |-----------|---------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------|-------------|----------|
 * | program   | The program to which buffer will be added to                                                                                                      | Program &                                                |             | Yes      |
 * | core_spec | Either a single logical core, a range of logical cores or a set of logical core ranges that indicate where the circular buffer will be configured | const std::variant<CoreCoord, CoreRange, CoreRangeSet> & |             | Yes      |
 * | config    | Config for circular buffer                                                                                                                        | const CircularBufferConfig &                             |             | Yes      |
 */
CircularBufferID CreateCircularBuffer(Program &program, const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec, const CircularBufferConfig &config);

/**
 * Gets a reference to the config owned by circular buffer at the given circular buffer ID. This should be used to update the circular buffer config.
 * This API will invalidate circular buffer address allocation for all circular buffers in the program. Circular buffers will be allocated when the program is run.
 *
 * Return value: CircularBufferConfig &
 *
 * | Argument           | Description                                                    | Type                         | Valid Range | Required |
 * |--------------------|----------------------------------------------------------------|------------------------------|-------------|----------|
 * | program            | The program containing the circular buffer                     | Program &                    |             | Yes      |
 * | circular_buffer_id | ID of the circular buffer, returned by `CreateCircularBuffer`  | CircularBufferID (uintptr_t) |             | Yes      |
*/
CircularBufferConfig &GetCircularBufferConfig(Program &program, CircularBufferID circular_buffer_id);

/**
 * Initializes semaphore on all cores within core range (inclusive). Each core can have up to four 32B semaphores.
 *
 * Return value: Semaphore address (uint32_t)
 *
 * | Argument      | Description                                          | Type                                                  | Valid Range                                              | Required |
 * |---------------|------------------------------------------------------|-------------------------------------------------------|----------------------------------------------------------|----------|
 * | program       | The program to which semaphore will be added to      | Program &                                             |                                                          | Yes      |
 * | core_range    | Range of the Tensix co-ordinates using the semaphore | const CoreRange & (std::pair<CoreCoord, CoreCoord>)   | Pair of logical coords where first coord <= second coord | Yes      |
 * | initial_value | Initial value of the semaphore                       | uint32_t                                              |                                                          | Yes      |
 */
uint32_t CreateSemaphore(Program &program, const CoreRange &core_range, uint32_t initial_value);

/**
 * Initializes semaphore on all cores within core range (inclusive). Each core can have up to four 32B semaphores.
 *
 * Return value: Semaphore address (uint32_t)
 *
 * | Argument       | Description                                                 | Type                   | Valid Range                                               | Required |
 * |----------------|-------------------------------------------------------------|------------------------|-----------------------------------------------------------|----------|
 * | program        | The program to which semaphore will be added to             | Program &              |                                                           | Yes      |
 * | core_range_set    | Set of Range of the Tensix co-ordinates using the semaphore | const CoreRangeSet &   | Pairs of logical coords where first coord <= second coord | Yes      |
 * | initial_value  | Initial value of the semaphore                              | uint32_t               |                                                           | Yes      |
 */
uint32_t CreateSemaphore(Program &program, const CoreRangeSet &core_range_set, uint32_t initial_value);

/**
*  Allocates a DRAM or L1 buffer on device
*
*  Return value: Buffer
*
*  | Argument       | Description                                | Type           | Valid Range | Required |
*  |----------------|------------------------------------------- |----------------|-------------|----------|
*  | device         | The device that the buffer will reside     | Device         |             | Yes      |
*  | size           | size of buffer                             | uint64_t       |             | Yes      |
*  | page_size      | buffer page size                           | uint64_t       |             | Yes      |
*  | buffer_storage | storage of buffer (L1 or DRAM)             | BufferStorage  |             | Yes      |
*/
Buffer CreateBuffer(Device *device, std::uint64_t size, std::uint64_t page_size, const BufferStorage buffer_storage,
        const TensorMemoryLayout buffer_layout=TensorMemoryLayout::INTERLEAVED,
        std::optional<ShardSpec> shard_parameter = std::nullopt);

/**
* Copies data from a host buffer into the specified buffer
*
* Return value: void
*
* | Argument    | Description                                     | Data type               | Valid range                                      | Required |
* |-------------|-------------------------------------------------|-------------------------|--------------------------------------------------|----------|
* | buffer      | Buffer to send data to                          | const Buffer &          |                                                  | Yes      |
* | host_buffer | Buffer on host to copy data from                | std::vector<uint32_t> & | Host buffer size must match buffer               | Yes      |
*/
void WriteToBuffer(const Buffer &buffer, const std::vector<uint32_t> &host_buffer);

/**
* Copies data from a buffer into a host buffer
*
* Return value: void
*
* | Argument        | Description                                                                | Data type                           | Valid range                                      | Required |
* |-----------------|----------------------------------------------------------------------------|-------------------------------------|--------------------------------------------------|----------|
* | buffer          | Buffer to read data from                                                   | const Buffer &                      |                                                  | Yes      |
* | host_buffer     | Buffer on host to copy data into                                           | std::vector<uint32_t> &             |                                                  | Yes      |
* | override_layout | Overrides tensor memory layout to a diff layout than buffer                | std::optional<TensorMemoryLayout> & |                                                  | No       |
*/
void ReadFromBuffer(const Buffer &buffer, std::vector<uint32_t> &host_buffer, std::optional<TensorMemoryLayout> override_layout = std::nullopt);

/**
*  Deallocates buffer from device by marking its memory as free.
*
*  Return value: void
*
*  | Argument | Description                          | Type     | Valid Range | Required |
*  |----------|--------------------------------------|----------|-------------|----------|
*  | buffer   | The buffer to deallocate from device | Buffer & |             | Yes      |
*/
void DeallocateBuffer(Buffer &buffer);



// ==================================================
//           COMPILE & EXECUTE KENRNELS
//
// ==================================================

/**
 * Set runtime args for a kernel that are sent to the core during runtime. This API needs to be called to update the runtime args for the kernel.
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                          | Valid Range                                                      | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------------------------------------|----------|
 * | program      | The program containing kernels, circular buffers, semaphores           | const Program &               |                                                                  | Yes      |
 * | kernel_id    | ID of the kernel that will receive the runtime args                    | KernelID (u64)                |                                                                  | Yes      |
 * | logical_core | The location of the Tensix core where the runtime args will be written | const CoreCoord &             | Any logical Tensix core coordinate on which the kernel is placed | Yes      |
 * | runtime_args | The runtime args to be written                                         | const std::vector<uint32_t> & |                                                                  | Yes      |
 */
void SetRuntimeArgs(const Program &program, KernelID kernel, const CoreCoord &logical_core, const std::vector<uint32_t> &runtime_args);

/**
 * Set runtime args for a kernel that are shared amongst a range of cores. Runtime args are sent to cores during runtime. This API needs to be called to update the runtime args for the kernel.
 *
 * Return value: void
 *
 * | Argument     | Description                                                                                            | Type                          | Valid Range                                                             | Required |
 * |--------------|--------------------------------------------------------------------------------------------------------|-------------------------------|-------------------------------------------------------------------------|----------|
 * | program      | The program containing kernels, circular buffers, semaphores                                           | const Program &               |                                    | Yes      |
 * | kernel_id    | ID of the kernel that will receive the runtime args                                                    | KernelID (u64)                |                                    | Yes      |
 * | core_range   | The range of the Tensix co-ordinates which receive the runtime args (Logical co-ordinates)             | const CoreRange &             | A range of any logical Tensix core coordinate on which the kernel is placed | Yes      |
 * | runtime_args | The runtime args to be written to the core range                                                       | const std::vector<uint32_t> & |                                                                         | Yes      |
 */
void SetRuntimeArgs(const Program &program, KernelID kernel, const CoreRange &core_range, const std::vector<uint32_t> &runtime_args);

/**
 * Set runtime args for a kernel that are shared amongst a CoreRangeSet. Runtime args are sent to cores during runtime. This API needs to be called to update the runtime args for the kernel.
 *
 * Return value: void
 *
 * | Argument       | Description                                                                                            | Type                          | Valid Range                        | Required |
 * |----------------|--------------------------------------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | program      | The program containing kernels, circular buffers, semaphores                                             | const Program &               |                                    | Yes      |
 * | kernel_id    | ID of the kernel that will receive the runtime args                                                      | KernelID (u64)                |                                    | Yes      |
 * | core_range_set | Set of ranges of Tensix co-ordinates which receive the runtime args (Logical co-ordinates)             | const CoreRangeSet &          | Ranges of any logical Tensix core coordinate on which the kernel is placed | Yes      |
 * | runtime_args   | The runtime args to be written to the core ranges                                                      | const std::vector<uint32_t> & |                                    | Yes      |
 */
void SetRuntimeArgs(const Program &program, KernelID kernel, const CoreRangeSet &core_range_set, const std::vector<uint32_t> &runtime_args);

/**
 * Get the runtime args for a kernel.
 *
 * Return value: std::vector<uint32_t>
 *
 * | Argument     | Description                                                            | Type                          | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | program      | The program containing kernels, circular buffers, semaphores           | const Program &               |                                    | Yes      |
 * | kernel_id    | ID of the kernel that will receive the runtime args                    | KernelID (u64)                |                                    | Yes      |
 * | logical_core | The location of the Tensix core where the runtime args will be written | const CoreCoord &             | Any logical Tensix core coordinate | Yes      |
 */
std::vector<uint32_t> GetRuntimeArgs(const Program &program, KernelID kernel_id, const CoreCoord &logical_core);

// Launches all kernels on cores specified with kernels in the program.
// All kernels on a given Tensix core must be launched.
void LaunchProgram(Device *device, Program &program);

/**
 * Reads a buffer from the device
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                          | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | cq           | The command queue object which dispatches the command to the hardware  | CommandQueue &                |                                    | Yes      |
 * | buffer       | The device buffer we are reading from                                  | Buffer &                      |                                    | Yes      |
 * | dst          | The vector where the results that are read will be stored              | vector<u32> &                 |                                    | Yes      |
 * | blocking     | Whether or not this is a blocking operation                            | bool                          |                                    | Yes      |
 */
void EnqueueReadBuffer(CommandQueue& cq, Buffer& buffer, std::vector<u32>& dst, bool blocking);

/**
 * Writes a buffer to the device
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                          | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | cq           | The command queue object which dispatches the command to the hardware  | CommandQueue &                |                                    | Yes      |
 * | buffer       | The device buffer we are writing to                                    | Buffer &                      |                                    | Yes      |
 * | src          | The vector we are writing to the device                                | vector<u32> &                 |                                    | Yes      |
 * | blocking     | Whether or not this is a blocking operation                            | bool                          |                                    | Yes      |
 */
void EnqueueWriteBuffer(CommandQueue& cq, Buffer& buffer, std::vector<u32>& src, bool blocking);

/**
 * Writes a program to the device and launches it
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                          | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | cq           | The command queue object which dispatches the command to the hardware  | CommandQueue &                |                                    | Yes      |
 * | program      | The program that will be executed on the device that cq is bound to    | Program &                     |                                    | Yes      |
 * | blocking     | Whether or not this is a blocking operation                            | bool                          |                                    | Yes      |
 */
void EnqueueProgram(CommandQueue& cq, Program& program, bool blocking);

/**
 * Blocks until all previously dispatched commands on the device have completed
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                          | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | cq           | The command queue object which dispatches the command to the hardware  | CommandQueue &                |                                    | Yes      |
 */
void Finish(CommandQueue& cq);

}  // namespace tt_metal

}  // namespace tt
