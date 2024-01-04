// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>
#include "common/core_coord.h"
#include "tt_metal/impl/program/program.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"

/** @file */

/** \mainpage tt-metal Internal C++ Documentation
 *
 * Welcome. Please navigate using the Files menu. All APIs are documented
 * under the files listed in the Files menu.
 *
 * If you want to contribute to the documentation and are looking for a good
 * resource for generating Markdown tables, refer to
 * https://www.tablesgenerator.com/markdown_tables
 * */

namespace tt {

namespace tt_metal {

class Program;
class Host;
class Device;
class CommandQueue;
class CircularBuffer;

// ==================================================
//                  HOST API: Device management
// ==================================================

/**
 * Returns number of Tenstorrent devices that can be targeted
 *
 * Return value: size_t
 */
size_t GetNumAvailableDevices();

/**
 * Returns number of Tenstorrent devices that are connected to host via PCIe and can be targeted
 *
 * Return value: size_t
 */
size_t GetNumPCIeDevices();

/**
 * Instantiates a device object.
 *
 * Return value: Device *
 *
 * | Argument   | Description                | Type            | Valid Range                       | Required |
 * |------------|----------------------------|-----------------|-----------------------------------|----------|
 * | device_id  | ID of the device to target| chip_id_t (int) | 0 to (GetNumAvailableDevices - 1) | Yes      |
 * */
Device *CreateDevice(chip_id_t device_id, const uint8_t num_hw_cqs = 1, const std::vector<uint32_t>& l1_bank_remap = {});

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
 * Creates a Program object which is the main container that bundles kernels, circular buffers, and/or semaphores for execution on device
 *
 * Return value: Program
 */
Program CreateProgram();

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
 * | config       | Config for data movement or compute kernel                                                                                           | const std::variant<DataMovementConfig,ComputeConfig,EthernetConfig> &   |             | No       |
 */
KernelHandle CreateKernel(Program &program, const std::string &file_name, const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec, const std::variant<DataMovementConfig,ComputeConfig,experimental::EthernetConfig> & config);

// ==================================================
//                  HOST API: buffers
// ==================================================
/**
 * Creates a Circular Buffer (CB) in L1 memory of all cores within core ranges (inclusive) and adds it to the program. There can be a total of NUM_CIRCULAR_BUFFERS (32) circular buffers per core.
 * Circular buffers hold data and have an associated config which indicates usage of the address space.
 * If the config is specified for multiple buffer indices, the circular buffer address space is shared and each buffer index can potentially have a unique view of the shared space.
 *
 * Circular buffers can be dynamically allocated or program-local allocated. If the config is created with an L1 buffer or sets a globally allocated address it is dynamic and shares the same address space as the L1 buffer.
 * Otherwise, the circular buffer address space is managed by the program. Address space for program-local circular buffers does not persist across programs.
 *
 * Return value: Circular Buffer ID (uintptr_t)
 *
 * | Argument  | Description                                                                                                                                       | Type                                                     | Valid Range | Required |
 * |-----------|---------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------|-------------|----------|
 * | program   | The program to which buffer will be added to                                                                                                      | Program &                                                |             | Yes      |
 * | core_spec | Either a single logical core, a range of logical cores or a set of logical core ranges that indicate where the circular buffer will be configured | const std::variant<CoreCoord, CoreRange, CoreRangeSet> & |             | Yes      |
 * | config    | Config for circular buffer                                                                                                                        | const CircularBufferConfig &                             |             | Yes      |
 */
CBHandle CreateCircularBuffer(Program &program, const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec, const CircularBufferConfig &config);

/**
 * Gets a reference to the config owned by circular buffer at the given circular buffer ID.
 *
 * Return value: const CircularBufferConfig &
 *
 * | Argument  | Description                                                    | Type                         | Valid Range | Required |
 * |-----------|----------------------------------------------------------------|------------------------------|-------------|----------|
 * | program   | The program containing the circular buffer                     | Program &                    |             | Yes      |
 * | cb_handle | ID of the circular buffer, returned by `CreateCircularBuffers` | CBHandle (uintptr_t) |             | Yes      |
*/
const CircularBufferConfig &GetCircularBufferConfig(Program &program, CBHandle cb_handle);

/**
 * Update the total size of the circular buffer at the given circular buffer handle. Updating a program-local circular buffer requires all circular buffers in the program to be reallocated.
 *
 * Return value: void
 *
 * | Argument   | Description                                                    | Type                         | Valid Range | Required |
 * |------------|----------------------------------------------------------------|------------------------------|-------------|----------|
 * | program    | The program containing the circular buffer                     | Program &                    |             | Yes      |
 * | cb_handle  | ID of the circular buffer, returned by `CreateCircularBuffers` | CBHandle (uintptr_t) |             | Yes      |
 * | total_size | New size of the circular buffer in bytes                       | uint32_t                     |             | Yes      |
*/
void UpdateCircularBufferTotalSize(Program &program, CBHandle cb_handle, uint32_t total_size);

/**
 * Update the page size at specified `buffer_index` of the circular buffer at the given circular buffer handle.
 *
 * Return value: void
 *
 * | Argument     | Description                                                                                                                | Type                         | Valid Range                   | Required |
 * |--------------|----------------------------------------------------------------------------------------------------------------------------|------------------------------|-------------------------------|----------|
 * | program      | The program containing the circular buffer                                                                                 | Program &                    |                               | Yes      |
 * | cb_handle    | ID of the circular buffer, returned by `CreateCircularBuffers`                                                             | CBHandle (uintptr_t) |                               | Yes      |
 * | buffer_index | Circular buffer index to update page size. `cb_handle` must be a circular buffer that had previously programmed this index | uint8_t                      | 0 to NUM_CIRCULAR_BUFFERS - 1 | Yes      |
 * | page_size    | Updated page size in bytes                                                                                                 | uint32_t                     |                               | Yes      |
*/
void UpdateCircularBufferPageSize(Program &program, CBHandle cb_handle, uint8_t buffer_index, uint32_t page_size);

/**
 * Update the address of a dynamic circular buffer. Dynamic circular buffers share the same address space as L1 buffers.
 *
 * Return value: void
 *
 * | Argument  | Description                                                                              | Type                         | Valid Range | Required |
 * |-----------|------------------------------------------------------------------------------------------|------------------------------|-------------|----------|
 * | program   | The program containing the circular buffer                                               | Program &                    |             | Yes      |
 * | cb_handle | ID of the circular buffer, returned by `CreateCircularBuffers`                           | CBHandle (uintptr_t) |             | Yes      |
 * | buffer    | Dynamically allocated L1 buffer that shares address space of circular buffer `cb_handle` | const Buffer &               | L1 buffer   | Yes      |
 */
void UpdateDynamicCircularBufferAddress(Program &program, CBHandle cb_handle, const Buffer &buffer);

/**
 * Initializes semaphore on all cores within core range (inclusive). Each core can have up to four 32B semaphores.
 *
 * Return value: Semaphore address (uint32_t)
 *
 * | Argument      | Description                                          | Type                                                      | Valid Range  | Required |
 * |---------------|------------------------------------------------------|-----------------------------------------------------------|--------------|----------|
 * | program       | The program to which semaphore will be added to      | Program &                                                 |              | Yes      |
 * | core_spec     | Range of the Tensix co-ordinates using the semaphore | const std::variant<CoreRange,CoreRangeSet> &              |              | Yes      |
 * | initial_value | Initial value of the semaphore                       | uint32_t                                                  |              | Yes      |
 */
uint32_t CreateSemaphore(Program &program, const std::variant<CoreRange,CoreRangeSet> &core_spec, uint32_t initial_value);

/**
*  Allocates a DRAM or L1 buffer on device
*
*  Return value: Buffer
*
*  | Argument        | Description                             | Type                     | Valid Range | Required |
*  |-----------------|---------------------------------------- |--------------------------|-------------|----------|
*  | config          | config for buffer                       | BufferConfig             |             | Yes      |
*/
Buffer CreateBuffer(const std::variant<InterleavedBufferConfig, ShardedBufferConfig> & config);

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
// ==================================================

/**
 * Set runtime args for a kernel that are sent to the core during runtime. This API needs to be called to update the runtime args for the kernel.
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                                                   | Valid Range                                                         | Required |
 * |--------------|------------------------------------------------------------------------|--------------------------------------------------------|---------------------------------------------------------------------|----------|
 * | program      | The program containing kernels, circular buffers, semaphores           | const Program &                                        |                                                                     | Yes      |
 * | kernel_id    | ID of the kernel that will receive the runtime args                    | KernelHandle (uint64_t)                                    |                                                                     | Yes      |
 * | core_spec    | Location of Tensix core(s) where the runtime args will be written      | const std::variant<CoreCoord,CoreRange,CoreRangeSet> & | Any logical Tensix core coordinate(s) on which the kernel is placed | Yes      |
 * | runtime_args | The runtime args to be written                                         | const std::vector<uint32_t> &                          |                                                                     | Yes      |
 */
void SetRuntimeArgs(const Program &program, KernelHandle kernel, const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec, const std::vector<uint32_t> &runtime_args);

/**
 * Set multiple runtime arguments of a kernel at once during runtime, each mapping to a specific core. The runtime args for each core may be unique.
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                                                   | Valid Range                                                                | Required |
 * |--------------|------------------------------------------------------------------------|--------------------------------------------------------|----------------------------------------------------------------------------|----------|
 * | program      | The program containing kernels, circular buffers, semaphores           | const Program &                                        |                                                                            | Yes      |
 * | kernel_id    | ID of the kernel that will receive the runtime args                    | KernelHandle (uint64_t)                                    |                                                                            | Yes      |
 * | core_spec    | Location of Tensix core(s) where the runtime args will be written      | const std::vector<CoreCoord> &                         | Any set of logical Tensix core coordinates on which the kernel is placed   | Yes      |
 * | runtime_args | The runtime args to be written                                         | const std::vector< vector<uint32_t> > &                | outer vector size must be equal to size of core_spec vector                | Yes      |
 */
void SetRuntimeArgs(const Program &program, KernelHandle kernel, const std::vector< CoreCoord > & core_spec, const std::vector< std::vector<uint32_t> > &runtime_args);

/**
 * Get the runtime args for a kernel.
 *
 * Return value: std::vector<uint32_t> &
 *
 * | Argument     | Description                                                            | Type                          | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | program      | The program containing kernels, circular buffers, semaphores           | const Program &               |                                    | Yes      |
 * | kernel_id    | ID of the kernel that will receive the runtime args                    | KernelHandle (uint64_t)                |                                    | Yes      |
 * | logical_core | The location of the Tensix core where the runtime args will be written | const CoreCoord &             | Any logical Tensix core coordinate | Yes      |
 */
std::vector<uint32_t>& GetRuntimeArgs(const Program &program, KernelHandle kernel_id, const CoreCoord &logical_core);

/**
 * Reads a buffer from the device
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                          | Valid Range                            | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|----------------------------------------|----------|
 * | cq           | The command queue object which dispatches the command to the hardware  | CommandQueue &                |                                        | Yes      |
 * | buffer       | The device buffer we are reading from                                  | Buffer &                      |                                        | Yes      |
 * | dst          | The vector where the results that are read will be stored              | vector<uint32_t> &            |                                        | Yes      |
 * | blocking     | Whether or not this is a blocking operation                            | bool                          | Only blocking mode supported currently | Yes      |
 */
void EnqueueReadBuffer(CommandQueue& cq, Buffer& buffer, std::vector<uint32_t>& dst, bool blocking);

/**
 * Reads a buffer from the device
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                          | Valid Range                            | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|----------------------------------------|----------|
 * | cq           | The command queue object which dispatches the command to the hardware  | CommandQueue &                |                                        | Yes      |
 * | buffer       | The device buffer we are reading from                                  | Buffer &                      |                                        | Yes      |
 * | dst          | The memory where the result will be stored                             | void*                         |                                        | Yes      |
 * | blocking     | Whether or not this is a blocking operation                            | bool                          | Only blocking mode supported currently | Yes      |
 */
void EnqueueReadBuffer(CommandQueue& cq, Buffer& buffer, void* dst, bool blocking);

/**
 * Writes a buffer to the device
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                          | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | cq           | The command queue object which dispatches the command to the hardware  | CommandQueue &                |                                    | Yes      |
 * | buffer       | The device buffer we are writing to                                    | Buffer &                      |                                    | Yes      |
 * | src          | The vector we are writing to the device                                | vector<uint32_t> &            |                                    | Yes      |
 * | blocking     | Whether or not this is a blocking operation                            | bool                          |                                    | Yes      |
 */
void EnqueueWriteBuffer(CommandQueue& cq, Buffer& buffer, std::vector<uint32_t>& src, bool blocking);

/**
 * Writes a buffer to the device
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                          | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | cq           | The command queue object which dispatches the command to the hardware  | CommandQueue &                |                                    | Yes      |
 * | buffer       | The device buffer we are writing to                                    | Buffer &                      |                                    | Yes      |
 * | src          | The memory we are writing to the device                                | const void*                   |                                    | Yes      |
 * | blocking     | Whether or not this is a blocking operation                            | bool                          |                                    | Yes      |
 */
void EnqueueWriteBuffer(CommandQueue& cq, Buffer& buffer, const void* src, bool blocking);

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

/**
 * Read device side profiler data and dump results into device side CSV log
 *
 * This function only works in PROFILER builds. Please refer to the "Device Program Profiler" section for more information.
 *
 * Return value: void
 *
 * | Argument      | Description                                       | Type            | Valid Range               | Required |
 * |---------------|---------------------------------------------------|-----------------|---------------------------|----------|
 * | device        | The device holding the program being profiled.    | Device *        |                           | True     |
 * | program       | The program being profiled.                       | const Program & |                           | True     |
 * */
void DumpDeviceProfileResults(Device *device, const Program &program);


}  // namespace tt_metal

}  // namespace tt
