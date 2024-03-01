// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>
#include <future>
#include "common/core_coord.h"
#include "tt_metal/impl/program/program.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/event/event.hpp"

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
class Trace;
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
KernelHandle CreateKernel(
    Program &program,
    const std::string &file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig> &config);

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
 * | cb_handle | ID of the circular buffer, returned by `CreateCircularBuffers` | CBHandle (uintptr_t) |       |    Yes      |
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
 * | cb_handle  | ID of the circular buffer, returned by `CreateCircularBuffers` | CBHandle (uintptr_t) |       | Yes         |          |
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
 * | cb_handle | ID of the circular buffer, returned by `CreateCircularBuffers`                           | CBHandle (uintptr_t) |       | Yes         |          |
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
 * | core_type     | Tensix or Ethernet core to create semaphore on.      | CoreType                                                  |              | Yes      |
 */
uint32_t CreateSemaphore(Program &program, const std::variant<CoreRange,CoreRangeSet> &core_spec, uint32_t initial_value, CoreType core_type=CoreType::WORKER);

/**
*  Allocates an interleaved DRAM or L1 buffer on device
*
*  Return value: std::shared_ptr<Buffer>
*
*  | Argument        | Description                             | Type                     | Valid Range | Required |
*  |-----------------|---------------------------------------- |--------------------------|-------------|----------|
*  | config          | config for buffer                       | InterleavedBufferConfig  |             | Yes      |
*/
std::shared_ptr<Buffer> CreateBuffer(const InterleavedBufferConfig &config);

/**
*  Allocates a sharded DRAM or L1 buffer on device
*
*  Return value: std::shared_ptr<Buffer>
*
*  | Argument        | Description                             | Type                     | Valid Range | Required |
*  |-----------------|---------------------------------------- |--------------------------|-------------|----------|
*  | config          | config for buffer                       | ShardedBufferConfig      |             | Yes      |
*/
std::shared_ptr<Buffer> CreateBuffer(const ShardedBufferConfig &config);

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

/**
*  Gives the specified program ownership of the buffer: the buffer will remain on device at least until the program is enqueued. This is required for asynchronous Command Queues.
*
*  Return value: void
*
*  | Argument | Description                                  | Type                           | Valid Range | Required |
*  |----------|----------------------------------------------|--------------------------------|-------------|----------|
*  | buffer   | The buffer that will be owned by the program | std::shared_ptr<Buffer> buffer |             | Yes      |
*  | program  | The program getting ownership of the buffer  | std::shared_ptr<Buffer> buffer |             | Yes      |
*/
void AssignGlobalBufferToProgram(std::shared_ptr<Buffer> buffer, std::variant<std::reference_wrapper<Program>, std::shared_ptr<Program>> program);

// ==================================================
//           COMPILE & EXECUTE KENRNELS
// ==================================================
using RuntimeArgs = std::vector<std::variant<Buffer*, uint32_t>>;
/**
 * Set runtime args for a kernel that are sent to the core during runtime. This API needs to be called to update the runtime args for the kernel.
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                                                   | Valid Range                                                         | Required |
 * |--------------|------------------------------------------------------------------------|--------------------------------------------------------|---------------------------------------------------------------------|----------|
 * | program      | The program containing kernels, circular buffers, semaphores           | const Program &                                        |                                                                     | Yes      |
 * | kernel_id    | ID of the kernel that will receive the runtime args                    | KernelHandle (uint64_t)                                |                                                                     | Yes      |
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
 * | kernel_id    | ID of the kernel that will receive the runtime args                    | KernelHandle (uint64_t)                                |                                                                            | Yes      |
 * | core_spec    | Location of Tensix core(s) where the runtime args will be written      | const std::vector<CoreCoord> &                         | Any set of logical Tensix core coordinates on which the kernel is placed   | Yes      |
 * | runtime_args | The runtime args to be written                                         | const std::vector< vector<uint32_t> > &                | Outer vector size must be equal to size of core_spec vector                | Yes      |
 */
void SetRuntimeArgs(const Program &program, KernelHandle kernel, const std::vector< CoreCoord > & core_spec, const std::vector< std::vector<uint32_t> > &runtime_args);

/**
 * Set runtime args for a kernel that are sent to the specified cores using the command queue. This API must be used when Asynchronous Command Queue Mode is enabled.
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                                                   | Valid Range                                                                | Required |
 * |--------------|------------------------------------------------------------------------|--------------------------------------------------------|----------------------------------------------------------------------------|----------|
 * | device       | The device that runtime args are being written to.                     | Device*                                                |                                                                            | Yes      |
 * | kernel       | The kernel that will recieve these runtime args.                       | std::shared_ptr<Kernel>                                |                                                                            | Yes      |
 * | core_spec    | Location of Tensix core(s) where the runtime args will be written      | const std::variant<CoreCoord,CoreRange,CoreRangeSet> & | Any set of logical Tensix core coordinates on which the kernel is placed   | Yes      |
 * | runtime_args | The runtime args to be written                                         | std::shared_ptr<RuntimeArgs>                           |                                                                            | Yes      |
*/
void SetRuntimeArgs(Device* device, const std::shared_ptr<Kernel> kernel, const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec, std::shared_ptr<RuntimeArgs> runtime_args);

/**
 * Set multiple runtime arguments of a kernel using the command queue. Each core can have distinct arguments. This API must be used when Asynchronous Command Queue Mode is enabled.
 *
 * Return value: void
 * | Argument     | Description                                                            | Type                                                   | Valid Range                                                                | Required |
 * |--------------|------------------------------------------------------------------------|--------------------------------------------------------|----------------------------------------------------------------------------|----------|
 * | device       | The device that runtime args are being written to.                     | Device*                                                |                                                                            | Yes      |
 * | kernel       | The kernel that will recieve these runtime args.                       | std::shared_ptr<Kernel>                                |                                                                            | Yes      |
 * | core_spec    | Location of Tensix core(s) where the runtime args will be written      | const std::vector< CoreCoord > &                       | Any set of logical Tensix core coordinates on which the kernel is placed   | Yes      |
 * | runtime_args | The runtime args to be written                                         | const std::vector<std::shared_ptr<RuntimeArgs>>        | Outer vector size must be equal to size of core_spec vector                | Yes      |
 */
void SetRuntimeArgs(Device* device, const std::shared_ptr<Kernel> kernel, const std::vector< CoreCoord > & core_spec, const std::vector<std::shared_ptr<RuntimeArgs>> runtime_args);

/**
 * Get the runtime args for a kernel.
 *
 * Return value: std::vector<uint32_t> &
 *
 * | Argument     | Description                                                            | Type                          | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | program      | The program containing kernels, circular buffers, semaphores           | const Program &               |                                    | Yes      |
 * | kernel_id    | ID of the kernel that will receive the runtime args                    | KernelHandle (uint64_t)       |                                    | Yes      |
 * | logical_core | The location of the Tensix core where the runtime args will be written | const CoreCoord &             | Any logical Tensix core coordinate | Yes      |
 */
std::vector<uint32_t>& GetRuntimeArgs(const Program &program, KernelHandle kernel_id, const CoreCoord &logical_core);

/**
 * Update specific entries of the runtime args vector for a kernel using the command queue. This API must be used when Asynchronous Command Queue Mode is enabled.
 *
 * Return Value: void
 *
 * | Argument     | Description                                                            | Type                          | Valid Range                                                  | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|--------------------------------------------------------------|----------|
 * | cq           | The command queue used to send the runtime args update                 | CommandQueue &                |                                                              | Yes      |
 * | kernel       | The kernel for which the runtime args must be updated                  | std::shared_ptr<Kernel>       |                                                              | Yes      |
 * | core_coord   | The core receiving the runtime args update                             | const CoreCoord &             | A single core running the kernel                             | Yes      |
 * | update_idx   | The runtime arg vector indices that must be updated                    | std::vector<uint32_t> &       | Each index in this vector must be less than num runtime args | Yes      |
 * | runtime_args | Updated runtime args                                                   | std::shared_ptr<RuntimeArgs>  | 1:1 Mapping between each entry and the indices in update_idx | Yes      |
 */
void UpdateRuntimeArgs(Device* device, const std::shared_ptr<Kernel> kernel, const CoreCoord &core_coord, std::vector<uint32_t> &update_idx, std::shared_ptr<RuntimeArgs> runtime_args);

/**
 * Reads a buffer from the device
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                                | Valid Range                            | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------------|----------------------------------------|----------|
 * | cq           | The command queue object which dispatches the command to the hardware  | CommandQueue &                      |                                        | Yes      |
 * | buffer       | The device buffer we are reading from                                  | Buffer & or std::shared_ptr<Buffer> |                                        | Yes      |
 * | dst          | The vector where the results that are read will be stored              | vector<uint32_t> &                  |                                        | Yes      |
 * | blocking     | Whether or not this is a blocking operation                            | bool                                | Only blocking mode supported currently | Yes      |
 */
void EnqueueReadBuffer(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> > buffer, std::vector<uint32_t>& dst, bool blocking);

/**
 * Reads a buffer from the device
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                                | Valid Range                            | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------------|----------------------------------------|----------|
 * | cq           | The command queue object which dispatches the command to the hardware  | CommandQueue &                      |                                        | Yes      |
 * | buffer       | The device buffer we are reading from                                  | Buffer & or std::shared_ptr<Buffer> |                                        | Yes      |
 * | dst          | The memory where the result will be stored                             | void*                               |                                        | Yes      |
 * | blocking     | Whether or not this is a blocking operation                            | bool                                | Only blocking mode supported currently | Yes      |
 */
void EnqueueReadBuffer(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> > buffer, void * dst, bool blocking);

/**
 * Writes a buffer to the device
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                                | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------------|------------------------------------|----------|
 * | cq           | The command queue object which dispatches the command to the hardware  | CommandQueue &                      |                                    | Yes      |
 * | buffer       | The device buffer we are writing to                                    | Buffer & or std::shared_ptr<Buffer> |                                    | Yes      |
 * | src          | The vector we are writing to the device                                | vector<uint32_t> &                  |                                    | Yes      |
 * | blocking     | Whether or not this is a blocking operation                            | bool                                |                                    | Yes      |
 */
void EnqueueWriteBuffer(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> > buffer, std::vector<uint32_t>& src, bool blocking);

/**
 * Writes a buffer to the device
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                                | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------------|------------------------------------|----------|
 * | cq           | The command queue object which dispatches the command to the hardware  | CommandQueue &                      |                                    | Yes      |
 * | buffer       | The device buffer we are writing to                                    | Buffer & or std::shared_ptr<Buffer> |                                    | Yes      |
 * | src          | The memory we are writing to the device                                | HostDataType                        |                                    | Yes      |
 * | blocking     | Whether or not this is a blocking operation                            | bool                                |                                    | Yes      |
 */
void EnqueueWriteBuffer(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> > buffer, HostDataType src, bool blocking);

/**
 * Writes a program to the device and launches it
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                               | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|------------------------------------|------------------------------------|----------|
 * | cq           | The command queue object which dispatches the command to the hardware  | CommandQueue &                     |                                    | Yes      |
 * | program      | The program that will be executed on the device that cq is bound to    | Program &                          |                                    | Yes      |
 * | blocking     | Whether or not this is a blocking operation                            | bool                               |                                    | Yes      |
 */
void EnqueueProgram(CommandQueue& cq, std::variant<std::reference_wrapper<Program>, std::shared_ptr<Program> > program, bool blocking);

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
 * Begins capture on a trace, when the trace is in capture mode all programs pushed into the trace queue will have their execution delayed until the trace is instantiated and enqueued.
 * The capture must be later ended via EndTrace, and can be instantiated via InstantiateTrace on a device command queue, and finally scheduled to be executed via EnqueueTrace.
 *
 * Return value: CommandQueue&
 *
 * | Argument     | Description                                                            | Type                          | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | trace        | Trace in which to initiate the capture                                 | Trace &                       |                                    | Yes      |
*/
CommandQueue& BeginTrace(Trace &trace);

/**
 * Completes capture on a trace, if captured commands do not conform to the rules of the trace, the trace will be invalidated.
 * This trace can later be instantiated via InstantiateTrace on a device command queue, and enqueued for execution via EnqueueTrace on the same device command queue.
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                          | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | trace        | Trace in which to end the capture                                      | Trace &                       |                                    | Yes      |
 */
void EndTrace(Trace &trace);

/**
 * Instantiates a trace on a device command queue, triggering the staging of traced commands and data to the device.
 * Staging is a blocking operation and must be completed before the trace can be enqueued for exeuction. A unique trace instance id is returned
 *
 * Return value: uint32_t
 *
 * | Argument     | Description                                                            | Type                          | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | trace        | The trace object to instantiate                                        | Trace &                       |                                    | Yes      |
 * | cq           | The device command queue on which to instantiate the trace             | CommandQueue &                |                                    | Yes      |
*/
uint32_t InstantiateTrace(Trace &trace, CommandQueue &cq);

/**
 * Enqueues a trace of previously generated commands and data.
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                          | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | cq           | The command queue object which dispatches the command to the hardware  | CommandQueue &                |                                    | Yes      |
 * | trace_id     | A unique id representing an existing on-device trace, which has been   | uint32_t                      |                                    | Yes      |
 * |              | instantiated via InstantiateTrace where the trace_id is returned       |                               |                                    |          |
 * | blocking     | Whether or not this is a blocking operation                            | bool                          |                                    | Yes      |
 */
void EnqueueTrace(CommandQueue &cq, uint32_t trace_id, bool blocking);

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

/**
 * Enqueues a command to record an Event on the device for a given CQ, and updates the Event object for the user.
 * Return value: void
 * | Argument     | Description                                                            | Type                          | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | cq           | The command queue object which dispatches the command to the hardware  | CommandQueue &                |                                    | Yes      |
 * | event        | An event that will be populated by this function, and inserted in CQ   | std::shared_ptr<Event>        |                                    | Yes      |
 */
void EnqueueRecordEvent(CommandQueue& cq, std::shared_ptr<Event> event);

/**
 * Enqueues a command on the device for a given CQ (non-blocking). The command on device will block and wait for completion of the specified event (which may be in another CQ).
 * Return value: void
 * | Argument     | Description                                                            | Type                          | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | cq           | The command queue object which dispatches the command to the hardware  | CommandQueue &                |                                    | Yes      |
 * |              | and waits for the event to complete.                                   |                               |                                    |          |
 * | event        | The event object that this CQ will wait on for completion.             | std::shared_ptr<Event>        |                                    | Yes      |
 */
void EnqueueWaitForEvent(CommandQueue& cq, std::shared_ptr<Event> event);

/**
 * Blocking function for host to synchronize (wait) on an event completion on device.
 * Return value: void
 * | Argument     | Description                                                            | Type                          | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | event        | The event object that host will wait on for completion.                | std::shared_ptr<Event>        |                                    | Yes      |
 */
void EventSynchronize(std::shared_ptr<Event> event);

}  // namespace tt_metal

}  // namespace tt
