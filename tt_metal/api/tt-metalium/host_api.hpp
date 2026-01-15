// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <initializer_list>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/runtime_args_data.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/profiler_types.hpp>
#include <tt-metalium/profiler_optional_metadata.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>

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

namespace tt::tt_metal {

struct TraceDescriptor;

class Program;
class IDevice;
class Trace;
struct Event;
class Buffer;
class GlobalSemaphore;
class CoreRange;
class CoreRangeSet;

// ==================================================
//                  HOST API: Device management
// ==================================================

// clang-format off
/**
 * Sets the root directory for TT Metal meta data files like kernel sources.
 *
 * Return value: void
 *
 * | Argument  | Description                                 | Type                | Valid range | Required |
 * |-----------|---------------------------------------------|---------------------|-------------|----------|
 * | root_dir  | Path to the root directory                  | const std::string & |             | No       |
 */
// clang-format on
void SetRootDir(const std::string& root_dir);

/**
 * Returns number of Tenstorrent devices that can be targeted
 *
 * Return value: size_t
 */
size_t GetNumAvailableDevices();

/**
 * Returns whether Tenstorrent devices are in a Galaxy cluster
 *
 * Return value: bool
 */
bool IsGalaxyCluster();

/**
 * Returns number of Tenstorrent devices that are connected to host via PCIe and can be targeted
 *
 * Return value: size_t
 */
size_t GetNumPCIeDevices();

ChipId GetPCIeDeviceID(ChipId device_id);

// clang-format off
/**
 * Instantiates a device object.
 *
 * Return value: IDevice*
 *
 * | Argument   | Description                | Type            | Valid Range                       | Required |
 * |------------|----------------------------|-----------------|-----------------------------------|----------|
 * | device_id  | ID of the device to target| ChipId (int) | 0 to (GetNumAvailableDevices - 1) | Yes      |
 * */
// clang-format on
IDevice* CreateDevice(
    ChipId device_id,
    uint8_t num_hw_cqs = 1,
    size_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
    size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
    const DispatchCoreConfig& dispatch_core_config = DispatchCoreConfig{},
    const std::vector<uint32_t>& l1_bank_remap = {},
    size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE);

// clang-format off
/**
 * Instantiates a device with minimal setup, used to attach to a device in a bad state.
 *
 * Return value: IDevice*
 *
 * | Argument   | Description                | Type            | Valid Range                       | Required |
 * |------------|----------------------------|-----------------|-----------------------------------|----------|
 * | device_id  | ID of the device to target| ChipId (int) | 0 to (GetNumAvailableDevices - 1) | Yes      |
 * */
// clang-format on
IDevice* CreateDeviceMinimal(
    ChipId device_id, uint8_t num_hw_cqs = 1, const DispatchCoreConfig& dispatch_core_config = DispatchCoreConfig{});

// clang-format off
/**
 * Resets device and closes device
 *
 * Return value: bool
 *
 * | Argument | Description                | Type     | Valid Range | Required |
 * |----------|----------------------------|----------|-------------|----------|
 * | device   | Pointer to a device object | IDevice* |             | True     |
 */
// clang-format on
bool CloseDevice(IDevice* device);

// ==================================================
//                  HOST API: program & kernels
// ==================================================

// clang-format off
/**
 * Creates a Program object which is the main container that bundles kernels, circular buffers, and/or semaphores for execution on device
 *
 * Return value: Program
 */
// clang-format on
Program CreateProgram();

// clang-format off
/**
 * Creates a data movement kernel with no compile time arguments and adds it to the program.
 *
 * Return value: Kernel ID (uintptr_t)
 *
 * | Argument     | Description                                                                                                                                 | Type                                                     | Valid Range | Required |
 * |--------------|---------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------|-------------|----------|
 * | program      | The program to which this kernel will be added to                                                                                           | Program &                                                |             | Yes      |
 * | file_name    | Path to kernel src. Assumed to be absolute/relative to CWD, but will fall back to relative path from TT_METAL_HOME/TT_METAL_KERNEL_PATH.    | const std::string &                                      |             | Yes      |
 * | core_spec    | Either a single logical core, a range of logical cores or a set of logical core ranges that indicate which cores kernel is placed on        | const std::variant<CoreCoord, CoreRange, CoreRangeSet> & |             | Yes      |
 * | config       | Config for data movement or compute kernel                                                                                                  | const std::variant<DataMovementConfig,ComputeConfig,EthernetConfig> &   |             | No       |
 */
// clang-format on
KernelHandle CreateKernel(
    Program& program,
    const std::string& file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config);

// clang-format off
/**
 * Creates a compute or data movement kernel with the given compile time arguments and adds it to the program.
 *
 * Return value: Kernel ID (uintptr_t)
 *
 * | Argument           | Description                                                                                                                          | Type                                                     | Valid Range | Required |
 * |--------------------|--------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------|-------------|----------|
 * | program            | The program to which this kernel will be added to                                                                                    | Program &                                                |             | Yes      |
 * | kernel_src_code    | Source code for kernel                                                                                                               | const std::string &                                      |             | Yes      |
 * | core_spec          | Either a single logical core, a range of logical cores or a set of logical core ranges that indicate which cores kernel is placed on | const std::variant<CoreCoord, CoreRange, CoreRangeSet> & |             | Yes      |
 * | config             | Config for data movement or compute kernel                                                                                           | const std::variant<DataMovementConfig,ComputeConfig,EthernetConfig> &   |             | No       |
 */
// clang-format on
KernelHandle CreateKernelFromString(
    Program& program,
    const std::string& kernel_src_code,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config);

// clang-format off
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
// clang-format on
CBHandle CreateCircularBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const CircularBufferConfig& config);

// clang-format off
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
// clang-format on
const CircularBufferConfig& GetCircularBufferConfig(Program& program, CBHandle cb_handle);

// clang-format off
/**
 * Update the total size of the circular buffer at the given circular buffer handle. Updating a program-local circular buffer requires all circular buffers in the program to be reallocated.
 * If it is required to update the address and total size of a dynamic circular buffer, use `UpdateDynamicCircularBufferAddressAndTotalSize`.
 *
 * Return value: void
 *
 * | Argument   | Description                                                    | Type                         | Valid Range | Required |
 * |------------|----------------------------------------------------------------|------------------------------|-------------|----------|
 * | program    | The program containing the circular buffer                     | Program &                    |             | Yes      |
 * | cb_handle  | ID of the circular buffer, returned by `CreateCircularBuffers` | CBHandle (uintptr_t) |       | Yes         |          |
 * | total_size | New size of the circular buffer in bytes                       | uint32_t                     |             | Yes      |
*/
// clang-format on
void UpdateCircularBufferTotalSize(Program& program, CBHandle cb_handle, uint32_t total_size);

// clang-format off
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
// clang-format on
void UpdateCircularBufferPageSize(Program& program, CBHandle cb_handle, uint8_t buffer_index, uint32_t page_size);

// clang-format off
/**
 * Update the address of a dynamic circular buffer. Dynamic circular buffers share the same address space as L1 buffers.
 * If it is required to update the address and total size of a dynamic circular buffer, use `UpdateDynamicCircularBufferAddressAndTotalSize`.
 *
 * Return value: void
 *
 * | Argument  | Description                                                                              | Type                         | Valid Range | Required |
 * |-----------|------------------------------------------------------------------------------------------|------------------------------|-------------|----------|
 * | program   | The program containing the circular buffer                                               | Program &                    |             | Yes      |
 * | cb_handle | ID of the circular buffer, returned by `CreateCircularBuffers`                           | CBHandle (uintptr_t) |       | Yes         |          |
 * | buffer    | Dynamically allocated L1 buffer that shares address space of circular buffer `cb_handle` | const Buffer &               | L1 buffer   | Yes      |
 */
// clang-format on
void UpdateDynamicCircularBufferAddress(Program& program, CBHandle cb_handle, const Buffer& buffer);

// clang-format off
/**
 * Update the address and total size of a dynamic circular buffer. Dynamic circular buffers share the same address space as L1 buffers.
 *
 * Return value: void
 *
 * | Argument   | Description                                                                              | Type                         | Valid Range | Required |
 * |------------|------------------------------------------------------------------------------------------|------------------------------|-------------|----------|
 * | program    | The program containing the circular buffer                                               | Program &                    |             | Yes      |
 * | cb_handle  | ID of the circular buffer, returned by `CreateCircularBuffers`                           | CBHandle (uintptr_t) |       | Yes         |          |
 * | buffer     | Dynamically allocated L1 buffer that shares address space of circular buffer `cb_handle` | const Buffer &               | L1 buffer   | Yes      |
 * | total_size | New size of the circular buffer in bytes                                                 | uint32_t                     |             | Yes      |
 */
// clang-format on
void UpdateDynamicCircularBufferAddressAndTotalSize(
    Program& program, CBHandle cb_handle, const Buffer& buffer, uint32_t total_size);

// clang-format off
/**
 * Initializes semaphore on all cores within core range (inclusive). Each core can have up to eight 4B semaphores aligned to L1_ALIGNMENT.
 *
 * Return value: Semaphore id (uint32_t). This can be used inside a kernel to extract the address using get_semaphore
 *
 * | Argument      | Description                                          | Type                                                      | Valid Range  | Required |
 * |---------------|------------------------------------------------------|-----------------------------------------------------------|--------------|----------|
 * | program       | The program to which semaphore will be added to      | Program &                                                 |              | Yes      |
 * | core_spec     | Range of the Tensix co-ordinates using the semaphore | const std::variant<CoreRange,CoreRangeSet> &              |              | Yes      |
 * | initial_value | Initial value of the semaphore                       | uint32_t                                                  |              | Yes      |
 * | core_type     | Tensix or Ethernet core to create semaphore on.      | CoreType                                                  |              | No       |
 */
// clang-format on
uint32_t CreateSemaphore(
    Program& program,
    const std::variant<CoreRange, CoreRangeSet>& core_spec,
    uint32_t initial_value,
    CoreType core_type = CoreType::WORKER);

// clang-format off
/**
 * Initializes a global semaphore on all cores within the specified CoreRangeSet.
 * This only supports tensix cores, and can only use L1 buffer types like BufferType::L1 and BufferType::L1_SMALL.
 *
 * Return value: GlobalSemaphore
 *
 * | Argument       | Description                                            | Type                                                      | Valid Range  | Required |
 * |----------------|--------------------------------------------------------|-----------------------------------------------------------|--------------|----------|
 * | device         | The device to create the semaphore on                  | IDevice*                                                  |              | Yes      |
 * | cores          | Range of the Tensix co-ordinates using the semaphore   | const CoreRangeSet &                                      |              | Yes      |
 * | initial_value  | Initial value of the semaphore                         | uint32_t                                                  |              | Yes      |
 * | buffer_type    | Buffer type to store the semaphore                     | BufferType                                                | L1 types     | No       |
 */
// clang-format on
GlobalSemaphore CreateGlobalSemaphore(
    IDevice* device, const CoreRangeSet& cores, uint32_t initial_value, BufferType buffer_type = BufferType::L1);

// clang-format off
/**
 * Initializes a global semaphore on all cores within the specified CoreRangeSet.
 * This only supports tensix cores, and can only use L1 buffer types like BufferType::L1 and BufferType::L1_SMALL.
 *
 * Return value: GlobalSemaphore
 *
 * | Argument       | Description                                            | Type                                                      | Valid Range  | Required |
 * |----------------|--------------------------------------------------------|-----------------------------------------------------------|--------------|----------|
 * | device         | The device to create the semaphore on                  | IDevice*                                                  |              | Yes      |
 * | cores          | Range of the Tensix co-ordinates using the semaphore   | CoreRangeSet &&                                           |              | Yes      |
 * | initial_value  | Initial value of the semaphore                         | uint32_t                                                  |              | Yes      |
 * | buffer_type    | Buffer type to store the semaphore                     | BufferType                                                | L1 types     | No       |
 */
// clang-format on
GlobalSemaphore CreateGlobalSemaphore(
    IDevice* device, CoreRangeSet&& cores, uint32_t initial_value, BufferType buffer_type = BufferType::L1);

// clang-format off
/**
*  Creates a pre-allocated interleaved DRAM or L1 buffer with the global allocator on device
*
*  Return value: std::shared_ptr<Buffer>
*
*  | Argument        | Description                                                       | Type                      | Valid Range | Required |
*  |-----------------|------------------------------------------------------------------ |---------------------------|-------------|----------|
*  | config          | Config for the buffer                                             | InterleavedBufferConfig   |             | Yes      |
*/
// clang-format on
std::shared_ptr<Buffer> CreateBuffer(const InterleavedBufferConfig& config);

// clang-format off
/**
*  Creates a pre-allocated interleaved DRAM or L1 buffer with the global allocator on device
*
*  Return value: std::shared_ptr<Buffer>
*
*  | Argument        | Description                                                       | Type                      | Valid Range | Required |
*  |-----------------|------------------------------------------------------------------ |---------------------------|-------------|----------|
*  | config          | Config for the buffer                                             | InterleavedBufferConfig   |             | Yes      |
*  | address         | Device address of the buffer                                      | DeviceAddr                |             | No       |
*/
// clang-format on
std::shared_ptr<Buffer> CreateBuffer(const InterleavedBufferConfig& config, DeviceAddr address);

// clang-format off
/**
*  Creates a pre-allocated interleaved DRAM or L1 buffer on device
*
*  Return value: std::shared_ptr<Buffer>
*
*  | Argument        | Description                                                       | Type                      | Valid Range | Required |
*  |-----------------|------------------------------------------------------------------ |---------------------------|-------------|----------|
*  | config          | Config for the buffer                                             | InterleavedBufferConfig   |             | Yes      |
*  | sub_device_id   | The sub-device id to allocate on                                  | SubDeviceId               |             | No       |
*/
// clang-format on
std::shared_ptr<Buffer> CreateBuffer(const InterleavedBufferConfig& config, SubDeviceId sub_device_id);

// clang-format off
/**
*  Creates a pre-allocated sharded DRAM or L1 buffer with the global allocator on device
*
*  Return value: std::shared_ptr<Buffer>
*
*  | Argument        | Description                                                       | Type                      | Valid Range | Required |
*  |-----------------|------------------------------------------------------------------ |---------------------------|-------------|----------|
*  | config          | Config for the buffer                                             | ShardedBufferConfig       |             | Yes      |
*/
// clang-format on
std::shared_ptr<Buffer> CreateBuffer(const ShardedBufferConfig& config);

// clang-format off
/**
*  Creates a pre-allocated sharded DRAM or L1 buffer with the global allocator on device
*
*  Return value: std::shared_ptr<Buffer>
*
*  | Argument        | Description                                                       | Type                      | Valid Range | Required |
*  |-----------------|------------------------------------------------------------------ |---------------------------|-------------|----------|
*  | config          | Config for the buffer                                             | ShardedBufferConfig       |             | Yes      |
*  | address         | Device address of the buffer                                      | DeviceAddr                |             | No       |
*/
// clang-format on
std::shared_ptr<Buffer> CreateBuffer(const ShardedBufferConfig& config, DeviceAddr address);

// clang-format off
/**
*  Creates a pre-allocated sharded DRAM or L1 buffer on device
*
*  Return value: std::shared_ptr<Buffer>
*
*  | Argument        | Description                                                       | Type                      | Valid Range | Required |
*  |-----------------|------------------------------------------------------------------ |---------------------------|-------------|----------|
*  | config          | Config for the buffer                                             | ShardedBufferConfig       |             | Yes      |
*  | sub_device_id   | The sub-device id to allocate on                                  |                           |             | No       |
*/
// clang-format on
std::shared_ptr<Buffer> CreateBuffer(const ShardedBufferConfig& config, SubDeviceId sub_device_id);

// clang-format off
/**
*  Deallocates buffer from device by marking its memory as free.
*
*  Return value: void
*
*  | Argument | Description                          | Type     | Valid Range | Required |
*  |----------|--------------------------------------|----------|-------------|----------|
*  | buffer   | The buffer to deallocate from device | Buffer & |             | Yes      |
*/
// clang-format on
void DeallocateBuffer(Buffer& buffer);

// clang-format off
/**
*  Gives the specified program ownership of the buffer: the buffer will remain on device at least until the program is enqueued. This is required for asynchronous Command Queues.
*
*  Return value: void
*
*  | Argument | Description                                  | Type                           | Valid Range | Required |
*  |----------|----------------------------------------------|--------------------------------|-------------|----------|
*  | buffer   | The buffer that will be owned by the program | std::shared_ptr<Buffer> buffer |             | Yes      |
*  | program  | The program getting ownership of the buffer  | Program &                      |             | Yes      |
*/
// clang-format on
void AssignGlobalBufferToProgram(const std::shared_ptr<Buffer>& buffer, Program& program);

// clang-format off
// ==================================================
//           COMPILE & EXECUTE KENRNELS
// ==================================================
/**
 * Set runtime args for a kernel that are sent to the core during runtime. This API needs to be called to update the runtime args for the kernel.
 * Maximum of 341 allowed runtime args per core (unique and common runtime args count toward same limit).
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                                                   | Valid Range                                                         | Required |
 * |--------------|------------------------------------------------------------------------|--------------------------------------------------------|---------------------------------------------------------------------|----------|
 * | program      | The program containing kernels, circular buffers, semaphores           | const Program &                                        |                                                                     | Yes      |
 * | kernel_id    | ID of the kernel that will receive the runtime args                    | KernelHandle (uint64_t)                                |                                                                     | Yes      |
 * | core_spec    | Location of Tensix core(s) where the runtime args will be written      | const std::variant<CoreCoord,CoreRange,CoreRangeSet> & | Any logical Tensix core coordinate(s) on which the kernel is placed | Yes      |
 * | runtime_args | The runtime args to be written                                         | stl::Span<const uint32_t>                              |                                                                     | Yes      |
 */
// clang-format on
void SetRuntimeArgs(
    const Program& program,
    KernelHandle kernel,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    stl::Span<const uint32_t> runtime_args);

// clang-format off
/**
 * Set runtime args for a kernel that are sent to the core during runtime. This API needs to be called to update the runtime args for the kernel.
 * Maximum of 255 allowed runtime args per core (unique and common runtime args count toward same limit).
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                                                   | Valid Range                                                         | Required |
 * |--------------|------------------------------------------------------------------------|--------------------------------------------------------|---------------------------------------------------------------------|----------|
 * | program      | The program containing kernels, circular buffers, semaphores           | const Program &                                        |                                                                     | Yes      |
 * | kernel_id    | ID of the kernel that will receive the runtime args                    | KernelHandle (uint64_t)                                |                                                                     | Yes      |
 * | core_spec    | Location of Tensix core(s) where the runtime args will be written      | const std::variant<CoreCoord,CoreRange,CoreRangeSet> & | Any logical Tensix core coordinate(s) on which the kernel is placed | Yes      |
 * | runtime_args | The runtime args to be written                                         | initializer_list<uint32_t>                       |                                                                     | Yes      |
 */
// clang-format on
void SetRuntimeArgs(
    const Program& program,
    KernelHandle kernel,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    std::initializer_list<uint32_t> runtime_args);

// clang-format off
/**
 * Set multiple runtime arguments of a kernel at once during runtime, each mapping to a specific core. The runtime args for each core may be unique.
 * Maximum of 341 allowed runtime args per core (unique and common runtime args count toward same limit).
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
// clang-format on
void SetRuntimeArgs(
    const Program& program,
    KernelHandle kernel,
    const std::vector<CoreCoord>& core_spec,
    const std::vector<std::vector<uint32_t>>& runtime_args);

// clang-format off
/**
 * Set common (shared by all cores) runtime args for a kernel that are sent to all cores during runtime. This API needs to be called to update the common runtime args for the kernel.
 * Maximum of 341 allowed runtime args per core (unique and common runtime args count toward same limit).
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                                                   | Valid Range                                                         | Required |
 * |--------------|------------------------------------------------------------------------|--------------------------------------------------------|---------------------------------------------------------------------|----------|
 * | program      | The program containing kernels, circular buffers, semaphores           | const Program &                                        |                                                                     | Yes      |
 * | kernel_id    | ID of the kernel that will receive the runtime args                    | KernelHandle (uint64_t)                                |                                                                     | Yes      |
 * | runtime_args | The runtime args to be written                                         | stl::Span<const uint32_t>                              |                                                                     | Yes      |
 */
// clang-format on
void SetCommonRuntimeArgs(const Program& program, KernelHandle kernel_id, stl::Span<const uint32_t> runtime_args);

// clang-format off
/**
 * Set common (shared by all cores) runtime args for a kernel that are sent to all cores during runtime. This API needs to be called to update the common runtime args for the kernel.
 * Maximum of 341 allowed runtime args per core (unique and common runtime args count toward same limit).
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                                                   | Valid Range                                                         | Required |
 * |--------------|------------------------------------------------------------------------|--------------------------------------------------------|---------------------------------------------------------------------|----------|
 * | program      | The program containing kernels, circular buffers, semaphores           | const Program &                                        |                                                                     | Yes      |
 * | kernel_id    | ID of the kernel that will receive the runtime args                    | KernelHandle (uint64_t)                                |                                                                     | Yes      |
 * | runtime_args | The runtime args to be written                                         | std::initializer_list<uint32_t>                  |                                                                     | Yes      |
 */
// clang-format on
void SetCommonRuntimeArgs(const Program& program, KernelHandle kernel_id, std::initializer_list<uint32_t> runtime_args);

// clang-format off
/**
 * Get the runtime args for a kernel.
 *
 * Return value: uint32_t *
 *
 * | Argument     | Description                                                            | Type                          | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | program      | The program containing kernels, circular buffers, semaphores           | const Program &               |                                    | Yes      |
 * | kernel_id    | ID of the kernel that will receive the runtime args                    | KernelHandle (uint64_t)       |                                    | Yes      |
 * | logical_core | The location of the Tensix core where the runtime args will be written | const CoreCoord &             | Any logical Tensix core coordinate | Yes      |
 */
// clang-format on
RuntimeArgsData& GetRuntimeArgs(const Program& program, KernelHandle kernel_id, const CoreCoord& logical_core);

// clang-format off
/**
 * Get the runtime args for a kernel.
 *
 * Return value: std::vector< std::vector< RuntimeArgsData > > &
 *
 * | Argument     | Description                                                            | Type                          | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | program      | The program containing kernels, circular buffers, semaphores           | const Program &               |                                    | Yes      |
 * | kernel_id    | ID of the kernel that will receive the runtime args                    | KernelHandle (uint64_t)       |                                    | Yes      |
 */
// clang-format on
std::vector<std::vector<RuntimeArgsData>>& GetRuntimeArgs(const Program& program, KernelHandle kernel_id);

// clang-format off
/**
 * Get the common runtime args for a kernel.
 *
 * Return value: RuntimeArgsData &
 *
 * | Argument     | Description                                                            | Type                          | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | program      | The program containing kernels, circular buffers, semaphores           | const Program &               |                                    | Yes      |
 * | kernel_id    | ID of the kernel that will receive the runtime args                    | KernelHandle (uint64_t)       |                                    | Yes      |
 */
// clang-format on
RuntimeArgsData& GetCommonRuntimeArgs(const Program& program, KernelHandle kernel_id);

// clang-format off
/**
 * Read device side profiler data for all devices in the mesh device
 *
 * This function only works in PROFILER builds. Please refer to the "Device Program Profiler" section for more information.
 *
 * Return value: void
 *
 * | Argument      | Description                                           | Type                     | Valid Range               | Required |
 * |---------------|-------------------------------------------------------|--------------------------|---------------------------|----------|
 * | mesh_device   | The mesh device containing the devices to be profiled | MeshDevice&              |                           | Yes      |
 * | state         | The state to use for this profiler read               | ProfilerReadState        |                           | No       |
 * | metadata      | Metadata to include in the profiler results           | ProfilerOptionalMetadata |                           | No       |
 * */
// clang-format on
void ReadMeshDeviceProfilerResults(
    distributed::MeshDevice& mesh_device,
    ProfilerReadState state = ProfilerReadState::NORMAL,
    const std::optional<ProfilerOptionalMetadata>& metadata = {});

// clang-format off
/**
 * Host will query an event for completion status on device.
 * Return value: bool.  True if event is completed, false otherwise.
 * | Argument     | Description                                                            | Type                          | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | event        | The event object that host will query for completion.                  | std::shared_ptr<Event>        |                                    | Yes      |
 */
// clang-format on
bool EventQuery(const std::shared_ptr<Event>& event);

// clang-format off
/**
 * Push the current command queue id to the stack.
 * Return value: void
 * | Argument     | Description                                                                       | Type                          | Valid Range                        | Required |
 * |--------------|-----------------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | cq_id        | The command queue id to push.                                                     | uint8_t                       |                                    | Yes      |
 */
// clang-format on
void PushCurrentCommandQueueIdForThread(uint8_t cq_id);

// clang-format off
/**
 * Pop the current command queue id from the stack.
 * Return value: uint8_t
 */
// clang-format on
uint8_t PopCurrentCommandQueueIdForThread();

// clang-format off
/**
 * Get the current command queue id.
 * Return value: uint8_t
 */
// clang-format on
uint8_t GetCurrentCommandQueueIdForThread();

}  // namespace tt::tt_metal
