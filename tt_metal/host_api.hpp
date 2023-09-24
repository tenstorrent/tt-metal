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
 * | Argument       | Description                                                      | Data type | Valid range                                         | required |
 * |----------------|------------------------------------------------------------------|-----------|-----------------------------------------------------|----------|
 * | device_id      | ID of device to target                                           | int       | 0 to (Device::detect_num_available_devices - 1)     | Yes      |
 * */
Device *CreateDevice(int device_id, const std::vector<uint32_t>& l1_bank_remap = {});

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
 * Creates a Circular Buffer (CBs) in L1 memory at specified address and core and adds it to the program. L1 allocator reserves size_in_bytes bytes at manually specified addresses.
 *
 * Return value: Circular Buffer ID (uintptr_t)
 *
 * | Argument      | Description                                                                    | Type               | Valid Range                             | Required |
 * |---------------|--------------------------------------------------------------------------------|--------------------|-----------------------------------------|----------|
 * | program       | The program to which buffer will be added to.                                  | Program &          |                                         | True     |
 * | buffer_index  | The index/ID of the CB.                                                        | uint32_t           | 0 to 32 DOX-TODO: specify more detail here. | True     |
 * | core          | The location of the Tensix core on which the CB will reside (logical co-ordinates) | const CoreCoord & | DOX-TODO: { , } –> { , }                    | True     |
 * | num_tiles     | Total number of tiles to be stored in the CB                                   | uint32_t           | DOX-TODO: range?                            | True     |
 * | size_in_bytes | Size of CB buffer in Bytes                                                     | uint32_t           | 0 to 1 MB (DOX-TODO: in Bytes)              | True     |
 * | data_format   | The format of the data to be stored in the CB                                  | DataFormat enum    | DataFormat::Float16_b                   | True     |
 * | l1_address    | Address at which the CB buffer will reside                                     | optional<uint32_t>           | 200 kB to 1MB (DOX-TODO: in bytes)          | False     |
 */
CircularBufferID CreateCircularBuffer(
    Program &program,
    uint32_t buffer_index,
    const CoreCoord &core,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    DataFormat data_format,
    std::optional<uint32_t> l1_address = std::nullopt);

/**
 * Creates Circular Buffers (CBs) in L1 memory of all cores within core range (inclusive) at specified address and adds it to the program. L1 allocator reserves size_in_bytes bytes at manually specified addresses.
 *
 * Return value: Circular Buffer ID (uintptr_t)
 *
 * | Argument      | Description                                                                    | Type               | Valid Range                             | Required |
 * |---------------|--------------------------------------------------------------------------------|--------------------|-----------------------------------------|----------|
 * | program       | The program to which buffer will be added to.                                  | Program *          |                                         | True     |
 * | buffer_index  | The index/ID of the CB.                                                        | uint32_t           | 0 to 32 DOX-TODO: specify more detail here. | True     |
 * | core_range    | Range of the Tensix co-ordinates where buffer will reside (Logical co-ordinates)  | const CoreRange & (std::pair<CoreCoord, CoreCoord>) | DOX-TODO: { , } –> { , }                    | True     |
 * | num_tiles     | Total number of tiles to be stored in the CB                                   | uint32_t           | DOX-TODO: range?                            | True     |
 * | size_in_bytes | Size of CB buffer in Bytes                                                     | uint32_t           | 0 to 1 MB (DOX-TODO: in Bytes)              | True     |
 * | data_format   | The format of the data to be stored in the CB                                  | DataFormat enum    | DataFormat::Float16_b                   | True     |
 * | l1_address    | Address at which the CB buffer will reside                                     | optional<uint32_t>           | 200 kB to 1MB (DOX-TODO: in bytes)          | False     |
 */
CircularBufferID CreateCircularBuffers(
    Program &program,
    uint32_t buffer_index,
    const CoreRange &core_range,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    DataFormat data_format,
    std::optional<uint32_t> l1_address = std::nullopt);

/**
 * Creates Circular Buffers (CBs) in L1 memory of all cores within set of core ranges (inclusive) at specified address and adds it to the program. L1 allocator reserves size_in_bytes bytes at manually specified addresses.
 *
 * Return value: Circular Buffer ID (uintptr_t)
 *
 * | Argument      | Description                                                                    | Type               | Valid Range                             | Required |
 * |---------------|--------------------------------------------------------------------------------|--------------------|-----------------------------------------|----------|
 * | program       | The program to which buffer will be added to.                                  | Program *          |                                         | True     |
 * | buffer_index  | The index/ID of the CB.                                                        | uint32_t           | 0 to 32 DOX-TODO: specify more detail here. | True     |
 * | core_range_set   | Ranges of the Tensix co-ordinates where buffer will reside (Logical co-ordinates)  | const CoreRangeSet & (std::set<CoreRange>) | DOX-TODO: { , } –> { , }                    | True     |
 * | num_tiles     | Total number of tiles to be stored in the CB                                   | uint32_t           | DOX-TODO: range?                            | True     |
 * | size_in_bytes | Size of CB buffer in Bytes                                                     | uint32_t           | 0 to 1 MB (DOX-TODO: in Bytes)              | True     |
 * | data_format   | The format of the data to be stored in the CB                                  | DataFormat enum    | DataFormat::Float16_b                   | True     |
 * | l1_address    | Address at which the CB buffer will reside                                     | optional<uint32_t>           | 200 kB to 1MB (DOX-TODO: in bytes)          | False     |
 */
CircularBufferID CreateCircularBuffers(
    Program &program,
    uint32_t buffer_index,
    const CoreRangeSet &core_range_set,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    DataFormat data_format,
    std::optional<uint32_t> l1_address = std::nullopt);

/**
 * Creates Circular Buffers (CBs) in L1 memory of all cores within set of core ranges (inclusive) at specified address and adds it to the program. L1 allocator reserves size_in_bytes bytes at manually specified addresses.
 *
 * Return value: Circular Buffer ID (uintptr_t)
 *
 * | Argument      | Description                                                                    | Type               | Valid Range                             | Required |
 * |---------------|--------------------------------------------------------------------------------|--------------------|-----------------------------------------|----------|
 * | program       | The program to which buffer will be added to.                                  | Program *          |                                         | True     |
 * | buffer_indices  | Indices/IDs of the CB.                                                        | const std::set<uint32_t> &           | 0 to 32 DOX-TODO: specify more detail here. | True     |
 * | core_range_set   | Ranges of the Tensix co-ordinates where buffer will reside (Logical co-ordinates)  | const CoreRangeSet & (std::set<CoreRange>) | DOX-TODO: { , } –> { , }                    | True     |
 * | num_tiles     | Total number of tiles to be stored in the CB                                   | uint32_t           | DOX-TODO: range?                            | True     |
 * | size_in_bytes | Size of CB buffer in Bytes                                                     | uint32_t           | 0 to 1 MB (DOX-TODO: in Bytes)              | True     |
 * | data_format   | The format of the data to be stored in the CB                                  | DataFormat enum    | DataFormat::Float16_b                   | True     |
 * | l1_address    | Address at which the CB buffer will reside                                     | optional<uint32_t>           | 200 kB to 1MB (DOX-TODO: in bytes)          | True     |
 */
CircularBufferID CreateCircularBuffers(
    Program &program,
    const std::set<uint32_t> &buffer_indices,
    const CoreRangeSet &core_range_set,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    DataFormat data_format,
    std::optional<uint32_t> l1_address = std::nullopt);

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
*  | Argument    | Description                             | Type       | Valid Range | Required |
*  |-------------|---------------------------------------- |------------|-------------|----------|
*  | device      | The device that the buffer will reside  | Device     |             | Yes      |
*  | size        | size of buffer                          | uint64_t   |             | Yes      |
*  | page_size   | buffer page size                        | uint64_t   |             | Yes      |
*  | buffer_type | type of buffer (L1 or DRAM)             | BufferType |             | Yes      |
*/
Buffer CreateBuffer(Device *device, std::uint64_t size, std::uint64_t page_size, const BufferType buffer_type);

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
* | Argument    | Description                                     | Data type               | Valid range                                      | Required |
* |-------------|-------------------------------------------------|-------------------------|--------------------------------------------------|----------|
* | buffer      | Buffer to read data from                        | const Buffer &          |                                                  | Yes      |
* | host_buffer | Buffer on host to copy data into                | std::vector<uint32_t> & |                                                  | Yes      |
*/
void ReadFromBuffer(const Buffer &buffer, std::vector<uint32_t> &host_buffer);

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
