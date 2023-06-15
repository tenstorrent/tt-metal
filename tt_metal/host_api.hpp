#pragma once

#include <functional>
#include <iostream>
#include <random>
#include <tuple>
#include <utility>
#include <variant>

#include "tools/profiler/profiler.hpp"
#include "hostdevcommon/common_runtime_address_map.h"
#include "hostdevcommon/registers.hpp"
#include "hostdevcommon/common_values.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/buffers/circular_buffer.hpp"
#include "tt_metal/impl/buffers/semaphore.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/device/host.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/impl/program.hpp"
#include "llrt/llrt.hpp"

#include "tt_metal/impl/dispatch/command_queue.hpp"

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

// ==================================================
//                  HOST API: reports
// ==================================================

/**
 * Enable generation of reports for compilation statistics.
 * Two reports are generated in .reports/tt_metal:
 *  - `compile_program_summary.csv` has a table with an entry for each program that indicates number of CreateComputeKernel and CreateDataMovement API calls,
 *  and number of kernel compilation cache hits and misses
 *  - `compile_program.csv` expands on the summary report by including a per program table with an entry for each kernel, indicating the cores the kernel
 * is placed on, kernel attributes and whether there was a cache hit or miss when compiling the kernel.
 *
 * Return value: void
 *
 */
void EnableCompilationReports();


/**
 * Disable generation compilation statistics reports.
 *
 * Return value: void
 *
 */
void DisableCompilationReports();

// ==================================================
//                  HOST API: profiler
// ==================================================

/**
 * Dump host side profiler results into the host side CSV log
 *
 * Return value: void
 *
 * | Argument       | Description                                                                  |  Data type  | Valid range  | required |
 * |----------------|------------------------------------------------------------------------------|-------------|--------------|----------|
 * | name_prepend   | The name or description to be prepended to all rows in the CSV for this dump | std::string | Any string   | No       |
 * */
void DumpHostProfileResults(std::string name_prepend = "");

/**
 * Read device side profiler data and dump results into device side CSV log
 *
 * Return value: void
 *
 * | Argument      | Description                                       | Type            | Valid Range                                            | Required |
 * |---------------|---------------------------------------------------|-----------------|--------------------------------------------------------|----------|
 * | device        | The device holding the program being profiled.    | Device *        |                                                        | True     |
 * | program       | The program being profiled.                       | const Program & |                                                        | True     |
 * */
void DumpDeviceProfileResults(Device *device, const Program &program);

/**
 * Set the directory for all CSV logs produced by the profiler instance in the tt-metal module
 *
 * Return value: void
 *
 * | Argument     | Description                                             |  Data type  | Valid range              | required |
 * |--------------|---------------------------------------------------------|-------------|--------------------------|----------|
 * | output_dir   | The output directory that will hold the outpu CSV logs  | std::string | Any valid directory path | No       |
 * */
void SetProfilerDir(std::string output_dir = "");

/**
 * Start a fresh log for the host side profile results
 *
 * Return value: void
 *
 * | Argument     | Description                                             |  Data type  | Valid range              | required |
 * |--------------|---------------------------------------------------------|-------------|--------------------------|----------|
 * */
void FreshProfilerHostLog();

/**
 * Start a fresh log for the device side profile results
 *
 * Return value: void
 *
 * | Argument     | Description                                             |  Data type  | Valid range              | required |
 * |--------------|---------------------------------------------------------|-------------|--------------------------|----------|
 * */
void FreshProfilerDeviceLog();


// ==================================================
//                  HOST API: host and device
// ==================================================
Host *GetHost();

/**
 * Instantiates a device object.
 *
 * Return value: Device *
 *
 * | Argument       | Description                                                      | Data type | Valid range                                         | required |
 * |----------------|------------------------------------------------------------------|-----------|-----------------------------------------------------|----------|
 * | device_type    | Type of Tenstorrent device to be used                            | ARCH enum | “tt::ARCH::GRAYSKULL”                               | Yes      |
 * | pcie_slot      | The number of the PCIexpress slot in which the device is located | int       | 0 to 7                                              | Yes      |
 * */
Device *CreateDevice(tt::ARCH arch, int pcie_slot);

/**
 * Initializes a device by creating a tt_cluster object and memory manager. Puts device into reset.
 *
 * Currently device has a 1:1 mapping with tt_cluster and memory manager only allocates DRAM addresses.
 *
 * Return value: bool
 *
 * | Argument                    | Description                                 | Type                 | Valid Range         | Required |
 * |-----------------------------|---------------------------------------------|----------------------|---------------------|----------|
 * | device                      | Pointer to device object                    | Device *             |                     | Yes      |
 * | memory_allocator            | Type of memory allocator scheme to use      | MemoryAllocator enum | BASIC or L1_BANKING | No       |
 */
bool InitializeDevice(Device *device, const MemoryAllocator &memory_allocator = MemoryAllocator::BASIC);

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

/**
 * Starts a debug print server on core {1,1} in physical core grid coordinates.
*/
void StartDebugPrintServer(Device *device);

/**
 * Starts a debug print server on specified cores (in physical core grid coordinates).
 *
 * |      Argument     |                                                      Description                                                      |        Data type        |    Valid range    | required |
 * |:-----------------:|:---------------------------------------------------------------------------------------------------------------------:|:-----------------------:|:-----------------:|----------|
 * | device            | Device pointer                                                                                                        |                         |                   | Yes      |
 * | cores             | Array of x,y pairs with locations of the Tensix cores (physical coordinates)                                          | const CoreCoord &      | {0, 0} -> {9, 11} | Yes      |
*/
void StartDebugPrintServerOnCores(Device *device, const std::vector<std::vector<int>>& cores);

// ==================================================
//                  HOST API: program & kernels
// ==================================================

/**
 * Creates a single core data movement kernel and adds it to the program.
 *
 * Return value: DataMovementKernel *
 *
 * | Argument       | Description                                                                                                  | Data type                | Valid range                                                    | required |
 * |----------------|--------------------------------------------------------------------------------------------------------------|--------------------------|----------------------------------------------------------------|----------|
 * | program        | The program to which this kernel will be added to                                                            | Program &                |                                                                | Yes      |
 * | file_name      | Name of file containing the kernel                                                                           | const std::string        |                                                                | Yes      |
 * | core           | The location of the Tensix core on which the kernel will execute (Logical co-ordinates)                      | const CoreCoord &       | {0, 0} –> {9, 11}                                              | Yes      |
 * | compile_args   | Compile arguments passed to kernel at compile time                                                           | const std::vector<uint32_t> &       |                                                                | Yes      |
 * | processor_type | The target RISC-V processor on which the kernel will execute, on the given Tensix core (1 kernel per RISC-V) | enum                     | DataMovementProcessor::RISCV_0, DataMovementProcessor::RISCV_1 | Yes      |
 * | noc            | The NoC ID on which the kernel will perform data transfers                                                   | enum                     | RISCV_0_default, RISCV_1_default, NOC_0, NOC_1,                | Yes      |
 */

DataMovementKernel *CreateDataMovementKernel(
    Program &program,
    const std::string &file_name,
    const CoreCoord &core,
    const std::vector<uint32_t> &compile_args,
    DataMovementProcessor processor_type,
    NOC noc);

/**
 * Creates a single core data movement kernel with no default arguments and adds it to the program.
 *
 * Return value: DataMovementKernel *
 *
 * | Argument       | Description                                                                                                  | Data type                | Valid range                                                    | required |
 * |----------------|--------------------------------------------------------------------------------------------------------------|--------------------------|----------------------------------------------------------------|----------|
 * | program        | The program to which this kernel will be added to                                                            | Program &                |                                                                | Yes      |
 * | file_name      | Name of file containing the kernel                                                                           | const std::string        |                                                                | Yes      |
 * | core           | A range of the Tensix cores (inclusive) on which the kernel will execute (Logical co-ordinates)              | const CoreCoord &       | {0, 0} –> {9, 11}                                              | Yes      |
 * | processor_type | The target RISC-V processor on which the kernel will execute, on the given Tensix core (1 kernel per RISC-V) | enum                     | DataMovementProcessor::RISCV_0, DataMovementProcessor::RISCV_1 | Yes      |
 * | noc            | The NoC ID on which the kernel will perform data transfers                                                   | enum                     | RISCV_0_default, RISCV_1_default, NOC_0, NOC_1,                | Yes      |
 */
DataMovementKernel *CreateDataMovementKernel(
    Program &program,
    const std::string &file_name,
    const CoreCoord &core,
    DataMovementProcessor processor_type,
    NOC noc);

/**
 * Creates a multi-core data movement kernel and adds it to the program.
 *
 * Return value: DataMovementKernel *
 *
 * | Argument       | Description                                                                                                  | Data type                | Valid range                                                    | required |
 * |----------------|--------------------------------------------------------------------------------------------------------------|--------------------------|----------------------------------------------------------------|----------|
 * | program        | The program to which this kernel will be added to                                                            | Program &                |                                                                | Yes      |
 * | file_name      | Name of file containing the kernel                                                                           | const std::string        |                                                                | Yes      |
 * | core_range     | The range of the Tensix co-ordinates on which the kernel will execute (Logical co-ordinates)                 | const CoreRange &        | Any range encompassing cores within {0 , 0} –> {9, 11}         | Yes      |
 * | compile_args   | Compile arguments passed to kernel at compile time                                                           | const std::vector<uint32_t> &       |                                                                | Yes      |
 * | processor_type | The target RISC-V processor on which the kernel will execute, on the given Tensix core (1 kernel per RISC-V) | enum                     | DataMovementProcessor::RISCV_0, DataMovementProcessor::RISCV_1 | Yes      |
 * | noc            | The NoC ID on which the kernel will perform data transfers                                                   | enum                     | RISCV_0_default, RISCV_1_default, NOC_0, NOC_1,                | Yes      |
 */
DataMovementKernel *CreateDataMovementKernel(
    Program &program,
    const std::string &file_name,
    const CoreRange &core_range,
    const std::vector<uint32_t> &compile_args,
    DataMovementProcessor processor_type,
    NOC noc);

/**
 * Creates a multi-core data movement kernel with no default arguments and adds it to the program.
 *
 * Return value: DataMovementKernel *
 *
 * | Argument       | Description                                                                                                  | Data type                | Valid range                                                    | required |
 * |----------------|--------------------------------------------------------------------------------------------------------------|--------------------------|----------------------------------------------------------------|----------|
 * | program        | The program to which this kernel will be added to                                                            | Program *                |                                                                | Yes      |
 * | file_name      | Name of file containing the kernel                                                                           | const std::string        |                                                                | Yes      |
 * | core_range     | The range of the Tensix co-ordinates on which the kernel will execute (Logical co-ordinates)                 | const CoreRange &        | Any range encompassing cores within {0 , 0} –> {9, 11}         | Yes      |
 * | processor_type | The target RISC-V processor on which the kernel will execute, on the given Tensix core (1 kernel per RISC-V) | enum                     | DataMovementProcessor::RISCV_0, DataMovementProcessor::RISCV_1 | Yes      |
 * | noc            | The NoC ID on which the kernel will perform data transfers                                                   | enum                     | RISCV_0_default, RISCV_1_default, NOC_0, NOC_1,                | Yes      |
 */
DataMovementKernel *CreateDataMovementKernel(
    Program &program,
    const std::string &file_name,
    const CoreRange &core_range,
    DataMovementProcessor processor_type,
    NOC noc);

/**
 * Creates a multi-core data movement kernel and adds it to the program.
 *
 * Return value: DataMovementKernel *
 *
 * | Argument       | Description                                                                                                  | Data type                | Valid range                                                    | required |
 * |----------------|--------------------------------------------------------------------------------------------------------------|--------------------------|----------------------------------------------------------------|----------|
 * | program        | The program to which this kernel will be added to                                                            | Program &                |                                                                | Yes      |
 * | file_name      | Name of file containing the kernel                                                                           | const std::string        |                                                                | Yes      |
 * | core_range_set    | A set of ranges (inclusive) of Tensix co-ordinates on which the kernel will execute (Logical co-ordinates)   | const CoreRangeSet &     | Ranges encompassing cores within {0 , 0} –> {9, 11}            | Yes      |
 * | compile_args   | Compile arguments passed to kernel at compile time                                                           | const std::vector<uint32_t> &       |                                                                | Yes      |
 * | processor_type | The target RISC-V processor on which the kernel will execute, on the given Tensix core (1 kernel per RISC-V) | enum                     | DataMovementProcessor::RISCV_0, DataMovementProcessor::RISCV_1 | Yes      |
 * | noc            | The NoC ID on which the kernel will perform data transfers                                                   | enum                     | RISCV_0_default, RISCV_1_default, NOC_0, NOC_1,                | Yes      |
 */
DataMovementKernel *CreateDataMovementKernel(
    Program &program,
    const std::string &file_name,
    const CoreRangeSet &core_range_set,
    const std::vector<uint32_t> &compile_args,
    DataMovementProcessor processor_type,
    NOC noc);

/**
 * Creates a multi-core data movement kernel with no default arguments and adds it to the program.
 *
 * Return value: DataMovementKernel *
 *
 * | Argument       | Description                                                                                                  | Data type                | Valid range                                                    | required |
 * |----------------|--------------------------------------------------------------------------------------------------------------|--------------------------|----------------------------------------------------------------|----------|
 * | program        | The program to which this kernel will be added to                                                            | Program *                |                                                                | Yes      |
 * | file_name      | Name of file containing the kernel                                                                           | const std::string        |                                                                | Yes      |
 * | core_range_set    | A set of ranges (inclusive) of Tensix co-ordinates on which the kernel will execute (Logical co-ordinates)   | const CoreRangeSet &     | Ranges encompassing cores within {0 , 0} –> {9, 11}            | Yes      |
 * | processor_type | The target RISC-V processor on which the kernel will execute, on the given Tensix core (1 kernel per RISC-V) | enum                     | DataMovementProcessor::RISCV_0, DataMovementProcessor::RISCV_1 | Yes      |
 * | noc            | The NoC ID on which the kernel will perform data transfers                                                   | enum                     | RISCV_0_default, RISCV_1_default, NOC_0, NOC_1,                | Yes      |
 */
DataMovementKernel *CreateDataMovementKernel(
    Program &program,
    const std::string &file_name,
    const CoreRangeSet &core_range_set,
    DataMovementProcessor processor_type,
    NOC noc);

/**
 * Creates a single core compute kernel object, and adds it to the program.
 *
 * Return value: ComputeKernel *
 *
 * |     Argument     |                                       Description                                       |      Data type      |      Valid range      | required |
 * |:----------------:|:---------------------------------------------------------------------------------------:|:-------------------:|:---------------------:|----------|
 * | program          | The program to which this kernel will be added to                                       | Program &           |                       | Yes      |
 * | file_name        | Name of file containing the kernel                                                      | const std::string   |                       | Yes      |
 * | core             | The location of the Tensix core on which the kernel will execute (Logical co-ordinates) | const CoreCoord &  | {0, 0} –> {9, 11}     | Yes      |
 * | compile_args     | Compile arguments passed to kernel at compile time                                                           | const std::vector<uint32_t> &       |                                                                | Yes      |
 * | math_fidelity    | The percision of the matrix compute engine                                              | enum                | MathFidelity::HiFi4   | Yes      |
 * | fp32_dest_acc_en | Specifies the type of accumulation performed in the matrix compute engine.              | bool                | false (for Grayskull) | Yes      |
 * | math_approx_mode | Used by the vector compute engine. (will be depricated)                                 | bool                | true, false           | Yes      |
 */
ComputeKernel *CreateComputeKernel(
    Program &program,
    const std::string &file_name,
    const CoreCoord &core,
    const std::vector<uint32_t> &compile_args,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode);

/**
 * Creates a multi-core compute kernel object, and adds it to the program.
 *
 * Return value: ComputeKernel *
 *
 * | Argument         | Description                                                                                  | Data type           | Valid range                                            | required |
 * |------------------|----------------------------------------------------------------------------------------------|---------------------|--------------------------------------------------------|----------|
 * | program          | The program to which this kernel will be added to                                            | Program &           |                                                        | Yes      |
 * | file_name        | Name of file containing the kernel                                                           | const std::string   |                                                        | Yes      |
 * | core_range       | The range of the Tensix co-ordinates on which the kernel will execute (Logical co-ordinates) | const CoreRange &   | Any range encompassing cores within {0 , 0} –> {9, 11} | Yes      |
 * | compile_args     | Compile arguments passed to kernel at compile time                                                           | const std::vector<uint32_t> &       |                                                                | Yes      |
 * | math_fidelity    | The percision of the matrix compute engine                                                   | enum                | MathFidelity::HiFi4                                    | Yes      |
 * | fp32_dest_acc_en | Specifies the type of accumulation performed in the matrix compute engine.                   | bool                | false (for Grayskull)                                  | Yes      |
 * | math_approx_mode | Used by the vector compute engine. (will be depricated)                                      | bool                | true, false                                            | Yes      |
 */
ComputeKernel *CreateComputeKernel(
    Program &program,
    const std::string &file_name,
    const CoreRange &core_range,
    const std::vector<uint32_t> &compile_args,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode);

/**
 * Creates a multi-core compute kernel object, and adds it to the program.
 *
 * Return value: ComputeKernel *
 *
 * | Argument         | Description                                                                                                  | Data type           | Valid range                                            | required |
 * |------------------|--------------------------------------------------------------------------------------------------------------|---------------------|--------------------------------------------------------|----------|
 * | program          | The program to which this kernel will be added to                                                            | Program &           |                                                        | Yes      |
 * | file_name        | Name of file containing the kernel                                                                           | const std::string   |                                                        | Yes      |
 * | core_range_set   | A set of ranges (inclusive) of Tensix co-ordinates on which the kernel will execute (Logical co-ordinates)   | const CoreRangeSet &     | Ranges encompassing cores within {0 , 0} –> {9, 11}            | Yes      |
 * | compile_args     | Compile arguments passed to kernel at compile time                                                           | const std::vector<uint32_t> &       |                                                                | Yes      |
 * | math_fidelity    | The percision of the matrix compute engine                                                                   | enum                | MathFidelity::HiFi4                                    | Yes      |
 * | fp32_dest_acc_en | Specifies the type of accumulation performed in the matrix compute engine.                                   | bool                | false (for Grayskull)                                  | Yes      |
 * | math_approx_mode | Used by the vector compute engine. (will be depricated)                                                      | bool                | true, false                                            | Yes      |
 */
ComputeKernel *CreateComputeKernel(
    Program &program,
    const std::string &file_name,
    const CoreRangeSet &core_range_set,
    const std::vector<uint32_t> &compile_args,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode);

// ==================================================
//                  HOST API: data format
// ==================================================
/**
 * Returns datum size of given data format in bytes
 *
 * Return value: uint32_t
 *
 * | Argument    | Description    | Type                | Valid Range | Required |
 * |-------------|----------------|---------------------|-------------|----------|
 * | data_format | Format of data | tt::DataFormat enum |             | Yes      |
 */
uint32_t DatumSize(const DataFormat &data_format);

/**
 * Returns tile size of given data format in bytes
 *
 * Return value: uint32_t
 *
 * | Argument    | Description    | Type                | Valid Range | Required |
 * |-------------|----------------|---------------------|-------------|----------|
 * | data_format | Format of data | tt::DataFormat enum |             | Yes      |
 */
uint32_t TileSize(const DataFormat &data_format);

// ==================================================
//                  HOST API: buffers
// ==================================================
/**
 * Creates a Circular Buffer (CBs) in L1 memory at specified address and core and adds it to the program. L1 allocator does not track CBs with manually specified addresses.
 * Warning: Do not mix use of this API with CB creation APIs that are tracked by allocator since there is no gurantee that buffer space will not be overwritten.
 *
 * Return value: CircularBuffer *
 *
 * | Argument      | Description                                                                    | Type               | Valid Range                             | Required |
 * |---------------|--------------------------------------------------------------------------------|--------------------|-----------------------------------------|----------|
 * | program       | The program to which buffer will be added to.                                  | Program &          |                                         | True     |
 * | device        | The device where the L1 buffer resides.                                        | Device *           |                                         | True     |
 * | buffer_index  | The index/ID of the CB.                                                        | uint32_t           | 0 to 32 DOX-TODO: specify more detail here. | True     |
 * | core          | The location of the Tensix core on which the CB will reside (logical co-ordinates) | const CoreCoord & | DOX-TODO: { , } –> { , }                    | True     |
 * | num_tiles     | Total number of tiles to be stored in the CB                                   | uint32_t           | DOX-TODO: range?                            | True     |
 * | size_in_bytes | Size of CB buffer in Bytes                                                     | uint32_t           | 0 to 1 MB (DOX-TODO: in Bytes)              | True     |
 * | l1_address    | Address at which the CB buffer will reside                                     | uint32_t           | 200 kB to 1MB (DOX-TODO: in bytes)          | True     |
 * | data_format   | The format of the data to be stored in the CB                                  | DataFormat enum    | DataFormat::Float16_b                   | True     |
 */
CircularBuffer *CreateCircularBuffer(
    Program &program,
    Device *device,
    uint32_t buffer_index,
    const CoreCoord &core,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    uint32_t l1_address,
    DataFormat data_format);

/**
 * Allocates and creates Circular Buffers (CBs) in L1 memory of specified core and adds it to the program. L1 allocator is responsible for generating address.
 * Warning: Do not mix use of this API with CB creation APIs that are not tracked by allocator since there is no gurantee that buffer space will not be overwritten.
 *
 * Return value: CircularBuffer *
 *
 * | Argument      | Description                                                                    | Type               | Valid Range                             | Required |
 * |---------------|--------------------------------------------------------------------------------|--------------------|-----------------------------------------|----------|
 * | program       | The program to which buffer will be added to.                                  | Program &          |                                         | True     |
 * | device        | The device where the L1 buffer resides.                                        | Device *           |                                         | True     |
 * | buffer_index  | The index/ID of the CB.                                                        | uint32_t           | 0 to 32 DOX-TODO: specify more detail here. | True     |
 * | core          | The location of the Tensix core on which the CB will reside (logical co-ordinates) | const CoreCoord & | DOX-TODO: { , } –> { , }                    | True     |
 * | num_tiles     | Total number of tiles to be stored in the CB                                   | uint32_t           | DOX-TODO: range?                            | True     |
 * | size_in_bytes | Size of CB buffer in Bytes                                                     | uint32_t           | 0 to 1 MB (DOX-TODO: in Bytes)              | True     |
 * | data_format   | The format of the data to be stored in the CB                                  | DataFormat enum    | DataFormat::Float16_b                   | True     |
 */
CircularBuffer *CreateCircularBuffer(
    Program &program,
    Device *device,
    uint32_t buffer_index,
    const CoreCoord &core,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    DataFormat data_format);

/**
 * Creates Circular Buffers (CBs) in L1 memory of all cores within core range (inclusive) at specified address and adds it to the program. L1 allocator does not track CBs with manually specified addresses.
 * Warning: Do not mix use of this API with CB creation APIs that are tracked by allocator since there is no gurantee that buffer space will not be overwritten.
 *
 * Return value: CircularBuffer *
 *
 * | Argument      | Description                                                                    | Type               | Valid Range                             | Required |
 * |---------------|--------------------------------------------------------------------------------|--------------------|-----------------------------------------|----------|
 * | program       | The program to which buffer will be added to.                                  | Program *          |                                         | True     |
 * | device        | The device where the L1 buffer resides.                                        | Device *           |                                         | True     |
 * | buffer_index  | The index/ID of the CB.                                                        | uint32_t           | 0 to 32 DOX-TODO: specify more detail here. | True     |
 * | core_range    | Range of the Tensix co-ordinates where buffer will reside (Logical co-ordinates)  | const CoreRange & (std::pair<CoreCoord, CoreCoord>) | DOX-TODO: { , } –> { , }                    | True     |
 * | num_tiles     | Total number of tiles to be stored in the CB                                   | uint32_t           | DOX-TODO: range?                            | True     |
 * | size_in_bytes | Size of CB buffer in Bytes                                                     | uint32_t           | 0 to 1 MB (DOX-TODO: in Bytes)              | True     |
 * | l1_address    | Address at which the CB buffer will reside                                     | uint32_t           | 200 kB to 1MB (DOX-TODO: in bytes)          | True     |
 * | data_format   | The format of the data to be stored in the CB                                  | DataFormat enum    | DataFormat::Float16_b                   | True     |
 */
CircularBuffer *CreateCircularBuffers(
    Program &program,
    Device *device,
    uint32_t buffer_index,
    const CoreRange &core_range,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    uint32_t l1_address,
    DataFormat data_format);

/**
 * Creates Circular Buffer (CB) per core within core range (inclusive). CBs will be allocated to the same address on their respective cores, an error is returned is this is not possible.
 * Warning: Do not mix use of this API with CB creation APIs that are not tracked by allocator since there is no gurantee that buffer space will not be overwritten.
 *
 * Return value: CircularBuffer *
 *
 * | Argument      | Description                                                                    | Type               | Valid Range                             | Required |
 * |---------------|--------------------------------------------------------------------------------|--------------------|-----------------------------------------|----------|
 * | program       | The program to which buffer will be added to.                                  | Program *          |                                         | True     |
 * | device        | The device where the L1 buffer resides.                                        | Device *           |                                         | True     |
 * | buffer_index  | The index/ID of the CB.                                                        | uint32_t           | 0 to 32 DOX-TODO: specify more detail here. | True     |
 * | core_range    | Range of the Tensix co-ordinates where buffer will reside (Logical co-ordinates)  | const CoreRange & (std::pair<CoreCoord, CoreCoord>) | DOX-TODO: { , } –> { , }                    | True     |
 * | num_tiles     | Total number of tiles to be stored in the CB                                   | uint32_t           | DOX-TODO: range?                            | True     |
 * | size_in_bytes | Size of CB buffer in Bytes                                                     | uint32_t           | 0 to 1 MB (DOX-TODO: in Bytes)              | True     |
 * | data_format   | The format of the data to be stored in the CB                                  | DataFormat enum    | DataFormat::Float16_b                   | True     |
 */
CircularBuffer *CreateCircularBuffers(
    Program &program,
    Device *device,
    uint32_t buffer_index,
    const CoreRange &core_range,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    DataFormat data_format);

/**
 * Creates Circular Buffers (CBs) in L1 memory of all cores within set of core ranges (inclusive) at specified address and adds it to the program. L1 allocator does not track CBs with manually specified addresses.
 * Warning: Do not mix use of this API with CB creation APIs that are tracked by allocator since there is no gurantee that buffer space will not be overwritten.
 *
 * Return value: CircularBuffer *
 *
 * | Argument      | Description                                                                    | Type               | Valid Range                             | Required |
 * |---------------|--------------------------------------------------------------------------------|--------------------|-----------------------------------------|----------|
 * | program       | The program to which buffer will be added to.                                  | Program *          |                                         | True     |
 * | device        | The device where the L1 buffer resides.                                        | Device *           |                                         | True     |
 * | buffer_index  | The index/ID of the CB.                                                        | uint32_t           | 0 to 32 DOX-TODO: specify more detail here. | True     |
 * | core_range_set   | Ranges of the Tensix co-ordinates where buffer will reside (Logical co-ordinates)  | const CoreRangeSet & (std::set<CoreRange>) | DOX-TODO: { , } –> { , }                    | True     |
 * | num_tiles     | Total number of tiles to be stored in the CB                                   | uint32_t           | DOX-TODO: range?                            | True     |
 * | size_in_bytes | Size of CB buffer in Bytes                                                     | uint32_t           | 0 to 1 MB (DOX-TODO: in Bytes)              | True     |
 * | l1_address    | Address at which the CB buffer will reside                                     | uint32_t           | 200 kB to 1MB (DOX-TODO: in bytes)          | True     |
 * | data_format   | The format of the data to be stored in the CB                                  | DataFormat enum    | DataFormat::Float16_b                   | True     |
 */
CircularBuffer *CreateCircularBuffers(
    Program &program,
    Device *device,
    uint32_t buffer_index,
    const CoreRangeSet &core_range_set,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    uint32_t l1_address,
    DataFormat data_format);

/**
 * Creates Circular Buffer (CB) per core within set of core ranges (inclusive). CBs will be allocated to the same address on their respective cores, an error is returned is this is not possible.
 * Warning: Do not mix use of this API with CB creation APIs that are not tracked by allocator since there is no gurantee that buffer space will not be overwritten.
 *
 * Return value: CircularBuffer *
 *
 * | Argument      | Description                                                                    | Type               | Valid Range                             | Required |
 * |---------------|--------------------------------------------------------------------------------|--------------------|-----------------------------------------|----------|
 * | program       | The program to which buffer will be added to.                                  | Program *          |                                         | True     |
 * | device        | The device where the L1 buffer resides.                                        | Device *           |                                         | True     |
 * | buffer_index  | The index/ID of the CB.                                                        | uint32_t           | 0 to 32 DOX-TODO: specify more detail here. | True     |
 * | core_range_set   | Ranges of the Tensix co-ordinates where buffer will reside (Logical co-ordinates)  | const CoreRange & (std::set<CoreRange>) | DOX-TODO: { , } –> { , }                    | True     |
 * | num_tiles     | Total number of tiles to be stored in the CB                                   | uint32_t           | DOX-TODO: range?                            | True     |
 * | size_in_bytes | Size of CB buffer in Bytes                                                     | uint32_t           | 0 to 1 MB (DOX-TODO: in Bytes)              | True     |
 * | data_format   | The format of the data to be stored in the CB                                  | DataFormat enum    | DataFormat::Float16_b                   | True     |
 */
CircularBuffer *CreateCircularBuffers(
    Program &program,
    Device *device,
    uint32_t buffer_index,
    const CoreRangeSet &core_range_set,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    DataFormat data_format);

/**
 * Initializes semaphore on all cores within core range (inclusive). Each core can have up to four 32B semaphores.
 *
 * Return value: Semaphore *
 *
 * | Argument      | Description                                          | Type                                                  | Valid Range                                              | Required |
 * |---------------|------------------------------------------------------|-------------------------------------------------------|----------------------------------------------------------|----------|
 * | program       | The program to which semaphore will be added to      | Program &                                             |                                                          | Yes      |
 * | device        | The device where the semaphore resides               | Device *                                              |                                                          | Yes      |
 * | core_range    | Range of the Tensix co-ordinates using the semaphore | const CoreRange & (std::pair<CoreCoord, CoreCoord>)   | Pair of logical coords where first coord <= second coord | Yes      |
 * | initial_value | Initial value of the semaphore                       | uint32_t                                              |                                                          | Yes      |
 */
Semaphore *CreateSemaphore(Program &program, Device *device, const CoreRange &core_range, uint32_t initial_value);

/**
 * Initializes semaphore on all cores within core range (inclusive). Each core can have up to four 32B semaphores.
 *
 * Return value: Semaphore *
 *
 * | Argument       | Description                                                 | Type                   | Valid Range                                               | Required |
 * |----------------|-------------------------------------------------------------|------------------------|-----------------------------------------------------------|----------|
 * | program        | The program to which semaphore will be added to             | Program &              |                                                           | Yes      |
 * | device         | The device where the semaphore resides                      | Device *               |                                                           | Yes      |
 * | core_range_set    | Set of Range of the Tensix co-ordinates using the semaphore | const CoreRangeSet &   | Pairs of logical coords where first coord <= second coord | Yes      |
 * | initial_value  | Initial value of the semaphore                              | uint32_t               |                                                           | Yes      |
 */
Semaphore *CreateSemaphore(Program &program, Device *device, const CoreRangeSet &core_range_set, uint32_t initial_value);

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
void WriteToBuffer(const Buffer &buffer, std::vector<uint32_t> &host_buffer);

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

/**
 * Copies data from a host buffer into a buffer within the device DRAM channel
 *
 * Return value: bool
 *
 * | Argument     | Description                                            | Data type             | Valid range                               | required |
 * |--------------|--------------------------------------------------------|-----------------------|-------------------------------------------|----------|
 * | device       | The device whose DRAM to write data into               | Device *              |                                           | Yes      |
 * | dram_channel | Channel index of DRAM to write into                    | int                   | On Grayskull, [0, 7] inclusive            | Yes      |
 * | address      | Starting address on DRAM channel to begin writing data | uint32_t              |                                           | Yes      |
 * | host_buffer  | Buffer on host to copy data from                       | std::vector<uint32_t> | Host buffer must be fully fit DRAM buffer | Yes      |
 */
bool WriteToDeviceDRAMChannel(Device *device, int dram_channel, uint32_t address, std::vector<uint32_t> &host_buffer);

/**
 * Copy data from a device DRAM channel to a host buffer
 *
 * Return value: bool
 *
 * | Argument     | Description                                                  | Data type             | Valid range                    | required |
 * |--------------|--------------------------------------------------------------|-----------------------|--------------------------------|----------|
 * | device       | The device whose DRAM to read data from                      | Device *              |                                | Yes      |
 * | dram_channel | Channel index of DRAM to read from                           | int                   | On Grayskull, [0, 7] inclusive | Yes      |
 * | address      | Starting address on DRAM channel from which to begin reading | uint32_t              |                                | Yes      |
 * | size         | Size of buffer to read from device in bytes                  | uint32_t              |                                | Yes      |
 * | host_buffer  | Buffer on host to copy data into                             | std::vector<uint32_t> |                                | Yes      |
 */
bool ReadFromDeviceDRAMChannel(Device *device, int dram_channel, uint32_t address, uint32_t size, std::vector<uint32_t> &host_buffer);

/**
 * Copy data from a host buffer into an L1 buffer. (Note: Current Can not be a CircularBuffer.)
 *
 * Return value: bool
 *
 * | Argument      | Description                                     | Data type             | Valid range                                         | required |
 * |---------------|-------------------------------------------------|-----------------------|-----------------------------------------------------|----------|
 * | device        | The device whose DRAM to write data into        | Device *              |                                                     | Yes      |
 * | logical_core  | Logical coordinate of core whose L1 to write to | CoreCoord            | On Grayskull, any valid logical worker coordinate   | Yes      |
 * | address       | Starting address in L1 to write into            | uint32_t              | Any non-reserved address in L1 that fits for buffer | Yes      |
 * | host_buffer   | Buffer on host whose data to copy from          | std::vector<uint32_t> | Buffer must fit into L1                             | Yes      |
 */
bool WriteToDeviceL1(Device *device, const CoreCoord &logical_core, uint32_t address, std::vector<uint32_t> &host_buffer);

/**
 * Copy data from an L1 buffer into a host buffer. Must be a buffer, and not a CB.
 *
 * Return value: bool
 *
 * | Argument             | Description                                 | Data type             | Valid range                                       | required |
 * |----------------------|---------------------------------------------|-----------------------|---------------------------------------------------|----------|
 * | device               | The device whose DRAM to read data from     | Device *              |                                                   | Yes      |
 * | logical_core         | Logical coordinate of core whose L1 to read | CoreCoord            | On Grayskull, any valid logical worker coordinate | Yes      |
 * | address              | Starting address in L1 to read from         | uint32_t              |                                                   | Yes      |
 * | size                 | Size of L1 buffer in bytes                  | uint32_t              |                                                   | Yes      |
 * | host_buffer          | Buffer on host to copy data into            | std::vector<uint32_t> | Buffer must fit L1 buffer                         | Yes      |
 */
bool ReadFromDeviceL1(Device *device, const CoreCoord &logical_core, uint32_t address, uint32_t size, std::vector<uint32_t> &host_buffer);

// ==================================================
//           COMPILE & EXECUTE KENRNELS
//
// ==================================================

// Compiles all kernels within the program, and generates their binaries
bool CompileProgram(
    Device *device,                 // Device - device doesn't have to be initialized to compile the program.
    Program &program,               // Program
    bool profile_kernel = false);   // Set the compile flag for kernels to report profiling timer marks

// Configures a given device with a given program.
// - Loads all kernel binaries into L1s of assigned Tensix cores
// - Configures circular buffers (inits regs with buffer data)
// - Takes the device out of reset
bool ConfigureDeviceWithProgram(Device *device, const Program &program);

/**
 * Set runtime args on program. Program tracks which processor on core receives the runtime args based on the processor that the kernel is targetting. Args are sent to the core during runtime.
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                          | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | program      | The program to which runtime args will be written                      | Program &                     |                                    | Yes      |
 * | kernel       | The kernel that will receive the runtime args                          | Kernel *                      |                                    | Yes      |
 * | logical_core | The location of the Tensix core where the runtime args will be written | const CoreCoord &             | Any logical Tensix core coordinate | Yes      |
 * | runtime_args | The runtime args to be written                                         | const std::vector<uint32_t> & |                                    | Yes      |
 */
void SetRuntimeArgs(Program &program, Kernel *kernel, const CoreCoord &logical_core, const std::vector<uint32_t> &runtime_args);

// Loads all kernel args into L1s of assigned Tensix cores
bool WriteRuntimeArgsToDevice(Device *device, DataMovementKernel *kernel, const CoreCoord &logical_core, const std::vector<uint32_t> &runtime_args);

bool WriteRuntimeArgsToDevice(Device *device, DataMovementKernel *kernel, const CoreRange &core_range, const std::vector<uint32_t> &runtime_args);

bool WriteRuntimeArgsToDevice(Device *device, DataMovementKernel *kernel, const CoreRangeSet &core_ranges, const std::vector<std::vector<uint32_t>> &runtime_args_spec);

// Launches all kernels on cores specified with kernels in the program.
// All kernels on a given Tensix core must be launched.
bool LaunchKernels(Device *device, const Program &program, bool stagger_start = false);

bool WriteToDeviceL1(Device *device, const CoreCoord &core, op_info_t op_info, int op_idx);


void GenerateBankToNocCoordHeaders(
    Device *device,
    build_kernel_for_riscv_options_t *build_options,
    const std::string &op_path);

}  // namespace tt_metal

}  // namespace tt


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
void EnqueueReadBuffer(CommandQueue& cq, Buffer& buffer, vector<u32>& dst, bool blocking);

/**
 * Writes a buffer to the device
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                          | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | cq           | The command queue object which dispatches the command to the hardware  | CommandQueue &                |                                    | Yes      |
 * | buffer       | The device buffer we are writing to                                    | Buffer &                      |                                    | Yes      |
 * | dst          | The vector we are writing to the device                                | vector<u32> &                 |                                    | Yes      |
 * | blocking     | Whether or not this is a blocking operation                            | bool                          |                                    | Yes      |
 */
void EnqueueWriteBuffer(CommandQueue& cq, Buffer& buffer, vector<u32>& src, bool blocking);

/**
 * Writes a program to the device and launches it
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                          | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | cq           | The command queue object which dispatches the command to the hardware  | CommandQueue &                |                                    | Yes      |
 * | program      | The program we are writing to the device                               | Program &                     |                                    | Yes      |
 * | runtime_args | The runtime arg mapping onto the device for each core/RISCV we are     | const RuntimeArgs&            |                                    | Yes      |
 * |              | targetting on the device                                               |                               |                                    | Yes      |
 * | blocking     | Whether or not this is a blocking operation                            | bool                          |                                    | Yes      |
 */
void EnqueueProgram(CommandQueue& cq, Program& program, const RuntimeArgs& runtime_args, bool blocking);

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
