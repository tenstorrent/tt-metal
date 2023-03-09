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
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/buffers/circular_buffer.hpp"
#include "tt_metal/impl/buffers/interleaved_buffer.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/device/host.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/impl/program.hpp"
#include "llrt/llrt.hpp"

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
//                  HOST API: profiler
// ==================================================

// dump host side profiler results
void dumpProfilerResults(std::string name_append = "", bool add_header = false);

// set profiler log dir
void setProfilerDir(std::string output_dir = "");

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
 * | Argument | Description                | Type     | Valid Range | Required |
 * |----------|----------------------------|----------|-------------|----------|
 * | device   | Pointer to a device object | Device * |             | True     |
 */
bool InitializeDevice(Device *device);

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

void StartDebugPrintServer(Device *device);

// ==================================================
//                  HOST API: program & kernels
// ==================================================
// Kernel args are only initialized with compile time arguments

/**
 * Creates kernel arguments for a data movement kernel
 *
 * Return value: DataMovementKernelArgs *
 *
 * |      Argument     |                                                      Description                                                      |        Data type        |    Valid range    | required |
 * |:-----------------:|:---------------------------------------------------------------------------------------------------------------------:|:-----------------------:|:-----------------:|----------|
 * | logical_core      | The location of the Tensix core with a kernel that receives these arguments (Logical co-ordinates)                    | const tt_xy_pair &      | {0, 0} –> {9, 11} | Yes      |
 * | runtime_args      | Collection of runtime args for the kernel. Required kernel arguments are located in the *.cpp file of the kernel      | std::vector<uint32_t> & |                   | Yes      |
 * | compile_time_args | Collection of compile time args for the kernel. Required kernel arguments are located in the *.cpp file of the kernel | std::vector<uint32_t> & | Default empty     | Yes      |
 */
DataMovementKernelArgs *InitializeCompileTimeDataMovementKernelArgs(const tt_xy_pair &logical_core, const std::vector<uint32_t> &compile_time_args);

/**
 * Creates the same kernel arguments for a range of cores
 *
 * Return value: DataMovementKernelArgs *
 *
 * | Argument          | Description                                                                                                           | Data type                                             | Valid range                                            | required |
 * |-------------------|-----------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|--------------------------------------------------------|----------|
 * | core_range        | The range of the Tensix co-ordinates with a kernel that receives these arguments (Logical co-ordinates)               | const CoreRange & (std::pair<tt_xy_pair, tt_xy_pair>) | Any range encompassing cores within {0 , 0} –> {9, 11} | Yes      |
 * | runtime_args      | Collection of runtime args for the kernel. Required kernel arguments are located in the *.cpp file of the kernel      | std::vector<uint32_t> &                               |                                                        | Yes      |
 * | compile_time_args | Collection of compile time args for the kernel. Required kernel arguments are located in the *.cpp file of the kernel | std::vector<uint32_t> &                               | Default empty                                          | Yes      |
 */
DataMovementKernelArgs *InitializeCompileTimeDataMovementKernelArgs(const CoreRange &core_range, const std::vector<uint32_t> &compile_time_args);

/**
 * Creates kernel arguments specified by a combination of single core co-ordinates or a range of core co-ordinates
 *
 * Return value: DataMovementKernelArgs *
 *
 * |      Argument     |                                                                 Description                                                                |                               Data type                               |                                                      Valid range                                                      | required |
 * |:-----------------:|:------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------:|----------|
 * | core_blocks       | A collection containing a single Tensix co-ordinate or a range of Tensix co-ordinates that receives these arguments (Logical co-ordinates) | const CoreBlocks & (std::vector<std::variant<tt_xy_pair, CoreRange>>) | A single core or range encompassing cores within {0 , 0} –> {9, 11}                                                   | Yes      |
 * | runtime_args      | A collection of runtime args. Required kernel arguments are located in the *.cpp file of the kernel                                        | std::vector<std::vector<uint32_t>> &                                  | Same size as core_blocks. Args are assigned to core or range of cores from core_blocks in order of runtime_args.      | Yes      |
 * | compile_time_args | A collection of compile time args. Required kernel arguments are located in the *.cpp file of the kernel                                   | std::vector<std::vector<uint32_t>> &                                  | Same size as core_blocks. Args are assigned to core or range of cores from core_blocks in order of compile_time_args. | Yes      |
 */
DataMovementKernelArgs *InitializeCompileTimeDataMovementKernelArgs(const CoreBlocks &core_blocks, const std::vector<std::vector<uint32_t>> &compile_time_args_spec);

/**
 * Creates kernel arguments for compute kernel
 *
 * Return value: ComputeKernelArgs *
 *
 * |        Argument        |                                                Description                                                |      Data type     |    Valid range    | required |
 * |:----------------------:|:---------------------------------------------------------------------------------------------------------:|:------------------:|:-----------------:|----------|
 * | logical_core           | The location of the Tensix core with a kernel that receives these arguments (Logical co-ordinates)        | const tt_xy_pair & | {0, 0} –> {9, 11} | Yes      |
 * | compile_time_args      | A pointer to the struct containing the args. Struct definition is located in the *.cpp file of the kernel | void *             |                   | Yes      |
 * | compile_time_args_size | Size of struct containing the kernel arguments                                                            | size_t             | 0 to 512 Bytes    | Yes      |
 */
ComputeKernelArgs *InitializeCompileTimeComputeKernelArgs(const tt_xy_pair &logical_core, const vector<uint32_t> &compile_time_args);

/**
 * Creates the same kernel arguments for a range of cores
 *
 * Return value: ComputeKernelArgs *
 *
 * |        Argument        |                                                Description                                                |                       Data type                       |                       Valid range                      | required |
 * |:----------------------:|:---------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------:|:------------------------------------------------------:|----------|
 * | core_range             | The range of the Tensix co-ordinates with a kernel that receives these arguments (Logical co-ordinates)   | const CoreRange & (std::pair<tt_xy_pair, tt_xy_pair>) | Any range encompassing cores within {0 , 0} –> {9, 11} | Yes      |
 * | compile_time_args      | A pointer to the struct containing the args. Struct definition is located in the *.cpp file of the kernel | void *                                                |                                                        | Yes      |
 * | compile_time_args_size | Size of struct containing the kernel arguments                                                            | size_t                                                | 0 to 512 Bytes                                         | Yes      |
 */
ComputeKernelArgs *InitializeCompileTimeComputeKernelArgs(const CoreRange &core_range, const vector<uint32_t> &compile_time_args);

/**
 * Creates kernel arguments specified by a combination of single core co-ordinates or a range of core co-ordinates
 *
 * Return value: ComputeKernelArgs *
 *
 * | Argument               | Description                                                                                                                                | Data type                                                             | Valid range                                                                                                           | required |
 * |------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|----------|
 * | core_blocks            | A collection containing a single Tensix co-ordinate or a range of Tensix co-ordinates that receives these arguments (Logical co-ordinates) | const CoreBlocks & (std::vector<std::variant<tt_xy_pair, CoreRange>>) | A single core or range encompassing cores within {0 , 0} –> {9, 11}                                                   | Yes      |
 * | compile_time_args      | A collection of pointers to structs containing the args. Struct definition is located in the *.cpp file of the kernel.                     | const std::vector<void *> &                                           | Same size as core_blocks. Args are assigned to core or range of cores from core_blocks in order of compile_time_args. | Yes      |
 * | compile_time_args_size | Size of struct containing the kernel arguments                                                                                             | size_t                                                                | 0 to 512 Bytes                                                                                                        | Yes      |
 */
ComputeKernelArgs *InitializeCompileTimeComputeKernelArgs(const CoreBlocks &core_blocks, const std::vector<std::vector<uint32_t>> &compile_time_args_spec);
/**
 * Creates a single core data movement kernel and adds it to the program.
 *
 * Return value: DataMovementKernel *
 *
 * | Argument       | Description                                                                                                  | Data type                | Valid range                                                    | required |
 * |----------------|--------------------------------------------------------------------------------------------------------------|--------------------------|----------------------------------------------------------------|----------|
 * | program        | The program to which this kernel will be added to                                                            | Program *                |                                                                | Yes      |
 * | file_name      | Name of file containing the kernel                                                                           | const std::string        |                                                                | Yes      |
 * | core           | The location of the Tensix core on which the kernel will execute (Logical co-ordinates)                      | const tt_xy_pair &       | {0, 0} –> {9, 11}                                              | Yes      |
 * | kernel_args    | Compile and runtime kernel arguments passed at compile time and runtime respectively                         | DataMovementKernelArgs * |                                                                | Yes      |
 * | processor_type | The target RISC-V processor on which the kernel will execute, on the given Tensix core (1 kernel per RISC-V) | enum                     | DataMovementProcessor::RISCV_0, DataMovementProcessor::RISCV_1 | Yes      |
 * | noc            | The NoC ID on which the kernel will perform data transfers                                                   | enum                     | RISCV_0_default, RISCV_1_default, NOC_0, NOC_1,                | Yes      |
 */
DataMovementKernel *CreateDataMovementKernel(
    Program *program,
    const std::string &file_name,
    const tt_xy_pair &core,
    DataMovementKernelArgs *kernel_args,
    DataMovementProcessor processor_type,
    NOC noc);

/**
 * Creates a single core data movement kernel with no default arguments and
 * adds it to the program.
 *
 * Return value: DataMovementKernel *
 *
 * | Argument       | Description                                                                                                  | Data type                | Valid range                                                    | required |
 * |----------------|--------------------------------------------------------------------------------------------------------------|--------------------------|----------------------------------------------------------------|----------|
 * | program        | The program to which this kernel will be added to                                                            | Program *                |                                                                | Yes      |
 * | file_name      | Name of file containing the kernel                                                                           | const std::string        |                                                                | Yes      |
 * | core           | The location of the Tensix core on which the kernel will execute (Logical co-ordinates)                      | const tt_xy_pair &       | {0, 0} –> {9, 11}                                              | Yes      |
 * | processor_type | The target RISC-V processor on which the kernel will execute, on the given Tensix core (1 kernel per RISC-V) | enum                     | DataMovementProcessor::RISCV_0, DataMovementProcessor::RISCV_1 | Yes      |
 * | noc            | The NoC ID on which the kernel will perform data transfers                                                   | enum                     | RISCV_0_default, RISCV_1_default, NOC_0, NOC_1,                | Yes      |
 */
DataMovementKernel *CreateDataMovementKernel(
    Program *program,
    const std::string &file_name,
    const tt_xy_pair &core,
    DataMovementProcessor processor_type,
    NOC noc);

/**
 * Creates a multi-core data movement kernel and adds it to the program.
 *
 * Return value: DataMovementKernel *
 *
 * | Argument       | Description                                                                                                  | Data type                | Valid range                                                    | required |
 * |----------------|--------------------------------------------------------------------------------------------------------------|--------------------------|----------------------------------------------------------------|----------|
 * | program        | The program to which this kernel will be added to                                                            | Program *                |                                                                | Yes      |
 * | file_name      | Name of file containing the kernel                                                                           | const std::string        |                                                                | Yes      |
 * | core_range     | The range of the Tensix co-ordinates on which the kernel will execute (Logical co-ordinates)                 | const CoreRange &        | Any range encompassing cores within {0 , 0} –> {9, 11}         | Yes      |
 * | kernel_args    | Compile and runtime kernel arguments passed at compile time and runtime respectively                         | DataMovementKernelArgs * |                                                                | Yes      |
 * | processor_type | The target RISC-V processor on which the kernel will execute, on the given Tensix core (1 kernel per RISC-V) | enum                     | DataMovementProcessor::RISCV_0, DataMovementProcessor::RISCV_1 | Yes      |
 * | noc            | The NoC ID on which the kernel will perform data transfers                                                   | enum                     | RISCV_0_default, RISCV_1_default, NOC_0, NOC_1,                | Yes      |
 */
DataMovementKernel *CreateDataMovementKernel(
    Program *program,
    const std::string &file_name,
    const CoreRange &core_range,
    DataMovementKernelArgs *kernel_args,
    DataMovementProcessor processor_type,
    NOC noc);

/**
 * Creates a multi-core data movement kernel with no default arguments and adds
 * it to the program.
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
    Program *program,
    const std::string &file_name,
    const CoreRange &core_range,
    DataMovementProcessor processor_type,
    NOC noc);

/**
 * Creates a single core compute kernel object, and adds it to the program.
 *
 * Return value: ComputeKernel *
 *
 * |     Argument     |                                       Description                                       |      Data type      |      Valid range      | required |
 * |:----------------:|:---------------------------------------------------------------------------------------:|:-------------------:|:---------------------:|----------|
 * | program          | The program to which this kernel will be added to                                       | Program *           |                       | Yes      |
 * | file_name        | Name of file containing the kernel                                                      | const std::string   |                       | Yes      |
 * | core             | The location of the Tensix core on which the kernel will execute (Logical co-ordinates) | const tt_xy_pair &  | {0, 0} –> {9, 11}     | Yes      |
 * | kernel_args      | Kernel arguments, passed at compile time                                                | ComputeKernelArgs * |                       | Yes      |
 * | math_fidelity    | The percision of the matrix compute engine                                              | enum                | MathFidelity::HiFi4   | Yes      |
 * | fp32_dest_acc_en | Specifies the type of accumulation performed in the matrix compute engine.              | bool                | false (for Grayskull) | Yes      |
 * | math_approx_mode | Used by the vector compute engine. (will be depricated)                                 | bool                | true, false           | Yes      |
 */
ComputeKernel *CreateComputeKernel(
    Program *program,
    const std::string &file_name,
    const tt_xy_pair &core,
    ComputeKernelArgs *kernel_args,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode);

ComputeKernel *CreateComputeKernelNew(
    Program *program,
    const std::string &file_name,
    const tt_xy_pair &core,
    ComputeKernelArgs *kernel_args,
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
 * | program          | The program to which this kernel will be added to                                            | Program *           |                                                        | Yes      |
 * | file_name        | Name of file containing the kernel                                                           | const std::string   |                                                        | Yes      |
 * | core_range       | The range of the Tensix co-ordinates on which the kernel will execute (Logical co-ordinates) | const CoreRange &   | Any range encompassing cores within {0 , 0} –> {9, 11} | Yes      |
 * | kernel_args      | Kernel arguments, passed at compile time                                                     | ComputeKernelArgs * |                                                        | Yes      |
 * | math_fidelity    | The percision of the matrix compute engine                                                   | enum                | MathFidelity::HiFi4                                    | Yes      |
 * | fp32_dest_acc_en | Specifies the type of accumulation performed in the matrix compute engine.                   | bool                | false (for Grayskull)                                  | Yes      |
 * | math_approx_mode | Used by the vector compute engine. (will be depricated)                                      | bool                | true, false                                            | Yes      |
 */
ComputeKernel *CreateComputeKernel(
    Program *program,
    const std::string &file_name,
    const CoreRange &core_range,
    ComputeKernelArgs *kernel_args,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode);

// ==================================================
//                  HOST API: buffers
// ==================================================
DramBuffer *CreateDramBuffer(Device *device, int dram_channel, uint32_t size_in_bytes);

/**
 * Creates a DRAM Buffer object. DRAM buffers exist independently from a program
 *
 * Return value: DramBuffer *
 * | Argument      | Description                                  | Type     | Valid Range                                           | Required |
 * |---------------|----------------------------------------------|----------|-------------------------------------------------------|----------|
 * | device        | The device where the DRAM buffer is created  | Device * |                                                       | Yes      |
 * | dram_channel  | DRAM channel ID                              | int      | [0, 7]                                                | Yes      |
 * | size_in_bytes | Size of DRAM buffer in bytes                 | uint32_t | DOX-TODO: valid range? 0 to 2 GB (expressed in Bytes) | Yes      |
 * | address       | Address at which the DRAM buffer will reside | uint32_t | DOX-TODO:fix range. 0 to 2 GB??                       | Yes      |
 */
DramBuffer *CreateDramBuffer(Device *device, int dram_channel, uint32_t size_in_bytes, uint32_t address);

// Allocates multiple DRAM buffers across multiple banks to store interleaved data
InterleavedDramBuffer *CreateInterleavedDramBuffer(
    Device *device,                        // Device
    int num_bank_units,                         // Single bank unit is read at a given time, unit can be tile, stick, etc
    int num_entries_per_bank_unit,              // Number of entries in single unit, e.g. tile has 512 entries because a single tile has 1024 values packed as uint32_t
    int num_bytes_per_entry);                   // Size of single entry in DRAM bank in bytes

/**
 * Creates an L1 buffer and adds it to the program.
 *
 * Return value: L1Buffer *
 *
 * | Argument      | Description                                       | Type       | Valid Range                                            | Required |
 * |---------------|---------------------------------------------------|------------|--------------------------------------------------------|----------|
 * | device        | The device where the L1 buffer resides.           | Device *   |                                                        | True     |
 * | program       | The program to which buffer will be added to.     | Program *  |                                                        | True     |
 * | core          | Logical core coordinate of L1 to create buffer on | tt_xy_pair | Any logical worker core on Grayskull                   | True     |
 * | size_in_bytes | Size of DRAM buffer in Bytes                      | uint32_t   | DOX-TODO: valid range? 0 to 800 KB ?? (expressed in Bytes) | True     |
 * | address       | Address at which the DRAM buffer will reside      | uint32_t   | DOX-TODO: fix range. 200KB to 1MB ??? (expressed in Bytes) | True     |
 */
L1Buffer *CreateL1Buffer(Program *program, Device *device, const tt_xy_pair &core, uint32_t size_in_bytes, uint32_t address);

/**
 * Creates a Circular Buffer (CB) in L1 memory and adds it to the program.
 *
 * Return value: CircularBuffer *
 *
 * | Argument      | Description                                                                    | Type               | Valid Range                             | Required |
 * |---------------|--------------------------------------------------------------------------------|--------------------|-----------------------------------------|----------|
 * | program       | The program to which buffer will be added to.                                  | Program *          |                                         | True     |
 * | device        | The device where the L1 buffer resides.                                        | Device *           |                                         | True     |
 * | buffer_index  | The index/ID of the CB.                                                        | uint32_t           | 0 to 32 DOX-TODO: specify more detail here. | True     |
 * | core          | The location of the Tensix core on which the CB will reside (SoC co-ordinates) | const tt_xy_pair & | DOX-TODO: { , } –> { , }                    | True     |
 * | num_tiles     | Total number of tiles to be stored in the CB                                   | uint32_t           | DOX-TODO: range?                            | True     |
 * | size_in_bytes | Size of CB buffer in Bytes                                                     | uint32_t           | 0 to 1 MB (DOX-TODO: in Bytes)              | True     |
 * | l1_address    | Address at which the CB buffer will reside                                     | uint32_t           | 200 kB to 1MB (DOX-TODO: in bytes)          | True     |
 * | data_format   | The format of the data to be stored in the CB                                  | DataFormat enum    | DataFormat::Float16_b                   | True     |
 */
CircularBuffer *CreateCircularBuffer(
    Program *program,
    Device *device,
    uint32_t buffer_index,
    const tt_xy_pair &core,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    uint32_t l1_address,
    DataFormat data_format);

// ==================================================
//           COMPILE & EXECUTE KENRNELS
//
// ==================================================

// Compiles all kernels within the program, and generates their binaries
bool CompileProgram(
    Device *device,                 // Device - device doesn't have to be initialized to compile the program.
    Program *program,               // Program
    bool skip_hlkc,                 // Skips HLK to LLK compilation for all compute kernels
    bool profile_kernel = false);   // Set the compile flag for kernels to report profiling timer marks

bool CompileProgramNew(
    Device *device,
    Program *program,
    bool profile_kernel = false);


// Configures a given device with a given program.
// - Loads all kernel binaries into L1s of assigned Tensix cores
// - Configures circular buffers (inits regs with buffer data)
// - Takes the device out of reset
bool ConfigureDeviceWithProgram(Device *device, Program *program);

// Loads all kernel args into L1s of assigned Tensix cores
bool WriteRuntimeArgsToDevice(Device *device, DataMovementKernel *kernel, const tt_xy_pair &logical_core, const std::vector<uint32_t> &runtime_args);

bool WriteRuntimeArgsToDevice(Device *device, DataMovementKernel *kernel, const CoreRange &core_range, const std::vector<uint32_t> &runtime_args);

bool WriteRuntimeArgsToDevice(Device *device, DataMovementKernel *kernel, const CoreBlocks &core_blocks, const std::vector<std::vector<uint32_t>> &runtime_args_spec);

// Launches all kernels on cores specified with kernels in the program.
// All kernels on a given Tensix core must be launched.
bool LaunchKernels(Device *device, Program *program, bool stagger_start = false);

// Dump all device side profiling results
void ReadDeviceSideProfileData(Device *device, Program *program);

//
/**
 * Copy data from a device DRAM channel to a host buffer
 *
 * Returns: bool
 *
 * | Argument    | Description                                     | Data type             | Valid range                               | required |
 * |-------------|-------------------------------------------------|-----------------------|-------------------------------------------|----------|
 * | dram_buffer | Pointer of the DRAM buffer to read from         | DramBuffer *          |                                           | Yes      |
 * | host_buffer | Pointer of the buffer on host to copy data into | std::vector<uint32_t> | Host buffer must be fully fit DRAM buffer | Yes      |
 */
bool ReadFromDeviceDRAM(
    DramBuffer *dram_buffer,                // DRAM buffer on device
    std::vector<uint32_t> &host_buffer);    // Source buffer on host

/**
* Copies data from a host buffer into a buffer within the device DRAM channel
*
* Return value: bool
*
* | Argument    | Description                                     | Data type             | Valid range                                      | required |
* |-------------|-------------------------------------------------|-----------------------|--------------------------------------------------|----------|
* | dram_buffer | Pointer of the DRAM buffer to send data to      | DramBuffer *          |                                                  | Yes      |
* | host_buffer | Pointer of the buffer on host to copy data form | std::vector<uint32_t> | Host buffer must be fully fit inside DRAM buffer | Yes      |
*/
bool WriteToDeviceDRAM(
    DramBuffer *dram_buffer,              // DRAM buffer on device
    std::vector<uint32_t> &host_buffer);  // Source buffer on host)

/**
 * Copy data from a device DRAM channel to a host buffer
 *
 * Return value: bool
 *
 * | Argument     | Description                                                  | Data type             | Valid range                    | required |
 * |--------------|--------------------------------------------------------------|-----------------------|--------------------------------|----------|
 * | device       | The device whose DRAM to read data from                      | Device *              |                                | Yes      |
 * | dram_channel | Channel index of DRAM to read from                           | int                   | On Grayskull, [0, 7] inclusive | Yes      |
 * | dram_address | Starting address on DRAM channel from which to begin reading | uint32_t              |                                | Yes      |
 * | host_buffer  | Buffer on host to copy data into                             | std::vector<uint32_t> |                                | Yes      |
 * | size         | Size of buffer to read from device in bytes                  | uint32_t              |                                | Yes      |
 */
bool ReadFromDeviceDRAMChannel(
    Device *device,                      // Device
    int dram_channel,                    // DRAM channel on the device
    uint32_t dram_address,               // Address within the DRAM channel
    std::vector<uint32_t> &host_buffer,  // Source buffer on host
    uint32_t size);                      // Size of copy in Bytes

/**
 * Copies data from a host buffer into a buffer within the device DRAM channel
 *
 * Return value: bool
 *
 * | Argument     | Description                                            | Data type             | Valid range                               | required |
 * |--------------|--------------------------------------------------------|-----------------------|-------------------------------------------|----------|
 * | device       | The device whose DRAM to write data into               | Device *              |                                           | Yes      |
 * | dram_channel | Channel index of DRAM to write into                    | int                   | On Grayskull, [0, 7] inclusive            | Yes      |
 * | host_buffer  | Buffer on host to copy data from                       | std::vector<uint32_t> | Host buffer must be fully fit DRAM buffer | Yes      |
 * | dram_address | Starting address on DRAM channel to begin writing data | uint32_t              |                                           | Yes      |
 */
bool WriteToDeviceDRAMChannel(
    Device *device,                      // Device
    int dram_channel,                    // DRAM channel on the device
    std::vector<uint32_t> &host_buffer,  // Source buffer on host
    uint32_t dram_address);              // Address within the DRAM channel

// Generic interleaved reader
bool ReadFromDeviceDRAMChannelsInterleaved(
    Device *device,                             // Device
    std::vector<uint32_t> &host_buffer,         // Source buffer on host
    uint32_t start_dram_buffer_address,         // Base address, shared across all DRAM banks
    int num_bank_units,                         // Single bank unit is read at a given time, unit can be tile, stick, etc
    int num_entries_per_bank_unit,              // Number of entries in single unit, e.g. tile has 512 entries because a single tile has 1024 values packed as uint32_t
    int num_bytes_per_entry);                   // Size of single entry in DRAM bank in bytes

// Generic interleaved writer
bool WriteToDeviceDRAMChannelsInterleaved(
    Device *device,                             // Device
    std::vector<uint32_t> &host_buffer,         // Source buffer on host
    uint32_t start_dram_buffer_address,         // Base address, shared across all DRAM banks
    int num_bank_units,                         // Single bank unit is written at a given time, unit can be tile, stick, etc
    int num_entries_per_bank_unit,              // Number of entries in single unit, e.g. tile has 512 entries because a single tile has 1024 values packed as uint32_t
    int num_bytes_per_entry);                   // Size of single entry in DRAM bank in bytes

// Read from interleave tiles from 8 banks starting at a given address (same starting address for each bank)
// See comments for the Write version of this function
bool ReadFromDeviceDRAMChannelsInterleavedTiles(
    Device *device,
    uint32_t device_dram_address,
    std::vector<uint32_t> &dst_host_buffer,
    uint32_t size_bytes);

// Interleave tiles into 8 banks starting at a given address (same starting address for each bank)
// Each write is tile-sized, so performance is probably not ideal.
// This can probably be made more optimal with strided or chain or async DMAs or whatnot
bool WriteToDeviceDRAMChannelsInterleavedTiles(
    Device *device,
    std::vector<uint32_t> &host_buffer,
    uint32_t dram_address);

/**
 * Copy data from a host buffer into an L1 buffer. (Note: Current Can not be a CircularBuffer.)
 *
 * Return value: bool
 *
 * | Argument      | Description                                     | Data type             | Valid range                                         | required |
 * |---------------|-------------------------------------------------|-----------------------|-----------------------------------------------------|----------|
 * | device        | The device whose DRAM to write data into        | Device *              |                                                     | Yes      |
 * | core          | Logical coordinate of core whose L1 to write to | tt_xy_pair            | On Grayskull, any valid logical worker coordinate   | Yes      |
 * | host_buffer   | Buffer on host whose data to copy from          | std::vector<uint32_t> | Buffer must fit into L1                             | Yes      |
 * | buffer_addess | Starting address in L1 to write into            | uint32_t              | Any non-reserved address in L1 that fits for buffer | Yes      |
 */
bool WriteToDeviceL1(
    Device *device,                       // Device
    const tt_xy_pair &core,               // Tensix core
    std::vector<uint32_t> &host_buffer,   // Source buffer on host
    uint32_t buffer_addess);              // Address within L1


bool WriteToDeviceL1(
    L1Buffer *l1_buffer,                   // L1 buffer
    std::vector<uint32_t> &host_buffer);   // Source buffer on host

bool WriteToDeviceL1(
    Device *device,
    const tt_xy_pair &core,
    op_info_t op_info,
    int op_idx);

/**
 * Copy data from an L1 buffer into a host buffer. Must be a buffer, and not a CB.
 *
 * Return value: bool
 *
 * | Argument             | Description                                 | Data type             | Valid range                                       | required |
 * |----------------------|---------------------------------------------|-----------------------|---------------------------------------------------|----------|
 * | device               | The device whose DRAM to read data from     | Device *              |                                                   | Yes      |
 * | core                 | Logical coordinate of core whose L1 to read | tt_xy_pair            | On Grayskull, any valid logical worker coordinate | Yes      |
 * | device_buffer_addess | Starting address in L1 to read from         | int                   |                                                   | Yes      |
 * | host_buffer          | Buffer on host to copy data into            | std::vector<uint32_t> | Buffer must fit L1 buffer                         | Yes      |
 * | size                 | Size of L1 buffer in bytes                  | int                   |                                                   | Yes      |
 */
bool ReadFromDeviceL1(
    Device *device,                      // Device
    const tt_xy_pair &core,              // Logical Tensix core
    int device_buffer_addess,            // Address within L1
    std::vector<uint32_t> &host_buffer,  // Source buffer on host
    int size);                           // Size of copy in Bytes

bool ReadFromDeviceL1(
    L1Buffer *l1_buffer,                   // L1 buffer
    std::vector<uint32_t> &host_buffer);    // Buffer on host to copy data into

}  // namespace tt_metal

}  // namespace tt
