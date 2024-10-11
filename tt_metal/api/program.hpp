// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>
#include "types.hpp"

#include "tt_metal/impl/kernels/kernel_types.hpp"
#include "tt_metal/impl/buffers/circular_buffer_types.hpp"

//==================================================
//                  PROGRAM MANAGEMENT
//==================================================

namespace tt::tt_metal{
namespace v1 {

/**
 * @brief Creates a Program object, which bundles kernels, circular buffers, and semaphores for execution on the device.
 *
 * @return Program handle to the created program.
 */
Program CreateProgram();


/**
 * @brief Creates a data movement or compute kernel and adds it to the program.
 *
 * @param program The program to which this kernel will be added.
 * @param file_name Path to the kernel source file.
 * @param core_spec Specifies the cores on which the kernel will be placed.
 * @param config DataMovementConfig for the kernel.
 * @return KernelHandle representing the kernel ID.
 */
KernelHandle CreateKernel(
    Program program,
    std::string_view file_name,
    const CoreRangeSet &core_spec,
    const DataMovementConfig &config);

/**
 * @brief Creates a data movement or compute kernel and adds it to the program.
 *
 * @param program The program to which this kernel will be added.
 * @param file_name Path to the kernel source file.
 * @param core_spec Specifies the cores on which the kernel will be placed.
 * @param config ComputeConfig for the kernel.
 * @return KernelHandle representing the kernel ID.
 */
KernelHandle CreateKernel(
    Program program,
    std::string_view file_name,
    const CoreRangeSet &core_spec,
    const ComputeConfig &config);

/**
 * @brief Creates a data movement or compute kernel and adds it to the program.
 *
 * @param program The program to which this kernel will be added.
 * @param file_name Path to the kernel source file.
 * @param core_spec Specifies the cores on which the kernel will be placed.
 * @param config EthernetConfig for the kernel.
 * @return KernelHandle representing the kernel ID.
 */
KernelHandle CreateKernel(
    Program program,
    std::string_view file_name,
    const CoreRangeSet &core_spec,
    const EthernetConfig &config);


/**
 * @brief Initializes a semaphore on specified cores.
 *
 * @param program The program to which the semaphore will be added.
 * @param core_spec Range of cores using the semaphore.
 * @param initial_value Initial value of the semaphore.
 * @param core_type Core type on which to create the semaphore (default: CoreType::WORKER).
 * @return Semaphore address as a uint32_t.
 */
uint32_t CreateSemaphore(
    Program program,
    const CoreRangeSet &core_spec,
    uint32_t initial_value,
    CoreType core_type = CoreType::WORKER);


/**
 * @brief Creates a Circular Buffer in L1 memory of specified cores and adds it to the program.
 *
 * @param program The program to which the buffer will be added.
 * @param core_spec Specifies the cores where the circular buffer will be configured.
 * @param config Configuration for the circular buffer.
 * @return CBHandle representing the Circular Buffer ID.
 */
CBHandle CreateCircularBuffer(
    Program program,
    const CoreRangeSet &core_spec,
    const CircularBufferConfig &config);

/**
 * @brief Gets the configuration of a circular buffer.
 *
 * @param program The program containing the circular buffer.
 * @param cb_handle Handle of the circular buffer.
 * @return Reference to the CircularBufferConfig.
 */
const CircularBufferConfig &GetCircularBufferConfig(Program program, CBHandle cb_handle);

/**
 * @brief Retrieves the circular buffers associated with the program.
 *
 * @param program The program to query.
 * @return Reference to a vector of shared pointers to CircularBuffer objects.
 */
const std::vector<CircularBuffer> &GetCircularBuffers(Program program);

/**
 * @brief Retrieves the circular buffers associated with the program on a specific core range.
 *
 * @param program The program to query.
 * @param cr The core range to consider.
 * @return Vector of shared pointers to CircularBuffer objects on the core range.
 */
std::vector<CircularBuffer> GetCircularBuffersOnCoreRange(Program program, const CoreRange &cr);


//==================================================
//                 PROGRAM FUNCTIONS
//==================================================

/**
 * @brief Updates the total size of a circular buffer.
 *
 * @param program The program containing the circular buffer.
 * @param cb_handle Handle of the circular buffer.
 * @param total_size New total size of the circular buffer in bytes.
 */
void UpdateCircularBufferTotalSize(Program program, CBHandle cb_handle, uint32_t total_size);

/**
 * @brief Updates the address of a dynamic circular buffer.
 *
 * @param program The program containing the circular buffer.
 * @param cb_handle Handle of the circular buffer.
 * @param buffer Dynamically allocated L1 buffer that shares address space with the circular buffer.
 */
void UpdateDynamicCircularBufferAddress(Program program, CBHandle cb_handle, const Buffer buffer);


/**
 * @brief Captures dependencies for multi-device execution in the program.
 *
 * @param program The program to modify.
 */
void CaptureMultiDeviceDependencies(Program program);


} // namespace v1
} // namespace tt::tt_metal
