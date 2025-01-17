// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>
#include "types.hpp"

#include <kernel_types.hpp>
#include <circular_buffer_types.hpp>
#include "tt_metal/tt_stl/any_range.hpp"

//==================================================
//                  PROGRAM MANAGEMENT
//==================================================

namespace tt::tt_metal {
namespace v1 {

MAKE_ANY_RANGE(
    SizedCircularBufferRange, stl::AnySizedInputRange<CircularBufferHandle, stl::default_any_range_capacity, 24>);

MAKE_ANY_RANGE(CircularBufferRange, stl::AnyInputRange<CircularBufferHandle, 96, 32>);

/**
 * @brief Creates a Program object, which bundles kernels, circular buffers, and semaphores for execution on the device.
 *
 * @return Program handle to the created program.
 */
ProgramHandle CreateProgram();

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
    ProgramHandle& program,
    std::string_view file_name,
    const CoreRangeSet& core_spec,
    const DataMovementConfig& config);

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
    ProgramHandle& program, std::string_view file_name, const CoreRangeSet& core_spec, const ComputeConfig& config);

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
    ProgramHandle& program, std::string_view file_name, const CoreRangeSet& core_spec, const EthernetConfig& config);

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
    ProgramHandle& program,
    const CoreRangeSet& core_spec,
    std::uint32_t initial_value,
    CoreType core_type = CoreType::WORKER);

/**
 * @brief Creates a Circular Buffer in L1 memory of specified cores and adds it to the program.
 *
 * @param program The program to which the buffer will be added.
 * @param core_spec Specifies the cores where the circular buffer will be configured.
 * @param config Configuration for the circular buffer.
 * @return CBHandle representing the Circular Buffer ID.
 */
CircularBufferHandle CreateCircularBuffer(
    ProgramHandle& program, const CoreRangeSet& core_spec, const CircularBufferConfig& config);

/**
 * @brief Gets the configuration of a circular buffer.
 *
 * @param program The program containing the circular buffer.
 * @param cb_handle Handle of the circular buffer.
 * @return Reference to the CircularBufferConfig.
 */
const CircularBufferConfig& GetCircularBufferConfig(ProgramHandle& program, CircularBufferHandle cb_handle);

/**
 * @brief Retrieves the circular buffers associated with the program.
 *
 * @param program The program to query.
 * @return A sized input range of CircularBufferHandle.
 */
SizedCircularBufferRange GetCircularBuffers(ProgramHandle& program);

/**
 * @brief Retrieves the circular buffers associated with the program on a specific core range.
 *
 * @param program The program to query.
 * @param cr The core range to consider.
 * @return An input range of CircularBufferHandle on the given core range.
 */
CircularBufferRange GetCircularBuffersOnCoreRange(ProgramHandle& program, CoreRange cr);

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
void UpdateCircularBufferTotalSize(ProgramHandle& program, CircularBufferHandle cb_handle, std::uint32_t total_size);

/**
 * @brief Updates the address of a dynamic circular buffer.
 *
 * @param program The program containing the circular buffer.
 * @param cb_handle Handle of the circular buffer.
 * @param buffer Dynamically allocated L1 buffer that shares address space with the circular buffer.
 */
void UpdateDynamicCircularBufferAddress(
    ProgramHandle& program, CircularBufferHandle cb_handle, const BufferHandle& buffer);

namespace experimental {

/**
 * @brief Creates a Circular Buffer in L1 memory of specified cores using the address space of the
 * global circular bufferand adds it to the program.
 *
 * @param program The program to which the buffer will be added.
 * @param core_spec Specifies the cores where the circular buffer will be configured.
 * @param config Configuration for the circular buffer.
 * @param global_circular_buffer Global circular buffer to use the address space and configuration of.
 * @return CBHandle representing the Circular Buffer ID.
 */
CBHandle CreateCircularBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const CircularBufferConfig& config,
    const GlobalCircularBuffer& global_circular_buffer);

/**
 * @brief Updates the address of a dynamic global circular buffer.
 *
 * @param program The program containing the circular buffer.
 * @param cb_handle Handle of the circular buffer.
 * @param buffer Dynamically allocated global L1 buffer that shares address space with the circular buffer.
 */
void UpdateDynamicCircularBufferAddress(
    Program& program, CBHandle cb_handle, const GlobalCircularBuffer& global_circular_buffer);

}  // namespace experimental

}  // namespace v1
}  // namespace tt::tt_metal
