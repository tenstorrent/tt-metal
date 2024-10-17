// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include "types.hpp"
#include "tt_metal/impl/buffers/buffer_constants.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/dispatch/work_executor.hpp"

//==================================================
//               DEVICE MANAGEMENT
//==================================================

namespace tt::tt_metal{
namespace v1 {

/**
 * @brief Returns the number of Tenstorrent devices that can be targeted.
 *
 * @return Size_t representing the number of available devices.
 */
size_t GetNumAvailableDevices();

/**
 * @brief Returns the number of Tenstorrent devices connected via PCIe.
 *
 * @return Size_t representing the number of PCIe devices.
 */
size_t GetNumPCIeDevices();

/**
 * @brief Retrieves the PCIe device ID for a given device ID.
 *
 * @param device_id ID of the device to query.
 * @return Chip ID of the PCIe device.
 */
chip_id_t GetPCIeDeviceID(chip_id_t device_id);

/**
 * @brief Instantiates a Device object.
 *
 * @param device_id ID of the device to target (0 to GetNumAvailableDevices() - 1).
 * @param num_hw_cqs Number of hardware command queues (default: 1, valid range: 1 to 2).
 * @param l1_small_size L1 small space to reserve (default: DEFAULT_L1_SMALL_SIZE).
 * @param trace_region_size Trace region size to reserve (default: DEFAULT_TRACE_REGION_SIZE).
 * @param dispatch_core_type Dispatch core type to use (default: DispatchCoreType::WORKER).
 * @return Device handle to the created device.
 */
Device CreateDevice(
    chip_id_t device_id,
    uint8_t num_hw_cqs = 1,
    size_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
    size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
    DispatchCoreType dispatch_core_type = DispatchCoreType::WORKER);

/**
 * @brief Resets and closes the device.
 *
 * @param device Handle to the device to close.
 * @return True if the device was successfully closed; otherwise, false.
 */
bool CloseDevice(Device device);


/**
 * @brief Deallocates all buffers on the device.
 */
void DeallocateBuffers(Device device);


/**
 * @brief Dumps device-side profiler data to a CSV log.
 *
 * @param device The device holding the program being profiled.
 * @param worker_cores CoreRangeSet of worker cores being profiled.
 * @param last_dump If true, indicates the last dump before process termination.
 */
void DumpDeviceProfileResults(Device device, const CoreRangeSet &worker_cores, bool last_dump = false);


/**
 * @brief Retrieves the architecture of the device.
 *
 * @param device The device to query.
 * @return ARCH representing the device architecture.
 */
ARCH GetArch(Device device);

/**
 * @brief Retrieves the ID of the device.
 *
 * @param device The device to query.
 * @return Chip ID of the device.
 */
chip_id_t GetId(Device device);

/**
 * @brief Retrieves the number of DRAM channels on the device.
 *
 * @param device The device to query.
 * @return Number of DRAM channels.
 */
int GetNumDramChannels(Device device);

/**
 * @brief Retrieves the available L1 size per worker core on the device.
 *
 * @param device The device to query.
 * @return L1 size per core in bytes.
 */
uint32_t GetL1SizePerCore(Device device);

/**
 * @brief Computes the storage grid size for the device.
 *
 * @param device The device to query.
 * @return CoreCoord representing the storage grid size.
 */
CoreCoord GetComputeWithStorageGridSize(Device device);

/**
 * @brief Retrieves the DRAM grid size for the device.
 *
 * @param device The device to query.
 * @return CoreCoord representing the DRAM grid size.
 */
CoreCoord GetDramGridSize(Device device);

/**
 * @brief Converts a logical core coordinate to a physical core coordinate.
 *
 * @param device The device to query.
 * @param logical_core The logical core coordinate.
 * @param core_type The type of the core.
 * @return CoreCoord representing the physical core coordinate.
 */
CoreCoord PhysicalCoreFromLogical(Device device, const CoreCoord &logical_core, const CoreType &core_type);

/**
 * @brief Retrieves the worker core coordinate corresponding to a logical core.
 *
 * @param device The device to query.
 * @param logical_core The logical core coordinate.
 * @return CoreCoord representing the worker core coordinate.
 */
CoreCoord WorkerCoreFromLogical(Device device, const CoreCoord &logical_core);

/**
 * @brief Retrieves the Ethernet core coordinate corresponding to a logical core.
 *
 * @param device The device to query.
 * @param logical_core The logical core coordinate.
 * @return CoreCoord representing the Ethernet core coordinate.
 */
CoreCoord EthernetCoreFromLogical(Device device, const CoreCoord &logical_core);

/**
 * @brief Enables the program cache on the device.
 *
 * @param device The device to modify.
 */
void EnableProgramCache(Device device);

/**
 * @brief Disables and clears the program cache on the device.
 *
 * @param device The device to modify.
 */
void DisableAndClearProgramCache(Device device);

/**
 * @brief Pushes a work function onto the device's work queue.
 *
 * @param device The device to which the work will be pushed.
 * @param work The work function to execute.
 * @param blocking Indicates whether the operation should be blocking (default: false).
 */
void PushWork(Device device, std::function<void()> &&work, bool blocking = false);

/**
 * @brief Pushes a shared work function onto the device's work queue.
 *
 * @param device The device to which the work will be pushed.
 * @param work Shared pointer to the work function to execute.
 * @param blocking Indicates whether the operation should be blocking (default: false).
 */
void PushWork(Device device, std::function<void()> work, bool blocking = false);

/**
 * @brief Synchronizes operations on the given device.
 *
 * @param device The device to synchronize.
 */
void Synchronize(Device device);

/**
 * @brief Retrieves a list of Ethernet socket coordinates connected to a specific chip ID.
 *
 * @param device The device to query.
 * @param connected_chip_id The connected chip ID.
 * @return Vector of CoreCoord representing Ethernet socket coordinates.
 */
std::vector<CoreCoord> GetEthernetSockets(Device device, chip_id_t connected_chip_id);

/**
 * @brief Returns the number of banks for a specific buffer type on the device.
 *
 * @param device The device to query.
 * @param buffer_type The type of buffer.
 * @return Number of banks.
 */
uint32_t GetNumBanks(Device device, const BufferType &buffer_type);

/**
 * @brief Computes the offset of a specific bank for a buffer type on the device.
 *
 * @param device The device to query.
 * @param buffer_type The type of buffer.
 * @param bank_id The ID of the bank.
 * @return Offset of the bank.
 */
int32_t GetBankOffset(Device device, BufferType buffer_type, uint32_t bank_id);

/**
 * @brief Retrieves bank IDs associated with a logical core for a given buffer type.
 *
 * @param device The device to query.
 * @param buffer_type The type of buffer.
 * @param logical_core The logical core coordinate.
 * @return Reference to a vector of bank IDs.
 */
const std::vector<uint32_t> &BankIdsFromLogicalCore(Device device, BufferType buffer_type, const CoreCoord &logical_core);


/**
 * @brief Retrieves the machine epsilon for the SFPU on the device.
 *
 * @param device The device to query.
 * @return SFPU machine epsilon.
 */
float GetSfpuEps(Device device);

/**
 * @brief Retrieves the representation of NaN for the SFPU on the device.
 *
 * @param device The device to query.
 * @return SFPU NaN value.
 */
float GetSfpuNan(Device device);

/**
 * @brief Retrieves the representation of infinity for the SFPU on the device.
 *
 * @param device The device to query.
 * @return SFPU infinity value.
 */
float GetSfpuInf(Device device);

/**
 * @brief Retrieves a command queue from the device for a given queue ID.
 *
 * @param device The device to query.
 * @param cq_id The command queue ID.
 * @return CommandQueue handle.
 */
CommandQueue GetCommandQueue(Device device, size_t cq_id);

/**
 * @brief Retrieves the default command queue for the given device.
 *
 * @param device The device to query.
 * @return CommandQueue handle.
 */
CommandQueue GetDefaultCommandQueue(Device device);

/**
 * @brief Retrieves the current worker mode of the device.
 *
 * @param device The device to query.
 * @return WorkExecutorMode representing the current worker mode.
 */
WorkExecutorMode GetWorkerMode(Device device);

/**
 * @brief Retrieves the number of entries in the program cache on the device.
 *
 * @param device The device to query.
 * @return Number of program cache entries.
 */
std::size_t GetNumProgramCacheEntries(Device device);

/**
 * @brief Checks if the current execution is in the main thread for the device.
 *
 * @param device The device to query.
 * @return True if in the main thread; otherwise, false.
 */
bool InMainThread(Device device);


} // namespace v1
} // namespace tt::tt_metal
