// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

#include "tt_metal/impl/buffers/buffer_constants.hpp"
#include "tt_metal/impl/dispatch/work_executor.hpp"
#include "types.hpp"

//==================================================
//               DEVICE MANAGEMENT
//==================================================

namespace tt::tt_metal {
namespace v1 {

/**
 * @brief Returns the number of Tenstorrent devices that can be targeted.
 *
 * @return Size_t representing the number of available devices.
 */
std::size_t GetNumAvailableDevices();

/**
 * @brief Returns the number of Tenstorrent devices connected via PCIe.
 *
 * @return Size_t representing the number of PCIe devices.
 */
std::size_t GetNumPCIeDevices();

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
 * @param config Device configuration parameters
 * @return Device handle to the created device.
 */
DeviceHandle CreateDevice(chip_id_t device_id, DeviceConfig config);

/**
 * @brief Resets and closes the device.
 *
 * @param device Handle to the device to close.
 * @return True if the device was successfully closed; otherwise, false.
 */
bool CloseDevice(DeviceHandle device);

/**
 * @brief Deallocates all buffers on the device.
 */
void DeallocateBuffers(DeviceHandle device);

/**
 * @brief Dumps device-side profiler data to a CSV log.
 *
 * @param device The device holding the program being profiled.
 * @param worker_cores CoreRangeSet of worker cores being profiled.
 * @param last_dump If true, indicates the last dump before process termination.
 */
void DumpDeviceProfileResults(DeviceHandle device, const CoreRangeSet &worker_cores, bool last_dump = false);

/**
 * @brief Retrieves the architecture of the device.
 *
 * @param device The device to query.
 * @return ARCH representing the device architecture.
 */
ARCH GetArch(DeviceHandle device);

/**
 * @brief Retrieves the ID of the device.
 *
 * @param device The device to query.
 * @return Chip ID of the device.
 */
chip_id_t GetId(DeviceHandle device);

/**
 * @brief Retrieves the number of DRAM channels on the device.
 *
 * @param device The device to query.
 * @return Number of DRAM channels.
 */
int GetNumDramChannels(DeviceHandle device);

/**
 * @brief Retrieves the available L1 size per worker core on the device.
 *
 * @param device The device to query.
 * @return L1 size per core in bytes.
 */
uint32_t GetL1SizePerCore(DeviceHandle device);

/**
 * @brief Computes the storage grid size for the device.
 *
 * @param device The device to query.
 * @return CoreCoord representing the storage grid size.
 */
CoreCoord GetComputeWithStorageGridSize(DeviceHandle device);

/**
 * @brief Retrieves the DRAM grid size for the device.
 *
 * @param device The device to query.
 * @return CoreCoord representing the DRAM grid size.
 */
CoreCoord GetDramGridSize(DeviceHandle device);

/**
 * @brief Converts a logical core coordinate to a physical core coordinate.
 *
 * @param device The device to query.
 * @param logical_core The logical core coordinate.
 * @param core_type The type of the core.
 * @return CoreCoord representing the physical core coordinate.
 */
CoreCoord PhysicalCoreFromLogical(DeviceHandle device, CoreCoord logical_core, CoreType core_type);

/**
 * @brief Retrieves the worker core coordinate corresponding to a logical core.
 *
 * @param device The device to query.
 * @param logical_core The logical core coordinate.
 * @return CoreCoord representing the worker core coordinate.
 */
CoreCoord WorkerCoreFromLogical(DeviceHandle device, CoreCoord logical_core);

/**
 * @brief Retrieves the Ethernet core coordinate corresponding to a logical core.
 *
 * @param device The device to query.
 * @param logical_core The logical core coordinate.
 * @return CoreCoord representing the Ethernet core coordinate.
 */
CoreCoord EthernetCoreFromLogical(DeviceHandle device, CoreCoord logical_core);

/**
 * @brief Enables the program cache on the device.
 *
 * @param device The device to modify.
 */
void EnableProgramCache(DeviceHandle device);

/**
 * @brief Disables and clears the program cache on the device.
 *
 * @param device The device to modify.
 */
void DisableAndClearProgramCache(DeviceHandle device);

/**
 * @brief Pushes a shared work function onto the device's work queue.
 *
 * @param device The device to which the work will be pushed.
 * @param work Shared pointer to the work function to execute.
 * @param blocking Indicates whether the operation should be blocking (default: false).
 */
void PushWork(DeviceHandle device, std::function<void()> work, bool blocking = false);

/**
 * @brief Synchronizes operations on the given device.
 *
 * @param device The device to synchronize.
 */
void Synchronize(DeviceHandle device);

/**
 * @brief Retrieves a list of Ethernet socket coordinates connected to a specific chip ID.
 *
 * @param device The device to query.
 * @param connected_chip_id The connected chip ID.
 * @return Vector of CoreCoord representing Ethernet socket coordinates.
 */
std::vector<CoreCoord> GetEthernetSockets(DeviceHandle device, chip_id_t connected_chip_id);

/**
 * @brief Returns the number of banks for a specific buffer type on the device.
 *
 * @param device The device to query.
 * @param buffer_type The type of buffer.
 * @return Number of banks.
 */
std::uint32_t GetNumBanks(DeviceHandle device, BufferType buffer_type);

/**
 * @brief Computes the offset of a specific bank for a buffer type on the device.
 *
 * @param device The device to query.
 * @param buffer_type The type of buffer.
 * @param bank_id The ID of the bank.
 * @return Offset of the bank.
 */
std::int32_t GetBankOffset(DeviceHandle device, BufferType buffer_type, std::uint32_t bank_id);

/**
 * @brief Retrieves bank IDs associated with a logical core for a given buffer type.
 *
 * @param device The device to query.
 * @param buffer_type The type of buffer.
 * @param logical_core The logical core coordinate.
 * @return Reference to a vector of bank IDs.
 */
const std::vector<uint32_t> &BankIdsFromLogicalCore(
    DeviceHandle device, BufferType buffer_type, CoreCoord logical_core);

/**
 * @brief Retrieves the machine epsilon for the SFPU on the device.
 *
 * @param device The device to query.
 * @return SFPU machine epsilon.
 */
float GetSfpuEps(DeviceHandle device);

/**
 * @brief Retrieves the representation of NaN for the SFPU on the device.
 *
 * @param device The device to query.
 * @return SFPU NaN value.
 */
float GetSfpuNan(DeviceHandle device);

/**
 * @brief Retrieves the representation of infinity for the SFPU on the device.
 *
 * @param device The device to query.
 * @return SFPU infinity value.
 */
float GetSfpuInf(DeviceHandle device);

/**
 * @brief Retrieves a command queue from the device for a given queue ID.
 *
 * @param device The device to query.
 * @param cq_id The command queue ID.
 * @return CommandQueue handle.
 */
CommandQueueHandle GetCommandQueue(DeviceHandle device, std::uint8_t cq_id);

/**
 * @brief Retrieves the default command queue for the given device.
 *
 * @param device The device to query.
 * @return CommandQueue handle.
 */
CommandQueueHandle GetDefaultCommandQueue(DeviceHandle device);

/**
 * @brief Retrieves the current worker mode of the device.
 *
 * @param device The device to query.
 * @return WorkExecutorMode representing the current worker mode.
 */
WorkExecutorMode GetWorkerMode(DeviceHandle device);

/**
 * @brief Retrieves the number of entries in the program cache on the device.
 *
 * @param device The device to query.
 * @return Number of program cache entries.
 */
std::size_t GetNumProgramCacheEntries(DeviceHandle device);

/**
 * @brief Checks if the current execution is in the main thread for the device.
 *
 * @param device The device to query.
 * @return True if in the main thread; otherwise, false.
 */
bool InMainThread(DeviceHandle device);

}  // namespace v1
}  // namespace tt::tt_metal
