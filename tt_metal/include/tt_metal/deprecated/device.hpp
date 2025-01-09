// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

#include "tt_metal/impl/buffers/buffer_constants.hpp"
#include "tt_metal/impl/dispatch/work_executor.hpp"
#include "tt_metal/types.hpp"

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
 * Configuration options for CreateDevice
 */
struct CreateDeviceOptions {
    /**
     * Number of hardware command queues (default: 1, valid range: 1 to 2).
     */
    uint8_t num_hw_cqs = 1;
    /**
     * L1 small space to reserve (default: DEFAULT_L1_SMALL_SIZE).
     */
    std::size_t l1_small_size = DEFAULT_L1_SMALL_SIZE;
    /**
     * Trace region size to reserve (default: DEFAULT_TRACE_REGION_SIZE).
     */
    std::size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE;
    /**
     * Dispatch core config to use (default: DispatchCoreType::WORKER, DispatchCoreAxis::ROW).
     */
    DispatchCoreConfig dispatch_core_config = DispatchCoreConfig{};
    /**
     * For shuffling bank id offsets
     */
    stl::Span<const std::uint32_t> l1_bank_remap = {};
};

/**
 * @brief Instantiates a Device object.
 *
 * @param options Configuration parameter for requested device
 * @return Device handle to the created device.
 */
IDevice* CreateDevice(chip_id_t device_id, CreateDeviceOptions options = {});

/**
 * @brief Resets and closes the device.
 *
 * @param device Handle to the device to close.
 * @return True if the device was successfully closed; otherwise, false.
 */
bool CloseDevice(IDevice* device);

/**
 * @brief Deallocates all buffers on the device.
 */
void DeallocateBuffers(IDevice* device);

/**
 * @brief Dumps device-side profiler data to a CSV log.
 *
 * @param device The device holding the program being profiled.
 * @param worker_cores CoreRangeSet of worker cores being profiled.
 * @param last_dump If true, indicates the last dump before process termination.
 */
void DumpDeviceProfileResults(IDevice* device, const CoreRangeSet& worker_cores, bool last_dump = false);

/**
 * @brief Retrieves the architecture of the device.
 *
 * @param device The device to query.
 * @return ARCH representing the device architecture.
 */
ARCH GetArch(IDevice* device);

/**
 * @brief Retrieves the ID of the device.
 *
 * @param device The device to query.
 * @return Chip ID of the device.
 */
chip_id_t GetId(IDevice* device);

/**
 * @brief Retrieves the number of DRAM channels on the device.
 *
 * @param device The device to query.
 * @return Number of DRAM channels.
 */
int GetNumDramChannels(IDevice* device);

/**
 * @brief Retrieves the available L1 size per worker core on the device.
 *
 * @param device The device to query.
 * @return L1 size per core in bytes.
 */
std::uint32_t GetL1SizePerCore(IDevice* device);

/**
 * @brief Computes the storage grid size for the device.
 *
 * @param device The device to query.
 * @return CoreCoord representing the storage grid size.
 */
CoreCoord GetComputeWithStorageGridSize(IDevice* device);

/**
 * @brief Retrieves the DRAM grid size for the device.
 *
 * @param device The device to query.
 * @return CoreCoord representing the DRAM grid size.
 */
CoreCoord GetDramGridSize(IDevice* device);

/**
 * @brief Enables the program cache on the device.
 *
 * @param device The device to modify.
 */
void EnableProgramCache(IDevice* device);

/**
 * @brief Disables and clears the program cache on the device.
 *
 * @param device The device to modify.
 */
void DisableAndClearProgramCache(IDevice* device);

/**
 * @brief Pushes a work function onto the device's work queue.
 *
 * @param device The device to which the work will be pushed.
 * @param work The work function to execute.
 * @param blocking Indicates whether the operation should be blocking (default: false).
 */
void PushWork(IDevice* device, std::function<void()> work, bool blocking = false);

/**
 * @brief Synchronizes operations on the given device.
 *
 * @param device The device to synchronize.
 */
void Synchronize(IDevice* device);

/**
 * @brief Retrieves a list of Ethernet socket coordinates connected to a specific chip ID.
 *
 * @param device The device to query.
 * @param connected_chip_id The connected chip ID.
 * @return Vector of CoreCoord representing Ethernet socket coordinates.
 */
std::vector<CoreCoord> GetEthernetSockets(IDevice* device, chip_id_t connected_chip_id);

/**
 * @brief Returns the number of banks for a specific buffer type on the device.
 *
 * @param device The device to query.
 * @param buffer_type The type of buffer.
 * @return Number of banks.
 */
std::uint32_t GetNumBanks(IDevice* device, BufferType buffer_type);

/**
 * @brief Computes the offset of a specific bank for a buffer type on the device.
 *
 * @param device The device to query.
 * @param buffer_type The type of buffer.
 * @param bank_id The ID of the bank.
 * @return Offset of the bank.
 */
std::int32_t GetBankOffset(IDevice* device, BufferType buffer_type, std::uint32_t bank_id);

/**
 * @brief Retrieves bank IDs associated with a logical core for a given buffer type.
 *
 * @param device The device to query.
 * @param buffer_type The type of buffer.
 * @param logical_core The logical core coordinate.
 * @return span of const bank IDs.
 */
stl::Span<const std::uint32_t> BankIdsFromLogicalCore(
    IDevice* device, BufferType buffer_type, CoreCoord logical_core);

/**
 * @brief Retrieves the current worker mode of the device.
 *
 * @param device The device to query.
 * @return WorkExecutorMode representing the current worker mode.
 */
WorkExecutorMode GetWorkerMode(IDevice* device);

/**
 * @brief Retrieves the number of entries in the program cache on the device.
 *
 * @param device The device to query.
 * @return Number of program cache entries.
 */
std::size_t GetNumProgramCacheEntries(IDevice* device);

}  // namespace v1
}  // namespace tt::tt_metal
