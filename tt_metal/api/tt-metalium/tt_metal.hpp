// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <stdint.h>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <span>
#include <vector>

#include <tt_stl/span.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device_types.hpp>
// UMD: re-exports CoreType (used in SetRuntimeArgs/GetRuntimeArgs default parameter).
#include <umd/device/types/core_coordinates.hpp>

namespace tt::tt_metal {
class Buffer;
class IDevice;
class Program;

namespace detail {

/**
 * Returns a pointer to an active device with the given ID, NULL otherwise
 *
 * Return value: IDevice*
 *
 * | Argument    | Description                                     | Data type               | Valid range      |
 * Required |
 * |-------------|-------------------------------------------------|-------------------------|--------------------------------------------------|----------|
 * | device_id   | ID of the device to look for                    | ChipId                  | Valid device IDs | Yes |
 */
IDevice* GetActiveDevice(ChipId device_id);

void ReadShard(Buffer& buffer, uint8_t* host_buffer, const uint32_t& core_id);
/**
 * Copies data from a buffer into a host buffer
 *
 * Return value: void
 *
 * | Argument    | Description                                     | Data type               | Valid range | Required |
 * |-------------|-------------------------------------------------|-------------------------|--------------------------------------------------|----------|
 * | buffer      | Buffer to read data from                        | Buffer &                | | Yes      | |
 * host_buffer | Buffer on host to copy data into                | std::vector<DType> &    | | Yes      | | core_id | ID
 * of core                                      | const uint32_t &        | | Yes      |
 */
template <typename DType>
void ReadShard(Buffer& buffer, std::vector<DType>& host_buffer, const uint32_t& core_id) {
    host_buffer.resize(buffer.page_size() * buffer.shard_spec().num_pages());
    ReadShard(buffer, reinterpret_cast<uint8_t*>(host_buffer.data()), core_id);
}

/**
 * Generate a (unique) per device ID for a program (potentially) running across multiple devices. The generated ID is
 * used by the performance profiler.
 *
 * Return value: uint32_t
 *
 * | Argument             | Description                                                                         |  Data
 * type            | Valid range              | required |
 * |----------------------|-------------------------------------------------------------------------------------|-----------------------|--------------------------|----------|
 * | base_program_id      | ID assigned to a program or an op by the user, for use by the performance profiler  |
 * uint32_t              | 0 - 2^21 - 1             | yes      | | device_id            | The device id this op will be
 * launched on (0 if this op runs on host only)          | uint32_t              | 0 - 2^32 - 1             | yes      |
 * | is_host_fallback_op  | (Optional): Specifies if this op runs entirely on host                              | bool
 * |                          | no       |
 */
uint32_t EncodePerDeviceProgramID(uint32_t base_program_id, uint32_t device_id, bool is_host_fallback_op = false);

// clang-format off
/**
 * Copy data from a host buffer into an L1 buffer. (Note: Current Can not be a CircularBuffer.)
 *
 * Return value: bool
 *
 * | Argument      | Description                                     | Data type                | Valid range                                         | required |
 * |---------------|-------------------------------------------------|--------------------------|-----------------------------------------------------|----------|
 * | device        | The device whose L1 to write data into          | IDevice*                 |                                                     | Yes      |
 * | logical_core  | Logical coordinate of core whose L1 to write to | CoreCoord                | On Grayskull, any valid logical worker coordinate   | Yes      |
 * | address       | Starting address in L1 to write into            | uint32_t                 | Any non-reserved address in L1 that fits for buffer | Yes      |
 * | host_buffer   | Buffer on host whose data to copy from          | std::span<const uint8_t> | Buffer must fit into L1                             | Yes      |
 */
// clang-format on
bool WriteToDeviceL1(
    IDevice* device,
    const CoreCoord& logical_core,
    uint32_t address,
    std::span<const uint8_t> host_buffer,
    CoreType core_type = CoreType::WORKER);
/**
 * Copy data from a host buffer into an L1 buffer. (Note: Current Can not be a CircularBuffer.)
 *
 * Return value: bool
 *
 * | Argument      | Description                                     | Data type             | Valid range | required |
 * |---------------|-------------------------------------------------|-----------------------|-----------------------------------------------------|----------|
 * | device        | The device whose DRAM to write data into        | IDevice*              | | Yes      | |
 * logical_core  | Logical coordinate of core whose L1 to write to | CoreCoord             | On Grayskull, any valid
 * logical worker coordinate   | Yes      | | address       | Starting address in L1 to write into            | uint32_t
 * | Any non-reserved address in L1 that fits for buffer | Yes      | | host_buffer   | Buffer on host whose data to
 * copy from          | std::vector<uint32_t> | Buffer must fit into L1                             | Yes      |
 */
bool WriteToDeviceL1(
    IDevice* device,
    const CoreCoord& logical_core,
    uint32_t address,
    std::vector<uint32_t>& host_buffer,
    CoreType core_type = CoreType::WORKER);

// clang-format off
/**
 * Copy data from an L1 buffer into a host buffer. Must be a buffer, and not a CB.
 *
 * Return value: bool
 *
 * | Argument             | Description                                 | Data type             | Valid range                                       | required |
 * |----------------------|---------------------------------------------|-----------------------|---------------------------------------------------|----------|
 * | device               | The device whose L1 to read data from       | IDevice*              |                                                   | Yes      |
 * | logical_core         | Logical coordinate of core whose L1 to read | CoreCoord             | On Grayskull, any valid logical worker coordinate | Yes      |
 * | address              | Starting address in L1 to read from         | uint32_t              |                                                   | Yes      |
 * | host_buffer          | Buffer on host to copy data into            | std::span<uint8_t>    | Buffer must fit L1 buffer                         | Yes      |
 */
// clang-format on
bool ReadFromDeviceL1(
    IDevice* device,
    const CoreCoord& logical_core,
    uint32_t address,
    std::span<uint8_t> host_buffer,
    CoreType core_type = CoreType::WORKER);

/**
 * Copy data from an L1 buffer into a host buffer. Must be a buffer, and not a CB.
 *
 * Return value: bool
 *
 * | Argument             | Description                                 | Data type             | Valid range | required
 * |
 * |----------------------|---------------------------------------------|-----------------------|---------------------------------------------------|----------|
 * | device               | The device whose DRAM to read data from     | IDevice*              | | Yes      | |
 * logical_core         | Logical coordinate of core whose L1 to read | CoreCoord            | On Grayskull, any valid
 * logical worker coordinate | Yes      | | address              | Starting address in L1 to read from         |
 * uint32_t              |                                                   | Yes      | | size                 | Size
 * of L1 buffer in bytes                  | uint32_t              |                                                   |
 * Yes      | | host_buffer          | Buffer on host to copy data into            | std::vector<uint32_t> | Buffer must
 * fit L1 buffer                         | Yes      |
 */
bool ReadFromDeviceL1(
    IDevice* device,
    const CoreCoord& logical_core,
    uint32_t address,
    uint32_t size,
    std::vector<uint32_t>& host_buffer,
    CoreType core_type = CoreType::WORKER);

/**
 * Return the name of the architecture present.
 *
 * Return value: std::string
 */
std::string get_platform_architecture_name();

}  // namespace detail
}  // namespace tt::tt_metal
