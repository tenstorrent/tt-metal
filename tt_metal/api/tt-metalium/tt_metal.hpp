// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <stdint.h>
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <span>
#include <vector>

#include <hostdevcommon/common_values.hpp>
#include <tt_stl/span.hpp>
#include <tt_stl/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/cluster.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/profiler_optional_metadata.hpp>
#include <tt-metalium/profiler_types.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/soc_descriptor.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

namespace tt::tt_metal {
class Buffer;
class IDevice;
class Program;

namespace detail {

bool DispatchStateCheck(bool isFastDispatch);

std::map<ChipId, IDevice*> CreateDevices(
    // TODO: delete this in favour of DeviceManager
    const std::vector<ChipId>& device_ids,
    uint8_t num_hw_cqs = 1,
    size_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
    size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
    const tt_metal::DispatchCoreConfig& dispatch_core_config = tt_metal::DispatchCoreConfig{},
    const std::vector<uint32_t>& l1_bank_remap = {},
    size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE,
    bool init_profiler = true,
    [[deprecated]] bool ignored = false,  // This argument was not used
    bool initialize_fabric_and_dispatch_fw = true);

void CloseDevices(const std::map<ChipId, IDevice*>& devices);

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

/**
 * Copies data from a host buffer into the specified buffer
 *
 * Return value: void
 *
 * | Argument    | Description                                     | Data type               | Valid range | Required |
 * |-------------|-------------------------------------------------|-------------------------|--------------------------------------------------|----------|
 * | buffer      | Buffer to send data to                          | Buffer &                | | Yes      | |
 * host_buffer | Buffer on host to copy data from                | Span<const uint8_t> &   | Host buffer size must match
 * buffer               | Yes      |
 */
void WriteToBuffer(Buffer& buffer, tt::stl::Span<const uint8_t> host_buffer);
/**
 * Copies data from a host buffer into the specified buffer
 *
 * Return value: void
 *
 * | Argument    | Description                                     | Data type               | Valid range | Required |
 * |-------------|-------------------------------------------------|-------------------------|--------------------------------------------------|----------|
 * | buffer      | Buffer to send data to                          | Buffer &                | | Yes      | |
 * host_buffer | Buffer on host to copy data from                | std::vector<DType> &    | Host buffer size must match
 * buffer               | Yes      |
 */
template <typename DType>
void WriteToBuffer(Buffer& buffer, const std::vector<DType>& host_buffer) {
    WriteToBuffer(
        buffer,
        tt::stl::Span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(host_buffer.data()), host_buffer.size() * sizeof(DType)));
}
template <typename DType>
void WriteToBuffer(const std::shared_ptr<Buffer>& buffer, const std::vector<DType>& host_buffer) {
    WriteToBuffer(*buffer, host_buffer);
}

void ReadFromBuffer(Buffer& buffer, uint8_t* host_buffer);
/**
 * Copies data from a buffer into a host buffer
 *
 * Return value: void
 *
 * | Argument    | Description                                     | Data type               | Valid range | Required |
 * |-------------|-------------------------------------------------|-------------------------|--------------------------------------------------|----------|
 * | buffer      | Buffer to read data from                        | Buffer &                | | Yes      | |
 * host_buffer | Buffer on host to copy data into                | std::vector<DType> &    | | Yes      | |
 */
template <typename DType>
void ReadFromBuffer(Buffer& buffer, std::vector<DType>& host_buffer) {
    auto buffer_size = buffer.size();
    TT_FATAL(buffer_size % sizeof(DType) == 0, "Buffer size is not divisible by dtype size");
    host_buffer.resize(buffer.size() / sizeof(DType));
    ReadFromBuffer(buffer, reinterpret_cast<uint8_t*>(host_buffer.data()));
}
template <typename DType>
void ReadFromBuffer(const std::shared_ptr<Buffer>& buffer, std::vector<DType>& host_buffer) {
    ReadFromBuffer(*buffer, host_buffer);
}

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

// Launches all kernels on cores specified with kernels in the program.
// All kernels on a given Tensix core must be launched.
void LaunchProgram(
    IDevice* device, Program& program, bool wait_until_cores_done = true, bool force_slow_dispatch = false);
void LaunchProgram(
    IDevice* device,
    const std::shared_ptr<Program>& program,
    bool wait_until_cores_done = true,
    bool force_slow_dispatch = false);
void WaitProgramDone(IDevice* device, Program& program, bool read_device_profiler_results = true);

/**
 *  Compiles all kernels within the program, and generates binaries that are written to
 * `<tt-metal-cache directory>/<build_key>/kernels/<kernel name>/<kernel hash>`
 *
 *  The build key component accounts for device architecture as binaries are not compatible across architectures.
 *  To speed up compilation there is a kernel compilation cache that skips over generating binaries for the previously
 * compiled kernels. Kernel uniqueness is determined by the kernel hash which is computed based on compile time args,
 * defines, and kernel type specific attributes such as NOC for data movement kernels and math fidelity for compute
 * kernels.
 *  On cache hits the kernel is not recompiled if the output binary directory exists, otherwise the kernel is compiled.
 *  This cache is static and is enabled for the duration of the running process.
 *  Across runs, previously compiled kernels are recompiled if the source code or dependencies have changed.
 *
 *  Return value: void
 *
 * | Argument                  | Description                                                      | Type      | Valid
 * Range                                        | Required |
 * |---------------------------|------------------------------------------------------------------|-----------|----------------------------------------------------|----------|
 * | device                    | Which device the program is compiled for                         | IDevice*  | Must be
 * initialized via tt_metal::InitializeDevice | Yes      | | program                   | The program to compile |
 * Program & |                                                    | Yes      | | force_slow_dispatch        | Set when
 * a user wants to compile a program with Slow Dispatch Force Enabled (advanced feature, currently used internally to
 * launch Fast Dispatch Firmware and in the Device Performance Profiler)           | bool      | | No |
 */
void CompileProgram(IDevice* device, Program& program, bool force_slow_dispatch = false);

/**
 * Writes runtime args that are saved in the program to device
 *
 * Return value: void
 *
 * | Argument            | Description                                                            | Type | Valid Range
 * | Required |
 * |---------------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | device              | The device to whcih runtime args will be written                       | IDevice* | | Yes |
 * | program             | The program holding the runtime args                                   | const Program & | |
 * Yes      |
 */
void WriteRuntimeArgsToDevice(IDevice* device, Program& program, bool force_slow_dispatch = false);

// Configures a given device with a given program.
// - Loads all kernel binaries into L1s of assigned Tensix cores
// - Configures circular buffers (inits regs with buffer data)
// - Takes the device out of reset
bool ConfigureDeviceWithProgram(IDevice* device, Program& program, bool force_slow_dispatch = false);

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

/**
 * Decode per device program ID to get encoded values (base program id, device id, and a flag indicating whether
 * it's an op run entirely on host).
 *
 * Return value: tuple<uint32_t, uint32_t, bool>
 *
 * | Argument             | Description                                                                         |  Data
 * type            | Valid range              | required |
 * |----------------------|-------------------------------------------------------------------------------------|-----------------------|--------------------------|----------|
 * | device_program_id    | Encoded device specific id used by the performance profiler  |
 * uint32_t        | 0 - 2^32 - 1             | yes      |
 */
DeviceProgramId DecodePerDeviceProgramID(uint32_t device_program_id);

// clang-format off
/**
 * Copies data from a host buffer into a buffer within the device DRAM channel
 *
 * Return value: bool
 *
 * | Argument     | Description                                            | Data type                | Valid range                               | required |
 * |--------------|--------------------------------------------------------|--------------------------|-------------------------------------------|----------|
 * | device       | The device whose DRAM to write data into               | IDevice*                 |                                           | Yes      |
 * | dram_channel | Channel index of DRAM to write into                    | int                      | On Grayskull, [0, 7] inclusive            | Yes      |
 * | address      | Starting address on DRAM channel to begin writing data | uint32_t                 | [DRAM_UNRESERVED_BASE, dram_size)         | Yes      |
 * | host_buffer  | Buffer on host to copy data from                       | std::span<const uint8_t> | Host buffer must be fully fit DRAM buffer | Yes      |
 */
// clang-format on
bool WriteToDeviceDRAMChannel(
    IDevice* device, int dram_channel, uint32_t address, std::span<const uint8_t> host_buffer);
/**
 * Copies data from a host buffer into a buffer within the device DRAM channel
 *
 * Return value: bool
 *
 * | Argument     | Description                                            | Data type             | Valid range |
 * required |
 * |--------------|--------------------------------------------------------|-----------------------|-------------------------------------------|----------|
 * | device       | The device whose DRAM to write data into               | IDevice*              | | Yes      | |
 * dram_channel | Channel index of DRAM to write into                    | int                   | On Grayskull, [0, 7]
 * inclusive            | Yes      | | address      | Starting address on DRAM channel to begin writing data | uint32_t
 * | [DRAM_UNRESERVED_BASE, dram_size)         | Yes      | | host_buffer  | Buffer on host to copy data from |
 * std::vector<uint32_t> | Host buffer must be fully fit DRAM buffer | Yes      |
 */
bool WriteToDeviceDRAMChannel(IDevice* device, int dram_channel, uint32_t address, std::vector<uint32_t>& host_buffer);

// clang-format off
/**
 * Copy data from a device DRAM channel to a host buffer
 *
 * Return value: bool
 *
 * | Argument     | Description                                                  | Data type             | Valid range                    | required |
 * |--------------|--------------------------------------------------------------|-----------------------|--------------------------------|----------|
 * | device       | The device whose DRAM to read data from                      | IDevice*              |                                | Yes      |
 * | dram_channel | Channel index of DRAM to read from                           | int                   | On Grayskull, [0, 7] inclusive | Yes      |
 * | address      | Starting address on DRAM channel from which to begin reading | uint32_t              |                                | Yes      |
 * | host_buffer  | Buffer on host to copy data into                             | std::span<uint8_t>    |                                | Yes      |
 */
// clang-format on
bool ReadFromDeviceDRAMChannel(IDevice* device, int dram_channel, uint32_t address, std::span<uint8_t> host_buffer);

/**
 * Copy data from a device DRAM channel to a host buffer
 *
 * Return value: bool
 *
 * | Argument     | Description                                                  | Data type             | Valid range
 * | required |
 * |--------------|--------------------------------------------------------------|-----------------------|--------------------------------|----------|
 * | device       | The device whose DRAM to read data from                      | IDevice*              | | Yes      |
 * | dram_channel | Channel index of DRAM to read from                           | int                   | On Grayskull,
 * [0, 7] inclusive | Yes      | | address      | Starting address on DRAM channel from which to begin reading |
 * uint32_t              |                                | Yes      | | size         | Size of buffer to read from
 * device in bytes                  | uint32_t              |                                | Yes      | | host_buffer
 * | Buffer on host to copy data into                             | std::vector<uint32_t> | | Yes      |
 */
bool ReadFromDeviceDRAMChannel(
    IDevice* device, int dram_channel, uint32_t address, uint32_t size, std::vector<uint32_t>& host_buffer);

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

bool WriteRegToDevice(IDevice* device, const CoreCoord& logical_core, uint32_t address, const uint32_t& regval);

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

bool ReadRegFromDevice(IDevice* device, const CoreCoord& logical_core, uint32_t address, uint32_t& regval);

/**
 * Return the name of the architecture present.
 *
 * Return value: std::string
 */
std::string get_platform_architecture_name();

}  // namespace detail
}  // namespace tt::tt_metal
