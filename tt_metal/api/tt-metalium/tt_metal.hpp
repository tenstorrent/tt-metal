// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <stdint.h>
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <hostdevcommon/common_values.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/fabric_types.hpp>
#include <tt-metalium/profiler_optional_metadata.hpp>
#include <tt-metalium/profiler_types.hpp>
#include <umd/device/tt_core_coordinates.h>
#include <umd/device/tt_soc_descriptor.h>
#include <umd/device/types/cluster_descriptor_types.h>

namespace tt {
namespace tt_metal {
enum class FabricConfig : uint32_t;
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal {
class Buffer;
class IDevice;
class Program;

namespace detail {

bool DispatchStateCheck(bool isFastDispatch);

// Call before CreateDevices to enable fabric, which uses all free ethernet cores
void InitializeFabricConfig(FabricConfig fabric_config);

std::map<chip_id_t, IDevice*> CreateDevices(
    // TODO: delete this in favour of DevicePool
    const std::vector<chip_id_t>& device_ids,
    const uint8_t num_hw_cqs = 1,
    const size_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
    const size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
    const tt_metal::DispatchCoreConfig& dispatch_core_config = tt_metal::DispatchCoreConfig{},
    const std::vector<uint32_t>& l1_bank_remap = {},
    const size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE,
    bool init_profiler = true,
    bool use_max_eth_core_count_on_all_devices = false,
    bool initialize_fabric_and_dispatch_fw = true);

void CloseDevices(const std::map<chip_id_t, IDevice*>& devices);

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
void WriteToBuffer(std::shared_ptr<Buffer> buffer, const std::vector<DType>& host_buffer) {
    WriteToBuffer(*buffer, host_buffer);
}

// TODO: Remove shard_order from this function
void ReadFromBuffer(Buffer& buffer, uint8_t* host_buffer, bool shard_order = false);
/**
 * Copies data from a buffer into a host buffer
 *
 * Return value: void
 *
 * | Argument    | Description                                     | Data type               | Valid range | Required |
 * |-------------|-------------------------------------------------|-------------------------|--------------------------------------------------|----------|
 * | buffer      | Buffer to read data from                        | Buffer &                | | Yes      | |
 * host_buffer | Buffer on host to copy data into                | std::vector<DType> &    | | Yes      | | shard_order
 * | For a sharded buffer we can read in shard order | bool                    | | No       |
 */
template <typename DType>
void ReadFromBuffer(Buffer& buffer, std::vector<DType>& host_buffer, bool shard_order = false) {
    auto buffer_size = buffer.size();
    TT_FATAL(buffer_size % sizeof(DType) == 0, "Buffer size is not divisible by dtype size");
    host_buffer.resize(buffer.size() / sizeof(DType));
    ReadFromBuffer(buffer, reinterpret_cast<uint8_t*>(host_buffer.data()), shard_order);
}
template <typename DType>
void ReadFromBuffer(std::shared_ptr<Buffer> buffer, std::vector<DType>& host_buffer, bool shard_order = false) {
    ReadFromBuffer(*buffer, host_buffer, shard_order);
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
void WaitProgramDone(IDevice* device, Program& program, bool dump_device_profile_results = true);

/**
 *  Compiles all kernels within the program, and generates binaries that are written to
 * `$TT_METAL_HOME/built/<device>/kernels/<kernel name>/<kernel hash>`
 *
 *  To speed up compilation there is a kernel compilation cache that skips over generating binaries for the previously
 * compiled kernels. Kernel uniqueness is determined by the kernel hash which is computed based on compile time args,
 * defines, and kernel type specific attributes such as NOC for data movement kernels and math fidelity for compute
 * kernels
 *  TODO: Kernel hash needs to account for device architecture as binaries are not the same across architectures.
 *  On cache hits the kernel is not recompiled if the output binary directory exists, otherwise the kernel is compiled.
 *  This cache is static is enabled for the duration of the running process.
 *  By default the cache does not persistent across runs, but can be enabled by calling EnablePersistentKernelCache().
 * Setting this will skip compilation when output binary directory exists.
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
 * Clear profiler control buffer
 *
 * Return value: void
 *
 * | Argument      | Description                                                        | Type            | Valid Range
 * | Required |
 * |---------------|--------------------------------------------------------------------|-----------------|---------------------------|----------|
 * | device        | Clear profiler control buffer before any core attempts to profler  | IDevice*        | | True     |
 * */
void ClearProfilerControlBuffer(IDevice* device);

/**
 * Initialize device profiling data buffers
 *
 * Return value: void
 *
 * | Argument      | Description                                       | Type            | Valid Range               |
 * Required |
 * |---------------|---------------------------------------------------|-----------------|---------------------------|----------|
 * | device        | The device holding the program being profiled.    | IDevice*        |                           |
 * True     |
 * */
void InitDeviceProfiler(IDevice* device);

/**
 * Sync TT devices with host
 *
 * Return value: void
 *
 * | Argument      | Description                                       | Type            | Valid Range               | Required |
 * |---------------|---------------------------------------------------|-----------------|---------------------------|----------|
 * */
void ProfilerSync(ProfilerSyncState state);

/**
 * Read device side profiler data and dump results into device side CSV log
 *
 * Return value: void
 *
 * | Argument      | Description                                       | Type | Valid Range               | Required |
 * |---------------|---------------------------------------------------|--------------------------------------------------------------|---------------------------|----------|
 * | device        | The device holding the program being profiled.    | IDevice* |                           | True |
 * | core_coords   | The logical core coordinates being profiled.      | const std::unordered_map<CoreType,
 * std::vector<CoreCoord>> & |
 * | satate        | Dumpprofiler various states                       | ProfilerDumpState |                  | False |
 * */
void DumpDeviceProfileResults(
    IDevice* device,
    std::vector<CoreCoord>& worker_cores,
    ProfilerDumpState = ProfilerDumpState::NORMAL,
    const std::optional<ProfilerOptionalMetadata>& metadata = {});

/**
 * Traverse all cores and read device side profiler data and dump results into device side CSV log
 *
 * Return value: void
 *
 * | Argument      | Description                                       | Type | Valid Range               | Required |
 * |---------------|---------------------------------------------------|--------------------------------------------------------------|---------------------------|----------|
 * | device        | The device holding the program being profiled.    | Device * |                           | True |
 * | satate        | Dumpprofiler various states                       | ProfilerDumpState |                  | False |
 * */
void DumpDeviceProfileResults(
    IDevice* device,
    ProfilerDumpState = ProfilerDumpState::NORMAL,
    const std::optional<ProfilerOptionalMetadata>& metadata = {});

/**
 * Set the directory for device-side CSV logs produced by the profiler instance in the tt-metal module
 *
 * Return value: void
 *
 * | Argument     | Description                                             |  Data type  | Valid range              |
 * required |
 * |--------------|---------------------------------------------------------|-------------|--------------------------|----------|
 * | output_dir   | The output directory that will hold the output CSV logs  | std::string | Any valid directory path |
 * No       |
 * */
void SetDeviceProfilerDir(const std::string& output_dir = "");

/**
 * Start a fresh log for the device side profile results
 *
 * Return value: void
 *
 * | Argument     | Description                                             |  Data type  | Valid range              |
 * required |
 * |--------------|---------------------------------------------------------|-------------|--------------------------|----------|
 * */
void FreshProfilerDeviceLog();

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

}  // namespace detail
}  // namespace tt::tt_metal
