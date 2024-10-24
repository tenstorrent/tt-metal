// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <vector>
#include <map>

#include "tt_metal/third_party/umd/device/tt_cluster_descriptor_types.h"
#include "tt_metal/third_party/umd/device/tt_soc_descriptor.h"
#include "tt_metal/hostdevcommon/common_values.hpp"
#include "tt_metal/common/core_coord.hpp"
#include "tt_metal/impl/dispatch/dispatch_core_manager.hpp"

namespace tt::tt_metal {
inline namespace v0 {
    class Program;
    class Buffer;
    class Device;
}  // namespace v0

    namespace detail {

        bool DispatchStateCheck( bool isFastDispatch);

        bool InWorkerThread();

        std::map<chip_id_t, Device *> CreateDevices(
            // TODO: delete this in favour of DevicePool
            const std::vector<chip_id_t>& device_ids,
            const uint8_t num_hw_cqs = 1,
            const size_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
            const size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
            tt_metal::DispatchCoreType dispatch_core_type = tt_metal::DispatchCoreType::WORKER,
            const std::vector<uint32_t> &l1_bank_remap = {});

        void CloseDevices(std::map<chip_id_t, Device *> devices);

        /**
        * Copies data from a host buffer into the specified buffer
        *
        * Return value: void
        *
        * | Argument    | Description                                     | Data type               | Valid range                                      | Required |
        * |-------------|-------------------------------------------------|-------------------------|--------------------------------------------------|----------|
        * | buffer      | Buffer to send data to                          | Buffer &                |                                                  | Yes      |
        * | host_buffer | Buffer on host to copy data from                | std::vector<uint32_t> & | Host buffer size must match buffer               | Yes      |
        */
        void WriteToBuffer(Buffer &buffer, const std::vector<uint32_t> &host_buffer);
        void WriteToBuffer( std::shared_ptr<Buffer> buffer, const std::vector<uint32_t> &host_buffer);
        /**
        * Copies data from a buffer into a host buffer
        *
        * Return value: void
        *
        * | Argument    | Description                                     | Data type               | Valid range                                      | Required |
        * |-------------|-------------------------------------------------|-------------------------|--------------------------------------------------|----------|
        * | buffer      | Buffer to read data from                        | Buffer &                |                                                  | Yes      |
        * | host_buffer | Buffer on host to copy data into                | std::vector<uint32_t> & |                                                  | Yes      |
        * | shard_order | For a sharded buffer we can read in shard order | bool                    |                                                  | No       |
        */
        void ReadFromBuffer(Buffer &buffer, std::vector<uint32_t> &host_buffer, bool shard_order = false);
        void ReadFromBuffer(std::shared_ptr<Buffer> buffer, std::vector<uint32_t> &host_buffer, bool shard_order = false);

        /**
        * Copies data from a buffer into a host buffer
        *
        * Return value: void
        *
        * | Argument    | Description                                     | Data type               | Valid range                                      | Required |
        * |-------------|-------------------------------------------------|-------------------------|--------------------------------------------------|----------|
        * | buffer      | Buffer to read data from                        | Buffer &                |                                                  | Yes      |
        * | host_buffer | Buffer on host to copy data into                | std::vector<uint32_t> & |                                                  | Yes      |
        * | core_id     | ID of core                                      | const uint32_t &        |                                                  | Yes      |
        */
        void ReadShard(Buffer &buffer, std::vector<uint32_t> &host_buffer, const uint32_t & core_id);



        // Launches all kernels on cores specified with kernels in the program.
        // All kernels on a given Tensix core must be launched.
        void LaunchProgram(Device *device, Program &program, bool wait_until_cores_done = true);
        void LaunchProgram(Device *device, std::shared_ptr<Program> program, bool wait_until_cores_done = true);
        void WaitProgramDone(Device *device, Program &program);

        /**
         *  Compiles all kernels within the program, and generates binaries that are written to `$TT_METAL_HOME/built/<device>/kernels/<kernel name>/<kernel hash>`
         *
         *  To speed up compilation there is a kernel compilation cache that skips over generating binaries for the previously compiled kernels.
         *  Kernel uniqueness is determined by the kernel hash which is computed based on compile time args, defines, and kernel type specific attributes such as NOC for data movement kernels and math fidelity for compute kernels
         *  TODO: Kernel hash needs to account for device architecture as binaries are not the same across architectures.
         *  On cache hits the kernel is not recompiled if the output binary directory exists, otherwise the kernel is compiled.
         *  This cache is static is enabled for the duration of the running process.
         *  By default the cache does not persistent across runs, but can be enabled by calling EnablePersistentKernelCache(). Setting this will skip compilation when output binary directory exists.
         *
         *  Return value: void
         *
         * | Argument                  | Description                                                      | Type      | Valid Range                                        | Required |
         * |---------------------------|------------------------------------------------------------------|-----------|----------------------------------------------------|----------|
         * | device                    | Which device the program is compiled for                         | Device *  | Must be initialized via tt_metal::InitializeDevice | Yes      |
         * | program                   | The program to compile                                           | Program & |                                                    | Yes      |
         * | fd_bootloader_mode        | Set when compiling program to initialize fast dispatch           | bool      |                                                    | No       |
         */
        void CompileProgram(Device *device, Program &program, bool fd_bootloader_mode = false);


        /**
         * Writes runtime args that are saved in the program to device
         *
         * Return value: void
         *
         * | Argument            | Description                                                            | Type                          | Valid Range                        | Required |
         * |---------------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
         * | device              | The device to whcih runtime args will be written                       | Device *                      |                                    | Yes      |
         * | program             | The program holding the runtime args                                   | const Program &               |                                    | Yes      |
         */
        void WriteRuntimeArgsToDevice(Device *device, Program &program);

        // Configures a given device with a given program.
        // - Loads all kernel binaries into L1s of assigned Tensix cores
        // - Configures circular buffers (inits regs with buffer data)
        // - Takes the device out of reset
        bool ConfigureDeviceWithProgram(Device *device, Program &program, bool fd_bootloader_mode = false);


        /**
         * Clear profiler control buffer
         *
         * Return value: void
         *
         * | Argument      | Description                                                        | Type            | Valid Range               | Required |
         * |---------------|--------------------------------------------------------------------|-----------------|---------------------------|----------|
         * | device        | Clear profiler control buffer before any core attempts to profler  | Device *        |                           | True     |
         * */
	    void ClearProfilerControlBuffer(Device *device);

        /**
         * Initialize device profiling data buffers
         *
         * Return value: void
         *
         * | Argument      | Description                                       | Type            | Valid Range               | Required |
         * |---------------|---------------------------------------------------|-----------------|---------------------------|----------|
         * | device        | The device holding the program being profiled.    | Device *        |                           | True     |
         * */
	    void InitDeviceProfiler(Device *device);

        /**
         * Read device side profiler data and dump results into device side CSV log
         *
         * Return value: void
         *
         * | Argument      | Description                                       | Type                                                         | Valid Range               | Required |
         * |---------------|---------------------------------------------------|--------------------------------------------------------------|---------------------------|----------|
         * | device        | The device holding the program being profiled.    | Device *                                                     |                           | True     |
         * | core_coords   | The logical core coordinates being profiled.      | const std::unordered_map<CoreType, std::vector<CoreCoord>> & |                           | True     |
         * | last_dump     | Last dump before process dies                     | bool                                                         |                           | False    |
         * */
        void DumpDeviceProfileResults(Device *device, std::vector<CoreCoord> &worker_cores, bool last_dump = false);

        /**
         * Traverse all cores and read device side profiler data and dump results into device side CSV log
         *
         * Return value: void
         *
         * | Argument      | Description                                       | Type                                                         | Valid Range               | Required |
         * |---------------|---------------------------------------------------|--------------------------------------------------------------|---------------------------|----------|
         * | device        | The device holding the program being profiled.    | Device *                                                     |                           | True     |
         * | last_dump     | Last dump before process dies                     | bool                                                         |                           | False    |
         * */
        void DumpDeviceProfileResults(Device *device, bool last_dump = false);

        /**
         * Set the directory for device-side CSV logs produced by the profiler instance in the tt-metal module
         *
         * Return value: void
         *
         * | Argument     | Description                                             |  Data type  | Valid range              | required |
         * |--------------|---------------------------------------------------------|-------------|--------------------------|----------|
         * | output_dir   | The output directory that will hold the output CSV logs  | std::string | Any valid directory path | No       |
         * */
        void SetDeviceProfilerDir(std::string output_dir = "");

        /**
         * Set the directory for all host-side CSV logs produced by the profiler instance in the tt-metal module
         *
         * Return value: void
         *
         * | Argument     | Description                                             |  Data type  | Valid range              | required |
         * |--------------|---------------------------------------------------------|-------------|--------------------------|----------|
         * | output_dir   | The output directory that will hold the output CSV logs  | std::string | Any valid directory path | No       |
         * */
        void SetHostProfilerDir(std::string output_dir = "");

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

        /**
         * Copies data from a host buffer into a buffer within the device DRAM channel
         *
         * Return value: bool
         *
         * | Argument     | Description                                            | Data type             | Valid range                               | required |
         * |--------------|--------------------------------------------------------|-----------------------|-------------------------------------------|----------|
         * | device       | The device whose DRAM to write data into               | Device *              |                                           | Yes      |
         * | dram_channel | Channel index of DRAM to write into                    | int                   | On Grayskull, [0, 7] inclusive            | Yes      |
         * | address      | Starting address on DRAM channel to begin writing data | uint32_t              | [DRAM_UNRESERVED_BASE, dram_size)         | Yes      |
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
         * | logical_core  | Logical coordinate of core whose L1 to write to | CoreCoord             | On Grayskull, any valid logical worker coordinate   | Yes      |
         * | address       | Starting address in L1 to write into            | uint32_t              | Any non-reserved address in L1 that fits for buffer | Yes      |
         * | host_buffer   | Buffer on host whose data to copy from          | std::vector<uint32_t> | Buffer must fit into L1                             | Yes      |
         */
        bool WriteToDeviceL1(Device *device, const CoreCoord &logical_core, uint32_t address, std::vector<uint32_t> &host_buffer, CoreType core_type = CoreType::WORKER);

        bool WriteRegToDevice(Device *device, const CoreCoord &logical_core, uint32_t address, const uint32_t &regval);

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

        bool ReadRegFromDevice(Device *device, const CoreCoord &logical_core, uint32_t address, uint32_t &regval);

        void SetLazyCommandQueueMode(bool lazy);

        DeviceAddr AllocateBuffer(const Buffer* buffer, bool bottom_up);

        void DeallocateBuffer(Buffer *buffer);
    }  // namespace detail
}  // namespace tt::tt_metal
