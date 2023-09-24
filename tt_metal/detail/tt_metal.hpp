/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <mutex>
#include <variant>

#include "third_party/magic_enum/magic_enum.hpp"

#include "tt_metal/build_kernels_for_riscv/build_kernels_for_riscv.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/detail/program.hpp"
#include "tt_metal/llrt/watcher.hpp"
#include "tt_metal/third_party/umd/device/util.hpp"
#include "tt_metal/third_party/umd/device/tt_device.h"

using std::unique_lock;
using std::mutex;

namespace tt::tt_metal{

    namespace detail {
        // To be removed at a later time, but need a global
        // command queue for the time being.
        inline unique_ptr<CommandQueue> GLOBAL_CQ;

        inline static bool DispatchStateCheck( bool isFastDispatch){
            static bool fd = isFastDispatch;
            TT_ASSERT( fd == isFastDispatch, "Mixing fast and slow dispatch is prohibited!" );
            return fd;
        }


        /**
         *  Compiles all kernels within the program, and generates binaries that are written to `$TT_METAL_HOME/built/kernels/<kernel name>/<kernel hash>`
         *  Blank data movement kernel targeting RISCV_0 are placed onto cores that are missing a RISCV_0 kernel because RISCV_0 processor needs to run to enable Compute and RISCV_1 processors
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
         * | Argument       | Description                                                      | Type      | Valid Range                                        | Required |
         * |----------------|------------------------------------------------------------------|-----------|----------------------------------------------------|----------|
         * | device         | Which device the program is compiled for                         | Device *  | Must be initialized via tt_metal::InitializeDevice | Yes      |
         * | program        | The program to compile                                           | Program & |                                                    | Yes      |
         */
        void CompileProgram(Device *device, Program &program);

        /**
         * Writes runtime args that are saved in the program to device
         *
         * Return value: void
         *
         * | Argument     | Description                                                            | Type                          | Valid Range                        | Required |
         * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
         * | device       | The device to whcih runtime args will be written                       | Device *                      |                                    | Yes      |
         * | program      | The program holding the runtime args                                   | const Program &               |                                    | Yes      |
         */
        void WriteRuntimeArgsToDevice(Device *device, const Program &program);

        // Configures a given device with a given program.
        // - Loads all kernel binaries into L1s of assigned Tensix cores
        // - Configures circular buffers (inits regs with buffer data)
        // - Takes the device out of reset
        bool ConfigureDeviceWithProgram(Device *device, const Program &program);

        /**
         * Read device side profiler data and dump results into device side CSV log
         *
         * Return value: void
         *
         * | Argument      | Description                                       | Type            | Valid Range               | Required |
         * |---------------|---------------------------------------------------|-----------------|---------------------------|----------|
         * | device        | The device holding the program being profiled.    | Device *        |                           | True     |
         * | program       | The program being profiled.                       | const Program & |                           | True     |
         * */
        void DumpDeviceProfileResults(Device *device, const Program &program);

        /**
         * Set the directory for all CSV logs produced by the profiler instance in the tt-metal module
         *
         * Return value: void
         *
         * | Argument     | Description                                             |  Data type  | Valid range              | required |
         * |--------------|---------------------------------------------------------|-------------|--------------------------|----------|
         * | output_dir   | The output directory that will hold the outpu CSV logs  | std::string | Any valid directory path | No       |
         * */
        void SetProfilerDir(std::string output_dir = "");

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
         * Profile scopes in tt_metal API
         *
         * */

        class ProfileTTMetalScope
        {
            private:
                string scopeName = "";
            public:
                ProfileTTMetalScope (const string& scopeNameArg);
                ~ProfileTTMetalScope ();
        };

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
        inline bool WriteToDeviceDRAMChannel(Device *device, int dram_channel, uint32_t address, std::vector<uint32_t> &host_buffer)
        {
            bool pass = true;
            TT_ASSERT(address >= DRAM_UNRESERVED_BASE, "Cannot write to reserved DRAM region, addresses [0, {}) are reserved!", DRAM_UNRESERVED_BASE);
            device->cluster()->write_dram_vec(host_buffer, tt_target_dram{device->id(), dram_channel, 0}, address);
            return pass;
        }

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
        inline bool ReadFromDeviceDRAMChannel(Device *device, int dram_channel, uint32_t address, uint32_t size, std::vector<uint32_t> &host_buffer)
        {
            bool pass = true;
            device->cluster()->dram_barrier(device->id());
            device->cluster()->read_dram_vec(host_buffer, tt_target_dram{device->id(), dram_channel, 0}, address, size);
            return pass;
        }

        /**
         * Copy data from a host buffer into an L1 buffer. (Note: Current Can not be a CircularBuffer.)
         *
         * Return value: bool
         *
         * | Argument      | Description                                     | Data type             | Valid range                                         | required |
         * |---------------|-------------------------------------------------|-----------------------|-----------------------------------------------------|----------|
         * | device        | The device whose DRAM to write data into        | Device *              |                                                     | Yes      |
         * | logical_core  | Logical coordinate of core whose L1 to write to | CoreCoord            | On Grayskull, any valid logical worker coordinate   | Yes      |
         * | address       | Starting address in L1 to write into            | uint32_t              | Any non-reserved address in L1 that fits for buffer | Yes      |
         * | host_buffer   | Buffer on host whose data to copy from          | std::vector<uint32_t> | Buffer must fit into L1                             | Yes      |
         */
        inline bool WriteToDeviceL1(Device *device, const CoreCoord &logical_core, uint32_t address, std::vector<uint32_t> &host_buffer)
        {
            auto worker_core = device->worker_core_from_logical_core(logical_core);
            llrt::write_hex_vec_to_core(device->cluster(), device->id(), worker_core, host_buffer, address);
            return true;
        }

        inline bool WriteToDeviceL1(Device *device, const CoreCoord &core, op_info_t op_info, int op_idx)
        {
            auto worker_core = device->worker_core_from_logical_core(core);
            llrt::write_graph_interpreter_op_info_to_core(device->cluster(), device->id(), worker_core, op_info, op_idx);
            return true;
        }


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
        inline bool ReadFromDeviceL1(Device *device, const CoreCoord &logical_core, uint32_t address, uint32_t size, std::vector<uint32_t> &host_buffer)
        {
            device->cluster()->l1_barrier(device->id());
            auto worker_core = device->worker_core_from_logical_core(logical_core);
            host_buffer = llrt::read_hex_vec_from_core(device->cluster(), device->id(), worker_core, address, size);
            return true;
        }

        inline void Synchronize()
        {
            if (detail::GLOBAL_CQ) {
                Finish(*detail::GLOBAL_CQ);
            }
        }

        inline void DeallocateBuffers(Device * device)
        {
            device->deallocate_buffers();
        }

        inline void GenerateDeviceHeaders(Device *device,
                                          build_kernel_for_riscv_options_t *build_options,
                                          const std::string &op_path_suffix)
        {
            // Basic Allocator generates number of banks which may not be power of 2, so we could just pad and alias for now
            const size_t num_dram_banks = device->num_banks(BufferType::DRAM);
            const size_t num_dram_banks_pow2 = std::pow(2, std::ceil(std::log2(num_dram_banks)));
            std::vector<CoreCoord> dram_noc_coord_per_bank(num_dram_banks);
            std::vector<i32> dram_offsets_per_bank(num_dram_banks);
            for (unsigned bank_id = 0; bank_id < num_dram_banks; bank_id++) {
                dram_noc_coord_per_bank[bank_id] = device->core_from_dram_channel(device->dram_channel_from_bank_id(bank_id));
                dram_offsets_per_bank[bank_id] = device->dram_bank_offset_from_bank_id(bank_id);
            }
            const size_t num_l1_banks = device->num_banks(BufferType::L1);
            const size_t num_l1_banks_pow2 = std::pow(2, std::ceil(std::log2(num_l1_banks)));
            std::vector<CoreCoord> l1_noc_coord_per_bank(num_l1_banks_pow2);
            std::vector<i32> l1_offset_per_bank(num_l1_banks_pow2);
            for (unsigned bank_id = 0; bank_id < num_l1_banks_pow2; bank_id++) {
                if (bank_id < num_l1_banks) {
                    l1_noc_coord_per_bank[bank_id] = device->worker_core_from_logical_core(device->logical_core_from_bank_id(bank_id));
                    l1_offset_per_bank[bank_id] = device->l1_bank_offset_from_bank_id(bank_id);
                } else {
                    l1_noc_coord_per_bank[bank_id] = device->worker_core_from_logical_core(device->logical_core_from_bank_id(0));
                    l1_offset_per_bank[bank_id] = device->l1_bank_offset_from_bank_id(0);
                }
            }
            // Generate header file in proper location
            generate_bank_to_noc_coord_descriptor (
                build_options,
                op_path_suffix,
                dram_noc_coord_per_bank,
                dram_offsets_per_bank,
                l1_noc_coord_per_bank,
                l1_offset_per_bank
            );

            metal_SocDescriptor& soc_d = device->cluster()->get_soc_desc(device->id());

            // Determine which noc-coords are harvested
            // TODO(PGK/Almeet): fix this w/ new UMD
            vector<uint32_t> harvested_rows;
            uint32_t harvested_noc_rows = device->cluster()->get_harvested_rows(device->id());
            for (uint32_t y = 0; y < soc_d.grid_size.y; y++) {
                bool row_harvested = (harvested_noc_rows >> y) & 0x1;
                if (row_harvested) {
                    harvested_rows.push_back(y);
                }
            }

            CoreCoord dispatch_logical_core = device->worker_core_from_logical_core(*device->dispatch_cores().begin());

            // XXXX TODO(PGK): get addr range values from device descriptor...
            generate_noc_addr_ranges_header (
                build_options,
                op_path_suffix,
                0, (uint64_t)4 * 1024 * 1024 * 1024,
                0, 1 * 1024 * 1024 * 1024,
                soc_d.get_pcie_cores(),
                soc_d.get_dram_cores(),
                soc_d.get_ethernet_cores(),
                soc_d.grid_size,
                harvested_rows,
                {dispatch_logical_core});
        }

        inline DataMovementConfig GetDataMovementConfig(const Program &program, const std::string &file_name, const CoreRangeSet &core_ranges, const std::optional<DataMovementConfig> &dm_config) {
            bool riscv0_in_use = false; bool riscv1_in_use = false;
            bool noc0_in_use = false; bool noc1_in_use = false;

            auto set_global_and_local_noc_usage = [&](KernelID kernel_id, bool &local_noc0_usage, bool &local_noc1_usage) {
                const auto kernel = detail::GetKernel(program, kernel_id);
                auto kernel_config = std::get<DataMovementConfig>(kernel->config());
                auto noc_value = magic_enum::enum_integer(kernel_config.noc);
                noc0_in_use, local_noc0_usage = noc_value == 0;
                noc1_in_use, local_noc1_usage = noc_value == 1;
            };

            for (const auto &core_range : core_ranges.ranges()) {
                for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
                    for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
                        auto kernel_group = program.kernels_on_core(CoreCoord(x, y));
                        bool local_noc0_in_use = false; bool local_noc1_in_use = false;
                        if (kernel_group.riscv0_id.has_value()) {
                            riscv0_in_use = true;
                            set_global_and_local_noc_usage(kernel_group.riscv0_id.value(), local_noc0_in_use, local_noc1_in_use);
                        }
                        if (kernel_group.riscv1_id.has_value()) {
                            riscv1_in_use = true;
                            set_global_and_local_noc_usage(kernel_group.riscv1_id.value(), local_noc0_in_use, local_noc1_in_use);
                        }
                        if (kernel_group.riscv0_id.has_value() and kernel_group.riscv1_id.has_value()) {
                            TT_ASSERT(local_noc0_in_use and local_noc1_in_use, "Illegal NOC usage: data movement kernels on logical core {} cannot use the same NOC, doing so results in hangs!");
                        }
                    }
                }
            }

            TT_ASSERT(not (riscv0_in_use and riscv1_in_use), "DataMovementKernel creation failure: Cannot create data movement kernel for {} across specified cores because both data movement processors are in use!", file_name);
            TT_ASSERT(not (noc0_in_use and noc1_in_use), "DataMovementKernel creation failure: Cannot create data movement kernels for {} across specified cores because both NOCs are in use!", file_name);

            if (dm_config.has_value()) {
                return dm_config.value();
            }

            DataMovementProcessor processor = riscv0_in_use ? DataMovementProcessor::RISCV_1 : DataMovementProcessor::RISCV_0;
            NOC noc = noc0_in_use ? NOC::NOC_1 : NOC::NOC_0;
            return DataMovementConfig{.processor = processor, .noc = noc};
        }

        inline CoreRangeSet GetCoreRangeSet(const std::variant<CoreCoord, CoreRange, CoreRangeSet> &specified_core_spec) {
            return std::visit(
                [](auto&& core_spec) -> CoreRangeSet
                {
                    using T = std::decay_t<decltype(core_spec)>;
                    if constexpr (std::is_same_v<T, CoreCoord>) {
                        return CoreRangeSet({CoreRange{.start=core_spec, .end=core_spec}});
                    }
                    else if constexpr (std::is_same_v<T, CoreRange>) {
                        return CoreRangeSet({core_spec});
                    }
                    else if constexpr (std::is_same_v<T, CoreRangeSet>) {
                        return core_spec;
                    }
                },
                specified_core_spec
            );
        }

        // TODO (abhullar): Remove this when tt_cluster and tt_metal::Device abstractions are redesigned
        class ClusterWrapper {
           public:
            ClusterWrapper& operator=(const ClusterWrapper&) = delete;
            ClusterWrapper& operator=(ClusterWrapper&& other) noexcept = delete;
            ClusterWrapper(const ClusterWrapper&) = delete;
            ClusterWrapper(ClusterWrapper&& other) noexcept = delete;

            static const ClusterWrapper& inst() {
                static ClusterWrapper inst;
                return inst;
            }

            tt_cluster *cluster() const { return this->cluster_.get(); }

            size_t number_of_chips() const { return this->cluster_desc_->get_number_of_chips(); }

            tt::ARCH arch() const { return this->arch_; }

           private:
            ClusterWrapper() {
                ZoneScoped;

                TargetDevice target_type;
#ifdef TT_METAL_VERSIM_DISABLED
                target_type = TargetDevice::Silicon;
#else
                target_type = TargetDevice::Versim;
#endif

                std::vector<chip_id_t> physical_mmio_device_ids = tt_SiliconDevice::detect_available_device_ids(true, false);
                this->arch_ = detect_arch(physical_mmio_device_ids.at(0));
                for (int dev_index = 1; dev_index < physical_mmio_device_ids.size(); dev_index++) {
                    chip_id_t device_id = physical_mmio_device_ids.at(dev_index);
                    tt::ARCH detected_arch = detect_arch(device_id);
                    TT_ASSERT(this->arch_ == detected_arch, "Expected all devices to be {} but device {} is {}", get_arch_str(this->arch_), device_id, get_arch_str(detected_arch));
                }

                const std::string sdesc_file = get_soc_description_file(this->arch_, target_type);
                const std::string cluster_desc_path = (this->arch_ == tt::ARCH::WORMHOLE_B0) ? GetClusterDescYAML().string() : "";

                std::set<chip_id_t> logical_device_ids;
                if (cluster_desc_path == "") {
                    // All Grayskull devices are MMIO mapped so physical_mmio_device_ids correspond to all available devices
                    for (chip_id_t logical_mmio_device_id = 0; logical_mmio_device_id < physical_mmio_device_ids.size(); logical_mmio_device_id++) {
                        logical_device_ids.insert(logical_mmio_device_id);
                    }
                    this->cluster_desc_ = tt_ClusterDescriptor::create_for_grayskull_cluster(logical_device_ids);
                } else {
                    this->cluster_desc_ = tt_ClusterDescriptor::create_from_yaml(cluster_desc_path);
                    for (chip_id_t logical_device_id = 0; logical_device_id < this->cluster_desc_->get_number_of_chips(); logical_device_id++) {
                        logical_device_ids.insert(logical_device_id);
                    }
                }

                // init UMD with all available device IDs
                this->cluster_ = std::make_unique<tt_cluster>();
                this->cluster_->open_device(this->arch_, target_type, logical_device_ids, sdesc_file, cluster_desc_path);

                tt_device_params default_params;
                if (getenv("TT_METAL_VERSIM_DUMP_CORES")) {
                    std::string dump_cores_string = getenv("TT_METAL_VERSIM_DUMP_CORES");
                    default_params.vcd_dump_cores = tt::utils::strsplit(dump_cores_string, ',');
                }

                this->cluster_->start_device(default_params);
            }
            ~ClusterWrapper() {
                log_info(tt::LogMetal, "Closing device driver");
                this->cluster_->close_device();
            }

            std::unique_ptr<tt_cluster> cluster_ = nullptr;
            // Need to hold reference to cluster descriptor to detect total number of devices available in cluster
            // UMD static APIs `detect_available_device_ids` and `detect_number_of_chips` only returns number of MMIO mapped devices
            std::unique_ptr<tt_ClusterDescriptor> cluster_desc_;
            // `detect_arch` API expects physical device ID but there are no APIs to translate logical device ID to physical
            // So we hold reference to arch after querying UMD for available physical MMIO device IDs
            tt::ARCH arch_;
        };
    }
}
