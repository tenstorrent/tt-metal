#pragma once
#include <mutex>

#include "tt_metal/impl/dispatch/command_queue.hpp"

using std::unique_lock;
using std::mutex;

namespace tt::tt_metal{

    namespace detail {

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
         * | address      | Starting address on DRAM channel to begin writing data | uint32_t              |                                           | Yes      |
         * | host_buffer  | Buffer on host to copy data from                       | std::vector<uint32_t> | Host buffer must be fully fit DRAM buffer | Yes      |
         */
        inline bool WriteToDeviceDRAMChannel(Device *device, int dram_channel, uint32_t address, std::vector<uint32_t> &host_buffer)
        {
            bool pass = true;
            device->cluster()->write_dram_vec(host_buffer, tt_target_dram{device->pcie_slot(), dram_channel, 0}, address);
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
            device->cluster()->read_dram_vec(host_buffer, tt_target_dram{device->pcie_slot(), dram_channel, 0}, address, size);
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
            llrt::write_hex_vec_to_core(device->cluster(), device->pcie_slot(), worker_core, host_buffer, address);
            return true;
        }

        inline bool WriteToDeviceL1(Device *device, const CoreCoord &core, op_info_t op_info, int op_idx)
        {
            auto worker_core = device->worker_core_from_logical_core(core);
            llrt::write_graph_interpreter_op_info_to_core(device->cluster(), device->pcie_slot(), worker_core, op_info, op_idx);
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
            auto worker_core = device->worker_core_from_logical_core(logical_core);
            host_buffer = llrt::read_hex_vec_from_core(device->cluster(), device->pcie_slot(), worker_core, address, size);
            return true;
        }

        inline void GenerateBankToNocCoordHeaders(  Device *device,
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
        }

        struct HashLookup {
            static HashLookup& inst() {
                static HashLookup inst_;
                return inst_;
            }

            bool exists(size_t khash) {
                unique_lock<mutex> lock(mutex_);
                return hashes_.find(khash) != hashes_.end();
            }
            bool add(size_t khash) {
                unique_lock<mutex> lock(mutex_);
                bool ret = false;
                if (hashes_.find(khash) == hashes_.end() ){
                    hashes_.insert(khash);
                    ret = true;
                }
                return ret;
            }

            void clear() {
                unique_lock<mutex> lock(mutex_);
                hashes_.clear();
            }


        private:
            std::mutex mutex_;
            std::unordered_set<size_t > hashes_;
        };

    }
}
