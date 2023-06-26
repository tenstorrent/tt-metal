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
