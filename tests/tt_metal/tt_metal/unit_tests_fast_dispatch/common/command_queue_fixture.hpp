// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <unordered_set>
#include <variant>
#include <vector>
#include "common/core_coord.h"
#include "common/env_lib.hpp"
#include "gtest/gtest.h"
#include "hostdevcommon/common_values.hpp"
#include "impl/device/device.hpp"
#include "impl/kernels/data_types.hpp"
#include "impl/kernels/kernel_types.hpp"
#include "llrt/hal.hpp"
#include "tt_cluster_descriptor_types.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/common/tt_backend_api_types.hpp"
#include "tt_metal/llrt/rtoptions.hpp"
#include "tt_metal/tt_metal/unit_tests_common/common/test_utils.hpp"
#include "tt_soc_descriptor.h"

class CommandQueueFixture : public ::testing::Test {
   protected:
    tt::ARCH arch_;
    tt::tt_metal::Device* device_;
    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            tt::log_info(tt::LogTest, "This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            GTEST_SKIP();
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());

        const int device_id = 0;

        const auto &dispatch_core_type = tt::llrt::OptionsG.get_dispatch_core_type();
        this->device_ = tt::tt_metal::CreateDevice(device_id, 1, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
    }

    void TearDown() override {
        if (!getenv("TT_METAL_SLOW_DISPATCH_MODE")){
            tt::tt_metal::CloseDevice(this->device_);
        }
    }
};


class CommandQueueMultiDeviceFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            TT_THROW("This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            GTEST_SKIP();
        }
        arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());

        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        if (num_devices_ < 2 ) {
            GTEST_SKIP();
        }
        std::vector<chip_id_t> chip_ids;
        for (unsigned int id = 0; id < num_devices_; id++) {
            chip_ids.push_back(id);
        }

        const auto &dispatch_core_type = tt::llrt::OptionsG.get_dispatch_core_type();
        reserved_devices_ = tt::tt_metal::detail::CreateDevices(chip_ids, 1, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
        for (const auto &[id, device] : reserved_devices_) {
            devices_.push_back(device);
        }
    }

    void TearDown() override { tt::tt_metal::detail::CloseDevices(reserved_devices_); }

    std::vector<tt::tt_metal::Device*> devices_;
    std::map<chip_id_t, tt::tt_metal::Device*> reserved_devices_;
    tt::ARCH arch_;
    size_t num_devices_;
};

class CommandQueueSingleCardFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            TT_THROW("This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            GTEST_SKIP();
        }
        auto enable_remote_chip = getenv("TT_METAL_ENABLE_REMOTE_CHIP");
        arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());

        const auto &dispatch_core_type = tt::llrt::OptionsG.get_dispatch_core_type();
        const chip_id_t mmio_device_id = 0;
        reserved_devices_ = tt::tt_metal::detail::CreateDevices({mmio_device_id}, 1, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
        if (enable_remote_chip) {
            for (const auto &[id, device] : reserved_devices_) {
                devices_.push_back(device);
            }
        } else {
            devices_.push_back(reserved_devices_.at(mmio_device_id));
        }

        num_devices_ = reserved_devices_.size();
    }

    void TearDown() override { tt::tt_metal::detail::CloseDevices(reserved_devices_); }

    std::vector<tt::tt_metal::Device*> devices_;
    std::map<chip_id_t, tt::tt_metal::Device*> reserved_devices_;
    tt::ARCH arch_;
    size_t num_devices_;
};

class SingleDeviceTraceFixture: public ::testing::Test {
protected:
    tt::tt_metal::Device* device_;
    tt::ARCH arch_;

    void Setup(const size_t buffer_size, const uint8_t num_hw_cqs = 1) {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            tt::log_info(tt::LogTest, "This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            GTEST_SKIP();
        }
        if (num_hw_cqs > 1) {
            // Running multi-CQ test. User must set this explicitly.
            auto num_cqs = getenv("TT_METAL_GTEST_NUM_HW_CQS");
            if (num_cqs == nullptr or strcmp(num_cqs, "2")) {
                TT_THROW("This suite must be run with TT_METAL_GTEST_NUM_HW_CQS=2");
                GTEST_SKIP();
            }
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
        const int device_id = 0;
        const auto &dispatch_core_type = tt::llrt::OptionsG.get_dispatch_core_type();
        const chip_id_t mmio_device_id = 0;
        this->device_ = tt::tt_metal::detail::CreateDevices({mmio_device_id}, 1, DEFAULT_L1_SMALL_SIZE, buffer_size, dispatch_core_type).at(mmio_device_id);
    }

    void TearDown() override {
        if (!getenv("TT_METAL_SLOW_DISPATCH_MODE")) {
            tt::tt_metal::CloseDevice(this->device_);
        }
    }

};

class RandomProgramFixture : public CommandQueueSingleCardFixture {
    protected:
     static const uint32_t MIN_KERNEL_SIZE_BYTES = 20;
     static const uint32_t MAX_KERNEL_SIZE_BYTES = 4096;
     static const uint32_t MIN_KERNEL_RUNTIME_MICROSECONDS = 100;
     static const uint32_t MAX_KERNEL_RUNTIME_MICROSECONDS = 20000;
     static const uint32_t MIN_NUM_RUNTIME_ARGS = 0;
     static const uint32_t MAX_NUM_RUNTIME_ARGS = max_runtime_args;
     static const uint32_t UNIQUE_RUNTIME_ARGS_VAL_OFFSET = 50;
     static const uint32_t COMMON_RUNTIME_ARGS_VAL_OFFSET = 100;
     static const uint32_t MIN_NUM_SEMS = 0;
     static const uint32_t MAX_NUM_SEMS = NUM_SEMAPHORES;
     static const uint32_t SEM_VAL = 1;
     static const uint32_t NUM_PROGRAMS = 75;

     Device *device_;

     void SetUp() override {
         CommandQueueSingleCardFixture::SetUp();

         this->device_ = this->devices_[0];

         this->seed_ = tt::parse_env("TT_METAL_SEED", static_cast<uint32_t>(time(nullptr)));
         log_info(tt::LogTest, "Using seed: {}", this->seed_);
         srand(this->seed_);
     }

     void create_kernel(
         Program &program,
         const CoreType kernel_core_type,
         const bool simple_kernel = false,
         const uint32_t min_num_sems = MIN_NUM_SEMS,
         const uint32_t max_num_sems = MAX_NUM_SEMS,
         const uint32_t min_num_rt_args = MIN_NUM_RUNTIME_ARGS,
         const uint32_t max_num_rt_args = MAX_NUM_RUNTIME_ARGS,
         const uint32_t min_kernel_size_bytes = MIN_KERNEL_SIZE_BYTES,
         const uint32_t max_kernel_size_bytes = MAX_KERNEL_SIZE_BYTES,
         const uint32_t min_kernel_runtime_microseconds = MIN_KERNEL_RUNTIME_MICROSECONDS,
         const uint32_t max_kernel_runtime_microseconds = MAX_KERNEL_RUNTIME_MICROSECONDS) {
         CoreRangeSet cores = this->get_cores(kernel_core_type);
         const bool create_eth_config = kernel_core_type == CoreType::ETH;

         if (simple_kernel) {
             this->create_kernel(program, cores, create_eth_config, 0, 0, 0, 0);
         } else {
             const vector<uint32_t> sem_ids =
                 this->generate_semaphores(program, cores, kernel_core_type, min_num_sems, max_num_sems);
             const auto [unique_rt_args, common_rt_args] =
                 this->generate_runtime_args(sem_ids, min_num_rt_args, max_num_rt_args);
             const uint32_t num_unique_rt_args = unique_rt_args.size() - sem_ids.size();
             KernelHandle kernel_id = this->create_kernel(
                 program,
                 cores,
                 create_eth_config,
                 sem_ids.size(),
                 num_unique_rt_args,
                 common_rt_args.size(),
                 min_kernel_size_bytes,
                 max_kernel_size_bytes,
                 min_kernel_runtime_microseconds,
                 max_kernel_runtime_microseconds);
             SetRuntimeArgs(program, kernel_id, cores, unique_rt_args);
             SetCommonRuntimeArgs(program, kernel_id, common_rt_args);
         }
     }

     vector<uint32_t> generate_semaphores(
         Program &program,
         const CoreRangeSet &cores,
         const CoreType core_type = CoreType::WORKER,
         const uint32_t min = MIN_NUM_SEMS,
         const uint32_t max = MAX_NUM_SEMS) {
         const uint32_t num_sems = this->generate_random_num(min, max);
         vector<uint32_t> sem_ids;
         for (uint32_t i = 0; i < num_sems; i++) {
             const uint32_t sem_id = CreateSemaphore(program, cores, SEM_VAL, core_type);
             sem_ids.push_back(sem_id);
         }
         return sem_ids;
     }

     pair<vector<uint32_t>, vector<uint32_t>> generate_runtime_args(
         const vector<uint32_t> &sem_ids,
         const uint32_t min = MIN_NUM_RUNTIME_ARGS,
         const uint32_t max = MAX_NUM_RUNTIME_ARGS) {
         const uint32_t num_sems = sem_ids.size();
         TT_FATAL(max >= num_sems, "Max number of runtime args to generate must be >= number of semaphores created");
         const uint32_t max_num_unique_rt_args = max - num_sems;
         const uint32_t num_unique_rt_args = this->generate_random_num(min, max_num_unique_rt_args);

         const uint32_t max_num_common_rt_args = max_num_unique_rt_args - num_unique_rt_args;
         const uint32_t num_common_rt_args = this->generate_random_num(0, max_num_common_rt_args);

         auto [unique_rt_args, common_rt_args] = create_runtime_args(
             num_unique_rt_args, num_common_rt_args, UNIQUE_RUNTIME_ARGS_VAL_OFFSET, COMMON_RUNTIME_ARGS_VAL_OFFSET);

         unique_rt_args.insert(unique_rt_args.end(), sem_ids.begin(), sem_ids.end());

         return {unique_rt_args, common_rt_args};
     }

    private:
     uint32_t seed_;

     KernelHandle create_kernel(
         Program &program,
         const CoreRangeSet &cores,
         const bool create_eth_config,
         const uint32_t num_sems,
         const uint32_t num_unique_rt_args,
         const uint32_t num_common_rt_args,
         const uint32_t min_kernel_size_bytes = MIN_KERNEL_SIZE_BYTES,
         const uint32_t max_kernel_size_bytes = MAX_KERNEL_SIZE_BYTES,
         const uint32_t min_kernel_runtime_microseconds = MIN_KERNEL_RUNTIME_MICROSECONDS,
         const uint32_t max_kernel_runtime_microseconds = MAX_KERNEL_RUNTIME_MICROSECONDS) {
         std::vector<uint32_t> compile_args = {
             num_unique_rt_args,
             num_common_rt_args,
             UNIQUE_RUNTIME_ARGS_VAL_OFFSET,
             COMMON_RUNTIME_ARGS_VAL_OFFSET,
             num_sems,
             SEM_VAL};

         uint32_t divisible_by;
         if (create_eth_config) {
             divisible_by = 4;
         } else {
             divisible_by = 1;
         }

         const uint32_t kernel_size_bytes =
             this->generate_random_num(min_kernel_size_bytes, max_kernel_size_bytes, divisible_by);
         const uint32_t kernel_runtime_microseconds =
             this->generate_random_num(min_kernel_runtime_microseconds, max_kernel_runtime_microseconds);

         const std::map<string, string> defines = {
             {"KERNEL_SIZE_BYTES", std::to_string(kernel_size_bytes)},
             {"KERNEL_RUNTIME_MICROSECONDS", std::to_string(kernel_runtime_microseconds)}};

         std::variant<DataMovementConfig, ComputeConfig, EthernetConfig> config;
         if (create_eth_config) {
             compile_args.push_back(static_cast<uint32_t>(HalProgrammableCoreType::ACTIVE_ETH));
             config = EthernetConfig{.compile_args = compile_args, .defines = defines};
         } else {
             compile_args.push_back(static_cast<uint32_t>(HalProgrammableCoreType::TENSIX));
             DataMovementProcessor processor = this->get_processor();
             config = DataMovementConfig{.processor = processor, .compile_args = compile_args, .defines = defines};
         }

         KernelHandle kernel_id = CreateKernel(
             program,
             "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/"
             "dispatcher_kernel_size_and_runtime.cpp",
             cores,
             config);
         return kernel_id;
     }

     // Generates a random number within the given bounds (inclusive) that is divisible by divisible_by
     uint32_t generate_random_num(const uint32_t min, const uint32_t max, const uint32_t divisible_by = 1) {
        TT_FATAL(max >= min, "max: {}, min: {} - max must be >= min", max, min);
        return min + (rand() % ((max - min) / divisible_by + 1)) * divisible_by;
     }

     DataMovementProcessor get_processor() {
        const uint32_t num = this->generate_random_num(0, 1);
        DataMovementProcessor processor;
        if (num == 0) {
            processor = DataMovementProcessor::RISCV_0;
        } else {
            processor = DataMovementProcessor::RISCV_1;
        }
        return processor;
     }

     CoreRangeSet get_cores(const CoreType core_type) {
        CoreRangeSet all_cores({});
        if (core_type == CoreType::WORKER) {
            CoreCoord worker_grid_size = device_->compute_with_storage_grid_size();
            all_cores = CoreRangeSet({{{0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1}}});
        }
        else {
            std::set<CoreRange> core_ranges;
            const std::unordered_set<CoreCoord> active_eth_cores = this->device_->get_active_ethernet_cores(true);
            for (CoreCoord eth_core : active_eth_cores) {
                core_ranges.emplace(eth_core);
            }
            all_cores = CoreRangeSet(core_ranges);
        }
        CoreRangeSet empty_crs({});
        all_cores = empty_crs.merge(all_cores);

        CoreRangeSet cores({});
        const uint32_t num = this->generate_random_num(0, 2);
        switch (num) {
            case 0: cores = all_cores; break;
            case 1: cores = this->generate_subset_of_cores(all_cores, 2); break;
            case 2: cores = this->generate_subset_of_cores(all_cores, 4); break;
        }

        return cores;
     }

     CoreRangeSet generate_subset_of_cores(const CoreRangeSet& cores, const uint32_t resulting_ratio_of_cores) {
        uint32_t num_cores = 0;
        for (CoreRange cr : cores.ranges()) {
            num_cores += cr.size();
        }

        std::set<CoreRange> cores_subset;
        const uint32_t num_cores_to_include_in_subset = num_cores / resulting_ratio_of_cores;
        uint32_t num_cores_added = 0;
        while (num_cores_added != num_cores_to_include_in_subset) {
            for (CoreRange cr : cores.ranges()) {
                for (CoreCoord core : cr) {
                    const uint32_t random_num = this->generate_random_num(1, resulting_ratio_of_cores);
                    if (random_num == 1 && num_cores_added != num_cores_to_include_in_subset) {
                        cores_subset.emplace(core);
                        num_cores_added += 1;
                    }
                }
            }
        }

        CoreRangeSet empty_crs({});
        CoreRangeSet resulting_cores = empty_crs.merge(cores_subset);
        return resulting_cores;
     }
};
