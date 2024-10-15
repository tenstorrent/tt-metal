// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <variant>
#include <vector>
#include "common/core_coord.h"
#include "common/env_lib.hpp"
#include "gtest/gtest.h"
#include "impl/kernels/kernel_types.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/common/tt_backend_api_types.hpp"
#include "tt_metal/llrt/rtoptions.hpp"
#include "tt_metal/tt_metal/unit_tests_common/common/test_utils.hpp"

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
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

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
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

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
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

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
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
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

class RandomProgramFixture : public CommandQueueFixture {
    protected:
     uint32_t seed_;

     static const uint32_t MIN_KERNEL_SIZE_BYTES = 20;
     static const uint32_t MAX_KERNEL_SIZE_BYTES = 8192;
     static const uint32_t MIN_KERNEL_RUNTIME_CYCLES = 100;
     static const uint32_t MAX_KERNEL_RUNTIME_CYCLES = 20000;
     static const uint32_t MIN_NUM_RUNTIME_ARGS = 0;
     static const uint32_t MAX_NUM_RUNTIME_ARGS = max_runtime_args;
     static const uint32_t UNIQUE_RUNTIME_ARGS_VAL_OFFSET = 50;
     static const uint32_t COMMON_RUNTIME_ARGS_VAL_OFFSET = 100;
     static const uint32_t MIN_NUM_SEMS = 0;
     static const uint32_t MAX_NUM_SEMS = NUM_SEMAPHORES;
     static const uint32_t SEM_VAL = 1;
     static const uint32_t NUM_PROGRAMS = 100;

     void SetUp() override {
        CommandQueueFixture::SetUp();

        this->seed_ = tt::parse_env("SEED", 0);
        srand(this->seed_);
        log_info(tt::LogTest, "Using seed: {}", this->seed_);
     }

     vector<uint32_t> generate_semaphores(
         Program &program,
         const std::variant<CoreRange, CoreRangeSet> &cores,
         const uint32_t min = MIN_NUM_SEMS,
         const uint32_t max = MAX_NUM_SEMS) {
         const uint32_t num_sems = this->generate_random_num(min, max);
         vector<uint32_t> sem_ids;
         for (uint32_t i = 0; i < num_sems; i++) {
             const uint32_t sem_id = CreateSemaphore(program, cores, SEM_VAL);
             sem_ids.push_back(sem_id);
         }
         return sem_ids;
     }

     pair<vector<uint32_t>, vector<uint32_t>> generate_runtime_args(
         const vector<uint32_t> &sem_ids,
         const uint32_t min = MIN_NUM_RUNTIME_ARGS,
         const uint32_t max = MAX_NUM_RUNTIME_ARGS) {
         const uint32_t num_sems = sem_ids.size();
         const uint32_t max_num_unique_rt_args = max - num_sems;
         const uint32_t num_unique_rt_args = this->generate_random_num(min, max_num_unique_rt_args);

         const uint32_t max_num_common_rt_args = max_num_unique_rt_args - num_unique_rt_args;
         const uint32_t num_common_rt_args = this->generate_random_num(0, max_num_common_rt_args);

         auto [unique_rt_args, common_rt_args] = create_runtime_args(
             num_unique_rt_args, num_common_rt_args, UNIQUE_RUNTIME_ARGS_VAL_OFFSET, COMMON_RUNTIME_ARGS_VAL_OFFSET);

         unique_rt_args.insert(unique_rt_args.end(), sem_ids.begin(), sem_ids.end());

         return {unique_rt_args, common_rt_args};
     }

     KernelHandle create_kernel(
         Program &program,
         const std::variant<CoreCoord, CoreRange, CoreRangeSet> &cores,
         const uint32_t num_sems,
         const uint32_t num_unique_rt_args,
         const uint32_t num_common_rt_args,
         const bool create_eth_config,
         const uint32_t min_kernel_size_bytes = MIN_KERNEL_SIZE_BYTES,
         const uint32_t max_kernel_size_bytes = MAX_KERNEL_SIZE_BYTES,
         const uint32_t min_kernel_runtime_cycles = MIN_KERNEL_RUNTIME_CYCLES,
         const uint32_t max_kernel_runtime_cycles = MAX_KERNEL_RUNTIME_CYCLES) {
         const std::vector<uint32_t> compile_args = {
             num_unique_rt_args,
             num_common_rt_args,
             UNIQUE_RUNTIME_ARGS_VAL_OFFSET,
             COMMON_RUNTIME_ARGS_VAL_OFFSET,
             num_sems,
             SEM_VAL};

         const uint32_t kernel_size_bytes =
             this->generate_random_num(min_kernel_size_bytes, max_kernel_size_bytes);
         const uint32_t kernel_runtime_cycles =
             this->generate_random_num(min_kernel_runtime_cycles, max_kernel_runtime_cycles);

         const std::map<string, string> defines = {
             {"KERNEL_SIZE_BYTES", std::to_string(kernel_size_bytes)},
             {"KERNEL_RUNTIME_CYCLES", std::to_string(kernel_runtime_cycles)}};

         std::variant<DataMovementConfig, ComputeConfig, EthernetConfig> config;
         if (create_eth_config) {
             config = EthernetConfig{.compile_args = compile_args, .defines = defines};
         } else {
             config = DataMovementConfig{.compile_args = compile_args, .defines = defines};
         }

         KernelHandle kernel_id = CreateKernel(
             program,
             "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/"
             "dispatcher_kernel_size_and_runtime.cpp",
             cores,
             config);
         return kernel_id;
     }

    private:
     // Generates a random number within the given bounds (inclusive) that is divisible by divisible_by
     uint32_t generate_random_num(const uint32_t min, const uint32_t max, const uint32_t divisible_by = 1) {
        return min + (rand() % ((max - min) / divisible_by + 1)) * divisible_by;
     }
};
