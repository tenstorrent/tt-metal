// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/impl/program/program.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/device/device_pool.hpp"

// A dispatch-agnostic test fixture
class CommonFixture: public ::testing::Test {
public:
    // A function to run a program, according to which dispatch mode is set.
    void RunProgram(tt::tt_metal::Device* device, tt::tt_metal::Program& program) {
        static std::unordered_map<uint64_t, uint32_t> trace_captured;
        uint64_t program_id = program.get_id();
        if (this->slow_dispatch_) {
            tt::tt_metal::detail::LaunchProgram(device, program);
        } else {
            tt::tt_metal::CommandQueue& cq = device->command_queue();
            tt::tt_metal::EnqueueProgram(cq, program, false);
            tt::tt_metal::Finish(cq);
        }
    }
    void WriteBuffer(tt::tt_metal::Device* device, std::shared_ptr<tt::tt_metal::Buffer> in_buffer, std::vector<uint32_t> &src_vec){
        if (this->slow_dispatch_) {
            tt::tt_metal::detail::WriteToBuffer(in_buffer, src_vec);
        } else {
            tt::tt_metal::CommandQueue& cq = device->command_queue();
            tt::tt_metal::EnqueueWriteBuffer(cq, in_buffer, src_vec, false);
        }
    }
    void ReadBuffer(tt::tt_metal::Device* device, std::shared_ptr<tt::tt_metal::Buffer> out_buffer, std::vector<uint32_t> &dst_vec){
        if (this->slow_dispatch_) {
            tt::tt_metal::detail::ReadFromBuffer(out_buffer, dst_vec);
        } else {
            tt::tt_metal::CommandQueue& cq = device->command_queue();
            tt::tt_metal::EnqueueReadBuffer(cq, out_buffer, dst_vec, true);
        }
    }
    int NumDevices() { return this->devices_.size(); }
    bool IsSlowDispatch() { return this->slow_dispatch_; }


protected:
    tt::ARCH arch_;
    vector<tt::tt_metal::v1::DeviceHandle> devices_;
    bool slow_dispatch_;
    bool has_remote_devices_;

    void SetUp() override {
        // Skip for slow dispatch for now
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            tt::log_info(tt::LogTest, "Running test using Slow Dispatch");
            slow_dispatch_ = true;
        } else {
            tt::log_info(tt::LogTest, "Running test using Fast Dispatch");
            slow_dispatch_ = false;
        }

        // Set up all available devices
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
        auto num_devices = tt::tt_metal::GetNumAvailableDevices();
        auto num_pci_devices = tt::tt_metal::GetNumPCIeDevices();
        // An extra flag for if we have remote devices, as some tests are disabled for fast
        // dispatch + remote devices.
        this->has_remote_devices_ = num_devices > num_pci_devices;
        vector<chip_id_t> ids;
        for (unsigned int id = 0; id < num_devices; id++) {
            if (SkipTest(id))
                continue;
            ids.push_back(id);
        }
        const auto &dispatch_core_type = tt::llrt::OptionsG.get_dispatch_core_type();
        tt::DevicePool::initialize(ids, tt::llrt::OptionsG.get_num_hw_cqs(), DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
        devices_ = tt::DevicePool::instance().get_all_active_devices();
    }

    void TearDown() override {
        tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(false);
        // Close all opened devices
        for (unsigned int id = 0; id < devices_.size(); id++) {
            // The test may ahve closed the device already, so only close if active.
            if (devices_.at(id)->is_initialized())
                tt::tt_metal::CloseDevice(devices_.at(id));
        }
    }

    bool SkipTest(unsigned int device_id) {
        // Also skip all devices after device 0 for grayskull. This to match all other tests
        // targetting device 0 by default (and to not cause issues with BMs that have E300s).
        // TODO: when we can detect only supported devices, this check can be removed.
        if (this->arch_ == tt::ARCH::GRAYSKULL && device_id > 0) {
            log_info(
                tt::LogTest,
                "Skipping test on device {} due to unsupported E300",
                device_id
            );
            return true;
        }

        return false;
    }

    void RunTestOnDevice(
        const std::function<void()>& run_function,
        tt::tt_metal::Device* device
    ) {
        if (SkipTest(device->id()))
            return;
        log_info(tt::LogTest, "Running test on device {}.", device->id());
        run_function();
        log_info(tt::LogTest, "Finished running test on device {}.", device->id());
    }
};
