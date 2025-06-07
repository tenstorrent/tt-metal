// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gtest/gtest.h"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/program.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/device_pool.hpp>
#include "llrt.hpp"

namespace tt::tt_metal {

// A dispatch-agnostic test fixture
class DispatchFixture : public ::testing::Test {
public:
    // A function to run a program, according to which dispatch mode is set.
    void RunProgram(tt::tt_metal::IDevice* device, tt::tt_metal::Program& program, const bool skip_finish = false) {
        const uint64_t program_id = program.get_id();
        if (this->slow_dispatch_) {
            tt::tt_metal::detail::LaunchProgram(device, program);
        } else {
            tt::tt_metal::CommandQueue& cq = device->command_queue();
            tt::tt_metal::EnqueueProgram(cq, program, false);
            if (!skip_finish) {
                tt::tt_metal::Finish(cq);
            }
        }
    }
    void FinishCommands(tt::tt_metal::IDevice* device) {
        if (!this->IsSlowDispatch()) {
            tt::tt_metal::CommandQueue& cq = device->command_queue();
            tt::tt_metal::Finish(cq);
        }
    }
    void WriteBuffer(
        tt::tt_metal::IDevice* device, std::shared_ptr<tt::tt_metal::Buffer> in_buffer, std::vector<uint32_t>& src_vec) {
        if (this->slow_dispatch_) {
            tt::tt_metal::detail::WriteToBuffer(in_buffer, src_vec);
        } else {
            tt::tt_metal::CommandQueue& cq = device->command_queue();
            tt::tt_metal::EnqueueWriteBuffer(cq, in_buffer, src_vec, false);
        }
    }
    void ReadBuffer(
        tt::tt_metal::IDevice* device,
        const std::shared_ptr<tt::tt_metal::Buffer>& out_buffer,
        std::vector<uint32_t>& dst_vec) {
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
    std::vector<tt::tt_metal::IDevice*> devices_;
    bool slow_dispatch_;
    const size_t l1_small_size_{DEFAULT_L1_SMALL_SIZE};
    const size_t trace_region_size_{DEFAULT_TRACE_REGION_SIZE};

    DispatchFixture(
        size_t l1_small_size = DEFAULT_L1_SMALL_SIZE, size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE) :
        l1_small_size_{l1_small_size}, trace_region_size_{trace_region_size} {};

    void SetUp() override {
        this->DetectDispatchMode();
        // Set up all available devices
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        auto num_devices = tt::tt_metal::GetNumAvailableDevices();
        std::vector<chip_id_t> ids;
        for (unsigned int id = 0; id < num_devices; id++) {
            if (SkipTest(id)) {
                continue;
            }
            ids.push_back(id);
        }
        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        tt::DevicePool::initialize(
            ids,
            tt::tt_metal::MetalContext::instance().rtoptions().get_num_hw_cqs(),
            l1_small_size_,
            DEFAULT_TRACE_REGION_SIZE,
            dispatch_core_config);
        devices_ = tt::DevicePool::instance().get_all_active_devices();
    }

    void TearDown() override {
        tt::tt_metal::MetalContext::instance().get_cluster().set_internal_routing_info_for_ethernet_cores(false);
        // Close all opened devices
        for (unsigned int id = 0; id < devices_.size(); id++) {
            // The test may ahve closed the device already, so only close if active.
            if (devices_.at(id)->is_initialized()) {
                tt::tt_metal::CloseDevice(devices_.at(id));
            }
        }
    }

    bool SkipTest(unsigned int device_id) {
        // Also skip all devices after device 0 for grayskull. This to match all other tests
        // targetting device 0 by default (and to not cause issues with BMs that have E300s).
        // TODO: when we can detect only supported devices, this check can be removed.
        if (this->arch_ == tt::ARCH::GRAYSKULL && device_id > 0) {
            log_info(tt::LogTest, "Skipping test on device {} due to unsupported E300", device_id);
            return true;
        }

        return false;
    }

    void RunTestOnDevice(const std::function<void()>& run_function, tt::tt_metal::IDevice* device) {
        if (SkipTest(device->id())) {
            return;
        }
        log_info(tt::LogTest, "Running test on device {}.", device->id());
        run_function();
        log_info(tt::LogTest, "Finished running test on device {}.", device->id());
    }

    void DetectDispatchMode() {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            log_info(tt::LogTest, "Running test using Slow Dispatch");
            this->slow_dispatch_ = true;
        } else {
            log_info(tt::LogTest, "Running test using Fast Dispatch");
            this->slow_dispatch_ = false;
        }
    }
};

}  // namespace tt::tt_metal
