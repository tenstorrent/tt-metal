// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <umd/device/types/cluster_descriptor_types.h>
#include "gtest/gtest.h"
#include <map>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "hostdevcommon/common_values.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/program.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/device_pool.hpp>
#include "llrt.hpp"

namespace tt::tt_metal {

// A dispatch-agnostic test fixture
class DispatchFixture : public ::testing::Test {
private:
    std::map<chip_id_t, tt::tt_metal::IDevice*> id_to_device_;

public:
    // A function to run a program, according to which dispatch mode is set.
    void RunProgram(tt::tt_metal::IDevice* device, tt::tt_metal::Program& program, const bool skip_finish = false) {
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
        // Must set up all available devices
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        std::vector<chip_id_t> ids;
        for (chip_id_t id : tt::tt_metal::MetalContext::instance().get_cluster().user_exposed_chip_ids()) {
            ids.push_back(id);
        }
        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        id_to_device_ = tt::tt_metal::detail::CreateDevices(
            ids,
            tt::tt_metal::MetalContext::instance().rtoptions().get_num_hw_cqs(),
            l1_small_size_,
            DEFAULT_TRACE_REGION_SIZE,
            dispatch_core_config);
        devices_.clear();
        for (auto [device_id, device] : id_to_device_) {
            devices_.push_back(device);
        }
    }

    void TearDown() override {
        // Checking if devices are empty because DPrintFixture.TensixTestPrintFinish already
        // closed all devices
        if (!id_to_device_.empty()) {
            tt::tt_metal::detail::CloseDevices(id_to_device_);
            id_to_device_.clear();
            devices_.clear();
        }
    }

    void RunTestOnDevice(const std::function<void()>& run_function, tt::tt_metal::IDevice* device) {
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
