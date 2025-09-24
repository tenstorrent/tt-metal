// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <umd/device/types/cluster_descriptor_types.hpp>
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
#include <tt-metalium/distributed.hpp>
#include "llrt.hpp"

namespace tt::tt_metal {

// A dispatch-agnostic test fixture
class MeshDispatchFixture : public ::testing::Test {
private:
    std::map<chip_id_t, std::shared_ptr<distributed::MeshDevice>> id_to_device_;

public:
    // A function to run a program, according to which dispatch mode is set.
    void RunProgram(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device,
        distributed::MeshWorkload& workload,
        const bool skip_finish = false) {
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
        if (!skip_finish) {
            distributed::Finish(mesh_device->mesh_command_queue());
        }
    }
    void FinishCommands(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
        distributed::Finish(mesh_device->mesh_command_queue());
    }
    void WriteBuffer(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device,
        const std::shared_ptr<distributed::MeshBuffer>& in_buffer,
        std::vector<uint32_t>& src_vec) {
        distributed::WriteShard(
            mesh_device->mesh_command_queue(), in_buffer, src_vec, distributed::MeshCoordinate(0, 0));
    }
    void ReadBuffer(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device,
        const std::shared_ptr<distributed::MeshBuffer>& out_buffer,
        std::vector<uint32_t>& dst_vec) {
        distributed::ReadShard(
            mesh_device->mesh_command_queue(), dst_vec, out_buffer, distributed::MeshCoordinate(0, 0));
    }
    int NumDevices() { return this->devices_.size(); }
    bool IsSlowDispatch() { return this->slow_dispatch_; }

protected:
    tt::ARCH arch_{tt::ARCH::Invalid};
    std::vector<std::shared_ptr<distributed::MeshDevice>> devices_;
    bool slow_dispatch_{};
    const size_t l1_small_size_{DEFAULT_L1_SMALL_SIZE};
    const size_t trace_region_size_{DEFAULT_TRACE_REGION_SIZE};

    MeshDispatchFixture(
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
        id_to_device_ = distributed::MeshDevice::create_unit_meshes(
            ids, l1_small_size_, trace_region_size_, 1, dispatch_core_config);
        devices_.clear();
        for (const auto& [device_id, device] : id_to_device_) {
            devices_.push_back(device);
        }
    }

    void TearDown() override {
        // Checking if devices are empty because DPrintFixture.TensixTestPrintFinish already
        // closed all devices
        if (!id_to_device_.empty()) {
            for (auto [device_id, device] : id_to_device_) {
                device->close();
                device.reset();
            }

            id_to_device_.clear();
            devices_.clear();
        }
    }

    void RunTestOnDevice(
        const std::function<void()>& run_function, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
        auto device = mesh_device->get_devices()[0];
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
