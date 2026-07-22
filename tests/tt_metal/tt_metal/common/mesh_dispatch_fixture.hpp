// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <umd/device/types/cluster_descriptor_types.hpp>
#include "gtest/gtest.h"
#include <functional>
#include <map>
#include <memory>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "hostdevcommon/common_values.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/program.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include "llrt.hpp"
#include "common/tt_backend_api_types.hpp"
#include <llrt/tt_cluster.hpp>

namespace tt::tt_metal {

struct SharedDevices {
    std::map<ChipId, std::shared_ptr<distributed::MeshDevice>> id_to_device;
    std::vector<std::shared_ptr<distributed::MeshDevice>> devices;
    size_t l1_small_size {DEFAULT_L1_SMALL_SIZE};
    size_t trace_region_size {DEFAULT_TRACE_REGION_SIZE};
    bool initialized {false};
    bool needs_recovery {false};
};

// A dispatch-agnostic test fixture
class MeshDispatchFixture : public ::testing::Test {
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
    bool IsSlowDispatch() const { return this->slow_dispatch_; }

protected:
    tt::ARCH arch_{tt::ARCH::Invalid};
    std::vector<std::shared_ptr<distributed::MeshDevice>> devices_;
    bool slow_dispatch_{};
    const size_t l1_small_size_{DEFAULT_L1_SMALL_SIZE};
    const size_t trace_region_size_{DEFAULT_TRACE_REGION_SIZE};
    uint32_t max_cbs_{};

    MeshDispatchFixture(
        size_t l1_small_size = DEFAULT_L1_SMALL_SIZE, size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE) :
        l1_small_size_{l1_small_size}, trace_region_size_{trace_region_size} {};

    static SharedDevices& get_shared_devices() {
        static SharedDevices devices;
        return devices;
    }

    static void create_shared_devices(size_t l1_small_size = DEFAULT_L1_SMALL_SIZE, size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE) {
        auto& shared = get_shared_devices();
        if (shared.initialized) {
            return;
        }

        std::vector<ChipId> ids;
        for (ChipId id : tt::tt_metal::MetalContext::instance().get_cluster().user_exposed_chip_ids()) {
            ids.push_back(id);
        }
        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        shared.id_to_device = distributed::MeshDevice::create_unit_meshes(
            ids, l1_small_size, trace_region_size, 1, dispatch_core_config);
        shared.devices.clear();
        for (const auto& [device_id, device] : shared.id_to_device) {
            shared.devices.push_back(device);
        }
        shared.l1_small_size = l1_small_size;
        shared.trace_region_size = trace_region_size;
        shared.initialized = true;
        shared.needs_recovery = false;
    }

    static void destroy_shared_devices() {
        auto& shared = get_shared_devices();
        if (!shared.initialized) {
            shared.needs_recovery = false;
            return;
        }

        for (auto& [device_id, device] : shared.id_to_device) {
            device->close();
            device.reset();
        }
        shared.id_to_device.clear();
        shared.devices.clear();
        shared.initialized = false;
        shared.needs_recovery = false;
    }

    // Derived fixtures should override SetUpTestSuite/TearDownTestSuite and SetUp/TearDown if
    // different device instantiation is needed
    static void SetUpTestSuite() {
        // Create a shared device to reduce individual test startup time
        create_shared_devices();
    }

    static void TearDownTestSuite() {
        destroy_shared_devices();
    }

    void SetUp() override {
        auto& shared = get_shared_devices();
        ASSERT_TRUE(shared.initialized) << "Shared devices not initialized, if this is a derived fixture then "
                                           "MeshDispatchFixture::SetUpTestSuite may have been overridden but not "
                                           "MeshDispatchFixture::SetUp";

        if (shared.needs_recovery || shared.l1_small_size != l1_small_size_ || shared.trace_region_size != trace_region_size_) {
            destroy_shared_devices();
        }
        if (!shared.initialized) {
            create_shared_devices(l1_small_size_, trace_region_size_);
        }
        this->devices_ = shared.devices;

        this->DetectDispatchMode();
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        init_max_cbs();
    }

    void TearDown() override {
        ASSERT_TRUE(get_shared_devices().initialized)
            << "Shared devices not initialized, if this is a derived fixture then "
               "MeshDispatchFixture::SetUpTestSuite may have been overridden but not "
               "MeshDispatchFixture::TearDown";
        devices_.clear();
        if (HasFailure()) {
            get_shared_devices().needs_recovery = true;
        }
    }

    void RunTestOnDevice(
        const std::function<void()>& run_function, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
        auto* device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "Running test on device {}.", device->id());
        run_function();
        log_info(tt::LogTest, "Finished running test on device {}.", device->id());
    }

    void DetectDispatchMode() {
        auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            log_info(tt::LogTest, "Running test using Slow Dispatch");
            this->slow_dispatch_ = true;
        } else {
            log_info(tt::LogTest, "Running test using Fast Dispatch");
            this->slow_dispatch_ = false;
        }
    }

    void init_max_cbs() { max_cbs_ = tt::tt_metal::MetalContext::instance().hal().get_arch_num_circular_buffers(); }
};

}  // namespace tt::tt_metal
