// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/kernel.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "erisc_datamover_builder.hpp"
#include "tt-metalium/kernel_types.hpp"
#include <tt-metalium/fabric.hpp>
#include "erisc_datamover_builder_helper.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/common/executor.hpp"
#include "tt_metal/fabric/erisc_datamover_builder_helper.hpp"
#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tt_stl/small_vector.hpp"
#include <tt-metalium/fabric_types.hpp>

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/mesh_buffer.hpp>

#include <umd/device/types/arch.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include "gtest/gtest.h"

#include <algorithm>
#include <cstddef>
#include <limits>

#include <tt_metal/fabric/ccl/ccl_common.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

using tt::tt_metal::distributed::MeshBuffer;
using tt::tt_metal::distributed::MeshCommandQueue;
using tt::tt_metal::distributed::MeshContainer;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshCoordinateRange;
using tt::tt_metal::distributed::MeshDevice;
using tt::tt_metal::distributed::MeshDeviceConfig;
using tt::tt_metal::distributed::MeshDeviceView;
using tt::tt_metal::distributed::MeshShape;
using tt::tt_metal::distributed::MeshWorkload;
using tt::tt_metal::distributed::SystemMesh;

class Fabric1DFixture {
public:
    tt::ARCH arch_{tt::ARCH::Invalid};
    std::size_t num_devices_{};
    bool device_open = false;

    // Common constants for both fixtures
    static constexpr size_t TG_NUM_DEVICES = 36;
    static constexpr size_t GALAXY_6U_NUM_DEVICES = 32;

    std::shared_ptr<MeshDevice> mesh_device_;

    // Gets the appropriate mesh shape based on device configuration
    MeshShape GetDeterminedMeshShape() const { return SystemMesh::instance().shape(); }

    // Validates environment and hardware for tests
    void ValidateEnvironment() {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            TT_THROW("This suite can only be run without TT_METAL_SLOW_DISPATCH_MODE set");
        }

        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        num_devices_ = tt::tt_metal::GetNumAvailableDevices();

        if (num_devices_ < 2) {
            TT_THROW("This suite can only be run on 2+ device systems");
        }
    }

    void SetupDevices() {
        ValidateEnvironment();

        const MeshShape cluster_shape = GetDeterminedMeshShape();
        tt::tt_metal::MetalContext::instance().get_control_plane();
        mesh_device_ = MeshDevice::create(MeshDeviceConfig(cluster_shape));
        device_open = true;
    }

    void TearDown() {
        if (device_open) {
            mesh_device_->close();
            device_open = false;
        }
    }

    Fabric1DFixture() : device_open(false) { this->SetupDevices(); }

    Fabric1DFixture(
        tt::tt_fabric::FabricConfig fabric_config,
        tt::tt_fabric::FabricReliabilityMode reliability_mode =
            tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE) :
        device_open(false) {
        tt::tt_fabric::SetFabricConfig(fabric_config, reliability_mode);
        this->SetupDevices();
    }

    ~Fabric1DFixture() {
        TearDown();
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    }
};

class Fabric1DLineDeviceInitFixture : public Fabric1DFixture {
public:
    Fabric1DLineDeviceInitFixture() : Fabric1DFixture(tt::tt_fabric::FabricConfig::FABRIC_1D) {}
};

class Fabric1DRingDeviceInitFixture : public Fabric1DFixture {
public:
    Fabric1DRingDeviceInitFixture() : Fabric1DFixture(tt::tt_fabric::FabricConfig::FABRIC_1D_RING) {}
};

template <typename ProgramContainer>
static void build_and_enqueue(
    const std::vector<std::shared_ptr<MeshDevice>>& devices, ProgramContainer& programs, bool enqueue_only = false) {
    static_assert(
        std::is_same_v<ProgramContainer, std::vector<Program*>> ||
            std::is_same_v<ProgramContainer, std::vector<Program>>,
        "programs must be a vector of Program* or Program");
    TT_FATAL(
        devices.size() == programs.size(),
        "Number of devices must match number of programs when calling build_and_enqueue in test");

    // Parallel compile and enqueue as a single atomic operation per device
    std::vector<std::shared_future<void>> futures;
    futures.reserve(devices.size());

    for (size_t i = 0; i < devices.size(); i++) {
        futures.emplace_back(tt::tt_metal::detail::async([&devices, &programs, i, enqueue_only]() {
            if constexpr (std::is_same_v<ProgramContainer, std::vector<Program*>>) {
                if (!enqueue_only) {
                    tt::tt_metal::detail::CompileProgram(devices[i]->get_devices()[0], *programs[i]);
                }
                MeshWorkload mesh_workload = tt::tt_metal::distributed::CreateMeshWorkload();
                MeshCoordinateRange device_range = MeshCoordinateRange({0, 0}, {0, 0});  // Single device range
                tt::tt_metal::distributed::AddProgramToMeshWorkload(
                    mesh_workload, std::move(*programs[i]), device_range);
                tt::tt_metal::distributed::EnqueueMeshWorkload(devices[i]->mesh_command_queue(), mesh_workload, false);
            } else {
                if (!enqueue_only) {
                    tt::tt_metal::detail::CompileProgram(devices[i]->get_devices()[0], programs[i]);
                }
                MeshWorkload mesh_workload = tt::tt_metal::distributed::CreateMeshWorkload();
                MeshCoordinateRange device_range = MeshCoordinateRange({0, 0}, {0, 0});  // Single device range
                tt::tt_metal::distributed::AddProgramToMeshWorkload(
                    mesh_workload, std::move(programs[i]), device_range);
                tt::tt_metal::distributed::EnqueueMeshWorkload(devices[i]->mesh_command_queue(), mesh_workload, false);
            }
        }));
    }

    // Wait for all compile and enqueue operations to complete
    for (const auto& future : futures) {
        future.get();
    }
}

static void wait_for_worker_program_completion(const std::vector<std::shared_ptr<MeshDevice>>& devices) {
    std::ranges::for_each(
        devices, [&](const std::shared_ptr<MeshDevice>& d) { tt_metal::distributed::Finish(d->mesh_command_queue()); });
}
