// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// JIT-compile smoke tests for GlobalSemaphore bindings: a data-movement kernel constructs
// GlobalSemaphore(global_sem::<accessor>) and calls every on-chip method (mock Wormhole),
// and every fabric method (mock 2-chip Wormhole with 1D and 2D fabric).

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestGen1;

constexpr const char* kKernelSource = R"(
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/global_semaphore.h"

void kernel_main() {
    GlobalSemaphore first(global_sem::first);
    GlobalSemaphore second(global_sem::second);
    Noc noc;
    first.up();
    first.up(noc, 1, 1, 2);
    first.wait_min(1);
    first.down(1);
    second.set(0);
    second.wait(0);
    second.set_multicast(noc, 1, 1, 2, 2, 4);
    second.inc_multicast(noc, 1, 1, 2, 2, 1, 4);
    volatile uint32_t value = first.value() + second.value();
    (void)value;
}
)";

ProgramSpec MakeSpec(KernelSpec dm_kernel) {
    dm_kernel.source = KernelSpec::SourceCode{kKernelSource};
    for (const char* name : {"first", "second"}) {
        dm_kernel.global_semaphore_bindings.push_back(GlobalSemaphoreBinding{
            .global_semaphore_parameter_name = GlobalSemaphoreParamName{name}, .accessor_name = name});
    }
    ProgramSpec spec;
    spec.name = "global_semaphore_smoke";
    spec.global_semaphore_parameters = {
        GlobalSemaphoreParameter{.unique_id = GlobalSemaphoreParamName{"first"}},
        GlobalSemaphoreParameter{.unique_id = GlobalSemaphoreParamName{"second"}},
    };
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", NodeCoord{0, 0}, {"dm_kernel"})};
    spec.kernels = {std::move(dm_kernel)};
    return spec;
}

TEST_F(ProgramSpecTestGen1, CPU_GlobalSemaphoreBindingJITSmoke) {
    Program program = MakeProgramFromSpec(*mesh_device_, MakeSpec(MakeMinimalGen1DMKernel("dm_kernel")));
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

constexpr const char* kFabricKernelSource = R"(
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/global_semaphore.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

#if !defined(GLOBAL_SEMAPHORE_FABRIC)
#error "GlobalSemaphore fabric calls are missing with fabric enabled"
#endif

void kernel_main() {
    size_t arg_idx = 0;
    auto sender = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
    auto* hdr = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(PacketHeaderPool::allocate_header());
    GlobalSemaphore sem(global_sem::remote);
    Noc noc;
#if defined(FABRIC_2D)
    const GlobalSemaphore::ToChip one{.dst_dev_id = 1, .dst_mesh_id = 0};
    const GlobalSemaphore::ToChips many{.dst_dev_id = 1, .dst_mesh_id = 0, .e = 1, .w = 0, .n = 0, .s = 0};
#else
    const GlobalSemaphore::ToChip one{.num_hops = 1};
    const GlobalSemaphore::ToChips many{.start_distance = 1, .range = 1};
#endif
    UnicastEndpoint ep;
    const noc_traits_t<UnicastEndpoint>::src_args_type src_args{.noc_x = 1, .noc_y = 1, .addr = 0x10000};
    const noc_traits_t<UnicastEndpoint>::dst_args_type dst_args{.noc_x = 1, .noc_y = 1, .addr = 0x20000};

    sem.up(sender, hdr, one, noc, 1, 1);
    sem.up(sender, hdr, many, noc, 1, 1, 2);
    sem.write_and_up(sender, hdr, one, noc, ep, src_args, ep, dst_args, 64, 1, 1);
    sem.write_and_up(sender, hdr, many, noc, ep, src_args, ep, dst_args, 64, 1, 1);
    sem.up_set_state(hdr, one, noc, 1, 1);
    sem.up_with_state(sender, hdr);
    sem.up_with_state(sender, hdr, noc, 2, 2);
    sem.up_set_state(hdr, many, noc, 1, 1);
    sem.up_with_state<GlobalSemaphore::ToChips>(sender, hdr);
    sem.up_with_state<GlobalSemaphore::ToChips>(sender, hdr, noc, 2, 2);
}
)";

// Fabric needs every chip open, so both mock chips are opened and the program is built on the first.
class GlobalSemaphoreFabricJITTest : public ::testing::TestWithParam<tt_fabric::FabricConfig> {
protected:
    void SetUp() override {
        env_ = std::make_unique<MetalEnv>(MetalEnvDescriptor{
            .mock_cluster_desc_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 2)});
        env_->configure_fabric({.fabric_config = GetParam()});
        devices_ = distributed::MeshDevice::open_unit_meshes(*env_, std::array{0, 1});
    }
    void TearDown() override {
        devices_.clear();
        env_.reset();
    }

    std::unique_ptr<MetalEnv> env_;  // outlives the meshes
    std::map<ChipId, std::shared_ptr<distributed::MeshDevice>> devices_;
};

TEST_P(GlobalSemaphoreFabricJITTest, CPU_GlobalSemaphoreFabricJITSmoke) {
    KernelSpec dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.source = KernelSpec::SourceCode{kFabricKernelSource};
    if (GetParam() == tt_fabric::FabricConfig::FABRIC_2D) {
        dm_kernel.compiler_options.defines.insert({"FABRIC_2D", "1"});
    }
    dm_kernel.global_semaphore_bindings = {GlobalSemaphoreBinding{
        .global_semaphore_parameter_name = GlobalSemaphoreParamName{"remote"}, .accessor_name = "remote"}};

    ProgramSpec spec;
    spec.name = "global_semaphore_fabric_smoke";
    spec.global_semaphore_parameters = {GlobalSemaphoreParameter{.unique_id = GlobalSemaphoreParamName{"remote"}}};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", NodeCoord{0, 0}, {"dm_kernel"})};
    spec.kernels = {std::move(dm_kernel)};

    auto& mesh_device = devices_.at(0);
    Program program = MakeProgramFromSpec(*mesh_device, spec);
    EXPECT_NO_THROW(program.impl().compile(mesh_device.get()));
}

INSTANTIATE_TEST_SUITE_P(
    MockWormhole,
    GlobalSemaphoreFabricJITTest,
    ::testing::Values(tt_fabric::FabricConfig::FABRIC_1D, tt_fabric::FabricConfig::FABRIC_2D),
    [](const ::testing::TestParamInfo<tt_fabric::FabricConfig>& info) {
        return info.param == tt_fabric::FabricConfig::FABRIC_1D ? std::string("Fabric1D") : std::string("Fabric2D");
    });

}  // namespace
}  // namespace tt::tt_metal::experimental
