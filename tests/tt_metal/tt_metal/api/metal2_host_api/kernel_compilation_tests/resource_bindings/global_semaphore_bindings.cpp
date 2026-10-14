// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// JIT-compile smoke tests for GlobalSemaphore bindings (mock Wormhole): the generated sem:: token,
// the device GlobalSemaphore class, and every fabric overload on both 1D and 2D fabric builds.
// Compile-only; the calls sit behind a branch that never runs.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <map>
#include <string>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/global_semaphore_test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindGlobalSemaphoreToKernel;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestGen1;

void CompileGlobalSemaphoreKernel(
    distributed::MeshDevice& mesh_device, const char* source, std::map<std::string, std::string> defines = {}) {
    ProgramSpec spec;
    spec.name = "global_sem_smoke";
    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.source = KernelSpec::SourceCode{source};
    dm_kernel.compiler_options.defines = KernelSpec::CompilerOptions::Defines(std::move(defines));
    BindGlobalSemaphoreToKernel(dm_kernel, "gsem", "gsem");
    spec.kernels = {dm_kernel};
    spec.global_semaphore_parameters = {GlobalSemaphoreParameter{.unique_id = GlobalSemaphoreParamName{"gsem"}}};
    spec.work_units = {MakeMinimalWorkUnit("work_unit", NodeCoord{0, 0}, {"dm_kernel"})};

    Program program = MakeProgramFromSpec(mesh_device, spec);
    EXPECT_NO_THROW(program.impl().compile(&mesh_device));
}

TEST_F(ProgramSpecTestGen1, CPU_GlobalSemaphoreOnChipApiJITSmoke) {
    CompileGlobalSemaphoreKernel(*mesh_device_, R"(
#include "api/dataflow/global_semaphore.h"

static_assert(sem::gsem_t::slot_id == 0);

void kernel_main() {
    GlobalSemaphore gs(sem::gsem);
    volatile uint32_t run = 0;
    if (run) {
        Noc noc;
        gs.set(0);
        gs.up();
        gs.up(2);
        gs.wait(3);
        gs.wait_min(1);
        gs.down(1);
        volatile uint32_t v = gs.value();
        (void)v;
        gs.up(noc, 1, 2);
        gs.up<true>(noc, 1, 2, 3);
        gs.set_multicast(noc, 1, 1, 2, 2, 4);
        gs.set_multicast<NocOptions::MCAST_INCL_SRC>(noc, 1, 1, 2, 2, 4, true);
        gs.inc_multicast(noc, 1, 1, 2, 2, 1, 4);
    }
}
)");
}

constexpr const char* kFabricSmokeSource = R"(
#include "api/dataflow/global_semaphore_fabric.h"

void kernel_main() {
    GlobalSemaphore gs(sem::gsem);
    volatile uint32_t run = 0;
    if (run) {
        auto& edm = *reinterpret_cast<tt::tt_fabric::WorkerToFabricEdmSender*>(run);
        auto& mux = *reinterpret_cast<tt::tt_fabric::WorkerToFabricMuxSender<8>*>(run);
        auto& mgr = *reinterpret_cast<tt::tt_fabric::RoutingPlaneConnectionManager*>(run);
        auto* hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(run);
#if defined(FABRIC_2D)
        const ToChip one{1, 0};
        const ToChips many{1, 0, 1, 1, 0, 0};
        gs.up(mgr, 0, 1, 2);
#else
        const ToChip one{1};
        const ToChips many{1, 2};
        uint8_t hops[1] = {1};
        gs.up(mgr, 0, hops, 1, 2);
#endif
        gs.up(edm, hdr, one, 1, 2);
        gs.up(edm, hdr, many, 1, 2, 2, false, 1);
        gs.up(mux, hdr, one, 1, 2);
        gs.write_and_up(edm, hdr, one, 0x1000, 0x2000, 64, 1, 2);
        gs.write_and_up(edm, hdr, many, 0x1000, 0x2000, 64, 1, 2);
        gs.up_set_state(hdr, one, 1, 2);
        gs.up_set_state(hdr, many, 1, 2, 4);
        gs.up_with_state(edm, hdr);
        gs.up_with_state(edm, hdr, 3, 4);
    }
}
)";

// ROUTING_MODE values match FabricContext::get_fabric_kernel_defines (fabric_routing_mode.h bits).
TEST_F(ProgramSpecTestGen1, CPU_GlobalSemaphoreFabric1DApiJITSmoke) {
    // ROUTING_MODE_1D | ROUTING_MODE_LINE | ROUTING_MODE_LOW_LATENCY
    CompileGlobalSemaphoreKernel(*mesh_device_, kFabricSmokeSource, {{"ROUTING_MODE", "137"}});
}

TEST_F(ProgramSpecTestGen1, CPU_GlobalSemaphoreFabric2DApiJITSmoke) {
    // ROUTING_MODE_2D | ROUTING_MODE_MESH | ROUTING_MODE_LOW_LATENCY
    CompileGlobalSemaphoreKernel(*mesh_device_, kFabricSmokeSource, {{"FABRIC_2D", "1"}, {"ROUTING_MODE", "162"}});
}

}  // namespace
}  // namespace tt::tt_metal::experimental
