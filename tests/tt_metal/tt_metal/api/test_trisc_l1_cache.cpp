// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar TRISC L1 read cache (#61944): relaunch one compute kernel many times with a runtime argument that
// changes every launch. Each TRISC echoes the argument it read into an L1 slot. With the TRISC L1 read cache on,
// a TRISC can read the previous launch's argument from a stale cached line; with it off (the default) it can't.

#include <cstdint>
#include <cstdlib>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "device_fixture.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include <tt-umd/driver_atomics.hpp>

namespace tt::tt_metal {

namespace m2 = experimental;

TEST_F(QuasarMeshDeviceSingleCardFixture, TriscL1CacheRuntimeArgNotStale) {
    // One Tensix engine runs four compute processors: unpack, math, pack, and isolate-SFPU.
    constexpr uint32_t num_trisc_threads = 4;
    // Launch count; override with TRISC_L1_CACHE_NUM_LAUNCHES. Keep it small enough for the emulator session budget.
    // A stale hit depends on timing, so a pass over this many launches is evidence, not proof the cache is off.
    uint32_t num_launches = 100;
    if (const char* env = std::getenv("TRISC_L1_CACHE_NUM_LAUNCHES")) {
        num_launches = static_cast<uint32_t>(std::strtoul(env, nullptr, 0));
    }

    const m2::NodeCoord node{0, 0};
    const m2::KernelSpecName KERNEL{"trisc_l1_rta_echo"};
    m2::KernelSpec kernel{
        .unique_id = KERNEL,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/trisc_l1_rta_echo.cpp",
        .num_threads = 1,
        .runtime_arg_schema = {.runtime_arg_names = {"result_l1_addr", "value"}},
        .hw_config = m2::ComputeHardwareConfig{},
    };
    m2::WorkUnitSpec wu{.name = "wu", .kernels = {KERNEL}, .target_nodes = node};
    m2::ProgramSpec spec{.name = "trisc_l1_rta_echo", .kernels = {kernel}, .work_units = {wu}};

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    workload.add_program(
        distributed::MeshCoordinateRange(zero_coord, zero_coord), m2::MakeProgramFromSpec(this->device(), spec));
    Program& program = workload.get_programs().begin()->second;

    // Mailboxes and the firmware mem map sit at the low end of L1. This test allocates no L1 buffers, so the
    // top of worker L1 stays unused. Four echo words fit in this gap.
    constexpr uint32_t result_region_bytes = 64;
    const uint32_t result_l1_addr = static_cast<uint32_t>(this->device().l1_size_per_core()) - result_region_bytes;

    uint32_t num_stale = 0;
    for (uint32_t launch = 0; launch < num_launches; ++launch) {
        const uint32_t value = 0xA0000000u + launch;
        m2::ProgramRunArgs params;
        params.kernel_run_args = {m2::ProgramRunArgs::KernelRunArgs{
            .kernel = KERNEL,
            .runtime_arg_values =
                m2::MakeRuntimeArgsForSingleNode(node, {{"result_l1_addr", result_l1_addr}, {"value", value}}),
        }};
        if (launch == 0) {
            m2::SetProgramRunArgs(program, params);
        } else {
            m2::UpdateProgramRunArgs(program, params);
        }

        distributed::EnqueueMeshWorkload(this->device().mesh_command_queue(), workload, /*blocking=*/true);

        tt_driver_atomics::mfence();
        std::vector<uint32_t> results;
        slow_dispatch::ReadFromL1(
            this->device(), CoreCoord(0, 0), result_l1_addr, num_trisc_threads * sizeof(uint32_t), results);
        ASSERT_EQ(results.size(), num_trisc_threads);
        for (uint32_t thread = 0; thread < num_trisc_threads; ++thread) {
            if (results[thread] != value) {
                ++num_stale;
                ADD_FAILURE() << "launch " << launch << " TRISC " << thread << " echoed 0x" << std::hex
                              << results[thread] << ", expected 0x" << value;
            }
        }
    }
    EXPECT_EQ(num_stale, 0u);
}

}  // namespace tt::tt_metal
