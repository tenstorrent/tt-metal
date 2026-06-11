// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>

#include <gtest/gtest.h>

#include "host_api.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::distributed::test {

using MeshEndToEnd2x4Tests = MeshDevice2x4Fixture;

TEST_F(MeshEndToEnd2x4Tests, ProgramDispatchTest) {
    auto& cq = mesh_device_->mesh_command_queue();

    uint8_t cq_id = cq.id();

    EXPECT_GE(cq_id, 0);

    auto example_program = CreateProgram();

    auto target_tensix_cores = CoreRange{
        CoreCoord{0, 0} /* start_coord */, CoreCoord{1, 1} /* end_coord */
    };

    auto compute_kernel_id = CreateKernel(
        example_program,
        "tt_metal/programming_examples/distributed/1_distributed_program_dispatch/kernels/void_kernel.cpp",
        target_tensix_cores,
        ComputeConfig{.compile_args = {}});

    auto runtime_args = std::vector<uint32_t>{};
    SetRuntimeArgs(example_program, compute_kernel_id, target_tensix_cores, runtime_args);

    auto rt_args_out = GetRuntimeArgs(example_program, compute_kernel_id);
    EXPECT_EQ(rt_args_out.size(), 2);

    auto mesh_workload = MeshWorkload();

    auto target_devices = MeshCoordinateRange(mesh_device_->shape());

    mesh_workload.add_program(target_devices, std::move(example_program));

    EnqueueMeshWorkload(cq, mesh_workload, false /* blocking */);

    EXPECT_EQ(mesh_workload.get_last_used_command_queue()->id(), cq_id);

    Finish(cq);
}

}  // namespace tt::tt_metal::distributed::test
