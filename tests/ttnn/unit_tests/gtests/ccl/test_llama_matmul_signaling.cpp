// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <vector>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include "tt_metal/tt_metal/common/device_fixture.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

namespace tt::tt_metal {

class LlamaMatmulSignaling : public AnyDispatchMeshDeviceSingleCardFixture, public testing::WithParamInterface<NOC> {};

// The canaries share the receivers' semaphore offset on cores that do not participate
// in reduce-scatter. Neither dispatch-time initialization nor device signaling may
// change them. In the split grid, one canary lies inside the receivers' bounding box.
TEST_P(LlamaMatmulSignaling, PreservesNonparticipatingCores) {
    auto& mesh_device = devices_.front();
    const auto grid = mesh_device->compute_with_storage_grid_size();
    if (grid.x < 4 || grid.y < 3) {
        GTEST_SKIP() << "Requires a 4 x 3 worker grid";
    }
    auto* device = mesh_device->get_devices().front();
    const CoreRangeSet observed(CoreRange({1, 0}, {3, 2}));
    const CoreRangeSet matmul_cores(CoreRange({2, 1}, {2, 2}));
    const std::vector<CoreRangeSet> receiver_grids = {
        // Run the regression case first: the old host code fails the canary-allocation
        // check before launching a kernel with its obsolete runtime-argument format.
        CoreRangeSet(std::vector<CoreRange>{CoreRange({1, 0}, {1, 2}), CoreRange({3, 0}, {3, 2})}),
        CoreRangeSet(CoreRange({1, 0}, {1, 0})),
        CoreRangeSet(CoreRange({1, 0}, {1, 2}))};
    constexpr uint32_t canary = 0x5a5a5a5a;
    const auto noc = GetParam();

    for (const auto& receivers : receiver_grids) {
        SCOPED_TRACE(receivers.str());
        Program program = CreateProgram();
        ttnn::experimental::ccl::MatmulFusedOpSignaler signaler(
            ttnn::experimental::ccl::MatmulFusedOpSignalerType::LLAMA_REDUCE_SCATTER);
        signaler.init_llama_rs_cores_rs(receivers, program);
        const auto canary_cores = observed.subtract(receivers).subtract(matmul_cores);
        const auto canary_id = CreateSemaphore(program, canary_cores, canary);
        // An out-of-grid semaphore allocation already violates isolation, before launch.
        ASSERT_EQ(canary_id, signaler.rs_semaphore);
        signaler.init_llama_rs_cores_mm(matmul_cores, program, device);

        const auto sender = CreateKernelFromString(
            program,
            R"(
                #include "ttnn/operations/ccl/kernel_common/llama_matmul_signaling.hpp"
                void kernel_main() {
                    uint32_t arg = 0;
                    const Noc noc;
                    llama_matmul_signal_reduce_scatter(noc, arg);
                }
            )",
            matmul_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = noc});
        for (const auto& core : corerange_to_cores(matmul_cores)) {
            std::vector<uint32_t> args;
            signaler.push_llama_rs_rt_args_for_mm(args, core, noc, device);
            SetRuntimeArgs(program, sender, core, args);
        }
        const auto receiver = CreateKernelFromString(
            program,
            R"(
                #include "api/dataflow/noc_semaphore.h"
                void kernel_main() {
                    Semaphore<> ready(get_arg_val<uint32_t>(0));
                    ready.wait(1);
                }
            )",
            receivers,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = noc});
        SetRuntimeArgs(program, receiver, receivers, {signaler.rs_semaphore});
        CreateKernelFromString(
            program,
            "void kernel_main() {}",
            canary_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = noc});

        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
        for (uint32_t iteration = 0; iteration < 40; ++iteration) {
            SCOPED_TRACE(iteration);
            RunProgram(mesh_device, workload);
            for (const auto& core : corerange_to_cores(receivers.merge(canary_cores))) {
                const auto address =
                    workload.get_sem_base_addr(mesh_device, core, CoreType::WORKER) +
                    signaler.rs_semaphore * MetalContext::instance().hal().get_alignment(HalMemType::L1);
                std::vector<uint32_t> actual;
                slow_dispatch::ReadFromL1(*mesh_device, core, address, sizeof(uint32_t), actual);
                ASSERT_EQ(actual.size(), 1);
                EXPECT_EQ(actual.front(), receivers.contains(core) ? 1u : canary) << core.str();
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(BothNocs, LlamaMatmulSignaling, testing::Values(NOC::NOC_0, NOC::NOC_1));

}  // namespace tt::tt_metal
