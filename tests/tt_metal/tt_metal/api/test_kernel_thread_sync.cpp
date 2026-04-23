// SPDX-FileCopyrightText: � 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "impl/context/metal_context.hpp"
#include "device_fixture.hpp"
#include "metal2_host_api/test_helpers.hpp"

namespace tt::tt_metal::experimental::metal2_host_api {
namespace {

// using test_helpers::MakeMinimalDMKernel;
// // using test_helpers::MakeMinimalGen1DMKernel;
// using test_helpers::MakeMinimalWorker;

// constexpr CoreCoord kCore{0, 0};
// constexpr const char* kKernelPath = "tests/tt_metal/tt_metal/test_kernels/dataflow/kernel_thread_barrier.cpp";
// constexpr uint32_t kRounds = 8;
// constexpr uint32_t kSkewIters = 64;
// constexpr uint32_t kKernelArgsCount = 6;

// struct ScratchLayout {
//     uint32_t arrivals_addr = 0;
//     uint32_t observed_addr = 0;
//     uint32_t post_addr = 0;
//     uint32_t total_words = 0;
// };

// ScratchLayout make_layout(uint32_t base_addr, uint32_t rounds) {
//     ScratchLayout layout{};
//     layout.arrivals_addr = base_addr;
//     layout.observed_addr = layout.arrivals_addr + rounds * sizeof(uint32_t);
//     layout.post_addr = layout.observed_addr + (rounds + 1) * sizeof(uint32_t);
//     layout.total_words = rounds + (rounds + 1) + rounds;
//     return layout;
// }

// ProgramRunParams::KernelRunParams make_run_params(
//     const KernelSpecName& kernel_name, const NodeCoord& node, const ScratchLayout& layout, uint32_t rounds, uint32_t skew_iters) {
//     return ProgramRunParams::KernelRunParams{
//         .kernel_spec_name = kernel_name,
//         .runtime_args =
//             {{node,
//               {
//                   layout.arrivals_addr,
//                   layout.observed_addr,
//                   layout.post_addr,
//                   rounds,
//                   skew_iters,
//                   layout.total_words,
//               }}},
//     };
// }

// class KernelThreadSyncTest : public tt::tt_metal::MeshDeviceFixture {};

// TEST_F(KernelThreadSyncTest, QuasarMultiDMBarrierSynchronizesThreads) {
//     if (this->arch_ != tt::ARCH::QUASAR) {
//         GTEST_SKIP() << "Quasar-only multithreaded DM barrier test";
//     }

//     auto mesh_device = devices_.at(0);
//     IDevice* device = mesh_device->get_devices()[0];
//     NodeCoord node{0, 0};

//     constexpr uint32_t num_dm_threads = 6;

//     ProgramSpec spec;
//     spec.program_id = "kernel_thread_barrier_quasar";

//     auto dm_kernel = MakeMinimalDMKernel("dm_barrier_kernel", node, static_cast<uint8_t>(num_dm_threads));
//     dm_kernel.source = KernelSpec::SourceFilePath{kKernelPath};
//     dm_kernel.runtime_arguments_schema.num_runtime_args_per_node = {{node, kKernelArgsCount}};
//     spec.kernels = {dm_kernel};
//     spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker_0", node, {"dm_barrier_kernel"})};

//     Program program = MakeProgramFromSpec(spec);

//     uint32_t l1_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
//     ScratchLayout layout = make_layout(l1_base, kRounds);
//     std::vector<uint32_t> zeros(layout.total_words, 0);
//     detail::WriteToDeviceL1(device, kCore, l1_base, zeros);

//     ProgramRunParams params;
//     params.kernel_run_params = {make_run_params("dm_barrier_kernel", node, layout, kRounds, kSkewIters)};
//     SetProgramRunParameters(program, params);
//     detail::LaunchProgram(device, program);

//     std::vector<uint32_t> arrivals;
//     std::vector<uint32_t> observed;
//     std::vector<uint32_t> post;
//     detail::ReadFromDeviceL1(device, kCore, layout.arrivals_addr, kRounds * sizeof(uint32_t), arrivals);
//     detail::ReadFromDeviceL1(device, kCore, layout.observed_addr, (kRounds + 1) * sizeof(uint32_t), observed);
//     detail::ReadFromDeviceL1(device, kCore, layout.post_addr, kRounds * sizeof(uint32_t), post);

//     ASSERT_EQ(arrivals.size(), kRounds);
//     ASSERT_EQ(observed.size(), kRounds + 1);
//     ASSERT_EQ(post.size(), kRounds);

//     for (uint32_t r = 0; r < kRounds; r++) {
//         EXPECT_EQ(arrivals[r], num_dm_threads) << "All DM threads must arrive before round barrier release";
//         EXPECT_EQ(observed[r], num_dm_threads) << "Thread 0 should observe full arrival count after barrier";
//         EXPECT_EQ(post[r], num_dm_threads) << "All DM threads must complete post-barrier phase";
//     }
//     EXPECT_EQ(observed[kRounds], num_dm_threads) << "Kernel should report total thread count";
// }

// TEST_F(KernelThreadSyncTest, Gen1BriscNcriscBarrierApiSmokeRuns) {
//     if (this->arch_ != tt::ARCH::WORMHOLE_B0 && this->arch_ != tt::ARCH::BLACKHOLE) {
//         GTEST_SKIP() << "WH/BH-only BRISC/NCRISC smoke test";
//     }

//     auto mesh_device = devices_.at(0);
//     IDevice* device = mesh_device->get_devices()[0];
//     NodeCoord node{0, 0};

//     ProgramSpec spec;
//     spec.program_id = "kernel_thread_barrier_gen1_smoke";

//     auto brisc_kernel = MakeMinimalGen1DMKernel("brisc_barrier_kernel", node, tt::tt_metal::DataMovementProcessor::RISCV_0);
//     brisc_kernel.source = KernelSpec::SourceFilePath{kKernelPath};
//     brisc_kernel.runtime_arguments_schema.num_runtime_args_per_node = {{node, kKernelArgsCount}};

//     auto ncrisc_kernel = MakeMinimalGen1DMKernel("ncrisc_barrier_kernel", node, tt::tt_metal::DataMovementProcessor::RISCV_1);
//     ncrisc_kernel.source = KernelSpec::SourceFilePath{kKernelPath};
//     ncrisc_kernel.runtime_arguments_schema.num_runtime_args_per_node = {{node, kKernelArgsCount}};

//     spec.kernels = {brisc_kernel, ncrisc_kernel};
//     spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker_0", node, {"brisc_barrier_kernel", "ncrisc_barrier_kernel"})};

//     Program program = MakeProgramFromSpec(spec);

//     uint32_t l1_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
//     ScratchLayout brisc_layout = make_layout(l1_base, kRounds);
//     ScratchLayout ncrisc_layout = make_layout(l1_base + brisc_layout.total_words * sizeof(uint32_t), kRounds);
//     std::vector<uint32_t> zeros(brisc_layout.total_words + ncrisc_layout.total_words, 0);
//     detail::WriteToDeviceL1(device, kCore, l1_base, zeros);

//     ProgramRunParams params;
//     params.kernel_run_params = {
//         make_run_params("brisc_barrier_kernel", node, brisc_layout, kRounds, kSkewIters),
//         make_run_params("ncrisc_barrier_kernel", node, ncrisc_layout, kRounds, kSkewIters),
//     };
//     SetProgramRunParameters(program, params);
//     detail::LaunchProgram(device, program);

//     std::vector<uint32_t> brisc_observed;
//     std::vector<uint32_t> ncrisc_observed;
//     detail::ReadFromDeviceL1(device, kCore, brisc_layout.observed_addr, (kRounds + 1) * sizeof(uint32_t), brisc_observed);
//     detail::ReadFromDeviceL1(device, kCore, ncrisc_layout.observed_addr, (kRounds + 1) * sizeof(uint32_t), ncrisc_observed);

//     ASSERT_EQ(brisc_observed.size(), kRounds + 1);
//     ASSERT_EQ(ncrisc_observed.size(), kRounds + 1);
//     EXPECT_EQ(brisc_observed[kRounds], 1u);
//     EXPECT_EQ(ncrisc_observed[kRounds], 1u);
// }

}  // namespace
}  // namespace tt::tt_metal::experimental::metal2_host_api
