// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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

using test_helpers::MakeMinimalDMKernel;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalWorkUnit;

constexpr CoreCoord kCore{0, 0};
constexpr const char* kKernelPath = "tests/tt_metal/tt_metal/test_kernels/dataflow/kernel_thread_barrier.cpp";
constexpr uint32_t kRounds = 8;
constexpr uint32_t kSkewIters = 64;
constexpr uint32_t kKernelArgsCount = 6;

struct ScratchLayout {
    uint32_t arrivals_addr = 0;
    uint32_t observed_addr = 0;
    uint32_t post_addr = 0;
    uint32_t total_words = 0;
};

ScratchLayout make_layout(uint32_t base_addr, uint32_t rounds) {
    ScratchLayout layout{};
    layout.arrivals_addr = base_addr;
    layout.observed_addr = layout.arrivals_addr + rounds * sizeof(uint32_t);
    layout.post_addr = layout.observed_addr + (rounds + 1) * sizeof(uint32_t);
    layout.total_words = rounds + (rounds + 1) + rounds;
    return layout;
}

ProgramRunParams::KernelRunParams make_run_params(
    const KernelSpecName& kernel_name, const NodeCoord& node, const ScratchLayout& layout, uint32_t rounds, uint32_t skew_iters) {
    return ProgramRunParams::KernelRunParams{
        .kernel_spec_name = kernel_name,
        .runtime_varargs =
            {{node,
              {
                  layout.arrivals_addr,
                  layout.observed_addr,
                  layout.post_addr,
                  rounds,
                  skew_iters,
                  layout.total_words,
              }}},
    };
}

class KernelThreadSyncTest : public tt::tt_metal::MeshDeviceFixture {};

TEST_F(KernelThreadSyncTest, BarrierSynchronizesThreads) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];
    NodeCoord node{0, 0};

    // Arch-specific config: kernels to launch, one scratch layout per kernel,
    // expected thread count per kernel, and whether to verify full barrier semantics.
    struct KernelConfig {
        std::string name;
        KernelSpec spec;
        ScratchLayout layout;
    };

    const bool is_quasar = (this->arch_ == tt::ARCH::QUASAR);
    const uint32_t expected_num_threads = is_quasar ? 6u : 1u;

    uint32_t l1_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);

    std::vector<KernelConfig> kernel_configs;
    std::vector<std::string> work_unit_kernel_names;

    if (is_quasar) {
        auto spec = MakeMinimalDMKernel("dm_barrier_kernel", static_cast<uint8_t>(expected_num_threads));
        spec.source = KernelSpec::SourceFilePath{kKernelPath};
        spec.runtime_arguments_schema.num_runtime_varargs_per_node = {{node, kKernelArgsCount}};
        kernel_configs.push_back({"dm_barrier_kernel", spec, make_layout(l1_base, kRounds)});
        work_unit_kernel_names = {"dm_barrier_kernel"};
    } else {
        auto make_gen1 = [&](const std::string& name, tt::tt_metal::DataMovementProcessor proc, uint32_t layout_base) {
            auto spec = MakeMinimalGen1DMKernel(name, proc);
            spec.source = KernelSpec::SourceFilePath{kKernelPath};
            spec.runtime_arguments_schema.num_runtime_varargs_per_node = {{node, kKernelArgsCount}};
            return KernelConfig{name, spec, make_layout(layout_base, kRounds)};
        };
        kernel_configs.push_back(make_gen1("brisc_barrier_kernel", tt::tt_metal::DataMovementProcessor::RISCV_0, l1_base));
        uint32_t ncrisc_base = l1_base + kernel_configs[0].layout.total_words * sizeof(uint32_t);
        kernel_configs.push_back(make_gen1("ncrisc_barrier_kernel", tt::tt_metal::DataMovementProcessor::RISCV_1, ncrisc_base));
        work_unit_kernel_names = {"brisc_barrier_kernel", "ncrisc_barrier_kernel"};
    }

    ProgramSpec spec;
    spec.program_id = "kernel_thread_barrier";
    for (const auto& cfg : kernel_configs) { spec.kernels.push_back(cfg.spec); }
    spec.work_units = {MakeMinimalWorkUnit("work_unit_0", node, work_unit_kernel_names)};

    Program program = MakeProgramFromSpec(spec);

    uint32_t total_zeros = 0;
    for (const auto& cfg : kernel_configs) { total_zeros += cfg.layout.total_words; }
    std::vector<uint32_t> zeros(total_zeros, 0);
    detail::WriteToDeviceL1(device, kCore, l1_base, zeros);

    ProgramRunParams params;
    for (const auto& cfg : kernel_configs) {
        params.kernel_run_params.push_back(make_run_params(cfg.name, node, cfg.layout, kRounds, kSkewIters));
    }
    SetProgramRunParameters(program, params);
    detail::LaunchProgram(device, program);

    for (const auto& cfg : kernel_configs) {
        std::vector<uint32_t> observed;
        detail::ReadFromDeviceL1(device, kCore, cfg.layout.observed_addr, (kRounds + 1) * sizeof(uint32_t), observed);
        ASSERT_EQ(observed.size(), kRounds + 1);
        EXPECT_EQ(observed[kRounds], expected_num_threads) << cfg.name << ": get_num_threads() mismatch";

        if (is_quasar) {
            std::vector<uint32_t> arrivals, post;
            detail::ReadFromDeviceL1(device, kCore, cfg.layout.arrivals_addr, kRounds * sizeof(uint32_t), arrivals);
            detail::ReadFromDeviceL1(device, kCore, cfg.layout.post_addr, kRounds * sizeof(uint32_t), post);
            ASSERT_EQ(arrivals.size(), kRounds);
            ASSERT_EQ(post.size(), kRounds);
            for (uint32_t r = 0; r < kRounds; r++) {
                EXPECT_EQ(arrivals[r], expected_num_threads) << cfg.name << " round " << r << ": not all threads arrived before barrier release";
                EXPECT_EQ(observed[r], expected_num_threads) << cfg.name << " round " << r << ": thread 0 observed wrong count after barrier";
                EXPECT_EQ(post[r], expected_num_threads) << cfg.name << " round " << r << ": not all threads completed post-barrier phase";
            }
        }
    }
}

}  // namespace
}  // namespace tt::tt_metal::experimental::metal2_host_api
