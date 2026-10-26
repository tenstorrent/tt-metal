// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "device_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/program/dispatch.hpp"
#include "impl/program/program_impl.hpp"

namespace tt::tt_metal::shared_compute_runtime_test {

const CoreCoord core0{0, 0};
const CoreCoord core1{1, 0};

// Returns UNPACK, MATH and PACK kernels on the same cores. Runtime arguments go on UNPACK.
ProgramDescriptor split_per_trisc() {
    ProgramDescriptor descriptor;
    for (auto processor : {ComputeProcessor::UNPACK, ComputeProcessor::MATH, ComputeProcessor::PACK}) {
        descriptor.kernels.push_back({
            .kernel_source = "void kernel_main() {}",
            .source_type = KernelDescriptor::SourceType::SOURCE_CODE,
            .core_ranges = CoreRangeSet(CoreRange(core0, core1)),
            .config = ComputeConfigDescriptor{.processor = processor},
        });
    }
    return descriptor;
}

uint32_t tensix_index() {
    return MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
}

void finalize_rt_args(Program& program) {
    uint32_t rta_offset = 0;
    program_dispatch::finalize_rt_args(
        MetalContext::instance(),
        program.impl().get_kernels(tensix_index()),
        program.impl().get_kernel_groups(tensix_index()),
        32,
        tensix_index(),
        rta_offset);
}

class SharedComputeRuntimeArgs : public ::testing::Test {
protected:
    void SetUp() override { experimental::configure_mock_mode(tt::ARCH::BLACKHOLE, 1); }
    void TearDown() override { experimental::disable_mock_mode(); }
};

TEST_F(SharedComputeRuntimeArgs, CPU_SplitKernelLaysOutRuntimeArgsLikeUnsplitKernel) {
    auto split_descriptor = split_per_trisc();
    auto& unpack = split_descriptor.kernels[0];
    unpack.runtime_args = {{core0, std::vector<uint32_t>(8, 7)}, {core1, std::vector<uint32_t>(4, 8)}};
    unpack.common_runtime_args = {9};
    auto unsplit = unpack;
    std::get<ComputeConfigDescriptor>(unsplit.config).processor.reset();
    // UNPACK shares a kernel group with this reader on core1 only, so its offsets differ per group.
    KernelDescriptor reader{
        .kernel_source = "void kernel_main() {}",
        .source_type = KernelDescriptor::SourceType::SOURCE_CODE,
        .core_ranges = CoreRangeSet(CoreRange(core1)),
        .runtime_args = {{core1, {1, 2, 3, 4, 5}}},
        .common_runtime_args = {6},
        .config = ReaderConfigDescriptor{},
    };
    split_descriptor.kernels.push_back(reader);
    Program split_program(split_descriptor);
    Program unsplit_program(ProgramDescriptor{.kernels = {unsplit, reader}});

    finalize_rt_args(split_program);
    finalize_rt_args(unsplit_program);
    for (const auto& group : split_program.impl().get_kernel_groups(tensix_index())) {
        auto* baseline =
            unsplit_program.impl().kernels_on_core(group->core_ranges.ranges().begin()->start_coord, tensix_index());
        auto offsets = group->launch_msg.view().kernel_config().rta_offset();
        auto baseline_offsets = baseline->launch_msg.view().kernel_config().rta_offset();
        for (size_t processor = 0; processor < offsets.size(); ++processor) {
            EXPECT_EQ(offsets[processor].rta_offset(), baseline_offsets[processor].rta_offset());
            EXPECT_EQ(offsets[processor].crta_offset(), baseline_offsets[processor].crta_offset());
        }
    }
}

TEST_F(SharedComputeRuntimeArgs, CPU_MathAndPackRejectOwnRuntimeArgs) {
    auto with_unique = split_per_trisc();
    with_unique.kernels[1].runtime_args = {{core0, {1}}};
    auto with_common = split_per_trisc();
    with_common.kernels[2].common_runtime_args = {1};
    for (const auto& descriptor : {with_unique, with_common}) {
        EXPECT_THAT(
            [&] { Program program(descriptor); },
            ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("must not have its own")));
    }
}

using SharedComputeRuntimeArgsDevice = UnitMeshAnyDispatchFixture;

TEST_F(SharedComputeRuntimeArgsDevice, TensixAllTriscsReadUnpackRuntimeArgsAcrossUpdates) {
    if (arch_ == tt::ARCH::QUASAR ||
        MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Emule) {
        GTEST_SKIP() << "Physical TRISC selection requires Wormhole or Blackhole silicon";
    }
    auto mesh_device = devices_.front();
    auto* device = mesh_device->get_devices()[0];
    const auto output_address = mesh_device->allocator()->get_base_allocator_addr(HalMemType::L1);
    auto descriptor = split_per_trisc();
    auto set_unpack_args = [&](uint32_t increment) {
        auto& named = descriptor.kernels[0].blaze_named_args;
        named.named_per_core_runtime_args = {{"probe.unique", {{core0, 10 + increment}, {core1, 20 + increment}}}};
        named.named_common_runtime_args = {{"probe.common", 30 + increment}};
    };
    set_unpack_args(0);
    for (auto& kernel : descriptor.kernels) {
        kernel.compile_time_args = {output_address};
        kernel.kernel_source = R"(
#include "api/compute/common.h"
#include "experimental/blaze_named_args.h"

void kernel_main() {
#if defined(TRISC_UNPACK)
    constexpr uint32_t processor = 0;
#elif defined(TRISC_MATH)
    constexpr uint32_t processor = 1;
#elif defined(TRISC_PACK)
    constexpr uint32_t processor = 2;
#endif
    auto* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_compile_time_arg_val(0));
    out[processor * 2 + 0] = blaze_rt_args::get<blaze_ct_args::probe::unique>();
    out[processor * 2 + 1] = blaze_rt_args::get<blaze_ct_args::probe::common>();
}
)";
    }
    // A data-movement kernel's runtime arguments come first, so UNPACK's offset is not 0.
    descriptor.kernels.push_back({
        .kernel_source = "void kernel_main() {}",
        .source_type = KernelDescriptor::SourceType::SOURCE_CODE,
        .core_ranges = CoreRangeSet(CoreRange(core0, core1)),
        .runtime_args = {{core0, {1, 2, 3}}, {core1, {4, 5, 6}}},
        .config = ReaderConfigDescriptor{},
    });
    const auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    distributed::MeshWorkload workload;
    workload.add_program(device_range, Program(descriptor));
    auto& program = workload.get_programs().at(device_range);

    auto run_and_check = [&](uint32_t increment) {
        SCOPED_TRACE(increment);
        std::vector<uint32_t> cleared(6, 0);
        for (auto core : {core0, core1}) {
            ASSERT_TRUE(detail::WriteToDeviceL1(device, core, output_address, cleared));
        }
        RunProgram(mesh_device, workload);
        for (auto core : {core0, core1}) {
            std::vector<uint32_t> output;
            ASSERT_TRUE(detail::ReadFromDeviceL1(device, core, output_address, 6 * sizeof(uint32_t), output));
            const uint32_t unique = 10 + 10 * core.x + increment;
            const uint32_t common = 30 + increment;
            EXPECT_EQ(output, std::vector<uint32_t>({unique, common, unique, common, unique, common}));
        }
    };
    ASSERT_NO_FATAL_FAILURE(run_and_check(0));

    // Update the same Program after it has run, when fast dispatch has moved its runtime args into commands.
    set_unpack_args(100);
    apply_descriptor_runtime_args(program, descriptor);
    ASSERT_NO_FATAL_FAILURE(run_and_check(100));
}

}  // namespace tt::tt_metal::shared_compute_runtime_test
