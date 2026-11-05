// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <memory>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "distributed/mesh_device_impl.hpp"
#include "impl/buffers/buffer_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_accessor.hpp"
#include "impl/device/mock_device_util.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "multi_device_fixture.hpp"

namespace tt::tt_metal {

class Metal1ProgramConstructionTest : public GenericMeshDeviceFixture {
protected:
    ContextId fixture_context_id() { return get_mesh_device()->impl().get_context_id(); }
    IDevice* fixture_device() { return get_mesh_device()->impl().get_devices().at(0); }
};

namespace {

KernelDescriptor make_test_kernel_descriptor(KernelDescriptor::ConfigDescriptor config) {
    return KernelDescriptor{
        .kernel_source = "void kernel_main() {}",
        .source_type = KernelDescriptor::SourceType::SOURCE_CODE,
        .core_ranges = CoreRangeSet{CoreRange{{0, 0}}},
        .config = std::move(config)};
}

}  // namespace

TEST(Metal1ProgramConstruction, CPU_DescriptorConstructionPreservesContext) {
    // A program constructed from a descriptor belongs to the supplied context.
    MetalEnv env({.mock_cluster_desc_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1)});
    ContextId context_id{MetalEnvAccessor(env).impl().ensure_context_registered(env)};
    detail::ProgramImpl program(context_id, ProgramDescriptor{}, detail::Metal1Quirks{});
    EXPECT_EQ(program.get_context_id(), context_id);
}

TEST_F(Metal1ProgramConstructionTest, DescriptorConstructionLeavesProgramUncompiled) {
    // Constructing a program prepares its resources without compiling or finalizing it.
    ContextId context_id = fixture_context_id();
    ProgramDescriptor descriptor;
    descriptor.kernels = {make_test_kernel_descriptor(DataMovementConfigDescriptor{})};
    detail::ProgramImpl program(context_id, descriptor, detail::Metal1Quirks{});
    EXPECT_FALSE(program.is_compiled());
    EXPECT_FALSE(program.is_finalized());
}

TEST_F(Metal1ProgramConstructionTest, DescriptorOrderDefinesKernelHandles) {
    // Kernel handles follow descriptor order for data movement and compute kernels.
    ContextId context_id = fixture_context_id();
    ProgramDescriptor descriptor;
    descriptor.kernels = {
        make_test_kernel_descriptor(DataMovementConfigDescriptor{}),
        make_test_kernel_descriptor(ComputeConfigDescriptor{})};
    detail::ProgramImpl program(context_id, descriptor, detail::Metal1Quirks{});
    EXPECT_EQ(program.get_kernel(0)->get_kernel_processor_class(), HalProcessorClassType::DM);
    EXPECT_EQ(program.get_kernel(1)->get_kernel_processor_class(), HalProcessorClassType::COMPUTE);
}

TEST_F(Metal1ProgramConstructionTest, DescriptorConstructionPreservesCircularBufferConfig) {
    // Circular buffer indices, sizes, and page sizes come from the descriptor.
    ContextId context_id = fixture_context_id();
    const CoreRangeSet cores{CoreRange{{0, 0}}};
    const CBFormatDescriptor format{.buffer_index = 3, .data_format = DataFormat::Float16_b, .page_size = 2048};
    ProgramDescriptor descriptor{.cbs = {{.total_size = 4096, .core_ranges = cores, .format_descriptors = {format}}}};
    Program program(std::make_shared<detail::ProgramImpl>(context_id, descriptor, detail::Metal1Quirks{}));
    CBHandle cb_handle = program.circular_buffers().at(0)->id();
    const CircularBufferConfig& config = GetCircularBufferConfig(program, cb_handle);
    EXPECT_EQ(config.total_size(), 4096);
    EXPECT_EQ(config.page_sizes().at(3), 2048);
    EXPECT_EQ(config.data_formats().at(3), DataFormat::Float16_b);
}

TEST_F(Metal1ProgramConstructionTest, DescriptorConstructionPreservesSemaphoreId) {
    // A descriptor keeps its chosen semaphore ID and initial value.
    ContextId context_id = fixture_context_id();
    ProgramDescriptor descriptor;
    descriptor.semaphores = {{.id = 5, .core_ranges = CoreRangeSet{CoreRange{{0, 0}}}, .initial_value = 42}};
    detail::ProgramImpl program(context_id, descriptor, detail::Metal1Quirks{});
    EXPECT_EQ(program.semaphores().at(0).id(), 5);
    EXPECT_EQ(program.semaphores().at(0).initial_value(), 42);
}

TEST_F(Metal1ProgramConstructionTest, DescriptorConstructionPreservesRuntimeArgs) {
    // Descriptor runtime arguments remain accessible through the Metal 1.0 API.
    ContextId context_id = fixture_context_id();
    ProgramDescriptor descriptor;
    descriptor.kernels = {make_test_kernel_descriptor(DataMovementConfigDescriptor{})};
    descriptor.kernels[0].runtime_args = {{{0, 0}, {12, 34}}};
    Program program(std::make_shared<detail::ProgramImpl>(context_id, descriptor, detail::Metal1Quirks{}));
    RuntimeArgsData& args = GetRuntimeArgs(program, 0, {0, 0});
    EXPECT_EQ(args.at(0), 12);
    EXPECT_EQ(args.at(1), 34);
}

TEST_F(Metal1ProgramConstructionTest, DescriptorConstructionPreservesCommonRuntimeArgs) {
    // Descriptor common arguments remain accessible through the Metal 1.0 API.
    ContextId context_id = fixture_context_id();
    ProgramDescriptor descriptor;
    descriptor.kernels = {make_test_kernel_descriptor(DataMovementConfigDescriptor{})};
    descriptor.kernels[0].common_runtime_args = {56, 78};
    Program program(std::make_shared<detail::ProgramImpl>(context_id, descriptor, detail::Metal1Quirks{}));
    RuntimeArgsData& args = GetCommonRuntimeArgs(program, 0);
    EXPECT_EQ(args.at(0), 56);
    EXPECT_EQ(args.at(1), 78);
}

TEST_F(Metal1ProgramConstructionTest, QuirksRetainBuffersAfterOwnershipTransfer) {
    // A program keeps a transferred buffer alive after the caller releases its reference.
    ContextId context_id = fixture_context_id();
    std::shared_ptr<Buffer> buffer = BufferImpl::create(fixture_device(), 2048, 2048, BufferType::DRAM);
    std::weak_ptr<Buffer> buffer_lifetime = buffer;
    detail::ProgramImpl program(
        context_id, ProgramDescriptor{}, detail::Metal1Quirks{.retained_buffers = {std::move(buffer)}});
    EXPECT_FALSE(buffer_lifetime.expired());
}

TEST_F(Metal1ProgramConstructionTest, ProgramDestructionReleasesRetainedBuffers) {
    // Destroying the program releases its final reference to a transferred buffer.
    ContextId context_id = fixture_context_id();
    std::shared_ptr<Buffer> buffer = BufferImpl::create(fixture_device(), 2048, 2048, BufferType::DRAM);
    std::weak_ptr<Buffer> buffer_lifetime = buffer;
    {
        detail::ProgramImpl program(
            context_id, ProgramDescriptor{}, detail::Metal1Quirks{.retained_buffers = {std::move(buffer)}});
    }
    EXPECT_TRUE(buffer_lifetime.expired());
}

}  // namespace tt::tt_metal
