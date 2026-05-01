// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the descriptor-patching fast cache-hit path.
//
// Test categories:
//   1. Pure descriptor-builder unit tests (no device required)
//      - emplace_runtime_args / emplace_common_runtime_args with uint32_t-only args
//      - RTArgList builder
//      - ResolvedBindings::empty()
//   2. Device integration tests (require a Tenstorrent device)
//      - resolve_bindings: correct (kernel_idx, core, arg_idx, tensor_buffer_idx, is_common)
//      - apply_resolved_bindings: patches per-core and common runtime args
//      - Two buffers at distinct positions in the same kernel
//      - Error path: buffer not enumerated in tensor_buffers

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/runtime_args_data.hpp>
#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include <tt_stl/assert.hpp>

#include "device_fixture.hpp"
#include "multi_device_fixture.hpp"

namespace tt::tt_metal {
namespace {

// ============================================================================
// Helpers
// ============================================================================

// Build a minimal single-kernel ProgramDescriptor using the blank dataflow kernel.
// The kernel occupies core {0,0} and all runtime args are set via emplace_runtime_args.
KernelDescriptor MakeBlankReaderKernel(CoreCoord core = {0, 0}) {
    KernelDescriptor kd;
    kd.kernel_source = "tt_metal/kernels/dataflow/blank.cpp";
    kd.source_type = KernelDescriptor::SourceType::FILE_PATH;
    kd.core_ranges = CoreRangeSet{CoreRange{core}};
    kd.config = ReaderConfigDescriptor{};
    return kd;
}

std::shared_ptr<Buffer> MakeDramBuffer(IDevice* device, uint32_t size = 2048) {
    InterleavedBufferConfig cfg{.device = device, .size = size, .page_size = size, .buffer_type = BufferType::DRAM};
    return CreateBuffer(cfg);
}

std::shared_ptr<Buffer> MakeL1Buffer(IDevice* device, uint32_t size = 2048) {
    InterleavedBufferConfig cfg{.device = device, .size = size, .page_size = size, .buffer_type = BufferType::L1};
    return CreateBuffer(cfg);
}

// ============================================================================
// SECTION 1: Pure unit tests — no device required
// ============================================================================

TEST(DescriptorPatching, EmplaceRuntimeArgs_AllUint32_NoBufferBindings) {
    KernelDescriptor kd;
    kd.emplace_runtime_args({0, 0}, {10u, 20u, 30u});

    ASSERT_EQ(kd.runtime_args.size(), 1u);
    EXPECT_EQ(kd.runtime_args[0].first, CoreCoord(0, 0));
    EXPECT_EQ(kd.runtime_args[0].second, (std::vector<uint32_t>{10u, 20u, 30u}));
    EXPECT_TRUE(kd.buffer_bindings.empty());
}

TEST(DescriptorPatching, EmplaceRuntimeArgs_MultipleCores_AllUint32) {
    KernelDescriptor kd;
    kd.emplace_runtime_args({0, 0}, {1u, 2u});
    kd.emplace_runtime_args({1, 0}, {3u, 4u});

    ASSERT_EQ(kd.runtime_args.size(), 2u);
    EXPECT_EQ(kd.runtime_args[0].second, (std::vector<uint32_t>{1u, 2u}));
    EXPECT_EQ(kd.runtime_args[1].second, (std::vector<uint32_t>{3u, 4u}));
    EXPECT_TRUE(kd.buffer_bindings.empty());
}

TEST(DescriptorPatching, EmplaceCommonRuntimeArgs_AllUint32_NoBindings) {
    KernelDescriptor kd;
    kd.emplace_common_runtime_args({100u, 200u, 300u});

    EXPECT_EQ(kd.common_runtime_args, (std::vector<uint32_t>{100u, 200u, 300u}));
    EXPECT_TRUE(kd.common_buffer_bindings.empty());
}

TEST(DescriptorPatching, RTArgList_Uint32Only_NoBufferBindings) {
    KernelDescriptor kd;
    KernelDescriptor::RTArgList args;
    args.push_back(7u);
    args.push_back(8u);
    args.push_back(9u);
    kd.emplace_runtime_args({0, 0}, args);

    ASSERT_EQ(kd.runtime_args.size(), 1u);
    EXPECT_EQ(kd.runtime_args[0].second, (std::vector<uint32_t>{7u, 8u, 9u}));
    EXPECT_TRUE(kd.buffer_bindings.empty());
}

TEST(DescriptorPatching, RTArgList_Append_ConcatenatesUint32s) {
    KernelDescriptor kd;
    KernelDescriptor::RTArgList args;
    args.push_back(1u);
    args.append({2u, 3u, 4u});
    kd.emplace_runtime_args({0, 0}, args);

    EXPECT_EQ(kd.runtime_args[0].second, (std::vector<uint32_t>{1u, 2u, 3u, 4u}));
}

TEST(DescriptorPatching, ResolvedBindings_DefaultIsEmpty) {
    ResolvedBindings b;
    EXPECT_TRUE(b.empty());
}

TEST(DescriptorPatching, ResolvedBindings_EmptyAfterAddingRtArg_IsFalse) {
    ResolvedBindings b;
    b.rt_args.push_back({});
    EXPECT_FALSE(b.empty());
}

TEST(DescriptorPatching, ResolvedBindings_EmptyAfterAddingCb_IsFalse) {
    ResolvedBindings b;
    b.cbs.push_back({});
    EXPECT_FALSE(b.empty());
}

// ============================================================================
// SECTION 2: Device integration tests
// ============================================================================

class DescriptorPatchingDeviceTest : public GenericMeshDeviceFixture {
protected:
    IDevice* device() { return get_mesh_device()->get_devices()[0]; }
};

// resolve_bindings correctly maps a single per-core buffer arg.
// Layout: args = [buf_a, 42u] at core {0,0}.
// Expected: one ResolvedRtArgBinding with kernel_idx=0, core={0,0}, arg_idx=0,
//           tensor_buffer_idx=0, is_common=false.
TEST_F(DescriptorPatchingDeviceTest, Tensix_ResolveBindings_PerCoreBuffer_CorrectTuples) {
    auto buf_a = MakeDramBuffer(device());

    KernelDescriptor kd = MakeBlankReaderKernel({0, 0});
    kd.emplace_runtime_args({0, 0}, {buf_a.get(), 42u});

    ProgramDescriptor desc;
    desc.kernels = {kd};

    Program program{desc};
    ResolvedBindings resolved = resolve_bindings(program, desc, {buf_a.get()});

    ASSERT_EQ(resolved.rt_args.size(), 1u);
    EXPECT_TRUE(resolved.cbs.empty());
    EXPECT_FALSE(resolved.empty());

    const auto& b = resolved.rt_args[0];
    EXPECT_EQ(b.kernel_idx, 0u);
    EXPECT_EQ(b.core, CoreCoord(0, 0));
    EXPECT_EQ(b.arg_idx, 0u);
    EXPECT_EQ(b.tensor_buffer_idx, 0u);
    EXPECT_FALSE(b.is_common);
}

// resolve_bindings with a buffer at arg index 1 (non-zero position).
// Layout: args = [99u, buf_a] at core {0,0}. arg_idx should be 1.
TEST_F(DescriptorPatchingDeviceTest, Tensix_ResolveBindings_BufferAtNonZeroArgIdx) {
    auto buf_a = MakeDramBuffer(device());

    KernelDescriptor kd = MakeBlankReaderKernel({0, 0});
    kd.emplace_runtime_args({0, 0}, {99u, buf_a.get()});

    ProgramDescriptor desc;
    desc.kernels = {kd};

    Program program{desc};
    ResolvedBindings resolved = resolve_bindings(program, desc, {buf_a.get()});

    ASSERT_EQ(resolved.rt_args.size(), 1u);
    EXPECT_EQ(resolved.rt_args[0].arg_idx, 1u);
    EXPECT_EQ(resolved.rt_args[0].tensor_buffer_idx, 0u);
}

// resolve_bindings with two buffer args in the same kernel/core.
// Layout: args = [buf_a, buf_b] at core {0,0}.
// Expected: two entries with arg_idx 0 and 1, tensor_buffer_idx 0 and 1.
TEST_F(DescriptorPatchingDeviceTest, Tensix_ResolveBindings_TwoBuffers_CorrectIndices) {
    auto buf_a = MakeDramBuffer(device(), 2048);
    auto buf_b = MakeDramBuffer(device(), 4096);

    KernelDescriptor kd = MakeBlankReaderKernel({0, 0});
    kd.emplace_runtime_args({0, 0}, {buf_a.get(), buf_b.get()});

    ProgramDescriptor desc;
    desc.kernels = {kd};

    Program program{desc};
    ResolvedBindings resolved = resolve_bindings(program, desc, {buf_a.get(), buf_b.get()});

    ASSERT_EQ(resolved.rt_args.size(), 2u);
    // Bindings should be recorded in arg-index order.
    EXPECT_EQ(resolved.rt_args[0].arg_idx, 0u);
    EXPECT_EQ(resolved.rt_args[0].tensor_buffer_idx, 0u);
    EXPECT_EQ(resolved.rt_args[1].arg_idx, 1u);
    EXPECT_EQ(resolved.rt_args[1].tensor_buffer_idx, 1u);
}

// resolve_bindings for a common (non-per-core) buffer arg.
// is_common should be true; core field is irrelevant / unused.
TEST_F(DescriptorPatchingDeviceTest, Tensix_ResolveBindings_CommonBuffer_IsCommonTrue) {
    auto buf_a = MakeDramBuffer(device());

    KernelDescriptor kd = MakeBlankReaderKernel({0, 0});
    kd.emplace_common_runtime_args({50u, buf_a.get()});

    ProgramDescriptor desc;
    desc.kernels = {kd};

    Program program{desc};
    ResolvedBindings resolved = resolve_bindings(program, desc, {buf_a.get()});

    ASSERT_EQ(resolved.rt_args.size(), 1u);
    const auto& b = resolved.rt_args[0];
    EXPECT_EQ(b.kernel_idx, 0u);
    EXPECT_EQ(b.arg_idx, 1u);  // buf_a is at index 1 in common args
    EXPECT_EQ(b.tensor_buffer_idx, 0u);
    EXPECT_TRUE(b.is_common);
}

// Descriptor with only uint32_t args → ResolvedBindings is empty.
TEST_F(DescriptorPatchingDeviceTest, Tensix_ResolveBindings_NoBufferArgs_ReturnsEmpty) {
    KernelDescriptor kd = MakeBlankReaderKernel({0, 0});
    kd.emplace_runtime_args({0, 0}, {1u, 2u, 3u});

    ProgramDescriptor desc;
    desc.kernels = {kd};

    Program program{desc};
    ResolvedBindings resolved = resolve_bindings(program, desc, {});

    EXPECT_TRUE(resolved.empty());
}

// apply_resolved_bindings patches per-core runtime args with the new buffer's address.
TEST_F(DescriptorPatchingDeviceTest, Tensix_ApplyResolvedBindings_PatchesPerCoreAddress) {
    auto buf_a = MakeDramBuffer(device(), 2048);
    auto buf_b = MakeDramBuffer(device(), 4096);

    KernelDescriptor kd = MakeBlankReaderKernel({0, 0});
    kd.emplace_runtime_args({0, 0}, {buf_a.get(), 42u});

    ProgramDescriptor desc;
    desc.kernels = {kd};

    Program program{desc};
    ResolvedBindings resolved = resolve_bindings(program, desc, {buf_a.get()});

    // Pre-apply: arg[0] should be buf_a's address.
    RuntimeArgsData& before = GetRuntimeArgs(program, 0, {0, 0});
    EXPECT_EQ(before[0], buf_a->address());
    EXPECT_EQ(before[1], 42u);  // uint32 arg unchanged

    // Apply with buf_b.
    apply_resolved_bindings(program, resolved, {buf_b.get()});

    RuntimeArgsData& after = GetRuntimeArgs(program, 0, {0, 0});
    EXPECT_EQ(after[0], buf_b->address());
    EXPECT_EQ(after[1], 42u);  // uint32 arg still unchanged
}

// apply_resolved_bindings patches common runtime args with the new buffer's address.
TEST_F(DescriptorPatchingDeviceTest, Tensix_ApplyResolvedBindings_PatchesCommonAddress) {
    auto buf_a = MakeDramBuffer(device(), 2048);
    auto buf_b = MakeDramBuffer(device(), 4096);

    KernelDescriptor kd = MakeBlankReaderKernel({0, 0});
    kd.emplace_common_runtime_args({77u, buf_a.get()});

    ProgramDescriptor desc;
    desc.kernels = {kd};

    Program program{desc};
    ResolvedBindings resolved = resolve_bindings(program, desc, {buf_a.get()});

    // Pre-apply: common arg[1] should be buf_a's address.
    RuntimeArgsData& before = GetCommonRuntimeArgs(program, 0);
    EXPECT_EQ(before[0], 77u);
    EXPECT_EQ(before[1], buf_a->address());

    apply_resolved_bindings(program, resolved, {buf_b.get()});

    RuntimeArgsData& after = GetCommonRuntimeArgs(program, 0);
    EXPECT_EQ(after[0], 77u);
    EXPECT_EQ(after[1], buf_b->address());
}

// apply_resolved_bindings can be called repeatedly with different buffer sets.
// Each call should update the runtime arg to the current buffer's address.
TEST_F(DescriptorPatchingDeviceTest, Tensix_ApplyResolvedBindings_RepeatedApplication) {
    auto buf_a = MakeDramBuffer(device(), 2048);
    auto buf_b = MakeDramBuffer(device(), 4096);
    auto buf_c = MakeDramBuffer(device(), 8192);

    KernelDescriptor kd = MakeBlankReaderKernel({0, 0});
    kd.emplace_runtime_args({0, 0}, {buf_a.get()});

    ProgramDescriptor desc;
    desc.kernels = {kd};

    Program program{desc};
    ResolvedBindings resolved = resolve_bindings(program, desc, {buf_a.get()});

    apply_resolved_bindings(program, resolved, {buf_b.get()});
    EXPECT_EQ(GetRuntimeArgs(program, 0, {0, 0})[0], buf_b->address());

    apply_resolved_bindings(program, resolved, {buf_c.get()});
    EXPECT_EQ(GetRuntimeArgs(program, 0, {0, 0})[0], buf_c->address());

    // Apply back to buf_a.
    apply_resolved_bindings(program, resolved, {buf_a.get()});
    EXPECT_EQ(GetRuntimeArgs(program, 0, {0, 0})[0], buf_a->address());
}

// Regression: a descriptor with sharded CB buffers but no emplace_runtime_args()
// Buffer* calls must produce empty ResolvedBindings, so the adapter falls through
// to the slow path.  Before the fix, CB-only bindings made empty() return false,
// causing the fast path to activate for factories that never opted in.
TEST_F(DescriptorPatchingDeviceTest, Tensix_ResolveBindings_CbOnlyBuffers_ReturnsEmpty) {
    auto buf_dram = MakeDramBuffer(device());
    auto buf_l1 = MakeL1Buffer(device());

    KernelDescriptor kd = MakeBlankReaderKernel({0, 0});
    // Old-style: push buffer address as plain uint32_t (no Buffer* binding).
    kd.runtime_args.emplace_back(CoreCoord{0, 0}, KernelDescriptor::CoreRuntimeArgs{buf_dram->address(), 42u});

    ProgramDescriptor desc;
    desc.kernels = {kd};

    // Simulate a sharded CB: set .buffer on the CB descriptor (must be L1).
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2048,
        .core_ranges = CoreRangeSet{CoreRange{{0, 0}}},
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = 0,
            .data_format = tt::DataFormat::Float16_b,
            .page_size = 2048,
        }}},
        .buffer = buf_l1.get(),
    });

    Program program{desc};
    ResolvedBindings resolved = resolve_bindings(program, desc, {buf_l1.get()});

    // No rt_arg bindings were declared, so both rt_args and cbs must be empty.
    EXPECT_TRUE(resolved.rt_args.empty());
    EXPECT_TRUE(resolved.cbs.empty());
    EXPECT_TRUE(resolved.empty());
}

// resolve_bindings fires TT_FATAL when a binding buffer is not in tensor_buffers.
TEST_F(DescriptorPatchingDeviceTest, Tensix_ResolveBindings_BufferNotInTensorList_Throws) {
    auto buf_a = MakeDramBuffer(device());
    auto buf_other = MakeDramBuffer(device());

    KernelDescriptor kd = MakeBlankReaderKernel({0, 0});
    kd.emplace_runtime_args({0, 0}, {buf_a.get()});

    ProgramDescriptor desc;
    desc.kernels = {kd};

    Program program{desc};

    // buf_a is not in the tensor list — should throw.
    EXPECT_ANY_THROW(resolve_bindings(program, desc, {buf_other.get()}));
}

}  // namespace
}  // namespace tt::tt_metal
