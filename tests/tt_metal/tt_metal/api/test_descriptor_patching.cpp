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
#include <functional>
#include <memory>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/runtime_args_data.hpp>
#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>
#include <tt_stl/assert.hpp>

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
    kd.kernel_source = "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp";
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

MeshTensor MakeSingleTileL1MeshTensor(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    auto tensor_layout = TensorLayout(
        DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::L1});
    auto spec = TensorSpec(Shape{32, 32}, tensor_layout);
    return MeshTensor::allocate_on_device(*mesh_device, spec, TensorTopology());
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

// Regression: emplace_runtime_args must accept nullptr Buffer* as a placeholder for
// an absent optional tensor.  It emits 0u into the runtime arg slot and registers no
// binding, so the cache-hit fast path stays valid for ops with optional inputs.
TEST(DescriptorPatching, EmplaceRuntimeArgs_NullBuffer_EmitsZero_NoBinding) {
    KernelDescriptor kd;
    Buffer* null_buf = nullptr;
    kd.emplace_runtime_args({0, 0}, {1u, null_buf, 3u});

    ASSERT_EQ(kd.runtime_args.size(), 1u);
    EXPECT_EQ(kd.runtime_args[0].second, (std::vector<uint32_t>{1u, 0u, 3u}));
    EXPECT_TRUE(kd.buffer_bindings.empty());
}

TEST(DescriptorPatching, EmplaceCommonRuntimeArgs_NullBuffer_EmitsZero_NoBinding) {
    KernelDescriptor kd;
    Buffer* null_buf = nullptr;
    kd.emplace_common_runtime_args({7u, null_buf, 9u});

    EXPECT_EQ(kd.common_runtime_args, (std::vector<uint32_t>{7u, 0u, 9u}));
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
    ResolvedBindings resolved = resolve_bindings(program, desc, std::vector<Buffer*>{buf_a.get()});

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
    ResolvedBindings resolved = resolve_bindings(program, desc, std::vector<Buffer*>{buf_a.get()});

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
    ResolvedBindings resolved = resolve_bindings(program, desc, std::vector<Buffer*>{buf_a.get(), buf_b.get()});

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
    ResolvedBindings resolved = resolve_bindings(program, desc, std::vector<Buffer*>{buf_a.get()});

    ASSERT_EQ(resolved.rt_args.size(), 1u);
    const auto& b = resolved.rt_args[0];
    EXPECT_EQ(b.kernel_idx, 0u);
    EXPECT_EQ(b.arg_idx, 1u);  // buf_a is at index 1 in common args
    EXPECT_EQ(b.tensor_buffer_idx, 0u);
    EXPECT_TRUE(b.is_common);
}

TEST_F(DescriptorPatchingDeviceTest, Tensix_ResolveBindings_PerCoreMeshTensorArg_CorrectTuples) {
    auto tensor = MakeSingleTileL1MeshTensor(get_mesh_device());
    Buffer* tensor_buffer = tensor.mesh_buffer().get_reference_buffer();
    ASSERT_NE(tensor_buffer, nullptr);

    KernelDescriptor kd = MakeBlankReaderKernel({0, 0});
    kd.emplace_runtime_args({0, 0}, {std::cref(tensor), 42u});

    ProgramDescriptor desc;
    desc.kernels = {kd};

    Program program{desc};
    ResolvedBindings resolved = resolve_bindings(program, desc, std::vector<Buffer*>{tensor_buffer});

    ASSERT_EQ(resolved.rt_args.size(), 1u);
    const auto& b = resolved.rt_args[0];
    EXPECT_EQ(b.kernel_idx, 0u);
    EXPECT_EQ(b.core, CoreCoord(0, 0));
    EXPECT_EQ(b.arg_idx, 0u);
    EXPECT_EQ(b.tensor_buffer_idx, 0u);
    EXPECT_FALSE(b.is_common);
}

TEST_F(DescriptorPatchingDeviceTest, Tensix_ResolveBindings_CommonMeshTensorArg_IsCommonTrue) {
    auto tensor = MakeSingleTileL1MeshTensor(get_mesh_device());
    Buffer* tensor_buffer = tensor.mesh_buffer().get_reference_buffer();
    ASSERT_NE(tensor_buffer, nullptr);

    KernelDescriptor kd = MakeBlankReaderKernel({0, 0});
    kd.emplace_common_runtime_args({7u, std::cref(tensor)});

    ProgramDescriptor desc;
    desc.kernels = {kd};

    Program program{desc};
    ResolvedBindings resolved = resolve_bindings(program, desc, std::vector<Buffer*>{tensor_buffer});

    ASSERT_EQ(resolved.rt_args.size(), 1u);
    const auto& b = resolved.rt_args[0];
    EXPECT_EQ(b.kernel_idx, 0u);
    EXPECT_EQ(b.arg_idx, 1u);
    EXPECT_EQ(b.tensor_buffer_idx, 0u);
    EXPECT_TRUE(b.is_common);
}

TEST_F(DescriptorPatchingDeviceTest, Tensix_ResolveBindings_CbTensor_ResolvesToTensorBufferSlot) {
    auto tensor = MakeSingleTileL1MeshTensor(get_mesh_device());
    Buffer* tensor_buffer = tensor.mesh_buffer().get_reference_buffer();
    ASSERT_NE(tensor_buffer, nullptr);

    KernelDescriptor kd = MakeBlankReaderKernel({0, 0});
    // Add at least one runtime binding so resolve_bindings processes CB descriptors.
    kd.emplace_runtime_args({0, 0}, {std::cref(tensor)});

    ProgramDescriptor desc;
    desc.kernels = {kd};
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2048,
        .core_ranges = CoreRangeSet{CoreRange{{0, 0}}},
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = 0,
            .data_format = tt::DataFormat::Float16_b,
            .page_size = 2048,
        }}},
        .tensor = &tensor,
    });

    Program program{desc};
    ResolvedBindings resolved = resolve_bindings(program, desc, std::vector<Buffer*>{tensor_buffer});

    ASSERT_EQ(resolved.rt_args.size(), 1u);
    ASSERT_EQ(resolved.cbs.size(), 1u);
    EXPECT_EQ(resolved.cbs[0].tensor_buffer_idx, 0u);
    EXPECT_EQ(resolved.cbs[0].address_offset, 0u);
}

TEST_F(DescriptorPatchingDeviceTest, Tensix_ProgramConstruction_CbWithBothBufferAndTensor_Throws) {
    auto tensor = MakeSingleTileL1MeshTensor(get_mesh_device());
    Buffer* tensor_buffer = tensor.mesh_buffer().get_reference_buffer();
    ASSERT_NE(tensor_buffer, nullptr);

    ProgramDescriptor desc;
    desc.kernels = {MakeBlankReaderKernel({0, 0})};
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2048,
        .core_ranges = CoreRangeSet{CoreRange{{0, 0}}},
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = 0,
            .data_format = tt::DataFormat::Float16_b,
            .page_size = 2048,
        }}},
        .buffer = tensor_buffer,
        .tensor = &tensor,
    });

    EXPECT_ANY_THROW((void)Program{desc});
}

TEST_F(DescriptorPatchingDeviceTest, Tensix_ApplyDescriptorRuntimeArgs_CbTensor_UpdatesAddress) {
    auto tensor_a = MakeSingleTileL1MeshTensor(get_mesh_device());
    auto tensor_b = MakeSingleTileL1MeshTensor(get_mesh_device());
    Buffer* buf_a = tensor_a.mesh_buffer().get_reference_buffer();
    Buffer* buf_b = tensor_b.mesh_buffer().get_reference_buffer();
    ASSERT_NE(buf_a, nullptr);
    ASSERT_NE(buf_b, nullptr);
    ASSERT_NE(buf_a, buf_b);

    KernelDescriptor kd = MakeBlankReaderKernel({0, 0});
    kd.emplace_runtime_args({0, 0}, {1u});

    ProgramDescriptor desc;
    desc.kernels = {kd};
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2048,
        .core_ranges = CoreRangeSet{CoreRange{{0, 0}}},
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = 0,
            .data_format = tt::DataFormat::Float16_b,
            .page_size = 2048,
        }}},
        .tensor = &tensor_a,
    });

    Program program{desc};

    auto cbs = program.circular_buffers();
    ASSERT_EQ(cbs.size(), 1u);
    CBHandle cb_id = cbs[0]->id();

    // Initial CB address should be tensor_a's buffer address.
    EXPECT_EQ(GetCircularBufferConfig(program, cb_id).globally_allocated_address(), buf_a->address());

    // Switch to tensor_b via apply_descriptor_runtime_args (the slow-path cache-hit call).
    ProgramDescriptor desc2 = desc;
    desc2.cbs[0].tensor = &tensor_b;
    apply_descriptor_runtime_args(program, desc2);

    // CB address must now reflect tensor_b.
    EXPECT_EQ(GetCircularBufferConfig(program, cb_id).globally_allocated_address(), buf_b->address());
}

// apply_descriptor_runtime_args must fire TT_FATAL when a CBDescriptor has both
// buffer and tensor set (mutual-exclusion invariant, same guard as program construction).
TEST_F(DescriptorPatchingDeviceTest, Tensix_ApplyDescriptorRuntimeArgs_CbBothBufferAndTensor_Throws) {
    auto tensor = MakeSingleTileL1MeshTensor(get_mesh_device());
    Buffer* tensor_buffer = tensor.mesh_buffer().get_reference_buffer();
    ASSERT_NE(tensor_buffer, nullptr);

    KernelDescriptor kd = MakeBlankReaderKernel({0, 0});
    kd.emplace_runtime_args({0, 0}, {1u});

    // Build a valid program first (tensor only, no buffer).
    ProgramDescriptor valid_desc;
    valid_desc.kernels = {kd};
    valid_desc.cbs.push_back(CBDescriptor{
        .total_size = 2048,
        .core_ranges = CoreRangeSet{CoreRange{{0, 0}}},
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = 0,
            .data_format = tt::DataFormat::Float16_b,
            .page_size = 2048,
        }}},
        .tensor = &tensor,
    });
    Program program{valid_desc};

    // Now pass a malformed descriptor (both fields set) to apply_descriptor_runtime_args.
    ProgramDescriptor bad_desc = valid_desc;
    bad_desc.cbs[0].buffer = tensor_buffer;
    EXPECT_ANY_THROW(apply_descriptor_runtime_args(program, bad_desc));
}

// Descriptor with only uint32_t args → ResolvedBindings is empty.
TEST_F(DescriptorPatchingDeviceTest, Tensix_ResolveBindings_NoBufferArgs_ReturnsEmpty) {
    KernelDescriptor kd = MakeBlankReaderKernel({0, 0});
    kd.emplace_runtime_args({0, 0}, {1u, 2u, 3u});

    ProgramDescriptor desc;
    desc.kernels = {kd};

    Program program{desc};
    ResolvedBindings resolved = resolve_bindings(program, desc, std::vector<Buffer*>{});

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
    ResolvedBindings resolved = resolve_bindings(program, desc, std::vector<Buffer*>{buf_a.get()});

    // Pre-apply: arg[0] should be buf_a's address.
    RuntimeArgsData& before = GetRuntimeArgs(program, 0, {0, 0});
    EXPECT_EQ(before[0], buf_a->address());
    EXPECT_EQ(before[1], 42u);  // uint32 arg unchanged

    // Apply with buf_b.
    apply_resolved_bindings(program, resolved, std::vector<Buffer*>{buf_b.get()});

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
    ResolvedBindings resolved = resolve_bindings(program, desc, std::vector<Buffer*>{buf_a.get()});

    // Pre-apply: common arg[1] should be buf_a's address.
    RuntimeArgsData& before = GetCommonRuntimeArgs(program, 0);
    EXPECT_EQ(before[0], 77u);
    EXPECT_EQ(before[1], buf_a->address());

    apply_resolved_bindings(program, resolved, std::vector<Buffer*>{buf_b.get()});

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
    ResolvedBindings resolved = resolve_bindings(program, desc, std::vector<Buffer*>{buf_a.get()});

    apply_resolved_bindings(program, resolved, std::vector<Buffer*>{buf_b.get()});
    EXPECT_EQ(GetRuntimeArgs(program, 0, {0, 0})[0], buf_b->address());

    apply_resolved_bindings(program, resolved, std::vector<Buffer*>{buf_c.get()});
    EXPECT_EQ(GetRuntimeArgs(program, 0, {0, 0})[0], buf_c->address());

    // Apply back to buf_a.
    apply_resolved_bindings(program, resolved, std::vector<Buffer*>{buf_a.get()});
    EXPECT_EQ(GetRuntimeArgs(program, 0, {0, 0})[0], buf_a->address());
}

// Regression: a descriptor with sharded CB buffers but no emplace_runtime_args()
// Buffer* calls must produce empty ResolvedBindings, so the adapter falls through
// to the slow path.  Before the fix, CB-only bindings made empty() return false,
// causing the fast path to activate for factories that never opted in.
TEST_F(DescriptorPatchingDeviceTest, Tensix_ResolveBindings_CbOnly_PopulatesCbs_RtArgsEmpty) {
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
    ResolvedBindings resolved = resolve_bindings(program, desc, std::vector<Buffer*>{buf_l1.get()});

    // resolve_bindings is policy-free: every CB `.buffer = ...` binding is
    // resolved, even when no rt-arg buffer bindings are declared.  The caller
    // (the adapter) decides whether to fast-path with a CB-only ResolvedBindings
    // or to take a slow-path rebuild for factories that still bind raw uint32
    // runtime args.  See DescriptorMeshWorkloadAdapter::apply_descriptor.
    EXPECT_TRUE(resolved.rt_args.empty());
    EXPECT_EQ(resolved.cbs.size(), 1u);
    EXPECT_EQ(resolved.cbs[0].tensor_buffer_idx, 0u);
    EXPECT_FALSE(resolved.empty());
}

// Regression: when tensor_buffers contains the same Buffer* more than once
// (e.g. matmul(X, X) in newton-schulz, or an output that aliases an input),
// resolve_bindings cannot disambiguate which binding maps to which slot from
// Buffer* alone.  std::find would silently return the first occurrence for
// every binding, causing apply_resolved_bindings to write the same address to
// every slot and producing wrong results on a future cache hit with two
// distinct tensors.
//
// The fix bails out and returns an empty ResolvedBindings, forcing the
// adapter onto the slow path (rebuild the descriptor) for that cache hit.
TEST_F(DescriptorPatchingDeviceTest, Tensix_ResolveBindings_DuplicateBuffer_ReturnsEmpty) {
    auto buf_a = MakeDramBuffer(device());

    KernelDescriptor kd = MakeBlankReaderKernel({0, 0});
    // Two bindings for the same Buffer*, simulating an op that takes the same
    // tensor at two distinct positional slots.
    kd.emplace_runtime_args({0, 0}, {buf_a.get(), buf_a.get()});

    ProgramDescriptor desc;
    desc.kernels = {kd};

    Program program{desc};
    // tensor_buffers reflects "X used twice" — the same Buffer* appears at
    // both slots.  resolve_bindings must report empty so the adapter takes
    // the slow path.
    ResolvedBindings resolved = resolve_bindings(program, desc, std::vector<Buffer*>{buf_a.get(), buf_a.get()});

    EXPECT_TRUE(resolved.empty());
    EXPECT_TRUE(resolved.rt_args.empty());
    EXPECT_TRUE(resolved.cbs.empty());
}

// #48928: an OUTPUT buffer that aliases an INPUT (in-place op writing back into its input) is a
// same-buffer-by-construction alias, not the ambiguous matmul(X, X) case.  With num_input_buffers
// marking the input/output boundary, resolve_bindings must KEEP the fast path (non-empty bindings).
TEST_F(DescriptorPatchingDeviceTest, Tensix_ResolveBindings_OutputAliasesInput_KeepsFastPath) {
    auto buf_a = MakeDramBuffer(device());

    KernelDescriptor kd = MakeBlankReaderKernel({0, 0});
    kd.emplace_runtime_args({0, 0}, {buf_a.get()});

    ProgramDescriptor desc;
    desc.kernels = {kd};

    Program program{desc};
    // [input=buf_a, output=buf_a]; first entry is the input, the second is the in-place output.
    ResolvedBindings resolved =
        resolve_bindings(program, desc, std::vector<Buffer*>{buf_a.get(), buf_a.get()}, /*num_input_buffers=*/1);

    EXPECT_FALSE(resolved.empty());
    ASSERT_EQ(resolved.rt_args.size(), 1u);
    EXPECT_EQ(resolved.rt_args[0].tensor_buffer_idx, 0u);
}

// #48928: a duplicate purely WITHIN the input region (two distinct input operands that coincide,
// matmul(X, X)) must still bail even when num_input_buffers is given.
TEST_F(DescriptorPatchingDeviceTest, Tensix_ResolveBindings_InputRegionDuplicate_ReturnsEmpty) {
    auto buf_a = MakeDramBuffer(device());

    KernelDescriptor kd = MakeBlankReaderKernel({0, 0});
    kd.emplace_runtime_args({0, 0}, {buf_a.get(), buf_a.get()});

    ProgramDescriptor desc;
    desc.kernels = {kd};

    Program program{desc};
    // Both entries are inputs (no output region); the repeat is ambiguous → bail.
    ResolvedBindings resolved =
        resolve_bindings(program, desc, std::vector<Buffer*>{buf_a.get(), buf_a.get()}, /*num_input_buffers=*/2);

    EXPECT_TRUE(resolved.empty());
}

// #48928: op(X, X, out=X) — X at two genuine input positions plus the in-place output_tensor slot.
// Even with the in-place opt-in ON, the output-alias skip is granted ONCE; the extra input
// occurrence still bails, so a later same-shape op(X, Y, out=X) cache hit can't patch input-b to X.
TEST_F(DescriptorPatchingDeviceTest, Tensix_ResolveBindings_OutputAliasRepeatedInInputs_ReturnsEmpty) {
    auto buf_a = MakeDramBuffer(device());

    KernelDescriptor kd = MakeBlankReaderKernel({0, 0});
    kd.emplace_runtime_args({0, 0}, {buf_a.get()});

    ProgramDescriptor desc;
    desc.kernels = {kd};

    Program program{desc};
    // [input_a=X, input_b=X, output_tensor=X | return=X]; num_input_buffers=3 (tensor_args), 1 output.
    ResolvedBindings resolved = resolve_bindings(
        program,
        desc,
        std::vector<Buffer*>{buf_a.get(), buf_a.get(), buf_a.get(), buf_a.get()},
        /*num_input_buffers=*/3,
        /*allow_inplace_output_tensor_alias=*/true);

    EXPECT_TRUE(resolved.empty());
}

// #48928/#49573: an in-place op's output_tensor carried INSIDE tensor_args (input region) aliasing
// an input — [input=X, output_tensor=X | return=X].  This is the SDXL-silu / MorehAdamW shape.
//   - Default (opt-out): treated as an ambiguous input-region duplicate → BAIL to slow-path rebuild.
//     This is the safe behavior for ops whose get_dynamic_runtime_args is incomplete.
TEST_F(DescriptorPatchingDeviceTest, Tensix_ResolveBindings_InplaceOutputTensorInInputs_Default_ReturnsEmpty) {
    auto buf_a = MakeDramBuffer(device());

    KernelDescriptor kd = MakeBlankReaderKernel({0, 0});
    kd.emplace_runtime_args({0, 0}, {buf_a.get()});

    ProgramDescriptor desc;
    desc.kernels = {kd};

    Program program{desc};
    // num_input_buffers=2 (input + output_tensor), 1 output (return). Default opt-in (false) → bail.
    ResolvedBindings resolved = resolve_bindings(
        program, desc, std::vector<Buffer*>{buf_a.get(), buf_a.get(), buf_a.get()}, /*num_input_buffers=*/2);

    EXPECT_TRUE(resolved.empty());
}

//   - Opt-in (allow_inplace_output_tensor_alias=true): the op re-applies every cache-hit-varying arg
//     itself (binary_ng), so the output_tensor is a safe in-place alias → KEEP the fast path.
TEST_F(DescriptorPatchingDeviceTest, Tensix_ResolveBindings_InplaceOutputTensorInInputs_OptIn_KeepsFastPath) {
    auto buf_a = MakeDramBuffer(device());

    KernelDescriptor kd = MakeBlankReaderKernel({0, 0});
    kd.emplace_runtime_args({0, 0}, {buf_a.get()});

    ProgramDescriptor desc;
    desc.kernels = {kd};

    Program program{desc};
    ResolvedBindings resolved = resolve_bindings(
        program,
        desc,
        std::vector<Buffer*>{buf_a.get(), buf_a.get(), buf_a.get()},
        /*num_input_buffers=*/2,
        /*allow_inplace_output_tensor_alias=*/true);

    EXPECT_FALSE(resolved.empty());
    ASSERT_EQ(resolved.rt_args.size(), 1u);
    EXPECT_EQ(resolved.rt_args[0].tensor_buffer_idx, 0u);
}

// resolve_bindings fires TT_FATAL when a runtime-arg binding buffer is not in
// tensor_buffers — RT-arg bindings are the only mechanism that lets the fast
// path patch a changing input/output address on cache hits, so a missing
// buffer is genuinely unrecoverable.
TEST_F(DescriptorPatchingDeviceTest, Tensix_ResolveBindings_RtArgBufferNotInTensorList_Throws) {
    auto buf_a = MakeDramBuffer(device());
    auto buf_other = MakeDramBuffer(device());

    KernelDescriptor kd = MakeBlankReaderKernel({0, 0});
    kd.emplace_runtime_args({0, 0}, {buf_a.get()});

    ProgramDescriptor desc;
    desc.kernels = {kd};

    Program program{desc};

    // buf_a is not in the tensor list — should throw.
    EXPECT_ANY_THROW(resolve_bindings(program, desc, std::vector<Buffer*>{buf_other.get()}));
}

// Regression: CB `.buffer = ...` bindings whose buffer is NOT in tensor_buffers
// must be silently skipped (not fatal).  Such buffers come from workload-scoped
// resources the factory injects directly into a CBDescriptor — for example,
// `dram_prefetcher`'s reader CB pegged to a `GlobalCircularBuffer`'s backing
// buffer that arrives via `operation_attributes`, not tensor_args.  These
// buffers have stable addresses across dispatches, so the fast path doesn't
// need a binding for them.
TEST_F(DescriptorPatchingDeviceTest, Tensix_ResolveBindings_CbBufferNotInTensorList_SkipsBindingNoThrow) {
    auto buf_l1_pegged = MakeL1Buffer(device());   // simulates GlobalCircularBuffer's backing buffer
    auto buf_l1_in_args = MakeL1Buffer(device());  // a CB buffer that IS in tensor_args

    KernelDescriptor kd = MakeBlankReaderKernel({0, 0});
    ProgramDescriptor desc;
    desc.kernels = {kd};
    // First CB: backed by a non-tensor buffer (should be skipped silently).
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2048,
        .core_ranges = CoreRangeSet{CoreRange{{0, 0}}},
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = 0,
            .data_format = tt::DataFormat::Float16_b,
            .page_size = 2048,
        }}},
        .buffer = buf_l1_pegged.get(),
    });
    // Second CB: backed by a tensor-arg buffer (should be resolved normally).
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2048,
        .core_ranges = CoreRangeSet{CoreRange{{0, 0}}},
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = 1,
            .data_format = tt::DataFormat::Float16_b,
            .page_size = 2048,
        }}},
        .buffer = buf_l1_in_args.get(),
    });

    Program program{desc};

    // tensor_buffers contains only buf_l1_in_args.  buf_l1_pegged is the
    // workload-scoped resource — must NOT throw, just skip.
    ResolvedBindings resolved;
    ASSERT_NO_THROW(resolved = resolve_bindings(program, desc, std::vector<Buffer*>{buf_l1_in_args.get()}));

    // The CB backed by the in-args buffer should be resolved; the pegged one omitted.
    EXPECT_TRUE(resolved.rt_args.empty());
    ASSERT_EQ(resolved.cbs.size(), 1u);
    EXPECT_EQ(resolved.cbs[0].tensor_buffer_idx, 0u);
}

// Regression: a scalar runtime arg whose value happens to numerically equal a
// registered buffer's address must NOT trigger a safety-scan false positive.
//
// An earlier version of resolve_bindings ran a value-match scan that looked for
// any uint32_t arg matching a registered buffer address but lacking a
// BufferBinding at that position, and fired TT_FATAL. The intent was to catch
// the push_back(buf->address()) factory-author mistake, but the check produced
// false positives whenever a legitimate scalar arg (loop counter, shape dim,
// etc.) happened to share the same numeric value as a buffer's address — most
// notably under graph-capture (sentinel addresses) and for low-address buffers.
//
// The check was removed; this test pins that behavior so it is not silently
// reintroduced.
TEST_F(DescriptorPatchingDeviceTest, Tensix_ResolveBindings_ScalarMatchingBufferAddress_DoesNotThrow) {
    auto buf_a = MakeDramBuffer(device());
    const uint32_t collision_value = buf_a->address();

    KernelDescriptor kd = MakeBlankReaderKernel({0, 0});
    // arg[0] is the registered buffer; arg[1] is a plain scalar that happens to
    // equal buf_a's address. The legacy value-match scan would have flagged
    // arg[1] as an undeclared address; the current implementation must not.
    kd.emplace_runtime_args({0, 0}, {buf_a.get(), collision_value});

    ProgramDescriptor desc;
    desc.kernels = {kd};

    Program program{desc};
    EXPECT_NO_THROW(resolve_bindings(program, desc, std::vector<Buffer*>{buf_a.get()}));
}

// Regression: same idea for common runtime args.
TEST_F(DescriptorPatchingDeviceTest, Tensix_ResolveBindings_CommonScalarMatchingBufferAddress_DoesNotThrow) {
    auto buf_a = MakeDramBuffer(device());
    const uint32_t collision_value = buf_a->address();

    KernelDescriptor kd = MakeBlankReaderKernel({0, 0});
    kd.emplace_common_runtime_args({buf_a.get(), collision_value});

    ProgramDescriptor desc;
    desc.kernels = {kd};

    Program program{desc};
    EXPECT_NO_THROW(resolve_bindings(program, desc, std::vector<Buffer*>{buf_a.get()}));
}

// ============================================================================
// Metal 2.0 named-binding descriptor path (rand is the reference op)
// ============================================================================

// A valid DFB binding (accessor -> a named CBDescriptor in the same descriptor) builds a Metal 2.0
// kernel without error.
TEST_F(DescriptorPatchingDeviceTest, Metal2Bindings_ValidDFBReference_Builds) {
    ProgramDescriptor desc;
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2048,
        .core_ranges = CoreRangeSet{CoreRange{{0, 0}}},
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = 0,
            .data_format = tt::DataFormat::Float16_b,
            .page_size = 2048,
        }}},
        .name = "intermed",
    });
    KernelDescriptor kd = MakeBlankReaderKernel({0, 0});
    kd.dfb_bindings = {{.accessor_name = "cb", .cb_name = "intermed"}};
    desc.kernels = {kd};

    EXPECT_NO_THROW({ Program program{desc}; });
}

// A DFB binding referencing a CB name with no matching named CBDescriptor must fail loudly rather than
// silently bind the kernel to an unrelated circular buffer.
TEST_F(DescriptorPatchingDeviceTest, Metal2Bindings_DanglingDFBReference_Throws) {
    ProgramDescriptor desc;
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2048,
        .core_ranges = CoreRangeSet{CoreRange{{0, 0}}},
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = 0,
            .data_format = tt::DataFormat::Float16_b,
            .page_size = 2048,
        }}},
        .name = "intermed",
    });
    KernelDescriptor kd = MakeBlankReaderKernel({0, 0});
    kd.dfb_bindings = {{.accessor_name = "cb", .cb_name = "does_not_exist"}};
    desc.kernels = {kd};

    EXPECT_ANY_THROW({ Program program{desc}; });
}

// Two named CBDescriptors sharing a name would let a DFB binding resolve to the wrong CB; the build
// must reject the duplicate.
TEST_F(DescriptorPatchingDeviceTest, Metal2Bindings_DuplicateCBName_Throws) {
    ProgramDescriptor desc;
    for (uint32_t idx : {0u, 1u}) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = 2048,
            .core_ranges = CoreRangeSet{CoreRange{{0, 0}}},
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = idx,
                .data_format = tt::DataFormat::Float16_b,
                .page_size = 2048,
            }}},
            .name = "intermed",
        });
    }
    desc.kernels = {MakeBlankReaderKernel({0, 0})};

    EXPECT_ANY_THROW({ Program program{desc}; });
}

}  // namespace
}  // namespace tt::tt_metal
