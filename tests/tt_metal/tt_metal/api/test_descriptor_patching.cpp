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
// SECTION 3: apply_descriptor_runtime_args — slow-path CB re-application
// ============================================================================
//
// Contract-1 factories use apply_descriptor_runtime_args on cache hit when
// there are no Buffer* bindings (factory hands the freshly-re-run descriptor
// back to the framework; rt args, common rt args, dynamic CB addresses, and —
// since this PR — CB total_size / per-format page_size are patched into the
// cached program in place).
//
// The 4 size-mutation ops (attn_matmul, group_attn_matmul, slice_rm,
// generic_op) deliberately keep CB total_size out of the program hash and
// rely on cheap UpdateCircularBufferTotalSize on cache hit.  Without this
// patching the cached program would keep the first-build size forever.

namespace {

ProgramDescriptor MakeOneCbDescriptor(uint32_t total_size, uint32_t page_size) {
    ProgramDescriptor desc;
    desc.cbs.push_back(CBDescriptor{
        .total_size = total_size,
        .core_ranges = CoreRangeSet{CoreRange{CoreCoord{0, 0}}},
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = 0,
            .data_format = tt::DataFormat::Float16_b,
            .page_size = page_size,
        }},
    });
    desc.kernels = {MakeBlankReaderKernel({0, 0})};
    return desc;
}

}  // namespace

// total_size in a fresh descriptor is applied to the cached program's CB.
TEST_F(DescriptorPatchingDeviceTest, Tensix_ApplyDescriptorRuntimeArgs_PatchesCbTotalSize) {
    ProgramDescriptor desc = MakeOneCbDescriptor(/*total_size=*/1024, /*page_size=*/64);
    Program program{desc};

    auto cbs = program.circular_buffers();
    ASSERT_EQ(cbs.size(), 1u);
    EXPECT_EQ(cbs[0]->size(), 1024u);

    // Mutate total_size as a factory would on rebuild — kernel hash unchanged
    // because total_size is not hashed (see hash_cb_descriptor).
    desc.cbs[0].total_size = 2048;
    apply_descriptor_runtime_args(program, desc);

    EXPECT_EQ(program.circular_buffers()[0]->size(), 2048u);
}

// page_size in a fresh descriptor is applied to the cached program's CB.
// (Today page_size IS hashed, but the framework still re-applies it for
// future opt-out scenarios — see comment in apply_descriptor_runtime_args.)
TEST_F(DescriptorPatchingDeviceTest, Tensix_ApplyDescriptorRuntimeArgs_PatchesCbPageSize) {
    ProgramDescriptor desc = MakeOneCbDescriptor(/*total_size=*/1024, /*page_size=*/64);
    Program program{desc};
    EXPECT_EQ(program.circular_buffers()[0]->page_size(0), 64u);

    desc.cbs[0].format_descriptors[0].page_size = 128;
    apply_descriptor_runtime_args(program, desc);

    EXPECT_EQ(program.circular_buffers()[0]->page_size(0), 128u);
}

// When the descriptor's sizes match the cached program's, no Update* call is
// issued (early-out keeps the cache-hit fast path lean for the common case).
TEST_F(DescriptorPatchingDeviceTest, Tensix_ApplyDescriptorRuntimeArgs_NoMutation_NoSideEffect) {
    ProgramDescriptor desc = MakeOneCbDescriptor(/*total_size=*/1024, /*page_size=*/64);
    Program program{desc};

    apply_descriptor_runtime_args(program, desc);  // identical sizes

    EXPECT_EQ(program.circular_buffers()[0]->size(), 1024u);
    EXPECT_EQ(program.circular_buffers()[0]->page_size(0), 64u);
}

// Regression: when total_size and page_size BOTH change such that the new
// total_size is not divisible by the OLD page_size (or vice versa), the
// patcher must not trip the divisibility invariant mid-update.  Comparing
// against CircularBuffer::page_size() (which validates total_size % page_size)
// after UpdateCircularBufferTotalSize would throw.  We compare against the
// cached config's raw page_sizes() array instead.
TEST_F(DescriptorPatchingDeviceTest, Tensix_ApplyDescriptorRuntimeArgs_IncompatibleIntermediate_DoesNotThrow) {
    // Cached: total_size=1024, page_size=64 (1024 % 64 == 0).
    ProgramDescriptor desc = MakeOneCbDescriptor(/*total_size=*/1024, /*page_size=*/64);
    Program program{desc};

    // New: total_size=192, page_size=96.  192 % 64 != 0 (incompatible with the
    // OLD cached page_size between the two Update calls); 192 % 96 == 0
    // (consistent once both updates land).
    desc.cbs[0].total_size = 192;
    desc.cbs[0].format_descriptors[0].page_size = 96;
    EXPECT_NO_THROW(apply_descriptor_runtime_args(program, desc));

    EXPECT_EQ(program.circular_buffers()[0]->size(), 192u);
    EXPECT_EQ(program.circular_buffers()[0]->page_size(0), 96u);
}

// remote_format_descriptors (GlobalCircularBuffer-backed CBs) must also be
// walked when patching page_size.  Today page_size is hashed so values match,
// but this guards the path for future opt-outs and matches the legacy
// GenericMeshProgramFactory behavior.
TEST_F(DescriptorPatchingDeviceTest, Tensix_ApplyDescriptorRuntimeArgs_RemoteFormatDescriptors_AlsoPatched) {
    ProgramDescriptor desc;
    desc.cbs.push_back(CBDescriptor{
        .total_size = 1024,
        .core_ranges = CoreRangeSet{CoreRange{CoreCoord{0, 0}}},
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = 0,
            .data_format = tt::DataFormat::Float16_b,
            .page_size = 64,
        }},
        .remote_format_descriptors = {CBFormatDescriptor{
            .buffer_index = 1,
            .data_format = tt::DataFormat::Float16_b,
            .page_size = 64,
        }},
    });
    desc.kernels = {MakeBlankReaderKernel({0, 0})};
    Program program{desc};
    EXPECT_EQ(program.circular_buffers()[0]->page_size(0), 64u);
    EXPECT_EQ(program.circular_buffers()[0]->page_size(1), 64u);

    // Mutating only the remote descriptor's page_size should still produce a
    // patch call (no throw, value updated).
    desc.cbs[0].remote_format_descriptors[0].page_size = 128;
    EXPECT_NO_THROW(apply_descriptor_runtime_args(program, desc));

    EXPECT_EQ(program.circular_buffers()[0]->page_size(0), 64u);
    EXPECT_EQ(program.circular_buffers()[0]->page_size(1), 128u);
}

// Regression: when a sharded CB rebinds its buffer AND grows total_size beyond
// the OLD buffer's bank size, the patcher must use the combined
// UpdateDynamicCircularBufferAddressAndTotalSize so max_size_ is refreshed
// atomically with total_size_.  Two separate Update* calls would trip the
// "total_size <= max_size_" assertion inside set_total_size against the stale
// max_size from the smaller previous buffer.  Surfaced by
// test_group_attn_matmul_with_program_cache when shape changes shrink/grow
// the sharded src0 CB across cache hits.
TEST_F(DescriptorPatchingDeviceTest, Tensix_ApplyDescriptorRuntimeArgs_ShardedRebind_LargerTotalSize_DoesNotThrow) {
    // Cached: total_size=2048 (1 tile), backed by a 2048-byte L1 buffer.
    auto small_buf = MakeL1Buffer(device(), /*size=*/2048);
    ProgramDescriptor desc;
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2048,
        .core_ranges = CoreRangeSet{CoreRange{CoreCoord{0, 0}}},
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = 0,
            .data_format = tt::DataFormat::Float16_b,
            .page_size = 2048,
        }},
        .buffer = small_buf.get(),
    });
    desc.kernels = {MakeBlankReaderKernel({0, 0})};
    Program program{desc};
    EXPECT_EQ(program.circular_buffers()[0]->size(), 2048u);

    // New: total_size=4096 backed by a larger 4096-byte buffer.  If the patcher
    // calls UpdateDynamicCircularBufferAddress (rebinds to large_buf, but does
    // not refresh max_size_) and then UpdateCircularBufferTotalSize(4096), the
    // latter trips the cached max_size_ (still 2048 from small_buf).  The
    // combined Update*AndTotalSize path resets max_size_ from large_buf.
    auto large_buf = MakeL1Buffer(device(), /*size=*/4096);
    desc.cbs[0].buffer = large_buf.get();
    desc.cbs[0].total_size = 4096;
    EXPECT_NO_THROW(apply_descriptor_runtime_args(program, desc));

    EXPECT_EQ(program.circular_buffers()[0]->size(), 4096u);
}

}  // namespace
}  // namespace tt::tt_metal
