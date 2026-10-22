// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Local invariants of DataflowBufferSpec::borrowed_from (dataflow_buffer_spec.hpp): the TensorParameter it names is
// L1-resident and large enough to hold the DFB.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeBorrowedDFBProgramSpec;
using test_helpers::MakeNdShardedTensorParameter;
using test_helpers::MakeShardedTensorParameter;
using test_helpers::ProgramSpecTestQuasar;

TEST_F(ProgramSpecTestQuasar, CPU_BorrowedMemoryDFBSucceeds) {
    // Positive baseline: borrowed-memory DFB whose TensorParameter is L1-resident and large enough.
    ProgramSpec spec = MakeBorrowedDFBProgramSpec();
    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_BorrowedMemoryDFBNonL1TensorParameterFails) {
    // DRAM-resident TensorParameter is not a legal borrow source.
    ProgramSpec spec = MakeBorrowedDFBProgramSpec("borrowed_tensor", tt::tt_metal::BufferType::DRAM);

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("is not L1-resident")));
}

TEST_F(ProgramSpecTestQuasar, CPU_BorrowedMemoryDFBOversizedFails) {
    // DFB total bytes exceed the TensorParameter's per-bank allocation. The default parameter is
    // interleaved and a single page -- 1*32*sizeof(bfloat16) = 64 bytes -- so it lands wholly in
    // one bank whatever the bank count, and 128 bytes of DFB (entry_size 64, num_entries 2)
    // overruns. This covers the interleaved branch of the bound, which the sharded cases below
    // do not reach.
    ProgramSpec spec = MakeBorrowedDFBProgramSpec(
        "borrowed_tensor", tt::tt_metal::BufferType::L1, /*dfb_entry_size=*/64, /*dfb_num_entries=*/2);

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("is larger than the per-bank allocation of its borrowed TensorParameter")));
}

TEST_F(ProgramSpecTestQuasar, CPU_BorrowedMemoryDFBShardLargerThanWholeTensorSucceeds) {
    // Regression: a borrowed DFB is sized for ONE shard, so it must be validated against the
    // backing buffer's per-bank allocation -- not the tensor's packed size, which is whole-tensor
    // and unpadded. A row-major sharded tensor pads on width only, so a 1x32 bf16 tensor with a
    // 32x32 shard on one core packs to 64 bytes while allocating 32 * 64 = 2048 bytes per bank.
    // Sizing the DFB at the shard (the convention every sharded op follows) used to be rejected
    // as "larger than its borrowed TensorParameter (64 bytes)".
    ProgramSpec spec = MakeBorrowedDFBProgramSpec(
        "borrowed_tensor", tt::tt_metal::BufferType::L1, /*dfb_entry_size=*/64, /*dfb_num_entries=*/32);
    spec.tensor_parameters = {MakeShardedTensorParameter(
        "borrowed_tensor",
        tt::tt_metal::Shape{1, 32},
        {32, 32},
        /*num_cores=*/1,
        tt::tt_metal::Layout::ROW_MAJOR)};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_BorrowedMemoryDFBLargerThanShardStillFails) {
    // Companion to the above: widening the bound to the per-bank allocation must not disarm the
    // check. Same tensor (2048 bytes per bank), but a DFB of 64 * 64 = 4096 bytes still overruns.
    ProgramSpec spec = MakeBorrowedDFBProgramSpec(
        "borrowed_tensor", tt::tt_metal::BufferType::L1, /*dfb_entry_size=*/64, /*dfb_num_entries=*/64);
    spec.tensor_parameters = {MakeShardedTensorParameter(
        "borrowed_tensor",
        tt::tt_metal::Shape{1, 32},
        {32, 32},
        /*num_cores=*/1,
        tt::tt_metal::Layout::ROW_MAJOR)};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("is larger than the per-bank allocation of its borrowed TensorParameter")));
}

TEST_F(ProgramSpecTestQuasar, CPU_BorrowedMemoryDFBNdShardLargerThanWholeTensorSucceeds) {
    // compute_consumed_memory_bytes_per_bank has a THIRD branch, for specs built from an
    // NdShardSpec (max_num_dev_pages_per_core) rather than a 2D ShardSpec. It over-covers the
    // logical data the same way, so it needs its own regression alongside the 2D case above.
    ProgramSpec spec = MakeBorrowedDFBProgramSpec(
        "borrowed_tensor", tt::tt_metal::BufferType::L1, /*dfb_entry_size=*/64, /*dfb_num_entries=*/32);
    spec.tensor_parameters = {MakeNdShardedTensorParameter(
        "borrowed_tensor", tt::tt_metal::Shape{1, 32}, tt::tt_metal::Shape{32, 32}, /*num_cores=*/1)};

    // Without these the test can silently re-cover the 2D case or assert nothing at all: the ND
    // branch is only reached when the spec keeps no 2D shard_spec (see MakeNdShardedTensorParameter
    // on why CONTIGUOUS_1D is what guarantees that), and the bound only has teeth when the shard
    // allocates more than the tensor packs.
    const tt::tt_metal::TensorSpec& tensor_spec = spec.tensor_parameters[0].spec;
    ASSERT_FALSE(tensor_spec.memory_config().shard_spec().has_value());
    const auto& allocator = mesh_device_->allocator();
    ASSERT_GT(
        tensor_spec.compute_consumed_memory_bytes_per_bank(
            allocator->get_alignment(tt::tt_metal::BufferType::L1),
            allocator->get_num_banks(tt::tt_metal::BufferType::L1)),
        tensor_spec.compute_packed_buffer_size_bytes());

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
