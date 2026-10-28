// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Local invariants of DFB borrowed memory: L1 tensor backing without per-node allocation,
// raw-allocation backing, node coverage, capacity, and borrowing offsets.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <tt_stl/fmt.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "distributed/mesh_device_impl.hpp"

#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"
#include "metal2_host_api/program_spec/validation/validate_spec.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeBorrowedDFBProgramSpec;
using test_helpers::MakeNdShardedTensorParameter;
using test_helpers::MakePerNodeTensorParameter;
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

TEST_F(ProgramSpecTestQuasar, CPU_BorrowedMemoryDFBPerNodeTensorParameterFails) {
    ProgramSpec spec = MakeBorrowedDFBProgramSpec();
    spec.tensor_parameters = {MakePerNodeTensorParameter("borrowed_tensor", /*num_cores=*/1)};
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("which is per-node allocated")));
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

TEST_F(ProgramSpecTestQuasar, RawAllocationDFBExactFitAtZeroOffset) {
    // `entry_size * num_entries` may equal the allocation’s per-node capacity.
    const uint32_t alignment = mesh_device_->allocator()->get_alignment(BufferType::L1);
    auto spec = MakeBorrowedDFBProgramSpec("tensor", BufferType::L1, alignment, 2, false);
    spec.tensor_parameters.clear();
    spec.dataflow_buffers[0].borrowed_from = RawAllocationParamName{"arena"};
    spec.dataflow_buffers[0].advanced_options.borrowed_memory_offset = 0;
    spec.advanced_options.raw_allocation_parameters = {RawAllocationParameter{
        .unique_id = RawAllocationParamName{"arena"},
        .properties =
            RangeLockstepAllocationProperties{
                .banks_to_be_allocated_on = NodeRangeSet{NodeRange{NodeCoord{0, 0}, NodeCoord{0, 0}}},
                .per_bank_size = 2 * alignment,
            },
    }};

    EXPECT_NO_THROW({
        const auto collected = CollectSpecData(spec);
        ValidateProgramSpec(spec, collected, mesh_device_->impl().metal_context(), *mesh_device_->allocator());
    });
}

TEST_F(ProgramSpecTestQuasar, RawAllocationDFBExactFitAtNonzeroOffset) {
    // An aligned offset is allowed, and `offset + entry_size * num_entries` may equal capacity.
    const uint32_t alignment = mesh_device_->allocator()->get_alignment(BufferType::L1);
    auto spec = MakeBorrowedDFBProgramSpec("tensor", BufferType::L1, alignment, 2, false);
    spec.tensor_parameters.clear();
    spec.dataflow_buffers[0].borrowed_from = RawAllocationParamName{"arena"};
    spec.dataflow_buffers[0].advanced_options.borrowed_memory_offset = alignment;
    spec.advanced_options.raw_allocation_parameters = {RawAllocationParameter{
        .unique_id = RawAllocationParamName{"arena"},
        .properties =
            RangeLockstepAllocationProperties{
                .banks_to_be_allocated_on = NodeRangeSet{NodeRange{NodeCoord{0, 0}, NodeCoord{0, 0}}},
                .per_bank_size = 3 * alignment,
            },
    }};

    EXPECT_NO_THROW({
        const auto collected = CollectSpecData(spec);
        ValidateProgramSpec(spec, collected, mesh_device_->impl().metal_context(), *mesh_device_->allocator());
    });
}

TEST_F(ProgramSpecTestQuasar, RawAllocationDFBExtraAllocationNodes) {
    // The allocation may cover more nodes than the DFB occupies.
    const uint32_t alignment = mesh_device_->allocator()->get_alignment(BufferType::L1);
    auto spec = MakeBorrowedDFBProgramSpec("tensor", BufferType::L1, alignment, 2, false);
    spec.tensor_parameters.clear();
    spec.dataflow_buffers[0].borrowed_from = RawAllocationParamName{"arena"};
    spec.dataflow_buffers[0].advanced_options.borrowed_memory_offset = 0;
    spec.advanced_options.raw_allocation_parameters = {RawAllocationParameter{
        .unique_id = RawAllocationParamName{"arena"},
        .properties =
            RangeLockstepAllocationProperties{
                .banks_to_be_allocated_on = NodeRangeSet{NodeRange{NodeCoord{0, 0}, NodeCoord{1, 0}}},
                .per_bank_size = 2 * alignment,
            },
    }};

    EXPECT_NO_THROW({
        const auto collected = CollectSpecData(spec);
        ValidateProgramSpec(spec, collected, mesh_device_->impl().metal_context(), *mesh_device_->allocator());
    });
}

TEST_F(ProgramSpecTestQuasar, RawAllocationDFBMissingOccupiedNode) {
    // The allocation must cover every node occupied by the DFB.
    const uint32_t alignment = mesh_device_->allocator()->get_alignment(BufferType::L1);
    auto spec = MakeBorrowedDFBProgramSpec("tensor", BufferType::L1, alignment, 2, false);
    spec.tensor_parameters.clear();
    spec.dataflow_buffers[0].borrowed_from = RawAllocationParamName{"arena"};
    spec.dataflow_buffers[0].advanced_options.borrowed_memory_offset = 0;
    spec.advanced_options.raw_allocation_parameters = {RawAllocationParameter{
        .unique_id = RawAllocationParamName{"arena"},
        .properties =
            RangeLockstepAllocationProperties{
                .banks_to_be_allocated_on = NodeRangeSet{NodeRange{NodeCoord{0, 0}, NodeCoord{0, 0}}},
                .per_bank_size = 2 * alignment,
            },
    }};
    spec.work_units[0].target_nodes = NodeRange{NodeCoord{0, 0}, NodeCoord{1, 0}};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
            "DFB 'dfb' borrows from RawAllocationParameter 'arena', which does not cover every occupied node.")));
}

TEST_F(ProgramSpecTestQuasar, RawAllocationDFBExceedsCapacity) {
    // Capacity must accommodate all entries, rather than just one entry.
    const uint32_t alignment = mesh_device_->allocator()->get_alignment(BufferType::L1);
    auto spec = MakeBorrowedDFBProgramSpec("tensor", BufferType::L1, alignment, 2, false);
    spec.tensor_parameters.clear();
    spec.dataflow_buffers[0].borrowed_from = RawAllocationParamName{"arena"};
    spec.dataflow_buffers[0].advanced_options.borrowed_memory_offset = 0;
    spec.advanced_options.raw_allocation_parameters = {RawAllocationParameter{
        .unique_id = RawAllocationParamName{"arena"},
        .properties =
            RangeLockstepAllocationProperties{
                .banks_to_be_allocated_on = NodeRangeSet{NodeRange{NodeCoord{0, 0}, NodeCoord{0, 0}}},
                .per_bank_size = 2 * alignment - 1,
            },
    }};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(fmt::format(
            "DFB 'dfb' borrowing offset {} + size_per_node {} exceeds RawAllocationParameter 'arena' per_bank_size {}.",
            0,
            2 * alignment,
            2 * alignment - 1))));
}

TEST_F(ProgramSpecTestQuasar, RawAllocationDFBOffsetCausesOverrun) {
    // The capacity check includes the offset, even when all entries fit without it.
    const uint32_t alignment = mesh_device_->allocator()->get_alignment(BufferType::L1);
    auto spec = MakeBorrowedDFBProgramSpec("tensor", BufferType::L1, alignment, 2, false);
    spec.tensor_parameters.clear();
    spec.dataflow_buffers[0].borrowed_from = RawAllocationParamName{"arena"};
    spec.dataflow_buffers[0].advanced_options.borrowed_memory_offset = alignment;
    spec.advanced_options.raw_allocation_parameters = {RawAllocationParameter{
        .unique_id = RawAllocationParamName{"arena"},
        .properties =
            RangeLockstepAllocationProperties{
                .banks_to_be_allocated_on = NodeRangeSet{NodeRange{NodeCoord{0, 0}, NodeCoord{0, 0}}},
                .per_bank_size = 3 * alignment - 1,
            },
    }};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(fmt::format(
            "DFB 'dfb' borrowing offset {} + size_per_node {} exceeds RawAllocationParameter 'arena' per_bank_size {}.",
            alignment,
            2 * alignment,
            3 * alignment - 1))));
}

TEST_F(ProgramSpecTestQuasar, RawAllocationDFBMisalignedOffset) {
    // The borrowing offset must satisfy L1 alignment, even when capacity is sufficient.
    const uint32_t alignment = mesh_device_->allocator()->get_alignment(BufferType::L1);
    ASSERT_GT(alignment, 1u);
    auto spec = MakeBorrowedDFBProgramSpec("tensor", BufferType::L1, alignment, 2, false);
    spec.tensor_parameters.clear();
    spec.dataflow_buffers[0].borrowed_from = RawAllocationParamName{"arena"};
    spec.dataflow_buffers[0].advanced_options.borrowed_memory_offset = 1;
    spec.advanced_options.raw_allocation_parameters = {RawAllocationParameter{
        .unique_id = RawAllocationParamName{"arena"},
        .properties =
            RangeLockstepAllocationProperties{
                .banks_to_be_allocated_on = NodeRangeSet{NodeRange{NodeCoord{0, 0}, NodeCoord{0, 0}}},
                .per_bank_size = 4 * alignment,
            },
    }};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
            fmt::format("DFB 'dfb' borrowing offset 1 must be a multiple of L1 alignment {}.", alignment))));
}

TEST_F(ProgramSpecTestQuasar, RawAllocationDFBAllBankLockstep) {
    // DFB borrowing accepts the all-bank property alternative as well as range-lockstep.
    const uint32_t alignment = mesh_device_->allocator()->get_alignment(BufferType::L1);
    auto spec = MakeBorrowedDFBProgramSpec("tensor", BufferType::L1, alignment, 2, false);
    spec.tensor_parameters.clear();
    spec.dataflow_buffers[0].borrowed_from = RawAllocationParamName{"arena"};
    spec.dataflow_buffers[0].advanced_options.borrowed_memory_offset = 0;
    spec.advanced_options.raw_allocation_parameters = {RawAllocationParameter{
        .unique_id = RawAllocationParamName{"arena"},
        .properties = LockstepAllocationProperties{.per_bank_size = 2 * alignment},
    }};

    EXPECT_NO_THROW({
        const auto collected = CollectSpecData(spec);
        ValidateProgramSpec(spec, collected, mesh_device_->impl().metal_context(), *mesh_device_->allocator());
    });
}

TEST_F(ProgramSpecTestQuasar, RawAllocationDFBOffsetWithoutBorrowing) {
    // A DFB with no backing-memory reference must have zero borrowing offset.
    const uint32_t alignment = mesh_device_->allocator()->get_alignment(BufferType::L1);
    auto spec = MakeBorrowedDFBProgramSpec("tensor", BufferType::L1, alignment, 2, false);
    spec.tensor_parameters.clear();
    spec.dataflow_buffers[0].borrowed_from.reset();
    spec.dataflow_buffers[0].advanced_options.borrowed_memory_offset = alignment;

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("DFB 'dfb' borrowed_memory_offset must be zero when borrowed_from is unset.")));
}

TEST_F(ProgramSpecTestQuasar, RawAllocationDFBOffsetWithTensorBorrowing) {
    // A tensor-backed DFB must have zero borrowing offset, even if a nonzero offset is aligned and fits.
    const uint32_t alignment = mesh_device_->allocator()->get_alignment(BufferType::L1);
    auto spec = MakeBorrowedDFBProgramSpec("tensor", BufferType::L1, alignment, 2, false);
    spec.tensor_parameters = {
        MakeShardedTensorParameter("tensor", Shape{1, 2 * alignment}, {1, 2 * alignment}, 1, Layout::ROW_MAJOR)};
    spec.dataflow_buffers[0].advanced_options.borrowed_memory_offset = alignment;

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
            "DFB 'dfb' borrowed_memory_offset must be zero when borrowing from a TensorParameter.")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
