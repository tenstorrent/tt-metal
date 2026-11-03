// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The codegen scatter prim takes an optional preallocated output that no public entry exposes, so
// its contract on that argument is only reachable from C++: the destination must be the tensor the
// op would create for itself (standard padding included) and must not be one of the operands the
// kernels still read. Likewise the shape relations ttnn::scatter establishes before calling (an
// index no wider than the input on any leading axis) are only testable against the prim directly.

#include <gtest/gtest.h>

#include <cstdint>
#include <exception>
#include <optional>
#include <vector>

#include <tt-metalium/allocator.hpp>

#include "ttnn/device.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/scatter/codegen/scatter_codegen_device_operation.hpp"
#include "ttnn/operations/data_movement/scatter/codegen/scatter_codegen_program_factory.hpp"
#include "ttnn/operations/data_movement/scatter/codegen/scatter_codegen_supported.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::data_movement::scatter::test {

class ScatterCodegenPrimFixture : public TTNNFixtureWithSuiteDevice<ScatterCodegenPrimFixture> {};

namespace {

struct Operands {
    Tensor input;
    Tensor index;
    Tensor src;
};

// bf16 TILE operands with the scatter axis last and a 32x32 index unique along it, the shape every
// TILE factory serves. `width` is the input's last extent; index and src share the input's rows.
Operands make_operands(tt::tt_metal::distributed::MeshDevice* device, uint32_t width) {
    const ttnn::Shape input_shape({1, 1, 32, width});
    const ttnn::Shape index_shape({1, 1, 32, 32});
    std::vector<int32_t> index(32 * 32);
    for (uint32_t r = 0; r < 32; ++r) {
        for (uint32_t c = 0; c < 32; ++c) {
            index[r * 32 + c] = static_cast<int32_t>((c * (width / 32) + r) % width);
        }
    }
    const tt::tt_metal::TensorSpec index_spec(
        index_shape,
        tt::tt_metal::TensorLayout(
            DataType::INT32, tt::tt_metal::PageConfig(Layout::TILE), tt::tt_metal::MemoryConfig{}));
    return {
        ttnn::ones(input_shape, DataType::BFLOAT16, Layout::TILE, *device),
        Tensor::from_vector(index, index_spec, device),
        ttnn::zeros(index_shape, DataType::BFLOAT16, Layout::TILE, *device),
    };
}

Tensor run(const Operands& ops, const std::optional<Tensor>& output) {
    return ttnn::prim::scatter_codegen(
        ops.input, ops.index, ops.src, kReductionNone, ops.input.memory_config(), output, std::nullopt);
}

// Index and src of `shape` (scatter axis last, values within `width`) against a bf16 TILE input of
// `input_shape`.
Operands make_operands_of(
    tt::tt_metal::distributed::MeshDevice* device,
    const ttnn::Shape& input_shape,
    const ttnn::Shape& shape,
    uint32_t width) {
    std::vector<int32_t> index(shape.volume());
    for (size_t i = 0; i < index.size(); ++i) {
        index[i] = static_cast<int32_t>(i % width);
    }
    const tt::tt_metal::TensorSpec index_spec(
        shape,
        tt::tt_metal::TensorLayout(
            DataType::INT32, tt::tt_metal::PageConfig(Layout::TILE), tt::tt_metal::MemoryConfig{}));
    return {
        ttnn::ones(input_shape, DataType::BFLOAT16, Layout::TILE, *device),
        Tensor::from_vector(index, index_spec, device),
        ttnn::zeros(shape, DataType::BFLOAT16, Layout::TILE, *device),
    };
}

}  // namespace

TEST_F(ScatterCodegenPrimFixture, RejectsAnIndexWiderThanTheInputOnALeadingAxis) {
    // The readers walk the input's pages and clip each page coordinate against the index's extent,
    // so index pages past the input's extent on a leading axis are never visited: without the check
    // the call returns a well-formed result with those updates silently dropped. A narrower index
    // is served, so only the wider one is refused.
    const Operands wider = make_operands_of(device_, ttnn::Shape({1, 1, 32, 64}), ttnn::Shape({1, 2, 32, 32}), 64);
    EXPECT_THROW(run(wider, std::nullopt), std::exception);
    const Operands lower_rank = make_operands_of(device_, ttnn::Shape({1, 1, 32, 64}), ttnn::Shape({32, 32}), 64);
    EXPECT_THROW(run(lower_rank, std::nullopt), std::exception);
    const Operands narrower = make_operands_of(device_, ttnn::Shape({1, 2, 32, 64}), ttnn::Shape({1, 1, 32, 32}), 64);
    EXPECT_NO_THROW(run(narrower, std::nullopt));
}

TEST_F(ScatterCodegenPrimFixture, AcceptsTheDestinationItWouldCreateItself) {
    const Operands ops = make_operands(device_, 64);
    Tensor destination = ttnn::create_device_tensor(ops.input.tensor_spec(), device_);
    const Tensor out = run(ops, destination);
    EXPECT_EQ(out.buffer(), destination.buffer());
}

TEST_F(ScatterCodegenPrimFixture, RejectsADestinationPaddedPastTheStandardGeometry) {
    const Operands ops = make_operands(device_, 64);
    // Same logical shape as the input, but tilized with twice the padded rows: the TILE writers
    // compute page ids from the input's tile-rounded geometry, so these pages would not line up.
    const Tensor host = ttnn::zeros(ttnn::Shape({1, 1, 32, 64}), DataType::BFLOAT16, Layout::ROW_MAJOR, *device_);
    const Tensor padded = ttnn::tilize_with_val_padding(host, ttnn::Shape({1, 1, 64, 64}), ttnn::PadValue{0.0f});
    ASSERT_EQ(padded.logical_shape(), ops.input.logical_shape());
    ASSERT_EQ(padded.padded_shape(), ttnn::Shape({1, 1, 64, 64}));
    EXPECT_FALSE(supported_execution_controls(ops.input, ops.input.memory_config(), padded));
    EXPECT_THROW(run(ops, padded), std::exception);
}

TEST_F(ScatterCodegenPrimFixture, RejectsADestinationThatAliasesSrc) {
    // With src the input's own shape, src is a spec-exact destination; aliasing it would let the
    // writers overwrite source values the streaming readers have not consumed yet.
    const Operands ops = make_operands(device_, 32);
    ASSERT_EQ(ops.src.tensor_spec(), ops.input.tensor_spec());
    EXPECT_THROW(run(ops, ops.src), std::exception);
    EXPECT_NO_THROW(run(ops, std::nullopt));
}

TEST_F(ScatterCodegenPrimFixture, RejectsADestinationOfAnotherShapeDtypeLayoutOrTile) {
    // Every factory sizes its page count and per-core split from the input's geometry, so only the
    // spec the op would create for itself is a destination it can address.
    const Operands ops = make_operands(device_, 64);
    const auto spec_with = [&](const ttnn::Shape& shape, DataType dtype, const tt::tt_metal::PageConfig& page) {
        return tt::tt_metal::TensorSpec(shape, tt::tt_metal::TensorLayout(dtype, page, ops.input.memory_config()));
    };
    const tt::tt_metal::PageConfig tile_page(Layout::TILE);
    const std::vector<tt::tt_metal::TensorSpec> wrong = {
        spec_with(ttnn::Shape({1, 1, 32, 128}), DataType::BFLOAT16, tile_page),
        spec_with(ops.input.logical_shape(), DataType::FLOAT32, tile_page),
        spec_with(ops.input.logical_shape(), DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::ROW_MAJOR)),
        spec_with(
            ops.input.logical_shape(),
            DataType::BFLOAT16,
            tt::tt_metal::PageConfig(Layout::TILE, tt::tt_metal::Tile({16, 32}))),
    };
    for (const auto& spec : wrong) {
        const Tensor destination = ttnn::create_device_tensor(spec, device_);
        EXPECT_FALSE(supported_execution_controls(ops.input, ops.input.memory_config(), destination));
        EXPECT_THROW(run(ops, destination), std::exception);
    }
}

TEST_F(ScatterCodegenPrimFixture, RejectsTransposedTilesOnAProgramCacheHit) {
    // A Tile's attribute values carry its shape and faces but not its transpose flags, so operands in
    // transposed 32x32 tiles hash like default-tile ones: the second call below hits the first
    // call's program, and only the hit-time checks keep its tiles off the fixed-face readers.
    device_->enable_program_cache();
    const Operands plain = make_operands(device_, 64);
    EXPECT_NO_THROW(run(plain, std::nullopt));
    const tt::tt_metal::Tile transposed({32, 32}, /*transpose_tile=*/true);
    const auto with_tile = [&](const Tensor& t) {
        const tt::tt_metal::TensorSpec spec(
            t.logical_shape(),
            tt::tt_metal::TensorLayout(
                t.dtype(), tt::tt_metal::PageConfig(Layout::TILE, transposed), t.memory_config()));
        return ttnn::create_device_tensor(spec, device_);
    };
    const Operands transposed_ops{with_tile(plain.input), with_tile(plain.index), with_tile(plain.src)};
    EXPECT_FALSE(
        supported_by_codegen(transposed_ops.input, -1, transposed_ops.index, transposed_ops.src, kReductionNone));
    EXPECT_THROW(run(transposed_ops, std::nullopt), std::exception);
    device_->disable_and_clear_program_cache();
}

TEST_F(ScatterCodegenPrimFixture, TileRouteNeedsItsStreamingPagesInLiveL1) {
    // The streaming plan's six tile pages are the least a TILE call seats, so an L1 that cannot hold
    // them must send the call native rather than fail the program's CB allocation. A 72-wide stick
    // keeps its raw bytes off a 32 B boundary, so the row-major detour, which would otherwise answer
    // for the call, declines it.
    const Operands ops = make_operands_of(device_, ttnn::Shape({1, 1, 32, 72}), ttnn::Shape({1, 1, 32, 32}), 72);
    const auto fits = [&]() {
        return codegen_plan_fits_live_l1(ops.input, -1, ops.index, ops.src, kReductionNone, ops.input.memory_config());
    };
    EXPECT_TRUE(fits());
    const uint64_t streaming_bytes = ttnn::prim::scatter_tile_streaming_bytes(
        ops.input.buffer()->aligned_page_size(),
        ops.input.buffer()->aligned_page_size(),
        ops.index.buffer()->aligned_page_size(),
        ops.src.buffer()->aligned_page_size());
    // One ROW_MAJOR stick per L1 bank, leaving every bank short of those pages.
    const uint32_t banks = device_->allocator()->get_num_banks(tt::tt_metal::BufferType::L1);
    const uint64_t stick_bytes = ttnn::prim::scatter_usable_l1(ops.input) - streaming_bytes / 2;
    const tt::tt_metal::MemoryConfig l1{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1};
    Tensor blocker = ttnn::zeros(
        ttnn::Shape({banks, static_cast<uint32_t>(stick_bytes / sizeof(uint16_t))}),
        DataType::BFLOAT16,
        Layout::ROW_MAJOR,
        *device_,
        l1);
    ASSERT_LT(ttnn::prim::scatter_usable_l1(ops.input), streaming_bytes);
    EXPECT_FALSE(fits());
    blocker.deallocate();
    EXPECT_TRUE(fits());
}

TEST(ScatterCodegenRmChunkPlanner, BudgetsTheRoundedCbPagesOfAnUnalignedChunk) {
    // An index stick of 42 int32 elements with bf16 sources: 252 raw bytes, but the two CB pages are
    // rounded to 16 B each (176 + 96 = 272). A budget between the two must scale the chunk down to
    // the 32-element floor rather than hand the factory pages it cannot allocate.
    constexpr uint32_t kIndexElems = 42, kIndexBytes = 4, kSrcBytes = 2;
    constexpr uint64_t kFixed = 1024;
    EXPECT_EQ(ttnn::prim::scatter_rm_chunk_elems(kFixed + 272, kFixed, kIndexElems, kIndexBytes, kSrcBytes), 42u);
    EXPECT_EQ(ttnn::prim::scatter_rm_chunk_elems(kFixed + 260, kFixed, kIndexElems, kIndexBytes, kSrcBytes), 32u);
}

// A device opened with an L1_SMALL region: the allocator's L1 bank stops below that region, and the
// occupancy query the live budget reads never reports the buffers living in it.
class ScatterCodegenL1SmallFixture : public TTNNFixtureWithDevice {
protected:
    static constexpr int kL1SmallSize = 16384;
    ScatterCodegenL1SmallFixture() : TTNNFixtureWithDevice(DEFAULT_TRACE_REGION_SIZE, kL1SmallSize) {}
};

TEST_F(ScatterCodegenL1SmallFixture, BudgetsStopAtTheL1Bank) {
    // The RM chunk planner fills whatever budget it is given, so a budget past the bank would plan
    // CBs over the L1_SMALL region.
    const Operands ops = make_operands(device_, 64);
    const uint64_t bank = device_->allocator()->get_bank_size(tt::tt_metal::BufferType::L1);
    const uint64_t base = device_->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    ASSERT_LE(bank + kL1SmallSize + base, static_cast<uint64_t>(device_->l1_size_per_core()));
    EXPECT_LE(ttnn::prim::scatter_static_l1(ops.input), bank);
    EXPECT_LE(ttnn::prim::scatter_usable_l1(ops.input), bank);
}

}  // namespace ttnn::operations::data_movement::scatter::test
