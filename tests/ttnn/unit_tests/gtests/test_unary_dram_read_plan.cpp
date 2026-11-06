// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Flow choice of get_dram_read_plan for unary ops on DRAM tensors. Host only: the plan depends on the tensor specs,
// the op chain and the core and DRAM bank counts, so both chip families are checked on any machine. The device tests
// (test_unary_dram_reads.py) check that every flow gives the default path's output.

#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/eltwise/unary/common/unary_utils.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::unary::test {
namespace {

using ::tt::tt_metal::BufferType;
using ::tt::tt_metal::DataType;
using ::tt::tt_metal::Layout;
using ::tt::tt_metal::MemoryConfig;
using ::tt::tt_metal::PageConfig;
using ::tt::tt_metal::ShardOrientation;
using ::tt::tt_metal::ShardSpec;
using ::tt::tt_metal::TensorLayout;
using ::tt::tt_metal::TensorMemoryLayout;
using ::tt::tt_metal::TensorSpec;

struct Chip {
    uint32_t cores;
    uint32_t banks;
};
// Many cores per DRAM bank (Blackhole p300a) and few (Wormhole).
constexpr Chip kManyCoresPerBank{110, 8};
constexpr Chip kFewCoresPerBank{72, 12};

constexpr uint32_t kTile = 32;

TensorSpec make_spec(const ttnn::Shape& shape, const MemoryConfig& memory_config, DataType dtype) {
    return TensorSpec(shape, TensorLayout(dtype, PageConfig(Layout::TILE), memory_config));
}

// tiles pages, one tile wide.
TensorSpec interleaved(uint32_t tiles, DataType dtype = DataType::BFLOAT16) {
    return make_spec(ttnn::Shape{1, 1, tiles * kTile, kTile}, ttnn::DRAM_MEMORY_CONFIG, dtype);
}

// shards height shards of shard_rows tiles, one tile wide.
TensorSpec height_shards(uint32_t shards, uint32_t shard_rows, DataType dtype = DataType::BFLOAT16) {
    const MemoryConfig memory_config(
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::DRAM,
        ShardSpec(
            CoreRangeSet(CoreRange({0, 0}, {shards - 1, 0})),
            {shard_rows * kTile, kTile},
            ShardOrientation::ROW_MAJOR));
    return make_spec(ttnn::Shape{1, 1, shards * shard_rows * kTile, kTile}, memory_config, dtype);
}

// shards width shards of shard_width tiles, rows tiles high.
TensorSpec width_shards(uint32_t shards, uint32_t rows, uint32_t shard_width) {
    const MemoryConfig memory_config(
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::DRAM,
        ShardSpec(
            CoreRangeSet(CoreRange({0, 0}, {shards - 1, 0})),
            {rows * kTile, shard_width * kTile},
            ShardOrientation::ROW_MAJOR));
    return make_spec(ttnn::Shape{1, 1, rows * kTile, shards * shard_width * kTile}, memory_config, DataType::BFLOAT16);
}

DramReadPlan plan(
    const std::vector<EltwiseUnaryWithParam>& op_chain,
    const TensorSpec& input,
    const TensorSpec& output,
    Chip chip,
    uint32_t num_cores) {
    return get_dram_read_plan(op_chain, input, output, num_cores, chip.cores, chip.banks);
}

// One op on the whole grid, output like the input.
DramReadPlan plan(UnaryOpType op, const TensorSpec& spec, Chip chip) {
    return plan({EltwiseUnaryWithParam{op}}, spec, spec, chip, chip.cores);
}

DramReadFlow flow(UnaryOpType op, const TensorSpec& spec, Chip chip) { return plan(op, spec, chip).flow; }

constexpr auto kNone = DramReadFlow::None;
constexpr auto kBurst = DramReadFlow::StaticBurst;
constexpr auto kOnePage = DramReadFlow::StaticOnePage;
constexpr auto kQueue = DramReadFlow::WorkQueue;

TEST(UnaryDramReadPlan, ManyCoresPerBankGates) {
    const Chip chip = kManyCoresPerBank;
    // Interleaved light ops: bursts from 65 pages per core, the queue from 128.
    EXPECT_EQ(flow(UnaryOpType::RELU, interleaved(chip.cores * 64), chip), kNone);
    EXPECT_EQ(flow(UnaryOpType::RELU, interleaved(chip.cores * 65), chip), kBurst);
    EXPECT_EQ(flow(UnaryOpType::RELU, interleaved(chip.cores * 128), chip), kQueue);
    // Mid ops keep the default path with every bank serving the tensor.
    EXPECT_EQ(flow(UnaryOpType::SILU, interleaved(chip.cores * 200), chip), kNone);
    // Height shards over part of the banks: the queue from 40 pages per core (7 * 628 / 110 = 39, 7 * 629 / 110 = 40),
    // mid ops from 32.
    EXPECT_EQ(flow(UnaryOpType::RELU, height_shards(7, 628), chip), kNone);
    EXPECT_EQ(flow(UnaryOpType::RELU, height_shards(7, 629), chip), kQueue);
    EXPECT_EQ(flow(UnaryOpType::SILU, height_shards(7, 629), chip), kQueue);
    // Shards on every bank: bursts from 96 pages per core, the queue from 128 (8 * 1375 / 110 = 100,
    // 8 * 1800 / 110 = 130).
    EXPECT_EQ(flow(UnaryOpType::RELU, height_shards(8, 1375), chip), kBurst);
    EXPECT_EQ(flow(UnaryOpType::RELU, height_shards(8, 1800), chip), kQueue);
    // Width shards over part of the banks: the queue from 73 pages per core (224 * 35 / 110 = 71, 224 * 36 / 110 = 73).
    EXPECT_EQ(flow(UnaryOpType::RELU, width_shards(7, 35, 32), chip), kNone);
    EXPECT_EQ(flow(UnaryOpType::RELU, width_shards(7, 36, 32), chip), kQueue);
}

TEST(UnaryDramReadPlan, ManyCoresPerBankLargePages) {
    const Chip chip = kManyCoresPerBank;
    // 7 shards of 4715 rows: 300 pages per core; of 8100 rows: 515.
    EXPECT_EQ(flow(UnaryOpType::RELU, height_shards(7, 4715, DataType::FLOAT32), chip), kQueue);
    EXPECT_EQ(flow(UnaryOpType::LOGIT, height_shards(7, 4715, DataType::FLOAT32), chip), kOnePage);
    // From 512 pages per core heavy ops take the queue; logit's and logsigmoid's kernels cannot run it, so bursts.
    EXPECT_EQ(flow(UnaryOpType::LOGIT, height_shards(7, 8100, DataType::FLOAT32), chip), kBurst);
    EXPECT_EQ(flow(UnaryOpType::LOGSIGMOID, height_shards(7, 8100, DataType::FLOAT32), chip), kBurst);
}

TEST(UnaryDramReadPlan, FewCoresPerBankGates) {
    const Chip chip = kFewCoresPerBank;
    // Interleaved light ops: the queue from 96 pages per core.
    EXPECT_EQ(flow(UnaryOpType::RELU, interleaved(chip.cores * 95), chip), kNone);
    EXPECT_EQ(flow(UnaryOpType::RELU, interleaved(chip.cores * 96), chip), kQueue);
    // Height shards over part of the banks: bursts from 24 pages per core and 384 per bank, the queue from 96.
    EXPECT_EQ(flow(UnaryOpType::RELU, height_shards(6, 384), chip), kBurst);
    EXPECT_EQ(flow(UnaryOpType::RELU, height_shards(6, 1152), chip), kQueue);
    // The rule was measured on the whole grid.
    const TensorSpec part_grid = interleaved(64 * 200);
    EXPECT_EQ(plan({EltwiseUnaryWithParam{UnaryOpType::RELU}}, part_grid, part_grid, chip, 64).flow, kNone);
}

TEST(UnaryDramReadPlan, FewCoresPerBankMidOpsInterleaved) {
    const Chip chip = kFewCoresPerBank;
    // 48 of 72 ranges are 459 pages, 459 apart: most cores start on 4 banks and the queue wins.
    EXPECT_EQ(flow(UnaryOpType::SILU, interleaved(chip.cores * 458 + 48), chip), kQueue);
    // Half the ranges or fewer are longer: their starts spread differently (one 459, 71 of 458 start on 7 banks), and
    // the rule was not measured there.
    EXPECT_EQ(flow(UnaryOpType::SILU, interleaved(chip.cores * 458 + 36), chip), kNone);
    EXPECT_EQ(flow(UnaryOpType::SILU, interleaved(chip.cores * 458 + 1), chip), kNone);
    // Every range starts on one bank, outside the measured 3 to 4.
    EXPECT_EQ(flow(UnaryOpType::SILU, interleaved(chip.cores * 912), chip), kNone);
}

TEST(UnaryDramReadPlan, FewCoresPerBankWidthShards) {
    const Chip chip = kFewCoresPerBank;
    // 32-page shards on every bank, 384-page rows. At 192 pages per core the default split's cores start in 2 of the 12
    // shards, so even very heavy ops take the queue.
    EXPECT_EQ(flow(UnaryOpType::GELU, width_shards(12, 36, 32), chip), kQueue);
    // At 64 pages per core they start in 6: heavy ops take the queue, very heavy ones keep the default path.
    EXPECT_EQ(flow(UnaryOpType::ELU, width_shards(12, 12, 32), chip), kQueue);
    EXPECT_EQ(flow(UnaryOpType::GELU, width_shards(12, 12, 32), chip), kNone);
}

TEST(UnaryDramReadPlan, FewCoresPerBankCosts) {
    const Chip chip = kFewCoresPerBank;
    // On interleaved tensors only light ops take the queue, and a chain of two light ops counts as moderate.
    const TensorSpec tensor = interleaved(chip.cores * 200);
    const std::vector<EltwiseUnaryWithParam> two_ops = {
        EltwiseUnaryWithParam{UnaryOpType::RELU}, EltwiseUnaryWithParam{UnaryOpType::RELU}};
    EXPECT_EQ(flow(UnaryOpType::RELU, tensor, chip), kQueue);
    EXPECT_EQ(plan(two_ops, tensor, tensor, chip, chip.cores).flow, kNone);
    // With 4 KB pages relu stays light, and softplus counts as very heavy.
    EXPECT_EQ(flow(UnaryOpType::RELU, interleaved(chip.cores * 200, DataType::FLOAT32), chip), kQueue);
    EXPECT_EQ(flow(UnaryOpType::SOFTPLUS, interleaved(chip.cores * 200, DataType::FLOAT32), chip), kNone);
}

TEST(UnaryDramReadPlan, KeepsDefaultPath) {
    // With many cores per bank heavy ops and softplus keep the default path, also on shards over part of the banks.
    EXPECT_EQ(flow(UnaryOpType::ELU, height_shards(7, 4715), kManyCoresPerBank), kNone);
    EXPECT_EQ(flow(UnaryOpType::SOFTPLUS, height_shards(7, 4715), kManyCoresPerBank), kNone);
    for (const Chip chip : {kManyCoresPerBank, kFewCoresPerBank}) {
        const TensorSpec tensor = interleaved(chip.cores * 200);
        // Very heavy ops on interleaved tensors, and ops with no measured cost.
        EXPECT_EQ(flow(UnaryOpType::LOGIT, tensor, chip), kNone);
        EXPECT_EQ(flow(UnaryOpType::LGAMMA, tensor, chip), kNone);
        // Pages under 2 KB.
        EXPECT_EQ(flow(UnaryOpType::RELU, interleaved(chip.cores * 200, DataType::BFLOAT8_B), chip), kNone);
        // Input and output in different layouts.
        EXPECT_EQ(
            plan(
                {EltwiseUnaryWithParam{UnaryOpType::RELU}}, tensor, height_shards(8, chip.cores * 25), chip, chip.cores)
                .flow,
            kNone);
    }
}

TEST(UnaryDramReadPlan, WorkQueueAnyWorkerCount) {
    // The scheduler keeps its per-worker state in L1, sized by the grid: no cap on the worker count.
    EXPECT_EQ(flow(UnaryOpType::RELU, interleaved(200 * 200), Chip{200, 8}), kQueue);
    EXPECT_EQ(flow(UnaryOpType::RELU, interleaved(150 * 200), Chip{150, 20}), kQueue);
}

TEST(UnaryDramReadPlan, WorkQueueChunks) {
    // A chunk is 1/150 of a core's share, 16 to 256 pages.
    const DramReadPlan many = plan(UnaryOpType::RELU, interleaved(kManyCoresPerBank.cores * 15000), kManyCoresPerBank);
    ASSERT_EQ(many.flow, kQueue);
    EXPECT_EQ(many.chunk_pages, 100u);
    EXPECT_EQ(read_burst(many.flow), 8u);
    EXPECT_EQ(write_burst(many.flow), 2u);
    // With few cores per bank a chunk that is a multiple of the banks grows by a page: 2700 / 150 = 18 on 6 shards.
    const DramReadPlan few = plan(UnaryOpType::RELU, height_shards(6, 32400), kFewCoresPerBank);
    ASSERT_EQ(few.flow, kQueue);
    EXPECT_EQ(few.chunk_pages, 19u);
}

}  // namespace
}  // namespace ttnn::operations::unary::test
