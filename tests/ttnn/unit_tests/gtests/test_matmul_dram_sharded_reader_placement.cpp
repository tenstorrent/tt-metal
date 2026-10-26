// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Program-spec tests for the DRAM-sharded matmul's reader placement on a mesh.
//
// Each chip has its own harvesting, so the worker core next to a given DRAM bank differs from chip to chip.
// The factory must place each chip's in1 readers on that chip's own optimal cores (#62443). Placing every chip
// on the first chip's cores still produces correct results, only slower, so running the op does not show the
// bug. These tests read the placement out of the program specs instead.

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/math.hpp>

#include "ttnn/mesh_device_operation_utils.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/types.hpp"
#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace ttnn::prim::test {
namespace {

using ::tt::tt_metal::BufferType;
using ::tt::tt_metal::CoreCoord;
using ::tt::tt_metal::CoreRange;
using ::tt::tt_metal::CoreRangeSet;
using ::tt::tt_metal::DataType;
using ::tt::tt_metal::Layout;
using ::tt::tt_metal::MemoryConfig;
using ::tt::tt_metal::PageConfig;
using ::tt::tt_metal::ShardOrientation;
using ::tt::tt_metal::ShardSpec;
using ::tt::tt_metal::TensorLayout;
using ::tt::tt_metal::TensorMemoryLayout;
using ::tt::tt_metal::TensorSpec;
using ::tt::tt_metal::distributed::MeshCoordinate;
using ::tt::tt_metal::distributed::MeshDevice;
using ::ttnn::device_operation::MeshWorkloadArtifacts;
using ::ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig;
using Factory = MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory;

// A Wormhole served decode shape that both readers handle: Llama-3.2-1B-Instruct's 1024 x 2048 projection
// (test_matmul_dram_sharded_multicore.py), activation on 8 x 2 cores, 4 output tiles per core.
constexpr uint32_t K = 1024;
constexpr uint32_t N = 2048;
constexpr uint32_t TILE = 32;

std::vector<CoreCoord> assignment_by_bank(MeshDevice& device, tt::tt_metal::NOC noc, const MeshCoordinate& coord) {
    const auto by_bank = device.get_optimal_dram_bank_to_logical_worker_assignment(noc, coord);
    std::vector<CoreCoord> cores(by_bank.size());
    for (const auto& [bank, core] : by_bank) {
        cores.at(bank) = core;
    }
    return cores;
}

// Banks holding real columns: N is padded up to whole shards, so the last banks may hold padding only.
uint32_t banks_with_columns(MeshDevice* device) {
    const uint32_t banks = device->dram_grid_size().x;
    const uint32_t shard_tiles = tt::div_up(N / TILE, banks);
    return tt::div_up(N / TILE, shard_tiles);
}

struct Operands {
    MatmulParams attributes;
    MatmulInputs inputs;
    std::vector<Tensor> outputs;
};

Operands make_operands(MeshDevice* device, uint32_t in0_block_w, uint32_t workers_per_bank) {
    const uint32_t banks = device->dram_grid_size().x;
    const uint32_t padded_n = tt::div_up(N, TILE * banks) * TILE * banks;

    const CoreRangeSet in0_grid(CoreRange({0, 0}, {7, 1}));
    const MemoryConfig in0_config(
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::L1,
        ShardSpec(in0_grid, {TILE, K / in0_grid.num_cores()}, ShardOrientation::ROW_MAJOR));
    const CoreRangeSet dram_grid(CoreRange({0, 0}, {banks - 1, 0}));
    const MemoryConfig in1_config(
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::DRAM,
        ShardSpec(dram_grid, {K, padded_n / banks}, ShardOrientation::ROW_MAJOR));

    const Tensor in0 = ttnn::create_device_tensor(
        TensorSpec(ttnn::Shape{1, 1, TILE, K}, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), in0_config)),
        device);
    const Tensor in1 = ttnn::create_device_tensor(
        TensorSpec(ttnn::Shape{1, 1, K, N}, TensorLayout(DataType::BFLOAT8_B, PageConfig(Layout::TILE), in1_config)),
        device);

    MatmulParams parameters{
        .program_config =
            MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig{
                .in0_block_w = in0_block_w,
                .per_core_M = 1,
                .per_core_N = 4,
                .fused_activation = std::nullopt,
                .num_workers_per_dram_bank = workers_per_bank,
                .use_multicore_pipeline = workers_per_bank >= 2,
            },
        .output_mem_config = MemoryConfig(TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1),
        .output_dtype = DataType::BFLOAT16,
        .compute_kernel_config = ttnn::init_device_compute_kernel_config(
            device->arch(), std::nullopt, tt::tt_metal::MathFidelity::HiFi2, false, true, true),
    };
    Operands operands{
        .attributes = create_matmul_attributes(in0, in1, parameters, {std::nullopt}),
        .inputs = MatmulInputs{{in0, in1}, {std::nullopt}, {std::nullopt}},
    };
    operands.outputs = MatmulDeviceOperation::create_output_tensors(operands.attributes, operands.inputs);
    return operands;
}

MeshWorkloadArtifacts build_workload(MeshDevice* device, Operands& operands) {
    ttnn::MeshCoordinateRangeSet coords;
    for (const auto& coord :
         ttnn::device_operation::mesh_device_operation_utils::extract_tensor_coordinates(operands.inputs, device)) {
        coords.merge(ttnn::MeshCoordinateRange(coord));
    }
    return Factory::create_mesh_workload_artifacts(operands.attributes, operands.inputs, operands.outputs, coords);
}

// One program per occupied coordinate, each covering exactly that coordinate.
std::vector<MeshCoordinate> program_coords(MeshDevice* device, const MeshWorkloadArtifacts& artifacts) {
    std::vector<MeshCoordinate> coords;
    std::set<MeshCoordinate> seen;
    for (const auto& program : artifacts.programs) {
        EXPECT_EQ(program.range.shape().mesh_size(), 1u) << "program covers " << program.range;
        const MeshCoordinate coord = program.range.start_coord();
        EXPECT_TRUE(seen.insert(coord).second) << "two programs for " << coord;
        coords.push_back(coord);
    }
    EXPECT_EQ(coords.size(), device->shape().mesh_size());
    return coords;
}

// Whether any two chips of the mesh disagree on a bank's optimal worker. Without that, a placement copied
// from the first chip would pass these checks too.
bool assignments_differ(MeshDevice* device, tt::tt_metal::NOC noc) {
    std::optional<std::vector<CoreCoord>> first;
    for (const auto& coord : ttnn::MeshCoordinateRange(device->shape())) {
        auto cores = assignment_by_bank(*device, noc, coord);
        if (!first) {
            first = std::move(cores);
        } else if (cores != *first) {
            return true;
        }
    }
    return false;
}

class MatmulDramShardedReaderPlacementTest : public tt::tt_metal::GenericMeshDeviceFixture {
protected:
    void SetUp() override {
        GenericMeshDeviceFixture::SetUp();
        if (IsSkipped()) {
            return;
        }
        if (mesh_device_->arch() != tt::ARCH::WORMHOLE_B0) {
            GTEST_SKIP() << "the shape is a Wormhole served shape (12 DRAM banks)";
        }
        if (mesh_device_->shape().mesh_size() < 2) {
            GTEST_SKIP() << "needs a mesh of at least two chips";
        }
        if (!assignments_differ(
                mesh_device_.get(), tt::tt_metal::detail::preferred_noc_for_dram_read(mesh_device_->arch()))) {
            std::cout << "Every chip has the same DRAM bank to worker assignment; the placement checks cannot tell "
                         "per-chip placement from the first chip's placement on this mesh.\n";
        }
    }
};

// The single-reader program: on every chip, the in1 reader of bank b sits on that chip's optimal worker of b.
TEST_F(MatmulDramShardedReaderPlacementTest, StandardReadersFollowEachChipsAssignment) {
    auto* device = mesh_device_.get();
    auto operands = make_operands(device, /*in0_block_w=*/16, /*workers_per_bank=*/1);
    const auto artifacts = build_workload(device, operands);
    const auto coords = program_coords(device, artifacts);
    const auto noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
    const uint32_t banks = banks_with_columns(device);

    for (size_t i = 0; i < artifacts.programs.size(); ++i) {
        const auto expected = assignment_by_bank(*device, noc, coords[i]);
        const auto& run_args = artifacts.programs[i].run_params.kernel_run_args;
        const auto in1 =
            std::ranges::find_if(run_args, [](const auto& args) { return *args.kernel == "in1_sender_writer"; });
        ASSERT_NE(in1, run_args.end()) << coords[i] << ": no in1_sender_writer run args";
        const auto is_worker = in1->runtime_arg_values.find("is_worker_core");
        const auto bank_id = in1->runtime_arg_values.find("dram_bank_id");
        ASSERT_NE(is_worker, in1->runtime_arg_values.end());
        ASSERT_NE(bank_id, in1->runtime_arg_values.end());

        std::set<uint32_t> banks_read;
        for (const auto& [core, worker] : is_worker->second) {
            if (worker == 0) {
                continue;
            }
            const auto bank = bank_id->second.find(core);
            ASSERT_NE(bank, bank_id->second.end()) << coords[i] << ": reader " << core.str() << " has no bank";
            ASSERT_LT(bank->second, expected.size());
            EXPECT_EQ(core, expected[bank->second])
                << coords[i] << ": bank " << bank->second << " is read from " << core.str() << ", this chip's "
                << "optimal worker is " << expected[bank->second].str();
            EXPECT_TRUE(banks_read.insert(bank->second).second) << coords[i] << ": bank " << bank->second << " twice";
        }
        EXPECT_EQ(banks_read.size(), banks) << coords[i];
    }
}

// The multi-core pipeline: work item w = bank * cores_per_bank + c, role w + 1, in the DSMC_ROLE table. The
// first core of each bank is that chip's optimal worker of the bank; the others are its nearest free cores.
TEST_F(MatmulDramShardedReaderPlacementTest, MulticorePipelineFollowsEachChipsAssignment) {
    constexpr uint32_t cores_per_bank = 2;
    auto* device = mesh_device_.get();
    auto operands = make_operands(device, /*in0_block_w=*/2, cores_per_bank);
    const auto artifacts = build_workload(device, operands);
    const auto coords = program_coords(device, artifacts);
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t banks = banks_with_columns(device);

    for (size_t i = 0; i < artifacts.programs.size(); ++i) {
        const auto expected = assignment_by_bank(*device, tt::tt_metal::NOC::NOC_0, coords[i]);
        const auto& kernels = artifacts.programs[i].spec.kernels;
        const auto compute = std::ranges::find_if(kernels, [](const auto& k) { return *k.unique_id == "compute"; });
        ASSERT_NE(compute, kernels.end()) << coords[i] << ": no compute kernel";
        const auto table = compute->compiler_options.defines.find("DSMC_ROLE");
        ASSERT_NE(table, compute->compiler_options.defines.end()) << coords[i] << ": no DSMC_ROLE define";

        std::unordered_map<uint32_t, CoreCoord> core_of_role;
        size_t pos = 0;
        for (uint32_t index = 0; pos <= table->second.size(); ++index) {
            const size_t comma = std::min(table->second.find(',', pos), table->second.size());
            const uint32_t role = std::stoul(table->second.substr(pos, comma - pos));
            if (role != 0) {
                EXPECT_TRUE(core_of_role.emplace(role, CoreCoord{index % grid.x, index / grid.x}).second);
            }
            pos = comma + 1;
        }
        ASSERT_EQ(core_of_role.size(), banks * cores_per_bank) << coords[i];
        for (uint32_t bank = 0; bank < banks; ++bank) {
            const CoreCoord& anchor = core_of_role.at((bank * cores_per_bank) + 1);
            EXPECT_EQ(anchor, expected[bank]) << coords[i] << ": bank " << bank << " is anchored at " << anchor.str()
                                              << ", this chip's optimal worker is " << expected[bank].str();
        }
    }
}

}  // namespace
}  // namespace ttnn::prim::test
