// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The matmul config model's candidates against the real matmul device op, on a mock Blackhole P150 (no hardware).
// For candidates the selector enumerates for a set of matmuls, the device op's own validation must accept them and
// build their programs, and the L1 buffers those programs allocate per core must fit in what the selector budgets for
// them (circular_buffer_bytes; today they match it exactly). The selector checks a pick against the live L1 with that
// estimate and the device op does not check buffers against L1, so an estimate below the real allocation could pick a
// config that overflows L1. A change to a factory's buffers that the selector doesn't follow fails here.

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/mesh_device.hpp>

#include <tt-umd/types/arch.hpp>

#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/config/model/matmul_candidates.hpp"
#include "ttnn/operations/matmul/device/config/model/matmul_model_hook.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::matmul::model {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

using TtDataType = tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::MathFidelity;
using tt::tt_metal::MemoryConfig;
using tt::tt_metal::PageConfig;
using tt::tt_metal::TensorLayout;
using tt::tt_metal::TensorSpec;

constexpr std::uint32_t kTile = 32;
constexpr std::uint32_t kBudget = 1461120 - 16 * 1024;  // a free Blackhole p150 L1 bank less the selector's headroom

class MatmulModelMockDevice : public ::testing::Test {
protected:
    std::unique_ptr<tt::tt_metal::MetalEnv> env_;
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> device_;

    void SetUp() override {
        env_ = std::make_unique<tt::tt_metal::MetalEnv>(tt::tt_metal::MetalEnvDescriptor{
            .mock_cluster_desc_path = tt::tt_metal::experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 1)});
        device_ = tt::tt_metal::distributed::MeshDevice::open_mesh_device(
            *env_, tt::tt_metal::distributed::MeshDeviceConfig(env_->get_system_mesh().shape()));
        ASSERT_GT(device_->num_devices(), 0u);
        // As in test_query_op_constraints_mock_device.cpp: cached workloads outlive their sub-devices at teardown
        device_->disable_and_clear_program_cache();
    }

    void TearDown() override {
        device_.reset();
        env_.reset();
    }
};

struct Case {
    const char* name;
    ttnn::Shape a, b;  // as stored, before transposes
    TtDataType a_dtype = TtDataType::BFLOAT16, b_dtype = TtDataType::BFLOAT16, out_dtype = TtDataType::BFLOAT16;
    MemoryConfig a_mem = ttnn::DRAM_MEMORY_CONFIG, out_mem = ttnn::DRAM_MEMORY_CONFIG;
    bool transpose_a = false, transpose_b = false;
    MathFidelity fidelity = MathFidelity::HiFi2;
    bool fp32_acc = false, packer_l1_acc = true;
};

model::DataType model_dtype(TtDataType t) {
    switch (t) {
        case TtDataType::BFLOAT8_B: return model::DataType::kBfloat8B;
        case TtDataType::BFLOAT4_B: return model::DataType::kBfloat4B;
        case TtDataType::FLOAT32: return model::DataType::kFloat32;
        default: return model::DataType::kBfloat16;
    }
}

Fidelity model_fidelity(MathFidelity f) {
    switch (f) {
        case MathFidelity::LoFi: return Fidelity::kLoFi;
        case MathFidelity::HiFi2: return Fidelity::kHiFi2;
        case MathFidelity::HiFi3: return Fidelity::kHiFi3;
        default: return Fidelity::kHiFi4;
    }
}

// The selector's view of the call (matmul_model_hook.cpp, describe): sizes in tiles after transposes
MatmulDesc desc_of(const Case& c) {
    MatmulDesc p;
    const auto& a = c.a;
    const auto& b = c.b;
    p.rank_a = static_cast<std::uint32_t>(a.rank());
    p.rank_b = static_cast<std::uint32_t>(b.rank());
    p.batch_a = static_cast<std::uint32_t>(a.volume() / (static_cast<std::uint64_t>(a[-2]) * a[-1]));
    p.batch_b = static_cast<std::uint32_t>(b.volume() / (static_cast<std::uint64_t>(b[-2]) * b[-1]));
    p.Mt = (c.transpose_a ? a[-1] : a[-2]) / kTile;
    p.Kt = (c.transpose_a ? a[-2] : a[-1]) / kTile;
    p.Nt = (c.transpose_b ? b[-2] : b[-1]) / kTile;
    p.in0 = model_dtype(c.a_dtype);
    p.in1 = model_dtype(c.b_dtype);
    p.out = model_dtype(c.out_dtype);
    p.transpose_a = c.transpose_a;
    p.fidelity = model_fidelity(c.fidelity);
    p.fp32_dest_acc = c.fp32_acc;
    p.packer_l1_acc = c.packer_l1_acc;
    p.a.in_l1 = c.a_mem.buffer_type() == tt::tt_metal::BufferType::L1;
    p.out_place.in_l1 = c.out_mem.buffer_type() == tt::tt_metal::BufferType::L1;
    return p;
}

std::string describe(const Candidate& c) {
    std::ostringstream s;
    const auto& b = c.blocking;
    s << "family " << static_cast<int>(c.family) << " grid " << c.grid_x << "x" << c.grid_y << " in0_block_w "
      << b.in0_block_w << " per core " << b.per_core_M << "x" << b.per_core_N << " block " << b.out_block_h << "x"
      << b.out_block_w << " subblock " << b.out_subblock_h << "x" << b.out_subblock_w << " fuse_batch " << c.fuse_batch;
    return s.str();
}

std::vector<Case> cases() {
    using S = ttnn::Shape;
    std::vector<Case> v;
    v.push_back(
        {"prefill, 2D",
         S(ttnn::Array4D{1, 1, 1024, 4096}),
         S(ttnn::Array4D{1, 1, 4096, 4096}),
         TtDataType::BFLOAT16,
         TtDataType::BFLOAT8_B});
    v.push_back(
        {"decode, 1D in0",
         S(ttnn::Array4D{1, 1, 32, 2048}),
         S(ttnn::Array4D{1, 1, 2048, 8192}),
         TtDataType::BFLOAT16,
         TtDataType::BFLOAT8_B});
    v.push_back({"tall, A in L1", S(ttnn::Array4D{1, 1, 4096, 256}), S(ttnn::Array4D{1, 1, 256, 1024})});
    v.back().a_mem = ttnn::L1_MEMORY_CONFIG;
    v.push_back({"router, deep K", S(ttnn::Array4D{1, 1, 128, 2880}), S(ttnn::Array4D{1, 1, 2880, 128})});
    v.push_back({"batched, fp32 acc", S(ttnn::Array4D{1, 8, 128, 64}), S(ttnn::Array4D{1, 8, 64, 128})});
    v.back().fidelity = MathFidelity::HiFi4;
    v.back().fp32_acc = true;
    v.push_back(
        {"bfp8 out, no packer L1 acc",
         S(ttnn::Array4D{1, 1, 32, 2880}),
         S(ttnn::Array4D{1, 1, 2880, 4096}),
         TtDataType::BFLOAT8_B,
         TtDataType::BFLOAT8_B,
         TtDataType::BFLOAT8_B});
    v.back().fidelity = MathFidelity::LoFi;
    v.back().packer_l1_acc = false;
    v.push_back({"transposed B", S(ttnn::Array4D{1, 1, 512, 1024}), S(ttnn::Array4D{1, 1, 2048, 1024})});
    v.back().transpose_b = true;
    v.push_back({"transposed A", S(ttnn::Array4D{1, 1, 1024, 512}), S(ttnn::Array4D{1, 1, 1024, 1024})});
    v.back().transpose_a = true;
    v.push_back({"output in L1", S(ttnn::Array4D{1, 1, 256, 1024}), S(ttnn::Array4D{1, 1, 1024, 512})});
    v.back().out_mem = ttnn::L1_MEMORY_CONFIG;
    return v;
}

// Up to `per_family` candidates of each layout, spread over the enumeration order (it is long, and each query builds
// the op's program)
std::vector<Candidate> sample(const std::vector<Candidate>& all, std::size_t per_family) {
    std::vector<Candidate> out;
    for (const Family f : {Family::k2D, Family::k1DMcastIn0, Family::k1DMcastIn1, Family::kReuse}) {
        std::vector<const Candidate*> of;
        for (const auto& c : all) {
            if (c.family == f) {
                of.push_back(&c);
            }
        }
        const std::size_t step = of.size() <= per_family ? 1 : of.size() / per_family;
        for (std::size_t i = 0; i < of.size() && i / step < per_family; i += step) {
            out.push_back(*of[i]);
        }
    }
    return out;
}

TEST_F(MatmulModelMockDevice, CandidatesPassValidationAndFitTheirL1Estimate) {
    const auto grid = device_->compute_with_storage_grid_size();
    HardwareDesc hw = HardwareDesc::blackhole(grid.x, grid.y, kBudget);
    hw.dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    std::size_t checked = 0;
    for (const auto& c : cases()) {
        const MatmulDesc p = desc_of(c);
        ASSERT_EQ(unsupported_reason(p), "") << c.name;
        const TensorSpec spec_a(c.a, TensorLayout(c.a_dtype, PageConfig(Layout::TILE), c.a_mem));
        const TensorSpec spec_b(c.b, TensorLayout(c.b_dtype, PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG));
        const auto kernel_config = ttnn::init_device_compute_kernel_config(
            tt::ARCH::BLACKHOLE, std::nullopt, c.fidelity, /*default_approx_mode=*/true, c.fp32_acc, c.packer_l1_acc);
        const auto all = enumerate_candidates_with_narrower_grid(p, hw);
        ASSERT_FALSE(all.empty()) << c.name;
        for (const auto& candidate : sample(all, 4)) {
            ++checked;
            const auto query = ttnn::graph::query_op_constraints(
                ttnn::matmul,
                device_.get(),
                spec_a,
                spec_b,
                c.transpose_a,
                c.transpose_b,
                c.out_mem,
                c.out_dtype,
                std::make_optional(to_program_config(candidate)),
                std::nullopt,  // activation
                std::make_optional(kernel_config),
                std::nullopt,  // core_grid
                std::nullopt,  // output_tile
                std::nullopt,  // optional_output_tensor
                std::nullopt,  // global_cb
                std::nullopt,  // sub_device_id
                ttnn::PrefetcherPipeList{});
            ASSERT_EQ(query.status, ttnn::graph::ExecutionStatus::Success)
                << c.name << ", " << describe(candidate) << ": " << query.error_message.value_or("no message");
            // The op's buffers: its peak per-core L1 less the L1 tensors (circular and dataflow buffers both count in
            // the peak; cb_peak_size_per_core leaves dataflow buffers out)
            const std::uint64_t buffers =
                query.resource_usage.peak_memory_usage_per_core - query.resource_usage.l1_buffers_peak_per_core;
            EXPECT_GT(buffers, 0u) << c.name << ", " << describe(candidate);
            EXPECT_LE(buffers, circular_buffer_bytes(p, hw, candidate)) << c.name << ", " << describe(candidate);
        }
    }
    EXPECT_GT(checked, 60u);
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace
}  // namespace ttnn::operations::matmul::model
