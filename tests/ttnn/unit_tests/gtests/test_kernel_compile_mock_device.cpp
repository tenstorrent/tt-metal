// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Runs representative op configs on a mock Wormhole / Blackhole device. Nothing executes, but every
// op's program factory runs and its kernels are JIT-compiled, so a kernel build error or a config
// that does not fit (CBs, kernel-config buffer) fails here without hardware.
// Run with an empty TT_METAL_CACHE so the kernels are actually compiled.

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/mesh_device.hpp>

#include <umd/device/types/arch.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/tilize/tilize.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/data_movement/untilize/untilize.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/ternary/ternary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/transformer/sdpa/sdpa.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/operations/transformer/sdpa_decode/sdpa_decode.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/types.hpp"

namespace {

using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::PageConfig;
using tt::tt_metal::TensorLayout;
using tt::tt_metal::TensorSpec;
using tt::tt_metal::distributed::MeshDevice;

ttnn::Tensor device_tensor(
    MeshDevice* device, const ttnn::Shape& shape, DataType dtype = DataType::BFLOAT16, Layout layout = Layout::TILE) {
    return ttnn::create_device_tensor(
        TensorSpec(shape, TensorLayout(dtype, PageConfig(layout), ttnn::DRAM_MEMORY_CONFIG)), device);
}

ttnn::DeviceComputeKernelConfig compute_config(MeshDevice* device, bool fp32_dest_acc, bool approx) {
    return ttnn::init_device_compute_kernel_config(
        device->arch(), std::nullopt, tt::tt_metal::MathFidelity::HiFi4, approx, fp32_dest_acc);
}

const ttnn::Shape kTileShape({1, 1, 64, 64});

struct CompileCase {
    std::string name;
    std::function<void(MeshDevice*)> run;
};

void PrintTo(const CompileCase& compile_case, std::ostream* os) { *os << compile_case.name; }

// One case per kernel file or compile-time branch (dtype, fp32 accumulation, approximation mode,
// fused activation, causal / masked); shapes are the smallest that select the kernel.
std::vector<CompileCase> eltwise_cases() {
    auto unary = [](auto op, DataType dtype = DataType::BFLOAT16) {
        return [op, dtype](MeshDevice* device) { op(device_tensor(device, kTileShape, dtype)); };
    };
    auto binary = [](auto op, const ttnn::Shape& rhs_shape, DataType dtype = DataType::BFLOAT16) {
        return [op, rhs_shape, dtype](MeshDevice* device) {
            op(device_tensor(device, kTileShape, dtype), device_tensor(device, rhs_shape, dtype));
        };
    };
    using ttnn::operations::unary::GeluVariant;
    return {
        {"exp", unary([](const auto& t) { ttnn::exp(t); })},
        {"exp_fast_approx", unary([](const auto& t) { ttnn::exp(t, true); })},
        {"exp_fp32", unary([](const auto& t) { ttnn::exp(t); }, DataType::FLOAT32)},
        {"gelu", unary([](const auto& t) { ttnn::gelu(t, GeluVariant::ACCURATE); })},
        {"gelu_fast_lut", unary([](const auto& t) { ttnn::gelu(t, GeluVariant::FAST_LUT); })},
        {"gelu_tanh", unary([](const auto& t) { ttnn::gelu(t, GeluVariant::TANH); })},
        {"sigmoid", unary([](const auto& t) { ttnn::sigmoid(t); })},
        {"tanh", unary([](const auto& t) { ttnn::tanh(t); })},
        {"sqrt", unary([](const auto& t) { ttnn::sqrt(t); })},
        {"rsqrt_fp32", unary([](const auto& t) { ttnn::rsqrt(t); }, DataType::FLOAT32)},
        {"log", unary([](const auto& t) { ttnn::log(t); })},
        {"erf", unary([](const auto& t) { ttnn::erf(t); })},
        {"reciprocal", unary([](const auto& t) { ttnn::reciprocal(t); })},
        {"silu", unary([](const auto& t) { ttnn::silu(t); })},
        {"softplus", unary([](const auto& t) { ttnn::softplus(t, 1.0f, 20.0f); })},
        {"softplus_fp32", unary([](const auto& t) { ttnn::softplus(t, 1.0f, 20.0f); }, DataType::FLOAT32)},
        {"relu_int32", unary([](const auto& t) { ttnn::relu(t); }, DataType::INT32)},
        {"add", binary([](const auto& a, const auto& b) { ttnn::add(a, b); }, kTileShape)},
        {"add_row_broadcast",
         binary([](const auto& a, const auto& b) { ttnn::add(a, b); }, ttnn::Shape({1, 1, 1, 64}))},
        {"add_int32", binary([](const auto& a, const auto& b) { ttnn::add(a, b); }, kTileShape, DataType::INT32)},
        {"multiply_fp32",
         binary([](const auto& a, const auto& b) { ttnn::multiply(a, b); }, kTileShape, DataType::FLOAT32)},
        {"where",
         [](MeshDevice* device) {
             ttnn::where(
                 device_tensor(device, kTileShape),
                 device_tensor(device, kTileShape),
                 device_tensor(device, kTileShape));
         }},
        {"where_scalars", [](MeshDevice* device) { ttnn::where(device_tensor(device, kTileShape), 1.0f, 0.0f); }},
    };
}

std::vector<CompileCase> matmul_cases() {
    // 128x128 @ 128x128 on a 2x2 grid: 2x2 output tiles and a 2-tile K block per core.
    auto matmul = [](ttnn::operations::matmul::MatmulProgramConfig program_config,
                     bool fp32_dest_acc,
                     std::optional<ttnn::Activation> activation = std::nullopt) {
        return [program_config, fp32_dest_acc, activation](MeshDevice* device) {
            ttnn::matmul(
                device_tensor(device, ttnn::Shape({1, 1, 128, 128})),
                device_tensor(device, ttnn::Shape({1, 1, 128, 128})),
                false,
                false,
                std::nullopt,
                std::nullopt,
                program_config,
                activation,
                compute_config(device, fp32_dest_acc, false));
        };
    };
    const auto mcast_2d = ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig{
        .compute_with_storage_grid_size = {2, 2},
        .in0_block_w = 2,
        .out_subblock_h = 1,
        .out_subblock_w = 2,
        .out_block_h = 2,
        .out_block_w = 2,
        .per_core_M = 2,
        .per_core_N = 2,
    };
    // 1D: the 4 output tile columns are split over 4 cores in one row, in0 is multicast.
    const auto mcast_1d = ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig{
        .compute_with_storage_grid_size = {4, 1},
        .in0_block_w = 2,
        .out_subblock_h = 1,
        .out_subblock_w = 1,
        .out_block_h = 4,
        .out_block_w = 1,
        .per_core_M = 4,
        .per_core_N = 1,
        .fuse_batch = true,
        .mcast_in0 = true,
    };
    return {
        {"matmul_2d_mcast", matmul(mcast_2d, false)},
        {"matmul_2d_mcast_fp32_acc", matmul(mcast_2d, true)},
        {"matmul_2d_mcast_gelu", matmul(mcast_2d, false, std::string("gelu"))},
        {"matmul_1d_mcast_in0", matmul(mcast_1d, false)},
        {"matmul_1d_mcast_in0_fp32_acc", matmul(mcast_1d, true)},
        {"matmul_auto_config",
         [](MeshDevice* device) {
             ttnn::matmul(
                 device_tensor(device, ttnn::Shape({1, 1, 128, 128})),
                 device_tensor(device, ttnn::Shape({1, 1, 128, 128})));
         }},
    };
}

std::vector<CompileCase> sdpa_cases() {
    // Decode: 8 users, 8 query heads, 1 KV head, 256 cached positions, head dim 128.
    auto decode = [](bool fp32_dest_acc) {
        return [fp32_dest_acc](MeshDevice* device) {
            constexpr uint32_t b = 8, nh = 8, s = 256, d = 128;
            const auto program_config = ttnn::operations::transformer::SDPAProgramConfig{
                .compute_with_storage_grid_size = device->compute_with_storage_grid_size(),
                .q_chunk_size = 32,
                .k_chunk_size = 128,
            };
            ttnn::transformer::scaled_dot_product_attention_decode(
                device_tensor(device, ttnn::Shape({1, b, nh, d})),
                device_tensor(device, ttnn::Shape({b, 1, s, d})),
                device_tensor(device, ttnn::Shape({b, 1, s, d})),
                true,
                std::nullopt,
                std::vector<uint32_t>(b, 100),
                std::nullopt,
                std::nullopt,
                1.0f / std::sqrt(static_cast<float>(d)),
                std::nullopt,
                std::nullopt,
                program_config,
                compute_config(device, fp32_dest_acc, false));
        };
    };
    // Prefill: 1 user, 2 heads, 128 tokens, head dim 64, 64-token Q and K chunks.
    auto prefill = [](bool is_causal, bool fp32_dest_acc) {
        return [is_causal, fp32_dest_acc](MeshDevice* device) {
            constexpr uint32_t b = 1, nh = 2, s = 128, d = 64;
            const auto program_config = ttnn::operations::transformer::SDPAProgramConfig{
                .compute_with_storage_grid_size = device->compute_with_storage_grid_size(),
                .q_chunk_size = 64,
                .k_chunk_size = 64,
            };
            std::optional<ttnn::Tensor> mask;
            if (!is_causal) {
                mask = device_tensor(device, ttnn::Shape({b, 1, s, s}));
            }
            ttnn::transformer::scaled_dot_product_attention(
                device_tensor(device, ttnn::Shape({b, nh, s, d})),
                device_tensor(device, ttnn::Shape({b, nh, s, d})),
                device_tensor(device, ttnn::Shape({b, nh, s, d})),
                mask,
                is_causal,
                1.0f / std::sqrt(static_cast<float>(d)),
                std::nullopt,
                std::nullopt,
                program_config,
                compute_config(device, fp32_dest_acc, false));
        };
    };
    return {
        {"sdpa_decode", decode(false)},
        {"sdpa_decode_fp32_acc", decode(true)},
        {"sdpa_prefill_causal", prefill(true, false)},
        {"sdpa_prefill_causal_fp32_acc", prefill(true, true)},
        {"sdpa_prefill_masked", prefill(false, false)},
    };
}

std::vector<CompileCase> data_movement_cases() {
    const ttnn::Shape shape({1, 2, 64, 64});
    return {
        {"pad_tile",
         [shape](MeshDevice* device) {
             ttnn::pad(device_tensor(device, shape), ttnn::Array4D{1, 2, 96, 96}, ttnn::Array4D{0, 0, 0, 0}, 0.0f);
         }},
        {"permute_tiled", [shape](MeshDevice* device) { ttnn::permute(device_tensor(device, shape), {0, 2, 1, 3}); }},
        {"permute_row_major",
         [shape](MeshDevice* device) {
             ttnn::permute(device_tensor(device, shape, DataType::BFLOAT16, Layout::ROW_MAJOR), {0, 2, 1, 3});
         }},
        {"transpose_wh", [shape](MeshDevice* device) { ttnn::transpose(device_tensor(device, shape), -2, -1); }},
        {"transpose_hc", [shape](MeshDevice* device) { ttnn::transpose(device_tensor(device, shape), 1, 2); }},
        {"tilize",
         [shape](MeshDevice* device) {
             ttnn::tilize(device_tensor(device, shape, DataType::BFLOAT16, Layout::ROW_MAJOR));
         }},
        {"untilize", [shape](MeshDevice* device) { ttnn::untilize(device_tensor(device, shape)); }},
        {"concat",
         [shape](MeshDevice* device) {
             ttnn::concat({device_tensor(device, shape), device_tensor(device, shape)}, 3);
         }},
    };
}

std::vector<CompileCase> compile_cases() {
    std::vector<CompileCase> cases;
    for (auto group : {eltwise_cases(), matmul_cases(), sdpa_cases(), data_movement_cases()}) {
        cases.insert(cases.end(), group.begin(), group.end());
    }
    return cases;
}

class KernelCompileMockDevice : public ::testing::TestWithParam<std::tuple<tt::ARCH, CompileCase>> {
protected:
    std::unique_ptr<tt::tt_metal::MetalEnv> env_;
    std::shared_ptr<MeshDevice> device_;

    void SetUp() override {
        env_ = std::make_unique<tt::tt_metal::MetalEnv>(tt::tt_metal::MetalEnvDescriptor{
            .mock_cluster_desc_path =
                tt::tt_metal::experimental::get_mock_cluster_desc_name(std::get<0>(GetParam()), 1)});
        device_ =
            env_->create_mesh_device(tt::tt_metal::distributed::MeshDeviceConfig(env_->get_system_mesh().shape()));
        // A cached workload outlives the sub-devices it references and crashes at teardown on mock (#45646).
        device_->disable_and_clear_program_cache();
    }

    void TearDown() override {
        device_.reset();
        env_.reset();
    }
};

TEST_P(KernelCompileMockDevice, Compiles) { std::get<1>(GetParam()).run(device_.get()); }

INSTANTIATE_TEST_SUITE_P(
    Ops,
    KernelCompileMockDevice,
    ::testing::Combine(
        ::testing::Values(tt::ARCH::WORMHOLE_B0, tt::ARCH::BLACKHOLE), ::testing::ValuesIn(compile_cases())),
    [](const auto& info) {
        const auto arch = std::get<0>(info.param) == tt::ARCH::WORMHOLE_B0 ? "wormhole" : "blackhole";
        return std::string(arch) + "_" + std::get<1>(info.param).name;
    });

}  // namespace
