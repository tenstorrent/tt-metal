// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Calls ttnn ops on a mock Wormhole / Blackhole device: program factories run and kernels are JIT-compiled,
// but nothing executes. A kernel that does not build fails here, without hardware. Use an empty cache:
//   TT_METAL_CACHE=$(mktemp -d) tt-nn-validation-smoke --gtest_filter='CPU_KernelCompile*/*wormhole_*'
//
// To add coverage:
// - a config: one {"name", lambda} line in its *_cases() group, one per kernel file or compile-time branch;
// - an op-type enum: one sweep<Enum>() call in CPU_KernelCompileSweep; new enum values are then covered.
// CPU_KernelCompile runs in the PR gate, both suites in the merge gate.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/mesh_device.hpp>

#include <enchantum/enchantum.hpp>
#include <umd/device/types/arch.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/to_memory_config/to_memory_config_op.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/gather/gather.hpp"
#include "ttnn/operations/data_movement/move/move.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/tilize/tilize.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/data_movement/untilize/untilize.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/common/binary_op_types.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.hpp"
#include "ttnn/operations/eltwise/ternary/common/ternary_op_types.hpp"
#include "ttnn/operations/eltwise/ternary/device/ternary_device_operation.hpp"
#include "ttnn/operations/eltwise/ternary/ternary.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/eltwise/unary/device/unary_device_operation.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/normalization/layernorm/layernorm.hpp"
#include "ttnn/operations/reduction/prod/prod.hpp"
#include "ttnn/operations/transformer/sdpa/sdpa.hpp"
#include "ttnn/operations/transformer/sdpa/sparse_sdpa.hpp"
#include "ttnn/operations/transformer/sdpa/sparse_sdpa_msa.hpp"
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
    MeshDevice* device,
    const ttnn::Shape& shape,
    DataType dtype = DataType::BFLOAT16,
    Layout layout = Layout::TILE,
    const ttnn::MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG) {
    return ttnn::create_device_tensor(
        TensorSpec(shape, TensorLayout(dtype, PageConfig(layout), memory_config)), device);
}

ttnn::DeviceComputeKernelConfig compute_config(MeshDevice* device, bool fp32_dest_acc) {
    return ttnn::init_device_compute_kernel_config(
        device->arch(), std::nullopt, tt::tt_metal::MathFidelity::HiFi4, false, fp32_dest_acc);
}

const ttnn::Shape kTileShape({1, 1, 64, 64});

struct CompileCase {
    std::string name;
    std::function<void(MeshDevice*)> run;
};

void PrintTo(const CompileCase& compile_case, std::ostream* os) { *os << compile_case.name; }

const std::vector<DataType> kSweepDtypes = {DataType::BFLOAT16, DataType::INT32};

// Errors from finding or building a kernel; any other error is the op rejecting the config on host.
constexpr std::string_view kKernelErrors[] = {
    "build failed", "Failed to generate binaries", "doesn't exist in any of the searched paths"};

// Runs with the first dtype the op accepts on host. If it accepts none the case is skipped; a kernel that
// is missing or does not build always fails.
void run_with_supported_dtype(
    const std::function<void(DataType)>& run, const std::vector<DataType>& dtypes = kSweepDtypes) {
    std::string rejected;
    for (const auto dtype : dtypes) {
        try {
            run(dtype);
            return;
        } catch (const std::exception& e) {
            const std::string_view what = e.what();
            if (std::ranges::any_of(
                    kKernelErrors, [&](auto error) { return what.find(error) != std::string_view::npos; })) {
                throw;
            }
            rejected = what.substr(0, what.find('\n'));
        }
    }
    GTEST_SKIP() << rejected;
}

// One case per enum value, named <prefix>_<VALUE>.
template <typename Enum, typename Run>
std::vector<CompileCase> sweep(const std::string& prefix, Run run, const std::vector<DataType>& dtypes = kSweepDtypes) {
    std::vector<CompileCase> cases;
    for (const auto& [value, name] : enchantum::entries_generator<Enum>) {
        cases.push_back({prefix + "_" + std::string(name), [value, run, dtypes](MeshDevice* device) {
                             run_with_supported_dtype([&](DataType dtype) { run(device, value, dtype); }, dtypes);
                         }});
    }
    return cases;
}

std::vector<CompileCase> join(std::vector<std::vector<CompileCase>> groups) {
    std::vector<CompileCase> cases;
    for (auto& group : groups) {
        cases.insert(cases.end(), group.begin(), group.end());
    }
    return cases;
}

// One case per kernel file or compile-time branch; shapes are the smallest that select it.
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
        {"snake_beta",
         [](MeshDevice* device) {
             ttnn::snake_beta(
                 device_tensor(device, kTileShape),
                 device_tensor(device, ttnn::Shape({1, 1, 1, 64})),
                 device_tensor(device, ttnn::Shape({1, 1, 1, 64})));
         }},
    };
}

// Params are 1: valid as a value, a flag or an exponent.
std::vector<CompileCase> unary_sweep_cases() {
    using ttnn::operations::unary::EltwiseUnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    return sweep<UnaryOpType>("unary", [](MeshDevice* device, UnaryOpType op_type, DataType dtype) {
        auto op = EltwiseUnaryWithParam(op_type);
        if (ttnn::operations::unary::utils::is_parametrized_type(op_type)) {
            op = dtype == DataType::INT32 ? EltwiseUnaryWithParam(op_type, std::vector<std::int32_t>{1, 1})
                                          : EltwiseUnaryWithParam(op_type, std::vector<float>{1.0f, 1.0f});
        }
        ttnn::prim::unary(
            device_tensor(device, kTileShape, dtype), {op}, dtype, ttnn::DRAM_MEMORY_CONFIG, false, false, false);
    });
}

// Each broadcast kind has its own kernels. Tensor-tensor only: binary_ng has no tensor-scalar kernel for
// some ops (MAXIMUM, GCD, ...), which the public API sends through unary.
std::vector<CompileCase> binary_sweep_cases() {
    using ttnn::operations::binary::BinaryOpType;
    const std::vector<std::tuple<std::string, ttnn::Shape, ttnn::Shape>> broadcasts = {
        {"none", kTileShape, kTileShape},
        {"row", kTileShape, ttnn::Shape({1, 1, 1, 64})},
        {"col", kTileShape, ttnn::Shape({1, 1, 64, 1})},
        {"row_col", ttnn::Shape({1, 1, 1, 64}), ttnn::Shape({1, 1, 64, 1})},
        {"scalar", kTileShape, ttnn::Shape({1, 1, 1, 1})},
    };
    std::vector<std::vector<CompileCase>> groups;
    groups.reserve(broadcasts.size());
    for (const auto& [name, lhs, rhs] : broadcasts) {
        groups.push_back(
            sweep<BinaryOpType>("binary_" + name, [lhs, rhs](MeshDevice* device, BinaryOpType op_type, DataType dtype) {
                ttnn::prim::binary_ng(device_tensor(device, lhs, dtype), device_tensor(device, rhs, dtype), op_type);
            }));
    }
    return join(std::move(groups));
}

// Each broadcast of the third operand, a scalar second or third operand, and each dtype.
std::vector<CompileCase> ternary_sweep_cases() {
    using ttnn::operations::ternary::TernaryOpType;
    auto tensors = [](const ttnn::Shape& c_shape) {
        return [c_shape](MeshDevice* device, TernaryOpType op_type, DataType dtype) {
            ttnn::prim::ternary(
                op_type,
                device_tensor(device, kTileShape, dtype),
                device_tensor(device, kTileShape, dtype),
                device_tensor(device, c_shape, dtype));
        };
    };
    std::vector<std::vector<CompileCase>> groups = {
        sweep<TernaryOpType>("ternary_none", tensors(kTileShape)),
        sweep<TernaryOpType>("ternary_row", tensors(ttnn::Shape({1, 1, 1, 64}))),
        sweep<TernaryOpType>("ternary_col", tensors(ttnn::Shape({1, 1, 64, 1}))),
        sweep<TernaryOpType>("ternary_scalar", tensors(ttnn::Shape({1, 1, 1, 1}))),
        sweep<TernaryOpType>(
            "ternary_tts",
            [](MeshDevice* device, TernaryOpType op_type, DataType dtype) {
                ttnn::prim::ternary(
                    op_type, device_tensor(device, kTileShape, dtype), device_tensor(device, kTileShape, dtype), 1.0f);
            }),
        sweep<TernaryOpType>(
            "ternary_tst",
            [](MeshDevice* device, TernaryOpType op_type, DataType dtype) {
                ttnn::prim::ternary(
                    op_type, device_tensor(device, kTileShape, dtype), 1.0f, device_tensor(device, kTileShape, dtype));
            }),
    };
    for (const auto dtype : {DataType::FLOAT32, DataType::INT32, DataType::UINT32}) {
        groups.push_back(
            sweep<TernaryOpType>("ternary_" + std::string(enchantum::to_string(dtype)), tensors(kTileShape), {dtype}));
    }
    return join(std::move(groups));
}

// Each op with DRAM in and out, or a sharded input or output, in TILE and ROW_MAJOR: sharded readers and
// writers are separate kernels.
std::vector<CompileCase> data_movement_sweep_cases() {
    using ttnn::operations::data_movement::ShardStrategy;
    using Op = std::function<void(const ttnn::Tensor&, const ttnn::MemoryConfig&)>;
    const std::vector<std::pair<std::string, Op>> ops = {
        {"permute_row_invariant", [](const auto& t, const auto& mem) { ttnn::permute(t, {1, 0, 2, 3}, mem); }},
        {"permute_wh", [](const auto& t, const auto& mem) { ttnn::permute(t, {0, 1, 3, 2}, mem); }},
        {"transpose_wh", [](const auto& t, const auto& mem) { ttnn::transpose(t, -2, -1, mem); }},
        {"tilize", [](const auto& t, const auto& mem) { ttnn::tilize(t, mem); }},
        {"untilize", [](const auto& t, const auto& mem) { ttnn::untilize(t, mem); }},
        {"to_memory_config", [](const auto& t, const auto& mem) { ttnn::to_memory_config(t, mem); }},
    };
    // Name, input shard strategy, output shard strategy; none is DRAM.
    using Strategy = std::optional<ShardStrategy>;
    std::vector<std::tuple<std::string, Strategy, Strategy>> placements = {{"dram", std::nullopt, std::nullopt}};
    for (const auto& [name, strategy] : std::vector<std::pair<std::string, ShardStrategy>>{
             {"height", ShardStrategy::HEIGHT}, {"width", ShardStrategy::WIDTH}, {"block", ShardStrategy::BLOCK}}) {
        placements.emplace_back(name + "_in", strategy, std::nullopt);
        placements.emplace_back(name + "_out", std::nullopt, strategy);
    }
    std::vector<CompileCase> cases;
    for (const auto layout : {Layout::TILE, Layout::ROW_MAJOR}) {
        const std::string layout_name = layout == Layout::TILE ? "tile" : "row_major";
        for (const auto& [op_name, op] : ops) {
            for (const auto& [placement, input, output] : placements) {
                cases.push_back(
                    {"dm_" + op_name + "_" + placement + "_" + layout_name,
                     [op, layout, input, output](MeshDevice* device) {
                         const ttnn::Shape shape({1, 2, 128, 128});
                         auto memory_config = [&](Strategy strategy) {
                             return strategy ? ttnn::operations::data_movement::create_sharded_memory_config(
                                                   shape,
                                                   tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange({0, 0}, {1, 1})),
                                                   *strategy,
                                                   tt::tt_metal::ShardOrientation::ROW_MAJOR,
                                                   std::nullopt,
                                                   layout)
                                             : ttnn::DRAM_MEMORY_CONFIG;
                         };
                         run_with_supported_dtype(
                             [&](DataType dtype) {
                                 op(device_tensor(device, shape, dtype, layout, memory_config(input)),
                                    memory_config(output));
                             },
                             {DataType::BFLOAT16});
                     }});
            }
        }
    }
    return cases;
}

std::vector<CompileCase> matmul_cases() {
    // 128x128 @ 128x128 on a 2x2 grid: 2x2 output tiles and a 2-tile K block per core.
    auto matmul = [](const ttnn::operations::matmul::MatmulProgramConfig& program_config,
                     bool fp32_dest_acc,
                     const std::optional<ttnn::Activation>& activation = std::nullopt) {
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
                compute_config(device, fp32_dest_acc));
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
                compute_config(device, fp32_dest_acc));
        };
    };
    // Prefill: 1 user, 2 heads, 128 tokens, head dim 64, 64-token Q and K chunks. Non-causal uses a mask,
    // or cu_window_seqlens when windowed, which selects the windowed branch of the kernels.
    auto prefill = [](bool is_causal, bool fp32_dest_acc, bool windowed = false) {
        return [is_causal, fp32_dest_acc, windowed](MeshDevice* device) {
            constexpr uint32_t b = 1, nh = 2, s = 128, d = 64;
            const auto program_config = ttnn::operations::transformer::SDPAProgramConfig{
                .compute_with_storage_grid_size = device->compute_with_storage_grid_size(),
                .q_chunk_size = 64,
                .k_chunk_size = 64,
            };
            std::optional<ttnn::Tensor> mask, cu_window_seqlens;
            if (windowed) {
                cu_window_seqlens = device_tensor(device, ttnn::Shape({3}), DataType::INT32, Layout::ROW_MAJOR);
            } else if (!is_causal) {
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
                compute_config(device, fp32_dest_acc),
                std::nullopt,
                cu_window_seqlens);
        };
    };
    auto blackhole_only = [](const std::function<void(MeshDevice*)>& run) {
        return [run](MeshDevice* device) {
            if (device->arch() != tt::ARCH::BLACKHOLE) {
                GTEST_SKIP() << "Blackhole-only op";
            }
            run(device);
        };
    };
    return {
        {"sdpa_decode", decode(false)},
        {"sdpa_decode_fp32_acc", decode(true)},
        {"sdpa_prefill_causal", prefill(true, false)},
        {"sdpa_prefill_causal_fp32_acc", prefill(true, true)},
        {"sdpa_prefill_masked", prefill(false, false)},
        {"sdpa_windowed", prefill(false, false, true)},
        {"sdpa_windowed_causal", prefill(true, false, true)},
        {"sparse_sdpa", blackhole_only([](MeshDevice* device) {
             ttnn::transformer::sparse_sdpa(
                 device_tensor(device, ttnn::Shape({1, 32, 32, 64}), DataType::BFLOAT16, Layout::ROW_MAJOR),
                 device_tensor(device, ttnn::Shape({1, 1, 128, 64}), DataType::BFLOAT16, Layout::ROW_MAJOR),
                 device_tensor(device, ttnn::Shape({1, 1, 32, 128}), DataType::UINT32, Layout::ROW_MAJOR),
                 64,
                 ttnn::transformer::SparseKVFormat::BF16);
         })},
        {"sparse_sdpa_msa", blackhole_only([](MeshDevice* device) {
             ttnn::transformer::sparse_sdpa_msa(
                 device_tensor(device, ttnn::Shape({1, 16, 32, 64}), DataType::BFLOAT16, Layout::ROW_MAJOR),
                 device_tensor(device, ttnn::Shape({1, 1, 128, 64})),
                 device_tensor(device, ttnn::Shape({1, 1, 128, 64})),
                 device_tensor(device, ttnn::Shape({1, 1, 32, 16}), DataType::UINT32, Layout::ROW_MAJOR));
         })},
    };
}

std::vector<CompileCase> normalization_cases() {
    return {
        {"layernorm_welford",
         [](MeshDevice* device) {
             ttnn::layer_norm(
                 device_tensor(device, kTileShape),
                 1e-5f,
                 std::nullopt,
                 std::nullopt,
                 std::nullopt,
                 std::nullopt,
                 ttnn::prim::LayerNormDefaultProgramConfig{.use_welford = true},
                 compute_config(device, true));
         }},
    };
}

std::vector<CompileCase> data_movement_cases() {
    const ttnn::Shape shape({1, 2, 64, 64});
    // Gather picks a single-core or multi-core factory by index width (over 60 tiles is multi-core).
    auto gather = [](uint32_t width, Layout layout) {
        return [width, layout](MeshDevice* device) {
            const ttnn::Shape gather_shape({1, 1, 32, width});
            ttnn::gather(
                device_tensor(device, gather_shape, DataType::BFLOAT16, layout),
                -1,
                device_tensor(device, gather_shape, DataType::UINT32, layout),
                false,
                std::nullopt);
        };
    };
    // Freeing a buffer just below the input makes the move output overlap it.
    auto move_overlap = [](Layout layout) {
        return [layout](MeshDevice* device) {
            auto below = device_tensor(device, ttnn::Shape({1, 1, 32, 64}), DataType::BFLOAT16, layout);
            auto input = device_tensor(device, ttnn::Shape({1, 1, 256, 64}), DataType::BFLOAT16, layout);
            const auto below_address = below.buffer()->address();
            below.deallocate(true);
            const auto output = ttnn::move(input);
            ASSERT_EQ(output.buffer()->address(), below_address) << "output does not overlap the input";
        };
    };
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
        {"gather_single_core", gather(64, Layout::TILE)},
        {"gather_multi_core", gather(64 * 32, Layout::TILE)},
        {"gather_row_major_single_core", gather(64, Layout::ROW_MAJOR)},
        {"gather_row_major_multi_core", gather(64 * 32, Layout::ROW_MAJOR)},
        {"permute_row_invariant_height_sharded",
         [](MeshDevice* device) {
             ttnn::permute(
                 device_tensor(device, ttnn::Shape({2, 1, 1, 64})),
                 {2, 0, 1, 3},
                 ttnn::MemoryConfig(tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED, tt::tt_metal::BufferType::L1));
         }},
        {"move_overlap_row_major", move_overlap(Layout::ROW_MAJOR)},
        {"move_overlap_tile", move_overlap(Layout::TILE)},
        {"prod_dim_n", [](MeshDevice* device) { ttnn::prod(device_tensor(device, ttnn::Shape({2, 2, 64, 64})), 0); }},
        {"prod_dim_c", [](MeshDevice* device) { ttnn::prod(device_tensor(device, ttnn::Shape({2, 2, 64, 64})), 1); }},
    };
}

class KernelCompileMockDevice : public ::testing::TestWithParam<std::tuple<tt::ARCH, CompileCase>> {
protected:
    std::unique_ptr<tt::tt_metal::MetalEnv> env_;
    std::shared_ptr<MeshDevice> device_;

    void SetUp() override {
        env_ = std::make_unique<tt::tt_metal::MetalEnv>(tt::tt_metal::MetalEnvDescriptor{
            .mock_cluster_desc_path =
                tt::tt_metal::experimental::get_mock_cluster_desc_name(std::get<0>(GetParam()), 1)});
        device_ = MeshDevice::open_mesh_device(
            *env_, tt::tt_metal::distributed::MeshDeviceConfig(env_->get_system_mesh().shape()));
        // A cached workload outlives the sub-devices it references and crashes at teardown on mock (#45646).
        device_->disable_and_clear_program_cache();
    }

    void TearDown() override {
        if (HasFailure()) {
            const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
            std::cerr << "Reproduce without hardware:\n  TT_METAL_CACHE=$(mktemp -d) tt-nn-validation-smoke "
                      << "--gtest_filter='" << info->test_suite_name() << "." << info->name() << "'\n"
                      << "If your change makes the op reject this config on purpose, update the case in\n"
                      << "  tests/ttnn/unit_tests/gtests/test_kernel_compile_mock_device.cpp\n";
        }
        device_.reset();
        env_.reset();
    }
};

TEST_P(KernelCompileMockDevice, Compiles) { std::get<1>(GetParam()).run(device_.get()); }

std::string case_name(const ::testing::TestParamInfo<KernelCompileMockDevice::ParamType>& info) {
    const auto* const arch = std::get<0>(info.param) == tt::ARCH::WORMHOLE_B0 ? "wormhole" : "blackhole";
    return std::string(arch) + "_" + std::get<1>(info.param).name;
}

const auto kArchs = ::testing::Values(tt::ARCH::WORMHOLE_B0, tt::ARCH::BLACKHOLE);

// CPU_: host-only, so CI runs these on CPU runners and leaves them out of hardware runs.
INSTANTIATE_TEST_SUITE_P(
    CPU_KernelCompile,
    KernelCompileMockDevice,
    ::testing::Combine(
        kArchs,
        ::testing::ValuesIn(
            join({eltwise_cases(), matmul_cases(), sdpa_cases(), normalization_cases(), data_movement_cases()}))),
    case_name);

INSTANTIATE_TEST_SUITE_P(
    CPU_KernelCompileSweep,
    KernelCompileMockDevice,
    ::testing::Combine(
        kArchs,
        ::testing::ValuesIn(
            join({unary_sweep_cases(), binary_sweep_cases(), ternary_sweep_cases(), data_movement_sweep_cases()}))),
    case_name);

}  // namespace
