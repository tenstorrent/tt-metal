// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/bmm/bmm_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

#include <optional>
#include <algorithm>

using namespace tt::constants;

vector<uint32_t> _get_prime_factors(uint32_t n) {
    uint32_t i = 2;

    vector<uint32_t> prime_factors;
    while (i * i <= n) {
        if (n % i != 0) i++;
        else {
            n /= i;
            prime_factors.push_back(i);
        }
    }
    if (n > 1) prime_factors.push_back(n);

    return prime_factors;
}

vector<uint32_t> _get_possible_products(vector<uint32_t> factors) {
    if (factors.size() == 0) return {1};

    vector<uint32_t> products;
    for (uint32_t& fac : factors) {
        vector<uint32_t> new_products;
        if (not std::count(products.begin(), products.end(), fac))
            new_products.push_back(fac);
        for (uint32_t& prod : products) {
            if (not std::count(products.begin(), products.end(), fac * prod))
                new_products.push_back(fac * prod);
        }

        // Insert all new products to product
        products.reserve(products.size() + distance(new_products.begin(), new_products.end()));
        products.insert(products.end(), new_products.begin(), new_products.end());
    }

    // Sort products
    std::sort(products.begin(), products.end());

    return products;
}

uint32_t _get_maximum_block_dim(int32_t block_dim, int32_t in0_block_w) {
    int32_t other_dim = (400 - 2 * in0_block_w * block_dim) / (2 * in0_block_w + block_dim);
    if (other_dim > 0)
        return other_dim;
    return 0;
}

namespace {
using namespace tt;
using namespace tt::tt_metal;
operation::OpPerformanceModel create_op_performance_model_for_matmul(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor> &output_tensors,
    const DeviceComputeKernelConfig& compute_kernel_config
)  {

    const auto& in_a_shape = input_tensors.at(0).get_shape();
    const auto& in_b_shape = input_tensors.at(1).get_shape();
    const auto& out_shape = output_tensors.at(0).get_shape();

    const auto& t = output_tensors.at(0);
    if(t.storage_type() != StorageType::DEVICE) {
        tt::log_warning(tt::LogOp, "Output tensor not on DEVICE?!");
    }

    auto arch = t.storage_type() == StorageType::DEVICE ? t.device()->arch() : AutoFormat::GetDefaultDevice()->arch();
    const int num_cores = (arch == ARCH::WORMHOLE_B0) ? 8 * 8 : 9 * 12;
    const int tensix_mul_adds_per_cycle_lofi = (arch == ARCH::WORMHOLE_B0) ? 4096 : 2048;

    // Calculate number of mul/add operations
    // TODO: add bias modeling
    int64_t num_mul_adds_per_elem = in_a_shape[3] * 2; // 1 multiply and 1 add per element
    int64_t num_mul_adds = num_mul_adds_per_elem * out_shape[2] * out_shape[3] * out_shape[1] * out_shape[0];

    MathFidelity math_fidelity = MathFidelity::Invalid;

    std::visit([&](auto&& compute_kernel_config) {
        using T = std::decay_t<decltype(compute_kernel_config)>;
        if constexpr (std::is_same_v<T, GrayskullComputeKernelConfig>) {
            math_fidelity = compute_kernel_config.math_fidelity;
        } else if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
            math_fidelity = compute_kernel_config.math_fidelity;
        } else {
            TT_FATAL("arch not supported");
        }
    }, compute_kernel_config);


    int ideal_dev_clock_cycles = std::ceil(((float)num_mul_adds / (float)(num_cores * tensix_mul_adds_per_cycle_lofi)) * (float)operation::OpPerformanceModel::fidelity_multiplier(math_fidelity));

    operation::OpPerformanceModel result(input_tensors, output_tensors, ideal_dev_clock_cycles);
#if 0
    tt::log_info(tt::LogOp, "Matmul PerfModel:");
    tt::log_info(tt::LogOp, "\t Batch: ({}, {})", out_shape[0], out_shape[1]);
    tt::log_info(tt::LogOp, "\t In A (H, W): ({}, {})", in_a_shape[2], in_a_shape[3]);
    tt::log_info(tt::LogOp, "\t In B (H, W): ({}, {})", in_b_shape[2], in_b_shape[3]);
    tt::log_info(tt::LogOp, "\t Out (H, W): ({}, {})", out_shape[2], out_shape[3]);
    tt::log_info(tt::LogOp, "\t ideal_dev_clock_cycles: {}", ideal_dev_clock_cycles);
#endif
    return result;
    }
}
namespace bmm_op_utils {
using namespace tt;
using namespace tt::tt_metal;


tuple<uint32_t, uint32_t, uint32_t, uint32_t> get_large_matmul_params(uint32_t Mt, uint32_t Nt, uint32_t num_cores_y, uint32_t num_cores_x, uint32_t in0_block_w) {
    auto Nt_fac = _get_prime_factors(Nt);
    auto Mt_fac = _get_prime_factors(Mt);
    uint32_t Npc_min = 1;
    uint32_t Mpc_min = 1;

    for (auto it = Nt_fac.begin(); it != Nt_fac.end(); ++it) {
        auto ele = *it;
        if (ele > num_cores_x) {
            Npc_min *= ele;
            Nt_fac.erase(it);
            --it;
        }
    }
    for (auto it = Mt_fac.begin(); it != Mt_fac.end(); ++it) {
        auto ele = *it;
        if (ele > num_cores_y) {
            Mpc_min *= ele;
            Mt_fac.erase(it);
            --it;
        }
    }

    if (Npc_min > _get_maximum_block_dim(Mpc_min, in0_block_w))
        return {0, 0, 0, 0};

    uint32_t Mpc = Mpc_min;
    uint32_t Npc = Npc_min;
    if (Mpc_min > 1) {
        auto Npc_choices = _get_possible_products(Nt_fac);
        auto Npc_max = _get_maximum_block_dim(Mpc_min, in0_block_w);
        for (auto &ele : Npc_choices) {
            if (ele *  Npc_min <= Npc_max)
                Npc = ele * Npc_min;
            else
                break;
        }

        if (Mt / Mpc > num_cores_y or Nt / Npc > num_cores_x)
            return {0, 0, 0, 0};

        for (auto &subblock_hw : SUBBLOCK_HW_CHOICES) {
            auto subblock_h = std::get<0>(subblock_hw);
            auto subblock_w = std::get<1>(subblock_hw);
            if (Mpc % subblock_h == 0 and Npc % subblock_w == 0)
                return {Mpc, Npc, subblock_h, subblock_w};
        }
    }

    else if (Npc_min > 1) {
        auto Mpc_choices = _get_possible_products(Mt_fac);
        auto Mpc_max = _get_maximum_block_dim(Npc_min, in0_block_w);
        for (auto &ele : Mpc_choices) {
            if (ele *  Mpc_min <= Mpc_max)
                Mpc = ele * Mpc_min;
            else
                break;
        }

        if (Mt / Mpc > num_cores_y or Nt / Npc > num_cores_x) {
            return {0, 0, 0, 0};
        }

        for (auto &subblock_hw : SUBBLOCK_HW_CHOICES) {
            auto subblock_h = std::get<0>(subblock_hw);
            auto subblock_w = std::get<1>(subblock_hw);
            if (Mpc % subblock_h == 0 and Npc % subblock_w == 0)
                return {Mpc, Npc, subblock_h, subblock_w};
        }
    }

    else {
        auto Mpc_choices = _get_possible_products(Mt_fac);
        auto Npc_choices = _get_possible_products(Nt_fac);
        for (auto &Npc : Npc_choices) {
            auto Mpc_max = _get_maximum_block_dim(Npc, in0_block_w);
            for (auto &ele : Mpc_choices) {
                if (ele <= Mpc_max)
                    Mpc = ele;
            }

            if (Mt / Mpc > num_cores_y or Nt / Npc > num_cores_x)
                continue;

            for (auto &subblock_hw : SUBBLOCK_HW_CHOICES) {
                auto subblock_h = std::get<0>(subblock_hw);
                auto subblock_w = std::get<1>(subblock_hw);
                if (Mpc % subblock_h == 0 and Npc % subblock_w == 0)
                    return {Mpc, Npc, subblock_h, subblock_w};
            }
        }
    }

    return {0, 0, 0, 0};
}


CoreCoord get_core_range(uint32_t num_blocks_rows, uint32_t num_blocks_cols, uint32_t max_num_rows, uint32_t max_num_cols) {
    CoreCoord core_range(0, 0);
    if (!(num_blocks_rows == 1 && num_blocks_cols == 1) && num_blocks_rows <= max_num_rows && num_blocks_cols <= max_num_cols) {
        core_range.x = num_blocks_cols;
        core_range.y = num_blocks_rows;
    }
    return core_range;
}

MatmulParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    const auto& ashape = input_tensor_a.get_legacy_shape(), bshape = input_tensor_b.get_legacy_shape();
    uint32_t num_output_tiles = ashape[0] * ashape[1] * ashape[2] * bshape[3] / TILE_HW; // Output M x N

    // Parameters for large matmul with reuse
    uint32_t B = ashape[0] * ashape[1];
    uint32_t Mt = ashape[2]/TILE_HEIGHT;
    uint32_t Kt = ashape[3]/TILE_WIDTH;
    uint32_t Nt = bshape[3]/TILE_WIDTH;
    uint32_t in0_block_w = 2;

    tt::tt_metal::Device *device = input_tensor_a.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    uint32_t per_core_M, per_core_N, out_subblock_h, out_subblock_w;
    uint32_t num_blocks_x, num_blocks_y;

    // out_subblock h/w doesn't matter
    per_core_M = 16;
    per_core_N = 16;

    // Calculate number of blocks along x and y; tensor dims are padded up to 512
    num_blocks_y = (Mt - 1) / per_core_M + 1;
    num_blocks_x = (Nt - 1) / per_core_N + 1;

    if (num_blocks_x * num_blocks_y <= num_cores_x * num_cores_y and Kt % in0_block_w == 0) {
        CoreCoord core_range = get_core_range(num_blocks_y, num_blocks_x, num_cores_y, num_cores_x);
        if (core_range.y == 1) {
            return MatmulParallelizationStrategy::MULTI_CORE_REUSE_MCAST_1D_IN0_OPTIMIZED;
        } else if (core_range.x == 1) {
            return MatmulParallelizationStrategy::MULTI_CORE_REUSE_MCAST_1D_IN1_OPTIMIZED;
        } else if (core_range.y > 0) {
            return MatmulParallelizationStrategy::MULTI_CORE_REUSE_MCAST_2D_OPTIMIZED;
        }
        // If we don't need padding, use the default multi_core reuse/reuse_mcast
        else if (Mt % per_core_M == 0 and Nt % per_core_N == 0) {
            return MatmulParallelizationStrategy::MULTI_CORE_REUSE;
        } else {
            return MatmulParallelizationStrategy::MULTI_CORE;
        }
    }
    else if (num_output_tiles > 1) {
        return MatmulParallelizationStrategy::MULTI_CORE;
    }
    else {
        return MatmulParallelizationStrategy::SINGLE_CORE;
    }
}

tt::operations::primary::MatmulMultiCoreReuseMultiCast1DProgramConfig get_mcast_1d_config(const Tensor &input_tensor_a, const Tensor &input_tensor_b, bool fuse_batch, std::optional<UnaryWithParam> fused_activation, bool mcast_in0, bool out_sharded) {
    auto device = input_tensor_a.device();
    auto grid_size = device->compute_with_storage_grid_size();
    uint32_t M = fuse_batch ? input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] : input_tensor_a.get_legacy_shape()[-2];
    uint32_t K = input_tensor_a.get_legacy_shape()[-1];
    uint32_t N = input_tensor_b.get_legacy_shape()[-1];
    uint32_t per_core_M, per_core_N;
    if (mcast_in0) {
        per_core_M = M / TILE_HEIGHT;
        per_core_N = div_up(div_up(N, grid_size.x * grid_size.y), TILE_WIDTH);
    } else {
        per_core_M = div_up(div_up(M, grid_size.x * grid_size.y), TILE_HEIGHT);
        per_core_N = N / TILE_WIDTH;
    }
    uint32_t in0_block_w = K / TILE_WIDTH % 2 == 0 ? 2 : 1;
    // TODO: Replace with get_matmul_subblock_params
    uint32_t out_subblock_h, out_subblock_w;
    bool params_found = false;
    for (auto &subblock_hw : bmm_op_utils::SUBBLOCK_HW_CHOICES) {
        out_subblock_h = std::get<0>(subblock_hw);
        out_subblock_w = std::get<1>(subblock_hw);
        if (out_sharded) {
            if (!mcast_in0) {
                if (out_subblock_w != per_core_N || out_subblock_h != 1) {
                    continue;
                }
            } else {
                if (out_subblock_h != per_core_M || out_subblock_w != 1) {
                    continue;
                }
            }
        }
        if (per_core_M % out_subblock_h == 0 and per_core_N % out_subblock_w == 0) {
            params_found = true;
            break;
        }
    }
    TT_FATAL(params_found, "Matmul parameters could not be determined for given input shapes");

    return tt::operations::primary::MatmulMultiCoreReuseMultiCast1DProgramConfig{
        .compute_with_storage_grid_size = grid_size,
        .in0_block_w = in0_block_w,
        .out_subblock_h = out_subblock_h,
        .out_subblock_w = out_subblock_w,
        .per_core_M = per_core_M,
        .per_core_N = per_core_N,
        .fuse_batch = fuse_batch,
        .fused_activation = fused_activation,
        .mcast_in0 = mcast_in0
    };
}

tuple<uint32_t, uint32_t> get_matmul_subblock_params(const uint32_t per_core_M, const uint32_t per_core_N, const bool per_core_M_equals_subblock_h_constraint, bool per_core_N_equals_subblock_w_constraint) {
    TT_FATAL(!(per_core_M_equals_subblock_h_constraint and per_core_N_equals_subblock_w_constraint), "Only one constraint may be true for h or w!");

    uint32_t out_subblock_h, out_subblock_w;
    for (auto &subblock_hw : bmm_op_utils::SUBBLOCK_HW_CHOICES) {
        out_subblock_h = std::get<0>(subblock_hw);
        out_subblock_w = std::get<1>(subblock_hw);
        if (per_core_N_equals_subblock_w_constraint) {
            if (out_subblock_w != per_core_N || out_subblock_h != 1) {
                continue;
            }
        }
        if (per_core_M_equals_subblock_h_constraint) {
            if (out_subblock_h != per_core_M || out_subblock_w != 1) {
                continue;
            }
        }
        if (per_core_M % out_subblock_h == 0 and per_core_N % out_subblock_w == 0) {
            return {out_subblock_h, out_subblock_w};
        }
    }

    TT_FATAL(false, "Matmul subblock parameters could not be determined for given input shapes");
}


// TODO: Only supports sharded matmul for now; infer most matmul params from shard spec
tt::operations::primary::MatmulProgramConfig get_matmul_program_config(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const MemoryConfig &output_mem_config, std::optional<UnaryWithParam> fused_activation, const bool matmul) {
    TT_FATAL(input_tensor_a.is_sharded());
    // TODO: Should we check if grid_size is valid against device->compute_with_storage_grid_size()?
    auto grid_size = input_tensor_a.shard_spec().value().grid.bounding_box().grid_size();

    // MCAST matmuls only support input_b in INTERLEAVED
    if (matmul) {
        TT_FATAL(input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        if ((input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED
              or input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED)
            and (grid_size.x > 1 or grid_size.y > 1)) {
            TT_FATAL(input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR);

            bool per_core_N_equals_subblock_w_constraint = false;
            if (output_mem_config.is_sharded()) {
                TT_FATAL(input_tensor_a.memory_config().buffer_type == output_mem_config.buffer_type);
                TT_FATAL(input_tensor_a.memory_config().memory_layout == output_mem_config.memory_layout);
                per_core_N_equals_subblock_w_constraint = true;
            }

            uint32_t M = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / TILE_HEIGHT;
            uint32_t K = input_tensor_a.get_legacy_shape()[-1] / TILE_WIDTH;
            uint32_t N = input_tensor_b.get_legacy_shape()[-1] / TILE_WIDTH;
            auto shard_shape = input_tensor_a.shard_spec().value().shape;

            bool mcast_in0;
            uint32_t per_core_M;
            uint32_t per_core_N;
            uint32_t in0_block_w;
            if (input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
                mcast_in0 = true;
                per_core_M = M;
                per_core_N = div_up(N, input_tensor_a.shard_spec().value().grid.num_cores());
                in0_block_w = shard_shape[1] / TILE_WIDTH;
            } else if (input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
                mcast_in0 = false;
                per_core_M = shard_shape[0] / TILE_HEIGHT;
                per_core_N = N; // Only necessary if output is sharded; otherwise, can set this to be < N
                in0_block_w = K;
            } else {
                TT_FATAL(false, "Input tensor must be WIDTH or HEIGHT sharded for 1D mcast matmul!");
            }

            auto subblock_hw = get_matmul_subblock_params(per_core_M, per_core_N, false, per_core_N_equals_subblock_w_constraint);
            auto out_subblock_h = std::get<0>(subblock_hw);
            auto out_subblock_w = std::get<1>(subblock_hw);

            return tt::operations::primary::MatmulMultiCoreReuseMultiCast1DProgramConfig{
                .compute_with_storage_grid_size = grid_size,
                .in0_block_w = in0_block_w,
                .out_subblock_h = out_subblock_h,
                .out_subblock_w = out_subblock_w,
                .per_core_M = per_core_M,
                .per_core_N = per_core_N,
                .fuse_batch = true,
                .fused_activation = fused_activation,
                .mcast_in0 = mcast_in0,
            };
        } else if (input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED
                   and (grid_size.x > 1 and grid_size.y > 1)) {
            bool transpose_mcast = input_tensor_a.shard_spec().value().orientation == ShardOrientation::COL_MAJOR;

            bool per_core_N_equals_subblock_w_constraint = false;
            if (output_mem_config.is_sharded()) {
                TT_FATAL(input_tensor_a.memory_config().buffer_type == output_mem_config.buffer_type);
                TT_FATAL(input_tensor_a.memory_config().memory_layout == output_mem_config.memory_layout);
                per_core_N_equals_subblock_w_constraint = true;
            }

            uint32_t M = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / TILE_HEIGHT;
            uint32_t K = input_tensor_a.get_legacy_shape()[-1] / TILE_WIDTH;
            uint32_t N = input_tensor_b.get_legacy_shape()[-1] / TILE_WIDTH;

            auto shard_shape = input_tensor_a.shard_spec().value().shape;
            uint32_t virtual_x = transpose_mcast ? grid_size.y : grid_size.x;
            uint32_t virtual_y = transpose_mcast ? grid_size.x : grid_size.y;
            TT_FATAL(virtual_y == (M / (shard_shape[0] / TILE_HEIGHT)), "Num cores along y must match provided grid size!");
            TT_FATAL(virtual_x == (K / (shard_shape[1] / TILE_WIDTH)), "Num cores along x must match provided grid size!");

            uint32_t per_core_M = M / virtual_y;
            uint32_t per_core_N = N / virtual_x;
            uint32_t in0_block_w = shard_shape[1] / TILE_WIDTH;

            auto subblock_hw = get_matmul_subblock_params(per_core_M, per_core_N, false, per_core_N_equals_subblock_w_constraint);
            auto out_subblock_h = std::get<0>(subblock_hw);
            auto out_subblock_w = std::get<1>(subblock_hw);

            return tt::operations::primary::MatmulMultiCoreReuseMultiCastProgramConfig{
                .compute_with_storage_grid_size = grid_size,
                .in0_block_w = in0_block_w,
                .out_subblock_h = out_subblock_h,
                .out_subblock_w = out_subblock_w,
                .per_core_M = per_core_M,
                .per_core_N = per_core_N,
                .transpose_mcast = transpose_mcast,
                .fused_activation = fused_activation,
            };
        }
    } else {
        // TODO: Need a better criteria for BMMs and MatmulMultiCoreReuseProgramConfig
        TT_FATAL(input_tensor_a.memory_config().memory_layout != TensorMemoryLayout::WIDTH_SHARDED);

        bool per_core_N_equals_subblock_w_constraint = false;
        if (output_mem_config.is_sharded()) {
            TT_FATAL(input_tensor_a.memory_config().buffer_type == output_mem_config.buffer_type);
            TT_FATAL(input_tensor_a.memory_config().memory_layout == output_mem_config.memory_layout);
            per_core_N_equals_subblock_w_constraint = true;
        }

        uint32_t M = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / TILE_HEIGHT;
        uint32_t K = input_tensor_a.get_legacy_shape()[-1] / TILE_WIDTH;
        uint32_t N = input_tensor_b.get_legacy_shape()[-1] / TILE_WIDTH;

        auto in0_shard_shape = input_tensor_a.shard_spec().value().shape;
        uint32_t per_core_M = in0_shard_shape[0] / TILE_HEIGHT;
        uint32_t per_core_N = N;
        uint32_t in0_block_w = in0_shard_shape[1] / TILE_WIDTH;

        auto subblock_hw = get_matmul_subblock_params(per_core_M, per_core_N, false, per_core_N_equals_subblock_w_constraint);
        auto out_subblock_h = std::get<0>(subblock_hw);
        auto out_subblock_w = std::get<1>(subblock_hw);

        // TODO: Temporarily allow for single core; should support bcast_batch in general
        bool broadcast_batch = (
            input_tensor_a.get_legacy_shape()[0] * input_tensor_a.get_legacy_shape()[1] > 1
            and input_tensor_b.get_legacy_shape()[0] * input_tensor_b.get_legacy_shape()[1] == 1
        );
        TT_FATAL(!broadcast_batch);

        if (input_tensor_b.is_sharded()) {
            TT_FATAL(input_tensor_a.memory_config().buffer_type == input_tensor_b.memory_config().buffer_type);
            TT_FATAL(input_tensor_a.memory_config().memory_layout == input_tensor_b.memory_config().memory_layout);
            TT_FATAL(input_tensor_a.shard_spec().value().grid == input_tensor_b.shard_spec().value().grid);
            TT_FATAL(input_tensor_a.shard_spec().value().orientation == input_tensor_b.shard_spec().value().orientation);
        }

        return tt::operations::primary::MatmulMultiCoreReuseProgramConfig{
            .compute_with_storage_grid_size = grid_size,
            .in0_block_w = in0_block_w,
            .out_subblock_h = out_subblock_h,
            .out_subblock_w = out_subblock_w,
            .per_core_M = per_core_M,
            .per_core_N = per_core_N,
        };
    }
    TT_FATAL(false, "Matmul program config could not be determined for given input shapes!");
}

} // namespace bmm_op_utils

namespace tt {
namespace tt_metal {

void Matmul::validate(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    TT_FATAL((input_tensor_a.get_layout() == Layout::TILE && input_tensor_b.get_layout() == Layout::TILE), "Inputs to matmul must be tilized");
    if (this->bcast_batch) {
        TT_FATAL(input_tensor_b.get_legacy_shape()[0] * input_tensor_b.get_legacy_shape()[1] == 1 && "matmul (batch bcast variant) expects input tensors of shapes BCMK*11KN=BCMN");
    } else {
        TT_FATAL(input_tensor_a.get_legacy_shape()[1] == input_tensor_b.get_legacy_shape()[1] && input_tensor_a.get_legacy_shape()[0] == input_tensor_b.get_legacy_shape()[0] && "bmm (non-bcast matmul) expects input tensors of shapes BCMK*BCKN=BCMN");
    }
    TT_FATAL(input_tensor_a.get_legacy_shape()[3] == input_tensor_b.get_legacy_shape()[2] && "Dimension K (A.shape[3] and B.shape[2]) must match for A and B in bmm_op"); // A.K == B.K

    // TODO: Uplift get_parallelization_strategy to be struct param? We should do separate dtype validations for different parallelizations
    // This requires sweeping across shapes with different dtypes/dataformats; for now, ignore dtype assertions here and uplift to actual matmul/bmm implementations
    TT_FATAL(input_tensor_a.get_dtype() == tt::tt_metal::DataType::BFLOAT16 || input_tensor_a.get_dtype() == tt::tt_metal::DataType::BFLOAT8_B, "Unsupported data format");
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE and input_tensor_b.storage_type() == StorageType::DEVICE, "Operands to matmul need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr and input_tensor_b.buffer() != nullptr, "Operands to matmul need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.device() == input_tensor_b.device(), "Operands to matmul need to be on the same device!");

    if (this->untilize_out) {
        TT_FATAL(this->output_dtype == DataType::BFLOAT16);
    }
}

std::vector<Shape> Matmul::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto input_shape_a = input_tensors.at(0).get_legacy_shape();
    const auto input_shape_b = input_tensors.at(1).get_legacy_shape();

    auto output_shape = input_shape_a;
    output_shape[-1] = input_shape_b[-1];
    auto dimensions_pads = std::vector<Padding::PadDimension>();
    for (auto index = 0; index < input_shape_a.rank() - 1; index++) {
        dimensions_pads.push_back(input_shape_a.padding()[index]);
    }
    dimensions_pads.push_back(input_shape_b.padding()[input_shape_b.rank() - 1]);
    const auto padding = Padding(dimensions_pads, Padding::PadValue::Any);
    return {Shape(output_shape, padding)};
}

std::vector<Tensor> Matmul::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, this->output_dtype, Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks Matmul::create_program(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    auto& output_tensor = output_tensors.at(0);

    auto parallelization_strategy = this->get_parallelization_strategy(input_tensors);
    tt::operations::primary::MatmulMultiCoreReuseMultiCast1DProgramConfig config;
    switch (parallelization_strategy){
        case MatmulParallelizationStrategy::MULTI_CORE:
            return matmul_multi_core(input_tensor_a, input_tensor_b, output_tensor, this->bcast_batch);
        case MatmulParallelizationStrategy::MULTI_CORE_REUSE:
            return matmul_multi_core_reuse(input_tensor_a, input_tensor_b, output_tensor, this->bcast_batch);
        case MatmulParallelizationStrategy::MULTI_CORE_REUSE_MCAST_2D_OPTIMIZED:
            return matmul_multi_core_reuse_mcast_2d_optimized(
                input_tensor_a, input_tensor_b, std::nullopt, output_tensor,
                this->bcast_batch,
                input_tensor_a.device()->compute_with_storage_grid_size(),
                // MathFidelity::HiFi4, false, true, false,
                this->compute_kernel_config,
                2, 4, 2,
                16, 16, false, false, std::nullopt,
                this->untilize_out
            );
        case MatmulParallelizationStrategy::MULTI_CORE_REUSE_MCAST_1D_IN0_OPTIMIZED:
            config = bmm_op_utils::get_mcast_1d_config(input_tensor_a, input_tensor_b, false, std::nullopt, true, false);
            return matmul_multi_core_reuse_mcast_1d_optimized(
                input_tensor_a, input_tensor_b, std::nullopt, output_tensor,
                this->bcast_batch,
                input_tensor_a.device()->compute_with_storage_grid_size(),
                // MathFidelity::HiFi4, false, true, false,
                this->compute_kernel_config,
                config.in0_block_w, config.out_subblock_h, config.out_subblock_w,
                config.per_core_M, config.per_core_N, false, std::nullopt, true,
                this->untilize_out
            );
        case MatmulParallelizationStrategy::MULTI_CORE_REUSE_MCAST_1D_IN1_OPTIMIZED:
            config = bmm_op_utils::get_mcast_1d_config(input_tensor_a, input_tensor_b, false, std::nullopt, false, false);
            return matmul_multi_core_reuse_mcast_1d_optimized(
                input_tensor_a, input_tensor_b, std::nullopt, output_tensor,
                this->bcast_batch,
                input_tensor_a.device()->compute_with_storage_grid_size(),
                // MathFidelity::HiFi4, false, true, false,
                this->compute_kernel_config,
                config.in0_block_w, config.out_subblock_h, config.out_subblock_w,
                config.per_core_M, config.per_core_N, false, std::nullopt, false,
                this->untilize_out
            );
        case MatmulParallelizationStrategy::MULTI_CORE_REUSE_PADDING:
            return matmul_multi_core_reuse_padding(input_tensor_a, input_tensor_b, output_tensor, this->bcast_batch);
        case MatmulParallelizationStrategy::SINGLE_CORE:
        default:
            return matmul_single_core(input_tensor_a, input_tensor_b, output_tensor, this->bcast_batch);
    }

}

operation::OpPerformanceModel Matmul::create_op_performance_model(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor> &output_tensors
    ) const {
        return ::create_op_performance_model_for_matmul(input_tensors, optional_input_tensors, output_tensors, this->compute_kernel_config);
}
MatmulParallelizationStrategy Matmul::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    return bmm_op_utils::get_parallelization_strategy(input_tensors);
}

/**
 * Bert large matmuls using operations::primary::matmul + program_config
 */
Tensor bert_large_fused_qkv_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype) {
    auto batch_size = input_tensor_a.get_legacy_shape()[0];

    TT_FATAL((input_tensor_a.get_legacy_shape() == Shape({batch_size, 1, 384, 1024})), "Unsupported input shape");
    TT_FATAL((input_tensor_b.get_legacy_shape() == Shape({1, 1, 1024, 3072})), "Unsupported input shape");

    auto program_config = operations::primary::MatmulMultiCoreReuseMultiCastProgramConfig{
        .compute_with_storage_grid_size = {12, batch_size},
        .in0_block_w = 4,
        .out_subblock_h = 4,
        .out_subblock_w = 2,
        .per_core_M = 12,
        .per_core_N = 8,
        .transpose_mcast = false,
        .fused_activation = std::nullopt,
    };
    return operations::primary::matmul(input_tensor_a, input_tensor_b, bias, program_config, mem_config, output_dtype);
}

Tensor bert_large_ff1_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, std::optional<UnaryWithParam> fused_activation, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype) {
    auto batch_size = input_tensor_a.get_legacy_shape()[0];

    TT_FATAL((input_tensor_a.get_dtype() != DataType::BFLOAT16 or input_tensor_b.get_dtype() != DataType::BFLOAT16 or output_dtype != DataType::BFLOAT16) or (mem_config.buffer_type == BufferType::DRAM) or (input_tensor_a.memory_config().buffer_type == BufferType::DRAM and input_tensor_b.memory_config().buffer_type == BufferType::DRAM), "For BFLOAT16, if output is on L1, one of in0 or in1 must be on DRAM!");
    TT_FATAL((input_tensor_a.get_legacy_shape() == Shape({batch_size, 1, 384, 1024})), "Unsupported input shape");
    TT_FATAL((input_tensor_b.get_legacy_shape() == Shape({1, 1, 1024, 4096})), "Unsupported input shape");

    auto program_config = operations::primary::MatmulMultiCoreReuseMultiCastProgramConfig{
        .compute_with_storage_grid_size = {12, batch_size},
        .in0_block_w = 4,
        .out_subblock_h = 6,
        .out_subblock_w = 1,
        .per_core_M = 12,
        .per_core_N = 11,
        .transpose_mcast = false,
        .fused_activation=fused_activation,
    };
    return operations::primary::matmul(input_tensor_a, input_tensor_b, bias, program_config, mem_config, output_dtype);
}

Tensor bert_large_ff2_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype) {
    auto batch_size = input_tensor_a.get_legacy_shape()[0];

    TT_FATAL((input_tensor_a.get_legacy_shape() == Shape({batch_size, 1, 384, 4096})), "Unsupported input shape");
    TT_FATAL((input_tensor_b.get_legacy_shape() == Shape({1, 1, 4096, 1024})), "Unsupported input shape");

    auto program_config = operations::primary::MatmulMultiCoreReuseMultiCastProgramConfig{
        .compute_with_storage_grid_size = {12, batch_size},
        .in0_block_w = 4,
        .out_subblock_h = 6,
        .out_subblock_w = 1,
        .per_core_M = 12,
        .per_core_N = 3,
        .transpose_mcast = false,
        .fused_activation= std::nullopt,
    };
    return operations::primary::matmul(input_tensor_a, input_tensor_b, bias, program_config, mem_config, output_dtype);
}

Tensor bert_large_selfout_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype) {
    auto batch_size = input_tensor_a.get_legacy_shape()[0];

    TT_FATAL((input_tensor_a.get_legacy_shape() == Shape({batch_size, 1, 384, 1024})), "Unsupported input shape");
    TT_FATAL((input_tensor_b.get_legacy_shape() == Shape({1, 1, 1024, 1024})), "Unsupported input shape");

    auto program_config = operations::primary::MatmulMultiCoreReuseMultiCastProgramConfig{
        .compute_with_storage_grid_size = {12, batch_size},
        .in0_block_w = 4,
        .out_subblock_h = 6,
        .out_subblock_w = 1,
        .per_core_M = 12,
        .per_core_N = 3,
        .transpose_mcast = false,
        .fused_activation= std::nullopt,
    };
    return operations::primary::matmul(input_tensor_a, input_tensor_b, bias, program_config, mem_config, output_dtype);
}

Tensor bert_large_pre_softmax_bmm(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype) {
    auto batch_size = input_tensor_a.get_legacy_shape()[0];

    TT_FATAL((input_tensor_a.get_legacy_shape() == Shape({batch_size, 16, 384, 64})), "Unsupported input shape");
    TT_FATAL((input_tensor_b.get_legacy_shape() == Shape({batch_size, 16, 64, 384})), "Unsupported input shape");

    auto program_config = operations::primary::MatmulMultiCoreReuseProgramConfig{
        .compute_with_storage_grid_size = {12, batch_size},
        .in0_block_w = 1,
        .out_subblock_h = 4,
        .out_subblock_w = 2,
        .per_core_M = 12,
        .per_core_N = 12,
    };
    return operations::primary::matmul(input_tensor_a, input_tensor_b, program_config, mem_config, output_dtype);

}

Tensor bert_large_post_softmax_bmm(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype) {
    auto batch_size = input_tensor_a.get_legacy_shape()[0];

    TT_FATAL((input_tensor_a.get_legacy_shape() == Shape({batch_size, 16, 384, 384})), "Unsupported input shape");
    TT_FATAL((input_tensor_b.get_legacy_shape() == Shape({batch_size, 16, 384, 64})), "Unsupported input shape");

    auto program_config = operations::primary::MatmulMultiCoreReuseProgramConfig{
        .compute_with_storage_grid_size = {12, batch_size},
        .in0_block_w = 2,
        .out_subblock_h = 4,
        .out_subblock_w = 2,
        .per_core_M = 12,
        .per_core_N = 2,
    };
    return operations::primary::matmul(input_tensor_a, input_tensor_b, program_config, mem_config, output_dtype);

}

/**
 * Falcon matmuls using operations::primary::matmul + program_config
 */
Tensor falcon_fused_qkv_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype) {
    auto program_config = bmm_op_utils::get_mcast_1d_config(input_tensor_a, input_tensor_b, true, std::nullopt, true, mem_config.is_sharded());
    return operations::primary::matmul_1d(input_tensor_a, input_tensor_b, bias, program_config, mem_config, output_dtype);
}

Tensor falcon_selfout_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype) {
    auto program_config = bmm_op_utils::get_mcast_1d_config(input_tensor_a, input_tensor_b, true, std::nullopt, true, mem_config.is_sharded());
    return operations::primary::matmul_1d(input_tensor_a, input_tensor_b, bias, program_config, mem_config, output_dtype);
}

Tensor falcon_dense_4h_to_h_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype, std::optional<bool> packer_l1_acc) {
    auto program_config = bmm_op_utils::get_mcast_1d_config(
        input_tensor_a, input_tensor_b, true, std::nullopt, true, mem_config.is_sharded());
    std::optional<const DeviceComputeKernelConfig> config = std::nullopt;
    auto compute_kernel_config = init_device_compute_kernel_config(input_tensor_a.device()->arch(), config, MathFidelity::LoFi, true, false, packer_l1_acc.value_or(false));
    return operations::primary::matmul_1d(
        input_tensor_a,
        input_tensor_b,
        bias,
        program_config,
        mem_config,
        output_dtype,
        compute_kernel_config);
}

Tensor falcon_dense_h_to_4h_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, std::optional<UnaryWithParam> fused_activation, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype) {
    auto seq_len = input_tensor_a.get_legacy_shape()[2];
    if (seq_len > 1024) {
        TT_FATAL(not fused_activation.has_value());
        // TODO: Check support for seq_len == 128, 256, 512, ..., 2048
        TT_FATAL(seq_len % TILE_HEIGHT == 0, "Falcon mm's seq_len must be a multiple of 32!");
        TT_FATAL(seq_len >=  128, "Falcon mm's seq_len must be greater than 128!");
        TT_FATAL((input_tensor_a.get_legacy_shape() == Shape({1, 1, seq_len, 4544})), "Unsupported input shape");
        TT_FATAL((input_tensor_b.get_legacy_shape() == Shape({1, 1, 4544, 18176})), "Unsupported input shape");
        TT_FATAL(!fused_activation.has_value());
        return operation::run_with_autoformat(Matmul{.bcast_batch=true, .output_mem_config=mem_config, .output_dtype=output_dtype.value_or(input_tensor_a.get_dtype())}, {input_tensor_a, input_tensor_b}).at(0);
    } else {
        auto program_config = bmm_op_utils::get_mcast_1d_config(input_tensor_a, input_tensor_b, true, fused_activation, true, mem_config.is_sharded());
        std::optional<const DeviceComputeKernelConfig> config = std::nullopt;
        auto compute_kernel_config = init_device_compute_kernel_config(input_tensor_a.device()->arch(), config, MathFidelity::LoFi, true /* math_approx_mode */, false /* fp32_dest_acc_en */, true /* packer_l1_acc */);
        return operations::primary::matmul_1d(input_tensor_a, input_tensor_b, bias, program_config, mem_config, output_dtype, compute_kernel_config);
    }
}

Tensor falcon_lm_head_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype) {
    auto seq_len = input_tensor_a.get_legacy_shape()[2];

    if (seq_len > 512) {
        // TODO: Check support for seq_len == 128, 256, 512, ..., 2048
        TT_FATAL(seq_len % TILE_HEIGHT == 0, "Falcon mm's seq_len must be a multiple of 32!");
        TT_FATAL(seq_len >=  128, "Falcon mm's seq_len must be greater than 128!");
        TT_FATAL((input_tensor_a.get_legacy_shape() == Shape({1, 1, seq_len, 4544})), "Unsupported input shape");
        TT_FATAL((input_tensor_b.get_legacy_shape() == Shape({1, 1, 4544, 65024})), "Unsupported input shape");
        return operation::run_with_autoformat(Matmul{.bcast_batch=true, .output_mem_config=mem_config, .output_dtype=output_dtype.value_or(input_tensor_a.get_dtype())}, {input_tensor_a, input_tensor_b}, {bias}).at(0);
    } else {
        auto program_config = bmm_op_utils::get_mcast_1d_config(input_tensor_a, input_tensor_b, true, std::nullopt, true, mem_config.is_sharded());
        std::optional<const DeviceComputeKernelConfig> config = std::nullopt;
        auto compute_kernel_config = init_device_compute_kernel_config(input_tensor_a.device()->arch(), config, MathFidelity::LoFi, true /* math_approx_mode */, false /* fp32_dest_acc_en */, true /* packer_l1_acc */);
        return operations::primary::matmul_1d(input_tensor_a, input_tensor_b, bias, program_config, mem_config, output_dtype, compute_kernel_config);
    }
}

/**
 * Resnet50 matmul with fused batch
 */
Tensor resnet_matmul(const Tensor& input_a, const Tensor& input_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype, const MathFidelity math_fidelity) {
    auto program_config = bmm_op_utils::get_mcast_1d_config(input_a, input_b, true);
    std::optional<const DeviceComputeKernelConfig> config = std::nullopt;
    auto compute_kernel_config = init_device_compute_kernel_config(input_a.device()->arch(), config, math_fidelity, true, false, false);
    return operations::primary::matmul_1d(input_a, input_b, bias, program_config, mem_config, output_dtype, compute_kernel_config);
}


}  // namespace tt_metal



namespace operations {

namespace primary {

void Matmul::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors
) const {

    TT_FATAL(input_tensors.size() == 2);
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    TT_FATAL((input_tensor_a.get_layout() == Layout::TILE && input_tensor_b.get_layout() == Layout::TILE), "Inputs to matmul must be tilized");
    TT_FATAL(input_tensor_a.get_legacy_shape()[3] == input_tensor_b.get_legacy_shape()[2] && "Dimension K (A.shape[3] and B.shape[2]) must match for A and B in bmm_op"); // A.K == B.K

    TT_FATAL(input_tensor_a.get_dtype() == tt::tt_metal::DataType::FLOAT32 || input_tensor_a.get_dtype() == tt::tt_metal::DataType::BFLOAT16 || input_tensor_a.get_dtype() == tt::tt_metal::DataType::BFLOAT8_B, "Unsupported data format");
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE and input_tensor_b.storage_type() == StorageType::DEVICE, "Operands to matmul need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr and input_tensor_b.buffer() != nullptr, "Operands to matmul need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.device() == input_tensor_b.device(), "Operands to matmul need to be on the same device!");

    std::visit(
        [input_tensor_a](const auto& program_config) {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (
                not std::is_same_v<ProgramConfigType, MatmulDefaultProgramConfig>
            ) {
                TT_FATAL(program_config.compute_with_storage_grid_size.x <= input_tensor_a.device()->compute_with_storage_grid_size().x, "Matmul grid size exceeds maximum device compute grid size!");
                TT_FATAL(program_config.compute_with_storage_grid_size.y <= input_tensor_a.device()->compute_with_storage_grid_size().y, "Matmul grid size exceeds maximum device compute grid size!");
            }
        },
        this->program_config
    );

    TT_FATAL(optional_input_tensors.size() == 1);
    const auto& optional_bias = optional_input_tensors.at(0);
    if (optional_bias.has_value()) {
        const auto& bias = optional_bias.value();
        TT_FATAL(bias.get_layout() == Layout::TILE, "Unsupported input layout");
        TT_FATAL(bias.get_legacy_shape() == Shape({1, 1, TILE_HEIGHT, input_tensor_b.get_legacy_shape()[3]}), "Unsupported bias shape");
    }

    if (this->untilize_out) {
        TT_FATAL((this->output_dtype == DataType::BFLOAT16) || (this->output_dtype == DataType::FLOAT32));
    }

    std::visit(
        [
            input_tensor_a,
            input_tensor_b,
            this
        ](const auto& program_config) {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            // TODO: For 1D and 2D mcasts, we don't check if tensor is single core or single row/col
            // We can uplift these variants to skip mcasting to support single core (1D) or single row/col (2D)
            if constexpr (
                std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCast1DProgramConfig>
            ) {
                if (program_config.mcast_in0) {
                    if (input_tensor_a.is_sharded()) {
                        TT_FATAL(program_config.fuse_batch);
                        TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED);
                        if (this->output_mem_config.is_sharded()) {
                            TT_FATAL(input_tensor_a.memory_config().buffer_type == this->output_mem_config.buffer_type);
                            TT_FATAL(input_tensor_a.memory_config().memory_layout == this->output_mem_config.memory_layout);
                        }
                        TT_FATAL(input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR);
                        uint32_t M = (program_config.fuse_batch ? input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] : input_tensor_a.get_legacy_shape()[-2]) / TILE_HEIGHT;
                        uint32_t N = input_tensor_b.get_legacy_shape()[-1] / TILE_WIDTH;
                        uint32_t K = input_tensor_a.get_legacy_shape()[-1] / TILE_WIDTH;
                        uint32_t per_core_M = program_config.per_core_M;
                        uint32_t per_core_N = program_config.per_core_N;
                        auto shard_shape = input_tensor_a.shard_spec().value().shape;

                        // No padding
                        TT_FATAL(M == per_core_M);
                        TT_FATAL(per_core_M == (shard_shape[0] / TILE_HEIGHT));
                        TT_FATAL(K % program_config.in0_block_w == 0);
                        TT_FATAL((shard_shape[1] / TILE_WIDTH) % program_config.in0_block_w == 0);
                        TT_FATAL(div_up(N, per_core_N) == input_tensor_a.shard_spec().value().grid.num_cores());
                    }
                    if (this->output_mem_config.is_sharded()) {
                        TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::WIDTH_SHARDED);
                        uint32_t M = (program_config.fuse_batch ? input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] : input_tensor_a.get_legacy_shape()[-2]) / TILE_HEIGHT;
                        uint32_t N = input_tensor_b.get_legacy_shape()[-1] / TILE_WIDTH;
                        uint32_t per_core_M = program_config.per_core_M;
                        uint32_t per_core_N = program_config.per_core_N;

                        // No padding
                        TT_FATAL(M == per_core_M);

                        TT_FATAL(program_config.out_subblock_w == per_core_N || program_config.out_subblock_h == 1);
                    }
                } else {
                    if (input_tensor_a.memory_config().is_sharded()) {
                        TT_FATAL(program_config.fuse_batch);
                        TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
                        if (this->output_mem_config.is_sharded()) {
                            TT_FATAL(input_tensor_a.memory_config().buffer_type == this->output_mem_config.buffer_type);
                            TT_FATAL(input_tensor_a.memory_config().memory_layout == this->output_mem_config.memory_layout);
                        }
                        TT_FATAL(input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR);
                        uint32_t M = (program_config.fuse_batch ? input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] : input_tensor_a.get_legacy_shape()[-2]) / TILE_HEIGHT;
                        uint32_t K = input_tensor_a.get_legacy_shape()[-1] / TILE_WIDTH;
                        uint32_t per_core_M = program_config.per_core_M;
                        auto shard_shape = input_tensor_a.shard_spec().value().shape;

                        TT_FATAL(div_up(M, per_core_M) == input_tensor_a.shard_spec().value().grid.num_cores());
                        TT_FATAL(per_core_M == (shard_shape[0] / TILE_HEIGHT));
                        TT_FATAL(K == program_config.in0_block_w);
                        TT_FATAL(program_config.in0_block_w == (shard_shape[1] / TILE_WIDTH));
                    }
                    if (this->output_mem_config.is_sharded()) {
                        TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
                        uint32_t M = (program_config.fuse_batch ? input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] : input_tensor_a.get_legacy_shape()[-2]) / TILE_HEIGHT;
                        uint32_t N = input_tensor_b.get_legacy_shape()[-1] / TILE_WIDTH;
                        uint32_t per_core_M = program_config.per_core_M;
                        uint32_t per_core_N = program_config.per_core_N;

                        TT_FATAL(N == per_core_N);
                        TT_FATAL(program_config.out_subblock_w == per_core_N || program_config.out_subblock_h == 1);
                    }
                }
                TT_FATAL(input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
            } else if constexpr (
                std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCastProgramConfig>
            ) {
                if (input_tensor_a.memory_config().is_sharded()) {
                    TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED);
                    if (program_config.transpose_mcast) {
                        TT_FATAL(input_tensor_a.shard_spec().value().orientation == ShardOrientation::COL_MAJOR);
                    } else {
                        TT_FATAL(input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR);
                    }
                    if (this->output_mem_config.is_sharded()) {
                        TT_FATAL(input_tensor_a.memory_config().buffer_type == this->output_mem_config.buffer_type);
                        TT_FATAL(input_tensor_a.memory_config().memory_layout == this->output_mem_config.memory_layout);
                    }

                    uint32_t M = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / TILE_HEIGHT;
                    uint32_t K = input_tensor_a.get_legacy_shape()[-1] / TILE_WIDTH;
                    uint32_t N = input_tensor_b.get_legacy_shape()[-1] / TILE_WIDTH;
                    uint32_t per_core_M = program_config.per_core_M;
                    auto shard_shape = input_tensor_a.shard_spec().value().shape;

                    TT_FATAL(per_core_M == (shard_shape[0] / TILE_HEIGHT));
                    TT_FATAL((shard_shape[1] / TILE_WIDTH) % program_config.in0_block_w == 0);
                    TT_FATAL(K / (shard_shape[1] / TILE_WIDTH) == N / program_config.per_core_N);
                }
                if (this->output_mem_config.is_sharded()) {
                    TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::BLOCK_SHARDED);
                    uint32_t M = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / TILE_HEIGHT;
                    uint32_t N = input_tensor_b.get_legacy_shape()[-1] / TILE_WIDTH;
                    uint32_t per_core_M = program_config.per_core_M;
                    uint32_t per_core_N = program_config.per_core_N;

                    TT_FATAL(program_config.out_subblock_w == per_core_N || program_config.out_subblock_h == 1);
                }
                TT_FATAL(input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
            } else if constexpr (
                std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseProgramConfig>
            ) {
                uint32_t M = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / TILE_HEIGHT;
                uint32_t N = input_tensor_b.get_legacy_shape()[-1] / TILE_WIDTH;
                uint32_t K = input_tensor_a.get_legacy_shape()[-1];
                uint32_t per_core_M = program_config.per_core_M;
                uint32_t per_core_N = program_config.per_core_N;
                TT_FATAL(per_core_M % (input_tensor_a.get_legacy_shape()[-2] / TILE_HEIGHT) == 0);
                TT_FATAL(N == per_core_N);
                if (input_tensor_a.is_sharded()) {
                    TT_FATAL(input_tensor_a.memory_config().memory_layout != TensorMemoryLayout::WIDTH_SHARDED);
                    auto in0_shard_shape = input_tensor_a.shard_spec().value().shape;

                    TT_FATAL(K == in0_shard_shape[1]);
                    TT_FATAL(in0_shard_shape[1] == program_config.in0_block_w * TILE_WIDTH);
                    TT_FATAL(per_core_M * TILE_HEIGHT == in0_shard_shape[0]);

                    if (input_tensor_b.is_sharded()) {
                        TT_FATAL(input_tensor_a.memory_config().buffer_type == input_tensor_b.memory_config().buffer_type);
                        TT_FATAL(input_tensor_a.memory_config().memory_layout == input_tensor_b.memory_config().memory_layout);
                        TT_FATAL(input_tensor_a.shard_spec().value().grid == input_tensor_b.shard_spec().value().grid);
                        TT_FATAL(input_tensor_a.shard_spec().value().orientation == input_tensor_b.shard_spec().value().orientation);
                    }
                    if (this->output_mem_config.is_sharded()) {
                        TT_FATAL(input_tensor_a.memory_config().buffer_type == this->output_mem_config.buffer_type);
                        TT_FATAL(input_tensor_a.memory_config().memory_layout == this->output_mem_config.memory_layout);
                    }
                }

                bool broadcast_batch = (
                    input_tensor_a.get_legacy_shape()[0] * input_tensor_a.get_legacy_shape()[1] > 1
                    and input_tensor_b.get_legacy_shape()[0] * input_tensor_b.get_legacy_shape()[1] == 1
                );
                TT_FATAL(!broadcast_batch);

                if (input_tensor_b.is_sharded()) {
                    TT_FATAL(input_tensor_b.memory_config().memory_layout != TensorMemoryLayout::WIDTH_SHARDED);
                    auto in1_shard_shape = input_tensor_b.shard_spec().value().shape;
                    TT_FATAL(in1_shard_shape[1] == input_tensor_b.get_legacy_shape()[-1]);
                    TT_FATAL(per_core_N * TILE_HEIGHT == in1_shard_shape[1]);
                    TT_FATAL(in1_shard_shape[0] % K == 0);
                }
                if (this->output_mem_config.is_sharded()) {
                    TT_FATAL(this->output_mem_config.memory_layout != TensorMemoryLayout::WIDTH_SHARDED);
                    TT_FATAL(program_config.out_subblock_w == per_core_N || program_config.out_subblock_h == 1);
                }
                TT_FATAL(M % per_core_M == 0);
                TT_FATAL(N % per_core_N == 0);
            } else {
                TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
                TT_FATAL(input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
                TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
            }
            if constexpr (
                std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseProgramConfig> ||
                std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCastProgramConfig> ||
                std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCast1DProgramConfig>
            ) {
                TT_FATAL((input_tensor_a.get_legacy_shape()[-1] / TILE_WIDTH) % program_config.in0_block_w == 0, "Kt must be divisible by in0_block_w");
                TT_FATAL(program_config.per_core_M % program_config.out_subblock_h == 0, "per_core_M must be divisible by out_subblock_h");
                TT_FATAL(program_config.per_core_N % program_config.out_subblock_w == 0, "per_core_N must be divisible by out_subblock_w");
            }
        },
        this->program_config
    );

}

std::vector<Shape> Matmul::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto input_shape_a = input_tensors.at(0).get_legacy_shape();
    const auto input_shape_b = input_tensors.at(1).get_legacy_shape();

    auto output_shape = input_shape_a;
    output_shape[-1] = input_shape_b[-1];
    auto dimensions_pads = std::vector<Padding::PadDimension>();
    for (auto index = 0; index < input_shape_a.rank() - 1; index++) {
        dimensions_pads.push_back(input_shape_a.padding()[index]);
    }
    dimensions_pads.push_back(input_shape_b.padding()[input_shape_b.rank() - 1]);
    const auto padding = Padding(dimensions_pads, Padding::PadValue::Any);
    return {Shape(output_shape, padding)};
}

std::vector<Tensor> Matmul::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    auto output_layout = this->untilize_out ? Layout::ROW_MAJOR : Layout::TILE;
    if (this->output_mem_config.is_sharded()) {
        return std::visit(
            [&](const auto& program_config) -> std::vector<Tensor> {
                using ProgramConfigType = std::decay_t<decltype(program_config)>;
                if constexpr (
                    std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCast1DProgramConfig>
                ) {
                    uint32_t M = (program_config.fuse_batch ? input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] : input_tensor_a.get_legacy_shape()[-2]) / TILE_HEIGHT;
                    uint32_t N = input_tensor_b.get_legacy_shape()[-1]/TILE_WIDTH;
                    uint32_t per_core_M = program_config.per_core_M;
                    uint32_t per_core_N = program_config.per_core_N;

                    uint32_t num_blocks_y = (M - 1) / per_core_M + 1;
                    uint32_t num_blocks_x = (N - 1) / per_core_N + 1;
                    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
                    uint32_t num_cores = num_blocks_x * num_blocks_y;
                    CoreRangeSet all_cores = num_cores_to_corerange_set(num_cores, program_config.compute_with_storage_grid_size, true);
                    ShardSpec shard_spec = ShardSpec{all_cores, {per_core_M * TILE_HEIGHT, per_core_N * TILE_WIDTH}, ShardOrientation::ROW_MAJOR};
                    auto mem_config = this->output_mem_config;
                    mem_config.shard_spec = shard_spec;
                    return {create_sharded_device_tensor(this->compute_output_shapes(input_tensors).at(0), this->output_dtype, output_layout, input_tensor_a.device(), mem_config)};
                } else if constexpr (
                    std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCastProgramConfig>
                ) {
                    uint32_t M = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / TILE_HEIGHT;
                    uint32_t N = input_tensor_b.get_legacy_shape()[-1]/TILE_WIDTH;
                    uint32_t per_core_M = program_config.per_core_M;
                    uint32_t per_core_N = program_config.per_core_N;

                    uint32_t num_blocks_y = (M - 1) / per_core_M + 1;
                    uint32_t num_blocks_x = (N - 1) / per_core_N + 1;
                    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
                    uint32_t num_cores = num_blocks_x * num_blocks_y;
                    CoreRangeSet all_cores({});
                    ShardOrientation shard_orientation;
                    if (program_config.transpose_mcast) {
                        all_cores = CoreRangeSet({CoreRange({0, 0}, {num_blocks_y - 1, num_blocks_x - 1})});
                        shard_orientation = ShardOrientation::COL_MAJOR;
                    } else {
                        all_cores = CoreRangeSet({CoreRange({0, 0}, {num_blocks_x - 1, num_blocks_y - 1})});
                        shard_orientation = ShardOrientation::ROW_MAJOR;
                    }
                    ShardSpec shard_spec = ShardSpec{all_cores, {per_core_M * TILE_HEIGHT, per_core_N * TILE_WIDTH}, shard_orientation};
                    auto mem_config = this->output_mem_config;
                    mem_config.shard_spec = shard_spec;
                    return {create_sharded_device_tensor(this->compute_output_shapes(input_tensors).at(0), this->output_dtype, output_layout, input_tensor_a.device(), mem_config)};
                } else if constexpr (
                    std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseProgramConfig>
                ) {
                    uint32_t M = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / TILE_HEIGHT;
                    uint32_t N = input_tensor_b.get_legacy_shape()[-1]/TILE_WIDTH;
                    uint32_t per_core_M = program_config.per_core_M;
                    uint32_t per_core_N = program_config.per_core_N;

                    uint32_t num_blocks_y = (M - 1) / per_core_M + 1;
                    uint32_t num_blocks_x = (N - 1) / per_core_N + 1;
                    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
                    uint32_t num_cores = num_blocks_x * num_blocks_y;
                    ShardOrientation shard_orientation = ShardOrientation::COL_MAJOR;
                    if (input_tensor_a.is_sharded()) {
                        shard_orientation = input_tensor_a.shard_spec().value().orientation;
                    } else if (input_tensor_b.is_sharded()) {
                        shard_orientation = input_tensor_b.shard_spec().value().orientation;
                    }

                    CoreRangeSet all_cores = num_cores_to_corerange_set(num_cores, program_config.compute_with_storage_grid_size, shard_orientation==ShardOrientation::ROW_MAJOR);
                    ShardSpec shard_spec = ShardSpec{all_cores, {per_core_M * TILE_HEIGHT, per_core_N * TILE_WIDTH}, shard_orientation};
                    auto mem_config = this->output_mem_config;
                    mem_config.shard_spec = shard_spec;
                    return {create_sharded_device_tensor(this->compute_output_shapes(input_tensors).at(0), this->output_dtype, output_layout, input_tensor_a.device(), mem_config)};
                } else {
                    TT_FATAL(false, "Unsupported op for output sharding");
                    return {};
                }
            },
            this->program_config
        );
    }
    return operation::generic_create_output_tensors(*this, input_tensors, this->output_dtype, Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks Matmul::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor> &output_tensors
) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    const auto& bias = optional_input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    tt::tt_metal::DataType output_dtype = this->output_dtype;

    bool fuse_batch = true;
    // TODO: If input_tensor_a.get_legacy_shape()[0] * input_tensor_a.get_legacy_shape()[1] == 1, does matmuls work if we treat it as bmm
    // TODO: Only for MatmulMultiCoreReuseProgramConfig we allow this as single core matmul/bmm
    bool broadcast_batch = input_tensor_b.get_legacy_shape()[0] * input_tensor_b.get_legacy_shape()[1] == 1;

    return std::visit(
        [&](const auto& program_config) -> operation::ProgramWithCallbacks {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<ProgramConfigType, MatmulDefaultProgramConfig>) {
                auto parallelization_strategy = bmm_op_utils::get_parallelization_strategy(input_tensors);
                switch (parallelization_strategy){
                    case MatmulParallelizationStrategy::MULTI_CORE:
                        return matmul_multi_core(input_tensor_a, input_tensor_b, output_tensor, broadcast_batch);
                    case MatmulParallelizationStrategy::MULTI_CORE_REUSE:
                        return matmul_multi_core_reuse(input_tensor_a, input_tensor_b, output_tensor, broadcast_batch);
                    case MatmulParallelizationStrategy::MULTI_CORE_REUSE_MCAST_2D_OPTIMIZED:
                        return matmul_multi_core_reuse_mcast_2d_optimized(
                            input_tensor_a, input_tensor_b, bias, output_tensor,
                            broadcast_batch,
                            input_tensor_a.device()->compute_with_storage_grid_size(),
                            this->compute_kernel_config,
                            2, 4, 2,
                            16, 16, false, false, std::nullopt,
                            this->untilize_out
                        );
                    case MatmulParallelizationStrategy::MULTI_CORE_REUSE_MCAST_1D_IN0_OPTIMIZED:
                        return matmul_multi_core_reuse_mcast_1d_optimized(
                            input_tensor_a, input_tensor_b, std::nullopt, output_tensor,
                            broadcast_batch,
                            input_tensor_a.device()->compute_with_storage_grid_size(),
                            this->compute_kernel_config,
                            2, 4, 2,
                            16, 16, false, std::nullopt, true,
                            this->untilize_out
                        );
                    case MatmulParallelizationStrategy::MULTI_CORE_REUSE_MCAST_1D_IN1_OPTIMIZED:
                        return matmul_multi_core_reuse_mcast_1d_optimized(
                            input_tensor_a, input_tensor_b, std::nullopt, output_tensor,
                            broadcast_batch,
                            input_tensor_a.device()->compute_with_storage_grid_size(),
                            this->compute_kernel_config,
                            2, 4, 2,
                            16, 16, false, std::nullopt, false,
                            this->untilize_out
                        );
                    case MatmulParallelizationStrategy::MULTI_CORE_REUSE_PADDING:
                        return matmul_multi_core_reuse_padding(input_tensor_a, input_tensor_b, output_tensor, broadcast_batch);
                    case MatmulParallelizationStrategy::SINGLE_CORE:
                    default:
                        return matmul_single_core(input_tensor_a, input_tensor_b, output_tensor, broadcast_batch);
                }
            }
            else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseProgramConfig>) {
                broadcast_batch &= input_tensor_a.get_legacy_shape()[0] * input_tensor_a.get_legacy_shape()[1] > 1;
                return bmm_multi_core_reuse_optimized(
                    input_tensor_a, input_tensor_b, output_tensor,
                    broadcast_batch,
                    program_config.compute_with_storage_grid_size,
                    output_dtype,
                    this->compute_kernel_config,
                    program_config.in0_block_w, program_config.out_subblock_h, program_config.out_subblock_w,
                    program_config.per_core_M, program_config.per_core_N, fuse_batch,
                    this->untilize_out
                );
            }
            else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCastProgramConfig>) {
                return matmul_multi_core_reuse_mcast_2d_optimized(
                    input_tensor_a, input_tensor_b, bias, output_tensor,
                    broadcast_batch,
                    program_config.compute_with_storage_grid_size,
                    this->compute_kernel_config,
                    program_config.in0_block_w, program_config.out_subblock_h, program_config.out_subblock_w,
                    program_config.per_core_M, program_config.per_core_N, fuse_batch, program_config.transpose_mcast, program_config.fused_activation,
                    this->untilize_out
                );
            }
            else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                return matmul_multi_core_reuse_mcast_1d_optimized(
                    input_tensor_a, input_tensor_b, bias, output_tensor,
                    broadcast_batch,
                    program_config.compute_with_storage_grid_size,
                    this->compute_kernel_config,
                    program_config.in0_block_w, program_config.out_subblock_h, program_config.out_subblock_w,
                    program_config.per_core_M, program_config.per_core_N, program_config.fuse_batch, program_config.fused_activation,
                    program_config.mcast_in0,
                    this->untilize_out
                );
            } else {
                TT_THROW("Unrecognized Config");
            }
        },
        this->program_config
    );
}

MatmulParallelizationStrategy Matmul::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    return std::visit(
        [&](const auto& program_config) -> MatmulParallelizationStrategy {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<ProgramConfigType, MatmulDefaultProgramConfig>) {
                return bmm_op_utils::get_parallelization_strategy(input_tensors);
            }
            else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseProgramConfig>) {
                return MatmulParallelizationStrategy::MULTI_CORE_REUSE_OPTIMIZED;
            }
            else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCastProgramConfig>) {
                if (program_config.transpose_mcast) {
                    return MatmulParallelizationStrategy::MULTI_CORE_REUSE_MCAST_2D_TRANSPOSED_OPTIMIZED;
                } else {
                    return MatmulParallelizationStrategy::MULTI_CORE_REUSE_MCAST_2D_OPTIMIZED;
                }
            }
            else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                if (program_config.mcast_in0) {
                    return MatmulParallelizationStrategy::MULTI_CORE_REUSE_MCAST_1D_IN0_OPTIMIZED;
                } else {
                    return MatmulParallelizationStrategy::MULTI_CORE_REUSE_MCAST_1D_IN1_OPTIMIZED;
                }
            } else {
                TT_THROW("Unrecognized Config");
            }
        },
        this->program_config
    );
}

Tensor matmul_1d(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, std::optional<MatmulMultiCoreReuseMultiCast1DProgramConfig> program_config, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype, std::optional<const DeviceComputeKernelConfig> compute_kernel_config, bool untilize_out) {
    if (!program_config.has_value()) {
        program_config = bmm_op_utils::get_mcast_1d_config(input_tensor_a, input_tensor_b);
    }
    auto kernel_config_val = init_device_compute_kernel_config(input_tensor_a.device()->arch(), compute_kernel_config);
    return operations::primary::matmul(input_tensor_a, input_tensor_b, bias, program_config.value(), mem_config, output_dtype, kernel_config_val, untilize_out);
}

operation::OpPerformanceModel Matmul::create_op_performance_model(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor> &output_tensors
    ) const {
    return ::create_op_performance_model_for_matmul(input_tensors, optional_input_tensors, output_tensors, this->compute_kernel_config);
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
