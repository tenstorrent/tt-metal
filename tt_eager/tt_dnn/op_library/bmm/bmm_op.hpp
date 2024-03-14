// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>

#include "tensor/tensor.hpp"

#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_dnn/op_library/compute_kernel_config.hpp"

namespace tt {

namespace tt_metal {

// TODO: Accept parallelization
enum class MatmulParallelizationStrategy {
    MULTI_CORE = 0,
    MULTI_CORE_REUSE = 1,
    MULTI_CORE_REUSE_PADDING = 2,
    MULTI_CORE_REUSE_OPTIMIZED = 3,
    MULTI_CORE_REUSE_MCAST_2D_OPTIMIZED = 4,
    MULTI_CORE_REUSE_MCAST_2D_TRANSPOSED_OPTIMIZED = 5,
    MULTI_CORE_REUSE_MCAST_1D_IN0_OPTIMIZED = 6,
    MULTI_CORE_REUSE_MCAST_1D_IN1_OPTIMIZED = 7,
    SINGLE_CORE = 8
};


/*
 * GENERAL MATMUL AND BMM
 */
operation::ProgramWithCallbacks matmul_single_core  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, bool bcast_batch);
operation::ProgramWithCallbacks matmul_multi_core  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, bool bcast_batch);
operation::ProgramWithCallbacks matmul_multi_core_reuse  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, bool bcast_batch);
operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, bool bcast_batch);
operation::ProgramWithCallbacks matmul_multi_core_reuse_padding (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, bool bcast_batch);
operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_padding (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, bool bcast_batch);

struct Matmul {
    bool bcast_batch;
    const MemoryConfig output_mem_config;
    const DataType output_dtype; // TODO: Uplift output_dtype as an option for general matmul/bmm
    const DeviceComputeKernelConfig compute_kernel_config;
    const bool untilize_out;

    void validate(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, std::vector<Tensor> &output_tensors) const;
    operation::OpPerformanceModel create_op_performance_model(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor> &output_tensors
    ) const;
    MatmulParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;

    static constexpr auto attribute_names =
        std::make_tuple("bcast_batch", "output_mem_config", "output_dtype", "compute_kernel_config", "untilize_out");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->bcast_batch),
            std::cref(this->output_mem_config),
            std::cref(this->output_dtype),
            std::cref(this->compute_kernel_config),
            std::cref(this->untilize_out));
    }
};


operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_1d_optimized(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const std::optional<const Tensor> bias, Tensor &output_tensor, bool bcast_batch, CoreCoord compute_with_storage_grid_size, DeviceComputeKernelConfig compute_kernel_config, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t out_subblock_w, uint32_t per_core_M, uint32_t per_core_N, bool fuse_batch, std::optional<UnaryWithParam> fused_activation, bool mcast_in0, bool untilize_out);
operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_2d_optimized(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const std::optional<const Tensor> bias, Tensor &output_tensor, bool bcast_batch, CoreCoord compute_with_storage_grid_size, DeviceComputeKernelConfig compute_kernel_config, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t out_subblock_w, uint32_t per_core_M, uint32_t per_core_N, bool fuse_batch, bool transpose_mcast, std::optional<UnaryWithParam> fused_activation, bool untilize_out);
operation::ProgramWithCallbacks bmm_multi_core_reuse_optimized(const Tensor& input_tensor_a, const Tensor& input_tensor_b, Tensor &output_tensor, bool bcast_batch, CoreCoord compute_with_storage_grid_size, tt::tt_metal::DataType output_dtype, DeviceComputeKernelConfig compute_kernel_config, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t out_subblock_w, uint32_t per_core_M, uint32_t per_core_N, bool fuse_batch, bool untilize_out);


/**
 * Bert large matmuls using operations::primary::matmul + program_config
 */
Tensor bert_large_fused_qkv_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DataType> output_dtype=std::nullopt);
Tensor bert_large_ff1_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, std::optional<UnaryWithParam> fused_activation = std::nullopt, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DataType> output_dtype=std::nullopt);
Tensor bert_large_ff2_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DataType> output_dtype=std::nullopt);
Tensor bert_large_selfout_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DataType> output_dtype=std::nullopt);
Tensor bert_large_pre_softmax_bmm(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DataType> output_dtype=std::nullopt);
Tensor bert_large_post_softmax_bmm(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DataType> output_dtype=std::nullopt);

/**
 * Falcon matmuls using operations::primary::matmul + program_config
 */
Tensor falcon_fused_qkv_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DataType> output_dtype=std::nullopt);
Tensor falcon_selfout_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DataType> output_dtype=std::nullopt);
Tensor falcon_dense_4h_to_h_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DataType> output_dtype=std::nullopt, std::optional<bool> packer_l1_acc = std::nullopt);
Tensor falcon_dense_h_to_4h_matmul (const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, std::optional<UnaryWithParam> fused_activation = std::nullopt, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DataType> output_dtype=std::nullopt);
Tensor falcon_lm_head_matmul (const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DataType> output_dtype=std::nullopt);

/**
 * Resnet matmul for linear
 */
Tensor resnet_matmul(const Tensor& input_a, const Tensor& input_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DataType> output_dtype = std::nullopt, const MathFidelity math_fidelity = MathFidelity::LoFi);


/**
 * Generalized blocked matmul with support for tilize and untilize and mixed-prec
 */
struct BMMTilizeUntilize {
    const DataType out_dt_;
    const uint32_t in0_nblocks_h_, in0_nblocks_w_, in1_nblocks_w_;
    const uint32_t in0_block_ntiles_h_, in0_block_ntiles_w_, in1_block_ntiles_w_;
    const uint32_t out_subblock_ntiles_h_, out_subblock_ntiles_w_;
    const bool tilize_in0_, untilize_out_;
    const bool has_bias_;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::make_tuple(
        "out_dt",
        "in0_nblocks_h",
        "in0_nblocks_w",
        "in1_nblocks_w",
        "in0_block_ntiles_h",
        "in0_block_ntiles_w",
        "in1_block_ntiles_w",
        "out_subblock_ntiles_h",
        "out_subblock_ntiles_w",
        "tilize_in0",
        "untilize_out",
        "has_bias");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->out_dt_),
            std::cref(this->in0_nblocks_h_),
            std::cref(this->in0_nblocks_w_),
            std::cref(this->in1_nblocks_w_),
            std::cref(this->in0_block_ntiles_h_),
            std::cref(this->in0_block_ntiles_w_),
            std::cref(this->in1_block_ntiles_w_),
            std::cref(this->out_subblock_ntiles_h_),
            std::cref(this->out_subblock_ntiles_w_),
            std::cref(this->tilize_in0_),
            std::cref(this->untilize_out_),std::cref(this->has_bias_));
    }
};

/**
 * Blocked Matmul, with support for tilize a and untilize output.
 * NOTE: Takes blocks and subblock information as arguments.
 */
Tensor bmm_tilize_untilize(const Tensor& a, const Tensor& b, const Tensor& bias, DataType out_dt,
                           uint32_t a_height_nblocks, uint32_t a_width_nblocks, uint32_t b_width_nblocks,
                           uint32_t a_block_height_ntiles, uint32_t a_block_width_ntiles, uint32_t b_block_width_ntiles,
                           uint32_t out_subblock_height_ntiles, uint32_t out_subblock_width_ntiles,
                           bool tilize_in0, bool untilize_out, bool has_bias);
operation::ProgramWithCallbacks bmm_single_core_tilize_untilize(
                                    const Tensor &in0, const Tensor &in1, Tensor &bias, DataType out_dt,
                                    uint32_t in0_height_nblocks, uint32_t in0_width_nblocks, uint32_t in1_width_nblocks,
                                    uint32_t in0_block_height_ntiles, uint32_t in0_block_width_ntiles, uint32_t in1_block_width_ntiles,
                                    uint32_t out_subblock_height_ntiles, uint32_t out_subblock_width_ntiles,
                                    bool tilize_in0, bool untilize_out, bool has_bias,
                                    Tensor &out);

}  // namespace tt_metal


namespace operations {

namespace primary {

using namespace tt_metal;

struct MatmulDefaultProgramConfig{
    static constexpr auto attribute_names = std::make_tuple();
    const auto attribute_values() const { return std::make_tuple(); }
};

// TODO: Uplift this to support fused activation and bias
// TODO: Uplift this to support bcast batch for in1; currently, only allows B=1 for in1 iff B=1 for in0 (ie. single core)
struct MatmulMultiCoreReuseProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w;
    std::size_t out_subblock_h;
    std::size_t out_subblock_w;
    std::size_t per_core_M;
    std::size_t per_core_N;

    static constexpr auto attribute_names = std::make_tuple(
        "compute_with_storage_grid_size",
        "in0_block_w",
        "out_subblock_h",
        "out_subblock_w",
        "per_core_M",
        "per_core_N");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->compute_with_storage_grid_size),
            std::cref(this->in0_block_w),
            std::cref(this->out_subblock_h),
            std::cref(this->out_subblock_w),
            std::cref(this->per_core_M),
            std::cref(this->per_core_N));
    }
};

struct MatmulMultiCoreReuseMultiCastProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w;
    std::size_t out_subblock_h;
    std::size_t out_subblock_w;
    std::size_t per_core_M;
    std::size_t per_core_N;
    bool transpose_mcast;
    std::optional<UnaryWithParam> fused_activation;

    static constexpr auto attribute_names = std::make_tuple(
        "compute_with_storage_grid_size",
        "in0_block_w",
        "out_subblock_h",
        "out_subblock_w",
        "per_core_M",
        "per_core_N",
        "transpose_mcast",
        "fused_activation");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->compute_with_storage_grid_size),
            std::cref(this->in0_block_w),
            std::cref(this->out_subblock_h),
            std::cref(this->out_subblock_w),
            std::cref(this->per_core_M),
            std::cref(this->per_core_N),
            std::cref(this->transpose_mcast),
            std::cref(this->fused_activation));
    }
};

struct MatmulMultiCoreReuseMultiCast1DProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w;
    std::size_t out_subblock_h;
    std::size_t out_subblock_w;
    std::size_t per_core_M;
    std::size_t per_core_N;
    bool fuse_batch;
    std::optional<UnaryWithParam> fused_activation;
    bool mcast_in0;

    static constexpr auto attribute_names = std::make_tuple(
        "compute_with_storage_grid_size",
        "in0_block_w",
        "out_subblock_h",
        "out_subblock_w",
        "per_core_M",
        "per_core_N",
        "fuse_batch",
        "fused_activation",
        "mcast_in0");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->compute_with_storage_grid_size),
            std::cref(this->in0_block_w),
            std::cref(this->out_subblock_h),
            std::cref(this->out_subblock_w),
            std::cref(this->per_core_M),
            std::cref(this->per_core_N),
            std::cref(this->fuse_batch),
            std::cref(this->fused_activation),
            std::cref(this->mcast_in0));
    }
};

using MatmulProgramConfig = std::variant<
    MatmulDefaultProgramConfig,
    MatmulMultiCoreReuseProgramConfig,
    MatmulMultiCoreReuseMultiCastProgramConfig,
    MatmulMultiCoreReuseMultiCast1DProgramConfig
>;


struct Matmul {
    MatmulProgramConfig program_config;
    const MemoryConfig output_mem_config;
    const DataType output_dtype;
    const DeviceComputeKernelConfig compute_kernel_config;
    const bool untilize_out;

    void validate(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor> &output_tensors
    ) const;
    operation::OpPerformanceModel create_op_performance_model(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor> &output_tensors
    ) const;
    MatmulParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;

    static constexpr auto attribute_names =
        std::make_tuple("program_config", "output_mem_config", "output_dtype", "compute_kernel_config", "untilize_out");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->program_config),
            std::cref(this->output_mem_config),
            std::cref(this->output_dtype),
            std::cref(this->compute_kernel_config),
            std::cref(this->untilize_out));
    }
};

inline Tensor matmul(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    const MatmulProgramConfig& program_config = MatmulDefaultProgramConfig{},
    const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DataType> output_dtype = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    bool untilize_out = false
) {
    auto arch = input_tensor_a.storage_type() == StorageType::DEVICE ? input_tensor_a.device()->arch() : AutoFormat::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config);
    return operation::run(Matmul{program_config, mem_config, output_dtype.value_or(input_tensor_a.get_dtype()), kernel_config_val, untilize_out}, {input_tensor_a, input_tensor_b}, {std::nullopt}).at(0);
}

inline Tensor matmul(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    std::optional<const Tensor> bias,
    const MatmulProgramConfig &program_config = MatmulDefaultProgramConfig{},
    const MemoryConfig &mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DataType> output_dtype = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    bool untilize_out = false) {
    auto arch = input_tensor_a.storage_type() == StorageType::DEVICE ? input_tensor_a.device()->arch() : AutoFormat::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config);
    return operation::run(Matmul{program_config, mem_config, output_dtype.value_or(input_tensor_a.get_dtype()), kernel_config_val, untilize_out}, {input_tensor_a, input_tensor_b}, {bias}).at(0);
}

Tensor matmul_1d(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, std::optional<MatmulMultiCoreReuseMultiCast1DProgramConfig> program_config = std::nullopt, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DataType> output_dtype=std::nullopt, std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt, bool untilize_out = false);

}  // namespace primary

}  // namespace operations

}  // namespace tt


namespace bmm_op_utils {
using namespace tt::tt_metal;

constexpr std::array<tuple<uint32_t, uint32_t>, 20> SUBBLOCK_HW_CHOICES = {{
    {4, 2}, {2, 4}, {8, 1}, {1, 8},
    {7, 1}, {1, 7},
    {3, 2}, {2, 3}, {6, 1}, {1, 6},
    {5, 1}, {1, 5},
    {2, 2}, {4, 1}, {1, 4},
    {3, 1}, {1, 3},
    {2, 1}, {1, 2},
    {1, 1},
}};

tuple<uint32_t, uint32_t, uint32_t, uint32_t> get_large_matmul_params(uint32_t Mt, uint32_t Nt, uint32_t num_cores_y, uint32_t num_cores_x, uint32_t in0_block_w);

CoreCoord get_core_range(uint32_t num_blocks_rows, uint32_t num_blocks_cols, uint32_t max_num_rows, uint32_t max_num_cols);

// TODO: Remove get_mcast_1d_config and merge with general version?
tt::operations::primary::MatmulMultiCoreReuseMultiCast1DProgramConfig get_mcast_1d_config(const Tensor &input_tensor_a, const Tensor &input_tensor_b, bool fuse_batch = false, std::optional<UnaryWithParam> fused_activation = std::nullopt, bool mcast_in0 = true, bool out_sharded = false);

tuple<uint32_t, uint32_t> get_matmul_subblock_params(const uint32_t per_core_M, const uint32_t per_core_N, const bool per_core_M_equals_subblock_h_constraint, bool per_core_N_equals_subblock_w_constraint);

// TODO: Review usage of matmul bool; should probably infer this from batch
tt::operations::primary::MatmulProgramConfig get_matmul_program_config(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const MemoryConfig &output_mem_config, std::optional<UnaryWithParam> fused_activation = std::nullopt, const bool matmul = false);
}  // namespace bmm_op_utils


// TODO: We can get rid of this once we remove tt:tt_metal::Matmul
namespace tt {

namespace tt_metal {

inline Tensor matmul (const Tensor &input_tensor_a, const Tensor &input_tensor_b, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt, bool untilize_out = false) {
    TT_FATAL(input_tensor_a.get_dtype() == input_tensor_b.get_dtype());
    TT_FATAL(input_tensor_a.get_legacy_shape()[3] == input_tensor_b.get_legacy_shape()[2] && "Dimension K (A.shape[3] and B.shape[2]) must match for A and B in bmm_op"); // A.K == B.K
    TT_FATAL(input_tensor_b.get_legacy_shape()[0]*input_tensor_b.get_legacy_shape()[1] == 1 && "matmul (batch bcast variant) expects input tensors of shapes BCMK*11KN=BCMN");

    auto arch = input_tensor_a.storage_type() == StorageType::DEVICE ? input_tensor_a.device()->arch() : AutoFormat::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4);

    // TODO: Uplift interleaved path to call tt::operation::primary::Matmul and deprecate old tt::tt_metal::Matmul
    if (input_tensor_a.is_sharded()) {
        auto matmul_program_config = bmm_op_utils::get_matmul_program_config(input_tensor_a, input_tensor_b, mem_config, std::nullopt, true);
        return operation::run(
                   tt::operations::primary::Matmul{
                       .program_config = matmul_program_config,
                       .output_mem_config = mem_config,
                       .output_dtype = input_tensor_a.get_dtype(),
                       .compute_kernel_config = kernel_config_val,
                       .untilize_out = untilize_out,
                   },
                   {input_tensor_a, input_tensor_b},
                   {std::nullopt})
            .at(0);
    } else {
        return operation::run_with_autoformat(
                   Matmul{
                       .bcast_batch = true,
                       .output_mem_config = mem_config,
                       .output_dtype = input_tensor_a.get_dtype(),
                       .compute_kernel_config = kernel_config_val,
                       .untilize_out = untilize_out,
                   },
                   {input_tensor_a, input_tensor_b},
                   {std::nullopt})
            .at(0);
    }
}
// TODO: Should we merge this with matmul and expose an option (or infer it from shape) to bcast_batch
inline Tensor bmm    (const Tensor &input_tensor_a, const Tensor &input_tensor_b, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt, bool untilize_out = false) {
    TT_FATAL(input_tensor_a.get_dtype() == input_tensor_b.get_dtype());
    TT_FATAL(input_tensor_a.get_legacy_shape()[3] == input_tensor_b.get_legacy_shape()[2] && "Dimension K (A.shape[3] and B.shape[2]) must match for A and B in bmm_op"); // A.K == B.K
    TT_FATAL(input_tensor_a.get_legacy_shape()[1] == input_tensor_b.get_legacy_shape()[1] && input_tensor_a.get_legacy_shape()[0] == input_tensor_b.get_legacy_shape()[0]
        && "bmm (non-bcast matmul) expects input tensors of shapes BCMK*BCKN=BCMN");

    auto arch = input_tensor_a.storage_type() == StorageType::DEVICE ? input_tensor_a.device()->arch() : AutoFormat::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4);

    if (input_tensor_a.is_sharded()) {
        auto matmul_program_config = bmm_op_utils::get_matmul_program_config(input_tensor_a, input_tensor_b, mem_config, std::nullopt, false);
        return operation::run(tt::operations::primary::Matmul{
            .program_config=matmul_program_config,
            .output_mem_config=mem_config,
            .output_dtype=input_tensor_a.get_dtype(),
            .compute_kernel_config=kernel_config_val,
            .untilize_out=untilize_out
        }, {input_tensor_a, input_tensor_b}, {std::nullopt}).at(0);
    } else {
        return operation::run_with_autoformat(
                   Matmul{
                       .bcast_batch = false,
                       .output_mem_config = mem_config,
                       .output_dtype = input_tensor_a.get_dtype(),
                       .compute_kernel_config = kernel_config_val,
                       .untilize_out = untilize_out,
                   },
                   {input_tensor_a, input_tensor_b},
                   {std::nullopt})
            .at(0);
    }
}

}  // namespace tt_metal

}  // namespace tt
