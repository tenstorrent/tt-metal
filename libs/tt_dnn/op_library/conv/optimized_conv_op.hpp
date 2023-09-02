#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace tt_metal {

// TODO: Accept parallelization
enum class OptimizedConvOpParallelizationStrategy {
    MULTI_CORE = 0, MULTI_CORE_REUSE = 1, MULTI_CORE_REUSE_MCAST = 2, SINGLE_CORE = 3
};

struct OptimizedConvParallelizationConfig {
    CoreCoord grid_size; // (x,y)
    uint32_t per_core_act_matrix_height_ntiles;
    // std::size_t in0_block_w;
    // std::size_t out_subblock_h;
    // std::size_t out_subblock_w;
    // std::size_t per_core_M;
    // std::size_t per_core_N;

    tt::stl::reflection::Attributes attributes() const;
};

struct OptimizedConv {
    OptimizedConvParallelizationConfig parallelization_config;
    const std::vector<int> conv_params;
    const uint32_t act_block_h_ntiles, act_block_w_ntiles, weight_block_w_ntiles, out_subblock_h_ntiles, out_subblock_w_ntiles, output_channels;
    bool untilize_out, has_bias, fuse_relu;
    MathFidelity math_fidelity;
    uint32_t extra_padding_for_32B_alignment;
    OptimizedConv(uint32_t act_bh, uint32_t act_bw,
        uint32_t weight_bw, uint32_t out_sh,
        uint32_t out_sw, const std::vector<int>&c_params,
        uint32_t output_channels, bool untile_out,
        bool has_bias, bool fuse_relu,
        MathFidelity mfidelity, const OptimizedConvParallelizationConfig& p_config,
        uint32_t e_padding_for_32B_alignment) :
            act_block_h_ntiles(act_bh),
            act_block_w_ntiles(act_bw),
            weight_block_w_ntiles(weight_bw),
            out_subblock_h_ntiles(out_sh),
            out_subblock_w_ntiles(out_sw),
            output_channels(output_channels),
            conv_params(c_params),
            untilize_out(untile_out),
            has_bias(has_bias),
            fuse_relu(fuse_relu),
            math_fidelity(mfidelity),
            parallelization_config(p_config),
            extra_padding_for_32B_alignment(e_padding_for_32B_alignment) {}

    void validate(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

Tensor optimized_conv(const Tensor& a, const Tensor &b, std::optional<const Tensor> bias, const vector<int> conv_params, uint32_t act_block_h_ntiles, uint32_t act_block_w_ntiles, uint32_t weight_block_w_ntiles,
             uint32_t out_subblock_h_ntiles, uint32_t out_subblock_w_ntiles, uint32_t output_channels, bool untilize_out, bool has_bias, bool fuse_relu, MathFidelity math_fidelity, const OptimizedConvParallelizationConfig& parallelization_config, uint32_t extra_padding_for_32B_alignment=0);

}  // namespace tt_metal

}  // namespace tt
