#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace tt_metal {

// TODO: Accept parallelization
struct ConvOpParallelizationStrategy {
    enum Enum { MULTI_CORE = 0, MULTI_CORE_REUSE = 1, MULTI_CORE_REUSE_MCAST = 2, SINGLE_CORE = 3 };
    static const vector<Enum> all() { return { MULTI_CORE, MULTI_CORE_REUSE, MULTI_CORE_REUSE_MCAST, SINGLE_CORE }; }
};

struct Conv {
    void validate(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors) const;
    Program create_program(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors, std::vector<Tensor> &output_tensors) const;

    Conv(uint32_t in0_bh, uint32_t in0_bw, uint32_t in1_bw, uint32_t out_sh, uint32_t out_sw, const std::vector<int>&c_params, bool unt_out = true)
        : in0_block_h(in0_bh),
          in0_block_w(in0_bw),
          in1_block_w(in1_bw),
          out_subblock_h(out_sh),
          out_subblock_w(out_sw),
          untilize_out(unt_out),
          conv_params(c_params) {}

    // additional parameters
    std::vector<int> conv_params;
    uint32_t in0_block_h, in0_block_w, in1_block_w, out_subblock_h, out_subblock_w;
    bool untilize_out;
};

Tensor conv(const Tensor& a, const Tensor &b, const vector<int> conv_params, uint32_t in0_block_h, uint32_t in0_block_w, uint32_t in1_block_w,
             uint32_t out_subblock_h, uint32_t out_subblock_w, bool untilize_out);
Program conv_single_core(const Tensor& A, const Tensor& B, vector<int> conv_params, uint32_t in0_block_h, uint32_t in0_block_w, uint32_t in1_block_w,
             uint32_t out_subblock_h, uint32_t out_subblock_w, bool untilize_out, Tensor& output); // Tilizes a, untilizes b
Program conv_as_large_bmm_single_core_single_block(const Tensor& A, const Tensor& B, bool untilize_out, bool use_single_bank_reader); // Allows support for tilizing a, untilize b
std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> compute_conv_op_block_info(uint32_t M, uint32_t K, uint32_t N);
}  // namespace tt_metal

}  // namespace tt
