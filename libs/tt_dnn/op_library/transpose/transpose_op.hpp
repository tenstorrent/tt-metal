#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace tt_metal {

struct TransposeOpDim {
    enum Enum { WH = 0, HC = 1, CN = 2 };
    static const vector<Enum> all() { return { WH, HC, CN }; }
};

struct TransposeOpParallelizationStrategy {
    enum Enum { MULTI_CORE_WH = 0, MULTI_CORE_HC = 1, SINGLE_CORE = 2 };
    static const vector<Enum> all() { return { MULTI_CORE_WH, MULTI_CORE_HC, SINGLE_CORE }; }
};

struct Transpose {
    const TransposeOpDim::Enum dim;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;
    TransposeOpParallelizationStrategy::Enum get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
};

// TODO: Accept parallelization
Tensor transpose_(const Tensor &a, TransposeOpDim::Enum transpose_dim=TransposeOpDim::WH);
// TODO: Don't bind transpose as transpose_wh, should explicitly bind like the others
// Alternatively, bind only 1 transpose function and take 2 dims to transpose
inline Tensor transpose(const Tensor &a) { return transpose_(a, TransposeOpDim::WH); }
inline Tensor transpose_wh(const Tensor &a) { return transpose_(a, TransposeOpDim::WH); }
inline Tensor transpose_hc(const Tensor &a) { return transpose_(a, TransposeOpDim::HC); }
inline Tensor transpose_cn(const Tensor &a) { return transpose_(a, TransposeOpDim::CN); }

operation::ProgramWithCallbacks transpose_single_core(const Tensor &a, Tensor &output, TransposeOpDim::Enum transpose_dim);
operation::ProgramWithCallbacks transpose_wh_multi_core(const Tensor &a, Tensor &output);
operation::ProgramWithCallbacks transpose_hc_multi_core(const Tensor &a, Tensor &output);

}  // namespace tt_metal

}  // namespace tt
